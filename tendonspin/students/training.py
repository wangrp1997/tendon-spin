"""Online supervised adaptation on both students' visited states (DAgger).

Physics/rewards/termination are inherited from C. No policy-gradient updates.
Held-out teacher episodes select checkpoints; final test cases are separate.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from tendonspin.rl.resource_guard import atomic_json
from .model import TeacherLabels, TouchStudent, distillation_loss, save_student, sha256
from .observations import FRAME_DIM, HISTORY, sensor_contract

SELECTED_TEACHER_SHA256 = 'a1a4f3c20c62ed766bcce014eb15704d80b962cee0ffe4edff95ba1e2ef4b5a2'


class Replay:
    def __init__(self, capacity, device):
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.history = torch.empty(capacity, HISTORY, FRAME_DIM, device=device)
        self.action = torch.empty(capacity, 16, device=device)
        self.latent = torch.empty(capacity, 8, device=device)

    @torch.no_grad()
    def add(self, history, action, latent):
        n = min(len(history), self.capacity)
        indices = (torch.arange(n, device=history.device)+self.cursor) % self.capacity
        self.history[indices] = history[-n:]
        self.action[indices] = action[-n:]
        self.latent[indices] = latent[-n:]
        self.cursor = (self.cursor+n) % self.capacity
        self.size = min(self.size+n, self.capacity)

    def sample(self, batch_size):
        if not self.size:
            raise ValueError('Empty replay')
        indices = torch.randint(self.size, (batch_size,), device=self.history.device)
        return self.history[indices], self.action[indices], self.latent[indices]


def split_environments(num_envs):
    if num_envs < 18 or num_envs % 9:
        raise ValueError('Use at least two complete nine-scale groups')
    ids = torch.arange(num_envs)
    groups = num_envs//9
    val = ids//9 % 7 == 6 if groups >= 7 else ids//9 == groups-1
    return ids[~val], ids[val]


def train(args, root, out, status):
    from tendonspin.rl.isaac_spin_progress import SpinProgressBoyaEnv
    from .isaac_observations import IsaacStudentSensors
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    teacher = TeacherLabels(args.teacher, 'cuda:0')
    if teacher.checkpoint_sha256 != SELECTED_TEACHER_SHA256:
        raise ValueError('Checkpoint is not the selected C1260M teacher')
    scene = teacher.contract['scene_configuration']
    cache = root/scene['training_cache']
    if sha256(cache) != teacher.contract['cache_sha256']:
        raise ValueError('Teacher grasp cache changed')
    dr_config = deepcopy(teacher.contract['domain_randomization'])
    dr_config['static_seed'] = args.seed
    env = SpinProgressBoyaEnv(root, out, cache, num_envs=args.num_envs, trace_mode='summary',
        scene_configuration=scene, randomization_configuration=dr_config,
        training_randomization=True,
        engine_profile=teacher.contract['engine_configuration']['profile'])
    for key, actual in [('engine_configuration', env.physics.engine_configuration),
                        ('termination', env.termination.spec.to_dict()),
                        ('reward_configuration', env.reward_configuration)]:
        if teacher.contract[key] != actual:
            raise ValueError('Student changed teacher physics/task: '+key)
    sensors = IsaacStudentSensors(env)
    obs_contract = sensor_contract(sensors.history.limits)
    atomic_json(out/'observation_contract.json', obs_contract)
    train_ids, val_ids = (ids.to(env.device) for ids in split_environments(args.num_envs))
    replay = Replay(args.replay_capacity, env.device)
    validation_controls = min(args.validation_controls, args.warmup_controls+args.online_controls)
    validation = Replay(((validation_controls+3)//4)*len(val_ids), env.device)
    students = {name: TouchStudent(teacher, name).to(env.device) for name in ('touch', 'proprio')}
    optimizers = {name: torch.optim.Adam(model.history_encoder.parameters(), lr=3e-4)
                  for name, model in students.items()}
    frozen = {key: value.clone() for key, value in students['touch'].action_head.state_dict().items()}
    best = dict.fromkeys(students, float('inf'))
    step_counts = dict.fromkeys(students, 0)
    obs = env.reset()
    history = sensors.observe(obs)
    action, latent = teacher(obs)
    with torch.no_grad():
        parity = float((students['touch'].action_head(history, latent)-action).abs().max())
        native_obs = dict(obs, obs=teacher.normalizer(obs['obs']))
        native_parity = float((teacher.model.act_inference(native_obs).clamp(-1, 1)-action).abs().max())
    if max(parity, native_parity) > 1e-6:
        raise RuntimeError('C teacher/action-head parity failed')
    status.update(teacher_sha256=teacher.checkpoint_sha256,
        cache_sha256=sha256(cache), teacher_training_num_envs=int(teacher.contract['num_envs']),
        train_environment_ids=train_ids.cpu().tolist(), validation_environment_ids=val_ids.cpu().tolist(),
        selection='Mean action MSE + latent MSE on fixed held-out teacher episodes; no test selection',
        simulation_tactile='Ideal normal+friction resultant on distal rigid body only; no pad ROI or array model',
        tactile_body_names=sensors.distal_names, teacher_action_head_max_error=parity,
        teacher_native_action_max_error=native_parity,
        randomization_configuration=dr_config,
        property_readback=env.dr.verify_properties(), observation_contract=obs_contract,
        phase='teacher_collection', controls=0, optimizer_updates=step_counts.copy(),
        teacher_controlled_actions=0, student_controlled_actions=0, completed_episodes=0,
        first_update_verified=False, validation_collection_controls=validation_controls,
        recorded_controls=0)
    contact_sum = torch.zeros(5, device=env.device)
    force_max = torch.zeros(5, device=env.device)
    started = time.monotonic()
    data_dir = out/'data'
    data_dir.mkdir()
    budget = args.warmup_controls+args.online_controls
    def mapping(name, shape, dtype=np.float32):
        return np.lib.format.open_memmap(data_dir/(name+'.npy'), mode='w+', dtype=dtype, shape=shape)
    archive = dict(sensor_frame=mapping('sensor_frame', (budget, env.num_envs, FRAME_DIM)),
        teacher_action=mapping('teacher_action', (budget, env.num_envs, 16)),
        teacher_latent=mapping('teacher_latent', (budget, env.num_envs, 8)),
        executed_action=mapping('executed_action', (budget, env.num_envs, 16)),
        episode_id=mapping('episode_id', (budget, env.num_envs), np.int64),
        driver=mapping('driver', (budget, env.num_envs), np.int8),
        control_complete=mapping('control_complete', (budget,), np.bool_))
    archive['control_complete'][:] = False

    def record_control(expert, latent, executed, drivers):
        index = status['controls']
        for name, value in dict(sensor_frame=history[:, -1], teacher_action=expert,
                teacher_latent=latent, executed_action=executed, episode_id=env.episode_ids,
                driver=drivers).items():
            archive[name][index] = value.cpu().numpy()
        status['recorded_controls'] = index+1

    def check_stop():
        if (args.stop_file and args.stop_file.exists()) or (out/'STOP').exists():
            raise InterruptedError('User stop file detected')

    def save_status():
        status.update(updated_unix_s=time.time(), wall_s=time.monotonic()-started,
            replay_size=replay.size, validation_size=validation.size,
            completed_episodes=len(env.completed), optimizer_updates=step_counts.copy(),
            episode_stop_counts=dict(env.reason_counts))
        atomic_json(out/'status.json', status)
        print('TOUCH_STUDENT '+json.dumps({key: status.get(key) for key in
            ('phase', 'controls', 'optimizer_updates', 'replay_size', 'wall_s')}), flush=True)

    def assert_frozen():
        for model in students.values():
            if any(not torch.equal(value, frozen[key]) for key, value in model.action_head.state_dict().items()):
                raise RuntimeError('Frozen teacher action head or normalizer changed')

    def optimize(count):
        for _ in range(count):
            check_stop()
            h, a, z = replay.sample(args.batch_size)
            for name, model in students.items():
                model.train()
                prediction = model(h)
                losses = distillation_loss(prediction, a, z)
                if not torch.isfinite(losses['total']):
                    raise FloatingPointError('Nonfinite distillation loss')
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                losses['total'].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.history_encoder.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite encoder gradient')
                optimizer.step()
                step_counts[name] += 1
                if step_counts[name] % 100 == 0 or step_counts[name] == 1:
                    with (out/'losses.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(modality=name, update=step_counts[name],
                            **{k: float(v) for k, v in losses.items()}))+'\n')
        assert_frozen()

    validation_saved = False

    @torch.no_grad()
    def validate():
        nonlocal validation_saved
        check_stop()
        rows = {}
        selection_ready = status['controls'] >= validation_controls
        if selection_ready and not validation_saved:
            np.savez_compressed(out/'validation_samples.npz',
                history=validation.history[:validation.size].cpu().numpy(),
                action=validation.action[:validation.size].cpu().numpy(),
                latent=validation.latent[:validation.size].cpu().numpy(),
                environment_ids=val_ids.cpu().numpy())
            validation_saved = True
        for name, model in students.items():
            model.eval()
            totals = Counter()
            for start in range(0, validation.size, args.batch_size):
                end = min(validation.size, start+args.batch_size)
                losses = distillation_loss(model(validation.history[start:end]),
                    validation.action[start:end], validation.latent[start:end])
                for key, value in losses.items():
                    totals[key] += float(value)*(end-start)
            rows[name] = {k: v/validation.size for k, v in totals.items()}
            if not all(np.isfinite(v) for v in rows[name].values()):
                raise FloatingPointError('Nonfinite held-out validation loss')
            progress = dict(updates=step_counts[name], controls=status['controls'], validation=rows[name])
            save_student(out/f'{name}_last.pth', model, obs_contract, teacher.checkpoint_sha256, progress)
            if selection_ready and rows[name]['total'] < best[name] and step_counts[name] > 0:
                best[name] = rows[name]['total']
                save_student(out/f'{name}_best.pth', model, obs_contract, teacher.checkpoint_sha256, progress)
        status.update(validation=rows, validation_selection_ready=selection_ready,
            best_validation_loss={k: v if np.isfinite(v) else None for k, v in best.items()})
        with (out/'validation.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(controls=status['controls'], updates=step_counts.copy(), scores=rows))+'\n')
        # Keep learning state atomically; resumption deliberately starts fresh
        # simulator episodes/replay and is not a bitwise physics continuation.
        temporary = out/'learner_latest.pth.tmp'
        torch.save(dict(schema='tendonspin-touch-learner-v1',
            models={k: v.state_dict() for k, v in students.items()},
            optimizers={k: v.state_dict() for k, v in optimizers.items()},
            updates=step_counts.copy(), best_validation_loss=best.copy(),
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            teacher_sha256=teacher.checkpoint_sha256,
            environment_resume='Reset cache episodes and refill replay; not exact physical continuation'), temporary)
        temporary.replace(out/'learner_latest.pth')
        save_status()

    save_status()
    for control in range(args.warmup_controls):
        check_stop()
        with torch.no_grad():
            action, latent = teacher(obs)
            if control % 2 == 0:
                replay.add(history[train_ids], action[train_ids], latent[train_ids])
            if control < validation_controls and control % 4 == 0:
                validation.add(history[val_ids], action[val_ids], latent[val_ids])
            contact_sum += history[:, -1, 36:41].sum(0)
            force_max = torch.maximum(force_max, sensors.history.force.amax(0))
            record_control(action, latent, action, torch.full((env.num_envs,), -1, device=env.device))
            obs, _, done, _ = env.step(action)
            history = sensors.observe(obs, torch.where(done)[0])
        archive['control_complete'][status['controls']] = True
        status['controls'] += 1
        status['teacher_controlled_actions'] += env.num_envs
        if env.num_envs <= 18 or (control+1) % 4 == 0:
            save_status()
    if not validation.size or not replay.size or not bool((contact_sum > 0).any()):
        raise RuntimeError('No usable training/validation/contact samples collected')
    status.update(phase='warmup_fitting', contact_fraction=(contact_sum/(args.warmup_controls*env.num_envs)).cpu().tolist(),
                  force_max_n=force_max.cpu().tolist())
    initial = {name: {k: v.clone() for k, v in model.history_encoder.state_dict().items()}
               for name, model in students.items()}
    validate()
    for start in range(0, args.warmup_updates, args.updates_per_block):
        optimize(min(args.updates_per_block, args.warmup_updates-start))
        changed = {name: any(not torch.equal(value, initial[name][key])
            for key, value in model.history_encoder.state_dict().items()) for name, model in students.items()}
        if not all(changed.values()):
            raise RuntimeError('A student encoder did not update')
        status.update(first_update_verified=True, encoder_parameters_changed=changed)
        validate()
    status['phase'] = 'student_rollouts'
    # Both students contribute their own visited states to a shared training
    # replay; validation environments remain teacher-controlled and never fit.
    proprio_driver = (train_ids//9) % 2 == 1
    for block_start in range(0, args.online_controls, 16):
        for local in range(min(16, args.online_controls-block_start)):
            check_stop()
            with torch.no_grad():
                for model in students.values():
                    model.eval()
                expert, latent = teacher(obs)
                if status['controls'] < validation_controls and status['controls'] % 4 == 0:
                    validation.add(history[val_ids], expert[val_ids], latent[val_ids])
                replay.add(history[train_ids], expert[train_ids], latent[train_ids])
                action = expert.clone()
                touch_action = students['touch'](history[train_ids])['action']
                proprio_action = students['proprio'](history[train_ids])['action']
                student_action = torch.where(proprio_driver[:, None], proprio_action, touch_action)
                beta = max(0., 1.-(block_start+local)/max(1, args.online_controls//4))
                use_teacher = torch.rand(len(train_ids), device=env.device) < beta
                action[train_ids] = torch.where(use_teacher[:, None], expert[train_ids], student_action)
                drivers = torch.full((env.num_envs,), -1, dtype=torch.int8, device=env.device)
                drivers[train_ids] = torch.where(use_teacher, -1, proprio_driver.to(torch.int8))
                record_control(expert, latent, action, drivers)
                status['teacher_controlled_actions'] += len(val_ids)+int(use_teacher.sum())
                status['student_controlled_actions'] += int((~use_teacher).sum())
                obs, _, done, _ = env.step(action)
                history = sensors.observe(obs, torch.where(done)[0])
            archive['control_complete'][status['controls']] = True
            status['controls'] += 1
            if env.num_envs <= 18 or status['controls'] % 4 == 0:
                save_status()
        status['teacher_mixture_probability'] = beta
        optimize(args.updates_per_block)
        validate()
    assert_frozen()
    for value in archive.values():
        value.flush()
    with (out/'episodes.jsonl').open('w') as stream:
        for row in env.completed:
            stream.write(json.dumps(row)+'\n')
    status.update(phase='training_completed', status='completed',
        independent_closed_loop_test_completed=False,
        simulation_actions=env.actions_executed,
        final_property_readback=env.dr.verify_properties(),
        artifacts={name: dict(path=str(out/f'{name}_best.pth'), sha256=sha256(out/f'{name}_best.pth'))
                   for name in students})
    save_status()
    return status
