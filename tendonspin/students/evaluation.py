"""Frozen, paired 30-second test on fresh seeds after checkpoint selection.

All controllers start from identical cached states and physical parameters.
After a first task failure, retain the valid prefix; never reset or take over.
"""
from collections import Counter
from copy import deepcopy
import json
import time

import numpy as np
import torch

from tendonspin.rl.resource_guard import atomic_json
from .model import TeacherLabels, load_student, sha256
from .observations import sensor_contract
from .training import SELECTED_TEACHER_SHA256


def summarize(net_deg, valid_s, first_failure, requested_s):
    net_deg, valid_s, first_failure = map(np.asarray, (net_deg, valid_s, first_failure))
    return dict(cases=len(net_deg), mean_net_deg=float(net_deg.mean()),
        p10_net_deg=float(np.quantile(net_deg, .1)),
        mean_valid_s=float(valid_s.mean()), retained=int((first_failure == 0).sum()),
        retention_fraction=float((first_failure == 0).mean()), requested_s=requested_s,
        failure_codes=dict(Counter(map(str, first_failure.tolist()))))


def evaluate(args, root, out, status):
    from tendonspin.rl.isaac_spin_progress import SpinProgressBoyaEnv
    from tendonspin.rl.isaac_hora import spin_increment
    from tendonspin.physics.isaac_parallel import tensor
    from .isaac_observations import IsaacStudentSensors
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    training_status = json.loads((args.training/'status.json').read_text())
    if training_status['status'] != 'completed':
        raise ValueError('Training must complete before the independent test')
    if args.seed == training_status['arguments']['seed']:
        raise ValueError('Test seed overlaps training seed')
    for source in training_status['sources']:
        if sha256(root/source['path']) != source['sha256']:
            raise RuntimeError('Frozen training dependency changed: '+source['path'])
    teacher = TeacherLabels(args.teacher, 'cuda:0')
    if teacher.checkpoint_sha256 != SELECTED_TEACHER_SHA256:
        raise ValueError('Wrong C teacher identity')
    models = {}
    metadata = {}
    for name in ('touch', 'proprio'):
        models[name], metadata[name] = load_student(args.training/f'{name}_best.pth', 'cuda:0')
        if metadata[name]['teacher_sha256'] != teacher.checkpoint_sha256 or models[name].modality != name:
            raise ValueError('Selected student identity mismatch')
    config = deepcopy(teacher.contract['domain_randomization'])
    config['static_seed'] = args.seed
    env = SpinProgressBoyaEnv(root, out, root/teacher.contract['scene_configuration']['training_cache'],
        num_envs=args.num_envs, trace_mode='summary', scene_configuration=teacher.contract['scene_configuration'],
        randomization_configuration=config, training_randomization=True,
        engine_profile=teacher.contract['engine_configuration']['profile'])
    p, dr = env.physics, env.dr
    env.reset()
    sensors = IsaacStudentSensors(env)
    expected = sensor_contract(sensors.history.limits)
    for name in models:
        if metadata[name]['observation_contract'] != expected:
            raise ValueError('Student preprocessing differs from evaluator')
    initial_q = tensor(p.hand.data.joint_pos).clone()
    initial_obj = tensor(p.object.data.root_state_w).clone()
    initial_commands = p.adapter.commands.clone()
    props = dict(mass_kg=dr.mass.cpu().numpy(), scale=dr.scale.cpu().numpy(),
        friction=dr.friction.cpu().numpy(), com_m=dr.com.cpu().numpy(),
        p_ratio=dr.p_ratio.cpu().numpy(), d_ratio=dr.d_ratio.cpu().numpy(),
        cache_index=env.cache_ids.cpu().numpy())
    np.savez_compressed(out/'fixed_cases.npz', joint_pos=initial_q.cpu().numpy(),
        object_state=initial_obj.cpu().numpy(), commands=initial_commands.cpu().numpy(), **props)
    status.update(phase='paired_closed_loop', seed=args.seed,
        fixed_cases_sha256=sha256(out/'fixed_cases.npz'), property_readback=dr.verify_properties(),
        acceptance=dict(min_teacher_mean_ratio=.9, max_retention_fraction_drop=.1),
        limitation='Fresh ID physical conditions on the existing training grasp cache; not hardware or unseen coupling',
        teacher_sha256=teacher.checkpoint_sha256,
        student_sha256={name: sha256(args.training/f'{name}_best.pth') for name in models},
        controllers={}, training_updates=0, resets_within_case=0, teacher_takeovers=0)
    control_dt = p.dt*p.adapter.steps_per_control
    controls = round(args.seconds/control_dt)
    if abs(controls*control_dt-args.seconds) > 1e-9:
        raise ValueError('Evaluation duration must contain whole controls')
    started = time.monotonic()
    stimulus = None
    for controller in ('teacher', 'touch', 'proprio'):
        p._write_state(initial_q, torch.zeros_like(initial_q), initial_obj, initial_commands)
        env.termination.reset()
        dr.force.zero_()
        torch.manual_seed(args.seed+1)
        noise_rng = torch.Generator(device=p.device).manual_seed(args.seed+2)
        env.history[:] = env._frame(noise=False)[:, None]
        obs = env.observe()
        sensors.reset()
        history = sensors.observe(obs)
        measure = p.measure()
        error = max(float((measure['joint_pos']-initial_q).abs().max()),
                    float((measure['object_state']-initial_obj).abs().max()),
                    float((p.adapter.commands-initial_commands).abs().max()))
        if error > 1e-6:
            raise RuntimeError('Paired initial-state readback failed')
        net = torch.zeros(p.num_envs, device=p.device, dtype=torch.float64)
        valid_steps = torch.zeros(p.num_envs, device=p.device, dtype=torch.long)
        failure = torch.zeros_like(valid_steps)
        forces, noises, actions, histories, increments, failures = [], [], [], [], [], []
        for control in range(controls):
            if (out.parent/'STOP').exists():
                raise InterruptedError('User stop file detected')
            with torch.no_grad():
                action = teacher(obs)[0] if controller == 'teacher' else models[controller](history)['action']
                if not torch.isfinite(action).all():
                    raise FloatingPointError('Nonfinite frozen-controller output')
                action = torch.where((failure == 0)[:, None], action, 0.)
                actions.append(action.cpu().numpy())
                histories.append(history[:, -1].cpu().numpy())
                dr.begin_control()
                forces.append(dr.force.cpu().numpy().copy())
                p.adapter.set_action(action)
                previous = tensor(p.object.data.root_state_w)[:, 3:7].clone()
                substeps = []
                for substep in range(p.adapter.steps_per_control):
                    p.advance()
                    m = p.measure()
                    codes = env.termination.failure_codes(m, p.origins,
                        control_boundary=substep == p.adapter.steps_per_control-1)
                    if not m['finite'].all() or (codes == 1).any():
                        raise FloatingPointError('Nonfinite physics in paired evaluation')
                    failure = torch.where(failure == 0, codes, failure)
                    inc = spin_increment(previous, m['object_state'][:, 3:7]).double()
                    previous = m['object_state'][:, 3:7].clone()
                    increment = torch.where(failure == 0, inc, 0.)
                    net += increment
                    valid_steps += failure == 0
                    substeps.append(increment)
                increments.append(torch.stack(substeps).cpu().numpy())
                failures.append(failure.cpu().numpy())
                noise = (torch.rand((p.num_envs, 16), generator=noise_rng, device=p.device)*2-1)*config['joint_noise_rad']
                noises.append(noise.cpu().numpy())
                q = tensor(p.hand.data.joint_pos)[:, p.active_joint_ids]
                core = torch.cat((2*(q+noise-env.lower)/(env.upper-env.lower)-1,
                                  p.adapter.commands[:, p.adapter.active]), dim=-1)
                env.history = torch.roll(env.history, -1, dims=1)
                env.history[:, -1] = core
                obs = env.observe()
                history = sensors.observe(obs)
            if control % 20 == 19 or control == controls-1:
                status.update(current_controller=controller, current_control=control+1,
                    requested_controls=controls, wall_s=time.monotonic()-started, updated_unix_s=time.time())
                atomic_json(out/'status.json', status)
                print('STUDENT_EVAL '+json.dumps({k: status[k] for k in
                    ('current_controller', 'current_control', 'requested_controls', 'wall_s')}), flush=True)
            if (failure != 0).all():
                break
        force_array, noise_array = np.asarray(forces), np.asarray(noises)
        if stimulus is None:
            stimulus = (force_array, noise_array)
        else:
            for original, actual in zip(stimulus, (force_array, noise_array)):
                count = min(len(original), len(actual))
                if not np.array_equal(original[:count], actual[:count]):
                    raise RuntimeError('Paired external-force/encoder-noise stimulus differs')
        net_deg = torch.rad2deg(net).cpu().numpy()
        valid_s = (valid_steps*p.dt).cpu().numpy()
        codes = failure.cpu().numpy()
        np.savez_compressed(out/f'{controller}_rollout.npz', action=np.asarray(actions),
            sensor_frame=np.asarray(histories), force_n=force_array, noise_rad=noise_array,
            valid_spin_increments_rad=np.asarray(increments), first_failure=np.asarray(failures),
            net_deg=net_deg, valid_s=valid_s)
        status['controllers'][controller] = summarize(net_deg, valid_s, codes, args.seconds)
        status['controllers'][controller].update(initial_state_max_error=error,
            rollout_sha256=sha256(out/f'{controller}_rollout.npz'))
        atomic_json(out/'status.json', status)
    baseline = status['controllers']['teacher']
    for name in models:
        row = status['controllers'][name]
        ratio = row['mean_net_deg']/baseline['mean_net_deg'] if baseline['mean_net_deg'] > 0 else None
        row.update(teacher_mean_ratio=ratio,
            meets_declared_simulation_threshold=(ratio is not None and ratio >= .9
                and row['retention_fraction'] >= baseline['retention_fraction']-.1))
    status.update(status='completed', phase='completed', paired_stimuli_verified=True,
        hardware_validated=False, wall_s=time.monotonic()-started, updated_unix_s=time.time())
    atomic_json(out/'comparison.json', status)
    atomic_json(out/'status.json', status)
    return status
