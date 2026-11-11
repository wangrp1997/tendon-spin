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
from tendonspin.physics.hand_profiles import resolve_hand_profile
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


def side_normalized_spin_mean(mean_net_deg, hand_side):
    """Return the right-hand-canonical signed spin for a bilateral scene."""
    if hand_side not in ('right', 'left'):
        raise ValueError('hand_side must be right or left')
    return float(mean_net_deg if hand_side == 'right' else -mean_net_deg)


def declared_transfer_metrics(student, teacher, hand_side):
    """Compare spin magnitude and retention in the hand's canonical direction."""
    student_spin = side_normalized_spin_mean(student['mean_net_deg'], hand_side)
    teacher_spin = side_normalized_spin_mean(teacher['mean_net_deg'], hand_side)
    ratio = (student_spin/teacher_spin) if teacher_spin > 0 else None
    retention_drop = teacher['retention_fraction']-student['retention_fraction']
    return dict(
        side_normalized_mean_net_deg=student_spin,
        teacher_side_normalized_mean_net_deg=teacher_spin,
        teacher_mean_ratio=ratio,
        retention_fraction_delta=-retention_drop,
        meets_declared_simulation_threshold=(
            ratio is not None and ratio >= .9 and retention_drop <= .1))


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
    hand_profile = resolve_hand_profile(root, getattr(args, 'hand_side', 'right'),
        getattr(args, 'left_asset', None))
    hand_profile_identity = hand_profile.identity()
    trained_hand_profile = training_status.get('hand_profile')
    if trained_hand_profile is not None:
        identity_keys = ('side', 'asset_path', 'asset_sha256', 'asset_dependencies',
                         'source_prim', 'joint_prefix', 'body_prefix', 'mirror_world_y')
        if any(trained_hand_profile.get(key) != hand_profile_identity.get(key)
               for key in identity_keys):
            raise ValueError('Evaluation hand asset/profile differs from training')
    right_hand = hand_profile.side == 'right'
    if right_hand:
        for source in training_status['sources']:
            if sha256(root/source['path']) != source['sha256']:
                raise RuntimeError('Frozen training dependency changed: '+source['path'])
    teacher = TeacherLabels(args.teacher, 'cuda:0')
    if teacher.checkpoint_sha256 != SELECTED_TEACHER_SHA256:
        raise ValueError('Wrong C teacher identity')
    scene_config_path = getattr(args, 'scene_config', None)
    if scene_config_path is not None:
        from tendonspin.physics.learning_scene import load_scene_configuration
        scene = load_scene_configuration(
            scene_config_path if scene_config_path.is_absolute() else root/scene_config_path)
        if scene.get('hand_side', 'right') != hand_profile.side:
            raise ValueError('Scene hand_side differs from --hand-side')
    else:
        scene = hand_profile.adapt_scene_configuration(teacher.contract['scene_configuration'])
    controllers = tuple(getattr(args, 'controllers', ('teacher', 'touch', 'proprio')))
    model_names = tuple(name for name in ('touch', 'proprio') if name in controllers)
    cache = root/scene['training_cache']
    if sha256(cache) != scene.get('training_cache_sha256', teacher.contract['cache_sha256']):
        raise ValueError('Evaluation grasp cache changed')
    models = {}
    metadata = {}
    for name in model_names:
        models[name], metadata[name] = load_student(args.training/f'{name}_best.pth', 'cuda:0')
        if metadata[name]['teacher_sha256'] != teacher.checkpoint_sha256 or models[name].modality != name:
            raise ValueError('Selected student identity mismatch')
    config = deepcopy(teacher.contract['domain_randomization'])
    config['static_seed'] = args.seed
    env = SpinProgressBoyaEnv(root, out, cache,
        num_envs=args.num_envs, trace_mode='summary', scene_configuration=scene,
        randomization_configuration=config, training_randomization=True,
        engine_profile=teacher.contract['engine_configuration']['profile'],
        hand_profile=hand_profile)
    p, dr = env.physics, env.dr
    if (trained_hand_profile is not None and trained_hand_profile.get('usd_structure')
            != p.hand_profile_identity.get('usd_structure')):
        raise ValueError('Evaluation USD structure differs from training')
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
        hand_profile=env.physics.hand_profile_identity,
        scene_configuration=scene,
        historical_training_sources_verified=right_hand,
        migration_note=('Left-hand closed-loop transfer; original right-hand training sources '
                        'are identity evidence, not an unchanged left environment.') if not right_hand else None,
        teacher_sha256=teacher.checkpoint_sha256,
        student_sha256={name: sha256(args.training/f'{name}_best.pth') for name in models},
        controllers={}, training_updates=0, resets_within_case=0, teacher_takeovers=0)
    control_dt = p.dt*p.adapter.steps_per_control
    controls = round(args.seconds/control_dt)
    if abs(controls*control_dt-args.seconds) > 1e-9:
        raise ValueError('Evaluation duration must contain whole controls')
    started = time.monotonic()
    stimulus = None
    initial_physics = {}
    for controller in controllers:
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
        initial_physics[controller] = dict(
            max_speed_rad_s=float(measure['max_speed'].max()),
            max_normal_force_N=float(measure['max_normal'].max()),
            support_groups=[int(value) for value in measure['support_groups'].cpu()],
            coupling_error=float(measure['coupling_error'].max()),
            drift_mm=float(measure['drift_mm'].max()),
            tilt_deg=float(measure['tilt_deg'].max()))
        error = max(float((measure['joint_pos']-initial_q).abs().max()),
                    float((measure['object_state']-initial_obj).abs().max()),
                    float((p.adapter.commands-initial_commands).abs().max()))
        if error > 1e-6:
            raise RuntimeError('Paired initial-state readback failed')
        net = torch.zeros(p.num_envs, device=p.device, dtype=torch.float64)
        valid_steps = torch.zeros(p.num_envs, device=p.device, dtype=torch.long)
        failure = torch.zeros_like(valid_steps)
        forces, noises, actions, histories, increments, failures = [], [], [], [], [], []
        debug_rows = []
        for control in range(controls):
            if (out.parent/'STOP').exists():
                raise InterruptedError('User stop file detected')
            with torch.no_grad():
                action = teacher(obs)[0] if controller == 'teacher' else models[controller](history)['action']
                if getattr(args, 'zero_actions', False):
                    action = torch.zeros_like(action)
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
                    _, _, effort = p.advance()
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
                    if getattr(args, 'debug_physics', False):
                        selected = int(m['max_speed'].argmax())
                        joint_id = int(m['joint_vel'][selected].abs().argmax())
                        body_force = m['normal_force_matrix_w'][selected].norm(dim=-1)
                        force_order = body_force.argsort(descending=True)
                        top_forces = {
                            p.filter_names[int(index)]: float(body_force[int(index)])
                            for index in force_order[:8]
                            if float(body_force[int(index)]) > 1e-6}
                        debug_rows.append(dict(
                            controller=controller, control=control, substep=substep,
                            elapsed_s=(control*p.adapter.steps_per_control+substep+1)*p.dt,
                            max_speed_rad_s=float(m['max_speed'].max()),
                            joint=p.names[joint_id],
                            joint_vel=float(m['joint_vel'][selected, joint_id]),
                            effort=float(effort[selected, joint_id]),
                            normal_force_N=float(m['max_normal'].max()),
                            contact_forces_by_body_N=top_forces,
                            support_groups=int(m['support_groups'].max()),
                            coupling_error=float(m['coupling_error'].max()),
                            drift_mm=float(m['drift_mm'].max()),
                            tilt_deg=float(m['tilt_deg'].max()),
                            failure_code=int(codes.max())))
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
        status['controllers'][controller]['side_normalized_mean_net_deg'] = (
            side_normalized_spin_mean(
                status['controllers'][controller]['mean_net_deg'],
                hand_profile.side))
        if debug_rows:
            status['controllers'][controller]['physics_debug'] = debug_rows
        status['controllers'][controller].update(initial_state_max_error=error,
            rollout_sha256=sha256(out/f'{controller}_rollout.npz'))
        atomic_json(out/'status.json', status)
    baseline = status['controllers'].get('teacher')
    for name in models:
        row = status['controllers'][name]
        if baseline is not None:
            row.update(declared_transfer_metrics(
                row, baseline, hand_profile.side))
        else:
            row.update(teacher_mean_ratio=None,
                meets_declared_simulation_threshold=None)
    status.update(status='completed', phase='completed', paired_stimuli_verified=True,
        initial_physics=initial_physics,
        hardware_validated=False, wall_s=time.monotonic()-started, updated_unix_s=time.time())
    atomic_json(out/'comparison.json', status)
    atomic_json(out/'status.json', status)
    return status
