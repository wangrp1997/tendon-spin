"""Record one controller's measured hand/object poses for visual playback."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--side', choices=('left', 'right'), required=True)
    parser.add_argument('--actions', type=Path, required=True)
    parser.add_argument('--fixed-cases', type=Path, required=True)
    parser.add_argument('--scene-config', type=Path)
    parser.add_argument('--case-index', type=int, default=9)
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--seed', type=int, default=2026110203)
    parser.add_argument('--teacher', type=Path, default=Path(
        'outputs/boya_c_fresh1p5b_server_interrupted_recovery_20261028_v1/'
        'actions_1260000000/training/teacher_final.pth'))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    actions_path = args.actions if args.actions.is_absolute() else ROOT/args.actions
    fixed_path = args.fixed_cases if args.fixed_cases.is_absolute() \
        else ROOT/args.fixed_cases
    teacher_path = args.teacher if args.teacher.is_absolute() else ROOT/args.teacher

    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False, device='cuda:0').app
    started = time.monotonic()
    status = dict(status='running', phase='starting',
                  started_unix_s=time.time(), side=args.side,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    try:
        import torch
        from tendonspin.physics.hand_profiles import resolve_hand_profile
        from tendonspin.physics.isaac_parallel import tensor
        from tendonspin.physics.learning_scene import load_scene_configuration
        from tendonspin.physics.physx_property_sync import install
        from tendonspin.rl.isaac_hora import spin_increment
        from tendonspin.rl.isaac_spin_progress import SpinProgressBoyaEnv
        from tendonspin.students.training import SELECTED_TEACHER_SHA256

        if digest(teacher_path) != SELECTED_TEACHER_SHA256:
            raise ValueError('Selected C1260M teacher identity mismatch')
        status['property_sync'] = install(lambda name: None)
        profile = resolve_hand_profile(ROOT, args.side)
        teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
        config = deepcopy(teacher['contract']['domain_randomization'])
        config['static_seed'] = args.seed
        if args.scene_config is None:
            scene = profile.adapt_scene_configuration(
                teacher['contract']['scene_configuration'])
            scene_path = None
        else:
            scene_path = args.scene_config if args.scene_config.is_absolute() \
                else ROOT/args.scene_config
            scene = load_scene_configuration(scene_path)
            if scene.get('hand_side', 'right') != args.side:
                raise ValueError('Scene hand side differs from recorder side')
        with np.load(actions_path, allow_pickle=False) as saved:
            actions = saved['action'].copy()
        with np.load(fixed_path, allow_pickle=False) as saved:
            fixed = {name: saved[name].copy() for name in saved.files}
        if not 0 <= args.case_index < actions.shape[1]:
            raise ValueError('Case index out of range')
        case = args.case_index
        cache_path = ROOT/scene['training_cache']
        with np.load(cache_path, allow_pickle=False) as saved:
            cache_object_state = profile.adapt_object_state(
                saved['object_state'].copy())

        selected_scale = float(fixed['scale'][case])
        nominal_scale = float(config['logical_nominal_scale'])
        size_ratio = selected_scale/nominal_scale
        video_scene = deepcopy(scene)
        video_scene['geometry']['diameter_m'] *= size_ratio
        video_scene['geometry']['height_m'] *= size_ratio
        video_scene['geometry']['dimensions_m'] = [
            video_scene['geometry']['diameter_m']] * 2 + [
            video_scene['geometry']['height_m']]
        config['scale_centers'] = list(config['scale_centers'])
        config['scale_centers'][0] = selected_scale
        config['logical_nominal_scale'] = selected_scale

        env = SpinProgressBoyaEnv(
            ROOT, out, cache_path, num_envs=1, trace_mode='summary',
            scene_configuration=video_scene,
            randomization_configuration=config, training_randomization=False,
            engine_profile=teacher['contract']['engine_configuration']['profile'],
            hand_profile=profile)
        p = env.physics
        env.reset()
        q = torch.as_tensor(fixed['joint_pos'][case], device=p.device)
        obj = torch.as_tensor(fixed['object_state'][case], device=p.device)
        cache_index = int(fixed['cache_index'][case])
        source_object = cache_object_state[cache_index]
        original_origin = np.asarray(fixed['object_state'][case][:3])-source_object[:3]
        obj[:3] -= torch.as_tensor(original_origin, device=p.device)
        commands = torch.as_tensor(fixed['commands'][case], device=p.device)
        p._write_state(
            q[None], torch.zeros_like(q)[None], obj[None], commands[None])
        env.cache_ids[0] = cache_index
        env.dr.mass[0] = float(fixed['mass_kg'][case])
        env.dr.p_ratio[0] = torch.as_tensor(
            fixed['p_ratio'][case], device=p.device)
        env.dr.d_ratio[0] = torch.as_tensor(
            fixed['d_ratio'][case], device=p.device)
        env.dr.friction[0] = float(fixed['friction'][case])
        env.dr.com[0] = torch.as_tensor(
            fixed['com_m'][case], device=p.device)
        p.object.set_masses_index(
            masses=env.dr.mass[:1, None],
            env_ids=torch.tensor([0], device=p.device))
        coms = tensor(p.object.data.body_com_pose_b).clone()
        coms[..., :3] = env.dr.com
        p.object.set_coms_index(
            coms=coms, env_ids=torch.tensor([0], device=p.device))
        import warp as wp
        materials = tensor(p.object.root_view.get_material_properties()).clone()
        materials[..., 0] = env.dr.friction
        materials[..., 1] = env.dr.friction
        ids_cpu = torch.zeros(1, dtype=torch.int32)
        p.object.root_view.set_material_properties(
            wp.from_torch(materials.contiguous(), dtype=wp.float32),
            wp.from_torch(ids_cpu, dtype=wp.int32))
        active_motors = p.adapter.active
        p.adapter.kp[:, active_motors] = (
            env.dr.base_kp[active_motors][None] * env.dr.p_ratio[:1])
        p.adapter.motor_damping[:, active_motors] = (
            env.dr.base_damping[active_motors][None] * env.dr.d_ratio[:1])
        p.sim.forward()

        measure = p.measure()
        records = [dict(
            body_pose=tensor(measure['body_pose'])[0].cpu().numpy().copy(),
            object_state=tensor(measure['object_state'])[0].cpu().numpy().copy(),
            net_deg=0., valid=True)]
        controls = min(
            round(args.seconds/(p.dt*p.adapter.steps_per_control)),
            actions.shape[0])
        net = torch.zeros(1, device=p.device, dtype=torch.float64)
        valid_steps = torch.zeros(1, device=p.device, dtype=torch.long)
        failure = torch.zeros_like(valid_steps)
        status.update(phase='rollout', requested_controls=controls)
        for control in range(controls):
            with torch.no_grad():
                action = torch.as_tensor(
                    actions[control, case], device=p.device)[None]
                p.adapter.set_action(action)
                previous = tensor(
                    p.object.data.root_state_w)[:, 3:7].clone()
                for substep in range(p.adapter.steps_per_control):
                    _, _, _ = p.advance()
                    measure = p.measure()
                    codes = env.termination.failure_codes(
                        measure, p.origins,
                        control_boundary=(
                            substep == p.adapter.steps_per_control-1))
                    failure = torch.where(failure == 0, codes, failure)
                    increment = spin_increment(
                        previous,
                        measure['object_state'][:, 3:7]).double()
                    previous = measure['object_state'][:, 3:7].clone()
                    net += torch.where(failure == 0, increment, 0.)
                    valid_steps += failure == 0
                records.append(dict(
                    body_pose=tensor(
                        measure['body_pose'])[0].cpu().numpy().copy(),
                    object_state=tensor(
                        measure['object_state'])[0].cpu().numpy().copy(),
                    net_deg=float(torch.rad2deg(net[0]).cpu()),
                    valid=bool(failure[0] == 0)))
            if control % 20 == 19:
                status.update(
                    current_control=control+1,
                    wall_s=time.monotonic()-started)
                print('POSE_TRACK '+json.dumps(dict(
                    phase='rollout', side=args.side,
                    control=control+1, requested=controls)), flush=True)
            if bool(failure[0]):
                break

        body_pose = np.stack([record['body_pose'] for record in records])
        object_state = np.stack([record['object_state'] for record in records])
        net_deg = np.asarray([record['net_deg'] for record in records])
        valid = np.asarray([record['valid'] for record in records])
        np.savez_compressed(
            out/'pose_track.npz',
            body_pose=body_pose,
            object_state=object_state,
            net_deg=net_deg,
            valid=valid,
            body_names=np.asarray(list(p.hand.body_names)),
            selected_scale=np.asarray(selected_scale),
            diameter_m=np.asarray(video_scene['geometry']['diameter_m']),
            height_m=np.asarray(video_scene['geometry']['height_m']))
        status.update(
            status='completed',
            phase='completed',
            side=args.side,
            case_index=case,
            controls=controls,
            selected_net_deg=float(torch.rad2deg(net[0]).cpu()),
            controls_completed=len(records)-1,
            selected_scale=selected_scale,
            body_names=list(p.hand.body_names),
            pose_track=str(out/'pose_track.npz'),
            pose_track_sha256=digest(out/'pose_track.npz'),
            scene_configuration=str(scene_path) if scene_path else None,
            hand_profile=profile.identity(),
            wall_s=time.monotonic()-started)
    except BaseException as error:
        import traceback
        status.update(
            status='error',
            error=str(error),
            traceback=traceback.format_exc())
        raise
    finally:
        (out/'status.json').write_text(
            json.dumps(status, indent=2)+'\n')
        app.close(exit_code=0 if status['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
