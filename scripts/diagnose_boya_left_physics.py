"""One-control left-hand physics diagnostic before policy evaluation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scene-config', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, default=Path(
        'outputs/boya_c_fresh1p5b_server_interrupted_recovery_20261028_v1/'
        'actions_1260000000/training/teacher_final.pth'))
    parser.add_argument('--left-asset', type=Path)
    parser.add_argument('--num-envs', type=int, default=9)
    parser.add_argument('--controller', choices=('zero', 'teacher'), default='zero')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False, device='cuda:0').app
    try:
        import torch
        from tendonspin.physics.hand_profiles import resolve_hand_profile
        from tendonspin.physics.learning_scene import load_scene_configuration
        from tendonspin.rl.isaac_spin_progress import SpinProgressBoyaEnv
        from tendonspin.students.model import TeacherLabels

        scene = load_scene_configuration(args.scene_config)
        profile = resolve_hand_profile(root, 'left', args.left_asset)
        teacher = TeacherLabels(args.teacher, 'cuda:0')
        config = deepcopy(teacher.contract['domain_randomization'])
        config['static_seed'] = 2026110203
        env = SpinProgressBoyaEnv(root, out, root/scene['training_cache'],
            num_envs=args.num_envs, trace_mode='summary', scene_configuration=scene,
            randomization_configuration=config, training_randomization=True,
            engine_profile=teacher.contract['engine_configuration']['profile'],
            hand_profile=profile)
        obs = env.reset()
        if args.controller == 'teacher':
            action = teacher(obs)[0]
        else:
            action = torch.zeros((env.num_envs, 16), device=env.device)
        env.physics.adapter.set_action(action)
        rows = []
        started = time.monotonic()
        for substep in range(env.physics.adapter.steps_per_control):
            q, v, effort = env.physics.advance()
            measure = env.physics.measure()
            codes = env.termination.failure_codes(
                measure, env.physics.origins,
                control_boundary=substep == env.physics.adapter.steps_per_control-1)
            speed = measure['max_speed']
            selected = int(speed.argmax())
            joint_ids = v.abs().argmax(-1)
            rows.append(dict(substep=substep,
                elapsed_s=(substep+1)*env.physics.dt,
                max_speed_rad_s=speed.cpu().numpy(),
                max_speed_joint=[env.physics.names[int(i)] for i in joint_ids.cpu()],
                selected_env=selected,
                selected_joint=env.physics.names[int(joint_ids[selected])],
                selected_joint_pos=float(q[selected, int(joint_ids[selected])]),
                selected_joint_vel=float(v[selected, int(joint_ids[selected])]),
                selected_effort=float(effort[selected, int(joint_ids[selected])]),
                normal_force_N=measure['max_normal'].cpu().numpy(),
                support_groups=measure['support_groups'].cpu().numpy(),
                coupling_error=measure['coupling_error'].cpu().numpy(),
                drift_mm=measure['drift_mm'].cpu().numpy(),
                tilt_deg=measure['tilt_deg'].cpu().numpy(),
                failure_codes=codes.cpu().numpy()))
            if (codes != 0).any():
                break
        status = dict(status='completed', wall_s=time.monotonic()-started,
            controller=args.controller, steps=len(rows),
            first_failure_by_env=rows[-1]['failure_codes'].tolist(),
            max_speed_rad_s=float(max(row['max_speed_rad_s'].max() for row in rows)),
            scene_configuration=str(args.scene_config),
            hand_profile=env.physics.hand_profile_identity)
        (out/'status.json').write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
        (out/'trace.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
        print(json.dumps(status), flush=True)
    finally:
        app.close()


if __name__ == '__main__':
    main()
