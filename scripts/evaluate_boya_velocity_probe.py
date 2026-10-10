# Derived from TendonSpin scripts/evaluate_boya_hora.py; exact original identity
# is recorded in the source manifest. Hora v0.0.1 ActorCritic/RunningMeanStd and
# reward (HaozhiQi/hora, MIT; Qi et al., CoRL2022, arXiv:2210.04887).
# Reuse BoyaParallel.advance/measure and declared termination/scoring unchanged.
# Port change: raw PhysX getter instrumentation and one explicit scene velocity
# iteration lower-bound override, only for the user-approved diagnostic pair.
"""One frozen original-state30s episode, with raw velocity/pose provenance."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROTOCOL = ROOT/'docs/experiments/2026-10-10-boya-velocity-diagnostic/PROTOCOL.md'
EXPECTED_CHECKPOINT = '9781442fbfe0865946b7d3268a1ad379f796ca967aa8997f6dc427b8263f226e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration_json(value):
    """Encode authored non-finite configuration limits, never measured state."""
    if isinstance(value, dict):
        return {str(k): configuration_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [configuration_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--velocity-iterations', type=int, choices=(4, 16), required=True)
    args = parser.parse_args()
    run = ROOT/'outputs/boya_hora1024_workspace50m_v1/actions_020000000'
    checkpoint = run/'training/teacher_final.pth'
    cache = ROOT/'outputs/boya_settled_cache_v2/grasp_cache.npz'
    training = json.loads((run/'training/result.json').read_text())
    assert sha(checkpoint) == EXPECTED_CHECKPOINT
    for source in training['sources']:
        if sha(ROOT/source['path']) != source['sha256']:
            raise RuntimeError('Original source identity mismatch: '+source['path'])
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.nice(10)
    started = time.monotonic()
    record = dict(controller='frozen20M_raw_velocity_probe_v1',
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        training_actions_in_checkpoint=training['actions_executed'],
        configuration=f'TGS_position16_velocity_min{args.velocity_iterations}',
        requested_s=30., wall_budget_s=1500., seed=43, headless=True, enable_cameras=False,
        initial_state='original grasp44, not cache', physics_steps=0, valid_steps=0,
        actual_s=0., valid_s=0., net_deg=0., episode_resets=0, controller_switches=0,
        training_actions=0, benchmark_validated=False, automatic_resource_stop=False,
        stop_reason='time limit', phase='setup', sources=training['sources'],
        diagnostic_sources=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in
            (Path(__file__), PROTOCOL, ROOT/'scripts/evaluate_boya_hora.py')])
    source_out = out/'sources'
    for p in (Path(__file__), PROTOCOL):
        target = source_out/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())

    def save(status):
        record.update(status=status, wall_s=time.monotonic()-started)
        serialized = json.dumps(record, indent=2, allow_nan=False)+'\n'
        temporary = out/'result.json.tmp'
        temporary.write_text(serialized)
        temporary.replace(out/'result.json')
        print('VELOCITY_PROBE '+json.dumps({k: record[k] for k in
            ('status', 'configuration', 'actual_s', 'valid_s', 'net_deg', 'stop_reason', 'wall_s')}), flush=True)

    save('launching')
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False, device='cuda:0').app
    failed = False
    frames, controls, reward_rows, angles = [], [], [], []
    chunks = 0
    try:
        import inspect
        import numpy as np
        import torch
        from tendonspin.physics import isaac_parallel as parallel
        from tendonspin.physics.isaac_parallel import tensor
        from tendonspin.baselines.reference_models import build_model, reference_module
        from tendonspin.evaluation.rotation import score_prefix
        from tendonspin.rl.termination import legacy_failure_codes
        torch.set_num_threads(4)
        torch.manual_seed(43)
        np.random.seed(43)
        original_factory = parallel.PhysxCfg

        def declared_factory(*factory_args, **kwargs):
            if kwargs.get('min_velocity_iteration_count') != 4:
                raise RuntimeError('Unexpected original iteration request')
            kwargs['min_velocity_iteration_count'] = args.velocity_iterations
            return original_factory(*factory_args, **kwargs)

        parallel.PhysxCfg = declared_factory
        from tendonspin.rl.isaac_hora import HoraBoyaEnv, spin_increment
        env = HoraBoyaEnv(ROOT, out, cache, num_envs=1, termination_profile='boya_workspace')
        p = env.physics
        parallel.PhysxCfg = original_factory
        record['termination'] = env.termination.spec.to_dict()
        assert record['termination'] == training['termination']
        stage = p.sim.stage
        scene_prim = stage.GetPrimAtPath(p.cfg.physics_prim_path)
        record['scene_physics_attributes'] = {a.GetName(): a.Get() for a in scene_prim.GetAttributes()
            if a.GetName().startswith('physxScene:') and isinstance(a.Get(), (str, int, float, bool))}
        assert record['scene_physics_attributes']['physxScene:minVelocityIterationCount'] == args.velocity_iterations
        assert record['scene_physics_attributes']['physxScene:minPositionIterationCount'] == 16
        record['scene_physics_attributes'] = configuration_json(record['scene_physics_attributes'])
        record['actor_iteration_requests'] = configuration_json({str(prim.GetPath()): {
            a.GetName(): a.Get() for a in prim.GetAttributes() if 'IterationCount' in a.GetName()}
            for prim in stage.Traverse() if any('IterationCount' in a.GetName() for a in prim.GetAttributes())})
        record['physics_cfg'] = configuration_json(p.cfg.physics.to_dict())
        record['timing'] = dict(physics_dt=p.dt, control_steps=p.adapter.steps_per_control, control_hz=20)
        manager_type = p.sim.physics_manager if inspect.isclass(p.sim.physics_manager) else type(p.sim.physics_manager)
        runtime_paths = {
            Path(inspect.getfile(type(p.object.data))), Path(inspect.getfile(type(p.object))),
            Path(inspect.getfile(type(p.sim))), Path(inspect.getfile(type(p.scene))),
            Path(inspect.getfile(manager_type)), Path(inspect.getfile(p.object.root_view.__class__)),
        }
        record['runtime_sources'] = [dict(path=str(path), sha256=sha(path)) for path in sorted(runtime_paths)]
        p.reset_nominal()
        env.history[:] = env._frame(noise=False)[:, None, :]
        weights = torch.load(checkpoint, map_location=p.device, weights_only=False)
        model = build_model().to(p.device)
        model.load_state_dict(weights['model'])
        model.eval()
        norm = reference_module('hora_normalizer').RunningMeanStd((96,)).to(p.device)
        norm.load_state_dict(weights['running_mean_std'])
        norm.eval()
        record.update(joint_names=p.names, body_names=list(p.hand.body_names), filter_names=p.filter_names)
        initial = p.measure()
        initial_np = {k: v.cpu().numpy().copy() for k, v in initial.items()}
        initial_np['commands'] = p.adapter.commands.cpu().numpy().copy()
        np.savez(out/'initial_state.npz', **initial_np)
        with np.load(run/'evaluation/initial_state.npz') as original:
            record['initial_state_max_errors'] = {k: float(np.max(np.abs(initial_np[k]-original[k])))
                for k in ('joint_pos', 'joint_vel', 'object_state', 'commands')}
        assert max(record['initial_state_max_errors'].values()) == 0, record['initial_state_max_errors']
        prev = initial['object_state'][:, 3:7].clone()
        net, legacy_steps, legacy_stopped, legacy_reason = 0., 0, False, 'time limit'
        record['phase'] = 'episode'
        save('evaluating')
        for step in range(round(30./p.dt)):
            if step % p.adapter.steps_per_control == 0:
                if time.monotonic()-started >= 1500.:
                    record['stop_reason'] = 'wall budget'
                    break
                obs = env.observe()
                with torch.no_grad():
                    action = model.act_inference(dict(obs=norm(obs['obs']), priv_info=obs['priv_info'])).clamp(-1, 1)
                controls.append(dict(step=step, obs=obs['obs'].cpu().numpy()[0],
                    priv_info=obs['priv_info'].cpu().numpy()[0], action=action.cpu().numpy()[0]))
                p.adapter.set_action(action)
            q, v, effort = p.advance()
            # Clone each API buffer before another getter can overwrite its view.
            raw_pose_before = tensor(p.object.root_view.get_transforms()).clone()
            raw_velocity_before = tensor(p.object.root_view.get_velocities()).clone()
            m = p.measure()
            raw_pose_after = tensor(p.object.root_view.get_transforms()).clone()
            raw_velocity_after = tensor(p.object.root_view.get_velocities()).clone()
            code = int(env.termination.failure_codes(m, p.origins,
                control_boundary=(step+1) % p.adapter.steps_per_control == 0)[0])
            valid = code == 0
            legacy_code = int(legacy_failure_codes(m)[0])
            if not legacy_stopped:
                if legacy_code:
                    legacy_stopped, legacy_reason = True, env.reasons[legacy_code]
                else:
                    legacy_steps += 1
            if valid:
                net += float(torch.rad2deg(spin_increment(prev, m['object_state'][:, 3:7]))[0])
                record['valid_steps'] += 1
            prev = m['object_state'][:, 3:7].clone()
            angles.append(net)
            frame = {k: value.detach().cpu().numpy()[0].copy() for k, value in m.items()}
            frame.update(joint_pos_before=q.cpu().numpy()[0].copy(), joint_vel_before=v.cpu().numpy()[0].copy(),
                action=action.cpu().numpy()[0].copy(), commands=p.adapter.commands.cpu().numpy()[0].copy(),
                motor_effort_requested=effort.cpu().numpy()[0].copy(),
                actuator_effort_forwarded=tensor(p.hand.actuators.applied_effort).cpu().numpy()[0].copy(),
                raw_pose_before_lab=raw_pose_before.cpu().numpy()[0].copy(),
                raw_velocity_before_lab=raw_velocity_before.cpu().numpy()[0].copy(),
                raw_pose_after_lab=raw_pose_after.cpu().numpy()[0].copy(),
                raw_velocity_after_lab=raw_velocity_after.cpu().numpy()[0].copy(),
                valid=valid, failure_code=code, legacy_strict_code=legacy_code,
                legacy_strict_valid=not legacy_stopped, net_angle_deg=net, elapsed_s=(step+1)*p.dt)
            frames.append(frame)
            record.update(physics_steps=step+1, actual_s=(step+1)*p.dt,
                valid_s=record['valid_steps']*p.dt, net_deg=net)
            if (step+1) % p.adapter.steps_per_control == 0:
                active = p.active_joint_ids
                tau = effort[:, active]
                velocity = (m['joint_pos'][:, active]-q[:, active])/p.dt
                pose = ((m['joint_pos'][:, active]-env.init_q)**2).sum(-1)
                torque = tau.square().sum(-1)
                work = (tau*velocity).sum(-1).square()
                reward, rot, lin = env.reward_fn(m['object_state'][:, 7:10], -.3, m['object_state'][:, 10:13],
                    env.rotation_axis, 1., .5, -.5, pose, -.3, torque, -.1, work, -2.)
                reward_rows.append(dict(elapsed_s=(step+1)*p.dt, valid=valid, reward=float(reward[0]),
                    rotation=float(rot[0]), linear_cost=float(-.3*lin[0]), pose_cost=float(-.3*pose[0]),
                    torque_cost=float(-.1*torque[0]), work_cost=float(-2*work[0])))
            if len(frames) >= 2000:
                np.savez(out/f'physics_{chunks:03d}.npz', **{k: np.asarray([f[k] for f in frames]) for k in frames[0]})
                frames = []
                chunks += 1
            if not valid:
                record['stop_reason'] = env.reasons[code]
                break
            if (step+1) % p.adapter.steps_per_control == 0:
                env.history = torch.roll(env.history, -1, dims=1)
                env.history[:, -1] = env._frame()
            if (step+1) % 10000 == 0:
                save('evaluating')
        record['metrics'] = score_prefix(angles, p.dt, record['valid_steps'], window_s=30.)
        record['legacy_strict_shadow'] = dict(stop_reason=legacy_reason if legacy_stopped else record['stop_reason'],
            metrics=score_prefix(angles, p.dt, min(legacy_steps, record['valid_steps']), window_s=30.))
    except BaseException as error:
        failed = True
        traceback.print_exc()
        record.update(stop_reason='process error', error=dict(type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc()))
    finally:
        try:
            if frames:
                np.savez(out/f'physics_{chunks:03d}.npz', **{k: np.asarray([f[k] for f in frames]) for k in frames[0]})
                chunks += 1
            if controls:
                np.savez(out/'control_inputs.npz', **{k: np.asarray([f[k] for f in controls]) for k in controls[0]})
            if reward_rows:
                np.savez(out/'reconstructed_control_reward.npz', **{k: np.asarray([f[k] for f in reward_rows]) for k in reward_rows[0]})
            if angles:
                np.save(out/'angles_deg.npy', np.asarray(angles))
            record['physics_trace_chunks'] = chunks
            save('error' if failed else 'completed')
        finally:
            app.close(exit_code=1 if failed else 0)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
