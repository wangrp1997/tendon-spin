"""Inspect archived teacher actions/rewards; no new physics integration.

Source: TendonSpin teacher-v2 reward in tendonspin/rl/environment.py, adapted
from botyard-inhand/research/learning/environment.py at commit
204d197a9606fb7266e884f3b2e6110195be01cb. This is a diagnostic, not a controller.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from tendonspin.rl.config import ROOT, digest, write
from tendonspin.physics.native import scene, active_ids


def inspect(episode):
    result = json.loads((episode / 'results.json').read_text())
    config = result['attribution']['config']
    execution = ROOT / result['execution']['path']
    trace_path = ROOT / result['policy_trace']['path']
    assert digest(execution) == result['execution']['sha256']
    assert digest(trace_path) == result['policy_trace']['sha256']
    assert not result['physical_failure']
    protocol_path = ROOT / result['protocol']['path']
    assert digest(protocol_path) == result['protocol']['sha256']
    protocol = json.loads(protocol_path.read_text())
    s, _, metadata = scene()
    assert metadata['scene_sha256'] == protocol['scene']['scene_sha256']
    assert metadata['source_scene_sha256'] == protocol['scene']['source_scene_sha256']
    ids = active_ids(s)
    qids = np.array([s.model.jnt_qposadr[s.model.actuator_trnid[i, 0]] for i in ids])
    lo, hi = s.model.actuator_ctrlrange[ids].T
    with np.load(execution, allow_pickle=False) as raw, np.load(trace_path, allow_pickle=False) as trace:
        dt = float(raw['physics_dt'])
        stride = round(config['control_dt'] / dt)
        angles = raw['angle_deg'][:result['valid_steps']]
        ctrl = raw['ctrl'][:result['valid_steps'], ids]
        q = raw['qpos'][:result['valid_steps'], qids]
        metrics = raw['metrics'][:result['valid_steps']]
        actions = trace['actions'].astype(np.float64)
        total = dict(rotation=0., position=0., tilt=0., tracking=0., action=0.)
        for j, action in enumerate(actions):
            start, end = j * stride, min((j + 1) * stride, len(angles))
            if end <= start:
                break
            met = metrics[start:end]
            duration = (end - start) * dt
            increment = angles[end - 1] - (angles[start - 1] if start else 0.)
            total['rotation'] += config['rotation_reward_per_rad'] * duration * np.clip(np.radians(increment) / duration, -config['reward_spin_cap_rad_s'], config['reward_spin_cap_rad_s'])
            total['position'] += duration * config['position_cost_per_s'] * np.mean((met[:, 0] / 3.) ** 2)
            total['tilt'] += duration * config['tilt_cost_per_s'] * np.mean((met[:, 1] / 10.) ** 2)
            total['tracking'] += duration * config['tracking_cost_per_s'] * np.mean(((ctrl[start:end] - q[start:end]) / .1) ** 2)
            total['action'] += duration * config['action_cost_per_s'] * np.mean(action ** 2)
        reproduced_return = total['rotation'] - sum(total[k] for k in ('position', 'tilt', 'tracking', 'action'))
        error = abs(reproduced_return - result['return_'])
        assert error < 1e-8
        segments = []
        for first, last in ((0, 5), (5, 30), (30, 90), (90, 120)):
            if last > result['valid_seconds']:
                continue
            start, end = round(first / dt), round(last / dt)
            ai, aj = round(first / config['control_dt']), round(last / config['control_dt'])
            target, actual = ctrl[start:end], q[start:end]
            on_low, on_high = abs(target - lo) < 1e-6, abs(target - hi) < 1e-6
            limited = on_low | on_high
            step_targets = target[stride - 1::stride]
            step_low, step_high = abs(step_targets - lo) < 1e-6, abs(step_targets - hi) < 1e-6
            outward = (step_low & (actions[ai:aj] < 0)) | (step_high & (actions[ai:aj] > 0))
            segments.append(dict(first_seconds=first, last_seconds=last,
                net_deg=float(angles[end - 1] - (angles[start - 1] if start else 0.)),
                mean_abs_action=float(abs(actions[ai:aj]).mean()),
                mean_command_movement_per_motor_rad=float(abs(np.diff(target, axis=0)).sum(axis=0).mean()),
                mean_actual_movement_per_motor_rad=float(abs(np.diff(actual, axis=0)).sum(axis=0).mean()),
                at_command_limit_fraction=limited.mean(0).tolist(),
                outward_action_at_limit_fraction=outward.mean(0).tolist()))
    names = [s.model.actuator(int(i)).name for i in ids]
    later = [v for v in segments if v['first_seconds'] >= 30]
    at_bound_outward = (np.all([np.array(v['at_command_limit_fraction']) == 1 for v in later], axis=0)
                       & np.all([np.array(v['outward_action_at_limit_fraction']) == 1 for v in later], axis=0)) if later else np.zeros(len(ids), dtype=bool)
    saturated_names = [name for name, limited in zip(names, at_bound_outward) if limited]
    return dict(kind='archived trace diagnosis', new_physics_integration_steps=0,
        source=result['execution'], policy_trace=result['policy_trace'],
        controller=result['controller'], training_update=result['training_updates'],
        actuator_names=names,
        after30s_constant_outward_at_command_bound_names=saturated_names,
        limit_definition='Position-command software bounds; not evidence all physical joints are at hard stops',
        final_target_rad=ctrl[-1].tolist(), final_actual_position_rad=q[-1].tolist(),
        reward_components={k: float(v) for k, v in total.items()},
        stored_return=result['return_'], reconstructed_return=float(reproduced_return),
        return_reconstruction_error=float(error), segments=segments,
        diagnosis=f'{len(saturated_names)} targets remain at command bounds throughout the inspected segments after30s with continuing outward policy actions; inspect interval motion separately. This trace does not establish repeated gait.',
        causal_limits='Command clipping is directly verified; relative roles of reward, exploration, training duration and physical contact require separate controlled experiments.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.episode)
    write(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
