# Sources: TendonSpin archived diagnostic pair; original score_prefix (unchanged),
# independent SciPy Rotation world-vector calculation. No simulation or learning.
"""Compare rawPhysX/Lab/pose signals for the two user-approved configurations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DT = .0005


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    result = dict(schema=1, training_actions=0, new_evaluation_episodes=2,
        interpretation='Two separately evaluated configurations, one closed-loop episode each; no pooled angles, no statistical/causal solver proof',
        records={}, sources=[], analysis_source=dict(path=str(Path(__file__)), sha256=sha(Path(__file__))))
    traces = {}
    initials = {}
    for iterations in (4, 16):
        folder = out/f'velocity_{iterations:02d}'
        record = json.loads((folder/'result.json').read_text())
        assert record['status'] == 'completed'
        assert record['episode_resets'] == record['controller_switches'] == record['training_actions'] == 0
        with np.load(folder/'initial_state.npz') as f:
            initial = {k: f[k].copy() for k in f.files}
        initials[iterations] = initial
        keys = ['elapsed_s', 'valid', 'object_state', 'raw_pose_before_lab', 'raw_velocity_before_lab',
                'raw_pose_after_lab', 'raw_velocity_after_lab', 'support_groups', 'net_angle_deg',
                'drift_mm', 'tilt_deg', 'max_normal']
        blocks = {k: [] for k in keys}
        for path in sorted(folder.glob('physics_*.npz')):
            result['sources'].append(dict(path=str(path), sha256=sha(path)))
            with np.load(path) as f:
                for k in keys:
                    blocks[k].append(f[k])
        d = {k: np.concatenate(v) for k, v in blocks.items()}
        assert len(d['valid']) == record['physics_steps']
        rotation = Rotation.from_quat(d['raw_pose_after_lab'][:, 3:7])
        previous = Rotation.concatenate([Rotation.from_quat(initial['object_state'][0, 3:7]), rotation[:-1]])
        pose_omega = (rotation*previous.inv()).as_rotvec()/DT
        axes = rotation.apply([0, 0, 1])+previous.apply([0, 0, 1])
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        original_axis = -Rotation.from_quat(initial['object_state'][0, 3:7]).apply([0, 0, 1])
        d['omega_error_norm'] = np.linalg.norm(d['raw_velocity_after_lab'][:, 3:6]-pose_omega, axis=1)
        d['raw_fixed_rate'] = d['raw_velocity_after_lab'][:, 3:6]@original_axis
        d['pose_fixed_rate'] = pose_omega@original_axis
        d['pose_moving_rate'] = -(pose_omega*axes).sum(-1)
        with np.load(folder/'reconstructed_control_reward.npz') as f:
            reward = {k: f[k].copy() for k in f.files}
        d['reward'] = reward
        traces[iterations] = d
        valid = d['valid']
        component = lambda a, b: float(np.max(np.abs(a-b)))
        agreement = dict(
            raw_before_after_pose_max_abs=component(d['raw_pose_before_lab'], d['raw_pose_after_lab']),
            raw_before_after_velocity_max_abs=component(d['raw_velocity_before_lab'], d['raw_velocity_after_lab']),
            raw_vs_lab_pose_max_abs=component(d['raw_pose_after_lab'], d['object_state'][:, :7]),
            raw_vs_lab_velocity_max_abs=component(d['raw_velocity_after_lab'], d['object_state'][:, 7:13]))
        entry = dict(record=record, raw_read_agreement=agreement,
            raw_lab_read_agreement_within1e_6=all(x <= 1e-6 for x in agreement.values()),
            independent_pose_net_deg=float(d['pose_moving_rate'][valid].sum()*DT*180/np.pi),
            valid_prefix_drift_max_mm=float(d['drift_mm'][valid].max()),
            valid_prefix_tilt_max_deg=float(d['tilt_deg'][valid].max()),
            valid_prefix_normal_max_N=float(d['max_normal'][valid].max()), windows={})
        # Post-hoc interpretation of the observed terminal spin, never a score
        # filter or a replacement for the predeclared time windows.
        last_valid = np.flatnonzero(valid)[-1]
        valid_end = float(d['elapsed_s'][last_valid])
        final_angle = float(d['net_angle_deg'][last_valid])

        def angle_at(time_s):
            index = np.searchsorted(d['elapsed_s'], time_s+1e-10, side='right')-1
            return float(d['net_angle_deg'][index]) if index >= 0 else 0.

        visible = np.flatnonzero(valid & (d['support_groups'] > 0))
        last_contact = int(visible[-1]) if len(visible) else None
        tail = valid & (d['elapsed_s'] > valid_end-.2+1e-10)
        entry['posthoc_endpoint_motion'] = dict(
            purpose='Explain observed terminal motion; no contact-based score filtering',
            valid_end_s=valid_end, net_at25s_deg=angle_at(min(25., valid_end)),
            last_0p2s_net_deg=final_angle-angle_at(max(0., valid_end-.2)),
            last_0p2s_visible_contact_fraction=float((d['support_groups'][tail] > 0).mean()),
            last_visible_contact_s=float(d['elapsed_s'][last_contact]) if last_contact is not None else None,
            net_at_last_visible_contact_deg=float(d['net_angle_deg'][last_contact]) if last_contact is not None else None,
            net_after_last_visible_contact_deg=final_angle-float(d['net_angle_deg'][last_contact]) if last_contact is not None else None)
        log = out/f'velocity_{iterations:02d}.log'
        entry['runtime_iteration_warnings'] = [line for line in log.read_text(errors='replace').splitlines()
            if 'more than 4 velocity iterations' in line]
        result['sources'].append(dict(path=str(log), sha256=sha(log)))
        result['records'][iterations] = entry
        for path in (folder/'result.json', folder/'initial_state.npz', folder/'control_inputs.npz', folder/'reconstructed_control_reward.npz'):
            result['sources'].append(dict(path=str(path), sha256=sha(path)))
    a, b = (result['records'][i]['record'] for i in (4, 16))
    assert a['checkpoint_sha256'] == b['checkpoint_sha256']
    assert a['runtime_sources'] == b['runtime_sources']
    assert a['termination'] == b['termination']
    # The USD iteration snapshot also contains the scene prim. Compare authored
    # actor/articulation requests separately from the intentionally changed scene.
    def actor_requests(record):
        return {path: {k: v for k, v in attributes.items() if not k.startswith('physxScene:')}
            for path, attributes in record['actor_iteration_requests'].items()
            if any(not k.startswith('physxScene:') for k in attributes)}
    assert actor_requests(a) == actor_requests(b)
    result['authored_actor_iteration_requests_unchanged'] = True
    differences = {}
    for key in ('physics_cfg', 'scene_physics_attributes'):
        differences[key] = {k: [a[key].get(k), b[key].get(k)] for k in a[key].keys() | b[key].keys()
                            if a[key].get(k) != b[key].get(k)}
    assert differences['physics_cfg'] == {'min_velocity_iteration_count': [4, 16]}, differences
    assert differences['scene_physics_attributes'] == {'physxScene:minVelocityIterationCount': [4, 16]}, differences
    result['configuration_differences'] = differences
    result['initial_state_max_errors'] = {k: float(np.max(np.abs(initials[4][k]-initials[16][k])))
        for k in ('joint_pos', 'joint_vel', 'object_state', 'commands')}
    assert max(result['initial_state_max_errors'].values()) == 0
    common = min(a['valid_s'], b['valid_s'])
    for iterations, d in traces.items():
        for label, start, end in [('first5s', 0, 5), ('5to20s', 5, 20), ('20to30s', 20, 30), ('common_prefix', 0, common)]:
            m = (d['elapsed_s'] > start+1e-10) & (d['elapsed_s'] <= end+1e-10) & d['valid']
            entry = None
            if m.any():
                r = d['reward']
                c = (r['elapsed_s'] > start+1e-10) & (r['elapsed_s'] <= end+1e-10) & r['valid']
                contact = m & (d['support_groups'] > 0)
                entry = dict(start_s=start, actual_end_s=float(d['elapsed_s'][m][-1]), requested_end_s=end,
                    samples=int(m.sum()), visible_contact_fraction=float((d['support_groups'][m] > 0).mean()),
                    net_deg=float(d['pose_moving_rate'][m].sum()*DT*180/np.pi),
                    omega_pose_vector_rmse_rad_s=float(np.sqrt(np.mean(d['omega_error_norm'][m]**2))),
                    contact_omega_pose_vector_rmse_rad_s=float(np.sqrt(np.mean(d['omega_error_norm'][contact]**2))) if contact.any() else None,
                    raw_fixed_axis_mean_rad_s=float(d['raw_fixed_rate'][m].mean()),
                    pose_fixed_axis_mean_rad_s=float(d['pose_fixed_rate'][m].mean()),
                    pose_moving_axis_mean_rad_s=float(d['pose_moving_rate'][m].mean()),
                    reconstructed_reward_means={k: float(r[k][c].mean()) for k in
                        ('rotation', 'linear_cost', 'pose_cost', 'torque_cost', 'work_cost', 'reward')} if c.any() else None)
            result['records'][iterations]['windows'][label] = entry
    (out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for i, r in result['records'].items():
        print(json.dumps(dict(iterations=i, metrics=r['record']['metrics'], raw_agreement=r['raw_read_agreement'],
            windows=r['windows']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
