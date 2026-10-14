# References: TendonSpin821b8ab analyze_boya_small_budget.py (MIT),
# github.com/wangrp1997/tendon-spin; HaozhiQi/hora v0.0.1 (MIT), arXiv:2210.04887.
# Offline candidate uses the moving-axis metric and positive-only reward cap.
# SciPy independently reconstructs saved poses; no engine, learner or optimizer.
"""Validate a separate self-spin reward against existing frozen Isaac traces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from tendonspin.rl.spin_progress_reward import CONFIG, SpinProgressRewardSignal

ROOT = Path(__file__).resolve().parents[1]


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_episode(folder, config):
    record = json.loads((folder/'result.json').read_text())
    if record['status'] != 'completed' or record['reward_profile'] != 'hora_pose_delta_drop16':
        raise ValueError('Use completed frozen B evaluations with the declared original reward')
    if record['episode_resets'] or record['controller_switches'] or record['training_actions']:
        raise ValueError('One frozen uninterrupted episode required')
    states, valid, sources = [], [], []
    for path in sorted(folder.glob('physics_*.npz')):
        sources.append(identity(path))
        with np.load(path) as data:
            states.append(data['object_state']); valid.append(data['valid'])
    state, valid = np.concatenate(states), np.concatenate(valid).astype(bool)
    with np.load(folder/'initial_state.npz') as data: initial = data['object_state'][0]
    with np.load(folder/'reconstructed_control_reward.npz') as data:
        old = {k:data[k].copy() for k in data.files}
    signal = SpinProgressRewardSignal(config)
    steps = signal.steps_per_control
    if len(state) % steps:
        raise ValueError('This rescore requires complete recorded control intervals')
    current = Rotation.from_quat(state[:, 3:7])
    previous = Rotation.concatenate([Rotation.from_quat(initial[3:7]), current[:-1]])
    rv = (current*previous.inv()).as_rotvec()
    axis = current.apply([0, 0, 1])+previous.apply([0, 0, 1])
    axis /= np.linalg.norm(axis, axis=1)[:, None]
    delta = -(rv*axis).sum(-1)
    independent_speed = delta.reshape(-1, steps).sum(-1)/signal.control_dt
    independent_rotation = np.minimum(independent_speed, signal.configuration['positive_speed_cap_rad_s'])

    starts = np.concatenate([initial[None, 3:7], state[:-1, 3:7]])[::steps]
    samples = state[:, 3:7].reshape(-1, steps, 4)
    signal.begin(torch.tensor(starts, dtype=torch.float64))
    for step in range(steps): signal.advance(torch.tensor(samples[:, step], dtype=torch.float64))
    speed = signal.angular_speed().numpy()
    rotation = signal.rotation_reward().numpy()
    speed_error = float(np.max(np.abs(speed-independent_speed)))
    if speed_error > 1e-9: raise AssertionError('Candidate differs from independent SciPy self-spin')
    np.testing.assert_allclose(rotation, independent_rotation, rtol=0, atol=1e-9)

    # Recover the original penalty scalars from already recorded weighted terms;
    # no additional policy inference, dynamics, state filtering or task gate.
    linear = np.zeros((len(speed), 3)); linear[:, 0] = old['linear_cost']/-.3
    result = signal.compute_reward(torch.tensor(linear), torch.tensor(old['pose_cost']/-.3),
        torch.tensor(old['torque_cost']/-.1), torch.tensor(old['work_cost']/-2.),
        torch.tensor(old['failure_code']))
    terminal = result['terminal_failure_cost'].numpy()
    np.testing.assert_array_equal(terminal, old['terminal_failure_cost'])
    nonrotation = old['linear_cost']+old['pose_cost']+old['torque_cost']+old['work_cost']
    np.testing.assert_allclose(result['nonrotation_reward'].numpy(), nonrotation, rtol=0, atol=1e-10)
    independent_total = nonrotation+independent_rotation+terminal
    np.testing.assert_allclose(result['reward'].numpy(), independent_total, rtol=0, atol=1e-9)

    complete_valid = old['valid'].astype(bool)
    if len(complete_valid) != len(speed): raise AssertionError('Control/pose count mismatch')
    original_fixed = np.clip((rv@np.asarray(record['fixed_reward_axis'])).reshape(-1, steps).sum(-1)/signal.control_dt, -.5, .5)
    np.testing.assert_allclose(original_fixed, old['rotation'], rtol=0, atol=1e-6)
    spin_deg = float(np.rad2deg(delta[valid].sum()))
    metric_error = abs(spin_deg-record['net_deg'])
    if metric_error > 2e-4: raise AssertionError('Independent self-spin differs from saved primary metric')
    # min(v, cap) <= v, so a negative net angle cannot earn positive summed
    # raw rotation reward over the SAME equally timed complete controls.
    for mask in (complete_valid, np.ones_like(complete_valid)):
        if rotation[mask].sum() > independent_speed[mask].sum()+1e-9:
            raise AssertionError('Clipping hides reverse motion')
        if independent_speed[mask].sum() < -1e-9 and rotation[mask].sum() >= 0:
            raise AssertionError('Net reverse earns positive aggregate rotation reward')

    sources += [identity(folder/name) for name in ('result.json', 'initial_state.npz', 'reconstructed_control_reward.npz')]
    return dict(training_actions_in_checkpoint=record['training_actions_in_checkpoint'],
        checkpoint_sha256=record['checkpoint_sha256'], original_episode_stop_reason=record['stop_reason'],
        actual_s=record['actual_s'], valid_s=record['valid_s'],
        original_primary_metrics=record['metrics']['30.0'],
        original_fixed_rotation_mean=float(old['rotation'][complete_valid].mean()),
        moving_symmetric_clip_rotation_mean=float(np.clip(independent_speed, -.5, .5)[complete_valid].mean()),
        candidate_rotation_mean=float(rotation[complete_valid].mean()),
        complete_valid_control_self_spin_deg=float(np.rad2deg(independent_speed[complete_valid].sum()*signal.control_dt)),
        independent_valid_prefix_self_spin_deg=spin_deg, metric_error_deg=metric_error,
        signal_error_rad_s=speed_error,
        raw_total_old_reward_including_terminal=float(old['reward'].sum()),
        raw_total_candidate_reward_including_terminal=float(result['reward'].sum()),
        terminal_failure_cost_sum=float(terminal.sum()),
        window_note='Reward means use whole valid controls; primary net angle also includes the valid portion of the final terminal control. These are explicitly different intervals.',
        sources=sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--nodes', nargs='+', default=['010000000', '020000000', '030000000', '040000000'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [validate_episode(args.experiment/f'actions_{node}'/'evaluation', args.config) for node in args.nodes]
    result = dict(status='passed', training_actions=0, new_physics_steps=0, new_policy_inferences=0,
        reward_configuration=SpinProgressRewardSignal(args.config).configuration, rows=rows,
        acceptance='Independent pose integration, original other penalties/drop16, and no positive undiscounted aggregate rotation credit for net reverse on matched intervals.',
        interpretation='Same archived episodes rescored. No improved controller, stability, training efficacy or native-engine integration has been demonstrated. Unbounded reverse costs require subsequent PPO validation; fixed world-axis Hora baseline remains distinct.',
        sources=[identity(p) for p in (Path(__file__), args.config,
            ROOT/'tendonspin/rl/spin_progress_reward.py', ROOT/'tendonspin/rl/rotation_reward.py',
            ROOT/'third_party/hora/hora/tasks/allegro_hand_hora.py')])
    (args.out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(args.out/'summary.npz',
        actions=np.array([r['training_actions_in_checkpoint'] for r in rows]),
        old_rotation_mean=np.array([r['original_fixed_rotation_mean'] for r in rows]),
        moving_clip_mean=np.array([r['moving_symmetric_clip_rotation_mean'] for r in rows]),
        candidate_rotation_mean=np.array([r['candidate_rotation_mean'] for r in rows]))
    print(json.dumps(dict(status=result['status'], training_actions=0, new_physics_steps=0,
        episodes_rescored=len(rows), max_metric_error_deg=max(r['metric_error_deg'] for r in rows),
        max_signal_error_rad_s=max(r['signal_error_rad_s'] for r in rows),
        reward_means=[dict(actions=r['training_actions_in_checkpoint'], old=r['original_fixed_rotation_mean'],
            candidate=r['candidate_rotation_mean']) for r in rows]), indent=2))


if __name__ == '__main__': main()
