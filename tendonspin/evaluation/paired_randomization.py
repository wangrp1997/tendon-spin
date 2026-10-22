"""Frozen, matched ID cases; no Isaac import or training behavior changes.

References: Hora arXiv:2210.04887v1, Sections 4/5.1; official eval_s1.sh.
Local protocol: docs/experiments/2026-10-16-randomized-evaluation/.
492 cases and clustered intervals are local evaluation design choices.
"""
from pathlib import Path
import hashlib
import json

import numpy as np


TEST_SEEDS = (2026101601, 2026101602, 2026101603)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def generate_cases(cfg, cache_groups, controls=600, seeds=TEST_SEEDS):
    """Each case has independent streams, invariant to batching or model failure."""
    if controls < 1 or len(set(seeds)) != len(seeds):
        raise ValueError('Positive horizon and distinct test seeds required')
    rows = []
    forces, noise = [], []
    for cache_index, group in enumerate(cache_groups):
        for seed in seeds:
            children = np.random.SeedSequence([int(seed), cache_index]).spawn(3)
            params, disturbance, observation = [np.random.default_rng(s) for s in children]
            mass = np.float32(params.uniform(*cfg['mass_kg']))
            rows.append(dict(cache_index=cache_index, scale_id=int(group), test_seed=int(seed),
                scale=float(np.float32(cfg['scale_centers'][int(group)] + params.uniform(
                    -cfg['scale_jitter'], cfg['scale_jitter']))),
                friction=float(np.float32(params.uniform(*cfg['friction']))),
                com_m=params.uniform(*cfg['com_m'], 3).astype(np.float32).tolist(),
                mass_kg=float(mass),
                p_ratio=params.uniform(*cfg['p_gain_ratio'], 16).astype(np.float32).tolist(),
                d_ratio=params.uniform(*cfg['d_gain_ratio'], 16).astype(np.float32).tolist()))
            force = np.zeros(3, dtype=np.float32)
            schedule = np.empty((controls, 3), dtype=np.float32)
            decay = cfg['force_decay'] ** (cfg['force_decay_reference_dt_s'] / cfg['force_decay_interval_s'])
            for t in range(controls):
                force *= decay
                if disturbance.random() < cfg['force_probability_per_control']:
                    force = (disturbance.standard_normal(3) * float(mass) * cfg['force_scale']).astype(np.float32)
                schedule[t] = force
            forces.append(schedule)
            noise.append(observation.uniform(-cfg['joint_noise_rad'], cfg['joint_noise_rad'],
                (controls, 16)).astype(np.float32))
    arrays = dict(force_N=np.stack(forces, axis=1), noise_rad=np.stack(noise, axis=1))
    for index, row in enumerate(rows):
        row['case_id'] = f"grasp{row['cache_index']:03d}_seed{row['test_seed']}"
        row['case_index'] = index
    return rows, arrays


def describe(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n=0, mean=None, median=None, p10=None, std=None)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
        p10=float(np.quantile(values, .1)), std=float(values.std()))


def summarize(rows):
    metrics = ('net_deg', 'valid_s', 'backward_deg', 'last10s_net_deg', 'max_drift_mm', 'max_tilt_deg', 'reward_sum')
    result = dict(cases=len(rows), complete_window=sum(r['complete_window'] for r in rows),
        complete_window_fraction=float(np.mean([r['complete_window'] for r in rows])),
        infrastructure_incomplete=sum(r['stop_reason'] == 'wall budget' for r in rows),
        stop_reasons={})
    for row in rows:
        reason = row['stop_reason']
        result['stop_reasons'][reason] = result['stop_reasons'].get(reason, 0) + 1
    result['cache_weighted'] = {key: describe([r[key] for r in rows if r[key] is not None]) for key in metrics}
    result['by_scale'] = {}
    for group in sorted({r['scale_id'] for r in rows}):
        subset = [r for r in rows if r['scale_id'] == group]
        result['by_scale'][str(group)] = dict(cases=len(subset),
            complete_window_fraction=float(np.mean([r['complete_window'] for r in subset])),
            **{key: describe([r[key] for r in subset if r[key] is not None]) for key in metrics})
    result['scale_macro_net_deg'] = float(np.mean([g['net_deg']['mean'] for g in result['by_scale'].values()]))
    result['scale_macro_complete_fraction'] = float(np.mean([g['complete_window_fraction'] for g in result['by_scale'].values()]))
    return result


def paired_comparison(baseline, candidate, draws=2000, seed=20261016):
    """Resample grasp clusters within scale; retain each grasp's test conditions."""
    left = {r['case_id']: r for r in baseline}
    right = {r['case_id']: r for r in candidate}
    if set(left) != set(right):
        raise ValueError('Paired cases differ')
    if any(r['stop_reason'] == 'wall budget' for r in baseline + candidate):
        raise ValueError('Incomplete infrastructure cases must not enter a complete paired claim')
    keys = sorted(left)
    for key in keys:
        if (left[key]['cache_index'], left[key]['scale_id']) != (right[key]['cache_index'], right[key]['scale_id']):
            raise ValueError('Pair identity differs')
    groups = sorted({left[k]['scale_id'] for k in keys})
    clusters = {g: sorted({left[k]['cache_index'] for k in keys if left[k]['scale_id'] == g}) for g in groups}
    deltas = {grasp: np.mean([right[k]['net_deg'] - left[k]['net_deg'] for k in keys
        if left[k]['cache_index'] == grasp]) for ids in clusters.values() for grasp in ids}
    rng = np.random.default_rng(seed)
    weighted, macro = [], []
    for _ in range(draws):
        sampled = [np.array([deltas[int(i)] for i in rng.choice(ids, len(ids), replace=True)])
            for ids in clusters.values()]
        weighted.append(float(np.concatenate(sampled).mean()))
        macro.append(float(np.mean([s.mean() for s in sampled])))
    return dict(paired_cases=len(keys), baseline='70M',
        mean_net_deg_difference=float(np.mean([right[k]['net_deg'] - left[k]['net_deg'] for k in keys])),
        cache_weighted_95ci=np.quantile(weighted, [.025, .975]).tolist(),
        scale_macro_net_deg_difference=float(np.mean([
            np.mean([deltas[i] for i in ids]) for ids in clusters.values()])),
        scale_macro_95ci=np.quantile(macro, [.025, .975]).tolist(),
        bootstrap_draws=draws, bootstrap_unit='grasp cluster, stratified by scale; common model pairs',
        interpretation='Test-case sampling uncertainty; not independent training-seed variance')
