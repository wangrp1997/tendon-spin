"""Score a single archived, unwrapped physical rotation trace.

References: Hora (arXiv:2210.04887v1, Section 4), AnyRotate
(arXiv:2405.07391v3, Section 4), Touch Dexterity (arXiv:2303.10880v4,
Section V), and TacBPM (arXiv:2609.18174v1, Section IV-D).
New TendonSpin scoring code; this is not a policy or paper implementation.
Definitions and adaptation limits: docs/research/2026-10-08-rotation-metrics.
"""
from __future__ import annotations

import numpy as np


def score_prefix(angles_deg, physics_dt, valid_steps, *, window_s,
                 targets_deg=(90., 360., 720.), tail_s=30.):
    """Angles are signed, unwrapped, post-step samples relative to initial zero.

    The first invalid frame is excluded by valid_steps. A failed short prefix
    is not relabeled as having executed until window_s. Target times are first
    sampled crossings, with resolution physics_dt; an unreached goal is None.
    No stop/stall threshold or new success gate is introduced.
    """
    data = np.asarray(angles_deg, dtype=float)
    if data.ndim != 1 or valid_steps < 0 or valid_steps > len(data):
        raise ValueError('A one-dimensional trace and valid prefix length are required')
    if not np.isfinite(physics_dt) or physics_dt <= 0 or not np.isfinite(window_s) or window_s <= 0:
        raise ValueError('Positive finite physics_dt and window_s are required')
    if not np.isfinite(tail_s) or tail_s <= 0:
        raise ValueError('Positive finite tail_s is required')
    targets = np.asarray(targets_deg, dtype=float)
    if targets.ndim != 1 or not np.all(np.isfinite(targets) & (targets > 0)):
        raise ValueError('Goal angles must be finite and positive')
    count = min(int(valid_steps), int(np.floor(window_s / physics_dt + 1e-8)))
    angle = data[:count]
    if not np.all(np.isfinite(angle)):
        raise ValueError('Nonfinite values inside the physically valid prefix')
    elapsed = count * physics_dt
    completed = elapsed >= window_s - physics_dt * 1e-6
    net = float(angle[-1]) if count else 0.
    increments = np.diff(np.r_[0., angle])
    def angle_at(seconds):
        index = min(count, max(0, int(np.floor(seconds / physics_dt + 1e-8))))
        return float(angle[index - 1]) if index else 0.
    tail = net - angle_at(elapsed - tail_s) if elapsed >= tail_s - 1e-9 else None
    goals = []
    for target in targets:
        indices = np.flatnonzero(angle >= target)
        goals.append(dict(target_deg=float(target), reached=bool(len(indices)),
                          first_sampled_time_s=float((indices[0] + 1) * physics_dt) if len(indices) else None,
                          observed_until_s=float(elapsed), time_resolution_s=float(physics_dt)))
    return dict(window_s=float(window_s), valid_seconds=float(elapsed),
                complete_window_observed=bool(completed), net_deg=net,
                peak_deg=float(max(0., np.max(angle))) if count else 0.,
                forward_deg=float(np.maximum(increments, 0.).sum()),
                backward_deg=float(np.maximum(-increments, 0.).sum()),
                net_deg_per_budget_second=net / window_s,
                net_deg_per_valid_second=net / elapsed if elapsed else None,
                valid_prefix_last_tail_s=float(tail_s),
                valid_prefix_last_tail_net_deg=float(tail) if tail is not None else None,
                full_window_final_tail_net_deg=float(tail) if completed and tail is not None else None,
                goals=goals, stalled=None,
                stall_definition='Pending calibrated observation resolution and agreed window; no guessed threshold')
