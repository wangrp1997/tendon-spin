"""Rescore completed teacher-v2 archives; zero new simulator integration.

Source definitions: Hora2210.04887v1/AnyRotate2405.07391v3/TouchDexterity
2303.10880v4/TacBPM2609.18174v1. New TendonSpin audit; protocol and raw
execution hashes are preserved. Historical primary net-angle scores stay intact.
"""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np

from tendonspin.evaluation.rotation import score_prefix
from tendonspin.rl.config import ROOT, digest, write


def score_episode(directory, windows=(30., 120.)):
    directory = directory.resolve()
    path = directory / 'results.json'
    result = json.loads(path.read_text())
    if not result['original_state'] or result['episode_resets'] or result['controller_switches']:
        raise ValueError('Standalone original-state, uninterrupted episodes required')
    if max(windows) > result['evaluation_seconds_limit']:
        raise ValueError('Requested window exceeds the archived declared evaluation budget')
    execution = ROOT / result['execution']['path']
    if digest(execution) != result['execution']['sha256']:
        raise ValueError('Raw execution hash changed')
    protocol_path = ROOT / result['protocol']['path']
    if digest(protocol_path) != result['protocol']['sha256']:
        raise ValueError('Protocol hash changed')
    with np.load(execution, allow_pickle=False) as raw:
        angle = raw['angle_deg']
        dt = float(raw['physics_dt'])
        scores = [score_prefix(angle, dt, result['valid_steps'], window_s=t) for t in windows]
        first_invalid_s = (result['valid_steps'] + 1) * dt if result['physical_failure'] else None
        for score in scores:
            failed = first_invalid_s is not None and first_invalid_s <= score['window_s'] + dt * 1e-6
            score['physical_failure_within_window'] = bool(failed)
            score['window_stop_reason'] = result['reason'] if failed else 'requested window observed' if score['complete_window_observed'] else 'source archive ended'
    longest = scores[-1]
    if abs(longest['net_deg'] - result['net_deg']) > 1e-9:
        raise ValueError('Longest-window endpoint differs from the saved headline')
    return dict(kind='archived episode rescoring', new_physics_steps=0,
                results_source=dict(path=str(path.relative_to(ROOT)), sha256=digest(path)),
                execution=result['execution'], protocol=result['protocol'],
                controller=result['controller'], training_updates=result['training_updates'],
                stage='imported-weight initialization' if result['training_updates'] == 0 else 'new v2 training',
                original_declared_window_s=result['evaluation_seconds_limit'],
                original_stop_reason=result['reason'], source_episode_physical_failure=result['physical_failure'],
                source_first_invalid_time_s=first_invalid_s,
                angle_definition='Signed unwrapped spin about the moving cylinder axis, direction -1; not SO(3) endpoint distance',
                scores=scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [score_episode(args.run / 'evaluations' / f'update_{u:04d}') for u in (0, 32, 64, 128)]
    report = dict(kind='archived rescoring only', new_physics_steps=0, records=rows)
    write(args.out / 'archived_metrics.json', report)
    fields = ['training_updates', 'stage', 'window_s', 'valid_seconds', 'complete_window_observed',
              'net_deg', 'peak_deg', 'forward_deg', 'backward_deg', 'full_window_final_tail_net_deg',
              'time_to_90deg_s', 'time_to_360deg_s', 'window_stop_reason', 'physical_failure_within_window',
              'original_stop_reason', 'source_episode_physical_failure']
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n'); writer.writeheader()
    for row in rows:
        for score in row['scores']:
            value = {key: row[key] for key in ('training_updates', 'stage', 'original_stop_reason', 'source_episode_physical_failure')}
            value.update({key: score[key] for key in fields if key in score})
            for target in (90, 360):
                value[f'time_to_{target}deg_s'] = next(g['first_sampled_time_s'] for g in score['goals'] if g['target_deg'] == target)
            writer.writerow(value)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'archived_metrics.csv').write_text(buf.getvalue())
    print(json.dumps(dict(records=len(rows), new_physics_steps=0, out=str(args.out))))


if __name__ == '__main__':
    main()
