# References: TendonSpin run_boya_hora_background.py, checkpoint.py and
# evaluation/comparison.py; original Hora v0.0.1 learner restoration (MIT).
# This orchestrator changes scheduling/reporting only; learner and physics are reused.
"""Resume through cumulative action milestones, evaluating each before continuing."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from tendonspin.evaluation.comparison import (
    compare_initial_states, diagnostics, digest, read_json, validate_result,
    verify_checkpoint, verify_sources)
from tendonspin.rl.resource_guard import atomic_json

ROOT = Path(__file__).resolve().parents[1]
BATCH = 1024 * 8


def milestones(start, total, interval, batch=BATCH):
    if min(interval, batch) <= 0 or total <= start:
        raise ValueError('Require a positive interval/batch and a larger cumulative target')
    targets = list(range((start // interval + 1) * interval, total, interval)) + [total]
    return [dict(requested_actions=t, rounded_actions=((t + batch - 1) // batch) * batch)
            for t in targets if ((t + batch - 1) // batch) * batch > start]


def require_completed_stage(stage, actions):
    if stage.get('phase') != 'completed':
        raise RuntimeError('Milestone interrupted or failed; do not continue')
    for key in ('training', 'evaluation'):
        item = stage.get(key, {})
        if not isinstance(item, dict) or item.get('exit_code') != 0 or item.get('status') != 'completed':
            raise RuntimeError('Incomplete milestone ' + key)
    train = stage['training']
    if train.get('stop_reason') != 'update budget' or train.get('actions_executed') != actions:
        raise RuntimeError('Milestone did not complete its declared cumulative training budget')


def evaluation_row(run, actions, reference, reference_eval):
    training = read_json(run / 'training/result.json')
    if training['status'] != 'completed' or training['stop_reason'] != 'update budget':
        raise ValueError('Cannot evaluate a partial training milestone')
    identity = verify_checkpoint(ROOT, training, ROOT / training['checkpoint'], actions)
    result = read_json(run / 'evaluation/result.json')
    plan = dict(requested_s=120., seed=43, initial_state='original grasp44, not cache',
                termination=reference['termination'], expected_training_actions=actions,
                runtime_sources=training['sources'])
    validate_result(result, plan, identity['checkpoint_sha256'], 'frozen_' + training['controller'])
    agreement = compare_initial_states(run / 'evaluation', reference_eval)
    metric = result['metrics']['120.0']
    row = dict(training_actions=actions, net_deg=metric['net_deg'], valid_s=metric['valid_seconds'],
               peak_deg=metric['peak_deg'], backward_deg=metric['backward_deg'],
               complete_window=metric['complete_window_observed'], stop_reason=result['stop_reason'],
               **diagnostics(run / 'evaluation', windows=(120.,))['120.0'],
               checkpoint=str(ROOT / training['checkpoint']), checkpoint_sha256=identity['checkpoint_sha256'],
               evaluation=str(run / 'evaluation/result.json'),
               initial_state_max_error=max(agreement['max_abs_difference'].values()))
    return row, training


def report(out, rows):
    atomic_json(out / 'evaluations.json', dict(window_s=120., rows=rows,
        interpretation='One independent frozen original-state episode per checkpoint; no summed angles or resets.'))
    with (out / 'evaluations.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# 累计训练节点评估', '',
             '同一原始抓取、boya_workspace规则、120秒窗口；每个节点独立执行一次，不拼接角度。', '',
             '| 累计动作 | 净转角° | 有效秒 | 倒转° | 最大倾斜° | 终止原因 |',
             '|---:|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f'| {row["training_actions"]:,} | {row["net_deg"]:.3f} | {row["valid_s"]:.4f} | '
                     f'{row["backward_deg"]:.3f} | {row["tilt_deg"]:.3f} | {row["stop_reason"]} |')
    (out / 'evaluations.md').write_text('\n'.join(lines) + '\n')
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(out / 'evaluation_tb')) as writer:
        last = rows[-1]
        for name in ('net_deg', 'valid_s', 'peak_deg', 'backward_deg', 'drift_mm', 'tilt_deg', 'max_normal'):
            if last[name] is not None:
                writer.add_scalar('evaluation120s/' + name, last[name], last['training_actions'])
        writer.add_scalar('evaluation120s/complete_window', int(last['complete_window']), last['training_actions'])
        writer.add_text('evaluation120s/stop_reason', last['stop_reason'], last['training_actions'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-run', type=Path, required=True)
    parser.add_argument('--total-actions', type=int, default=50000000)
    parser.add_argument('--interval-actions', type=int, default=10000000)
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    out, source_run = args.out.resolve(), args.resume_run.resolve()
    origin = read_json(source_run / 'training/result.json')
    if origin['termination']['profile'] != 'boya_workspace':
        raise ValueError('This continuation requires the declared workspace task')
    require_completed_stage(read_json(source_run / 'stage.json'), origin['actions_executed'])
    targets = milestones(origin['actions_executed'], args.total_actions, args.interval_actions)
    out.mkdir(parents=True, exist_ok=False)
    pins = list(origin['sources']) + [dict(path=p, sha256=digest(ROOT / p)) for p in
        (args.protocol, 'scripts/run_boya_hora_milestones.py', 'scripts/run_boya_hora_background.py',
         'tendonspin/evaluation/comparison.py')]
    plan = dict(resume_run=str(source_run), start_actions=origin['actions_executed'], targets=targets,
                protocol=args.protocol, sources=pins, termination=origin['termination'],
                evaluation_window_s=120., extra_evaluations=len(targets), reuse_existing_10m_evaluation=True,
                resume_environment='Reset from the same cache at each resume; learner/RNG/optimizer retained',
                resource_watchdog=False, training_wall_limit=False, auto_retry=False)
    atomic_json(out / 'plan.json', plan)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    state = dict(pid=os.getpid(), phase='preparing', completed_milestones=[],
                 requested_total_actions=args.total_actions, start_actions=origin['actions_executed'])
    child = None; requested = False; started = time.monotonic()
    def save():
        state['wall_s'] = time.monotonic() - started
        state['updated_unix_s'] = time.time()
        atomic_json(out / 'status.json', state)
    def stop(signum, frame):
        nonlocal requested
        requested = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    save()
    try:
        verify_sources(ROOT, pins)
        row, previous = evaluation_row(source_run, origin['actions_executed'], origin, source_run / 'evaluation')
        rows = [row]; report(out, rows)
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(ROOT), OMNI_KIT_ACCEPT_EULA='YES', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        for target in targets:
            if requested:
                break
            verify_sources(ROOT, pins)
            checkpoint = ROOT / previous['checkpoint']
            verify_checkpoint(ROOT, previous, checkpoint, previous['actions_executed'])
            run = out / f'actions_{target["requested_actions"]:09d}'
            command = [sys.executable, str(ROOT / 'scripts/run_boya_hora_background.py'),
                '--out', str(run), '--resume', str(checkpoint), '--total-actions', str(target['requested_actions']),
                '--termination-profile', 'boya_workspace', '--controller', 'hora_boya_workspace_teacher_continue50m_v4',
                '--protocol', args.protocol]
            state.update(phase='training_then_evaluation', current_target=target, current_run=str(run),
                         resume_checkpoint=str(checkpoint), command=command)
            save()
            with (out / f'actions_{target["requested_actions"]:09d}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid; save()
                if requested:
                    child.send_signal(signal.SIGTERM)
                exit_code = child.wait()
            if requested:
                break
            if exit_code:
                raise RuntimeError(f'Milestone supervisor exited {exit_code}; no automatic retry')
            require_completed_stage(read_json(run / 'stage.json'), target['rounded_actions'])
            row, current = evaluation_row(run, target['rounded_actions'], origin, source_run / 'evaluation')
            if current['resumed_checkpoint_sha256'] != previous['checkpoint_sha256']:
                raise ValueError('Continuation did not load the preceding milestone checkpoint')
            if current['lineage_id'] != origin['lineage_id']:
                raise ValueError('Continuation lineage differs')
            rows.append(row); report(out, rows)
            state['completed_milestones'].append(dict(**target, run=str(run), evaluation=row['evaluation']))
            previous = current
            save()
        state['phase'] = 'stopped' if requested else 'completed'
    except BaseException as error:
        state.update(phase='stopped' if requested else 'error', error=dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        save()
    return 1 if state['phase'] == 'error' else 0


if __name__ == '__main__':
    sys.exit(main())
