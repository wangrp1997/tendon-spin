# Sources: github.com/wangrp1997/tendon-spin, commitd7d07f7,
# run_boya_hora_fresh_milestones.py / run_boya_hora_milestones.py (MIT).
# Learner: github.com/HaozhiQi/hora v0.0.1 (MIT), arXiv:2210.04887.
# Port change: schedule the authorized own-arm10M->50M continuation only;
# reuse unchanged learner, checkpoint contract, physics, rewards and30s evaluation.
"""Continue one completed fresh A/B learner through four fixed cumulative nodes."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts.run_boya_hora_fresh_milestones import (
    ARM_PROFILES, CACHE, CACHE_SHA256, REFERENCE, ROOT,
    command_for, evaluation_row, report, require_stage)
from scripts.run_boya_hora_milestones import BATCH, milestones
from tendonspin.evaluation.comparison import (
    digest, read_json, validate_result, verify_checkpoint, verify_sources)
from tendonspin.rl.resource_guard import atomic_json

TOTAL_ACTIONS = 50000000
INTERVAL_ACTIONS = 10000000
START_ACTIONS = 10002432


def continuation_targets():
    return milestones(START_ACTIONS, TOTAL_ACTIONS, INTERVAL_ACTIONS)


def validate_origin(arm, origin):
    expected = dict(status='completed', stop_reason='update budget',
        actions_executed=START_ACTIONS, completed_updates=START_ACTIONS // BATCH,
        seed=43, num_envs=1024, engine_profile='original_tgs16_4',
        reward_profile=ARM_PROFILES[arm], controller='hora_boya_pose_ab_fresh10m_' + arm.lower())
    for key, value in expected.items():
        if origin.get(key) != value:
            raise ValueError('Origin differs from authorized own-arm10M learner: ' + key)
    if origin.get('termination', {}).get('profile') != 'boya_workspace':
        raise ValueError('Origin task differs from the authorized workspace task')
    if origin.get('cache_sha256') != CACHE_SHA256 or 'reward_branch_migration' in origin:
        raise ValueError('Origin cache or fresh-arm ancestry differs')


def origin_row(arm, source_run, origin):
    """Reuse the recorded10M evaluation; never execute an extra episode."""
    result = read_json(source_run / 'evaluation/result.json')
    validate_result(result, dict(requested_s=30., seed=43,
        initial_state='original grasp44, not cache', termination=origin['termination'],
        expected_training_actions=START_ACTIONS, runtime_sources=origin['sources']),
        origin['checkpoint_sha256'], 'frozen_' + origin['controller'])
    if max(result['initial_state_max_errors'].values()) != 0:
        raise ValueError('Origin evaluation initial state differs from the frozen reference')
    rows = read_json(source_run.parent / 'evaluations.json')['rows']
    row = next(r for r in rows if r['training_actions'] == START_ACTIONS)
    if row['arm'] != arm or row['checkpoint_sha256'] != origin['checkpoint_sha256']:
        raise ValueError('Recorded origin evaluation belongs to another arm or checkpoint')
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=tuple(ARM_PROFILES), required=True)
    parser.add_argument('--resume-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    source_run, out = args.resume_run.resolve(), args.out.resolve()
    origin = read_json(source_run / 'training/result.json')
    validate_origin(args.arm, origin)
    require_stage(read_json(source_run / 'stage.json'), START_ACTIONS)
    verify_sources(ROOT, origin['sources'])
    identity = verify_checkpoint(ROOT, origin, ROOT / origin['checkpoint'], START_ACTIONS)
    if digest(CACHE) != CACHE_SHA256:
        raise ValueError('Declared28-state cache changed')
    row = origin_row(args.arm, source_run, origin)
    paths = (args.protocol, 'scripts/run_boya_hora_ab_continuation.py',
        'scripts/run_boya_hora_fresh_milestones.py', 'scripts/run_boya_hora_milestones.py',
        'scripts/run_boya_hora_background.py', 'scripts/analyze_boya_small_budget.py',
        'tendonspin/evaluation/comparison.py')
    pins = origin['sources'] + [dict(path=p, sha256=digest(ROOT / p)) for p in paths]
    targets = continuation_targets()
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(arm=args.arm, reward_profile=ARM_PROFILES[args.arm], resume_run=str(source_run),
        origin_checkpoint=identity, start_actions=START_ACTIONS, start_updates=START_ACTIONS // BATCH,
        requested_total_actions=TOTAL_ACTIONS, rounded_final_actions=targets[-1]['rounded_actions'],
        new_actions=targets[-1]['rounded_actions'] - START_ACTIONS, targets=targets,
        protocol=args.protocol, sources=pins, cache_sha256=CACHE_SHA256,
        initial_reference=str(REFERENCE), initial_reference_sha256=digest(REFERENCE),
        evaluation_window_s=30., new_frozen_episodes=4, reuse_existing_10m_evaluation=True,
        num_envs=1024, seed=43, lineage_id=origin['lineage_id'],
        initialization='Restore OWN full learner/network/normalizers/Adam/LR/scheduler/RNG/counts; no migration',
        resume_environment='Reset physics from same28-state cache at every segment; discard unfinished episodes/rollout',
        resource_watchdog=False, training_wall_limit=False, auto_retry=False, auto_extension=False)
    atomic_json(out / 'plan.json', plan)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    state = dict(pid=os.getpid(), arm=args.arm, phase='preparing', completed_milestones=[],
        start_actions=START_ACTIONS, requested_total_actions=TOTAL_ACTIONS,
        rounded_final_actions=targets[-1]['rounded_actions'], reused_origin_evaluation=row['evaluation'])
    started = time.monotonic()
    child = None
    requested = False

    def save():
        state.update(wall_s=time.monotonic() - started, updated_unix_s=time.time())
        atomic_json(out / 'status.json', state)

    def stop(signum, frame):
        nonlocal requested
        requested = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    save()
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMNI_KIT_ACCEPT_EULA='YES',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    previous = origin
    rows = [row]
    try:
        report(out, rows)
        for target in targets:
            if requested:
                break
            verify_sources(ROOT, pins)
            if digest(REFERENCE) != plan['initial_reference_sha256'] or digest(CACHE) != CACHE_SHA256:
                raise ValueError('Frozen initial reference or training cache changed')
            verify_sources(ROOT, previous['sources'])
            verify_checkpoint(ROOT, previous, ROOT / previous['checkpoint'], previous['actions_executed'])
            run = out / f'actions_{target["requested_actions"]:09d}'
            command = command_for(args.arm, run, target, args.protocol, previous)
            state.update(phase='training_then_evaluation', current_target=target,
                current_run=str(run), resume_checkpoint=str(ROOT / previous['checkpoint']), command=command)
            save()
            with (out / f'actions_{target["requested_actions"]:09d}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save()
                if requested:
                    child.send_signal(signal.SIGTERM)
                code = child.wait()
            if requested:
                break
            if code:
                raise RuntimeError(f'Milestone supervisor exited{code}; stop without retry')
            require_stage(read_json(run / 'stage.json'), target['rounded_actions'])
            row, current = evaluation_row(args.arm, run, target, previous)
            rows.append(row)
            report(out, rows)
            previous = current
            state['completed_milestones'].append(dict(**target, run=str(run), evaluation=row['evaluation']))
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
