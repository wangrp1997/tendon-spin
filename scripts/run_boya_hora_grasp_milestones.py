# Reuse: TendonSpin a96c9a6 run_boya_hora_fresh_milestones.py,
# run_boya_hora_ab_continuation.py and their checkpoint/evaluation helpers (MIT).
# Learner: github.com/HaozhiQi/hora v0.0.1 (MIT), arXiv:2210.04887.
# Port change: fresh B50M schedule with an explicit migrated scene/cache.
"""Train one fresh B teacher and evaluate its fixed nominal state every10M."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts.run_boya_hora_fresh_milestones import (
    ROOT, evaluation_row, initial_identity, report, require_stage)
from scripts.run_boya_hora_milestones import milestones
from tendonspin.evaluation.comparison import digest, read_json, verify_checkpoint, verify_sources
from tendonspin.physics.learning_scene import load_scene_configuration
from tendonspin.rl.resource_guard import atomic_json


def command_for(run, target, protocol, cache, scene, previous=None):
    command = [sys.executable, str(ROOT/'scripts/run_boya_hora_background.py'),
        '--out', str(run), '--cache', str(cache), '--scene-config', str(scene),
        '--total-actions', str(target['requested_actions']),
        '--termination-profile', 'boya_workspace', '--engine-profile', 'original_tgs16_4',
        '--reward-profile', 'hora_pose_delta_drop16', '--eval-seconds', '30', '--reward-diagnostics',
        '--controller', 'hora_boya_uniform48p8_cache24_b_fresh50m_v1', '--protocol', protocol]
    if previous is not None:
        command += ['--resume', str(ROOT/previous['checkpoint'])]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scene-config', type=Path, required=True)
    parser.add_argument('--protocol', required=True)
    args = parser.parse_args()
    scene, out = args.scene_config.resolve(), args.out.resolve()
    spec = load_scene_configuration(scene)
    cache = ROOT/spec['training_cache']
    if digest(cache) != spec['training_cache_sha256']:
        raise ValueError('Declared migrated cache changed')
    targets = milestones(0, 50000000, 10000000)
    paths = (args.protocol, str(scene.relative_to(ROOT)),
        'scripts/run_boya_hora_grasp_milestones.py', 'scripts/run_boya_hora_fresh_milestones.py',
        'scripts/run_boya_hora_milestones.py', 'scripts/run_boya_hora_background.py',
        'scripts/train_boya_hora.py', 'scripts/evaluate_boya_hora.py',
        'scripts/analyze_boya_small_budget.py', 'tendonspin/evaluation/comparison.py',
        'tendonspin/physics/learning_scene.py')
    pins = [dict(path=p, sha256=digest(ROOT/p)) for p in paths]
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'plan.json', dict(arm='B', reward_profile='hora_pose_delta_drop16',
        initialization='fresh network/normalizers/Adam/adaptiveLR/RNG seed43; zero actions',
        targets=targets, protocol=args.protocol, sources=pins, scene_configuration=spec,
        cache_sha256=spec['training_cache_sha256'], num_envs=1024, seed=43,
        requested_total_actions=50000000, rounded_final_actions=targets[-1]['rounded_actions'],
        evaluation_window_s=30., new_frozen_episodes=5,
        resume_environment='Restore OWN full learner at each segment; reset physics from same24-state cache',
        resource_watchdog=False, training_wall_limit=False, auto_retry=False, auto_extension=False))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    state = dict(pid=os.getpid(), arm='B', phase='preparing', completed_milestones=[],
        start_actions=0, requested_total_actions=50000000,
        rounded_final_actions=targets[-1]['rounded_actions'])
    started = time.monotonic()
    child = None
    requested = False
    previous = None
    rows = []

    def save():
        state.update(wall_s=time.monotonic()-started, updated_unix_s=time.time())
        atomic_json(out/'status.json', state)

    def stop(signum, frame):
        nonlocal requested
        requested = True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMNI_KIT_ACCEPT_EULA='YES',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    save()
    try:
        for target in targets:
            if requested:
                break
            verify_sources(ROOT, pins)
            if digest(cache) != spec['training_cache_sha256']:
                raise ValueError('Migrated cache changed')
            if previous is not None:
                verify_sources(ROOT, previous['sources'])
                verify_checkpoint(ROOT, previous, ROOT/previous['checkpoint'], previous['actions_executed'])
            run = out/f'actions_{target["requested_actions"]:09d}'
            command = command_for(run, target, args.protocol, cache, scene, previous)
            state.update(phase='training_then_evaluation', current_target=target,
                current_run=str(run), command=command)
            save()
            with (out/f'actions_{target["requested_actions"]:09d}.log').open('x') as log:
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
                raise RuntimeError(f'Milestone exited{code}; stop without retry')
            require_stage(read_json(run/'stage.json'), target['rounded_actions'])
            row, current = evaluation_row('B', run, target, previous)
            if current.get('scene_configuration') != spec:
                raise ValueError('Training scene configuration differs from frozen plan')
            if previous is None:
                atomic_json(out/'initial_identity.json', initial_identity(run))
            rows.append(row)
            report(out, rows)
            # Selection reuses these milestone episodes; it is not held-out validation.
            eligible = [r for r in rows if r['complete_window'] and r['net_deg']>0
                        and r['final_10s_net_deg'] is not None and r['final_10s_net_deg']>0]
            if eligible:
                atomic_json(out/'best_observed.json', max(eligible, key=lambda r:r['net_deg']))
            previous = current
            state['completed_milestones'].append(dict(**target, run=str(run), evaluation=row['evaluation']))
            save()
        state['phase'] = 'stopped' if requested else 'completed'
    except BaseException as error:
        state.update(phase='stopped' if requested else 'error', error=dict(type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc()))
    finally:
        save()
    return 1 if state['phase']=='error' else 0


if __name__ == '__main__':
    sys.exit(main())
