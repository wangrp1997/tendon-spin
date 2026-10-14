# Derived from run_boya_hora_grasp_milestones.py at8cca30e (MIT).
# Separate C size/physical-DR route using the existing frozen evaluation workflow.
# Reuse: TendonSpin a96c9a6 run_boya_hora_fresh_milestones.py,
# run_boya_hora_ab_continuation.py and their checkpoint/evaluation helpers (MIT).
# Learner: github.com/HaozhiQi/hora v0.0.1 (MIT), arXiv:2210.04887.
# Port change: fresh C schedule with an explicit declared scene/cache and budget.
"""Train one fresh C teacher through a declared cumulative action budget."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts.run_boya_hora_fresh_milestones import (
    ROOT, report, require_stage)
from scripts.spin_progress_milestone_checks import evaluation_row, initial_identity
from tendonspin.rl.spin_progress_reward import PROFILE
from scripts.run_boya_hora_milestones import milestones
from tendonspin.evaluation.comparison import digest, read_json, verify_checkpoint, verify_sources
from tendonspin.physics.learning_scene import load_scene_configuration
from tendonspin.rl.resource_guard import atomic_json


def command_for(run, target, protocol, cache, scene, randomization, previous=None, num_envs=1024,
        controller='boya_ninescale_dr_cache164_c_fresh_v1'):
    command = [sys.executable, str(ROOT/'scripts/run_boya_spin_progress_stage.py'),
        '--out', str(run), '--cache', str(cache), '--scene-config', str(scene),
        '--randomization-config', str(randomization),
        '--total-actions', str(target['requested_actions']), '--num-envs', str(num_envs),
        '--termination-profile', 'boya_workspace', '--engine-profile', 'original_tgs16_4',
        '--reward-profile', PROFILE, '--eval-seconds', '30', '--reward-diagnostics',
        '--controller', controller, '--protocol', protocol]
    if previous is not None:
        command += ['--resume', str(ROOT/previous['checkpoint'])]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs',type=int,required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scene-config', type=Path, required=True)
    parser.add_argument('--randomization-config',type=Path,required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--total-actions', type=int, help='Defaults to the declared C configuration budget')
    parser.add_argument('--evaluation-every-actions', type=int, help='Defaults to the declared C evaluation cadence')
    args = parser.parse_args()
    scene, out = args.scene_config.resolve(), args.out.resolve()
    randomization=args.randomization_config.resolve()
    spec = load_scene_configuration(scene)
    cache = ROOT/spec['training_cache']
    if digest(cache) != spec['training_cache_sha256']:
        raise ValueError('Declared migrated cache changed')
    if args.num_envs < 1:raise ValueError('Positive num_envs required')
    paths = (args.protocol, str(scene.relative_to(ROOT)), str(randomization.relative_to(ROOT)),
        'scripts/train_boya_spin_progress.py','scripts/evaluate_boya_spin_progress.py',
        'scripts/run_boya_spin_progress_stage.py','scripts/run_boya_spin_progress_milestones.py',
        'tendonspin/rl/isaac_hora_randomized.py','tendonspin/rl/domain_randomization.py',
        'tendonspin/rl/isaac_spin_progress.py','tendonspin/rl/spin_progress_reward.py',
        'scripts/spin_progress_milestone_checks.py','assets/config/boya_spin_progress_reward.json',
        'tendonspin/physics/isaac_parallel_randomized.py',
        'scripts/run_boya_hora_grasp_milestones.py', 'scripts/run_boya_hora_fresh_milestones.py',
        'scripts/run_boya_hora_milestones.py', 'scripts/run_boya_hora_background.py',
        'scripts/train_boya_hora.py', 'scripts/evaluate_boya_hora.py',
        'scripts/analyze_boya_small_budget.py', 'tendonspin/evaluation/comparison.py',
        'tendonspin/physics/learning_scene.py')
    from tendonspin.rl.domain_randomization import load_configuration
    declared=load_configuration(randomization)['training']
    if (declared['num_envs'],declared.get('arm'),declared['reward_profile']) != (args.num_envs,'C',PROFILE):
        raise ValueError('C config/environment count differs from milestone plan')
    total_actions = args.total_actions if args.total_actions is not None else declared['requested_actions']
    evaluation_every = args.evaluation_every_actions if args.evaluation_every_actions is not None else declared['evaluation_every_actions']
    if (total_actions, evaluation_every) != (declared['requested_actions'], declared['evaluation_every_actions']):
        raise ValueError('Declared C budget/evaluation cadence differs from command')
    targets = milestones(0, total_actions, evaluation_every, batch=args.num_envs*8)
    controller = f'boya_ninescale_dr_cache164_c_fresh_{total_actions}_actions_v1'
    pins = [dict(path=p, sha256=digest(ROOT/p)) for p in paths]
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'plan.json', dict(arm='C', reward_profile=PROFILE, controller=controller,
        initialization='fresh network/normalizers/Adam/adaptiveLR/RNG seed43; zero actions',
        targets=targets, protocol=args.protocol, sources=pins, scene_configuration=spec,
        cache_sha256=spec['training_cache_sha256'], num_envs=args.num_envs, seed=43,
        requested_total_actions=total_actions, evaluation_every_actions=evaluation_every,
        rounded_final_actions=targets[-1]['rounded_actions'],
        total_updates=targets[-1]['rounded_actions']//(args.num_envs*8),
        evaluation_window_s=30., new_frozen_episodes=len(targets),
        resume_environment='Restore OWN full learner at each segment; reset physics from matching size-group cache164; same static properties from fixedseed43',
        resource_watchdog=False, training_wall_limit=False, auto_retry=False, auto_extension=False))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    state = dict(pid=os.getpid(), arm='C', reward_profile=PROFILE, num_envs=args.num_envs,
        controller=controller, phase='preparing', completed_milestones=[],
        start_actions=0, requested_total_actions=total_actions,
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
            command = command_for(run, target, args.protocol, cache, scene, randomization, previous,
                num_envs=args.num_envs, controller=controller)
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
            row, current = evaluation_row(run, target, previous, args.num_envs)
            if current.get('scene_configuration') != spec:
                raise ValueError('Training scene configuration differs from frozen plan')
            if previous is None:
                atomic_json(out/'initial_identity.json', initial_identity(run, args.num_envs))
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
