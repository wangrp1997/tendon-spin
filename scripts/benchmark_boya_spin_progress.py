"""Independent C capacity trials: two warmups, eight full PPO updates, no resource guard."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from tendonspin.evaluation.comparison import digest, read_json
from tendonspin.evaluation.comparison import verify_sources
from tendonspin.rl.resource_guard import atomic_json
from tendonspin.rl.spin_progress_reward import PROFILE
from scripts.run_boya_hora_milestones import milestones

ROOT = Path(__file__).resolve().parents[1]
COUNTS = tuple(1024 * 2**i for i in range(6))
OOM = re.compile(r'CUDA out of memory|cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY|'
    r'CUDA error[^\n]*out of memory|out of memory[^\n]*GPU|std::bad_alloc', re.I)
PHYSX_CAPACITY = re.compile(r'PhysX error:[^\n]*limit of 64K materials reached', re.I)


def select_trial(rows, total_mib, reserve_mib):
    eligible = [r for r in rows if r['status'] == 'completed'
        and r['peak_device_MiB'] <= total_mib-reserve_mib]
    if not eligible:
        raise RuntimeError('No completed trial retains checkpoint/evaluation memory headroom')
    return max(eligible, key=lambda r: r['actions_per_s'])


def projected_run_wall(row, actions, evaluation_every):
    targets = milestones(0, actions, evaluation_every, batch=row['num_envs']*8)
    return (targets[-1]['rounded_actions']/row['actions_per_s']
        + len(targets)*row['initialization_and_shutdown_s'])


def select_total_time_trial(rows, total_mib, reserve_mib, actions, evaluation_every):
    eligible = [r for r in rows if r['status'] == 'completed'
        and r['peak_device_MiB'] <= total_mib-reserve_mib]
    if not eligible:
        raise RuntimeError('No complete trial retains checkpoint/evaluation memory headroom')
    # Frozen one-environment evaluation has the same nominal window and process
    # overhead across counts; it is additional to these comparable estimates.
    return min(eligible, key=lambda r: projected_run_wall(r, actions, evaluation_every))


def gpu_memory(index):
    result = subprocess.run(['nvidia-smi', '-i', str(index),
        '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=5)
    return tuple(int(s.strip()) for s in result.stdout.strip().split(','))


def rss_mib(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1])/1024
    except FileNotFoundError:
        pass
    return 0.


def reuse_trials(directory, config, spec, gpu, total, scene):
    """Reuse finished measurements only when the complete runtime identity still matches."""
    previous = read_json(directory/'benchmark.json')
    if (previous['gpu'], previous['total_device_MiB'], previous['warmup_updates'],
            previous['measured_updates'], previous['reward_profile']) != (gpu, total, 2, 8, PROFILE):
        raise ValueError('Previous capacity measurements use a different device or protocol')
    rows = []
    for old in previous['trials']:
        if old['status'] != 'completed':
            continue
        command = old['command']
        result_path = Path(command[command.index('--out')+1])/'result.json'
        result = read_json(result_path)
        # The root handoff is a mutable operational status document. All frozen
        # protocols, executable sources, assets and trial configurations must match.
        verify_sources(ROOT, [s for s in result['sources'] if s['path'] != 'SERVER_TRAINING_HANDOFF.md'])
        if any(importlib.metadata.version(name) != version
                for name, version in result['runtime_versions'].items()):
            raise ValueError('Previous capacity measurements use a different installed runtime')
        declared = json.loads(json.dumps(result['domain_randomization_configuration']))
        expected = json.loads(json.dumps(config))
        declared['training']['num_envs'] = expected['training']['num_envs']
        if declared != expected or result['cache_sha256'] != spec['training_cache_sha256']:
            raise ValueError('Previous capacity measurements use a different DR/cache configuration')
        if result['scene_configuration']['configuration_sha256'] != digest(scene):
            raise ValueError('Previous capacity measurements use a different scene')
        row = dict(old, reused_from=str(result_path))
        row['initialization_and_shutdown_s'] = max(0.,
            row['trial_wall_s']-result['updates'][-1]['completed_at_training_wall_s'])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--counts', type=int, nargs='+', default=COUNTS)
    parser.add_argument('--wall-budget-s', type=float, default=1200,
        help='0 permits the user-authorized extended capacity sweep without a wall cutoff')
    parser.add_argument('--reuse-from', type=Path,
        help='Verify and reuse completed measurements from a previous sweep')
    parser.add_argument('--protocol', default='SERVER_TRAINING_HANDOFF.md')
    parser.add_argument('--minimum-relative-gain', type=float, default=0.,
        help='Stop expanding when a larger completed bin improves throughput by no more than this fraction; 0 disables')
    parser.add_argument('--selection-objective', choices=('throughput','total-run-wall'), default='throughput')
    parser.add_argument('--target-actions', type=int, default=50000000)
    parser.add_argument('--evaluation-every-actions', type=int, default=10000000)
    parser.add_argument('--scene-config', type=Path, default=ROOT/'assets/config/boya_hora_randomized_c_scene.json')
    parser.add_argument('--randomization-config', type=Path, default=ROOT/'assets/config/boya_hora_randomized_c.json')
    args = parser.parse_args()
    if args.target_actions < 1 or args.evaluation_every_actions < 1:
        raise ValueError('Positive target and evaluation cadence required')
    if args.wall_budget_s < 0 or not 0 <= args.minimum_relative_gain < 1 or any(c <= 0 or c*8 % 512 for c in args.counts):
        raise ValueError('Counts must fill original 512-action minibatches; wall budget cannot be negative')
    counts = tuple(sorted(set(args.counts)))
    out = args.out.resolve()
    config = read_json(args.randomization_config)
    spec = read_json(args.scene_config)
    cache = ROOT/spec['training_cache']
    if digest(cache) != spec['training_cache_sha256']:
        raise ValueError('Cache identity differs from the handoff')
    used, total = gpu_memory(args.gpu)
    if used > 1024:
        raise RuntimeError('Selected GPU is occupied; do not interfere with another task')
    reserve = max(4096, round(total*.05))
    reused = reuse_trials(args.reuse_from.resolve(), config, spec, args.gpu, total,
        args.scene_config.resolve()) if args.reuse_from else []
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(arm='C', reward_profile=PROFILE, gpu=args.gpu, total_device_MiB=total,
        reserve_device_MiB=reserve, counts=counts, warmup_updates=2, measured_updates=8,
        wall_budget_s=args.wall_budget_s, resource_watchdog=False, automatic_memory_stop=False,
        reuse_from=str(args.reuse_from) if args.reuse_from else None,
        minimum_relative_gain=args.minimum_relative_gain,
        selection_objective=args.selection_objective, target_actions=args.target_actions,
        evaluation_every_actions=args.evaluation_every_actions,
        common_frozen_evaluation_wall='Additional; identical nominal 30s-or-failure window across counts',
        trials=reused, phase='benchmarking')
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=str(ROOT),
        OMNI_KIT_ACCEPT_EULA='YES', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    child = None
    try:
        for count in counts:
            if any(r['num_envs'] == count for r in reused):
                continue
            remaining = max(0., args.wall_budget_s-(time.monotonic()-started)) if args.wall_budget_s else 0.
            if args.wall_budget_s and remaining <= 0:
                state['stop_reason'] = 'benchmark wall budget'
                break
            cfg_path = out/f'config_{count}.json'
            trial_cfg = json.loads(json.dumps(config))
            trial_cfg['training'].update(num_envs=count, arm='C', reward_profile=PROFILE)
            atomic_json(cfg_path, trial_cfg)
            trial_out = out/f'envs_{count}'
            command = [sys.executable, str(ROOT/'scripts/train_boya_spin_progress.py'),
                '--out', str(trial_out), '--cache', str(cache),
                '--scene-config', str(args.scene_config.resolve()),
                '--randomization-config', str(cfg_path), '--num-envs', str(count),
                '--updates', '10', '--wall-s', str(remaining), '--trace-mode', 'summary',
                '--save-every', '16', '--minibatch-size', '512', '--max-gpu-memory-mib', '0',
                '--protocol', args.protocol,
                '--controller', 'boya_c_capacity_trial_seed43']
            row = dict(num_envs=count, status='running', peak_device_MiB=used,
                sampled_peak_host_rss_MiB=0., command=command)
            state['trials'].append(row)
            atomic_json(out/'benchmark.json', state)
            trial_started = time.monotonic()
            log_path = out/f'envs_{count}.log'
            with log_path.open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                row['pid'] = child.pid
                atomic_json(out/'benchmark.json', state)
                timed_out = False
                physics_capacity_error = False
                last_snapshot = time.monotonic()
                while child.poll() is None:
                    current, _ = gpu_memory(args.gpu)
                    row['peak_device_MiB'] = max(row['peak_device_MiB'], current)
                    row['sampled_peak_host_rss_MiB'] = max(row['sampled_peak_host_rss_MiB'], rss_mib(child.pid))
                    if time.monotonic()-last_snapshot >= 5:
                        if not physics_capacity_error and PHYSX_CAPACITY.search(log_path.read_text(errors='replace')):
                            physics_capacity_error = True
                            child.terminate()
                        atomic_json(out/'benchmark.json', state)
                        last_snapshot = time.monotonic()
                    if args.wall_budget_s and time.monotonic()-started >= args.wall_budget_s and not timed_out:
                        # Wall budget only; memory sampling never stops a process.
                        timed_out = True
                        child.terminate()
                    time.sleep(.5)
                code = child.wait()
            row.update(exit_code=code, trial_wall_s=time.monotonic()-trial_started)
            result_path = trial_out/'result.json'
            result = read_json(result_path) if result_path.exists() else {}
            if physics_capacity_error or PHYSX_CAPACITY.search(log_path.read_text(errors='replace')):
                row['status'] = 'physics capacity failure'
                row['capacity_reason'] = 'PhysX 64K material hard limit; not VRAM exhaustion'
                state['stop_reason'] = 'PhysX material limit; do not test larger counts or alter engine'
                break
            if timed_out or result.get('stop_reason') == 'wall budget':
                row['status'] = 'wall budget'
                state['stop_reason'] = 'benchmark wall budget'
                break
            if code or result.get('status') != 'completed' or result.get('stop_reason') != 'update budget':
                if OOM.search(log_path.read_text(errors='replace')+json.dumps(result.get('error', {}))):
                    row['status'] = 'capacity failure'
                    state['stop_reason'] = 'OOM; do not test larger counts'
                    break
                row['status'] = 'execution error'
                raise RuntimeError(f'{count} environments failed outside capacity; see {log_path}')
            updates = result['updates']
            if len(updates) != 10 or not result.get('first_update_finite') or not result.get('first_update_parameters_changed'):
                raise RuntimeError('Capacity trial lacks ten complete, finite updates and actual learning')
            measured = updates[2:]
            elapsed = updates[-1]['completed_at_training_wall_s']-updates[1]['completed_at_training_wall_s']
            row.update(status='completed', measured_actions=count*8*8,
                measured_wall_s=elapsed, actions_per_s=count*8*8/elapsed,
                initialization_and_shutdown_s=max(0., row['trial_wall_s']-updates[-1]['completed_at_training_wall_s']),
                mean_update_wall_s=elapsed/8,
                sampling_plus_ppo_update_wall_s=[u['update_wall_s'] for u in measured],
                host_peak_rss_MiB=result['host_peak_rss_MiB'],
                torch_peak_allocated_MiB=result['torch_peak_allocated_MiB'],
                torch_peak_reserved_MiB=result['torch_peak_reserved_MiB'])
            atomic_json(out/'benchmark.json', state)
            print('C_CAPACITY '+json.dumps(row), flush=True)
            if row['peak_device_MiB'] >= total-reserve:
                state['stop_reason'] = 'checkpoint/evaluation headroom reached'
                break
            previous = [r for r in state['trials'] if r is not row and r['status'] == 'completed']
            if args.minimum_relative_gain and previous:
                previous_best = max(previous, key=lambda r:r['actions_per_s'])
                gain = row['actions_per_s']/previous_best['actions_per_s']-1
                row['relative_gain_over_previous_best'] = gain
                if count > max(r['num_envs'] for r in previous) and gain <= args.minimum_relative_gain:
                    state['stop_reason'] = 'measured throughput plateau; prioritize training efficiency'
                    break
        for row in state['trials']:
            if row['status'] == 'completed':
                row['projected_training_and_initialization_s'] = projected_run_wall(
                    row, args.target_actions, args.evaluation_every_actions)
        best = (select_total_time_trial(state['trials'], total, reserve,
            args.target_actions, args.evaluation_every_actions) if args.selection_objective == 'total-run-wall'
            else select_trial(state['trials'], total, reserve))
        state.update(phase='completed', selected=best,
            fresh_training_required=True, benchmark_actions_count_toward_50m=False)
        chosen = json.loads(json.dumps(config))
        chosen['training'].update(num_envs=best['num_envs'], arm='C', reward_profile=PROFILE)
        atomic_json(out/'selected_config.json', chosen)
    except BaseException as error:
        state.update(phase='error', error=dict(type=type(error).__name__, message=str(error)))
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state['wall_s'] = time.monotonic()-started
        atomic_json(out/'benchmark.json', state)
    return 0 if state['phase'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
