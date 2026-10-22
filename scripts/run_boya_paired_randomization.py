"""Prepare and supervise a separate GPU-0 frozen three-checkpoint comparison.

One 9-environment validation precedes three fresh 492-environment scenes.
No automatic retries, training intervention, or modification of pinned sources.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from tendonspin.evaluation.paired_randomization import (
    TEST_SEEDS, generate_cases, paired_comparison, sha256, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = 'GPU-4d3badb8-5fb0-b880-f08e-6407f0093e86'
MODEL_DIRS = {
    '70M': 'outputs/boya_c_fresh1p5b_server_v1/actions_070000000/training',
    '230M': 'outputs/boya_c_fresh1p5b_server_recovery_sync_v1/actions_230000000/training',
    '300M': 'outputs/boya_c_fresh1p5b_server_recovery_sync_v1/actions_300000000/training',
}
NEW_SOURCES = (
    'scripts/run_boya_paired_randomization.py', 'scripts/evaluate_boya_paired_randomization.py',
    'tendonspin/evaluation/paired_randomization.py', 'tests/test_paired_randomization.py',
)


def prepare(out):
    import torch
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=False)
    cache = 'outputs/boya_hora_scale_caches_v1/grasp_cache.npz'
    expected_cache = '6cb72539fc3c40d1e75a49983019c9b4be679a37bd5d8dcef406c06696aaf993'
    if sha256(ROOT/cache) != expected_cache:
        raise ValueError('Original 164-grasp cache missing or changed')
    models, pins = {}, {}
    reference = None
    fields = ('reward_configuration', 'domain_randomization_configuration', 'engine_configuration',
        'termination', 'scene_configuration', 'engine_profile', 'reward_profile')
    for label, relative in MODEL_DIRS.items():
        base = ROOT/relative
        record = json.loads((base/'result.json').read_text())
        if record['status'] != 'completed' or record['arm'] != 'C':
            raise ValueError('Invalid checkpoint record: '+label)
        if reference is None:
            reference = record
        elif any(record[key] != reference[key] for key in fields):
            raise ValueError('Checkpoints have different task/physical contracts: '+label)
        learner = torch.load(base/'teacher_final.pth', map_location='cpu', weights_only=False)
        policy = torch.load(base/'teacher_final_policy.pth', map_location='cpu', weights_only=False)
        full_hash = sha256(base/'teacher_final.pth')
        if policy['learner_sha256'] != full_hash or learner['progress'] != policy['progress']:
            raise ValueError('Policy/full learner pairing failed: '+label)
        for key in ('model', 'running_mean_std', 'value_mean_std'):
            if set(learner[key]) != set(policy[key]):
                raise ValueError('Policy tensor keys differ: '+key)
            for name, tensor in learner[key].items():
                if not torch.isfinite(tensor).all() or not torch.equal(tensor, policy[key][name]):
                    raise ValueError('Nonfinite or unpaired tensor: '+label+'/'+key+'/'+name)
        if learner['progress']['actions_executed'] != record['actions_executed']:
            raise ValueError('Checkpoint action metadata differs')
        for ck, rk in [('reward_configuration', 'reward_configuration'),
                       ('domain_randomization', 'domain_randomization_configuration'),
                       ('scene_configuration', 'scene_configuration'), ('termination', 'termination'),
                       ('engine_configuration', 'engine_configuration')]:
            if learner['contract'][ck] != record[rk]:
                raise ValueError('Recorded checkpoint contract differs: '+ck)
        models[label] = dict(checkpoint=relative+'/teacher_final.pth', checkpoint_sha256=full_hash,
            policy=relative+'/teacher_final_policy.pth', policy_sha256=sha256(base/'teacher_final_policy.pth'),
            source_record=relative+'/result.json', source_record_sha256=sha256(base/'result.json'),
            actions_executed=record['actions_executed'], cpu_pair_verified=True)
        for source in record['sources']:
            path = source['path']
            if path in pins and pins[path] != source['sha256']:
                raise ValueError('Checkpoint source identities differ: '+path)
            pins[path] = source['sha256']
    recovery = json.loads((ROOT/'outputs/boya_c_fresh1p5b_server_recovery_sync_v1/plan.json').read_text())
    for source in recovery['sources'] + recovery['recovery_sources']:
        path = source['path']
        if path in pins and pins[path] != source['sha256']:
            raise ValueError('Recovery source identity differs: '+path)
        pins[path] = source['sha256']
    for name in NEW_SOURCES:
        pins[name] = sha256(ROOT/name)
    for name, digest in pins.items():
        if sha256(ROOT/name) != digest:
            raise ValueError('Frozen training source changed: '+name)
    with np.load(ROOT/cache) as saved:
        if len(saved['q']) != 164 or np.bincount(saved['scale_id']).tolist() != [3, 1, 5, 10, 12, 30, 20, 41, 42]:
            raise ValueError('Original cache population differs')
        rows, arrays = generate_cases(reference['domain_randomization_configuration'], saved['scale_id'])
    write_json(out/'cases.json', rows)
    np.savez_compressed(out/'stimuli.npz', **arrays)
    manifest = dict(schema='tendonspin-paired-dr-suite-v1', gpu_index=0, gpu_uuid=GPU_UUID,
        cache=cache, cache_sha256=expected_cache, test_seeds=list(TEST_SEEDS), seconds=30.,
        controls=600, num_envs=len(rows), models=models,
        fixture_files={name: sha256(out/name) for name in ('cases.json', 'stimuli.npz')},
        sources=[dict(path=name, sha256=digest) for name, digest in sorted(pins.items())],
        source_reference='docs/experiments/2026-10-16-randomized-evaluation/REFERENCES_AND_PROTOCOL.md',
        original_training_unchanged=True, automatic_retry=False,
        scope='Existing training grasp cache, independent ID parameter combinations, one training lineage',
        static_input_injection='Process-local sampler fixture; original size/material/COM authoring and PhysX unchanged')
    write_json(out/'suite.json', manifest)
    write_json(out/'status.json', dict(phase='prepared', cases_per_model=len(rows), models=list(models),
        gpu_index=0, gpu_uuid=GPU_UUID, cpu_checkpoint_pairs_verified=True, sources_verified=len(pins),
        automatic_retry=False, prepared_unix_s=time.time()))
    return manifest


def validate_result(path):
    result = json.loads((path/'result.json').read_text())
    if result['status'] != 'completed' or result['wall_budget_exhausted'] or not result['inference_state_unchanged']:
        raise RuntimeError('Incomplete or invalid worker result: '+str(path))
    rows = json.loads((path/'cases_result.json').read_text())
    for row in rows:
        expected = -16. if row['first_failure_code'] in (9, 10) else 0.
        if row['terminal_failure_cost_sum'] != expected:
            raise RuntimeError('Workspace failure cost must occur exactly once')
        if row['stop_reason'] == 'wall budget':
            raise RuntimeError('Infrastructure-incomplete case')
    return result


def compare_initial(reference, candidate):
    errors = {}
    for filename in ('initial_state.npz', 'properties.npz'):
        with np.load(reference/filename) as left, np.load(candidate/filename) as right:
            if left.files != right.files:
                raise RuntimeError('Paired readback fields differ')
            errors[filename] = {key: float(np.max(np.abs(left[key]-right[key]))) for key in left.files}
            if max(errors[filename].values()) > 1e-6:
                raise RuntimeError('Actual paired initial states/properties differ: '+str(errors))
    return errors


def run(suite_dir):
    suite = json.loads((suite_dir/'suite.json').read_text())
    status_path = suite_dir/'status.json'
    previous = json.loads(status_path.read_text())
    if previous['phase'] != 'prepared':
        raise ValueError('This suite was already started; no automatic restart or overwrite')
    query = subprocess.run(['nvidia-smi', '--id='+suite['gpu_uuid'],
        '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    parts = [p.strip() for p in query.split(',')]
    # The user selected physical GPU 0 on this shared server. Record concurrent
    # utilization rather than requiring exclusive access; never alter other jobs.
    if int(parts[0]) != 0 or int(parts[3]) < 8192:
        write_json(status_path, dict(phase='error', gpu_work_started=False,
            reason='Wrong GPU identity or insufficient current free memory', gpu_before=query,
            automatic_retry=False))
        raise RuntimeError('GPU 0 identity/free memory preflight failed: '+query)
    status = dict(phase='starting', pid=os.getpid(), started_unix_s=time.time(), gpu_index=0,
        gpu_uuid=suite['gpu_uuid'], gpu_before=query, shared_gpu_observed=int(parts[2]) > 2048,
        scheduling='Explicit GPU0 selection, concurrent load recorded, other processes untouched', num_envs=suite['num_envs'],
        cases_per_model=suite['num_envs'], models=list(suite['models']), completed_models=[],
        original_training_unchanged=True, automatic_retry=False)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=suite['gpu_uuid'], OMNI_KIT_ACCEPT_EULA='YES',
        PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    def save():
        status['updated_unix_s'] = time.time()
        write_json(status_path, status)
    def worker(label, name, extra=()):
        target = suite_dir/name
        command = [sys.executable, str(ROOT/'scripts/evaluate_boya_paired_randomization.py'),
            '--suite', str(suite_dir), '--model', label, '--out', str(target), *extra]
        status.update(phase='validation' if name == 'validation' else 'evaluating',
            current_model=label, current_output=str(target), command=command)
        with (suite_dir/(name+'.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = process.pid; save()
            while process.poll() is None:
                pulse = target/'result.json'
                if pulse.is_file():
                    data = json.loads(pulse.read_text())
                    status['worker'] = {k: data.get(k) for k in ('status', 'simulation_s', 'live_cases',
                        'wall_s', 'estimated_remaining_s', 'environment_actions_per_s', 'completed_cases')}
                save(); time.sleep(5)
            status['worker_exit_code'] = process.returncode; save()
            if process.returncode:
                raise RuntimeError(f'{label} worker exited {process.returncode}; see {name}.log and result.json')
        return validate_result(target)
    try:
        save()
        cases = json.loads((suite_dir/'cases.json').read_text())
        first_by_group = {}
        for row in cases:
            first_by_group.setdefault(row['scale_id'], row['case_index'])
        smoke = worker('70M', 'validation', ['--seconds', '1.', '--case-indices',
            ','.join(str(i) for i in first_by_group.values()), '--wall-s', '600'])
        if smoke['num_envs'] != 9 or smoke['geometry_max_error_m'] > 1e-9:
            raise RuntimeError('Nine-size validation failed')
        status['validation'] = dict(passed=True, cases=9, requested_s=1., result='validation/result.json')
        save()
        for label in suite['models']:
            result = worker(label, label)
            if result['num_envs'] != 492 or result['requested_s'] != 30.:
                raise RuntimeError('Full comparison population or horizon changed')
            if label != '70M':
                status.setdefault('actual_pairing', {})[label] = compare_initial(suite_dir/'70M', suite_dir/label)
            status['completed_models'].append(label)
            save()
        rows = {label: json.loads((suite_dir/label/'cases_result.json').read_text()) for label in suite['models']}
        comparison = {label: paired_comparison(rows['70M'], rows[label]) for label in ('230M', '300M')}
        summaries = {label: json.loads((suite_dir/label/'summary.json').read_text()) for label in suite['models']}
        write_json(suite_dir/'comparison.json', dict(summaries=summaries, paired_differences=comparison,
            case_files=suite['fixture_files'], actual_pairing=status['actual_pairing'],
            limitation='ID only; three checkpoints of one seed43 training lineage; no held-out grasp/shape or real-robot claim'))
        lines = ['# Paired randomized ID comparison', '',
            'Same 164 training grasps × 3 independent test conditions; 492 episodes per model, at most30s.', '',
            '| Model | Mean net deg | Median net deg | P10 net deg | Complete30s | Mean last10s deg (complete only) |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for label, summary in summaries.items():
            angle = summary['cache_weighted']['net_deg']
            tail = summary['cache_weighted']['last10s_net_deg']['mean']
            lines.append(f"| {label} | {angle['mean']:.3f} | {angle['median']:.3f} | {angle['p10']:.3f} | "
                f"{summary['complete_window']}/492 | {tail if tail is not None else 'N/A'} |")
        lines += ['', 'See comparison.json for per-scale results, equal-scale summaries and paired clustered intervals.',
            'Physical failure prefixes are retained, never extrapolated. Completion means observed30s, not spin success.',
            'Training source, weights and original nominal evaluations are preserved.']
        (suite_dir/'comparison.md').write_text('\n'.join(lines)+'\n')
        status.update(phase='completed', comparison='comparison.json', finished_unix_s=time.time())
        save()
    except BaseException as error:
        status.update(phase='error', error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        save()
        raise


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path, help='Create a new immutable suite')
    group.add_argument('--suite', type=Path, help='Run a previously prepared suite once')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.out:
        target = args.out.resolve()
        prepared = prepare(target)
        print(json.dumps(dict(phase='prepared', out=str(target), cases=prepared['num_envs'],
            models=list(prepared['models']), source_pins=len(prepared['sources'])), indent=2), flush=True)
        if args.prepare_only:
            return
    else:
        if args.prepare_only:
            raise ValueError('prepare-only requires a fresh --out')
        target = args.suite.resolve()
    run(target)


if __name__ == '__main__':
    main()
