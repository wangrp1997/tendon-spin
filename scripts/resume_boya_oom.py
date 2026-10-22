"""Explicit C recovery after scene-initialization OOM; unchanged learner/physics.

Reuse the already validated 920M evaluation, resume its full learner, and admit
each next worker only after sufficient free GPU memory. Worker errors still
stop the supervisor: waiting before admission is not retrying a failed worker.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from scripts.resume_boya_randomized_timeout import require_stopped_source
from scripts.run_boya_randomized_milestones import (
    ROOT, audit_checkpoint, audit_fresh_first_update, compare_initial, make_suite,
    paired_comparison, validate_evaluation, validate_lineage, validate_resume,
    digest, read_json, verify_sources, atomic_json,
)

OPERATIONAL_SOURCES = (
    'scripts/resume_boya_oom.py',
    'tests/test_boya_oom_recovery.py',
    'docs/experiments/2026-10-22-c930m-oom/RECOVERY.md',
)


def validate_oom_record(failed, log, training):
    if (failed.get('arm') != 'C' or failed.get('status') != 'error'
            or failed.get('stop_reason') != 'process error'
            or any(failed.get(k) != 0 for k in
                   ('actions_executed', 'completed_updates', 'session_actions_executed'))
            or 'CUDA error: out of memory' not in log
            or not failed.get('error', {}).get('message', '').startswith('Failed to allocate ')):
        raise ValueError('Require the reviewed scene OOM before any learner update')
    if Path(failed['resume_from']).resolve() != (ROOT/training['checkpoint']).resolve():
        raise ValueError('Failed worker did not target the completed own learner')


def recovered_plan(original, source, stage, training, target, provenance):
    plan = copy.deepcopy(original)
    plan.update(source_run=str(stage), start_actions=training['actions_executed'],
        initial_evaluation_target=copy.deepcopy(target),
        initialization='Resume own complete learner; reuse its completed paired-DR evaluation after explicit scene-OOM recovery',
        authorization='User2026-10-22 asked whether C can now restart; GPU1 admission checked against measured 16384-env memory',
        oom_recovery=dict(source_root=str(source), **provenance),
        resource_admission=dict(minimum_free_mib=15141, measured_peak_mib=10021,
            reserve_mib=5120, consecutive_samples=3, poll_seconds=5,
            scope='Before each worker only; no automatic retry after a worker error'))
    plan['remaining_targets'] = [t for t in plan['targets']
                                 if t['rounded_actions'] > plan['start_actions']]
    plan['new_frozen_episodes'] = len(plan['remaining_targets'])
    return plan


def validate_origin(plan):
    provenance = plan['oom_recovery']; source = Path(provenance['source_root'])
    for relative, field in [('plan.json', 'source_plan_sha256'),
                            ('status.json', 'source_status_sha256')]:
        if digest(source/relative) != provenance[field]:
            raise ValueError('Original failed supervisor changed: '+relative)
    state = read_json(source/'status.json'); require_stopped_source(state)
    failed_stage = Path(provenance['failed_stage'])
    if Path(state['current_run']).resolve() != failed_stage.resolve():
        raise ValueError('Original failed stage differs')
    for relative, field in [('training/result.json', 'failed_result_sha256'),
                            ('training.log', 'failed_log_sha256')]:
        if digest(failed_stage/relative) != provenance[field]:
            raise ValueError('Original OOM evidence changed')
    stage = Path(plan['source_run'])
    training = audit_checkpoint(stage/'training', 'C')
    if (training['checkpoint_sha256'] != provenance['checkpoint_sha256']
            or training['actions_executed'] != plan['start_actions']
            or state['last_completed_actions'] != plan['start_actions']):
        raise ValueError('Completed learner recovery boundary differs')
    validate_oom_record(read_json(failed_stage/'training/result.json'),
                        (failed_stage/'training.log').read_text(), training)
    for relative, sha in provenance['evaluation_files'].items():
        if digest(stage/relative) != sha:
            raise ValueError('Completed evaluation evidence changed: '+relative)
    result = validate_evaluation(stage/'evaluation', training)
    reference = read_json(Path(plan['reference_suite'])/'suite.json')
    if result['manifest_sha256'] != reference['fixture_files']:
        raise ValueError('Completed evaluation fixtures differ')
    for name, sha in reference['fixture_files'].items():
        if (digest(Path(plan['reference_suite'])/name) != sha
                or digest(stage/'evaluation_suite'/name) != sha):
            raise ValueError('Paired evaluation input bytes changed')
    compare_initial(Path(plan['reference_suite'])/'70M', stage/'evaluation')
    return training


def prepare(source, out):
    source = source.resolve(); out = out.resolve()
    if source.parent != ROOT/'outputs' or out.parent != ROOT/'outputs':
        raise ValueError('Recovery directories must be repository outputs children')
    if out.exists():
        raise ValueError('Refuse duplicate recovery or overwrite: '+str(out))
    original = read_json(source/'plan.json'); state = read_json(source/'status.json')
    require_stopped_source(state)
    if (original.get('arm'), original.get('gpu_index'), original.get('num_envs')) != ('C', 1, 16384):
        raise ValueError('Require original C/GPU1/16384 contract')
    verify_sources(ROOT, original['all_sources'])
    if digest(Path(original['backend_source'])) != original['backend_source_sha256']:
        raise ValueError('Installed backend changed')
    if digest(ROOT/original['scene_configuration']['training_cache']) != original['cache_sha256']:
        raise ValueError('164-grasp cache changed')
    failed_stage = Path(state['current_run']).resolve()
    stage = Path(state['completed_milestones'][-1]['run']).resolve()
    if failed_stage.parent != source or stage.parent != source:
        raise ValueError('Recovery stages escaped their original run')
    training = audit_checkpoint(stage/'training', 'C')
    target = next(t for t in original['targets']
                  if t['rounded_actions'] == training['actions_executed'])
    previous_stage = Path(state['completed_milestones'][-2]['run'])
    previous = audit_checkpoint(previous_stage/'training', 'C')
    validate_lineage(training, previous, target, original)
    evidence = ['evaluation/result.json', 'evaluation/cases_result.json',
                'evaluation/summary.json', 'evaluation/initial_state.npz',
                'evaluation/properties.npz', 'evaluation_suite/suite.json', 'stage.json']
    provenance = dict(source_plan_sha256=digest(source/'plan.json'),
        source_status_sha256=digest(source/'status.json'), failed_stage=str(failed_stage),
        failed_result_sha256=digest(failed_stage/'training/result.json'),
        failed_log_sha256=digest(failed_stage/'training.log'),
        checkpoint_sha256=training['checkpoint_sha256'],
        evaluation_files={name:digest(stage/name) for name in evidence})
    plan = recovered_plan(original, source, stage, training, target, provenance)
    pins = {p['path']:p['sha256'] for p in original['all_sources']}
    for name in OPERATIONAL_SOURCES:
        pins[name] = digest(ROOT/name)
    plan['all_sources'] = [dict(path=p, sha256=s) for p,s in sorted(pins.items())]
    verify_sources(ROOT, plan['all_sources']); validate_origin(plan)
    out.mkdir()
    for pin in plan['all_sources']:
        destination = out/'sources'/pin['path']; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/pin['path'], destination)
    atomic_json(out/'plan.json', plan)
    atomic_json(out/'status.json', dict(phase='prepared', arm='C', gpu_index=1,
        gpu_uuid=plan['gpu_uuid'], num_envs=plan['num_envs'], start_actions=plan['start_actions'],
        requested_total_actions=plan['requested_total_actions'], automatic_retry=False,
        evaluation_wall_s=plan['evaluation_wall_s'], evaluation_cases=492, completed_milestones=[]))
    atomic_json(out/'recovery_preflight.json', dict(status='passed',
        frozen_learning_contract_unchanged=True, original_source_count=len(original['all_sources']),
        operational_sources=list(OPERATIONAL_SOURCES), provenance=provenance,
        resource_admission=plan['resource_admission'], completed_evaluation_reused=True))
    return plan


def gpu_snapshot(plan):
    line = subprocess.check_output(['nvidia-smi', '--id='+plan['gpu_uuid'],
        '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
    fields = [v.strip() for v in line.split(',')]
    if len(fields) != 5 or int(fields[0]) != plan['gpu_index'] or fields[1] != plan['gpu_uuid']:
        raise ValueError('GPU identity differs: '+line)
    return dict(index=int(fields[0]), uuid=fields[1], used_mib=int(fields[2]),
                free_mib=int(fields[3]), utilization=int(fields[4]))


def wait_for_memory(plan, state, save, check, phase):
    admission = plan['resource_admission']; consecutive = 0
    while consecutive < admission['consecutive_samples']:
        check(); snapshot = gpu_snapshot(plan)
        consecutive = consecutive+1 if snapshot['free_mib'] >= admission['minimum_free_mib'] else 0
        state.update(phase='waiting_for_gpu_memory', pending_phase=phase,
            gpu_admission=dict(**snapshot, minimum_free_mib=admission['minimum_free_mib'],
                consecutive_passes=consecutive, required_samples=admission['consecutive_samples']))
        save()
        if consecutive < admission['consecutive_samples']:
            time.sleep(admission['poll_seconds'])
    state.pop('pending_phase', None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', type=Path)
    args = parser.parse_args()
    if args.run_prepared:
        if args.source or args.out or args.prepare_only:
            parser.error('run-prepared is mutually exclusive')
        directory = args.run_prepared.resolve(); plan = read_json(directory/'plan.json')
        if read_json(directory/'status.json')['phase'] != 'prepared':
            raise ValueError('Already started; no automatic retry')
        verify_sources(ROOT, plan['all_sources']); validate_origin(plan)
        return run(directory)
    if args.source is None or args.out is None:
        parser.error('source and out are required')
    plan = prepare(args.source, args.out)
    print(json.dumps(dict(status='prepared', start_actions=plan['start_actions'],
        next_training_actions=plan['remaining_targets'][0]['rounded_actions'],
        source_pins=len(plan['all_sources']), resource_admission=plan['resource_admission'])), flush=True)
    return 0 if args.prepare_only else run(args.out.resolve())


def run(directory):
    plan=read_json(directory/'plan.json');state=read_json(directory/'status.json')
    if state['phase']!='prepared':
        raise ValueError('Already started; no automatic retry')
    started=time.monotonic();child=None;requested=False;rows=[]
    state.update(pid=os.getpid(),phase='starting',started_unix_s=time.time())
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpu_uuid'],OMNI_KIT_ACCEPT_EULA='YES',
        PYTHONPATH=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')

    def save():
        state.update(wall_s=time.monotonic()-started,updated_unix_s=time.time())
        atomic_json(directory/'status.json',state)

    def stop(signum,frame):
        nonlocal requested
        requested=True
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    def check():
        if requested:raise RuntimeError('User stop requested; no next worker')
        verify_sources(ROOT,plan['all_sources'])
        if digest(ROOT/plan['scene_configuration']['training_cache'])!=plan['cache_sha256']:
            raise ValueError('Cache changed')
        if digest(Path(plan['backend_source']))!=plan['backend_source_sha256']:
            raise ValueError('Installed backend changed')

    def execute(command,log,phase,pulse_path):
        nonlocal child
        check();wait_for_memory(plan,state,save,check,phase)
        state.update(phase=phase,command=list(map(str,command)));save()
        with log.open('x') as handle:
            child=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=environment,
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            if requested:child.send_signal(signal.SIGTERM)
            while child.poll() is None:
                if pulse_path.is_file():
                    value=read_json(pulse_path)
                    state['worker']={k:value.get(k) for k in ('status','actions_executed',
                        'completed_updates','simulation_s','live_cases','wall_s','estimated_remaining_s',
                        'training_actions_per_wall_s','environment_actions_per_s')}
                    if (plan['arm']=='D' and phase=='training' and 'fresh_first_update' not in state
                            and value.get('first_update_parameters_changed')
                            and (pulse_path.parent/'teacher_u000001_policy.pth').is_file()):
                        state['fresh_first_update']=audit_fresh_first_update(pulse_path.parent,plan['num_envs'])
                save();time.sleep(5)
            state['worker_exit_code']=child.returncode;save()
            if child.returncode:
                raise RuntimeError(f'{phase} exited{child.returncode}; stop without retry')
        check()

    def evaluate(stage,target,training_dir):
        label=f"{plan['arm']}_{target['requested_actions']//1000000}M"
        suite_dir=stage/'evaluation_suite'
        record=make_suite(plan,suite_dir,training_dir,label)
        evaluation=stage/'evaluation'
        command=[sys.executable,ROOT/'scripts/evaluate_boya_randomized_checkpoint.py',
            '--suite',suite_dir,'--model',label,'--out',evaluation,'--wall-s',str(plan['evaluation_wall_s'])]
        execute(command,stage/'evaluation.log','randomized_evaluation',evaluation/'result.json')
        validate_evaluation(evaluation,record)
        actual=compare_initial(Path(plan['reference_suite'])/'70M',evaluation)
        baseline=read_json(Path(plan['reference_suite'])/'70M/cases_result.json')
        cases=read_json(evaluation/'cases_result.json')
        comparison=paired_comparison(baseline,cases)
        atomic_json(stage/'comparison_to_70M.json',dict(paired_differences=comparison,
            actual_initial_and_property_pairing=actual,
            reward_comparison='C and D have different explicit terminal costs; prioritize physical metrics'))
        summary=read_json(evaluation/'summary.json')
        row=dict(arm=plan['arm'],requested_actions=target['requested_actions'],
            training_actions=record['actions_executed'],checkpoint_sha256=record['checkpoint_sha256'],
            mean_net_deg=summary['cache_weighted']['net_deg']['mean'],
            median_net_deg=summary['cache_weighted']['net_deg']['median'],
            p10_net_deg=summary['cache_weighted']['net_deg']['p10'],
            complete_window=summary['complete_window'],complete_window_fraction=summary['complete_window_fraction'],
            scale_macro_net_deg=summary['scale_macro_net_deg'],
            scale_macro_complete_fraction=summary['scale_macro_complete_fraction'],
            mean_last10s_net_deg=summary['cache_weighted']['last10s_net_deg']['mean'],
            mean_reward=summary['cache_weighted']['reward_sum']['mean'],
            evaluation=str(evaluation/'result.json'),summary=str(evaluation/'summary.json'))
        rows.append(row)
        atomic_json(directory/'evaluations.json',dict(evaluation_mode=plan['evaluation_mode'],rows=rows))
        lines=['| Arm | Actions | Mean net deg | P10 deg | Retained30s | Equal-size net deg |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:
            lines.append(f"| {r['arm']} | {r['training_actions']} | {r['mean_net_deg']:.3f} | "
                f"{r['p10_net_deg']:.3f} | {r['complete_window']}/492 | {r['scale_macro_net_deg']:.3f} |")
        (directory/'evaluations.md').write_text('\n'.join(lines)+'\n')
        atomic_json(stage/'stage.json',dict(phase='completed',target=target,
            training=str(training_dir/'result.json'),randomized_evaluation=row,
            actual_initial_and_property_pairing=actual,automatic_retry=False))
        state['completed_milestones'].append(dict(**target,run=str(stage),evaluation=row['evaluation']))
        state['last_completed_actions']=record['actions_executed'];save()
        return record

    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:
        save();check()
        # The saved 920M evaluation is already complete and paired: do not rerun it.
        previous=validate_origin(plan)
        target=plan['initial_evaluation_target'];source_stage=Path(plan['source_run'])
        stage=directory/f"actions_{target['requested_actions']:09d}"
        stage.symlink_to(source_stage,target_is_directory=True)
        state.update(current_target=target,current_run=str(stage),
            last_completed_actions=previous['actions_executed'])
        state['completed_milestones'].append(dict(**target,run=str(stage),
            evaluation=str(source_stage/'evaluation/result.json'),reused_completed_evaluation=True))
        save()
        for target in plan['remaining_targets']:
            check()
            stage=directory/f"actions_{target['requested_actions']:09d}";stage.mkdir()
            state.update(current_target=target,current_run=str(stage));state.pop('worker',None);save()
            training=stage/'training'
            train=[ '--out',training,'--cache',ROOT/plan['scene_configuration']['training_cache'],
                '--randomization-config',ROOT/plan['randomization_config'],
                '--total-actions',target['requested_actions'],'--num-envs',plan['num_envs'],
                '--trace-mode','summary','--wall-s','0','--max-gpu-memory-mib','0',
                '--save-every','16','--minibatch-size','512','--seed','43',
                '--termination-profile','boya_workspace','--engine-profile','original_tgs16_4',
                '--reward-profile',plan['reward_profile'],'--scene-config',ROOT/plan['scene_config'],
                '--controller',plan['controller'],'--protocol',plan['protocol'] ]
            if previous:train+=['--resume',ROOT/previous['checkpoint']]
            wrapper='run_boya_with_property_sync.py' if plan['arm']=='C' else 'run_boya_no_drop_with_property_sync.py'
            command=[sys.executable,ROOT/'scripts'/wrapper,'--entry','train',
                '--audit',stage/'training_property_sync.json','--',*train]
            execute(command,stage/'training.log','training',training/'result.json')
            current=audit_checkpoint(training,plan['arm'])
            validate_lineage(current,previous,target,plan)
            previous=evaluate(stage,target,training)
        state.update(phase='completed',finished_unix_s=time.time());save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait()
        state.update(phase='stopped' if requested else 'error',
            error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        save()
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
