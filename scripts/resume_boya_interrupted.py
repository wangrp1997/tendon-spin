"""Explicit C/D continuation from an interrupted worker's paired periodic learner.

Preserve old evidence and all learning/physics/evaluation sources. Only unfinished
updates since the last paired snapshot are repeated. No worker is retried.
"""
import argparse
import copy
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from scripts.resume_boya_oom import wait_for_memory
from scripts.resume_boya_randomized_timeout import process_alive
from scripts.run_boya_randomized_milestones import (
    ROOT, audit_checkpoint, audit_fresh_first_update, compare_initial, make_suite,
    paired_comparison, validate_evaluation, validate_lineage, digest, read_json,
    verify_sources, atomic_json,
)

OPERATIONAL_SOURCES = (
    'scripts/resume_boya_interrupted.py',
    'tests/test_interrupted_recovery.py',
    'docs/experiments/2026-10-24-interrupted/RECOVERY.md',
)


def require_interrupted(state, record):
    if (state.get('arm') not in ('C', 'D') or state.get('phase') != 'training'
            or record.get('arm') != state['arm']
            or record.get('status') != 'update completed'
            or record.get('stop_reason') is not None or record.get('error')):
        raise ValueError('Require the reviewed interrupted training, not an execution failure')
    if any(process_alive(state.get(k)) for k in ('pid', 'child_pid')):
        raise ValueError('Previous supervisor/worker still alive; refuse duplicate training')


def audit_periodic(source):
    """CPU-only: validate a complete learner while its original result is unfinished."""
    import torch
    torch.set_num_threads(4)
    source = Path(source).resolve()
    plan = read_json(source/'plan.json'); state = read_json(source/'status.json')
    stage = Path(state['current_run']).resolve(); directory = stage/'training'
    record = read_json(directory/'result.json')
    require_interrupted(state, record)
    if stage.parent != source or (plan['arm'], plan['gpu_index'], plan['num_envs']) != (state['arm'], 1, 16384):
        raise ValueError('Interrupted stage/GPU/environment identity differs')
    verify_sources(ROOT, plan['all_sources']); verify_sources(ROOT, record['sources'])
    if (digest(Path(plan['backend_source'])) != plan['backend_source_sha256']
            or digest(ROOT/plan['scene_configuration']['training_cache']) != plan['cache_sha256']):
        raise ValueError('Frozen backend or grasp cache changed')
    meta = read_json(directory/'latest_checkpoint.json')
    for field, record_field, sha in [('path','checkpoint','sha256'),
                                    ('policy_path','policy_checkpoint','policy_sha256')]:
        path = Path(meta[field]).resolve()
        if (path.parent != directory or path != (ROOT/record[record_field]).resolve()
                or digest(path) != meta[sha]):
            raise ValueError('Latest learner/policy path or hash differs')
    full = torch.load(meta['path'], map_location='cpu', weights_only=False)
    policy = torch.load(meta['policy_path'], map_location='cpu', weights_only=False)
    prior = torch.load(record['resume_from'], map_location='cpu', weights_only=False)
    progress = full['progress']; batch = plan['num_envs']*8
    if (full.get('schema') != 'tendonspin-hora-resume-v1'
            or policy['learner_sha256'] != meta['sha256'] or policy['progress'] != progress
            or full['agent_steps'] != progress['actions_executed']
            or full['epoch_num'] != progress['completed_updates']
            or progress['actions_executed'] != progress['completed_updates']*batch
            or full['contract'] != prior['contract'] or full['contract']['arm'] != plan['arm']
            or policy['arm'] != plan['arm'] or meta['arm'] != plan['arm']
            or progress['lineage_id'] != prior['progress']['lineage_id']
            or record['lineage_id'] != progress['lineage_id']):
        raise ValueError('Own paired learner/progress/learning contract differs')
    for key, value in progress.items():
        if meta.get(key) != value:
            raise ValueError('Latest metadata progress differs: '+key)
    if not (prior['agent_steps'] < full['agent_steps'] <= record['actions_executed']
            < state['current_target']['rounded_actions']):
        raise ValueError('Periodic snapshot is outside the unfinished own milestone')
    for key in ('model','running_mean_std','value_mean_std'):
        if (full[key].keys() != policy[key].keys()
                or not all(torch.isfinite(v).all() and torch.equal(v,policy[key][k])
                           for k,v in full[key].items())):
            raise ValueError('Nonfinite or unpaired model/normalizer: '+key)
    optimizer = full['optimizer']['state']
    if (not optimizer or not all(torch.isfinite(v).all() for entry in optimizer.values()
                                for v in entry.values() if torch.is_tensor(v))
            or {float(entry['step']) for entry in optimizer.values()} !=
                {float(progress['completed_updates']*(batch//512)*5)}
            or not math.isfinite(full['learning_rate'])
            or not all(k in full for k in ('scheduler','torch_rng','cuda_rng','numpy_rng','python_rng'))):
        raise ValueError('Incomplete/nonfinite optimizer, scheduler or random state')
    descriptor = copy.deepcopy(record)
    descriptor.update(progress, checkpoint=str(Path(meta['path']).relative_to(ROOT)),
        checkpoint_sha256=meta['sha256'], policy_checkpoint=str(Path(meta['policy_path']).relative_to(ROOT)),
        policy_checkpoint_sha256=meta['policy_sha256'])
    return descriptor, dict(checkpoint_sha256=meta['sha256'], policy_sha256=meta['policy_sha256'],
        restored_actions=progress['actions_executed'], restored_updates=progress['completed_updates'],
        discarded_unsaved_updates=record['completed_updates']-progress['completed_updates'],
        discarded_unsaved_actions=record['actions_executed']-progress['actions_executed'],
        exact_saved_learning_state_verified=True, environment_reset_from_original_cache=True)


def continued_plan(original, source, stage, previous, evidence):
    plan = copy.deepcopy(original)
    plan.update(source_run=str(stage), start_actions=previous['actions_executed'],
        initialization='Resume own paired periodic learner after reviewed environment interruption',
        authorization='User2026-10-24 explicitly requested restoring C and D training after checkpoint diagnosis',
        interruption_recovery=dict(source_root=str(source), **evidence),
        evaluation_wall_s=original.get('evaluation_wall_s',7200),
        resource_admission=dict(minimum_free_mib=15141, measured_peak_mib=10021,
            reserve_mib=5120, consecutive_samples=3, poll_seconds=5,
            scope='Before each worker only; no automatic retry after a worker error'))
    plan['remaining_targets'] = [t for t in plan['targets'] if t['rounded_actions'] > plan['start_actions']]
    plan['new_frozen_episodes'] = len(plan['remaining_targets'])
    return plan


def validate_origin(plan):
    evidence=plan['interruption_recovery']; source=Path(evidence['source_root'])
    for path,sha in evidence['preserved_files'].items():
        if digest(ROOT/path)!=sha:
            raise ValueError('Preserved interruption evidence changed: '+path)
    previous, audit=audit_periodic(source)
    if audit!=evidence['audit'] or previous['actions_executed']!=plan['start_actions']:
        raise ValueError('Recovery learner identity changed')
    return previous


def prepare(source, out):
    source=source.resolve(); out=out.resolve()
    if source.parent!=ROOT/'outputs' or out.parent!=ROOT/'outputs' or out.exists():
        raise ValueError('Require existing own source and a new repository output directory')
    previous,audit=audit_periodic(source)
    original=read_json(source/'plan.json'); state=read_json(source/'status.json')
    stage=Path(state['current_run'])
    # The last completed evaluation remains historical; never label the partial
    # worker as completed and never evaluate its periodic boundary as a milestone.
    last=Path(state['completed_milestones'][-1]['run'])
    last_training=audit_checkpoint(last/'training',original['arm'])
    validate_evaluation(last/'evaluation',last_training)
    if last_training['checkpoint_sha256']!=previous['resumed_checkpoint_sha256']:
        raise ValueError('Interrupted worker did not resume the last completed own milestone')
    preserved=[source/'plan.json',source/'status.json',stage/'training.log',
        stage/'training/result.json',stage/'training/latest_checkpoint.json',
        ROOT/previous['checkpoint'],ROOT/previous['policy_checkpoint']]
    reference=Path(original['reference_suite'])
    for name,sha in read_json(reference/'suite.json')['fixture_files'].items():
        if digest(reference/name)!=sha:raise ValueError('Paired evaluation fixture changed')
        preserved.append(reference/name)
    evidence=dict(audit=audit,preserved_files={str(p.relative_to(ROOT)):digest(p) for p in preserved})
    plan=continued_plan(original,source,stage,previous,evidence)
    pins={p['path']:p['sha256'] for p in original['all_sources']}
    pins.update({p:digest(ROOT/p) for p in OPERATIONAL_SOURCES})
    plan['all_sources']=[dict(path=p,sha256=s) for p,s in sorted(pins.items())]
    verify_sources(ROOT,plan['all_sources']); validate_origin(plan)
    out.mkdir()
    for pin in plan['all_sources']:
        target=out/'sources'/pin['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/pin['path'],target)
    atomic_json(out/'plan.json',plan)
    atomic_json(out/'continuation_source.json',dict(record_type='periodic_learner_resume_descriptor',
        original_record=str(stage/'training/result.json'),audit=audit,
        checkpoint=previous['checkpoint'],checkpoint_sha256=previous['checkpoint_sha256']))
    atomic_json(out/'status.json',dict(phase='prepared',arm=plan['arm'],gpu_index=1,
        gpu_uuid=plan['gpu_uuid'],num_envs=plan['num_envs'],start_actions=plan['start_actions'],
        requested_total_actions=plan['requested_total_actions'],automatic_retry=False,
        evaluation_wall_s=plan['evaluation_wall_s'],evaluation_cases=492,completed_milestones=[]))
    atomic_json(out/'recovery_preflight.json',dict(status='passed',**audit,
        frozen_training_and_evaluation_unchanged=True,source_pin_count=len(plan['all_sources']),
        last_completed_evaluation_actions=last_training['actions_executed'],
        first_target=plan['remaining_targets'][0],resource_admission=plan['resource_admission']))
    return plan


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
        previous=validate_origin(plan)
        state.update(last_completed_actions=previous['actions_executed'],
            restored_updates=previous['completed_updates'],
            restoration=plan['interruption_recovery']['audit'])
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run-prepared',type=Path)
    args=parser.parse_args()
    if args.run_prepared:
        if args.source or args.out or args.prepare_only:parser.error('run-prepared is mutually exclusive')
        out=args.run_prepared.resolve(); plan=read_json(out/'plan.json')
    else:
        if args.source is None or args.out is None:parser.error('source and out are required')
        out=args.out.resolve(); plan=prepare(args.source,out)
        if args.prepare_only:
            print(read_json(out/'recovery_preflight.json'),flush=True)
            return 0
    if read_json(out/'status.json')['phase']!='prepared':
        raise ValueError('Already started; no automatic retry')
    with (ROOT/'outputs'/('.interrupted_resume_'+plan['arm']+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_sources(ROOT,plan['all_sources']);validate_origin(plan)
        return run(out)


if __name__=='__main__':
    sys.exit(main())
