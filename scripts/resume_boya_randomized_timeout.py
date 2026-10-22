"""Explicit C timeout recovery; preserve all frozen training/evaluation sources.

The supervision loop below is identical to the pinned C/D loop except for the
operational evaluation wall budget. A new output first evaluates the saved own
learner, then resumes the original remaining milestones. There is no retry loop.
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

from scripts.run_boya_randomized_milestones import (
    ROOT, audit_checkpoint, audit_fresh_first_update, compare_initial, make_suite,
    paired_comparison, validate_evaluation, validate_lineage, validate_resume,
    digest, read_json, verify_sources, atomic_json,
)

OPERATIONAL_SOURCES=(
    'scripts/resume_boya_randomized_timeout.py',
    'tests/test_randomized_timeout_recovery.py',
    'docs/experiments/2026-10-21-c840m-timeout/RECOVERY.md',
)
TIMEOUT_MESSAGE='Evaluation incomplete due to wall budget; preserve partial results'


def process_alive(pid):
    try:
        return (Path('/proc')/str(int(pid))/'stat').read_text().split()[2] not in ('Z','X')
    except (OSError,ValueError,TypeError,IndexError):
        return False


def require_stopped_source(state):
    if state.get('arm')!='C' or state.get('phase')!='error':
        raise ValueError('Require the explicitly reviewed failed C supervisor')
    if process_alive(state.get('pid')) or process_alive(state.get('child_pid')):
        raise ValueError('Previous supervisor/worker still alive; prevent duplicate C')


def validate_timeout_record(failed,training):
    if (failed.get('status')!='error' or failed.get('wall_budget_exhausted') is not True
            or failed.get('error',{}).get('message')!=TIMEOUT_MESSAGE):
        raise ValueError('Only the reviewed evaluation wall-budget failure is allowed')
    if (failed.get('arm')!='C' or failed.get('training_actions_in_checkpoint')!=training['actions_executed']
            or failed.get('checkpoint',{}).get('checkpoint_sha256')!=training['checkpoint_sha256']
            or failed.get('checkpoint',{}).get('policy_sha256')!=training['policy_checkpoint_sha256']):
        raise ValueError('Timeout evaluation belongs to another learner/policy')
    if (failed.get('num_envs')!=492 or failed.get('requested_s')!=30.
            or failed.get('normalizer_updates')!=0):
        raise ValueError('Timeout evaluation population/window/frozen state differs')


def recovered_plan(original,source,stage,training,target,wall_s,provenance):
    if not isinstance(wall_s,int) or wall_s<=7200:
        raise ValueError('Explicit recovery must extend the old 7200-second budget')
    plan=copy.deepcopy(original)
    plan.update(source_run=str(stage),start_actions=training['actions_executed'],
        initial_evaluation_target=copy.deepcopy(target),evaluation_wall_s=wall_s,
        initialization='Resume own complete C learner after explicit timeout recovery; all learning/physics contracts unchanged',
        authorization='User2026-10-21 requested resolving C840M evaluation timeout and continuing; independent output, extended operational wall budget, no automatic retry',
        timeout_recovery=dict(source_root=str(source),**provenance))
    plan['remaining_targets']=[t for t in plan['targets'] if t['rounded_actions']>plan['start_actions']]
    plan['new_frozen_episodes']=1+len(plan['remaining_targets'])
    return plan


def validate_origin(plan):
    recovery=plan['timeout_recovery'];source=Path(recovery['source_root'])
    for name,key in [('plan.json','source_plan_sha256'),('status.json','source_status_sha256')]:
        if digest(source/name)!=recovery[key]:
            raise ValueError('Original failed-run identity changed: '+name)
    state=read_json(source/'status.json');require_stopped_source(state)
    if Path(state['current_run']).resolve()!=Path(plan['source_run']).resolve():
        raise ValueError('Recovery boundary differs from original failed stage')
    failed_path=Path(plan['source_run'])/'evaluation/result.json'
    if digest(failed_path)!=recovery['failed_evaluation_sha256']:
        raise ValueError('Original failed evaluation changed')
    training=audit_checkpoint(Path(plan['source_run'])/'training','C')
    validate_timeout_record(read_json(failed_path),training)
    if training['checkpoint_sha256']!=recovery['checkpoint_sha256']:
        raise ValueError('Recovery learner identity changed')
    return training


def prepare(source,out,wall_s):
    source=source.resolve();out=out.resolve()
    if source.parent!=ROOT/'outputs' or out.parent!=ROOT/'outputs':
        raise ValueError('Recovery directories must be direct children of this repository outputs')
    if out.exists():raise ValueError('Refuse overwrite or duplicate recovery: '+str(out))
    original=read_json(source/'plan.json');state=read_json(source/'status.json')
    require_stopped_source(state)
    if original.get('arm')!='C' or (original.get('gpu_index'),original.get('num_envs'))!=(1,16384):
        raise ValueError('Require the existing C/GPU1/16384 plan')
    verify_sources(ROOT,original['all_sources'])
    if digest(Path(original['backend_source']))!=original['backend_source_sha256']:
        raise ValueError('Installed backend changed')
    if digest(ROOT/original['scene_configuration']['training_cache'])!=original['cache_sha256']:
        raise ValueError('164-grasp cache changed')
    stage=Path(state['current_run']).resolve()
    if stage.parent!=source:raise ValueError('Failed stage escaped its original run')
    training=audit_checkpoint(stage/'training','C')
    target=next(t for t in original['targets'] if t['rounded_actions']==training['actions_executed'])
    previous_stage=Path(state['completed_milestones'][-1]['run'])
    previous=audit_checkpoint(previous_stage/'training','C')
    validate_evaluation(previous_stage/'evaluation',previous)
    validate_lineage(training,previous,target,original)
    failed=read_json(stage/'evaluation/result.json');validate_timeout_record(failed,training)
    reference=read_json(Path(original['reference_suite'])/'suite.json')
    if failed.get('manifest_sha256')!=reference['fixture_files']:
        raise ValueError('Failed evaluation fixture differs from immutable reference')
    provenance=dict(source_plan_sha256=digest(source/'plan.json'),
        source_status_sha256=digest(source/'status.json'),
        failed_evaluation_sha256=digest(stage/'evaluation/result.json'),
        checkpoint_sha256=training['checkpoint_sha256'],
        previous_valid_actions=previous['actions_executed'],
        old_evaluation_wall_s=7200,new_evaluation_wall_s=wall_s)
    plan=recovered_plan(original,source,stage,training,target,wall_s,provenance)
    pins={p['path']:p['sha256'] for p in original['all_sources']}
    for name in OPERATIONAL_SOURCES:pins[name]=digest(ROOT/name)
    plan['all_sources']=[dict(path=p,sha256=s) for p,s in sorted(pins.items())]
    verify_sources(ROOT,plan['all_sources']);validate_origin(plan)
    out.mkdir()
    for pin in plan['all_sources']:
        destination=out/'sources'/pin['path'];destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/pin['path'],destination)
    atomic_json(out/'plan.json',plan)
    atomic_json(out/'status.json',dict(phase='prepared',arm='C',gpu_index=1,gpu_uuid=plan['gpu_uuid'],
        num_envs=plan['num_envs'],start_actions=plan['start_actions'],
        requested_total_actions=plan['requested_total_actions'],evaluation_cases=492,
        evaluation_wall_s=wall_s,automatic_retry=False,completed_milestones=[]))
    atomic_json(out/'recovery_preflight.json',dict(status='passed',checkpoint_pair_finite=True,
        frozen_learning_contract_unchanged=True,original_source_count=len(original['all_sources']),
        operational_sources=list(OPERATIONAL_SOURCES),provenance=provenance))
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
        check();state.update(phase=phase,command=list(map(str,command)));save()
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
        gpu=subprocess.check_output(['nvidia-smi','--id='+plan['gpu_uuid'],
            '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
        parts=[p.strip()for p in gpu.split(',')]
        if int(parts[0])!=plan['gpu_index'] or int(parts[3])<12288:
            raise ValueError('GPU identity/free memory preflight failed: '+gpu)
        state['gpu_before']=gpu;save()
        previous=None
        if plan['source_run']:
            target=plan['initial_evaluation_target'];source=Path(plan['source_run'])/'training'
            stage=directory/f"actions_{target['requested_actions']:09d}";stage.mkdir()
            (stage/'training').symlink_to(source,target_is_directory=True)
            state.update(current_target=target,current_run=str(stage));save()
            # Re-evaluate the saved handover node using the new protocol before continuation.
            previous=evaluate(stage,target,source)
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
    parser.add_argument('--evaluation-wall-s',type=int,default=21600)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run-prepared',type=Path)
    args=parser.parse_args()
    if args.run_prepared:
        if args.source or args.out or args.prepare_only:parser.error('run-prepared is mutually exclusive')
        directory=args.run_prepared.resolve();plan=read_json(directory/'plan.json')
        if read_json(directory/'status.json')['phase']!='prepared':raise ValueError('Already started; no automatic retry')
        verify_sources(ROOT,plan['all_sources']);validate_origin(plan)
        return run(directory)
    if args.source is None or args.out is None:parser.error('source and out are required')
    plan=prepare(args.source,args.out,args.evaluation_wall_s)
    print(json.dumps(dict(status='prepared',out=str(args.out.resolve()),
        start_actions=plan['start_actions'],next_training_actions=plan['remaining_targets'][0]['rounded_actions'],
        evaluation_wall_s=plan['evaluation_wall_s'],source_pins=len(plan['all_sources']))),flush=True)
    return 0 if args.prepare_only else run(args.out.resolve())


if __name__=='__main__':sys.exit(main())
