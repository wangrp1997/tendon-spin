"""Own C continuation / fresh D, with immutable paired-DR evaluation every10M.

The existing C trainer and installed engine remain pinned and untouched. D has a
separate native trainer and reward identity. Every stage saves a full learner,
then runs the same492 randomized cases before the next training segment.
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

import numpy as np

from scripts.resume_boya_spin_progress_milestones import validate_resume
from scripts.run_boya_paired_randomization import compare_initial
from tendonspin.evaluation.comparison import digest, read_json, verify_checkpoint, verify_sources
from tendonspin.evaluation.paired_randomization import paired_comparison
from tendonspin.physics.physx_property_sync import audited_source
from tendonspin.rl.resource_guard import atomic_json

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = ROOT/'outputs/boya_c_fresh1p5b_server_recovery_sync_v1'
REFERENCE = ROOT/'outputs/boya_c_paired_dr_eval_20261016_gpu0_v3'
GPU_UUIDS = {0: 'GPU-4d3badb8-5fb0-b880-f08e-6407f0093e86',
             1: 'GPU-a3f9cc58-fe6f-96d3-ab1a-e1dc33b09e7a'}
PROTOCOL = 'docs/experiments/2026-10-16-d-no-drop/PROTOCOL.md'
NEW_SOURCES = (
    'scripts/run_boya_randomized_milestones.py',
    'scripts/evaluate_boya_randomized_checkpoint.py',
    'scripts/train_boya_spin_progress_no_drop.py',
    'scripts/run_boya_no_drop_with_property_sync.py',
    'tendonspin/rl/isaac_spin_progress_no_drop.py',
    'tendonspin/rl/spin_progress_no_drop.py',
    'assets/config/boya_spin_progress_no_drop_reward.json',
    'assets/config/boya_hora_randomized_d.json',
    'tests/test_no_drop_randomized_milestones.py', PROTOCOL,
)


def audit_checkpoint(directory, expected_arm):
    """CPU: source/learner/policy pairing, finite state, physical/reward identity."""
    import torch
    torch.set_num_threads(4)
    directory = Path(directory)
    record = read_json(directory/'result.json')
    if (record['status'], record['stop_reason'], record['arm']) != ('completed', 'update budget', expected_arm):
        raise ValueError('Need normally completed own '+expected_arm+' learner')
    verify_sources(ROOT, record['sources'])
    full, policy_path = ROOT/record['checkpoint'], ROOT/record['policy_checkpoint']
    verify_checkpoint(ROOT, record, full, record['actions_executed'])
    if digest(policy_path) != record['policy_checkpoint_sha256']:
        raise ValueError('Export policy identity differs')
    learner = torch.load(full, map_location='cpu', weights_only=False)
    policy = torch.load(policy_path, map_location='cpu', weights_only=False)
    if policy['learner_sha256'] != digest(full) or policy['progress'] != learner['progress']:
        raise ValueError('Policy/full learner pairing differs')
    for key in ('model', 'running_mean_std', 'value_mean_std'):
        if set(learner[key]) != set(policy[key]):
            raise ValueError('Unpaired tensor names: '+key)
        for name, value in learner[key].items():
            if not torch.isfinite(value).all() or not torch.equal(value, policy[key][name]):
                raise ValueError('Nonfinite or unpaired tensor: '+key+'/'+name)
    for value in learner['optimizer']['state'].values():
        for item in value.values():
            if torch.is_tensor(item) and not torch.isfinite(item).all():
                raise ValueError('Nonfinite optimizer state')
    for ck, rk in [('arm','arm'), ('reward_configuration','reward_configuration'),
                   ('domain_randomization','domain_randomization_configuration'),
                   ('scene_configuration','scene_configuration'), ('termination','termination'),
                   ('engine_configuration','engine_configuration')]:
        if learner['contract'][ck] != record[rk]:
            raise ValueError('Checkpoint contract differs: '+ck)
    return record


def validate_lineage(record, previous, target, plan):
    validate_resume(record, plan, target)
    if previous is None:
        if record['resume_from'] is not None or record['session_actions_executed'] != target['rounded_actions']:
            raise ValueError('D must start from zero')
    elif (record['resumed_checkpoint_sha256'] != previous['checkpoint_sha256']
          or record['lineage_id'] != previous['lineage_id']
          or record['session_actions_executed'] != target['rounded_actions']-previous['actions_executed']):
        raise ValueError('Own full-learner chain or cumulative count differs')


def audit_fresh_first_update(directory, num_envs):
    import torch
    initial=torch.load(directory/'teacher_initial.pth',map_location='cpu',weights_only=False)
    first=torch.load(directory/'teacher_u000001.pth',map_location='cpu',weights_only=False)
    policy=torch.load(directory/'teacher_u000001_policy.pth',map_location='cpu',weights_only=False)
    batch=num_envs*8
    if (initial['arm'],initial['progress']['actions_executed'],
            initial['progress']['completed_updates'],initial['resume_source']) != ('D',0,0,None):
        raise ValueError('D inherited initial learner state')
    if first['progress']['actions_executed']!=batch or first['epoch_num']!=1 or first['agent_steps']!=batch:
        raise ValueError('D first update counts differ')
    expected_steps=batch//512*5
    if {float(v['step']) for v in first['optimizer']['state'].values()}!={float(expected_steps)}:
        raise ValueError('D first update optimizer count differs')
    if policy['learner_sha256']!=digest(directory/'teacher_u000001.pth'):
        raise ValueError('D first policy/full learner unpaired')
    for key in ('model','running_mean_std','value_mean_std'):
        for name,value in first[key].items():
            if not torch.isfinite(value).all() or not torch.equal(value,policy[key][name]):
                raise ValueError('D first update tensors nonfinite or unpaired')
    if not any(not torch.equal(v,initial['model'][k]) for k,v in first['model'].items()):
        raise ValueError('D first update failed to change model')
    evidence=dict(arm='D',fresh_actions=0,fresh_updates=0,first_actions=batch,
        first_updates=1,adam_steps_per_parameter=expected_steps,
        finite=True,policy_pair_verified=True,parameters_changed=True)
    if num_envs==16384:
        original=torch.load(ROOT/'outputs/boya_c_fresh1p5b_server_v1/actions_010000000/training/teacher_initial.pth',
            map_location='cpu',weights_only=False)
        evidence['same_seed_initial_tensors_as_fresh_C']={key:
            set(initial[key])==set(original[key]) and all(torch.equal(v,original[key][k])
                for k,v in initial[key].items()) for key in ('model','running_mean_std','value_mean_std')}
        if not all(evidence['same_seed_initial_tensors_as_fresh_C'].values()):
            raise ValueError('D fresh seed43 initial tensors differ from original fresh C')
    atomic_json(directory/'fresh_first_update_verified.json',evidence)
    return evidence


def validate_evaluation(directory, record):
    result = read_json(directory/'result.json')
    if (result['status'] != 'completed' or result['wall_budget_exhausted']
            or not result['inference_state_unchanged'] or result['num_envs'] != 492
            or result['requested_s'] != 30. or result['training_actions'] != 0
            or result['episode_resets'] != 0 or result['arm'] != record['arm']
            or result['training_actions_in_checkpoint'] != record['actions_executed']
            or result['checkpoint']['checkpoint_sha256'] != record['checkpoint_sha256']):
        raise ValueError('Incomplete or mismatched randomized evaluation')
    rows = read_json(directory/'cases_result.json')
    if len(rows) != 492 or [r['case_index'] for r in rows] != list(range(492)):
        raise ValueError('Randomized population/order differs')
    cost = record['reward_configuration']['terminal_failure_cost']
    for row in rows:
        expected = -cost if row['first_failure_code'] in (9,10) else 0.
        if row['terminal_failure_cost_sum'] != expected or row['stop_reason'] == 'wall budget':
            raise ValueError('Incorrect terminal reward or incomplete case')
    return result


def make_suite(plan, destination, training_dir, label):
    """Keep fixture bytes, case order and sources fixed; bind this own checkpoint."""
    destination.mkdir(parents=True, exist_ok=False)
    reference = read_json(Path(plan['reference_suite'])/'suite.json')
    for name, sha in reference['fixture_files'].items():
        original = Path(plan['reference_suite'])/name
        if digest(original) != sha:
            raise ValueError('Reference case/stimulus identity changed')
        shutil.copyfile(original, destination/name)
        if digest(destination/name) != sha:
            raise ValueError('Copied fixture differs')
    record = audit_checkpoint(training_dir, plan['arm'])
    model = dict(checkpoint=record['checkpoint'], checkpoint_sha256=record['checkpoint_sha256'],
        policy=record['policy_checkpoint'], policy_sha256=record['policy_checkpoint_sha256'],
        source_record=str((Path(training_dir)/'result.json').relative_to(ROOT)),
        source_record_sha256=digest(Path(training_dir)/'result.json'),
        actions_executed=record['actions_executed'], cpu_pair_verified=True)
    suite = dict(reference, gpu_index=plan['gpu_index'], gpu_uuid=plan['gpu_uuid'],
        models={label:model}, sources=plan['all_sources'], original_training_unchanged=plan['arm']=='C',
        authorization='User requested C/D randomized evaluation every10M',
        reference_fixture_suite=plan['reference_suite'])
    atomic_json(destination/'suite.json',suite)
    return record


def prepare(args):
    original = read_json(RECOVERY/'plan.json')
    verify_sources(ROOT,original['sources']+original['recovery_sources'])
    reference = read_json(args.reference_suite/'suite.json')
    if read_json(args.reference_suite/'status.json')['phase'] != 'completed':
        raise ValueError('Need the completed reference evaluation')
    verify_sources(ROOT,reference['sources'])
    for name, sha in reference['fixture_files'].items():
        if digest(args.reference_suite/name) != sha:
            raise ValueError('Reference fixture changed')
    backend, backend_sha = audited_source()
    if digest(backend) != original['backend_source_sha256']:
        raise ValueError('Installed backend changed')
    if digest(ROOT/reference['cache']) != reference['cache_sha256']:
        raise ValueError('164-grasp cache changed')
    dr = ROOT/f'assets/config/boya_hora_randomized_{args.arm.lower()}.json'
    plan = copy.deepcopy(original)
    plan.update(arm=args.arm, gpu_index=1,
        reference_suite=str(args.reference_suite), randomization_config=str(dr.relative_to(ROOT)),
        scene_config='assets/config/boya_hora_randomized_c_scene.json',
        source_run=str(args.resume_run) if args.resume_run else None,
        evaluation_protocol=PROTOCOL, evaluation_cases=492, evaluation_window_s=30.,
        evaluation_mode='same164grasps x3 fixed ID randomized conditions; native frozen policy',
        scheduling='Train segment then paired randomized evaluation, same physical GPU',
        automatic_retry=False, authorization='User2026-10-16: fresh D drop0; replace C/D10M eval with DR; then move D to GPU1 and leave GPU0 for others')
    plan['gpu_uuid']=GPU_UUIDS[plan['gpu_index']]
    if args.arm == 'C':
        if args.resume_run is None:
            raise ValueError('C requires its own completed saved milestone')
        previous = audit_checkpoint(args.resume_run/'training','C')
        target = next(t for t in original['targets'] if t['rounded_actions']==previous['actions_executed'])
        validate_resume(previous, original,target)
        status = read_json(RECOVERY/'status.json')
        if status['phase'] != 'stopped' or Path(status['current_run']) != args.resume_run:
            raise ValueError('Old supervisor must stop at this explicitly saved handover boundary')
        try:
            os.kill(status['pid'],0)
        except ProcessLookupError:
            pass
        else:
            proc=Path('/proc')/str(status['pid'])/'stat'
            if proc.exists() and proc.read_text().split()[2] != 'Z':
                raise ValueError('Old supervisor still alive; prevent duplicate training')
        plan.update(start_actions=previous['actions_executed'], initial_evaluation_target=target,
            initialization='Resume own C full learner; reward/DR/PPO/config/source contract unchanged')
    else:
        if args.resume_run is not None:
            raise ValueError('D is explicitly fresh; no C checkpoint migration')
        from tendonspin.rl.spin_progress_no_drop import PROFILE, configuration
        configuration()
        cfg = read_json(dr)
        c_cfg=read_json(ROOT/'assets/config/boya_hora_randomized_c.json')
        if {k:v for k,v in cfg.items() if k!='training'} != {k:v for k,v in c_cfg.items() if k!='training'}:
            raise ValueError('D changed physical domain randomization')
        plan.update(reward_profile=PROFILE,
            controller='boya_ninescale_dr_cache164_d_fresh_1500000000_actions_v1',
            protocol=PROTOCOL,start_actions=0,
            initialization='Fresh seed43 network/normalizers/Adam/LR/RNG/counts; only explicit drop16 becomes0')
    plan['remaining_targets']=[t for t in plan['targets'] if t['rounded_actions']>plan['start_actions']]
    plan['new_frozen_episodes']=len(plan['remaining_targets'])+(1 if args.arm=='C' else 0)
    pins={p['path']:p['sha256'] for p in reference['sources']}
    for path in NEW_SOURCES:
        pins[path]=digest(ROOT/path)
    plan['all_sources']=[dict(path=p,sha256=s) for p,s in sorted(pins.items())]
    verify_sources(ROOT,plan['all_sources'])
    if args.out.exists():
        raise ValueError('Refuse overwrite or automatic restart: '+str(args.out))
    args.out.mkdir(parents=True)
    for pin in plan['all_sources']:
        destination=args.out/'sources'/pin['path'];destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/pin['path'],destination)
    atomic_json(args.out/'plan.json',plan)
    atomic_json(args.out/'status.json',dict(phase='prepared',arm=args.arm,
        gpu_index=plan['gpu_index'],gpu_uuid=plan['gpu_uuid'],num_envs=plan['num_envs'],
        start_actions=plan['start_actions'],requested_total_actions=plan['requested_total_actions'],
        evaluation_cases=492,automatic_retry=False,completed_milestones=[]))
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
            '--suite',suite_dir,'--model',label,'--out',evaluation,'--wall-s','7200']
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
    parser.add_argument('--arm',choices=('C','D'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--resume-run',type=Path)
    parser.add_argument('--reference-suite',type=Path,default=REFERENCE)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--run-prepared',type=Path)
    args=parser.parse_args()
    if args.run_prepared:
        if args.arm or args.out or args.prepare_only:parser.error('run-prepared is mutually exclusive')
        return run(args.run_prepared.resolve())
    if not args.arm or not args.out:parser.error('arm and out are required')
    args.out=args.out.resolve();args.reference_suite=args.reference_suite.resolve()
    if args.resume_run:args.resume_run=args.resume_run.resolve()
    plan=prepare(args)
    print(json.dumps(dict(phase='prepared',arm=args.arm,out=str(args.out),
        gpu_index=plan['gpu_index'],num_envs=plan['num_envs'],start_actions=plan['start_actions'],
        evaluation_cases=492,source_pins=len(plan['all_sources']))),flush=True)
    return 0 if args.prepare_only else run(args.out)


if __name__=='__main__':
    sys.exit(main())
