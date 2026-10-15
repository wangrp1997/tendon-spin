"""Continue an own C learner after an explicitly reviewed initialization fault.

Original sources/contracts and failed outputs remain immutable. A separate run
links validated historical milestones, records the accepted diagnostic episode,
then trains each remaining target through the unchanged native workers. The only
runtime delta is the separately pinned host-copy synchronization adapter.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts.run_boya_hora_fresh_milestones import ROOT, report, require_stage
from scripts.spin_progress_milestone_checks import evaluation_row
from tendonspin.evaluation.comparison import digest, read_json, verify_checkpoint, verify_sources
from tendonspin.physics.physx_property_sync import audited_source
from tendonspin.physics.learning_scene import load_scene_configuration
from tendonspin.rl.resource_guard import atomic_json


def remaining_targets(targets, completed_actions):
    if not any(t['rounded_actions']==completed_actions for t in targets):
        raise ValueError('Resume actions must be a completed declared milestone')
    return [t for t in targets if t['rounded_actions']>completed_actions]


def validate_resume(training, plan, target):
    if (training['status'], training['stop_reason'], training['actions_executed']) != ('completed', 'update budget', target['rounded_actions']):
        raise ValueError('Resume requires a normally completed own training milestone')
    if (training['num_envs'], training['seed'], training['controller'], training['scene_configuration']) != (plan['num_envs'], plan['seed'], plan['controller'], plan['scene_configuration']):
        raise ValueError('Resume differs from the frozen configuration')
    verify_sources(ROOT, training['sources'])
    verify_checkpoint(ROOT, training, ROOT/training['checkpoint'], target['rounded_actions'])


def worker(entry, audit, arguments):
    return [sys.executable, str(ROOT/'scripts/run_boya_with_property_sync.py'),
            '--entry', entry, '--audit', str(audit), '--', *map(str, arguments)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-requested-actions', type=int, required=True)
    parser.add_argument('--recovered-evaluation', type=Path, required=True)
    parser.add_argument('--diagnostic-provenance', type=Path, required=True)
    parser.add_argument('--scene-config', type=Path, required=True)
    parser.add_argument('--randomization-config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Validate and report history/recovered node; launch no training')
    args = parser.parse_args()
    original, out = args.original_run.resolve(), args.out.resolve()
    plan = read_json(original/'plan.json')
    verify_sources(ROOT, plan['sources'])
    if read_json(original/'status.json')['phase']!='error':
        raise ValueError('Expected the explicitly reviewed stopped original run')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':
        raise ValueError('This server recovery is authorized on GPU 1 only')
    target = next(t for t in plan['targets'] if t['requested_actions']==args.resume_requested_actions)
    old_stage = original/f'actions_{args.resume_requested_actions:09d}'
    previous = read_json(old_stage/'training/result.json')
    validate_resume(previous, plan, target)
    future = remaining_targets(plan['targets'], previous['actions_executed'])
    scene, randomization = args.scene_config.resolve(), args.randomization_config.resolve()
    if load_scene_configuration(scene)!=plan['scene_configuration']:
        raise ValueError('Recovery scene differs from frozen plan')
    if read_json(randomization)!=previous['domain_randomization_configuration']:
        raise ValueError('Recovery domain randomization differs')
    cache = ROOT/plan['scene_configuration']['training_cache']
    if digest(cache)!=plan['cache_sha256']:
        raise ValueError('Grasp cache changed')
    backend, backend_sha = audited_source()
    recovered = args.recovered_evaluation.resolve()
    provenance = read_json(args.diagnostic_provenance)
    if (provenance['checkpoint_sha256']!=previous['checkpoint_sha256'] or provenance['installed_sources_modified']
            or provenance['evaluator_sha256']!=digest(ROOT/'scripts/evaluate_boya_spin_progress.py')):
        raise ValueError('Diagnostic episode provenance differs from own frozen worker/checkpoint')
    required = {'set_masses_index', 'set_coms_index', '_get_cpu_env_ids'}
    if not required.issubset(provenance['synchronized_copies']):
        raise ValueError('Diagnostic episode did not apply all declared synchronization boundaries')
    recovered_record=read_json(recovered/'result.json')
    properties=recovered_record['evaluation_domain_properties']
    if (properties['randomized'] or properties['mass_kg']!=.05 or properties['friction']!=.5
            or properties['com_m']!=[0.,0.,0.] or properties['external_force_N']!=[0.,0.,0.]
            or properties['p_gain_ratio']!=1. or properties['d_gain_ratio']!=1.
            or max(properties['readback']['errors'].values())>1e-6
            or max(recovered_record['initial_state_max_errors'].values())!=0
            or recovered_record['reward_configuration']!=previous['reward_configuration']
            or recovered_record['engine_configuration']!=previous['engine_configuration']):
        raise ValueError('Recovered episode differs from the verified nominal physical/reward contract')
    paths = ['scripts/resume_boya_spin_progress_milestones.py',
        'scripts/run_boya_with_property_sync.py','tendonspin/physics/physx_property_sync.py']
    pins = [dict(path=p, sha256=digest(ROOT/p)) for p in paths]
    out.mkdir(parents=True, exist_ok=False)
    for pin in pins:
        snapshot=out/'sources'/pin['path'];snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes((ROOT/pin['path']).read_bytes())
    recovery_plan = dict(plan, initialization='Resume OWN full learner after reviewed initialization failure; not fresh',
        original_run=str(original), start_actions=previous['actions_executed'], remaining_targets=future,
        recovery_sources=pins, backend_source=str(backend), backend_source_sha256=backend_sha,
        recovered_evaluation=str(recovered), diagnostic_provenance=str(args.diagnostic_provenance.resolve()),
        diagnostic_provenance_sha256=digest(args.diagnostic_provenance),
        compatibility_change='Scoped waits for existing PhysX mass/COM/index pinned CPU copies only',
        authorization='User: 先push再恢复？; publish verified recovery code before launching',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    atomic_json(out/'plan.json', recovery_plan)
    state = dict(pid=os.getpid(),phase='validating_history',arm='C',num_envs=plan['num_envs'],
        start_actions=previous['actions_executed'],requested_total_actions=plan['requested_total_actions'],
        rounded_final_actions=plan['rounded_final_actions'], original_run=str(original),
        completed_milestones=[],automatic_retry=False)
    started=time.monotonic();child=None;requested=False;rows=[]
    environment=os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT),OMNI_KIT_ACCEPT_EULA='YES',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')

    def save():
        state.update(wall_s=time.monotonic()-started,updated_unix_s=time.time())
        atomic_json(out/'status.json',state)

    def stop(signum,frame):
        nonlocal requested
        requested=True
        if child is not None and child.poll() is None:child.send_signal(signum)

    def execute(command, log, stage=None, phase=None):
        nonlocal child
        if requested:raise RuntimeError('Stop requested before worker launch')
        state.update(command=command)
        save()
        with log.open('x') as handle:
            child=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save()
            if requested:child.send_signal(signal.SIGTERM)
            code=child.wait()
        if code:raise RuntimeError(f'{phase or "worker"} exited {code}; stop without retry')
        if stage is not None:
            value=read_json(log.parent/phase/'result.json')
            stage[phase]=dict(exit_code=code,**{k:value[k] for k in ('status','stop_reason','actions_executed','actual_s','valid_s','net_deg') if k in value})
            if value['status']!='completed':raise RuntimeError(f'Incomplete {phase}; stop without retry')
            atomic_json(log.parent/'stage.json',stage)

    def completed(run,t,current_previous):
        row,current=evaluation_row(run,t,current_previous,plan['num_envs'])
        if current['scene_configuration']!=plan['scene_configuration']:
            raise ValueError('Completed milestone scene changed')
        rows.append(row);report(out,rows)
        eligible=[r for r in rows if r['complete_window'] and r['net_deg']>0 and r['final_10s_net_deg'] is not None and r['final_10s_net_deg']>0]
        if eligible:atomic_json(out/'best_observed.json',max(eligible,key=lambda r:r['net_deg']))
        state['completed_milestones'].append(dict(**t,run=str(run),evaluation=row['evaluation']))
        save()
        return current

    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop);save()
    try:
        historical_previous=None
        for t in plan['targets']:
            if t['rounded_actions']>=target['rounded_actions']:break
            run=original/f'actions_{t["requested_actions"]:09d}'
            require_stage(read_json(run/'stage.json'),t['rounded_actions'])
            historical_previous=completed(run,t,historical_previous)
        # Explicitly accept the separately audited episode without rewriting the
        # original failed stage. Training and evaluation artifacts are references.
        run=out/f'actions_{target["requested_actions"]:09d}';run.mkdir()
        (run/'training').symlink_to(old_stage/'training',target_is_directory=True)
        (run/'evaluation').symlink_to(recovered,target_is_directory=True)
        atomic_json(run/'recovery.json',dict(original_failed_stage=str(old_stage),
            accepted_diagnostic_evaluation=str(recovered),provenance=recovery_plan['diagnostic_provenance'],
            provenance_sha256=recovery_plan['diagnostic_provenance_sha256'],
            original_failed_result_sha256=digest(old_stage/'evaluation/result.json'),
            accepted_result_sha256=digest(recovered/'result.json'),
            evaluation_artifacts=[dict(path=str(p),sha256=digest(p)) for p in sorted(recovered.iterdir()) if p.is_file()],
            acceptance='Explicit recovery after user requested push then resume; diagnostic remains labelled and original failure preserved'))
        stage=dict(phase='analysis',training=dict(exit_code=0,status='completed',stop_reason='update budget',actions_executed=previous['actions_executed'],reused=True),
            evaluation=dict(exit_code=0,status='completed',reused_diagnostic=True))
        execute([sys.executable,str(ROOT/'scripts/analyze_boya_small_budget.py'),'--evaluation',str(run/'evaluation'),
            '--training',str(run/'training/result.json'),'--out',str(run/'analysis')],run/'analysis.log',stage,'analysis')
        stage['phase']='completed';atomic_json(run/'stage.json',stage)
        require_stage(stage,target['rounded_actions'])
        previous=completed(run,target,historical_previous)
        if args.prepare_only:
            state['phase']='prepared';save()
            return 0
        for t in future:
            if requested:break
            verify_sources(ROOT,plan['sources']);verify_sources(ROOT,pins);verify_sources(ROOT,previous['sources'])
            if digest(cache)!=plan['cache_sha256'] or digest(backend)!=backend_sha:
                raise ValueError('Cache or installed backend changed')
            verify_checkpoint(ROOT,previous,ROOT/previous['checkpoint'],previous['actions_executed'])
            run=out/f'actions_{t["requested_actions"]:09d}';run.mkdir()
            state.update(phase='training_then_evaluation',current_target=t,current_run=str(run));save()
            stage=dict(phase='training',num_envs=plan['num_envs'],resume_checkpoint=previous['checkpoint'],
                initialization='resume own full learner',automatic_retry=False,runtime_property_sync=True)
            atomic_json(run/'stage.json',stage)
            train_args=['--out',run/'training','--cache',cache,'--randomization-config',randomization,
                '--resume',ROOT/previous['checkpoint'],'--total-actions',t['requested_actions'],
                '--num-envs',plan['num_envs'],'--trace-mode','summary','--wall-s','0','--max-gpu-memory-mib','0',
                '--save-every','16','--minibatch-size','512','--seed','43','--termination-profile','boya_workspace',
                '--engine-profile','original_tgs16_4','--reward-profile',plan['reward_profile'],
                '--scene-config',scene,'--controller',plan['controller'],'--protocol',plan['protocol']]
            execute(worker('train',run/'training_property_sync.json',train_args),run/'training.log',stage,'training')
            stage['phase']='evaluation';atomic_json(run/'stage.json',stage)
            eval_args=['--out',run/'evaluation','--cache',cache,'--checkpoint',run/'training/teacher_final.pth',
                '--source-record',run/'training/result.json','--seconds','30','--wall-s','1500',
                '--reward-diagnostics','--initial-reference',recovered/'initial_state.npz']
            execute(worker('evaluate',run/'evaluation_property_sync.json',eval_args),run/'evaluation.log',stage,'evaluation')
            stage['phase']='analysis';atomic_json(run/'stage.json',stage)
            execute([sys.executable,str(ROOT/'scripts/analyze_boya_small_budget.py'),'--evaluation',str(run/'evaluation'),
                '--training',str(run/'training/result.json'),'--out',str(run/'analysis')],run/'analysis.log',stage,'analysis')
            stage['phase']='completed';atomic_json(run/'stage.json',stage)
            require_stage(stage,t['rounded_actions'])
            previous=completed(run,t,previous)
        state['phase']='stopped' if requested else 'completed'
    except BaseException as exc:
        state.update(phase='stopped' if requested else 'error',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        if 'current_run' in state:
            stage_path=Path(state['current_run'])/'stage.json'
            failed_stage=read_json(stage_path)
            failed_stage.update(phase=state['phase'],error=state['error'])
            atomic_json(stage_path,failed_stage)
    finally:save()
    return 1 if state['phase']=='error' else 0


if __name__=='__main__':
    sys.exit(main())
