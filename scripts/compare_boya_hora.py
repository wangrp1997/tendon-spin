# Reference: TendonSpin run_boya_hora_background.py and evaluate_boya_hora.py.
# Reuses the pinned evaluator unchanged; explicit old-policy inference migration.
"""Prepare/queue one matched old/new teacher comparison after the active run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from tendonspin.evaluation.comparison import (
    prepare, read_json, verify_sources, digest, verify_checkpoint,
    make_old_execution_contract, validate_result, compare_initial_states, write_report, dependency_completed)
from tendonspin.rl.resource_guard import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare','wait-run'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--old-run',type=Path)
    parser.add_argument('--new-run',type=Path)
    parser.add_argument('--protocol',default='docs/experiments/2026-10-08-boya-hora-matched-eval/PROTOCOL.md')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=args.out.resolve()
    if args.mode=='prepare':
        if args.old_run is None or args.new_run is None:parser.error('prepare requires --old-run and --new-run')
        plan=prepare(root,out,args.old_run,args.new_run,args.protocol)
        print(json.dumps(dict(status='prepared',out=str(out),additional_gpu_episodes=plan['extra_gpu_episodes'])))
        return
    plan=read_json(out/'plan.json')
    with (out/'queue_claim.json').open('x') as claim:
        json.dump(dict(pid=os.getpid(),started_unix_s=time.time()),claim)
    state=dict(phase='waiting_for_training_and_existing_evaluation',pid=os.getpid(),additional_gpu_episodes_launched=0)
    child=None;requested=False
    def stop(signum,frame):
        nonlocal requested
        requested=True
        if child is not None and child.poll() is None:child.send_signal(signum)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def save():
        state['updated_unix_s']=time.time();atomic_json(out/'status.json',state)
    os.nice(10);save()
    try:
        while not requested:
            stage=read_json(Path(plan['new_run'])/'stage.json')
            phase=stage['phase'];state['dependency_phase']=phase
            if dependency_completed(stage):break
            if not (Path('/proc')/str(stage['supervisor_pid'])).exists():
                raise RuntimeError('Training supervisor exited without completed stage')
            save();time.sleep(30)
        if requested:
            state['phase']='stopped';save();return
        verify_sources(root,plan['runtime_sources']);verify_sources(root,plan['comparison_pins'])
        old_path=Path(plan['old_run'])/'training/result.json'
        if digest(old_path)!=plan['old_training_record_sha256']:raise ValueError('Old training record changed')
        old_training=read_json(old_path);new_training=read_json(Path(plan['new_run'])/'training/result.json')
        for record in (old_training,new_training):
            if record['status']!='completed' or record['stop_reason']!='update budget':
                raise ValueError('Training did not normally complete its declared budget')
        old_identity=verify_checkpoint(root,old_training,plan['old_checkpoint'],plan['expected_training_actions'])
        new_identity=verify_checkpoint(root,new_training,plan['new_checkpoint'],plan['expected_training_actions'])
        new=read_json(Path(plan['reuse_new_evaluation'])/'result.json')
        command=stage['evaluation_command']
        if float(command[command.index('--wall-s')+1])!=plan['wall_s']:
            raise ValueError('Existing new evaluation wall budget differs')
        validate_result(new,plan,new_identity['checkpoint_sha256'],'frozen_'+plan['new_training_controller'])
        atomic_json(out/'checkpoint_identities.json',dict(old=old_identity,new=new_identity))
        contract=out/'old_execution_contract.json'
        atomic_json(contract,make_old_execution_contract(plan,old_training))
        command=[sys.executable,str(root/'scripts/evaluate_boya_hora.py'),
            '--out',str(out/'old_evaluation'),'--checkpoint',plan['old_checkpoint'],
            '--cache',str(root/'outputs/boya_settled_cache_v2/grasp_cache.npz'),
            '--source-record',str(contract),'--seconds',str(plan['requested_s']),'--wall-s',str(plan['wall_s'])]
        environment=os.environ.copy();environment.update(PYTHONPATH=str(root),OMNI_KIT_ACCEPT_EULA='YES',
                                                        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        state.update(phase='evaluating_old_weights_with_common_rule',command=command,additional_gpu_episodes_launched=1);save()
        with (out/'old_evaluation.log').open('x') as log:
            child=subprocess.Popen(command,cwd=root,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save();code=child.wait()
        if requested:raise RuntimeError('Comparison interrupted; preserve partial evidence')
        if code:raise RuntimeError(f'Old evaluation process exited {code}; no completed comparison')
        old=read_json(out/'old_evaluation/result.json')
        validate_result(old,plan,old_identity['checkpoint_sha256'],'frozen_'+plan['old_execution_controller'])
        initial=compare_initial_states(out/'old_evaluation',plan['reuse_new_evaluation'])
        write_report(out,plan,old,new,initial)
        state.update(phase='completed',report=str(out/'comparison.md'),independent_episodes=2)
    except BaseException as error:
        state.update(phase='error',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:save()
    return 1 if state['phase']=='error' else 0


if __name__=='__main__':sys.exit(main() or 0)
