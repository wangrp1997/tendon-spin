# References: existing TendonSpin Hora training/evaluation entrypoints. New local
# process supervision only; no model, reward or simulator changes.
"""Run one declared cache28 learning stage, then one frozen original-state episode."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--num-envs',type=int,default=128)
parser.add_argument('--updates',type=int,default=64)
parser.add_argument('--save-every',type=int,default=16)
parser.add_argument('--minibatch-size',type=int,default=512)
parser.add_argument('--protocol',default='docs/experiments/2026-10-08-boya-hora-cache28-128/PROTOCOL.md')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
os.nice(10)
started=time.monotonic()
state=dict(supervisor_pid=os.getpid(),phase='starting',training=None,evaluation=None,
    process_nice=os.getpriority(os.PRIO_PROCESS,0),num_envs=args.num_envs,
    requested_actions=args.num_envs*8*args.updates)
child_env=os.environ.copy();child_env['OMNI_KIT_ACCEPT_EULA']='YES';child_env['PYTHONPATH']=str(root)
child_env['OMP_NUM_THREADS']='4';child_env['MKL_NUM_THREADS']='4'
cache=root/'outputs/boya_settled_cache_v2/grasp_cache.npz'

def save():
    state['wall_s']=time.monotonic()-started
    (out/'stage.json').write_text(json.dumps(state,indent=2)+'\n')
    print('HORA_STAGE '+json.dumps(state),flush=True)

def run(phase,cmd,limit):
    state['phase']=phase;state[phase+'_command']=cmd;save()
    with (out/(phase+'.log')).open('w') as log:
        child=subprocess.Popen(cmd,cwd=root,env=child_env,stdout=log,stderr=subprocess.STDOUT)
        state['child_pid']=child.pid;save()
        try:code=child.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            state[phase]=dict(status='hard timeout',limit_s=limit);save()
            raise RuntimeError(phase+' exceeded its hard process budget')
    result=out/phase/'result.json'
    state[phase]=dict(exit_code=code,result=str(result))
    if result.exists():
        value=json.loads(result.read_text())
        for key in ('status','stop_reason','actions_executed','actual_s','valid_s','net_deg'):
            if key in value:state[phase][key]=value[key]
    save()
    if code:raise RuntimeError(phase+' failed; see saved result/log')

save()
try:
    run('training',[sys.executable,str(root/'scripts/train_boya_hora.py'),
        '--out',str(out/'training'),'--cache',str(cache),'--updates',str(args.updates),'--num-envs',str(args.num_envs),
        '--wall-s','1800','--save-every',str(args.save_every),'--seed','43',
        '--minibatch-size',str(args.minibatch_size),
        '--controller','hora_boya_cache28_teacher_v3_env128',
        '--protocol',args.protocol],1860)
    if state['training'].get('stop_reason')=='signal requested stop at update boundary':
        raise RuntimeError('Training was interrupted at user request; evaluation not launched')
    run('evaluation',[sys.executable,str(root/'scripts/evaluate_boya_hora.py'),
        '--out',str(out/'evaluation'),'--cache',str(cache),
        '--checkpoint',str(out/'training/teacher_final.pth'),
        '--source-record',str(out/'training/result.json'),'--seconds','120','--wall-s','1500'],1560)
    state['phase']='completed'
except BaseException as error:
    state['phase']='error';state['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
finally:save()
