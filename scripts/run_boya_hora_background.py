# Reference: TendonSpin run_boya_hora_stage.py training/evaluation sequencing.
# User-authorized cumulative resume, detached execution; no resource watchdog/quotas.
"""Continue one declared learner in the background and record its actual final evaluation."""
import argparse,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from tendonspin.rl.resource_guard import atomic_json
parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--resume',type=Path,help='Omit for a fresh learner; supply only for deliberate continuation')
parser.add_argument('--total-actions',type=int,default=10000000)
parser.add_argument('--protocol',default='docs/experiments/2026-10-08-boya-hora-1024-fresh10m/PROTOCOL.md')
args=parser.parse_args();root=Path(__file__).resolve().parents[1]
out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
os.nice(10);started=time.monotonic();child=None;requested=None
state=dict(supervisor_pid=os.getpid(),phase='starting',training=None,evaluation=None,
    process_nice=os.getpriority(os.PRIO_PROCESS,0),requested_total_actions=args.total_actions,
    resume_checkpoint=str(args.resume.resolve()) if args.resume else None,
    initialization='resume' if args.resume else 'fresh network/normalizers/Adam/RNG seed43',automatic_resource_stop=False)
(out/'run_boya_hora_background.py').write_bytes(Path(__file__).read_bytes())
environment=os.environ.copy();environment.update(OMNI_KIT_ACCEPT_EULA='YES',PYTHONPATH=str(root),
    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
cache=root/'outputs/boya_settled_cache_v2/grasp_cache.npz'
def stop(signum,frame):
    global requested
    requested=signum
    if child is not None and child.poll() is None:child.send_signal(signum)
signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
def save():
    state['wall_s']=time.monotonic()-started;atomic_json(out/'stage.json',state)
    print('HORA_BACKGROUND '+json.dumps(state),flush=True)
def run(phase,command):
    global child
    state['phase']=phase;state[phase+'_command']=command;save()
    with (out/(phase+'.log')).open('w') as log:
        child=subprocess.Popen(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
        state['child_pid']=child.pid;save();code=child.wait()
    result=out/phase/'result.json';state[phase]=dict(exit_code=code,result=str(result))
    if result.exists():
        value=json.loads(result.read_text())
        for key in ('status','stop_reason','actions_executed','session_actions_executed','actual_s','valid_s','net_deg'):
            if key in value:state[phase][key]=value[key]
    save()
    if code or state[phase].get('status')!='completed':raise RuntimeError(phase+' failed; preserve record and stop')
save()
try:
    run('training',[sys.executable,str(root/'scripts/train_boya_hora.py'),'--out',str(out/'training'),
        '--cache',str(cache),*(['--resume',str(args.resume.resolve())] if args.resume else []),'--total-actions',str(args.total_actions),
        '--num-envs','1024','--trace-mode','summary','--wall-s','0','--max-gpu-memory-mib','0',
        '--save-every','16','--minibatch-size','512','--seed','43',
        '--controller',('hora_boya1024_nominal_teacher_resume10m_v1' if args.resume else 'hora_boya1024_nominal_teacher_fresh10m_v1'),'--protocol',args.protocol])
    if requested is not None or state['training'].get('stop_reason')!='update budget':
        state['phase']='stopped';state['evaluation']='not launched after requested/non-budget stop'
    else:
        run('evaluation',[sys.executable,str(root/'scripts/evaluate_boya_hora.py'),
            '--out',str(out/'evaluation'),'--cache',str(cache),
            '--checkpoint',str(out/'training/teacher_final.pth'),
            '--source-record',str(out/'training/result.json'),'--seconds','120','--wall-s','1500'])
        state['phase']='completed'
except BaseException as error:
    state['phase']='error';state['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
finally:save()
