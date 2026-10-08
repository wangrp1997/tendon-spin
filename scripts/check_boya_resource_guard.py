# Required verification of the user-requested process protection. Synthetic
# telemetry drives tiny child processes; no GPU allocation or robot simulation.
from pathlib import Path
import json,sys,time
from tendonspin.rl.resource_guard import ResourceGuard,Limits,decision,atomic_json
root=Path(__file__).resolve().parents[1]
out=root/'outputs/boya_guard_selftest_v1'
out.mkdir(exist_ok=False)
base=dict(gpu_memory_MiB=4000,gpu_utilization_percent=60,available_memory_MiB=40000,
          disk_free_GiB=300,memory_full_pressure_avg10=0)
checks=[]
for name,change,kwargs,expected in (
    ('normal',{}, {},None),
    ('GPUsoft',dict(gpu_memory_MiB=11000),{},'stop'),
    ('GPUemergency',dict(gpu_memory_MiB=13000),{},'kill'),
    ('RAMsoft',dict(available_memory_MiB=10000),{},'stop'),
    ('RAMemergency',dict(available_memory_MiB=7000),{},'kill'),
    ('heartbeat',{},dict(heartbeat_age=91),'stop'),
    ('telemetry',dict(gpu_memory_MiB=None),dict(bad_gpu_samples=3),'stop'),
    ('GPUbusy',dict(gpu_utilization_percent=99),dict(busy_samples=15),'stop'),
    ('disk',dict(disk_free_GiB=10),{},'stop'),
    ('memory pressure',dict(memory_full_pressure_avg10=6),{},'stop')):
    options=dict(bad_gpu_samples=0,busy_samples=0,heartbeat_age=0,limits=Limits());options.update(kwargs)
    actual=decision(dict(base,**change),**options)
    assert actual[0]==expected,(name,actual,expected)
    checks.append(dict(name=name,action=actual[0],passed=True))

def sample(guard):
    return dict(base,gpu_memory_MiB=11000,elapsed_s=time.monotonic()-guard.started,
                synthetic_telemetry=True,child_rss_MiB=None)

results=[]
for kind in ('cooperative','unresponsive'):
    guard=ResourceGuard(root,out/kind,limits=Limits(wall_s=15,grace_s=1,terminate_s=1),sampler=sample)
    if kind=='cooperative':
        program='''import sys,time\nfrom tendonspin.rl.resource_guard import TrainingPulse,TrainingStopRequested\np=TrainingPulse(sys.argv[1],sys.argv[2])\ntry:\n while True:p.check();time.sleep(.05)\nexcept TrainingStopRequested:sys.exit(0)\n'''
    else:
        program='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'
    state=guard.run([sys.executable,'-c',program,str(guard.stop),str(guard.heartbeat)],{'PYTHONPATH':str(root)})
    kinds=[e['kind'] for e in state['events']]
    assert state['status']=='completed',state
    if kind=='cooperative':assert state['exit_code']==0 and kinds==['released','stop requested'],state
    else:assert 'SIGKILL' in kinds and state['exit_code']!=0,state
    assert not Path(f'/proc/{state["pid"]}').exists(),'child remains alive'
    results.append(dict(kind=kind,passed=True,events=state['events'],exit_code=state['exit_code'],
                        limits_readback=state['limits_readback'],child_reclaimed=True))
record=dict(policy_checks=checks,process_checks=results,new_physics_steps=0,
    telemetry_is_synthetic=True,large_memory_or_gpu_allocation=False)
atomic_json(out/'result.json',record)
print(json.dumps(dict(policy_checks_passed=len(checks),process_checks_passed=len(results),new_physics_steps=0)))
