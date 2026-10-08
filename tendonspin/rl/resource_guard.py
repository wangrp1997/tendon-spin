# References: Linux cgroup v2 cpu/memory/io controllers and systemd.resource-control
# (CPUQuota, MemoryHigh/Max/SwapMax, IOWeight); systemd.kill/service semantics.
# New TendonSpin per-job watchdog; no policy, reward or physics changes.
"""Isolate one owned simulation service; stop it before exhausting desktop resources."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

GIB=1024**3


def atomic_json(path,value):
    path=Path(path)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


class TrainingStopRequested(Exception):
    """Controlled stop at a control-action boundary; incomplete PPO rollout discarded."""


class TrainingPulse:
    def __init__(self,stop_file,heartbeat):
        self.stop_file=Path(stop_file) if stop_file else None
        self.heartbeat=Path(heartbeat) if heartbeat else None

    def check(self,phase='control'):
        if self.heartbeat:
            atomic_json(self.heartbeat,dict(pid=os.getpid(),phase=phase,unix_s=time.time()))
        if self.stop_file and self.stop_file.exists():
            raise TrainingStopRequested(self.stop_file.read_text().strip())


@dataclass
class Limits:
    gpu_stop_mib:int=10240
    gpu_kill_mib:int=12288
    available_stop_mib:int=12288
    available_kill_mib:int=8192
    heartbeat_s:float=90.
    wall_s:float=300.
    grace_s:float=8.
    terminate_s:float=5.


def decision(sample,*,bad_gpu_samples,busy_samples,heartbeat_age,limits):
    memory=sample.get('gpu_memory_MiB')
    available=sample['available_memory_MiB']
    if available<limits.available_kill_mib:return 'kill','system available memory <8GiB'
    if memory is not None and memory>limits.gpu_kill_mib:return 'kill','whole-device GPU memory >12GiB'
    if available<limits.available_stop_mib:return 'stop','system available memory <12GiB'
    if memory is not None and memory>limits.gpu_stop_mib:return 'stop','whole-device GPU memory >10GiB'
    if sample['disk_free_GiB']<20:return 'stop','disk free <20GiB'
    if sample['memory_full_pressure_avg10']>5:return 'stop','sustained system memory pressure'
    if bad_gpu_samples>=3:return 'stop','GPU telemetry unavailable for3samples'
    if busy_samples>=15:return 'stop','GPU utilization >95% for15samples'
    if heartbeat_age>limits.heartbeat_s:return 'stop','training heartbeat stale'
    return None,None


class ResourceGuard:
    def __init__(self,root,directory,*,limits=None,sampler=None):
        self.root=Path(root).resolve();self.directory=Path(directory).resolve()
        self.directory.mkdir(parents=True,exist_ok=False)
        self.limits=limits or Limits();self.sampler=sampler
        self.unit=f'tendonspin-{os.getpid()}-{time.monotonic_ns()}.service'
        self.ready=self.directory/'READY'
        self.stop=self.directory/'STOP'
        self.heartbeat=self.directory/'heartbeat.json'
        self.events=[];self.samples=[];self.pid=None
        self.state=dict(unit=self.unit,monitor_pid=os.getpid(),monitor_nice=os.getpriority(os.PRIO_PROCESS,0),events=self.events)

    def command(self,*args,check=True):
        return subprocess.run(['systemctl','--user',*args],capture_output=True,text=True,timeout=5,check=check)

    def properties(self):
        result=self.command('show',self.unit,'--property=MainPID,ControlGroup,ActiveState,SubState,Result,ExecMainCode,ExecMainStatus,Nice,CPUQuotaPerSecUSec,MemoryHigh,MemoryMax,MemorySwapMax,RuntimeMaxUSec,IOWeight,IOSchedulingClass',check=False)
        if result.returncode:
            return {'ActiveState':'unavailable','query_error':result.stderr.strip()}
        output=result.stdout
        return dict(line.split('=',1) for line in output.splitlines() if '=' in line)

    def event(self,kind,reason):
        self.events.append(dict(elapsed_s=time.monotonic()-self.started,kind=kind,reason=reason))
        atomic_json(self.directory/'guard.json',self.state)

    def sample(self):
        values={}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,value=line.split(':',1)
            values[key]=int(value.split()[0])
        pressure=next(line for line in Path('/proc/pressure/memory').read_text().splitlines() if line.startswith('full '))
        psi=dict(item.split('=') for item in pressure.split()[1:])
        sample=dict(elapsed_s=time.monotonic()-self.started,available_memory_MiB=values['MemAvailable']/1024,
            memory_full_pressure_avg10=float(psi['avg10']),disk_free_GiB=shutil.disk_usage(self.root).free/GIB)
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2.5,check=True)
            memory,util,temp=map(int,result.stdout.strip().splitlines()[0].split(','))
            sample.update(gpu_memory_MiB=memory,gpu_utilization_percent=util,gpu_temperature_C=temp)
        except (subprocess.SubprocessError,ValueError) as error:
            sample.update(gpu_memory_MiB=None,gpu_utilization_percent=None,gpu_error=type(error).__name__)
        try:
            line=next(x for x in Path(f'/proc/{self.pid}/status').read_text().splitlines() if x.startswith('VmRSS:'))
            sample['child_rss_MiB']=int(line.split()[1])/1024
        except (FileNotFoundError,StopIteration):sample['child_rss_MiB']=None
        cgroup=Path('/sys/fs/cgroup')/self.state['limits_readback']['properties']['ControlGroup'].lstrip('/')
        try:sample['job_memory_MiB']=int((cgroup/'memory.current').read_text())/1024**2
        except FileNotFoundError:sample['job_memory_MiB']=None
        return sample

    def kill(self,signum):
        self.command('kill','--kill-whom=all','--signal='+signum,self.unit,check=False)

    def run(self,cmd,environment):
        self.started=time.monotonic();run=None
        self.state.update(command=cmd,status='arming',limits=vars(self.limits))
        source=Path(__file__);(self.directory/'resource_guard.py').write_bytes(source.read_bytes())
        properties={'CPUQuota':'400%','MemoryHigh':'12G','MemoryMax':'16G','MemorySwapMax':'0',
            'OOMPolicy':'stop','KillMode':'control-group','Nice':'10','IOWeight':'10','IOSchedulingClass':'idle',
            'RuntimeMaxSec':str(self.limits.wall_s),'TimeoutStopSec':'8','SendSIGKILL':'yes'}
        launch=['systemd-run','--user','--quiet','--wait','--pipe','--unit',self.unit,'--working-directory',str(self.root)]
        for name,value in properties.items():launch+=['--property',name+'='+value]
        for name,value in environment.items():launch+=['--setenv',name+'='+value]
        launch += [sys.executable,str(self.root/'scripts/guarded_boya_entry.py'),'--ready',str(self.ready),'--',*cmd]
        try:
            with (self.directory/'process.log').open('w') as log:
                run=subprocess.Popen(launch,stdout=log,stderr=subprocess.STDOUT)
                for _ in range(100):
                    if run.poll() is not None:raise RuntimeError('resource group launch failed: '+(self.directory/'process.log').read_text()[-2000:])
                    prop=self.properties()
                    if int(prop.get('MainPID','0')) and prop.get('ControlGroup'):break
                    time.sleep(.1)
                else:raise RuntimeError('resource group startup timed out')
                self.pid=int(prop['MainPID'])
                cgroup=Path('/sys/fs/cgroup')/prop['ControlGroup'].lstrip('/')
                readback={name:(cgroup/name).read_text().strip() for name in
                    ('cpu.max','memory.high','memory.max','memory.swap.max')}
                quota,period=readback['cpu.max'].split()
                if quota=='max' or int(quota)/int(period)>4.01:raise RuntimeError('CPU quota not enforced')
                if int(readback['memory.high'])!=12*GIB or int(readback['memory.max'])!=16*GIB or int(readback['memory.swap.max'])!=0:
                    raise RuntimeError('memory cgroup limits not enforced')
                if prop.get('Nice')!='10':raise RuntimeError('process nice not enforced')
                self.state.update(pid=self.pid,status='running',limits_readback=dict(properties=prop,cgroup=readback))
                atomic_json(self.directory/'guard.json',self.state)
                self.ready.touch()
                self.event('released','effective cgroup limits verified before child execution')
                bad_gpu=busy=0;requested_at=None;terminated_at=None
                while run.poll() is None:
                    sample=self.sampler(self) if self.sampler else self.sample()
                    self.samples.append(sample)
                    memory=sample.get('gpu_memory_MiB');util=sample.get('gpu_utilization_percent')
                    bad_gpu=bad_gpu+1 if memory is None else 0
                    busy=busy+1 if util is not None and util>95 else 0
                    age=time.time()-self.heartbeat.stat().st_mtime if self.heartbeat.exists() else time.monotonic()-self.started
                    action,reason=decision(sample,bad_gpu_samples=bad_gpu,busy_samples=busy,heartbeat_age=age,limits=self.limits)
                    now=time.monotonic()
                    if now-self.started>self.limits.wall_s:
                        action,reason='kill','watchdog wall budget'
                    if action=='kill':
                        self.event('kill',reason);self.kill('SIGKILL');break
                    if action=='stop' and requested_at is None:
                        requested_at=now;self.stop.write_text(reason+'\n');self.event('stop requested',reason)
                    if requested_at is not None and now-requested_at>self.limits.grace_s and terminated_at is None:
                        terminated_at=now;self.event('SIGTERM','cooperative stop deadline');self.kill('SIGTERM')
                    if terminated_at is not None and now-terminated_at>self.limits.terminate_s:
                        self.event('SIGKILL','process did not exit after SIGTERM');self.kill('SIGKILL');break
                    atomic_json(self.directory/'resource_samples.json',self.samples)
                    time.sleep(1)
                try:code=run.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.kill('SIGKILL');run.kill();code=run.wait();self.event('launcher killed','systemd-run did not return')
            self.state.update(status='completed',exit_code=code,wall_s=time.monotonic()-self.started,
                              final_unit=self.properties(),sample_count=len(self.samples))
        except BaseException as error:
            self.state.update(status='error',error=dict(type=type(error).__name__,message=str(error)),wall_s=time.monotonic()-self.started)
            self.kill('SIGKILL')
            if run is not None:
                try:run.wait(timeout=5)
                except subprocess.TimeoutExpired:run.kill();run.wait()
            raise
        finally:
            atomic_json(self.directory/'guard.json',self.state)
            atomic_json(self.directory/'resource_samples.json',self.samples)
            # Only remove this uniquely owned job; no desktop or unrelated processes.
            self.command('stop',self.unit,'--no-block',check=False)
            self.command('reset-failed',self.unit,check=False)
        return self.state
