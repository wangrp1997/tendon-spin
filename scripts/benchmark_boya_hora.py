# Reference: TendonSpin512-env resource monitor from the942d1e9 experiment.
# Reuse its one-second sampling and bounded child supervision; parameterized CLI.
"""Measure one declared Isaac/Hora environment-count setting without competing jobs."""
from pathlib import Path
import subprocess,os,json,time,signal,argparse,hashlib,sys
parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--num-envs',type=int,default=1024)
parser.add_argument('--updates',type=int,default=4)
parser.add_argument('--protocol',default='docs/experiments/2026-10-08-boya-env1024-throughput/PROTOCOL.md')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
out=args.out.resolve()
os.nice(10)
env=os.environ.copy();env.update(OMNI_KIT_ACCEPT_EULA='YES',PYTHONPATH=str(root),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
cmd=[sys.executable,str(root/'scripts/train_boya_hora.py'),
     '--out',str(out),'--cache',str(root/'outputs/boya_settled_cache_v2/grasp_cache.npz'),
     '--num-envs',str(args.num_envs),'--updates',str(args.updates),'--wall-s','240','--save-every','4','--minibatch-size','512',
     '--max-gpu-memory-mib','10240','--seed','43','--controller',f'hora_boya_env{args.num_envs}_throughput_v1',
     '--protocol',args.protocol]
start=time.monotonic();samples=[];stop=None
with out.with_name(out.name+'_launch.log').open('x') as log:
    child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    while child.poll() is None:
        elapsed=time.monotonic()-start
        gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        rss=None
        try:
            status=Path(f'/proc/{child.pid}/status').read_text()
            rss=int(next(line for line in status.splitlines() if line.startswith('VmRSS:')).split()[1])/1024
        except (FileNotFoundError,StopIteration):pass
        if gpu.returncode==0:
            memory,util=[int(x.strip()) for x in gpu.stdout.strip().splitlines()[0].split(',')]
            samples.append(dict(elapsed_s=elapsed,gpu_memory_MiB=memory,gpu_utilization_percent=util,child_rss_MiB=rss))
            if memory>10240 and stop is None:
                stop='sampled device memory >10GiB';child.send_signal(signal.SIGINT)
        if out.exists():
            (out/'resource_samples.json').write_text(json.dumps(dict(pid=child.pid,stop_requested=stop,samples=samples),indent=2)+'\n')
        if elapsed>300:
            stop='hard300s wall budget';child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            break
        time.sleep(1)
    code=child.wait()
monitor=dict(monitor_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=cmd,exit_code=code,wall_s=time.monotonic()-start,stop_requested=stop,
    max_observed_gpu_memory_MiB=max(x['gpu_memory_MiB'] for x in samples),
    max_observed_gpu_utilization_percent=max(x['gpu_utilization_percent'] for x in samples),
    max_observed_child_rss_MiB=max(x['child_rss_MiB'] or 0 for x in samples),sample_count=len(samples))
(out/'resource_summary.json').write_text(json.dumps(monitor,indent=2)+'\n')
print(json.dumps(monitor),flush=True)
