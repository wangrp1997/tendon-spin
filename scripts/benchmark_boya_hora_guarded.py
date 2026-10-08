# Reference: TendonSpin benchmark_boya_hora.py (bd5f26d); original Hora PPO.
# Adds independently supervised per-job cgroup limits; task/physics unchanged.
"""Run one declared throughput trial with desktop resource headroom."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tendonspin.rl.resource_guard import ResourceGuard,atomic_json

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--num-envs',type=int,default=2048)
parser.add_argument('--updates',type=int,default=2)
parser.add_argument('--protocol',default='docs/experiments/2026-10-08-boya-env2048-guarded/PROTOCOL.md')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
out=args.out.resolve()
out.mkdir(parents=True,exist_ok=False)
guard=ResourceGuard(root,out/'guard')
source=Path(__file__).read_bytes()
(out/'benchmark_boya_hora_guarded.py').write_bytes(source)
cmd=[sys.executable,str(root/'scripts/train_boya_hora.py'),
     '--out',str(out/'training'),'--cache',str(root/'outputs/boya_settled_cache_v2/grasp_cache.npz'),
     '--num-envs',str(args.num_envs),'--updates',str(args.updates),'--wall-s','240',
     '--save-every','1','--minibatch-size','512','--max-gpu-memory-mib','10240','--seed','43',
     '--controller',f'hora_boya_env{args.num_envs}_guarded_v1','--protocol',args.protocol,
     '--stop-file',str(guard.stop),'--heartbeat',str(guard.heartbeat)]
state=guard.run(cmd,dict(OMNI_KIT_ACCEPT_EULA='YES',PYTHONPATH=str(root),
                         OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
state['benchmark_source_sha256']=hashlib.sha256(source).hexdigest()
atomic_json(out/'resource_summary.json',state)
print(json.dumps(dict(status=state['status'],exit_code=state['exit_code'],
                     wall_s=state['wall_s'],events=state['events'])),flush=True)
sys.exit(0 if state['exit_code']==0 else 1)
