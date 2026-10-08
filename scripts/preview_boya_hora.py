# References: existing evaluate_boya_hora.py; NVIDIA Isaac Lab 3
# IsaacRtxRendererCfg.enable_scene_partitioning and its documented process override.
# A /World spectator camera must see env_0; no policy or physics source changes.
"""Record one frozen Boya policy with single-environment RTX spectator visibility."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('--training',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--seconds',type=float,default=5.)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
training=args.training.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
record=json.loads((training/'result.json').read_text())
checkpoint=root/record['checkpoint']
command=[sys.executable,str(root/'scripts/evaluate_boya_hora.py'),
    '--out',str(out/'evaluation'),'--cache',str(root/'outputs/boya_settled_cache_v2/grasp_cache.npz'),
    '--checkpoint',str(checkpoint),'--source-record',str(training/'result.json'),
    '--seconds',str(args.seconds),'--wall-s','1500','--video']
environment=os.environ.copy()
environment.update(OMNI_KIT_ACCEPT_EULA='YES',PYTHONPATH=str(root),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
    ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION='0')
manifest=dict(command=command,training_record=str(training/'result.json'),
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    render_override={'ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION':'0'},
    reason='single-env /World spectator camera otherwise cannot see partitioned env_0 geometry',
    scope='this child process only; camera visibility, no physics/controller change',
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='running')
(out/'preview.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'preview_boya_hora.py').write_bytes(Path(__file__).read_bytes())
started=time.monotonic()
with (out/'evaluation.log').open('w') as log:
    child=subprocess.Popen(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
    code=child.wait()
manifest.update(status='completed' if code==0 else 'error',exit_code=code,wall_s=time.monotonic()-started)
(out/'preview.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest),flush=True)
sys.exit(code)
