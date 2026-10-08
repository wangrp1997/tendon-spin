"""Read-only environment reuse and optional minimal headless engine launch."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import time

parser=argparse.ArgumentParser()
parser.add_argument('--launch',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
record={}
for name in ('isaacsim','isaaclab','torch'):
    spec=importlib.util.find_spec(name)
    record[name]=dict(version=importlib.metadata.version(name),path=spec.origin)
record['reinstalled']=False
record['boya_isaac_model_validated']=False
if args.launch:
    start=time.monotonic()
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'hide_ui':True,'renderer':'RaytracedLighting'})
    try:
        for _ in range(3):app.update()
        record['headless_launch']=True
    finally:
        app.close()
    record['launch_wall_s']=time.monotonic()-start
(root/'docs/data/isaac_environment.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
