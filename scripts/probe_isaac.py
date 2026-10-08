# Project/source: New environment probe using NVIDIA Isaac Sim SimulationApp API from the existing installed6.1 runtime; no environment rebuild.
"""Read-only environment reuse and optional minimal headless engine launch."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time

parser=argparse.ArgumentParser()
parser.add_argument('--launch',action='store_true')
parser.add_argument('--out',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
record={}
for name in ('isaacsim','isaaclab','torch'):
    spec=importlib.util.find_spec(name)
    record[name]=dict(version=importlib.metadata.version(name),path=spec.origin)
record['reinstalled']=False
record['boya_isaac_model_validated']=False
record['eula_acceptance_supplied']=os.environ.get('OMNI_KIT_ACCEPT_EULA') == 'YES'
record['license_confirmation_scope']='current process only' if record['eula_acceptance_supplied'] else 'not supplied'
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
destination=args.out or root/'docs/data/isaac_environment.json'
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
