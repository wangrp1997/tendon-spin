# Sources: TendonSpin approved physical-v2 protocol; Python subprocess timeout API.
"""Launch exactly one versioned Isaac diagnostic under its 300s process budget."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path(__file__).resolve().parents[1]
audit = root/'outputs/isaac_boya_physical_v2_audit'
phase = root/'docs/data/isaac_boya_physical_v2_phases.json'
parent = root/'docs/data/isaac_boya_physical_v2.json'
if audit.exists() or phase.exists() or parent.exists():
    raise SystemExit('Versioned evidence exists; do not overwrite or silently repeat an approved trial')
audit.mkdir(parents=True)


def ref(path):
    return dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


sources = []
for name in ('scripts/probe_boya_physics.py', 'tendonspin/physics/isaac_boya.py',
             'tendonspin/physics/coordinates.py', 'scripts/run_boya_physical_diagnostic.py'):
    source = root/name
    snapshot = audit/source.name
    snapshot.write_bytes(source.read_bytes())
    sources.append(dict(**ref(source), snapshot=ref(snapshot)))
protocol = root/'docs/experiments/2026-10-08-boya-isaac-physical-v2/PROTOCOL.md'
(audit/'PROTOCOL.md').write_bytes(protocol.read_bytes())
command = ['/home/rw/miniconda3/envs/env_isaaclab/bin/python',
           str(root/'scripts/probe_boya_physics.py'), '--out', str(phase)]
env = os.environ.copy()
env.update(OMNI_KIT_ACCEPT_EULA='YES', PYTHONPATH=str(root), PYTHONUNBUFFERED='1')
log = audit/'launch.log'
record = dict(kind='one approved physical-transfer repair verification',
    controller='boya_source_pd_hold_v2', wall_budget_s=300, source=sources,
    protocol=ref(protocol), command=command, physics_steps=0, training_actions=0,
    benchmark_validated=False, grasp_validated=False)
parent.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
started = time.monotonic()
with log.open('wb') as stream:
    process = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    stop = 'process completed'
    try:
        process.wait(timeout=max(0.,298-(time.monotonic()-started)))
    except subprocess.TimeoutExpired:
        stop = 'process wall budget; terminated'
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=max(0.,300-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1)
record.update(actual_wall_s=time.monotonic()-started, return_code=process.returncode,
              parent_stop_reason=stop, log=ref(log))
if phase.exists():
    record['phase_record'] = ref(phase)
    result = json.loads(phase.read_text())
    for name in ('physics_steps', 'training_actions', 'original_physical_state_verified',
                 'stop_reason', 'diagnostic_elapsed_s', 'complete'):
        if name in result:
            record[name] = result[name]
parent.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('source','command')}, indent=2))
