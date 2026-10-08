# Source/project: NVIDIA Isaac Lab AppLauncher + SimulationContext API,
# installed Isaac Lab 3.0.0rc1 (BSD-3-Clause). New TendonSpin diagnostic wrapper;
# experience and installed source hashes are recorded per execution.
"""One empty-scene official-headless launch; run under the protocol timeout."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
started = time.monotonic()
record = dict(kind='empty-scene environment diagnostic',
              eula_acceptance_supplied=os.environ.get('OMNI_KIT_ACCEPT_EULA') == 'YES',
              license_confirmation_scope='current process only',
              reinstalled=False, boya_isaac_model_validated=False,
              training_actions=0, physics_steps=0, headless_launch=False,
              complete=False, versions={name: importlib.metadata.version(name)
                                       for name in ('isaacsim', 'isaaclab', 'torch')})

def save(phase):
    record['phase'] = phase
    record['wall_s_at_phase'] = time.monotonic() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    print('TENDONSPIN_PROBE ' + json.dumps(record), flush=True)

save('before AppLauncher import')
from isaaclab.app import AppLauncher
import inspect
source = Path(inspect.getfile(AppLauncher))
experience = source.parents[4] / 'apps/isaaclab.python.headless.kit'
record['launcher_source'] = dict(path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())
record['experience'] = dict(path=str(experience), sha256=hashlib.sha256(experience.read_bytes()).hexdigest())
record['configuration'] = dict(headless=True, enable_cameras=False, device='cuda:0',
                               physics_backend='default PhysX', dt=.005, gravity=[0.,0.,-9.81])
save('before AppLauncher constructor')
launcher = AppLauncher(headless=True, enable_cameras=False, device='cuda:0', experience=str(experience))
app = launcher.app
record['headless_launch'] = True
save('AppLauncher returned')
try:
    import isaaclab.sim as sim_utils
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005, device='cuda:0', visualizer_cfgs=[]))
    save('before reset')
    sim.reset()
    save('reset returned')
    for _ in range(3):
        sim.step(render=False)
        record['physics_steps'] += 1
        save('physics step returned')
finally:
    save('before close')
    app.close()
record['complete'] = True
save('close returned')
