"""Check packaged physics and observation ABI against the old executed episode."""
from dataclasses import replace
import json
from pathlib import Path
import mujoco
import numpy as np
from tendonspin.physics.native import scene, state, Runner
from tendonspin.rl.config import ROOT, Config, digest, write
from tendonspin.rl.environment import RotationEnv

source = ROOT / 'outputs/imported/ppo_v1/evaluations/update_0064/native/execution.npz'
s,d,meta=scene()
runner=Runner(s,d)
with np.load(source,allow_pickle=False) as old:
    np.testing.assert_array_equal(state(s.model,d),old['initial_state'])
    runner.original_quaternion=d.qpos[s.qa+3:s.qa+7].copy()
    z=runner.run(d,old['ctrl'],stop=True)
    runner.check_physical(d,z)
    errors={k:float(abs(z[k]-old[k]).max()) for k in ('ctrl','qpos','qvel','motor','metrics','flags','final_state')}
    assert not any(errors.values()),errors
with np.load(ROOT/'outputs/imported/ppo_v1/evaluations/update_0064/policy_trace.npz',allow_pickle=False) as trace:
    env=RotationEnv(Config())
    observation_error=0.
    for observation,action in zip(trace['observations'],trace['actions']):
        observation_error=max(observation_error,float(abs(env.observe()-observation).max()))
        env.step(action)
    assert env.done and observation_error==0,observation_error
result=dict(physical_replay_errors=errors,observation_error=observation_error,
    original_state=True,old_repository_imported_diagnostic=True,new_controller_success=False,
    source_record=dict(path=str(source.relative_to(ROOT)),sha256=digest(source)),scene=meta,
    code_sources=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in
                  (ROOT/'tendonspin/physics/native.py',ROOT/'tendonspin/physics/replay.c',ROOT/'tendonspin/rl/environment.py')])
write(ROOT/'docs/data/migration.json',result)
print(json.dumps(dict(physical_errors=errors,observation_error=observation_error)),flush=True)
