# Adapted botyard-inhand/research/learning/config.py
# Source revision: 204d197a9606fb7266e884f3b2e6110195be01cb.
# TendonSpin changes and evaluation identity are declared in docs/experiment_state.md.
# This local PPO is not a reproduced Hora/AnyRotate/Sharpa policy.
"""One declared nominal PPO baseline; no physics or contact-sequence changes."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'tendonspin_teacher_ppo_v2'
REPORT = ROOT / 'docs/experiments/2026-10-08-teacher-v2'
RUNTIME = ROOT / 'runtime/mujoco-patched'
SIM_PYTHON = os.environ.get('TENDONSPIN_SIM_PYTHON', '/home/rw/miniconda3/envs/free/bin/python')


@dataclass(frozen=True)
class Config:
    control_dt: float = .05
    command_speed_rad_s: float = .35
    train_episode_s: float = 30.
    evaluation_s: float = 120.
    rotation_reward_per_rad: float = 5.
    reward_spin_cap_rad_s: float = .2
    failure_penalty: float = 5.
    position_cost_per_s: float = .2
    tilt_cost_per_s: float = .2
    tracking_cost_per_s: float = .005
    action_cost_per_s: float = .01
    environments: int = 8
    rollout_steps: int = 256
    updates: int = 128
    epochs: int = 4
    minibatch: int = 256
    learning_rate: float = 3e-4
    gamma: float = .995
    gae_lambda: float = .95
    clip: float = .2
    entropy_coefficient: float = .002
    value_coefficient: float = .5
    max_grad_norm: float = .5
    target_kl: float = .03
    seed: int = 44
    train_wall_s: float = 1200.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze(out, config):
    paths = sorted((ROOT / 'tendonspin').rglob('*.py')) + [ROOT / 'tendonspin/physics/replay.c', REPORT / 'PROTOCOL.md']
    records = []
    for path in paths:
        relative = path.relative_to(ROOT)
        target = Path(out) / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        records.append(dict(path=str(relative), sha256=digest(path)))
    manifest = dict(controller=REVISION, config=asdict(config), files=records,
                    learning=True, privileged=True, physics='original patched MuJoCo 3.13 native',
                    components='single frozen neural policy and declared position command adapter',
                    fixed_grasp=True, fixed_contact_sequence=False,
                    domain_randomization=False, imitation=False, hardware_evaluated=False)
    write(Path(out) / 'protocol.json', manifest)
    return manifest
