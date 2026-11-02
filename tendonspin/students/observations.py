"""Shared simulation/deployment preprocessing, with no object-state inputs.

Force constants are declared engineering defaults, not measured calibration.
Raw pressure arrays and contact positions are intentionally absent from v1.
"""
from dataclasses import asdict, dataclass

import torch

from tendonspin.interfaces import ACTION_NAMES, FINGERS, PASSIVE_JOINT_NAMES

JOINT_NAMES = ACTION_NAMES + PASSIVE_JOINT_NAMES
HISTORY = 30
FRAME_DIM = 57
SCHEMA = 'tendonspin-c-touch-history30-v1'


@dataclass(frozen=True)
class SensorConfig:
    force_threshold_n: float = 0.05
    force_max_n: float = 20.0
    force_scale_n: float = 1.0
    force_alpha: float = 0.5
    max_tactile_age_s: float = 0.15

    def __post_init__(self):
        import math
        values = asdict(self)
        if not all(math.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError('Positive finite sensor settings required')
        if self.force_alpha > 1 or self.force_threshold_n >= self.force_max_n:
            raise ValueError('Invalid smoothing or contact threshold')


def joint_limits(contract):
    joints = {j['name']: j for j in contract['joints']}
    motors = {m['name']: m for m in contract['actuators']}
    # Match the teacher's active control limits; passive joints use CAD limits.
    return [motors[n]['control_range'] if n in ACTION_NAMES else joints[n]['joint_range']
            for n in JOINT_NAMES]


class SensorHistory:
    """Batchable real sensor history; J1 is always independently measured.

    Layout: 16 normalized active angles, 16 issued targets in radians,
    4 normalized passive J1s, 5 contacts, 5 log force magnitudes,
    5 tactile-valid flags, 5 ages in seconds, 1 real/padded-frame flag.
    """
    def __init__(self, limits, num_envs=1, device='cpu', config=SensorConfig()):
        self.config = config
        self.limits = torch.as_tensor(limits, dtype=torch.float32, device=device)
        if (self.limits.shape != (20, 2) or not torch.isfinite(self.limits).all()
                or (self.limits[:, 1] <= self.limits[:, 0]).any()):
            raise ValueError('Twenty ordered joint limits required')
        self.frames = torch.zeros(num_envs, HISTORY, FRAME_DIM, device=device)
        self.force = torch.zeros(num_envs, len(FINGERS), device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids=None):
        if ids is None:
            ids = torch.arange(len(self.frames), device=self.frames.device)
        self.force[ids] = 0
        self.initialized[ids] = False
        self.frames[ids] = 0

    @torch.no_grad()
    def push(self, joint_pos_rad, issued_targets_rad, force_magnitude_n,
             tactile_valid, tactile_age_s):
        n = len(self.frames)
        for value, shape in ((joint_pos_rad, (n, 20)), (issued_targets_rad, (n, 16)),
                             (force_magnitude_n, (n, 5)), (tactile_age_s, (n, 5))):
            if value.shape != shape or not torch.isfinite(value).all():
                raise ValueError('Finite, ordered sensor fields required')
        if tactile_valid.shape != (n, 5) or tactile_valid.dtype != torch.bool:
            raise ValueError('Explicit five-finger validity mask required')
        if (force_magnitude_n < 0).any() or (tactile_age_s < 0).any():
            raise ValueError('Negative force magnitude or sample age')
        c = self.config
        valid = tactile_valid & (tactile_age_s <= c.max_tactile_age_s)
        magnitude = force_magnitude_n.clamp(max=c.force_max_n)
        self.force = torch.where(valid, c.force_alpha * magnitude + (1-c.force_alpha)*self.force, 0.)
        contact = valid & (magnitude > c.force_threshold_n)
        scaled_force = torch.log1p(self.force / c.force_scale_n) * contact
        lower, upper = self.limits.unbind(-1)
        normalized = 2 * (joint_pos_rad-lower)/(upper-lower)-1
        frame = torch.cat((normalized[:, :16], issued_targets_rad, normalized[:, 16:],
                           contact.float(), scaled_force, valid.float(),
                           tactile_age_s.clamp(max=c.max_tactile_age_s),
                           torch.ones(n, 1, device=self.frames.device)), dim=-1)
        frame[:, :32] = frame[:, :32].clamp(-5, 5)
        fresh = ~self.initialized
        self.frames = torch.roll(self.frames, -1, dims=1)
        self.frames[:, -1] = frame
        # Repeat startup joint/target readings exactly as Hora does, but mark
        # padded samples explicitly. Never retain samples across episode resets.
        self.frames[fresh] = frame[fresh, None]
        self.frames[fresh, :-1, -1] = 0
        self.initialized[:] = True
        return self.frames


def sensor_contract(limits, config=SensorConfig()):
    return dict(schema=SCHEMA, joint_names=list(JOINT_NAMES), action_names=list(ACTION_NAMES),
                fingers=list(FINGERS), history_frames=HISTORY, frame_dim=FRAME_DIM,
                control_hz=20, limits=torch.as_tensor(limits).tolist(),
                preprocessing=asdict(config), force_units='newtons', angle_units='radians',
                force_features='contact and log1p(EMA magnitude / fixed scale), retaining amplitude',
                camera_required=False, raw_pressure_array_required=False,
                passive_j1_independently_measured=True)
