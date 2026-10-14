# References: HaozhiQi/hora v0.0.1 compute_hand_reward (MIT), arXiv:2210.04887,
# https://github.com/HaozhiQi/hora; TendonSpin821b8ab rotation_reward.py (MIT).
# Reuse original pose rotation-vector arithmetic, pure Hora penalty function and
# existing Boya terminal-cost16. Changes: moving object axis and positive-only cap.
# Standalone offline candidate; historical/pinned PPO entries remain unchanged.
"""Signed object self-spin reward, separately versioned from original Hora."""
import json
import math
from pathlib import Path

from tendonspin.baselines.hora_training import load_reward
from tendonspin.rl.rotation_reward import (
    POSE_DROP, RotationRewardSignal, world_rotation_increment_xyzw,
)

CONFIG = Path(__file__).resolve().parents[2] / 'assets/config/boya_spin_progress_reward.json'


def configuration(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    if cfg['schema'] != 'tendonspin-boya-spin-progress-reward-v1':
        raise ValueError('Unknown self-spin reward schema')
    if cfg['negative_speed_floor_rad_s'] is not None:
        raise ValueError('This candidate must retain the full reverse penalty')
    if not math.isfinite(cfg['positive_speed_cap_rad_s']) or cfg['positive_speed_cap_rad_s'] <= 0:
        raise ValueError('A finite positive reward cap is required')
    if cfg['physics_dt_s'] <= 0 or cfg['steps_per_control'] < 1:
        raise ValueError('Positive physics/control timing is required')
    if (cfg['terminal_failure_cost'], cfg['terminal_failure_codes'], cfg['ordinary_timeout_cost']) != (16., [9, 10], 0.):
        raise ValueError('Preserve the existing Boya terminal cost contract')
    return cfg


def normalized_quaternion(q):
    import torch
    q = q.to(torch.float64)
    norm = q.norm(dim=-1, keepdim=True)
    if not torch.isfinite(q).all() or (norm <= 1e-12).any():
        raise ValueError('Finite nonzero XYZW quaternions required')
    return q / norm


def object_axis_z(q):
    import torch
    x, y, z, w = q.unbind(-1)
    return torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), dim=-1)


class SpinProgressRewardSignal:
    """Accumulate one control, seeded after reset; no policy/engine integration."""
    def __init__(self, config_path=CONFIG):
        self.configuration = configuration(config_path)
        cfg = self.configuration
        self.steps_per_control = cfg['steps_per_control']
        self.control_dt = cfg['physics_dt_s'] * self.steps_per_control
        self.drop = RotationRewardSignal(POSE_DROP, cfg['physics_dt_s'], self.steps_per_control)
        self.hora_reward = load_reward()
        self.steps = 0
        self.previous = None
        self.total_spin = None

    def begin(self, orientation):
        import torch
        self.output_dtype = orientation.dtype
        self.previous = normalized_quaternion(orientation)
        self.total_spin = torch.zeros_like(self.previous[..., 0])
        self.steps = 0

    def advance(self, orientation):
        if self.previous is None or self.steps >= self.steps_per_control:
            raise RuntimeError('Begin one control before recording its poses')
        current = normalized_quaternion(orientation)
        if current.shape != self.previous.shape:
            raise ValueError('Quaternion batch shape changed inside a control')
        axis = object_axis_z(self.previous) + object_axis_z(current)
        norm = axis.norm(dim=-1, keepdim=True)
        if (norm <= 1e-12).any():
            raise ValueError('Adjacent antipodal object axes have ambiguous self-spin')
        increment = world_rotation_increment_xyzw(self.previous, current)
        self.total_spin -= (increment * (axis/norm)).sum(-1)
        self.previous = current
        self.steps += 1

    def angular_speed(self):
        if self.steps != self.steps_per_control:
            raise RuntimeError('Rotation reward requires one complete control')
        return (self.total_spin/self.control_dt).to(self.output_dtype)

    def rotation_reward(self):
        cfg = self.configuration
        return self.angular_speed().clamp(max=cfg['positive_speed_cap_rad_s']) * cfg['rotation_scale']

    def compute_reward(self, object_linvel, pose_penalty, torque_penalty, work_penalty, failure_codes):
        """Reuse Hora's remaining terms and the unchanged workspace terminal cost."""
        import torch
        scales = self.configuration['other_penalty_scales']
        zeros = torch.zeros_like(object_linvel)
        base, _, linear = self.hora_reward(object_linvel, scales['object_linear_velocity_l1'],
            zeros, zeros, 1., .5, -.5, pose_penalty, scales['joint_pose_squared'],
            torque_penalty, scales['motor_torque_squared'], work_penalty, scales['motor_power_squared'])
        rotation = self.rotation_reward()
        terminal = self.drop.terminal_cost(failure_codes, base)
        return dict(reward=base+rotation+terminal, rotation=rotation,
            nonrotation_reward=base, linear_penalty=linear, terminal_failure_cost=terminal)
