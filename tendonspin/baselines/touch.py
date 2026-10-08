"""Reference feature processing; no raw taxel-array simulator is required.

Sources:
- AnyRotate, arXiv:2405.07391v3, Appendix F, equations (16)-(19).
  Official project: https://maxyang27896.github.io/anyrotate/
  Paper-based implementation; official training code has not been verified.
- Sharpa reference commit 5accf024d376685eaa17da7aa4614498217eab4d,
  rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env.py, compute_observations().

Input: calibrated fingertip force vectors and contact-pose features from virtual
or real sensing. Output: binary contact, clipped/scaled pose, force magnitude.
Paper defaults are not measured Boya calibration. Missing poses remain masked.
Not used by teacher-v2; this is an observation adapter, not a trained student.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TouchConfig:
    fingers: int = 4
    force_threshold_N: float = .25
    alpha: float = .5
    force_max_N: float = 5.
    force_scale: float = .6
    pose_limit_rad: float = .53
    pose_scale: float = .6


class DenseTouch:
    def __init__(self, config=TouchConfig()):
        self.c = config
        if config.fingers <= 0 or not 0 < config.alpha <= 1:
            raise ValueError('Invalid finger count or smoothing factor')
        if min(config.force_threshold_N, config.force_max_N, config.force_scale,
               config.pose_limit_rad, config.pose_scale) <= 0:
            raise ValueError('Positive calibrated ranges required')
        self.reset()

    def reset(self):
        self.force = np.zeros(self.c.fingers)

    def step(self, force_N, contact_pose_rad=None, pose_available=None):
        force_N = np.asarray(force_N, dtype=float)
        if force_N.shape != (self.c.fingers, 3) or not np.isfinite(force_N).all():
            raise ValueError('Finite calibrated fingertip force vectors required')
        magnitude = np.linalg.norm(force_N, axis=1)
        contact = magnitude > self.c.force_threshold_N
        self.force = self.c.alpha * magnitude + (1 - self.c.alpha) * self.force
        pose = np.zeros((self.c.fingers, 2))
        available = np.zeros(self.c.fingers, dtype=bool)
        if contact_pose_rad is not None:
            pose = np.asarray(contact_pose_rad, dtype=float)
            if pose.shape != (self.c.fingers, 2) or not np.isfinite(pose).all():
                raise ValueError('Finite calibrated contact-pose angles required')
            available = np.ones(self.c.fingers, dtype=bool) if pose_available is None else np.asarray(pose_available, dtype=bool)
            if available.shape != (self.c.fingers,):
                raise ValueError('Pose availability shape mismatch')
        elif pose_available is not None and np.asarray(pose_available).any():
            raise ValueError('Cannot declare a missing contact pose available')
        sensed_force = self.c.force_scale * np.clip(self.force, 0, self.c.force_max_N) * contact
        sensed_pose = self.c.pose_scale * np.clip(pose, -self.c.pose_limit_rad, self.c.pose_limit_rad)
        sensed_pose = sensed_pose * (contact & available)[:, None]
        return dict(features=np.r_[contact.astype(float), sensed_pose.reshape(-1), sensed_force].astype(np.float32),
                    pose_available=available & contact, paper_defaults=self.c == TouchConfig(),
                    object_truth_used=False, raw_array_simulated=False)
