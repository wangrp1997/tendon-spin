# Adapted from botyard-inhand/rotation/metrics.py
# Source revision: 204d197a9606fb7266e884f3b2e6110195be01cb.
# Changes: standalone TendonSpin package/asset paths; original behavior retained.
"""Measure actual cylinder spin without quaternion wrapping or tilt rewards."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-12:
        raise ValueError('Expected a finite, nonzero MuJoCo wxyz quaternion')
    return Rotation.from_quat(q[[1, 2, 3, 0]])


@dataclass
class SpinTracker:
    """Integrate physical rotation about the moving cylinder axis, in radians.

    Consecutive orientations must differ by less than pi. The runner updates at
    each physics step. Quaternion sign flips are representation changes only.
    """
    initial_quaternion: np.ndarray
    direction: int = -1

    def __post_init__(self):
        if self.direction not in (-1, 1):
            raise ValueError('direction must be -1 or +1')
        self.previous = rotation(self.initial_quaternion)
        self.reference_axis = self.previous.as_matrix()[:, 2]
        self.net = self.forward = self.backward = 0.0

    def update(self, quaternion):
        current = rotation(quaternion)
        axes = self.previous.as_matrix()[:, 2] + current.as_matrix()[:, 2]
        norm = np.linalg.norm(axes)
        if norm < 1e-8:
            raise ValueError('Axis changed by 180 degrees in one physics step')
        increment = self.direction * float(np.dot(
            (current * self.previous.inv()).as_rotvec(), axes / norm))
        self.net += increment
        self.forward += max(increment, 0.0)
        self.backward += max(-increment, 0.0)
        self.previous = current
        return increment

    @property
    def tilt(self):
        axis = self.previous.as_matrix()[:, 2]
        return float(np.arccos(np.clip(axis @ self.reference_axis, -1.0, 1.0)))

    @property
    def directionality(self):
        return self.net / max(self.forward + self.backward, 1e-12)

    def report(self):
        return dict(net_turns=self.net / (2*np.pi), net_degrees=float(np.degrees(self.net)),
                    forward_degrees=float(np.degrees(self.forward)),
                    backward_degrees=float(np.degrees(self.backward)),
                    directionality=self.directionality, tilt_degrees=float(np.degrees(self.tilt)))
