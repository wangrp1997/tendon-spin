"""Sensor-only inference boundary; SDK acquisition/actuation is supplied by caller.

No teacher loading, object truth, online weight updates or calibration required.
This class returns actions, and does not send hardware commands itself.
"""
import torch

from .model import load_student
from .observations import SensorConfig, SensorHistory


class TouchStudentRuntime:
    def __init__(self, checkpoint, device='cpu'):
        self.model, self.metadata = load_student(checkpoint, device)
        contract = self.metadata['observation_contract']
        self.modality = self.metadata['modality']
        self.device = device
        self.history = SensorHistory(contract['limits'], device=device,
                                     config=SensorConfig(**contract['preprocessing']))
        self.last_timestamp_s = None

    def reset(self):
        self.history.reset()
        self.last_timestamp_s = None

    @torch.no_grad()
    def step(self, *, joint_pos_rad, issued_targets_rad, timestamp_s,
             force_magnitude_n=None, tactile_valid=None, tactile_age_s=None):
        import math
        if (not math.isfinite(timestamp_s) or (self.last_timestamp_s is not None
                and timestamp_s <= self.last_timestamp_s)):
            raise ValueError('Finite increasing acquisition timestamps required')
        # Avoid feeding discontinuous history after an SDK pause/reconnect.
        if self.last_timestamp_s is not None and timestamp_s-self.last_timestamp_s > .15:
            self.history.reset()
        tactile = (force_magnitude_n, tactile_valid, tactile_age_s)
        if all(value is None for value in tactile):
            if self.modality != 'proprio':
                raise ValueError('Touch modality requires force, validity and age inputs')
            force_magnitude_n = [0.]*5
            tactile_valid = [False]*5
            tactile_age_s = [0.]*5
        elif any(value is None for value in tactile):
            raise ValueError('All tactile inputs must be supplied together')
        def batch(value, dtype=torch.float32):
            return torch.as_tensor(value, dtype=dtype, device=self.device)[None]
        history = self.history.push(batch(joint_pos_rad), batch(issued_targets_rad),
            batch(force_magnitude_n), batch(tactile_valid, torch.bool), batch(tactile_age_s))
        output = self.model(history)['action'][0]
        if not torch.isfinite(output).all():
            raise RuntimeError('Nonfinite student action')
        self.last_timestamp_s = timestamp_s
        return output.cpu().numpy()
