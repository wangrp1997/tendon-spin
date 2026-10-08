# Project/source: New TendonSpin measured-input adapter; references Hora arXiv:2210.04887 and AnyRotate arXiv:2405.07391 deployment boundaries. No upstream robot driver code copied.
"""Offline deployment interface for an observation-limited student policy.

No ROS or hardware publish occurs here. Teacher checkpoints are rejected. Motor
and tactile measurements, calibration and command timing are explicit inputs.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from tendonspin.interfaces import ACTION_NAMES, STUDENT_SCHEMA, STUDENT_FRAME_DIM, STUDENT_HISTORY
from tendonspin.rl.policy import Policy


@dataclass(frozen=True)
class SensorFrame:
    timestamp: float
    motor_position_rad: np.ndarray
    commanded_position_rad: np.ndarray
    pad_force_N: np.ndarray
    previous_action: np.ndarray

    def features(self):
        for name,shape in [('motor_position_rad',(13,)),('commanded_position_rad',(13,)),
                           ('pad_force_N',(4,3)),('previous_action',(13,))]:
            value=np.asarray(getattr(self,name))
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('Invalid sensor field: '+name)
        if not math.isfinite(self.timestamp):raise ValueError('Invalid timestamp')
        return np.r_[self.motor_position_rad,self.commanded_position_rad,
                     np.asarray(self.pad_force_N).reshape(-1)/2.,self.previous_action].astype(np.float32)


class StudentRuntime:
    def __init__(self, checkpoint):
        bundle=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if bundle.get('role')!='student' or bundle.get('observation_schema')!=STUDENT_SCHEMA:
            raise ValueError('Deployment requires a student checkpoint with measured-input schema')
        if tuple(bundle['actuator_names'])!=ACTION_NAMES:raise ValueError('Actuator order mismatch')
        self.low=np.asarray(bundle['command_low_rad'],dtype=float)
        self.high=np.asarray(bundle['command_high_rad'],dtype=float)
        self.rate=np.asarray(bundle['command_rate_rad_s'],dtype=float)
        if any(a.shape!=(13,) for a in (self.low,self.high,self.rate)):
            raise ValueError('13 calibrated command bounds/rates required')
        if not all(np.isfinite(a).all() for a in (self.low,self.high,self.rate)) or np.any(self.low>=self.high) or np.any(self.rate<=0):
            raise ValueError('Invalid command calibration')
        self.calibration_id=bundle['calibration_id']
        self.dt=float(bundle['control_dt'])
        self.max_age=float(bundle['maximum_sensor_age_s'])
        if not self.calibration_id or self.dt<=0 or self.max_age<=0:
            raise ValueError('Timing and calibration identity required')
        self.policy=Policy(STUDENT_FRAME_DIM*STUDENT_HISTORY,13)
        self.policy.load_state_dict(bundle['policy'])
        self.policy.eval()
        self.history=None
        self.last_timestamp=None

    def predict(self, frame: SensorFrame, now):
        if not math.isfinite(now) or not 0<=now-frame.timestamp<=self.max_age:
            raise ValueError('Stale or future sensor frame')
        if self.last_timestamp is not None and frame.timestamp<=self.last_timestamp:
            raise ValueError('Sensor time must advance')
        features=frame.features()
        if self.history is None:
            self.history=np.repeat(features[None],STUDENT_HISTORY,axis=0)
        else:
            self.history=np.vstack((self.history[1:],features))
        self.last_timestamp=frame.timestamp
        with torch.no_grad():
            action=self.policy.deterministic(torch.from_numpy(self.history.reshape(1,-1)))[0].numpy()
        command=np.clip(np.asarray(frame.commanded_position_rad)+self.dt*self.rate*action,self.low,self.high)
        return dict(actuator_names=ACTION_NAMES,command_rad=command,action=action,
                    sensor_timestamp=frame.timestamp,calibration_id=self.calibration_id,published_to_hardware=False)
