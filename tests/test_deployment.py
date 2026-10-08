# Project/source: New TendonSpin offline deployment contract tests; no original paper policy or physical driver execution.
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from tendonspin.deploy.runtime import SensorFrame, StudentRuntime
from tendonspin.interfaces import ACTION_NAMES, ACTION_DIM, FINGERS, STUDENT_SCHEMA, STUDENT_FRAME_DIM, STUDENT_HISTORY
from tendonspin.rl.policy import Policy


class DeploymentBoundaryTests(unittest.TestCase):
    def test_teacher_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'teacher.pt'
            torch.save(dict(role='teacher'),path)
            with self.assertRaises(ValueError):StudentRuntime(path)

    def test_measured_student_commands_and_sensor_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'student.pt'
            policy=Policy(STUDENT_FRAME_DIM*STUDENT_HISTORY,ACTION_DIM)
            torch.save(dict(role='student',observation_schema=STUDENT_SCHEMA,actuator_names=ACTION_NAMES,
                command_low_rad=[-1.]*ACTION_DIM,command_high_rad=[1.]*ACTION_DIM,command_rate_rad_s=[.35]*ACTION_DIM,
                calibration_id='test-only',control_dt=.05,maximum_sensor_age_s=.1,policy=policy.state_dict()),path)
            runtime=StudentRuntime(path)
            frame=SensorFrame(1.,np.zeros(ACTION_DIM),np.zeros(ACTION_DIM),np.zeros((len(FINGERS),3)),np.zeros(ACTION_DIM))
            result=runtime.predict(frame,1.01)
            self.assertFalse(result['published_to_hardware'])
            self.assertLessEqual(np.max(abs(result['command_rad'])),.0175)
            with self.assertRaises(ValueError):runtime.predict(frame,1.02)
            with self.assertRaises(ValueError):runtime.predict(frame,1.2)


if __name__=='__main__':unittest.main()
