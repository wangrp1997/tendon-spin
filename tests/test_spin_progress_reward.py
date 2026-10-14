"""Analytic self-spin/tilt/reset cases; no Isaac app, optimizer or physics steps."""
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from tendonspin.baselines.hora_training import load_reward
from tendonspin.rl.spin_progress_reward import SpinProgressRewardSignal


def spin_controls(speeds, initial=None, signs=False):
    if initial is None: initial = Rotation.from_euler('xyz', [.7, -.2, 1.1])
    cfg = SpinProgressRewardSignal().configuration
    dt, steps = cfg['physics_dt_s'], cfg['steps_per_control']
    position = 0.
    values = []
    signal = SpinProgressRewardSignal()
    for speed in speeds:
        start = initial*Rotation.from_rotvec([0., 0., -position])
        signal.begin(torch.tensor(start.as_quat()[None], dtype=torch.float64))
        for i in range(1, steps+1):
            q = (initial*Rotation.from_rotvec([0., 0., -(position+speed*dt*i)])).as_quat()
            if signs and i % 2: q = -q
            signal.advance(torch.tensor(q[None], dtype=torch.float64))
        values.append((float(signal.angular_speed()[0]), float(signal.rotation_reward()[0])))
        position += speed*dt*steps
    return signal, np.asarray(values)


class SpinProgressRewardTests(unittest.TestCase):
    def test_true_spin_is_invariant_to_initial_tilt_and_quaternion_sign(self):
        speeds = [.2, -.2, 80., -80.]
        expected = np.column_stack((speeds, np.minimum(speeds, .5)))
        for initial in (Rotation.identity(), Rotation.from_euler('xyz', [2.1, -.5, .8])):
            _, values = spin_controls(speeds, initial, signs=True)
            np.testing.assert_allclose(values, expected, rtol=0, atol=1e-9)

    def test_stationary_pose_and_reset_jump_cannot_create_rotation(self):
        signal, values = spin_controls([0.])
        np.testing.assert_allclose(values, 0., rtol=0, atol=1e-12)
        q = torch.tensor(Rotation.from_euler('xyz', [1.8, -.9, 2.]).as_quat()[None])
        signal.begin(q)
        for i in range(signal.steps_per_control): signal.advance(q if i % 2 else -q)
        self.assertAlmostEqual(float(signal.rotation_reward()[0]), 0., places=12)

    def test_pure_tilt_does_not_earn_self_spin(self):
        initial = Rotation.from_euler('xyz', [.4, -.8, 1.])
        axis = initial.apply([1., 0., 0.])
        signal = SpinProgressRewardSignal()
        signal.begin(torch.tensor(initial.as_quat()[None]))
        for i in range(1, signal.steps_per_control+1):
            q = (Rotation.from_rotvec(axis*3.*.0005*i)*initial).as_quat()
            signal.advance(torch.tensor(q[None]))
        self.assertAlmostEqual(float(signal.rotation_reward()[0]), 0., places=10)

    def test_fast_reverse_cannot_be_hidden_by_symmetric_clipping(self):
        _, values = spin_controls([.4, .4, .4, .4, -2.])
        self.assertLess(values[:, 0].sum(), 0.)
        self.assertGreater(np.clip(values[:, 0], -.5, .5).sum(), 0.)
        self.assertLess(values[:, 1].sum(), 0.)
        self.assertLessEqual(values[:, 1].sum(), values[:, 0].sum()+1e-10)

    def test_original_penalties_and_only_workspace_cost16_are_preserved(self):
        signal = SpinProgressRewardSignal()
        q = torch.tensor([[0., 0., 0., 1.]]*6)
        signal.begin(q)
        for i in range(1, signal.steps_per_control+1):
            pose = Rotation.from_rotvec([0., 0., -.2*.0005*i]).as_quat()
            signal.advance(torch.tensor(np.tile(pose, (6, 1)), dtype=torch.float32))
        linear = torch.tensor([[.1, -.2, .3]]*6)
        pose, torque, work = torch.full((6,), .3), torch.full((6,), .4), torch.full((6,), .05)
        codes = torch.tensor([0, 7, 9, 10, 5, 6])
        output = signal.compute_reward(linear, pose, torque, work, codes)
        zero = torch.zeros(6, 3)
        original, _, _ = load_reward()(linear, -.3, zero, zero, 1., .5, -.5,
            pose, -.3, torque, -.1, work, -2.)
        cost = torch.tensor([0., 0., -16., -16., 0., 0.])
        torch.testing.assert_close(output['nonrotation_reward'], original, rtol=0, atol=0)
        torch.testing.assert_close(output['terminal_failure_cost'], cost, rtol=0, atol=0)
        torch.testing.assert_close(output['reward'], original+output['rotation']+cost, rtol=0, atol=0)

    def test_incomplete_control_is_rejected(self):
        signal = SpinProgressRewardSignal()
        signal.begin(torch.tensor([[0., 0., 0., 1.]]))
        signal.advance(torch.tensor([[0., 0., 0., 1.]]))
        with self.assertRaisesRegex(RuntimeError, 'complete control'): signal.angular_speed()


if __name__ == '__main__': unittest.main()
