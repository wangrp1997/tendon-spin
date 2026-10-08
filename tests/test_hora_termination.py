"""Reference-oracle and captured-failure regressions; no simulated success claims."""
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tendonspin.baselines.hora_training import load_termination
from tendonspin.rl.termination import TaskTermination, make_termination_spec, legacy_failure_codes

ROOT = Path(__file__).resolve().parents[1]
NOMINAL_Z = .10606109973956034


def measurement(heights):
    heights = torch.as_tensor(heights, dtype=torch.float64)
    n = len(heights)
    state = torch.zeros(n, 13, dtype=torch.float64)
    state[:, 2] = heights
    return dict(object_state=state, finite=torch.ones(n, dtype=torch.bool),
                drift_mm=torch.zeros(n), tilt_deg=torch.zeros(n),
                max_normal=torch.zeros(n), max_speed=torch.zeros(n),
                coupling_error=torch.zeros(n))


class HoraTerminationTests(unittest.TestCase):
    def setUp(self):
        self.spec = make_termination_spec(ROOT, 'hora_height', NOMINAL_Z)
        self.rule = TaskTermination(self.spec)

    def test_matches_upstream_height_and_time_after_scene_translation(self):
        z = torch.tensor([.644, .645, .646, .66, .65], dtype=torch.float64)
        steps = torch.tensor([1, 1, 1, 400, 399])
        reference = SimpleNamespace(reset_z_threshold=.645, progress_buf=steps, max_episode_length=400)
        xyz = torch.zeros(len(z), 3, dtype=torch.float64);xyz[:, 2] = z
        expected = load_termination()(reference, xyz)
        m = measurement(z + NOMINAL_Z - .65)
        # Mimic independent scene translations, including vertical origins.
        origins = torch.tensor([[0., 0., 0.], [1., 2., 0.], [2., -3., 0.],
                                [-1., -2., 10.], [3., 4., -10.]], dtype=torch.float64)
        m['object_state'][:, :3] += origins
        codes = self.rule.failure_codes(m, origins, control_boundary=True)
        actual = (codes != 0) | self.rule.timeouts(steps, codes)
        torch.testing.assert_close(actual, expected)
        self.assertAlmostEqual(self.spec.reset_height_m, .10106109973956034)

    def test_captured_10m_tilt_stop_is_diagnostic_in_new_task(self):
        m = measurement([NOMINAL_Z])
        m['tilt_deg'][:] = 15.0056505203
        m['drift_mm'][:] = 2.5597150326
        m['max_normal'][:] = 3.6838226318
        self.assertEqual(int(legacy_failure_codes(m)[0]), 3)
        self.assertEqual(int(self.rule.failure_codes(m, torch.zeros(1, 3), control_boundary=True)[0]), 0)
        # Horizontal movement, tilt and force cannot independently end new task.
        m['object_state'][0, 0] = 10.
        m['drift_mm'][:] = 100.;m['tilt_deg'][:] = 70.;m['max_normal'][:] = 20.
        self.assertEqual(int(self.rule.failure_codes(m, torch.zeros(1, 3), control_boundary=True)[0]), 0)
        self.assertEqual(int(legacy_failure_codes(m)[0]), 2)

    def test_height_is_checked_at_reference_control_boundary(self):
        m = measurement([self.spec.reset_height_m - .001])
        self.assertEqual(int(self.rule.failure_codes(m, torch.zeros(1, 3), control_boundary=False)[0]), 0)
        self.assertEqual(int(self.rule.failure_codes(m, torch.zeros(1, 3), control_boundary=True)[0]), 8)
        self.assertFalse(bool(self.rule.timeouts(torch.tensor([400]), torch.tensor([8]))[0]))
        self.assertTrue(bool(self.rule.timeouts(torch.tensor([400]), torch.tensor([0]))[0]))

    def test_numerical_faults_remain_distinct_from_task_failure(self):
        m = measurement([NOMINAL_Z] * 3)
        m['finite'][0] = False;m['max_speed'][1] = 101.;m['coupling_error'][2] = .051
        expected = torch.tensor([1, 4, 6])
        torch.testing.assert_close(self.rule.failure_codes(m, torch.zeros(3, 3), control_boundary=False), expected)

    def test_legacy_profile_preserves_original_gates(self):
        rule = TaskTermination(make_termination_spec(ROOT, 'legacy_strict', NOMINAL_Z))
        m = measurement([NOMINAL_Z] * 3)
        m['drift_mm'][0] = 5.01;m['tilt_deg'][1] = 15.01;m['max_normal'][2] = 12.01
        torch.testing.assert_close(rule.failure_codes(m, torch.zeros(3, 3), control_boundary=False),
                                   torch.tensor([2, 3, 5]))


if __name__ == '__main__':
    unittest.main()
