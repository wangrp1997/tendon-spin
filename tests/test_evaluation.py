"""TendonSpin evaluation integrity checks; source metric definitions are in rotation.py.

Analytic traces test early failure, backward cancellation and goal censorship,
without requiring or integrating a physics engine.
"""
import unittest

import numpy as np

from tendonspin.evaluation.rotation import score_prefix


class EvaluationTests(unittest.TestCase):
    def test_failure_frame_cannot_score_or_complete_a_goal(self):
        out = score_prefix([10., 200., 600.], 1., 2, window_s=30.)
        self.assertEqual(out['net_deg'], 200.)
        self.assertEqual(out['valid_seconds'], 2.)
        self.assertFalse(out['complete_window_observed'])
        self.assertIsNone(out['full_window_final_tail_net_deg'])
        self.assertEqual(out['goals'][0]['first_sampled_time_s'], 2.)
        self.assertFalse(out['goals'][1]['reached'])
        self.assertIsNone(out['goals'][1]['first_sampled_time_s'])

    def test_backward_motion_reduces_net_but_preserves_peak(self):
        out = score_prefix([40., 70., 10., 20.], 1., 4, window_s=4., tail_s=2.)
        self.assertEqual(out['net_deg'], 20.)
        self.assertEqual(out['peak_deg'], 70.)
        self.assertEqual(out['forward_deg'], 80.)
        self.assertEqual(out['backward_deg'], 60.)
        self.assertEqual(out['full_window_final_tail_net_deg'], -50.)

    def test_multiple_turns_are_not_wrapped_and_targets_are_first_crossings(self):
        out = score_prefix([90., 180., 270., 360., 450., 350.], 1., 6,
                           window_s=6., tail_s=2.)
        self.assertEqual(out['net_deg'], 350.)
        self.assertEqual(out['peak_deg'], 450.)
        self.assertEqual(out['goals'][1]['first_sampled_time_s'], 4.)
        self.assertFalse(out['goals'][2]['reached'])
        shorter = score_prefix([90., 180., 270., 360., 450., 350.], 1., 6, window_s=3.)
        self.assertFalse(shorter['goals'][1]['reached'])

    def test_nonfinite_valid_trace_is_rejected(self):
        with self.assertRaises(ValueError):
            score_prefix([90., np.nan], 1., 2, window_s=2.)
