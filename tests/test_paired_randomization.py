"""Check batching-invariant paired stimuli and statistical interpretation."""
import json
from pathlib import Path
import unittest

import numpy as np

from tendonspin.evaluation.paired_randomization import generate_cases, summarize, paired_comparison
from tendonspin.evaluation.rotation import score_prefix


class PairedRandomizationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path('assets/config/boya_hora_randomized_c.json').read_text())

    def test_case_streams_do_not_depend_on_other_cases_or_later_horizon(self):
        rows, a = generate_cases(self.cfg, [0, 1], controls=8)
        short_rows, b = generate_cases(self.cfg, [0], controls=3)
        self.assertEqual(rows[:3], short_rows)
        np.testing.assert_array_equal(a['force_N'][:3, :3], b['force_N'])
        np.testing.assert_array_equal(a['noise_rad'][:3, :3], b['noise_rad'])
        self.assertFalse(np.array_equal(a['noise_rad'][:, 0], a['noise_rad'][:, 1]))

    def test_parameters_and_external_force_keep_original_bounds_and_decay(self):
        rows, arrays = generate_cases(self.cfg, [0, 8], controls=100)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertLessEqual(abs(row['scale'] - self.cfg['scale_centers'][row['scale_id']]), .025001)
            self.assertTrue(.01 <= row['mass_kg'] <= .25)
        self.assertLessEqual(float(np.abs(arrays['noise_rad']).max()), .020001)
        force = arrays['force_N']
        factor = self.cfg['force_decay'] ** (self.cfg['force_decay_reference_dt_s'] / self.cfg['force_decay_interval_s'])
        unchanged = np.isclose(force[1:], force[:-1] * factor, atol=1e-7).all(-1)
        self.assertGreater(unchanged.mean(), .5)
        self.assertGreater(np.linalg.norm(force, axis=-1).max(), 0)

    def test_failure_prefix_is_not_extrapolated_or_given_a_tail(self):
        score = score_prefix([1., 2., 3.], 1., 2, window_s=30., tail_s=10.)
        self.assertFalse(score['complete_window_observed'])
        self.assertEqual(score['net_deg'], 2.)
        self.assertEqual(score['valid_seconds'], 2.)
        self.assertIsNone(score['full_window_final_tail_net_deg'])

    def test_pairing_rejects_missing_and_infrastructure_incomplete_cases(self):
        row = dict(case_id='a', cache_index=0, scale_id=0, net_deg=1., stop_reason='time limit')
        with self.assertRaisesRegex(ValueError, 'cases differ'):
            paired_comparison([row], [])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            paired_comparison([row], [dict(row, stop_reason='wall budget')])

    def test_equal_sized_effects_have_exact_paired_interval(self):
        baseline = [dict(case_id=f'{i}_{j}', cache_index=i, scale_id=i % 2,
            net_deg=i * 10., stop_reason='time limit') for i in range(4) for j in range(3)]
        candidate = [dict(r, net_deg=r['net_deg'] + 17.) for r in baseline]
        result = paired_comparison(baseline, candidate, draws=100)
        self.assertEqual(result['cache_weighted_95ci'], [17., 17.])
        self.assertEqual(result['scale_macro_95ci'], [17., 17.])

    def test_scale_macro_does_not_hide_sparse_group(self):
        def row(group, net):
            return dict(scale_id=group, net_deg=net, valid_s=30., backward_deg=0., last10s_net_deg=net/3,
                max_drift_mm=0., max_tilt_deg=0., reward_sum=0., complete_window=True, stop_reason='time limit')
        summary = summarize([row(0, 0.)] + [row(1, 100.) for _ in range(9)])
        self.assertEqual(summary['cache_weighted']['net_deg']['mean'], 90.)
        self.assertEqual(summary['scale_macro_net_deg'], 50.)


if __name__ == '__main__':
    unittest.main()
