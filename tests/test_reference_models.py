"""Reference-equivalence and frozen-teacher checks; no task-success claims."""
import unittest
import torch
from tendonspin.baselines.reference_models import reference_module, build_model, student_from_teacher, adaptation_loss
from tendonspin.baselines.touch import DenseTouch
import numpy as np
from tendonspin.interfaces import ACTION_DIM, PROPRIO_FRAME_DIM, FINGERS


class ReferenceTests(unittest.TestCase):
    def test_parameterized_tcn_matches_hora_at_original_width(self):
        torch.manual_seed(44)
        original = reference_module('hora_models').ProprioAdaptTConv()
        variable = reference_module('sharpa_models').ProprioAdaptTConv(32)
        variable.load_state_dict(original.state_dict())
        history = torch.randn(2, 30, 32)
        torch.testing.assert_close(original(history), variable(history), atol=0, rtol=0)

    def test_boya_adaptation_updates_history_only(self):
        torch.manual_seed(44)
        teacher = build_model()
        student = student_from_teacher(teacher)
        data = dict(obs=torch.randn(2, 3 * PROPRIO_FRAME_DIM), priv_info=torch.randn(2, 9),
                    proprio_hist=torch.randn(2, 30, PROPRIO_FRAME_DIM))
        with torch.no_grad():
            self.assertEqual(tuple(student.act_inference({k:v for k,v in data.items() if k!='priv_info'}).shape), (2, ACTION_DIM))
        before = {k:v.clone() for k,v in student.state_dict().items()}
        optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=3e-4)
        loss = adaptation_loss(student, data)
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad();loss.backward();optimizer.step()
        changed = [k for k,v in student.state_dict().items() if not torch.equal(v,before[k])]
        self.assertTrue(changed)
        self.assertTrue(all(k.startswith('adapt_tconv.') for k in changed))

    def test_contact_masking_and_missing_pose_are_explicit(self):
        processor = DenseTouch()
        out = processor.step(np.zeros((len(FINGERS),3)))
        np.testing.assert_array_equal(out['features'],np.zeros(4 * len(FINGERS)))
        force = np.zeros((len(FINGERS),3));force[0,2] = 1.
        out = processor.step(force)
        self.assertEqual(out['features'][0],1.)
        self.assertFalse(out['pose_available'].any())
        self.assertAlmostEqual(float(out['features'][3 * len(FINGERS)]),.3,places=6)
        out = processor.step(np.zeros((len(FINGERS),3)))
        self.assertEqual(out['features'][3 * len(FINGERS)],0.)


if __name__=='__main__':unittest.main()
