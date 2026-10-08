"""Credit assignment must not turn training resets into continuous successes."""
import unittest

import torch

from tendonspin.rl.policy import Policy, advantages


class PPOCreditTests(unittest.TestCase):
    def test_termination_vs_time_limit(self):
        rewards = torch.tensor([[1., 1.]])
        values = torch.tensor([[2., 2.]])
        successors = torch.tensor([[100., 100.]])
        term = torch.tensor([[True, False]])
        done = torch.tensor([[True, True]])
        adv, ret = advantages(rewards, values, successors, term, done, .9, .95)
        torch.testing.assert_close(adv, torch.tensor([[-1., 89.]]))
        torch.testing.assert_close(ret, torch.tensor([[1., 91.]]))

    def test_no_credit_across_reset(self):
        reward = torch.tensor([[1.], [1000.]])
        zero = torch.zeros_like(reward)
        done = torch.tensor([[True], [False]])
        adv, _ = advantages(reward, zero, zero, torch.zeros_like(done), done, .9, .95)
        torch.testing.assert_close(adv[0], torch.tensor([1.]))

    def test_saturation_keeps_likelihood_finite(self):
        policy = Policy()
        observation = torch.zeros((2, 95))
        latent = torch.full((2, 13), 100.)
        action, _, lp, _, value = policy.act(observation, latent)
        self.assertTrue(torch.isfinite(lp).all())
        self.assertTrue(torch.isfinite(value).all())
        self.assertLessEqual(action.abs().max().item(), 1.)
        (-lp.mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None))


if __name__ == '__main__':
    unittest.main()
