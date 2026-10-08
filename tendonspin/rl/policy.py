"""Squashed Gaussian policy and PPO credit assignment with explicit reset masks."""
import math

import torch
from torch import nn
from torch.distributions import Normal


class Policy(nn.Module):
    def __init__(self, observations=95, actions=13):
        super().__init__()
        layers = []
        width = observations
        for hidden in (256, 128, 128):
            layer = nn.Linear(width, hidden)
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.Tanh()))
            width = hidden
        self.body = nn.Sequential(*layers)
        self.mean = nn.Linear(width, actions)
        self.critic = nn.Linear(width, 1)
        nn.init.orthogonal_(self.mean.weight, .01)
        nn.init.orthogonal_(self.critic.weight, 1.)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.critic.bias)
        self.log_std = nn.Parameter(torch.full((actions,), -.75))

    def distribution(self, observation):
        hidden = self.body(observation)
        mean = self.mean(hidden)
        return Normal(mean, self.log_std.clamp(-3., 1.).exp()), self.critic(hidden).squeeze(-1)

    def act(self, observation, latent=None):
        dist, value = self.distribution(observation)
        if latent is None:
            latent = dist.sample()
        action = torch.tanh(latent)
        # Stable log derivative of tanh; latent is preserved in the PPO buffer.
        log_jacobian = 2 * (math.log(2.) - latent - nn.functional.softplus(-2 * latent))
        log_prob = (dist.log_prob(latent) - log_jacobian).sum(-1)
        # Entropy regularization uses the base Gaussian entropy as a surrogate,
        # not a claim to the exact entropy of the bounded action distribution.
        return action, latent, log_prob, dist.entropy().sum(-1), value

    def deterministic(self, observation):
        dist, _ = self.distribution(observation)
        return torch.tanh(dist.loc)


def advantages(rewards, values, next_values, terminated, done, gamma, lam):
    """Bootstrap time limits but never carry advantages across any reset."""
    result = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in range(len(rewards) - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        running = delta + gamma * lam * (~done[t]) * running
        result[t] = running
    return result, result + values
