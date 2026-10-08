"""Bounded nominal privileged PPO; training resets are never demo successes."""
import argparse
from collections import Counter, deque
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .config import Config, ROOT, REVISION, digest, freeze, write
from .evaluate import evaluate
from .policy import Policy, advantages
from .transport import Worker


def checkpoint(policy, optimizer, config, out, updates, action_steps):
    path = Path(out) / 'checkpoints' / f'update_{updates:04d}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(controller=REVISION, config=asdict(config), updates=updates,
                    action_steps=action_steps, policy={k: v.detach().cpu() for k, v in policy.state_dict().items()},
                    optimizer=optimizer.state_dict(), torch_version=str(torch.__version__)), path)
    return path


def train(config, out, initialize=None):
    out = Path(out).resolve()
    assert out.is_relative_to(ROOT / 'outputs')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = freeze(out, config)
    config_path = out / 'config.json'
    write(config_path, asdict(config))
    workers, evaluator = [], None
    histories, evaluations, episodes = [], [], []
    recent = deque(maxlen=100)
    started = time.monotonic()
    action_steps = physics_steps = completed_physics_steps = updates = 0
    try:
        for i in range(config.environments):
            workers.append(Worker(config_path, out / 'workers' / str(i)))
        evaluator = Worker(config_path, out / 'evaluation_worker')
        identity = workers[0].identity
        for worker in workers[1:] + [evaluator]:
            assert worker.identity == identity
        write(out / 'simulator_identity.json', identity)
        policy = Policy(identity['observation_dim'], identity['action_dim']).to(device)
        initialization = None
        if initialize is not None:
            imported = torch.load(initialize, map_location='cpu', weights_only=True)
            policy.load_state_dict(imported['policy'])
            initialization = dict(path=str(Path(initialize).resolve().relative_to(ROOT)),
                                  sha256=digest(initialize), source_controller=imported['controller'],
                                  source_updates=imported['updates'], optimizer_reused=False)
            write(out / 'initialization.json', initialization)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate, eps=1e-5)
        initial_checkpoint = checkpoint(policy, optimizer, config, out, 0, 0)
        result = evaluate(policy, evaluator, config, initial_checkpoint, out / 'evaluations' / 'update_0000')
        evaluations.append(result)
        print(json.dumps(dict(evaluation_update=0, valid_seconds=result['valid_seconds'],
                              net_deg=result['net_deg'], reason=result['reason'])), flush=True)
        obs = np.array([w.request(op='reset')['observation'] for w in workers], dtype=np.float32)
        training_started = time.monotonic()
        reason = 'update_budget'
        last_checkpoint = initial_checkpoint
        for update in range(1, config.updates + 1):
            if time.monotonic() - training_started >= config.train_wall_s:
                reason = 'training_wall_budget'
                break
            shape = (config.rollout_steps, config.environments)
            buffer = dict(observations=torch.empty((*shape, identity['observation_dim']), device=device),
                          latent=torch.empty((*shape, identity['action_dim']), device=device),
                          log_prob=torch.empty(shape, device=device), reward=torch.empty(shape, device=device),
                          value=torch.empty(shape, device=device), next_value=torch.empty(shape, device=device),
                          terminated=torch.empty(shape, device=device, dtype=torch.bool),
                          done=torch.empty(shape, device=device, dtype=torch.bool))
            rollout_started = time.monotonic()
            for t in range(config.rollout_steps):
                observation = torch.as_tensor(obs, device=device)
                with torch.no_grad():
                    action, latent, log_prob, _, value = policy.act(observation)
                bounded = action.cpu().numpy()
                for w, a in zip(workers, bounded):
                    w.send(dict(op='step', action=a.tolist()))
                responses = [w.receive() for w in workers]
                physics_steps += sum(r['physics_steps'] for r in responses)
                final_obs = np.array([r['observation'] for r in responses], dtype=np.float32)
                terminated = np.array([r['terminated'] for r in responses])
                done = np.array([r['terminated'] or r['truncated'] for r in responses])
                with torch.no_grad():
                    _, next_value = policy.distribution(torch.as_tensor(final_obs, device=device))
                buffer['observations'][t] = observation
                buffer['latent'][t] = latent
                buffer['log_prob'][t] = log_prob
                buffer['value'][t] = value
                buffer['next_value'][t] = next_value
                buffer['reward'][t] = torch.tensor([r['reward'] for r in responses], device=device)
                buffer['terminated'][t] = torch.as_tensor(terminated, device=device)
                buffer['done'][t] = torch.as_tensor(done, device=device)
                obs = final_obs
                for i, r in enumerate(responses):
                    if done[i]:
                        episode = dict(r['episode'], update=update, environment=i,
                                       training_episode=True, standalone_demo=False)
                        episodes.append(episode)
                        recent.append(episode)
                        completed_physics_steps += episode['total_steps']
                        obs[i] = workers[i].request(op='reset')['observation']
                action_steps += config.environments
            adv, returns = advantages(buffer['reward'], buffer['value'], buffer['next_value'],
                                       buffer['terminated'], buffer['done'], config.gamma, config.gae_lambda)
            flat = {k: v.flatten(0, 1) for k, v in buffer.items()}
            adv, returns = adv.flatten(), returns.flatten()
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            losses, kls, clip_fractions = [], [], []
            count = len(adv)
            for epoch in range(config.epochs):
                order = torch.randperm(count, device=device)
                epoch_kl = []
                for first in range(0, count, config.minibatch):
                    indices = order[first:first + config.minibatch]
                    _, _, lp, entropy, value = policy.act(flat['observations'][indices], flat['latent'][indices])
                    log_ratio = lp - flat['log_prob'][indices]
                    ratio = log_ratio.exp()
                    pg_loss = torch.maximum(-adv[indices] * ratio,
                                            -adv[indices] * ratio.clamp(1 - config.clip, 1 + config.clip)).mean()
                    value_loss = .5 * (returns[indices] - value).square().mean()
                    loss = pg_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy.mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite PPO loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    with torch.no_grad():
                        kl = ((ratio - 1) - log_ratio).mean().item()
                        kls.append(kl)
                        epoch_kl.append(kl)
                        losses.append(loss.item())
                        clip_fractions.append(((ratio - 1).abs() > config.clip).float().mean().item())
                if np.mean(epoch_kl) > config.target_kl:
                    break
            updates = update
            row = dict(update=update, action_steps=action_steps,
                       training_physics_steps=physics_steps,
                       completed_training_episode_physics_steps=completed_physics_steps,
                       completed_episodes=len(episodes), mean_reward=float(buffer['reward'].mean().item()),
                       recent_mean_return=float(np.mean([e['return_'] for e in recent])) if recent else None,
                       recent_mean_net_deg=float(np.mean([e['net_deg'] for e in recent])) if recent else None,
                       recent_mean_valid_s=float(np.mean([e['valid_seconds'] for e in recent])) if recent else None,
                       recent_stops=dict(Counter(e['reason'] for e in recent)),
                       mean_loss=float(np.mean(losses)), approximate_kl=float(np.mean(kls)),
                       clip_fraction=float(np.mean(clip_fractions)),
                       rollout_and_update_wall_s=time.monotonic() - rollout_started,
                       training_wall_s=time.monotonic() - training_started)
            histories.append(row)
            with (out / 'training.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row), flush=True)
            if update in (32, 64, config.updates):
                last_checkpoint = checkpoint(policy, optimizer, config, out, updates, action_steps)
                result = evaluate(policy, evaluator, config, last_checkpoint,
                                  out / 'evaluations' / f'update_{updates:04d}')
                evaluations.append(result)
                print(json.dumps(dict(evaluation_update=updates, valid_seconds=result['valid_seconds'],
                                      net_deg=result['net_deg'], peak_net_deg=result['peak_net_deg'], reason=result['reason'])), flush=True)
        if evaluations[-1]['training_updates'] != updates:
            last_checkpoint = checkpoint(policy, optimizer, config, out, updates, action_steps)
            evaluations.append(evaluate(policy, evaluator, config, last_checkpoint,
                                        out / 'evaluations' / f'update_{updates:04d}'))
        for record in manifest['files']:
            assert digest(ROOT / record['path']) == record['sha256']
        result = dict(controller=REVISION, config=asdict(config), reason=reason, initialization=initialization,
                      updates=updates, training_action_steps=action_steps,
                      training_physics_steps=physics_steps,
                      completed_training_episode_physics_steps=completed_physics_steps,
                      training_episodes=len(episodes), evaluations=evaluations,
                      last_checkpoint=dict(path=str(last_checkpoint.relative_to(ROOT)), sha256=digest(last_checkpoint)),
                      total_wall_s=time.monotonic() - started,
                      device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                      torch_version=str(torch.__version__), sources_unchanged=True,
                      policy_trained=True, privileged=True, domain_randomization=False,
                      indefinite_rotation_proven=False, hardware_robustness_proven=False,
                      last_update=histories[-1] if histories else None,
                      protocol=dict(path=str((out / 'protocol.json').relative_to(ROOT)), sha256=digest(out / 'protocol.json')),
                      simulator_identity=dict(path=str((out / 'simulator_identity.json').relative_to(ROOT)),
                                              sha256=digest(out / 'simulator_identity.json')))
        write(out / 'training_episodes.json', episodes)
        write(out / 'results.json', result)
        print(json.dumps(dict(final_update=updates, action_steps=action_steps,
                              evaluations=[{k: e[k] for k in ('training_updates', 'valid_seconds', 'net_deg', 'reason')}
                                           for e in evaluations], wall_s=result['total_wall_s'])), flush=True)
        return result
    finally:
        for w in workers:
            w.close()
        if evaluator is not None:
            evaluator.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--initialize', type=Path)
    args = parser.parse_args()
    config = Config(**json.loads(args.config.read_text())) if args.config else Config()
    train(config, args.out, args.initialize)


if __name__ == '__main__':
    main()
