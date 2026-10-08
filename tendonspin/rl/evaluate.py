# Adapted botyard-inhand/research/learning/evaluate.py
# Source revision: 204d197a9606fb7266e884f3b2e6110195be01cb.
# TendonSpin changes and evaluation identity are declared in docs/experiment_state.md.
# This local PPO is not a reproduced Hora/AnyRotate/Sharpa policy.
"""Independent original-state execution of one frozen policy checkpoint."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch

from .config import Config, ROOT, REVISION, digest, write
from .policy import Policy
from .transport import Worker


def evaluate(policy, worker, config, checkpoint, out, seconds=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(checkpoint).resolve()
    identity = dict(checkpoint=str(checkpoint.relative_to(ROOT)),
                    checkpoint_sha256=digest(checkpoint), inference='deterministic tanh(mean)',
                    policy_source_sha256=digest(ROOT / 'tendonspin/rl/policy.py'),
                    config=asdict(config), controller=REVISION)
    # Evaluation uses weights read back from the declared checkpoint, not a
    # mutable learner reference that might differ from what was saved.
    frozen = Policy(worker.identity['observation_dim'], worker.identity['action_dim'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['controller'] == REVISION and saved['config'] == asdict(config)
    frozen.load_state_dict(saved['policy'])
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), frozen.state_dict()[key], rtol=0, atol=0)
    frozen.eval()
    duration = config.evaluation_s if seconds is None else seconds
    started = time.monotonic()
    response = worker.request(op='reset', max_seconds=duration,
                              directory=str(out / 'native'), attribution=identity)
    observations, actions = [], []
    while True:
        observation = np.array(response['observation'], dtype=np.float32)
        with torch.no_grad():
            action = frozen.deterministic(torch.from_numpy(observation)[None])[0].numpy()
        observations.append(observation)
        actions.append(action)
        response = worker.request(op='step', action=action.tolist())
        if response['terminated'] or response['truncated']:
            break
    result = response['episode']
    trace = out / 'policy_trace.npz'
    np.savez_compressed(trace, observations=observations, actions=actions)
    # Repeat inference on the stored inputs with the same frozen checkpoint.
    with torch.no_grad():
        inference_error = 0.
        for observation, action in zip(observations, actions):
            actual = frozen.deterministic(torch.from_numpy(observation)[None])[0].numpy()
            inference_error = max(inference_error, float(abs(actual - action).max()))
    assert inference_error == 0.
    assert identity['checkpoint_sha256'] == digest(checkpoint)
    result.update(inference_error=inference_error,
                  policy_trace=dict(path=str(trace.relative_to(ROOT)), sha256=digest(trace)),
                  evaluation_wall_s=time.monotonic() - started, evaluation_seconds_limit=duration,
                  training_updates=saved['updates'], training_action_steps=saved['action_steps'])
    write(out / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = Config(**saved['config'])
    args.out.mkdir(parents=True, exist_ok=False)
    config_path = args.out / 'config.json'
    write(config_path, asdict(config))
    worker = Worker(config_path.resolve(), args.out / 'worker')
    try:
        policy = Policy(worker.identity['observation_dim'], worker.identity['action_dim'])
        policy.load_state_dict(saved['policy'])
        result = evaluate(policy, worker, config, args.checkpoint, args.out / 'episode')
        print({k: result[k] for k in ('valid_seconds', 'net_deg', 'peak_net_deg', 'reason')}, flush=True)
    finally:
        worker.close()


if __name__ == '__main__':
    main()
