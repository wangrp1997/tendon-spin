"""Patched-physics subprocess; stdout is reserved for JSON protocol responses."""
import argparse
import json
from pathlib import Path
import sys
import traceback

from .config import Config
from .environment import RotationEnv


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    env = RotationEnv(Config(**json.loads(args.config.read_text())))
    emit(env.identity())
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request['op'] == 'close':
                break
            if request['op'] == 'reset':
                obs = env.reset(request.get('max_seconds'), request.get('directory'), request.get('attribution'))
                emit(dict(observation=obs.tolist()))
            elif request['op'] == 'step':
                before = env.total_steps
                obs, reward, terminated, truncated, info = env.step(request['action'])
                emit(dict(observation=obs.tolist(), reward=reward, terminated=terminated,
                          truncated=truncated, episode=info, physics_steps=env.total_steps - before))
            else:
                raise ValueError('Unknown request')
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit(dict(error=f'{type(exc).__name__}: {exc}'))
            break


if __name__ == '__main__':
    main()
