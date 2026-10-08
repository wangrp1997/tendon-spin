"""Local JSON pipes keep the existing cp311 patched engine separate from Torch.

All workers execute the same engine/model. The GPU learner never imports another
MuJoCo installation. Sending every action before reading responses permits
parallel native simulations without changing a single physics setting.
"""
import json
import os
from pathlib import Path
import subprocess

from .config import ROOT, RUNTIME, SIM_PYTHON


class Worker:
    def __init__(self, config, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.log = (directory / 'stderr.txt').open('w')
        env = os.environ.copy()
        env.update(PYTHONPATH=str(RUNTIME) + os.pathsep + str(ROOT),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        self.process = subprocess.Popen(
            [SIM_PYTHON, '-u', '-m', 'tendonspin.rl.worker', '--config', str(config)],
            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1)
        try:
            self.identity = self.receive()
        except BaseException:
            self.close()
            raise

    def send(self, value):
        self.process.stdin.write(json.dumps(value, allow_nan=False) + '\n')
        self.process.stdin.flush()

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            self.log.flush()
            raise RuntimeError('Native worker exited; inspect ' + self.log.name)
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def request(self, **value):
        self.send(value)
        return self.receive()

    def close(self):
        process = getattr(self, 'process', None)
        if process is not None:
            if process.poll() is None:
                try:
                    self.send(dict(op='close'))
                    process.wait(timeout=10)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            if process.stdin:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
        if getattr(self, 'log', None):
            self.log.close()
