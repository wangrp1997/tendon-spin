"""Run an unchanged native C worker with the audited process-local adapter."""
import argparse
import atexit
import hashlib
from pathlib import Path
import os
import runpy
import sys
import time

from tendonspin.physics.physx_property_sync import install
from tendonspin.rl.resource_guard import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entry', choices=('train', 'evaluate'), required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('worker_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    audit = args.audit.resolve()
    if audit.exists():
        raise ValueError('Compatibility audit already exists; refuse overwrite')
    worker_args = args.worker_args[1:] if args.worker_args[:1]==['--'] else args.worker_args
    script = ROOT/'scripts'/('train_boya_spin_progress.py' if args.entry=='train' else 'evaluate_boya_spin_progress.py')
    data = dict(pid=os.getpid(), started_unix_s=time.time(), worker=str(script),
        worker_sha256=hashlib.sha256(script.read_bytes()).hexdigest(), synchronized_copies={})
    for path in (Path(__file__), ROOT/'tendonspin/physics/physx_property_sync.py'):
        data.setdefault('adapter_sources', []).append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def save():
        atomic_json(audit, data)

    def waited(name):
        counts = data['synchronized_copies']
        counts[name] = counts.get(name, 0)+1
        total = sum(counts.values())
        if total <= 10 or total % 1000 == 0:
            save()

    data['adapter'] = install(waited)
    save()
    atexit.register(save)
    sys.argv = [str(script), *worker_args]
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        data['finished_unix_s'] = time.time()
        save()


if __name__=='__main__':
    main()
