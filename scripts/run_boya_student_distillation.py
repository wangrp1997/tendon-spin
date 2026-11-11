"""Run training then an independent paired test, preserving all artifacts."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tendonspin.rl.resource_guard import atomic_json
from tendonspin.students.model import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hand-side', choices=('right', 'left'), default='right')
    parser.add_argument('--left-asset', type=Path,
        help='V3.2 left_hand.usd override; defaults to the bundled snapshot')
    parser.add_argument('--scene-config', type=Path,
        help='Optional scene and grasp-cache manifest used by training and evaluation')
    parser.add_argument('--num-envs', type=int, default=252)
    parser.add_argument('--warmup-controls', type=int, default=32)
    parser.add_argument('--online-controls', type=int, default=1968)
    parser.add_argument('--validation-controls', type=int, default=400)
    parser.add_argument('--warmup-updates', type=int, default=1024)
    parser.add_argument('--updates-per-block', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--replay-capacity', type=int, default=65536)
    parser.add_argument('--test-envs', type=int, default=27)
    parser.add_argument('--test-seconds', type=float, default=30.)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', phase='launching', pid=os.getpid(),
        started_unix_s=time.time(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        training_seed=2026110201, test_seed=2026110203, arguments=vars(args).copy(),
        teacher='C1260M', coupling_model='unchanged nominal 1:1',
        test_used_for_fitting_or_selection=False, real_hardware_commands_sent=False)
    state['arguments']['out'] = str(out)
    sources = (list((root/'tendonspin/students').glob('*.py'))
        + list((root/'tendonspin/physics').glob('*.py'))
        + list((root/'tendonspin/rl').glob('*.py'))
        + [Path(__file__).resolve(), root/'scripts/train_boya_touch_student.py',
           root/'scripts/evaluate_boya_touch_student.py'])
    state['sources'] = {str(path.relative_to(root)): sha256(path) for path in sources}
    for path in sources:
        target = out/'sources'/path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    atomic_json(out/'status.json', state)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        (out/'STOP').touch()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    training = [sys.executable, '-u', str(root/'scripts/train_boya_touch_student.py'), '--out', str(out/'training')]
    training += ['--stop-file', str(out/'STOP')]
    training += ['--hand-side', args.hand_side]
    if args.left_asset is not None:
        training += ['--left-asset', str(args.left_asset)]
    if args.scene_config is not None:
        training += ['--scene-config', str(args.scene_config)]
    for key in ('num_envs', 'warmup_controls', 'online_controls', 'validation_controls', 'warmup_updates',
                'updates_per_block', 'batch_size', 'replay_capacity'):
        training += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    evaluation = [sys.executable, '-u', str(root/'scripts/evaluate_boya_touch_student.py'),
        '--training', str(out/'training'), '--out', str(out/'evaluation'),
        '--hand-side', args.hand_side, '--num-envs', str(args.test_envs),
        '--seconds', str(args.test_seconds)]
    if args.left_asset is not None:
        evaluation += ['--left-asset', str(args.left_asset)]
    if args.scene_config is not None:
        evaluation += ['--scene-config', str(args.scene_config)]
    try:
        for phase, command in (('training', training), ('evaluation', evaluation)):
            for relative, digest in state['sources'].items():
                if sha256(root/relative) != digest:
                    raise RuntimeError('Pinned pipeline source changed: '+relative)
            if stopped or (out/'STOP').exists():
                raise InterruptedError('Stop requested')
            with (out/f'{phase}.log').open('xb') as log:
                worker = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                state.update(phase=phase, worker_pid=worker.pid, command=command, updated_unix_s=time.time())
                atomic_json(out/'status.json', state)
                code = worker.wait()
            if code:
                raise RuntimeError(f'{phase} worker exited with code {code}; see {phase}.log')
            result = json.loads((out/phase/'status.json').read_text())
            if result['status'] != 'completed':
                raise RuntimeError(f'{phase} worker did not complete')
        comparison = json.loads((out/'evaluation'/'comparison.json').read_text())
        state.update(status='completed', phase='completed', worker_pid=None,
            completed_unix_s=time.time(), comparison=comparison['controllers'],
            touch_student=str(out/'training'/'touch_best.pth'),
            proprio_student=str(out/'training'/'proprio_best.pth'),
            hardware_validated=False)
        atomic_json(out/'status.json', state)
    except BaseException as error:
        is_stop = stopped or (out/'STOP').exists()
        state.update(status='stopped' if is_stop else 'error', phase='stopped' if is_stop else 'error',
                     error=str(error), updated_unix_s=time.time())
        atomic_json(out/'status.json', state)
        raise


if __name__ == '__main__':
    main()
