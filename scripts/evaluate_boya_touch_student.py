"""Independent paired evaluation of the teacher and the two selected students."""
import argparse
import os
from pathlib import Path
import time
import traceback

from tendonspin.rl.resource_guard import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, default=Path('outputs/boya_c_fresh1p5b_server_interrupted_recovery_20261028_v1/actions_1260000000/training/teacher_final.pth'))
    parser.add_argument('--hand-side', choices=('right', 'left'), default='right')
    parser.add_argument('--left-asset', type=Path,
        help='V3.2 left_hand.usd override; defaults to the bundled snapshot')
    parser.add_argument('--scene-config', type=Path,
        help='Optional left/right scene and grasp-cache manifest')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=27)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--seed', type=int, default=2026110203)
    parser.add_argument('--debug-physics', action='store_true')
    parser.add_argument('--zero-actions', action='store_true')
    parser.add_argument('--controllers', nargs='+',
        choices=('teacher', 'touch', 'proprio'),
        default=('teacher', 'touch', 'proprio'))
    args = parser.parse_args()
    if args.num_envs < 9 or args.num_envs % 9 or args.seconds <= 0:
        parser.error('Require complete nine-scale groups and a positive horizon')
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', phase='launching', pid=os.getpid(),
                  started_unix_s=time.time(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    atomic_json(out/'status.json', status)
    from tendonspin.physics.physx_property_sync import install
    status['property_sync'] = install(lambda name: None)
    app = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False, device='cuda:0').app
        from tendonspin.students.evaluation import evaluate
        evaluate(args, root, out, status)
    except BaseException as error:
        status.update(status='stopped' if isinstance(error, InterruptedError) else 'error',
                      phase='stopped' if isinstance(error, InterruptedError) else 'error', error=str(error),
                      traceback=traceback.format_exc(), updated_unix_s=time.time())
        atomic_json(out/'status.json', status)
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
