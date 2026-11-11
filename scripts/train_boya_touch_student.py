"""Train TendonSpin's own sensor-history students with the frozen C1260M teacher."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import signal
import time
import traceback

from tendonspin.rl.resource_guard import atomic_json
from tendonspin.students.model import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, default=Path('outputs/boya_c_fresh1p5b_server_interrupted_recovery_20261028_v1/actions_1260000000/training/teacher_final.pth'))
    parser.add_argument('--hand-side', choices=('right', 'left'), default='right')
    parser.add_argument('--left-asset', type=Path,
        help='V3.2 left_hand.usd override; defaults to the bundled snapshot')
    parser.add_argument('--scene-config', type=Path,
        help='Optional left/right scene and grasp-cache manifest')
    parser.add_argument('--num-envs', type=int, default=252)
    parser.add_argument('--warmup-controls', type=int, default=32)
    parser.add_argument('--warmup-updates', type=int, default=1024)
    parser.add_argument('--online-controls', type=int, default=1968)
    parser.add_argument('--validation-controls', type=int, default=400)
    parser.add_argument('--updates-per-block', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--replay-capacity', type=int, default=65536)
    parser.add_argument('--seed', type=int, default=2026110201)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    if any(getattr(args, name) < 1 for name in ('num_envs', 'warmup_controls', 'warmup_updates',
            'online_controls', 'validation_controls', 'updates_per_block', 'batch_size', 'replay_capacity')):
        parser.error('Training budgets must be positive')
    from tendonspin.students.training import split_environments
    split_environments(args.num_envs)
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.stop_file is None:
        args.stop_file = out/'STOP'
    def stop(signum, frame):
        args.stop_file.touch()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1)
    status = dict(status='running', phase='launching', pid=os.getpid(),
        started_unix_s=time.time(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        dedicated_coupling_test=False, coupling_model='unchanged nominal 1:1',
        real_hardware_commands_sent=False)
    source_paths = list((root/'tendonspin/students').glob('*.py'))+[Path(__file__).resolve()]
    # Archive the teacher's exact physics/observation dependencies as well.
    import torch
    teacher_state = torch.load(args.teacher, map_location='cpu', weights_only=False)
    right_hand = args.hand_side == 'right'
    if right_hand:
        for relative, digest in teacher_state['contract']['sources'].items():
            if sha256(root/relative) != digest:
                raise RuntimeError('Original C teacher dependency changed: '+relative)
    source_paths += [root/name for name in teacher_state['contract']['sources']]
    source_paths += list((root/'tendonspin/physics').glob('*.py'))
    source_paths += list((root/'tendonspin/rl').glob('*.py'))
    status['historical_teacher_sources_verified'] = right_hand
    if not right_hand:
        status['historical_teacher_source_note'] = (
            'Left-hand migration intentionally replaces the right-hand asset profile; '
            'current left/physics sources are pinned instead of the original right environment hashes.')
    for path in sorted(set(source_paths)):
        relative = path.resolve().relative_to(root)
        content = path.read_bytes()
        snapshot = out/'sources'/relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
        status.setdefault('sources', []).append(dict(path=str(relative), sha256=sha256(path)))
    del teacher_state
    atomic_json(out/'status.json', status)
    from tendonspin.physics.physx_property_sync import install
    status['property_sync'] = install(lambda name: None)
    app = None
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False, device='cuda:0').app
        from tendonspin.students.training import train
        train(args, root, out, status)
        for source in status['sources']:
            if sha256(root/source['path']) != source['sha256']:
                raise RuntimeError('Source changed during training: '+source['path'])
    except BaseException as error:
        status.update(status='stopped' if isinstance(error, InterruptedError) else 'error',
                      phase='stopped' if isinstance(error, InterruptedError) else 'error',
                      error=str(error), traceback=traceback.format_exc(), updated_unix_s=time.time())
        atomic_json(out/'status.json', status)
        raise
    finally:
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
