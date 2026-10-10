# Source: TendonSpin run_boya_hora_background.py process sequencing. New bounded
# user-authorized two-episode diagnostic, no retries, learner or resource watcher.
"""Run4/16 velocity-iteration probes sequentially and analyze their archives."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/boya_velocity_diagnostic_v1')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMNI_KIT_ACCEPT_EULA='YES', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    started = time.monotonic()
    state = dict(phase='starting', pid=os.getpid(), requested_physics_seconds=60., training_actions=0,
        automatic_retry=False, automatic_resource_stop=False, runs=[])
    child = None
    requested = None

    def save():
        state['wall_s'] = time.monotonic()-started
        (out/'status.json').write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
        print('VELOCITY_PAIR '+json.dumps(state), flush=True)

    def stop(signum, frame):
        nonlocal requested
        requested = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        for iterations in (4, 16):
            if requested is not None:
                raise RuntimeError('Manual stop; no next episode')
            run = out/f'velocity_{iterations:02d}'
            command = [sys.executable, str(ROOT/'scripts/evaluate_boya_velocity_probe.py'),
                '--out', str(run), '--velocity-iterations', str(iterations)]
            row = dict(velocity_iterations=iterations, output=str(run), command=command)
            state['runs'].append(row)
            state['phase'] = f'velocity_{iterations:02d}'
            save()
            with (out/f'velocity_{iterations:02d}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                row['pid'] = child.pid
                save()
                row['exit_code'] = child.wait()
            if (run/'result.json').exists():
                value = json.loads((run/'result.json').read_text())
                row.update({k: value.get(k) for k in ('status', 'actual_s', 'valid_s', 'net_deg', 'stop_reason')})
            save()
            if row['exit_code'] or row.get('status') != 'completed' or requested is not None:
                raise RuntimeError('Episode error/manual stop; chain stopped without retry')
        state['phase'] = 'analyzing'
        save()
        command = [sys.executable, str(ROOT/'scripts/analyze_boya_velocity_probe.py'), '--out', str(out)]
        with (out/'analysis.log').open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            state['analysis_exit_code'] = child.wait()
        if state['analysis_exit_code']:
            raise RuntimeError('Analysis failed; retain both episodes')
        state['phase'] = 'completed'
    except BaseException as error:
        state.update(phase='stopped' if requested else 'error',
            error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        save()
    return 0 if state['phase'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
