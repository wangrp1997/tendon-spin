# Sources: github.com/wangrp1997/tendon-spin, f58b489 (MIT),
# run_boya_hora_background.py / evaluate_boya_hora.py / analyze_boya_small_budget.py.
# Learner reference: github.com/HaozhiQi/hora v0.0.1 (MIT), arXiv:2210.04887.
# Addition: five fixed independent frozen episodes; no training or policy changes.
"""Run the authorized B40M initial-state/long-window checklist, then stop."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

from tendonspin.evaluation.comparison import digest, read_json, verify_checkpoint, verify_sources
from tendonspin.evaluation.rotation import score_prefix
from tendonspin.rl.resource_guard import atomic_json

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/experiments/2026-10-13-boya-b40m-validation'
PLAN_DOC = ROOT / 'docs/next_steps.md'
TRAIN = ROOT / 'outputs/boya_uniform48p8_b50m_v1/actions_040000000/training'
CACHE = ROOT / 'outputs/boya_uniform48p8_isaac_transfer_v1/grasp_cache.npz'
CASES = [dict(name=f'cache_{i:03d}_30s', cache_index=i, seconds=30., wall_s=1500.)
         for i in (0, 7, 15, 23)] + [
             dict(name='nominal_120s', cache_index=None, seconds=120., wall_s=6000.)]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def report(state, rows):
    """Update only this stage's progress and its single checkbox."""
    phase = state['phase']
    count = len(rows)
    finished = phase == 'completed'
    ready = finished and all(row['finite_rotation_signal'] for row in rows)
    note = f'{phase}，已完成{count}/5项'
    if state.get('current_case') and not finished:
        note += '；当前' + state['current_case']
    if finished:
        note += f'；初步旋转信号{sum(r["finite_rotation_signal"] for r in rows)}/5'
    if state.get('error'):
        note += '；执行阻塞：' + state['error']['message'].replace('\n', ' ')
    progress_line = ('  进度：' + note + '；结果见'
        '[本阶段记录](experiments/2026-10-13-boya-b40m-validation/README.md)。')
    text = PLAN_DOC.read_text()
    lines = text.splitlines()
    step = next(i for i, line in enumerate(lines) if '第1步：冻结五指B40M' in line)
    lines[step] = lines[step].replace('[ ]', '[x]' if finished else '[ ]', 1)
    progress = next(i for i in range(step + 1, len(lines)) if lines[i].startswith('  进度：'))
    lines[progress] = progress_line
    PLAN_DOC.write_text('\n'.join(lines) + '\n')
    atomic_json(DOC / 'data/progress.json', dict(updated=now(), phase=phase,
        completed_episodes=count, planned_episodes=5, ready_for_next_stage=ready,
        checkpoint_sha256=state.get('checkpoint_sha256'), rows=rows,
        error=state.get('error'), output=state['output']))
    body = ['# 五指B40M教师验证', '', f'状态：{note}。更新时间：{now()}。', '',
        '固定4个训练缓存初态各30秒＋原名义初态120秒，最多240模拟秒，0训练动作。',
        '同一冻结B40M权重/normalizer、48.77355×32mm/50g圆柱、16动作、特权观测、Isaac原引擎/B奖励/workspace规则。',
        '每次独立episode；缓存样本不是未见抓取，角度不跨episode相加。', '',
        f'输出：{state["output"]}。',
        '按历史速度预计约65–85分钟；以实际进度与停止原因更新。', '',
        '[固定协议](PROTOCOL.md)、[后续计划](../../next_steps.md)、[结果与进度](data/progress.json)。']
    if rows:
        body += ['', '| 初态 | valid/requested秒 | 净角° | 峰值° | 回退° | 末段净角° | 漂移mm | 倾斜° | 停止原因 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        for row in rows:
            tail = row['full_window_final_tail_net_deg']
            tail_text = '未观察完整末段' if tail is None else f'{tail:.3f}'
            drift = row['drift_mm']; tilt = row['tilt_deg']
            drift_text = '无有效前缀' if drift is None else f'{drift:.3f}'
            tilt_text = '无有效前缀' if tilt is None else f'{tilt:.3f}'
            body += [f'| {row["name"]} | {row["valid_s"]:.4f}/{row["requested_s"]:.0f} | '
                f'{row["net_deg"]:+.3f} | {row["peak_deg"]:.3f} | {row["backward_deg"]:.3f} | '
                f'{tail_text} | {drift_text} | {tilt_text} | {row["stop_reason"]} |']
        body += ['', '30秒节点末段为最后10秒，120秒节点为最后30秒；漂移/倾斜是有效前缀最大值，沿用训练canonical参考。']
    if finished:
        body += ['', '固定评估清单已完成。' + ('全部满足预声明初步旋转信号。' if ready else
            '部分节点未满足预声明初步旋转信号；后续训练尚未启动，需要依据失败记录讨论。'),
            '所有权重/旧实验保留；单seed、训练初态检查不是未见条件泛化、无限旋转或实机结果。']
        (DOC / 'RESULTS.md').write_text('\n'.join(body) + '\n')
    (DOC / 'README.md').write_text('\n'.join(body) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (DOC / 'data').mkdir(exist_ok=True)
    os.nice(10)
    started = time.monotonic()
    child = None
    requested = None
    rows = []
    state = dict(pid=os.getpid(), phase='preparing', started=now(), output=str(out),
        completed_episodes=0, training_actions=0, planned_episodes=5,
        automatic_retry=False, automatic_extension=False, resource_watchdog=False)

    def stop(signum, frame):
        nonlocal requested
        requested = signum
        if child is not None and child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def save():
        state.update(updated=now(), wall_s=time.monotonic() - started,
                     completed_episodes=len(rows))
        atomic_json(out / 'status.json', state)
        report(state, rows)
        print('B40M_VALIDATION ' + str({key: state.get(key) for key in
            ('phase', 'current_case', 'completed_episodes', 'child_pid', 'wall_s')}), flush=True)

    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT), OMNI_KIT_ACCEPT_EULA='YES',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    try:
        training = read_json(TRAIN / 'result.json')
        verify_sources(ROOT, training['sources'])
        identity = verify_checkpoint(ROOT, training, TRAIN / 'teacher_final.pth', 40001536)
        if identity['checkpoint_sha256'] != 'c400562ff3d04497b451aa6594c6fa09c4268f8bc2c2203d312ce281ac0400e6':
            raise ValueError('Frozen B40M identity differs from the authorized protocol')
        if digest(CACHE) != training['cache_sha256']:
            raise ValueError('Declared training cache changed')
        names = ('scripts/run_boya_teacher_validation.py', 'scripts/evaluate_boya_hora_initial_state.py',
                 'scripts/analyze_boya_small_budget.py', str((DOC / 'PROTOCOL.md').relative_to(ROOT)))
        sources = [dict(path=name, sha256=digest(ROOT / name)) for name in names]
        state['checkpoint_sha256'] = identity['checkpoint_sha256']
        plan = dict(created=now(), checkpoint=identity, training_record=str(TRAIN / 'result.json'),
            training_record_sha256=digest(TRAIN / 'result.json'), cache=str(CACHE),
            cache_sha256=digest(CACHE), cases=CASES, maximum_new_episodes=5,
            maximum_physics_steps=480000, maximum_control_actions=4800,
            training_actions=0, unseen_grasp_validation=False, sources=sources,
            same_teacher_contract=training['scene_configuration'])
        atomic_json(out / 'plan.json', plan)
        atomic_json(DOC / 'data/launch.json', plan)
        for source in sources:
            path = out / 'sources' / source['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / source['path']).read_bytes())
        save()
        for case in CASES:
            if requested is not None:
                break
            verify_sources(ROOT, training['sources'])
            verify_sources(ROOT, sources)
            if digest(TRAIN / 'teacher_final.pth') != identity['checkpoint_sha256']:
                raise ValueError('Frozen checkpoint changed during the validation')
            if digest(CACHE) != training['cache_sha256']:
                raise ValueError('Frozen cache changed during the validation')
            run = out / case['name']
            run.mkdir(exist_ok=False)
            command = [sys.executable, str(ROOT / 'scripts/evaluate_boya_hora_initial_state.py'),
                '--out', str(run / 'evaluation'), '--checkpoint', str(TRAIN / 'teacher_final.pth'),
                '--cache', str(CACHE), '--source-record', str(TRAIN / 'result.json'),
                '--seconds', str(case['seconds']), '--wall-s', str(case['wall_s']), '--reward-diagnostics']
            if case['cache_index'] is not None:
                command += ['--cache-index', str(case['cache_index'])]
            else:
                command += ['--initial-reference', str(TRAIN.parent / 'evaluation/initial_state.npz')]
            state.update(phase='evaluation', current_case=case['name'], command=command)
            with (run / 'evaluation.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save()
                code = child.wait()
            if requested is not None:
                break
            if code:
                raise RuntimeError(f'{case["name"]} evaluation exit {code}; no retry')
            record = read_json(run / 'evaluation/result.json')
            expected = dict(status='completed', checkpoint_sha256=identity['checkpoint_sha256'],
                requested_s=case['seconds'], episode_resets=0, controller_switches=0, training_actions=0,
                training_actions_in_checkpoint=40001536, seed=43,
                termination=training['termination'], engine_configuration=training['engine_configuration'],
                reward_configuration=training['reward_configuration'])
            for key, value in expected.items():
                if record.get(key) != value:
                    raise ValueError('Frozen episode contract differs: ' + key)
            if record['initial_state_selection']['cache_index'] != case['cache_index']:
                raise ValueError('Evaluation selected an unexpected initial state')
            command = [sys.executable, str(ROOT / 'scripts/analyze_boya_small_budget.py'),
                '--evaluation', str(run / 'evaluation'), '--training', str(TRAIN / 'result.json'),
                '--out', str(run / 'analysis')]
            state.update(phase='analysis', command=command)
            with (run / 'analysis.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save()
                code = child.wait()
            if requested is not None:
                break
            if code:
                raise RuntimeError(f'{case["name"]} analysis exit {code}; no retry')
            import numpy as np
            analysis = read_json(run / 'analysis/result.json')
            angles = np.load(run / 'evaluation/angles_deg.npy')
            metric = score_prefix(angles, .0005, record['valid_steps'], window_s=case['seconds'],
                                  tail_s=10. if case['seconds'] == 30. else 30.)
            signal_ok = (metric['complete_window_observed'] and metric['net_deg'] > 0 and
                         metric['full_window_final_tail_net_deg'] > 0)
            row = dict(name=case['name'], cache_index=case['cache_index'],
                candidate_id=record['initial_state_selection'].get('candidate_id'),
                requested_s=case['seconds'], actual_s=record['actual_s'], valid_s=record['valid_s'],
                net_deg=metric['net_deg'], peak_deg=metric['peak_deg'], backward_deg=metric['backward_deg'],
                full_window_final_tail_net_deg=metric['full_window_final_tail_net_deg'],
                drift_mm=analysis['valid_prefix_maxima']['drift_mm'],
                tilt_deg=analysis['valid_prefix_maxima']['tilt_deg'],
                max_normal_N=analysis['valid_prefix_maxima']['max_normal'], stop_reason=record['stop_reason'],
                initial_state_max_errors=record['initial_state_max_errors'],
                finite_rotation_signal=signal_ok, checkpoint_sha256=record['checkpoint_sha256'],
                evaluation=str(run / 'evaluation/result.json'), analysis=str(run / 'analysis/result.json'))
            rows.append(row)
            atomic_json(out / 'results.json', dict(rows=rows))
            save()
        state.update(phase='stopped' if requested is not None else 'completed',
                     finished=now(), requested_signal=requested)
    except BaseException as error:
        state.update(phase='error', error=dict(type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc()), finished=now())
    finally:
        save()
    if state['phase'] == 'error':
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
