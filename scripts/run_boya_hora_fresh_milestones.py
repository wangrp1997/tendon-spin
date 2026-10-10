# Sources: github.com/wangrp1997/tendon-spin, commit95ded72,
# run_boya_hora_milestones.py / run_boya_hora_background.py / comparison.py.
# Learner: github.com/HaozhiQi/hora v0.0.1 (MIT), arXiv:2210.04887.
# Port changes: fresh A/B arm scheduling and matched30s reports only. No PPO,
# physics, reward formula or ordinary strict-resume contract changes.
"""Run one fresh reward arm to10M with four fixed frozen-policy evaluations."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts.run_boya_hora_milestones import BATCH, require_completed_stage
from tendonspin.evaluation.comparison import digest, read_json, validate_result, verify_checkpoint, verify_sources
from tendonspin.rl.resource_guard import atomic_json
from tendonspin.rl.rotation_reward import POSE_DELTA, POSE_DROP

ROOT = Path(__file__).resolve().parents[1]
REQUESTED_ACTIONS = (1000000, 3000000, 5000000, 10000000)
ARM_PROFILES = {'A': POSE_DELTA, 'B': POSE_DROP}
REFERENCE = ROOT / 'outputs/boya_hora1024_workspace50m_v1/actions_020000000/evaluation/initial_state.npz'
CACHE = ROOT / 'outputs/boya_settled_cache_v2/grasp_cache.npz'
CACHE_SHA256 = '7b575ca1203b1a03129aa7189c693680c3443095925d98afbf300eda22321eca'


def targets():
    return [dict(requested_actions=n, rounded_actions=((n+BATCH-1)//BATCH)*BATCH)
            for n in REQUESTED_ACTIONS]


def command_for(arm, run, target, protocol, previous=None):
    command = [sys.executable, str(ROOT/'scripts/run_boya_hora_background.py'),
        '--out', str(run), '--total-actions', str(target['requested_actions']),
        '--termination-profile', 'boya_workspace', '--engine-profile', 'original_tgs16_4',
        '--reward-profile', ARM_PROFILES[arm], '--eval-seconds', '30', '--reward-diagnostics',
        '--initial-reference', str(REFERENCE), '--controller', 'hora_boya_pose_ab_fresh10m_'+arm.lower(),
        '--protocol', protocol]
    if previous is not None:
        command += ['--resume', str(ROOT/previous['checkpoint'])]
    return command


def require_stage(stage, actions):
    require_completed_stage(stage, actions)
    analysis = stage.get('analysis', {})
    if not isinstance(analysis, dict) or analysis.get('exit_code') != 0 or analysis.get('status') != 'completed':
        raise RuntimeError('Incomplete milestone analysis; stop without retry')


def initial_identity(run):
    """Actual first-update proof; hash initial tensors for later A/B equality."""
    import torch
    initial = torch.load(run/'training/teacher_initial.pth', map_location='cpu', weights_only=False)
    first = torch.load(run/'training/teacher_u000001.pth', map_location='cpu', weights_only=False)
    if first['agent_steps'] != BATCH or first['epoch_num'] != 1:
        raise ValueError('Fresh arm did not start at zero actions/updates')
    steps = {float(v['step']) for v in first['optimizer']['state'].values()}
    if steps != {80.} or 'reward_branch_migration' in first['progress']:
        raise ValueError('Fresh arm inherited optimizer state or reward ancestry')
    hashes = {}
    for name, state in initial.items():
        h = hashlib.sha256()
        for key, value in sorted(state.items()):
            value = value.detach().cpu().contiguous()
            h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
            h.update(value.numpy().tobytes())
        hashes[name] = h.hexdigest()
    return dict(seed=43, fresh_actions_before_first_update=0, fresh_updates_before_first_update=0,
        first_checkpoint_actions=BATCH, first_checkpoint_update=1, first_adam_steps_per_parameter=80,
        initial_state_tensor_hashes=hashes, first_checkpoint_sha256=digest(run/'training/teacher_u000001.pth'))


def evaluation_row(arm, run, target, previous):
    import numpy as np
    from tendonspin.evaluation.rotation import score_prefix
    training = read_json(run/'training/result.json')
    verify_sources(ROOT, training['sources'])
    identity = verify_checkpoint(ROOT, training, ROOT/training['checkpoint'], target['rounded_actions'])
    if training['seed'] != 43 or training['num_envs'] != 1024 or training['reward_profile'] != ARM_PROFILES[arm]:
        raise ValueError('Fresh arm configuration differs from plan')
    if previous is None:
        if training['resume_from'] is not None or training['session_actions_executed'] != target['rounded_actions']:
            raise ValueError('First milestone is not a fresh learner')
    elif (training.get('resumed_checkpoint_sha256') != previous['checkpoint_sha256'] or
          training['lineage_id'] != previous['lineage_id'] or
          training['session_actions_executed'] != target['rounded_actions']-previous['actions_executed']):
        raise ValueError('Continuation did not preserve its own arm and cumulative counts')
    result = read_json(run/'evaluation/result.json')
    validate_result(result, dict(requested_s=30., seed=43, initial_state='original grasp44, not cache',
        termination=training['termination'], expected_training_actions=target['rounded_actions'],
        runtime_sources=training['sources']), identity['checkpoint_sha256'], 'frozen_'+training['controller'])
    if max(result['initial_state_max_errors'].values()) != 0:
        raise ValueError('Frozen evaluation original state differs from reference')
    analysis = read_json(run/'analysis/result.json')
    metric = result['metrics']['30.0']
    tail = score_prefix(np.load(run/'evaluation/angles_deg.npy'), .0005, result['valid_steps'],
        window_s=30., tail_s=10.)['full_window_final_tail_net_deg']
    return dict(arm=arm, reward_profile=ARM_PROFILES[arm], requested_actions=target['requested_actions'],
        training_actions=target['rounded_actions'], actual_s=result['actual_s'], valid_s=metric['valid_seconds'],
        net_deg=metric['net_deg'], peak_deg=metric['peak_deg'], backward_deg=metric['backward_deg'],
        final_10s_net_deg=tail, complete_window=metric['complete_window_observed'],
        stop_reason=result['stop_reason'], **analysis['valid_prefix_maxima'],
        checkpoint=identity['checkpoint'], checkpoint_sha256=identity['checkpoint_sha256'],
        evaluation=str(run/'evaluation/result.json'), initial_state_max_error=0.), training


def report(out, rows):
    atomic_json(out/'evaluations.json', dict(window_s=30., rows=rows, single_seed=True,
        interpretation='One uninterrupted frozen original-state episode per checkpoint; no reset/policy sums.'))
    with (out/'evaluations.csv').open('w', newline='') as file:
        writer=csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines=['# 从零训练节点评估', '',
        '| 组 | 实际训练动作 | 有效秒 | 净角° | 峰值° | 回退° | 20–30秒净角° | 停止原因 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        tail='未观测' if row['final_10s_net_deg'] is None else f'{row["final_10s_net_deg"]:.3f}'
        lines.append(f'| {row["arm"]} | {row["training_actions"]:,} | {row["valid_s"]:.4f} | '
            f'{row["net_deg"]:.3f} | {row["peak_deg"]:.3f} | {row["backward_deg"]:.3f} | '
            f'{tail} | {row["stop_reason"]} |')
    (out/'evaluations.md').write_text('\n'.join(lines)+'\n')
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(out/'evaluation_tb')) as writer:
        row=rows[-1]
        for key in ('valid_s','net_deg','peak_deg','backward_deg','final_10s_net_deg','drift_mm','tilt_deg','max_normal'):
            if row[key] is not None: writer.add_scalar('evaluation30s/'+key,row[key],row['training_actions'])
        writer.add_scalar('evaluation30s/complete_window',int(row['complete_window']),row['training_actions'])
        writer.add_text('evaluation30s/stop_reason',row['stop_reason'],row['training_actions'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm', choices=tuple(ARM_PROFILES), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protocol', required=True)
    args=parser.parse_args(); out=args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pins=[dict(path=p,sha256=digest(ROOT/p)) for p in (args.protocol,
        'scripts/run_boya_hora_fresh_milestones.py','scripts/run_boya_hora_milestones.py',
        'scripts/run_boya_hora_background.py','scripts/train_boya_hora.py','scripts/analyze_boya_small_budget.py',
        'tendonspin/evaluation/comparison.py','tendonspin/evaluation/rotation.py')]
    if digest(CACHE) != CACHE_SHA256: raise ValueError('Declared28-state cache changed')
    plan=dict(arm=args.arm,reward_profile=ARM_PROFILES[args.arm],initialization='fresh network/normalizers/Adam/RNG seed43; zero actions',
        targets=targets(),protocol=args.protocol,sources=pins,cache_sha256=CACHE_SHA256,
        initial_reference=str(REFERENCE),initial_reference_sha256=digest(REFERENCE),
        evaluation_window_s=30.,new_frozen_episodes=4,num_envs=1024,seed=43,
        resume_environment='At each later milestone restore full own-arm learner; reset physics from same28-state cache',
        resource_watchdog=False,training_wall_limit=False,auto_retry=False,auto_extension=False)
    atomic_json(out/'plan.json',plan)
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    state=dict(pid=os.getpid(),arm=args.arm,phase='preparing',completed_milestones=[],fresh_start_actions=0,
        requested_total_actions=10000000,rounded_final_actions=targets()[-1]['rounded_actions'])
    started=time.monotonic(); child=None; requested=False; previous=None; rows=[]
    def save():
        state.update(wall_s=time.monotonic()-started,updated_unix_s=time.time()); atomic_json(out/'status.json',state)
    def stop(signum,frame):
        nonlocal requested
        requested=True
        if child is not None and child.poll() is None: child.send_signal(signum)
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop); save()
    environment=os.environ.copy(); environment.update(PYTHONPATH=str(ROOT),OMNI_KIT_ACCEPT_EULA='YES',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    try:
        for target in targets():
            if requested: break
            verify_sources(ROOT,pins)
            if digest(REFERENCE) != plan['initial_reference_sha256']: raise ValueError('Initial reference changed')
            if previous is not None:
                verify_sources(ROOT,previous['sources'])
                verify_checkpoint(ROOT,previous,ROOT/previous['checkpoint'],previous['actions_executed'])
            run=out/f'actions_{target["requested_actions"]:09d}'
            command=command_for(args.arm,run,target,args.protocol,previous)
            state.update(phase='training_then_evaluation',current_target=target,current_run=str(run),command=command)
            save()
            with (out/f'actions_{target["requested_actions"]:09d}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid; save()
                if requested: child.send_signal(signal.SIGTERM)
                code=child.wait()
            if requested: break
            if code: raise RuntimeError(f'Milestone supervisor exited{code}; stop without retry')
            require_stage(read_json(run/'stage.json'),target['rounded_actions'])
            row,current=evaluation_row(args.arm,run,target,previous)
            if previous is None: atomic_json(out/'initial_identity.json',initial_identity(run))
            rows.append(row); report(out,rows); previous=current
            state['completed_milestones'].append(dict(**target,run=str(run),evaluation=row['evaluation']))
            save()
        state['phase']='stopped' if requested else 'completed'
    except BaseException as error:
        state.update(phase='stopped' if requested else 'error',error=dict(type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc()))
    finally: save()
    return 1 if state['phase']=='error' else 0


if __name__=='__main__': sys.exit(main())
