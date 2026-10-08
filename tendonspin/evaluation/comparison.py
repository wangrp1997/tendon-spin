# References: existing TendonSpin evaluate_boya_hora.py / score_prefix;
# Hora v0.0.1 actor/normalizer and height termination (MIT).
# New code only prepares provenance, validates a matched comparison and reports it.
"""Compare preserved teachers under one declared runtime without changing training."""
import ast
import csv
import hashlib
import json
from pathlib import Path

from tendonspin.rl.resource_guard import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def source_map(record):
    return {s['path']: s['sha256'] for s in record['sources']}


def verify_sources(root, sources):
    for item in sources:
        if digest(Path(root) / item['path']) != item['sha256']:
            raise ValueError('Runtime source changed: ' + item['path'])


def function_ast(path, name, class_name=None):
    tree = ast.parse(Path(path).read_text())
    body = tree.body
    if class_name:
        body = next(n.body for n in body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node, include_attributes=False)


def check_policy_compatibility(old, new, old_training, new_training):
    """Allow the declared termination/logging port, never an observation/model change."""
    a, b = source_map(old), source_map(new)
    prefixes = ('tendonspin/physics/', 'third_party/hora/hora/algo/',
                'tendonspin/interfaces.py', 'tendonspin/baselines/reference_models.py',
                'tendonspin/evaluation/rotation.py', 'third_party/hora/hora/tasks/')
    required = [p for p in a if p.startswith(prefixes)]
    if not required:
        raise ValueError('Missing physics/policy source identity')
    for path in required:
        if a[path] != b.get(path):
            raise ValueError('Policy/physics incompatibility: ' + path)
    for data, directory in ((old, old_training), (new, new_training)):
        verify_sources(Path(directory) / 'sources', data['sources'])
    env = 'sources/tendonspin/rl/isaac_hora.py'
    for name, cls in (('_frame', 'HoraBoyaEnv'), ('observe', 'HoraBoyaEnv'), ('spin_increment', None)):
        if function_ast(Path(old_training) / env, name, cls) != function_ast(Path(new_training) / env, name, cls):
            raise ValueError('Observation/scoring function differs: ' + name)
    if old['cache_sha256'] != new['cache_sha256']:
        raise ValueError('Training cache differs')
    if old['rounded_target_actions'] != new['rounded_target_actions']:
        raise ValueError('Different planned sample budgets')
    return dict(shared_core_paths=required, identical_observation_functions=['_frame', 'observe'],
                identical_scoring_function='spin_increment',
                changed_sources=[p for p in a.keys() & b.keys() if a[p] != b[p]])


def prepare(root, out, old_run, new_run, protocol):
    root, out, old_run, new_run = map(lambda p: Path(p).resolve(), (root, out, old_run, new_run))
    old_training, new_training = old_run / 'training', new_run / 'training'
    old, new = [read_json(p / 'result.json') for p in (old_training, new_training)]
    if old['status'] != 'completed' or old['stop_reason'] != 'update budget':
        raise ValueError('Old training did not normally complete its budget')
    compatibility = check_policy_compatibility(old, new, old_training, new_training)
    verify_sources(root, new['sources'])
    if new['termination']['profile'] != 'hora_height':
        raise ValueError('Expected the already-declared new height profile')
    old_checkpoint = root / old['checkpoint']
    if digest(old_checkpoint) != old['checkpoint_sha256']:
        raise ValueError('Preserved old checkpoint hash differs')
    extra_paths = [protocol, 'scripts/compare_boya_hora.py', 'tendonspin/evaluation/comparison.py',
                   'docs/data/boya_native_contract.json', 'docs/data/isaac_boya_import_phases.json',
                   'assets/grasp/scene.xml']
    imported = read_json(root / 'docs/data/isaac_boya_import_phases.json')
    extra_paths.append(imported['usd']['path'])
    pins = [dict(path=p, sha256=digest(root / p)) for p in extra_paths]
    plan = dict(schema='tendonspin-matched-teacher-v1', root=str(root), protocol=protocol,
        old_run=str(old_run), new_run=str(new_run), old_checkpoint=str(old_checkpoint),
        old_checkpoint_sha256=old['checkpoint_sha256'], old_training_record_sha256=digest(old_training / 'result.json'),
        new_checkpoint=str(new_training / 'teacher_final.pth'),
        expected_training_actions=old['actions_executed'], requested_s=120., wall_s=1500., seed=43,
        termination=new['termination'], runtime_sources=new['sources'], comparison_pins=pins,
        compatibility=compatibility, old_training_controller=old['controller'], new_training_controller=new['controller'],
        old_execution_controller='matched_height_old_weights__' + old['controller'],
        initial_state='original grasp44, not cache',
        reuse_new_evaluation=str(new_run / 'evaluation'),
        observation='privileged teacher: 96 proprio + 9 privileged, original actor and its own normalizer',
        scoring='signed moving-cylinder-axis prefix; 120s main and 30s supplementary',
        extra_gpu_episodes=1, multiple_seed_benchmark=False)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / 'plan.json', plan)
    atomic_json(out / 'status.json', dict(phase='prepared', training_actions=0, new_physics_steps=0))
    (out / 'comparison.md').write_text(
        '# 同条件教师对照（待执行）\n\n'
        '同一原始grasp44、同一hora_height规则、同一120s预算；主指标为有效前缀净转角。\n'
        '等待新训练及原定最终评估结束，再补跑旧权重。当前没有同条件对照成绩。\n\n'
        '| 权重 | 训练动作数 | 同条件评估 |\n|---|---:|---|\n'
        f'| 旧严格规则教师 | {old["actions_executed"]} | 待运行 |\n'
        '| 新高度规则教师 | 训练中 | 等待原定最终评估 |\n\n'
        '历史旧规则的63.626313°不填入本表；单初态对照不证明SOTA或统计显著性。\n')
    return plan


def verify_checkpoint(root, training, checkpoint, expected_actions):
    import torch
    checkpoint = Path(checkpoint)
    if digest(checkpoint) != training['checkpoint_sha256']:
        raise ValueError('Checkpoint does not match its training record')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    progress = state.get('progress', {})
    if state.get('schema') != 'tendonspin-hora-resume-v1' or progress.get('actions_executed') != expected_actions:
        raise ValueError('Checkpoint sample count/schema differs')
    if state.get('agent_steps') != expected_actions or training['actions_executed'] != expected_actions:
        raise ValueError('Training/checkpoint sample count differs')
    if progress.get('lineage_id') != training['lineage_id']:
        raise ValueError('Checkpoint lineage differs')
    if not state.get('model') or not state.get('running_mean_std'):
        raise ValueError('Missing model or its own observation normalizer')
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                actions=expected_actions, lineage_id=progress['lineage_id'])


def make_old_execution_contract(plan, old_training):
    """Explicit inference migration descriptor, NOT a rewritten training record."""
    return dict(record_type='evaluation_execution_contract_not_training_record',
        controller=plan['old_execution_controller'], actions_executed=plan['expected_training_actions'],
        sources=plan['runtime_sources'], termination=plan['termination'],
        training_origin=dict(record=str(Path(plan['old_run']) / 'training/result.json'),
            record_sha256=plan['old_training_record_sha256'], controller=old_training['controller'],
            original_termination=old_training.get('termination', {'profile': 'legacy_strict'}),
            checkpoint_sha256=plan['old_checkpoint_sha256'], sources=old_training['sources']),
        migration='Old preserved policy+own normalizer; new common height evaluation rule, declared before execution')


def validate_result(record, plan, checkpoint_sha256, controller):
    expected = dict(status='completed', requested_s=plan['requested_s'], seed=plan['seed'],
                    episode_resets=0, controller_switches=0, training_actions=0,
                    initial_state=plan['initial_state'], headless=True, enable_cameras=False,
                    checkpoint_sha256=checkpoint_sha256, termination=plan['termination'],
                    training_actions_in_checkpoint=plan['expected_training_actions'], controller=controller)
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError('Evaluation contract differs: ' + key)
    if source_map(record) != {s['path']: s['sha256'] for s in plan['runtime_sources']}:
        raise ValueError('Evaluation runtime differs')
    if record.get('stop_reason') == 'process error' or not record.get('metrics'):
        raise ValueError('Evaluation has no completed physical record')


def compare_initial_states(old_dir, new_dir):
    import numpy as np
    # Initial states are measured independently in two processes; no state transfer.
    with np.load(Path(old_dir) / 'initial_state.npz') as a, np.load(Path(new_dir) / 'initial_state.npz') as b:
        errors = {}
        for key in ('joint_pos', 'joint_vel', 'object_state', 'commands'):
            np.testing.assert_allclose(a[key], b[key], rtol=0., atol=1e-6,
                                       err_msg='Initial physical state differs: ' + key)
            errors[key] = float(np.max(np.abs(a[key] - b[key])))
    return dict(max_abs_difference=errors, atol=1e-6, rtol=0., independently_initialized=True)


def diagnostics(directory, windows=(30., 120.)):
    import numpy as np
    maxima = {str(w):{k:None for k in ('drift_mm', 'tilt_deg', 'max_normal')} for w in windows}
    for file in sorted(Path(directory).glob('physics_*.npz')):
        with np.load(file) as trace:
            for window in windows:
                valid = trace['valid'].astype(bool) & (trace['elapsed_s'] <= window)
                if not valid.any():
                    continue
                for key in maxima[str(window)]:
                    value = float(np.max(trace[key][valid]))
                    prior = maxima[str(window)][key]
                    maxima[str(window)][key] = value if prior is None else max(prior, value)
    return maxima


def write_report(out, plan, old, new, initial_agreement):
    out = Path(out)
    rows = []
    for label, result, directory in (('old_weights_new_criteria', old, out / 'old_evaluation'),
                                     ('new_weights_new_criteria', new, Path(plan['reuse_new_evaluation']))):
        window_maxima = diagnostics(directory)
        for window in (30., 120.):
            maxima = window_maxima[str(window)]
            metric = result['metrics'][str(window)]
            rows.append(dict(policy=label, window_s=window, net_deg=metric['net_deg'],
                valid_s=metric['valid_seconds'], peak_deg=metric['peak_deg'], backward_deg=metric['backward_deg'],
                final_30s_net_deg=metric['full_window_final_tail_net_deg'],
                complete_window=metric['complete_window_observed'],
                stop_reason='window completed' if metric['complete_window_observed'] else result['stop_reason'],
                source_episode_stop_reason=result['stop_reason'],
                max_drift_mm=maxima['drift_mm'], max_tilt_deg=maxima['tilt_deg'], max_normal_N=maxima['max_normal'],
                checkpoint_sha256=result['checkpoint_sha256'], result_path=str(directory / 'result.json')))
    payload=dict(status='completed', primary_window_s=120., rows=rows, initial_agreement=initial_agreement,
                 independent_episodes=2, reused_new_episode=True, additional_old_episode=True,
                 multiple_seed_benchmark=False, source_plan=str(out / 'plan.json'))
    atomic_json(out / 'comparison.json', payload)
    with (out / 'comparison.csv').open('w', newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 同条件教师对照', '', '相同原始grasp44、hora_height终止、120s预算、seed43；各自单独初始化，无重置或控制器切换。', '',
           '| 权重 | 窗口s | 净角° | 有效s | 峰角° | 倒转° | 停止原因 |', '|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f'| {r["policy"]} | {r["window_s"]:.0f} | {r["net_deg"]:.3f} | {r["valid_s"]:.4f} | {r["peak_deg"]:.3f} | {r["backward_deg"]:.3f} | {r["stop_reason"]} |')
    lines += ['', '新权重行复用其原定独立评估，旧权重行是新的同条件执行；不是旧轨迹重计分。',
              '位移、倾斜、力极值与末30s推进见JSON/CSV。新旧窗口均包含所有早停，不拼接角度。',
              '原旧规则63.626313°仅保留在历史记录，不混入本表。单初态对照不证明SOTA或统计显著性。', '']
    (out / 'comparison.md').write_text('\n'.join(lines))
    return payload


def dependency_completed(stage):
    """Never launch a new GPU evaluation after an interrupted/failed training run."""
    if stage['phase'] in ('error', 'stopped'):
        raise ValueError('Dependency stopped; do not launch comparison')
    if stage['phase'] != 'completed':
        return False
    for name in ('training', 'evaluation'):
        item=stage.get(name)
        if not isinstance(item,dict) or item.get('exit_code')!=0 or item.get('status')!='completed':
            raise ValueError('Incomplete dependency: '+name)
    if stage['training'].get('stop_reason')!='update budget':
        raise ValueError('Training budget did not complete normally')
    return True
