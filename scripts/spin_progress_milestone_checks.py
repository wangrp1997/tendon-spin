"""C fresh/own-learner and frozen-evaluation checks at the selected batch size."""
import hashlib
from pathlib import Path
from tendonspin.evaluation.comparison import digest, read_json, validate_result, verify_checkpoint, verify_sources
from tendonspin.rl.spin_progress_reward import PROFILE
ROOT=Path(__file__).resolve().parents[1]


def initial_identity(run, num_envs):
    """Validate fresh Adam updates at the actual batch and hash initial tensors."""
    batch=num_envs*8
    adam_steps=(batch//512)*5
    import torch
    initial = torch.load(run/'training/teacher_initial.pth', map_location='cpu', weights_only=False)
    first = torch.load(run/'training/teacher_u000001.pth', map_location='cpu', weights_only=False)
    if (initial.get('arm'),initial.get('reward_profile')) != ('C',PROFILE):
        raise ValueError('Initial policy does not identify the native C reward')
    if (initial['progress']['actions_executed'],initial['progress']['completed_updates'],initial.get('resume_source')) != (0,0,None):
        raise ValueError('Initial policy did not begin from zero without a resume source')
    if initial['reward_configuration'] != first['contract']['reward_configuration'] or first['contract'].get('arm') != 'C':
        raise ValueError('Initial policy and first full learner differ in C reward identity')
    if first['agent_steps'] != batch or first['epoch_num'] != 1:
        raise ValueError('Fresh arm did not start at zero actions/updates')
    steps = {float(v['step']) for v in first['optimizer']['state'].values()}
    if steps != {float(adam_steps)} or 'reward_branch_migration' in first['progress']:
        raise ValueError('Fresh arm inherited optimizer state or reward ancestry')
    hashes = {}
    for name in ('model','running_mean_std','value_mean_std'):
        state = initial[name]
        h = hashlib.sha256()
        for key, value in sorted(state.items()):
            value = value.detach().cpu().contiguous()
            h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
            h.update(value.numpy().tobytes())
        hashes[name] = h.hexdigest()
    return dict(arm='C',reward_profile=PROFILE,seed=43, fresh_actions_before_first_update=0, fresh_updates_before_first_update=0,
        first_checkpoint_actions=batch, first_checkpoint_update=1, first_adam_steps_per_parameter=adam_steps,
        initial_state_tensor_hashes=hashes, first_checkpoint_sha256=digest(run/'training/teacher_u000001.pth'))


def evaluation_row(run, target, previous, num_envs):
    import numpy as np
    from tendonspin.evaluation.rotation import score_prefix
    training = read_json(run/'training/result.json')
    verify_sources(ROOT, training['sources'])
    identity = verify_checkpoint(ROOT, training, ROOT/training['checkpoint'], target['rounded_actions'])
    if training['seed'] != 43 or training['num_envs'] != num_envs or training['reward_profile'] != PROFILE or training.get('arm') != 'C':
        raise ValueError('Fresh arm configuration differs from plan')
    if previous is None:
        if training['resume_from'] is not None or training['session_actions_executed'] != target['rounded_actions']:
            raise ValueError('First milestone is not a fresh learner')
    elif (training.get('resumed_checkpoint_sha256') != previous['checkpoint_sha256'] or
          training['lineage_id'] != previous['lineage_id'] or
          training['session_actions_executed'] != target['rounded_actions']-previous['actions_executed']):
        raise ValueError('Continuation did not preserve its own arm and cumulative counts')
    result = read_json(run/'evaluation/result.json')
    initial_state=training.get('scene_configuration',{}).get('evaluation_initial_state','original grasp44, not cache')
    validate_result(result, dict(requested_s=30., seed=43, initial_state=initial_state,
        termination=training['termination'], expected_training_actions=target['rounded_actions'],
        runtime_sources=training['sources']), identity['checkpoint_sha256'], 'frozen_'+training['controller'])
    limits=training.get('scene_configuration',{}).get('initial_state_atol',
        {key:0. for key in result['initial_state_max_errors']})
    if any(value>limits[key] for key,value in result['initial_state_max_errors'].items()):
        raise ValueError('Frozen evaluation original state differs from reference')
    analysis = read_json(run/'analysis/result.json')
    metric = result['metrics']['30.0']
    tail = score_prefix(np.load(run/'evaluation/angles_deg.npy'), .0005, result['valid_steps'],
        window_s=30., tail_s=10.)['full_window_final_tail_net_deg']
    return dict(arm='C', reward_profile=PROFILE, requested_actions=target['requested_actions'],
        training_actions=target['rounded_actions'], actual_s=result['actual_s'], valid_s=metric['valid_seconds'],
        net_deg=metric['net_deg'], peak_deg=metric['peak_deg'], backward_deg=metric['backward_deg'],
        final_10s_net_deg=tail, complete_window=metric['complete_window_observed'],
        stop_reason=result['stop_reason'], **analysis['valid_prefix_maxima'],
        checkpoint=identity['checkpoint'], checkpoint_sha256=identity['checkpoint_sha256'],
        evaluation=str(run/'evaluation/result.json'),
        initial_state_max_error=max(result['initial_state_max_errors'].values())), training

