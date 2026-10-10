# Sources: TendonSpin checkpoint.py and train_boya_hora.py, commit0bdb2d3;
# github.com/wangrp1997/tendon-spin. Hora v0.0.1 paired PPO state (MIT),
# github.com/HaozhiQi/hora, arXiv:2210.04887. No upstream algorithm edits.
# Port: one explicit, hash-pinned legacy20M -> selected pose-reward variant branch.
"""Copy the declared learner intact, updating only its contract and branch provenance."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from tendonspin.rl.checkpoint import SCHEMA
from tendonspin.rl.resource_guard import atomic_json
from tendonspin.rl.rotation_reward import configuration as reward_configuration, POSE_DELTA, POSE_DROP, REPORTED
from tendonspin.rl.termination import make_termination_spec
from tendonspin.physics.solver_profiles import configuration as engine_configuration, ORIGINAL

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA256 = '9781442fbfe0865946b7d3268a1ad379f796ca967aa8997f6dc427b8263f226e'
UPDATED_PATHS = {'tendonspin/rl/isaac_hora.py', 'tendonspin/physics/isaac_parallel.py'}
ADDED_PATHS = {'tendonspin/rl/rotation_reward.py', 'tendonspin/physics/solver_profiles.py'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_equal(left, right):
    """Verify serialized learner tensors, numpy RNG and scalar state exactly."""
    require(type(left) is type(right), 'Serialized learner type changed')
    if isinstance(left, torch.Tensor):
        require(torch.equal(left, right), 'Serialized learner tensor changed')
    elif isinstance(left, np.ndarray):
        require(np.array_equal(left, right), 'Serialized numpy RNG changed')
    elif isinstance(left, dict):
        require(left.keys() == right.keys(), 'Serialized learner keys changed')
        for key in left:
            require_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        require(len(left) == len(right), 'Serialized learner sequence changed')
        for a, b in zip(left, right):
            require_equal(a, b)
    else:
        require(left == right, 'Serialized learner scalar changed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-record', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lineage-id', required=True)
    parser.add_argument('--reward-profile', choices=(POSE_DELTA, POSE_DROP), default=POSE_DELTA)
    args = parser.parse_args()
    require(not args.out.exists(), 'Refuse to overwrite a checkpoint')
    plan = json.loads(args.plan.read_text())
    require(plan.get('branch_reward_profile',POSE_DELTA)==args.reward_profile,
            'Selected reward differs from the explicit migration plan')
    require(digest(args.checkpoint) == PARENT_SHA256 == plan['parent_checkpoint_sha256'],
            'Only the explicitly declared20M parent is authorized for this migration')
    require(digest(args.source_record) == plan['source_record_sha256'], 'Parent record changed')
    require(set(plan['updated_sources']) == UPDATED_PATHS, 'Undeclared source replacement')
    require(set(plan['added_sources']) == ADDED_PATHS, 'Undeclared source addition')
    record = json.loads(args.source_record.read_text())
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    require(state['schema'] == SCHEMA, 'Not a paired full learner checkpoint')
    require(record['status'] == 'completed' and record['stop_reason'] == 'update budget',
            'Parent did not complete its declared budget')
    require(record['checkpoint_sha256'] == PARENT_SHA256, 'Record does not identify parent checkpoint')
    require(record.get('engine_profile', ORIGINAL) == ORIGINAL and
            record.get('reward_profile', REPORTED) == REPORTED, 'Wrong parent engine/reward')
    original = state['contract']
    require(set(original) == {'cache_sha256', 'num_envs', 'ppo', 'network', 'sources', 'termination'},
            'Unexpected legacy contract schema; do not silently remigrate')
    require(original['num_envs'] == 1024 and record['seed'] == 43, 'Wrong learner population/seed')
    require(state['agent_steps'] == state['progress']['actions_executed'] == record['actions_executed'] == 20004864,
            'Parent action counts differ')
    require(state['epoch_num'] == state['progress']['completed_updates'] == record['completed_updates'] == 2442,
            'Parent is not the declared complete-update boundary')
    for source in record['sources']:
        require(digest(args.source_record.parent/'sources'/source['path']) == source['sha256'],
                'Archived parent source changed: '+source['path'])
    recorded_sources = {row['path']: row['sha256'] for row in record['sources']}
    require(all(recorded_sources.get(path) == sha for path, sha in original['sources'].items()),
            'Parent source contract differs from its archived record')

    target = deepcopy(original)
    for path, old_hash in original['sources'].items():
        current_hash = digest(ROOT/path)
        if path in UPDATED_PATHS:
            replacement = plan['updated_sources'][path]
            require(old_hash == replacement['before'] and current_hash == replacement['after'],
                    'Source does not match reviewed before/after identity: '+path)
        else:
            require(current_hash == old_hash, 'Unrelated source changed: '+path)
        target['sources'][path] = current_hash
    for path, sha in plan['added_sources'].items():
        require(digest(ROOT/path) == sha, 'Added source does not match plan: '+path)
        target['sources'][path] = sha
    cache = ROOT/'outputs/boya_settled_cache_v2/grasp_cache.npz'
    require(digest(cache) == original['cache_sha256'] == record['cache_sha256'], 'Grasp cache changed')
    native = json.loads((ROOT/'docs/data/boya_native_contract.json').read_text())
    require(make_termination_spec(ROOT, 'boya_workspace', native['object']['pos'][2]).to_dict() ==
            original['termination'] == record['termination'], 'Task termination changed')
    cfg = OmegaConf.create(dict(seed=43, rl_device='cuda:0', test=False, checkpoint=None,
        task=dict(env=dict(numEnvs=1024)),
        train=OmegaConf.load(ROOT/'third_party/hora/configs/train/AllegroHandHora.yaml')))
    cfg.train.ppo.priv_info = True
    cfg.train.ppo.minibatch_size = 512
    ppo = OmegaConf.to_container(cfg.train.ppo, resolve=True)
    for key in ('max_agent_steps', 'save_frequency', 'save_best_after', 'output_name'):
        ppo.pop(key, None)
    require(ppo == original['ppo'] and
            OmegaConf.to_container(cfg.train.network, resolve=True) == original['network'],
            'PPO/network configuration changed')
    target['engine_configuration'] = engine_configuration(ORIGINAL)
    target['reward_configuration'] = reward_configuration(args.reward_profile, .0005, 100)
    migration = dict(schema='tendonspin-explicit-reward-branch-v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        parent_checkpoint=str(args.checkpoint.resolve()), parent_checkpoint_sha256=PARENT_SHA256,
        source_record=str(args.source_record.resolve()), source_record_sha256=digest(args.source_record),
        plan=str(args.plan), plan_sha256=digest(args.plan),
        parent_lineage_id=state['progress']['lineage_id'], branch_lineage_id=args.lineage_id,
        parent_reward_profile=REPORTED, branch_reward_profile=args.reward_profile,
        target_reward_configuration=target['reward_configuration'],
        reward_start_actions=state['agent_steps'], reward_start_update=state['epoch_num'],
        pose_reward_training_actions_at_migration=0,
        source_contract_sha256=json_digest(original), target_contract_sha256=json_digest(target),
        updated_sources=plan['updated_sources'], added_sources=plan['added_sources'],
        engine_change=False, task_change=False, ppo_change=False,
        learner_fields_preserved=[key for key in state if key not in ('contract', 'progress')],
        environment_resume='reset from the same28-state cache; unfinished episodes/rollout are discarded',
        critic_and_adam='preserved from old objective; must adapt to changed reward, no reset or LR change')
    migrated = dict(state, contract=target, progress={**state['progress'],
        'lineage_id': args.lineage_id, 'reward_branch_migration': migration})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name+'.tmp')
    torch.save(migrated, temporary)
    temporary.replace(args.out)
    loaded = torch.load(args.out, map_location='cpu', weights_only=False)
    for key in migration['learner_fields_preserved']:
        require_equal(state[key], loaded[key])
    for key in state['progress']:
        if key != 'lineage_id':
            require_equal(state['progress'][key], loaded['progress'][key])
    require_equal(migrated['contract'], loaded['contract'])
    require_equal(migrated['progress'], loaded['progress'])
    require(digest(args.checkpoint) == PARENT_SHA256, 'Original checkpoint was modified')
    atomic_json(args.out.with_suffix('.migration.json'), dict(status='completed', **migration,
        migrated_checkpoint=str(args.out.resolve()), migrated_checkpoint_sha256=digest(args.out),
        serialized_learner_equality=True, original_checkpoint_unchanged=True,
        new_physics_steps=0, new_training_actions=0))
    print(json.dumps(dict(status='completed', checkpoint=str(args.out),
        actions=state['agent_steps'], updates=state['epoch_num'], preserved_learner_state=True)), flush=True)


if __name__ == '__main__':
    main()
