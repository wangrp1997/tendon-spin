# Reference: Hora v0.0.1 PPO.save/restore_train (MIT), torch optimizer/RNG APIs.
# Extends the original weights-only checkpoint with complete learner state.
# Simulation is deliberately reset on resume; no claim of bitwise physics replay.
"""Atomic, paired policy/optimizer snapshots for cumulative Hora training."""
from pathlib import Path
import random
import numpy as np
import torch

SCHEMA='tendonspin-hora-resume-v1'


def save_checkpoint(agent,path,*,contract,progress):
    path=Path(path)
    state=dict(schema=SCHEMA,model=agent.model.state_dict(),
        running_mean_std=agent.running_mean_std.state_dict(),
        value_mean_std=agent.value_mean_std.state_dict(),
        optimizer=agent.optimizer.state_dict(),learning_rate=agent.last_lr,
        scheduler=vars(agent.scheduler).copy(),agent_steps=agent.agent_steps,
        epoch_num=agent.epoch_num,data_collect_time=agent.data_collect_time,
        rl_train_time=agent.rl_train_time,best_rewards=agent.best_rewards,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        contract=contract,progress=progress,
        resume_environment='reset from the declared grasp cache; discard unfinished episodes and rollout')
    temporary=path.with_name(path.name+'.tmp')
    torch.save(state,temporary)
    temporary.replace(path)


def restore_checkpoint(agent,path,*,contract):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('schema')!=SCHEMA:raise ValueError('Checkpoint lacks paired learner state; not a resume checkpoint')
    if state['contract']!=contract:raise ValueError('Resume task/PPO/source contract mismatch; explicit migration required')
    agent.model.load_state_dict(state['model'])
    agent.running_mean_std.load_state_dict(state['running_mean_std'])
    agent.value_mean_std.load_state_dict(state['value_mean_std'])
    agent.optimizer.load_state_dict(state['optimizer'])
    agent.last_lr=state['learning_rate']
    for group in agent.optimizer.param_groups:group['lr']=agent.last_lr
    agent.scheduler.__dict__.update(state['scheduler'])
    for name in ('agent_steps','epoch_num','data_collect_time','rl_train_time','best_rewards'):
        setattr(agent,name,state[name])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
    np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    # The caller performs one declared environment reset before collecting anew.
    agent.current_rewards.zero_();agent.current_lengths.zero_();agent.dones.fill_(1)
    agent.storage.data_dict=None
    return state['progress']
