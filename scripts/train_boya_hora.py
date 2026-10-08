# Reference: Hora v0.0.1 hora/algo/ppo/ppo.py (MIT), original PPO train_epoch.
# Boya interface, logging modes and atomic resumable learner state are port additions.
"""Train a nominal Hora teacher; cumulative budgets support declared cache-reset resume."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import subprocess
import time
import traceback
from tendonspin.rl.resource_guard import TrainingPulse,TrainingStopRequested,atomic_json

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--cache',type=Path,required=True)
parser.add_argument('--updates',type=int,default=16)
parser.add_argument('--total-actions',type=int,help='Cumulative target, rounded up to a complete PPO rollout')
parser.add_argument('--resume',type=Path,help='Paired learner checkpoint; simulation resets from the same cache')
parser.add_argument('--trace-mode',choices=('full','summary'),default='full')
parser.add_argument('--termination-profile',choices=('legacy_strict','hora_height','boya_workspace'),default='legacy_strict')
parser.add_argument('--num-envs',type=int,default=64)
parser.add_argument('--wall-s',type=float,default=240.,help='0 disables the optional wall limit')
parser.add_argument('--seed',type=int,default=43)
parser.add_argument('--save-every',type=int,default=4)
parser.add_argument('--minibatch-size',type=int,default=None)
parser.add_argument('--max-gpu-memory-mib',type=int,default=12288,help='0 disables GPU polling and stopping')
parser.add_argument('--stop-file',type=Path)
parser.add_argument('--heartbeat',type=Path)
parser.add_argument('--controller',default='hora_boya_nominal_teacher_v1')
parser.add_argument('--protocol',default='docs/experiments/2026-10-08-boya-hora-pilot/PROTOCOL.md')
args=parser.parse_args()
if args.num_envs<1 or args.save_every<1 or (args.total_actions is not None and args.total_actions<1):
    parser.error('Environment count, save cadence and total action target must be positive')
root=Path(__file__).resolve().parents[1]
out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
started=time.monotonic();pulse=TrainingPulse(args.stop_file,args.heartbeat)
pulse.check('launching')
stop_request={'signal':None}
def request_stop(signum,frame):stop_request['signal']=signum
signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
record=dict(controller=args.controller,seed=args.seed,requested_updates=args.updates,
    requested_total_actions=args.total_actions,requested_wall_s=args.wall_s,
    max_gpu_memory_mib=args.max_gpu_memory_mib,save_every=args.save_every,protocol=args.protocol,
    headless=True,enable_cameras=False,trace_mode=args.trace_mode,
    num_envs=args.num_envs,updates=[],actions_executed=0,session_actions_executed=0,physics_steps=0,
    completed_updates=0,completed_episodes=0,stop_reason=None,
    full_hora_reproduction=False,benchmark_validated=False,domain_randomization=False,
    cache_path=str(args.cache),cache_size=None,resume_from=str(args.resume) if args.resume else None,
    environment_resume='reset from declared cache' if args.resume else 'fresh seed43 initialization',
    lineage_id=str(out),episode_stop_counts={})
files=('scripts/train_boya_hora.py','tendonspin/rl/isaac_hora.py','tendonspin/baselines/hora_training.py',
       'tendonspin/physics/isaac_parallel.py','tendonspin/physics/isaac_boya.py',
       'tendonspin/physics/isaac_boya_sharpa.py','tendonspin/interfaces.py',
       'third_party/hora/hora/algo/ppo/ppo.py','third_party/hora/hora/algo/ppo/experience.py',
       'third_party/hora/hora/algo/models/models.py','third_party/hora/hora/algo/models/running_mean_std.py',
       'third_party/hora/hora/tasks/allegro_hand_hora.py','third_party/hora/configs/train/AllegroHandHora.yaml',
       'tendonspin/baselines/reference_models.py','tendonspin/evaluation/rotation.py',
       'tendonspin/physics/coordinates.py','scripts/evaluate_boya_hora.py',
       'tendonspin/rl/termination.py','assets/grasp/rotation_workspace.json','third_party/hora/configs/task/AllegroHandHora.yaml',
       'tendonspin/rl/checkpoint.py','tendonspin/rl/native_video.py','tendonspin/rl/resource_guard.py','scripts/guarded_boya_entry.py',args.protocol)
for name in files:
    source=root/name;content=source.read_bytes()
    snapshot=out/'sources'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
    record.setdefault('sources',[]).append(dict(path=name,sha256=hashlib.sha256(content).hexdigest()))
record['cache_sha256']=hashlib.sha256(args.cache.read_bytes()).hexdigest()

def save(status):
    record['status']=status;record['wall_s']=time.monotonic()-started
    atomic_json(out/'result.json',record)
    print('BOYA_HORA '+json.dumps(dict(status=status,updates=record['completed_updates'],
        actions=record['actions_executed'],wall_s=record['wall_s'],stop_reason=record['stop_reason'],
        actions_per_s=record.get('training_actions_per_wall_s'))),flush=True)

save('launching')
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False,device='cuda:0').app
failed=False;env=None;agent=None;update=0;base_actions=0;base_episodes=0;processed_episodes=0
base_reason_counts={}
try:
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from tendonspin.baselines.hora_training import load_ppo
    from tendonspin.rl.isaac_hora import HoraBoyaEnv
    from tendonspin.rl.checkpoint import save_checkpoint,restore_checkpoint
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    cfg=OmegaConf.create(dict(seed=args.seed,rl_device='cuda:0',test=False,checkpoint=None,
        task=dict(env=dict(numEnvs=args.num_envs)),
        train=OmegaConf.load(root/'third_party/hora/configs/train/AllegroHandHora.yaml')))
    cfg.train.ppo.priv_info=True
    cfg.train.ppo.minibatch_size=args.minibatch_size or args.num_envs*cfg.train.ppo.horizon_length
    batch=args.num_envs*cfg.train.ppo.horizon_length
    target_updates=math.ceil(args.total_actions/batch) if args.total_actions else args.updates
    cfg.train.ppo.max_agent_steps=target_updates*batch
    record.update(requested_updates=target_updates,rounded_target_actions=target_updates*batch)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)),out/'config.yaml')
    ppo_contract=OmegaConf.to_container(cfg.train.ppo,resolve=True)
    for key in ('max_agent_steps','save_frequency','save_best_after','output_name'):ppo_contract.pop(key,None)
    contract=dict(cache_sha256=record['cache_sha256'],num_envs=args.num_envs,
        ppo=ppo_contract,network=OmegaConf.to_container(cfg.train.network,resolve=True),
        sources={x['path']:x['sha256'] for x in record['sources'] if
                 x['path'].startswith(('tendonspin/physics/','tendonspin/baselines/','third_party/')) or
                 x['path'] in ('tendonspin/rl/isaac_hora.py','tendonspin/rl/termination.py','assets/grasp/rotation_workspace.json','tendonspin/interfaces.py','tendonspin/evaluation/rotation.py')})
    PPO=load_ppo();pulse.check('constructing scene')
    env=HoraBoyaEnv(root,out,args.cache,args.num_envs,trace_mode=args.trace_mode,
        termination_profile=args.termination_profile);env.stop_check=pulse.check
    record['termination']=env.termination.spec.to_dict()
    contract['termination']=record['termination']
    record.update(cache_size=len(env.cache['q']),joint_names=env.physics.names,
        body_names=list(env.physics.hand.body_names),filter_names=env.physics.filter_names,reason_codes=env.reasons)
    if args.trace_mode=='full':record['origins']=env.physics.origins.cpu().tolist()
    agent=PPO(env,str(out/'training'),cfg)
    if args.resume:
        previous=restore_checkpoint(agent,args.resume,contract=contract)
        base_actions=previous['actions_executed'];base_episodes=previous['completed_episodes']
        base_reason_counts=previous['episode_stop_counts']
        record.update(actions_executed=base_actions,completed_updates=previous['completed_updates'],
            completed_episodes=base_episodes,lineage_id=previous['lineage_id'],
            resumed_checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest())
        if agent.agent_steps!=base_actions or agent.epoch_num!=previous['completed_updates']:
            raise ValueError('Checkpoint is not a completed PPO-update boundary')
        if target_updates<=agent.epoch_num:raise ValueError('Cumulative target is already reached by checkpoint')
    agent.obs=env.reset();agent.save(str(out/'teacher_initial'))
    # AppLauncher installs its own exit handlers; restore cooperative saving only
    # after Isaac and the scene have initialized, before training can be interrupted.
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    record['manual_stop_handler']='request_stop, installed after Isaac initialization'
    save('training');training_started=time.monotonic()

    def collect_episode_summary():
        rows=env.completed if args.trace_mode=='summary' else env.completed[processed_episodes:]
        stats=dict(count=len(rows),stop_counts=dict(Counter(row['stop_reason'] for row in rows)))
        if rows:
            for key in ('net_deg','peak_deg','backward_deg','valid_prefix_s','episode_max_drift_mm',
                        'episode_max_tilt_deg','episode_max_normal_N','legacy_strict_net_deg','legacy_strict_valid_s'):
                a=np.asarray([row[key] for row in rows])
                stats[key]=dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()))
        return stats

    def checkpoint(name):
        path=out/(name+'.pth')
        progress={k:record[k] for k in ('actions_executed','completed_updates','completed_episodes',
                                        'episode_stop_counts','lineage_id')}
        save_checkpoint(agent,path,contract=contract,progress=progress)
        record['checkpoint']=str(path.relative_to(root))
        record['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        atomic_json(out/'latest_checkpoint.json',dict(path=str(path),sha256=record['checkpoint_sha256'],**progress))

    for update in range(agent.epoch_num+1,target_updates+1):
        if stop_request['signal'] is not None:
            record['stop_reason']='signal requested stop at update boundary';break
        if args.wall_s>0 and time.monotonic()-started>args.wall_s:
            record['stop_reason']='wall budget';break
        pulse.check('update boundary');env.begin_trace(agent.horizon_length);agent.epoch_num=update
        losses=agent.train_epoch()
        values={name:float(torch.stack(tensors).mean().detach()) for name,tensors in zip(
            ('actor_loss','critic_loss','bounds_loss','entropy','kl'),losses)}
        if not np.isfinite(list(values.values())).all():raise RuntimeError('Nonfinite PPO loss')
        agent.storage.data_dict=None
        stats=collect_episode_summary();processed_episodes+=stats['count']
        record['completed_episodes']=base_episodes+processed_episodes
        if args.trace_mode=='summary':env.completed.clear()
        env.flush_trace(out/f'rollout_{update:06d}.npz')
        record.update(actions_executed=base_actions+env.actions_executed,session_actions_executed=env.actions_executed,
            physics_steps=(base_actions+env.actions_executed)*env.physics.adapter.steps_per_control,
            session_physics_steps=env.physical_steps,completed_updates=update,
            training_actions_per_wall_s=env.actions_executed/(time.monotonic()-training_started),
            episode_stop_counts={k:base_reason_counts.get(k,0)+env.reason_counts.get(k,0) for k in env.reasons[1:]})
        record['updates'].append(dict(update=update,actions=record['actions_executed'],learning_rate=agent.last_lr,
            completed_episodes=record['completed_episodes'],training_episode_summary=stats,
            mean_recent_episode_reward=float(agent.episode_rewards.get_mean()),**values))
        agent.write_stats(*losses)
        agent.writer.add_scalar('episode_rewards/step',float(agent.episode_rewards.get_mean()),agent.agent_steps)
        agent.writer.add_scalar('episode_lengths/step',float(agent.episode_lengths.get_mean()),agent.agent_steps)
        agent.writer.add_scalar('performance/session_actions_per_s',record['training_actions_per_wall_s'],agent.agent_steps)
        for key,metrics in stats.items():
            if isinstance(metrics,dict) and 'mean' in metrics:
                agent.writer.add_scalar('training_episodes/'+key,metrics['mean'],agent.agent_steps)
                for quantile in ('median','p90','max'):
                    agent.writer.add_scalar('training_episodes/'+key+'/'+quantile,metrics[quantile],agent.agent_steps)
        if stats['count']:
            for reason in env.reasons[1:]:
                agent.writer.add_scalar('training_episodes/termination_fraction/'+reason,
                    stats['stop_counts'].get(reason,0)/stats['count'],agent.agent_steps)
        agent.writer.flush()
        if update%args.save_every==0 or update==target_updates or len(record['updates'])==1:
            checkpoint(f'teacher_u{update:06d}')
        if args.max_gpu_memory_mib>0 and (update==1 or update%8==0):
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=5)
            if gpu.returncode==0:
                used=int(gpu.stdout.strip().splitlines()[0]);record['observed_gpu_memory_used_MiB']=used
                record['max_observed_gpu_memory_used_MiB']=max(used,record.get('max_observed_gpu_memory_used_MiB',0))
                if used>args.max_gpu_memory_mib:record['stop_reason']='device memory headroom limit'
        save('update completed')
        if record['stop_reason']:break
    if record['stop_reason'] is None:record['stop_reason']='update budget'
    checkpoint('teacher_final')
    initial=torch.load(out/'teacher_initial.pth',weights_only=False)['model']
    record['model_parameters_changed']=any(not torch.equal(v,initial[k]) for k,v in agent.model.state_dict().items())
except TrainingStopRequested as error:
    record['stop_reason']='resource guard requested stop';record['guard_reason']=str(error)
    if agent is not None:
        # Preserve diagnostic weights; only the last complete-update paired snapshot resumes.
        agent.save(str(out/'teacher_guard_stop'))
        record['diagnostic_checkpoint']=str((out/'teacher_guard_stop.pth').relative_to(root))
except BaseException as error:
    failed=True;record['stop_reason']='process error'
    record['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
finally:
    if env is not None:
        record.update(actions_executed=base_actions+env.actions_executed,session_actions_executed=env.actions_executed,
            physics_steps=(base_actions+env.actions_executed)*env.physics.adapter.steps_per_control,
            session_physics_steps=env.physical_steps,active_training_episodes_at_stop=env.num_envs)
        if args.trace_mode=='full':atomic_json(out/'training_episodes.json',env.completed)
        elif env.completed:record['unoptimized_rollout_completed_episodes']=len(env.completed)
        if env.trace is not None:env.flush_trace(out/f'interrupted_rollout_{update:06d}.npz')
    if agent is not None:agent.writer.close()
    save('error' if failed else 'completed');app.close(exit_code=1 if failed else 0)
