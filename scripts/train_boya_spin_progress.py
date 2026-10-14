# Derived from train_boya_hora.py at8cca30e (MIT), same pinned Hora PPO.
# Separate randomized task/property contract; historical entrypoint is unchanged.
# Reference: Hora v0.0.1 hora/algo/ppo/ppo.py (MIT), original PPO train_epoch.
# Boya interface, logging modes and atomic resumable learner state are port additions.
# Explicit solver profile is part of the recorded learner-resume contract.
"""Train a randomized C self-spin teacher with the unchanged Hora PPO learner."""
import argparse
from collections import Counter
import hashlib
import json
import importlib.metadata
import math
from pathlib import Path
import random
import resource
import signal
import subprocess
import time
import traceback
from tendonspin.rl.resource_guard import TrainingPulse,TrainingStopRequested,atomic_json
from tendonspin.rl.spin_progress_reward import PROFILE
REWARD_PROFILES = (PROFILE,)

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--cache',type=Path,required=True)
parser.add_argument('--randomization-config',type=Path,required=True)
parser.add_argument('--scene-config',type=Path,help='Optional frozen geometry, nominal grasp and rotation reference')
parser.add_argument('--updates',type=int,default=16)
parser.add_argument('--total-actions',type=int,help='Cumulative target, rounded up to a complete PPO rollout')
parser.add_argument('--resume',type=Path,help='Paired learner checkpoint; simulation resets from the same cache')
parser.add_argument('--trace-mode',choices=('full','summary'),default='summary')
parser.add_argument('--termination-profile',choices=('legacy_strict','hora_height','boya_workspace'),default='boya_workspace')
parser.add_argument('--engine-profile',choices=('original_tgs16_4','tgs16_velocity0'),default='original_tgs16_4')
parser.add_argument('--reward-profile',choices=REWARD_PROFILES,default=PROFILE)
parser.add_argument('--num-envs',type=int,default=1024)
parser.add_argument('--wall-s',type=float,default=0.,help='0 disables the optional wall limit')
parser.add_argument('--seed',type=int,default=43)
parser.add_argument('--save-every',type=int,default=4)
parser.add_argument('--minibatch-size',type=int,default=512)
parser.add_argument('--max-gpu-memory-mib',type=int,default=0,help='0 disables GPU polling and stopping')
parser.add_argument('--stop-file',type=Path)
parser.add_argument('--heartbeat',type=Path)
parser.add_argument('--controller',default='boya_ninescale_dr_cache164_c_fresh_v1')
parser.add_argument('--protocol',default='SERVER_TRAINING_HANDOFF.md')
args=parser.parse_args()
if args.seed != 43 or args.minibatch_size != 512 or args.engine_profile != 'original_tgs16_4' or args.termination_profile != 'boya_workspace':
    parser.error('C handoff fixes seed43/minibatch512/original_tgs16_4/boya_workspace')
if args.num_envs<1 or args.num_envs*8 % 512 or args.save_every<1 or (args.total_actions is not None and args.total_actions<1):
    parser.error('Environment count, save cadence and total action target must be positive')
root=Path(__file__).resolve().parents[1]
out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
started=time.monotonic();pulse=TrainingPulse(args.stop_file,args.heartbeat)
pulse.check('launching')
stop_request={'signal':None}
def request_stop(signum,frame):stop_request['signal']=signum
signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
record=dict(arm='C',controller=args.controller,seed=args.seed,requested_updates=args.updates,
    requested_total_actions=args.total_actions,requested_wall_s=args.wall_s,
    max_gpu_memory_mib=args.max_gpu_memory_mib,save_every=args.save_every,protocol=args.protocol,
    headless=True,enable_cameras=False,trace_mode=args.trace_mode,engine_profile=args.engine_profile,
    reward_profile=args.reward_profile,
    num_envs=args.num_envs,updates=[],actions_executed=0,session_actions_executed=0,physics_steps=0,
    completed_updates=0,completed_episodes=0,stop_reason=None,
    full_hora_reproduction=False,benchmark_validated=False,domain_randomization=True,
    cache_path=str(args.cache),cache_size=None,resume_from=str(args.resume) if args.resume else None,
    environment_resume='reset from declared cache' if args.resume else 'fresh seed43 initialization',
    lineage_id=str(out),episode_stop_counts={})
record['runtime_versions']={name:importlib.metadata.version(name) for name in
    ('isaaclab','isaacsim','torch','warp-lang','numpy','scipy')}
record['code_sha']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
files=('scripts/train_boya_spin_progress.py','scripts/evaluate_boya_spin_progress.py',
       'tendonspin/rl/isaac_spin_progress.py','tendonspin/rl/spin_progress_reward.py',
       'assets/config/boya_spin_progress_reward.json',
       'tendonspin/rl/domain_randomization.py','tendonspin/rl/isaac_hora_randomized.py',
       'tendonspin/physics/isaac_parallel_randomized.py',
       str(args.randomization_config.resolve().relative_to(root)),
       'docs/data/boya_native_contract.json','docs/data/isaac_boya_import_phases.json',
       'scripts/train_boya_hora.py','tendonspin/rl/isaac_hora.py','tendonspin/baselines/hora_training.py',
       'tendonspin/physics/isaac_parallel.py','tendonspin/physics/isaac_boya.py',
       'tendonspin/physics/isaac_boya_sharpa.py','tendonspin/interfaces.py',
       'third_party/hora/hora/algo/ppo/ppo.py','third_party/hora/hora/algo/ppo/experience.py',
       'third_party/hora/hora/algo/models/models.py','third_party/hora/hora/algo/models/running_mean_std.py',
       'third_party/hora/hora/tasks/allegro_hand_hora.py','third_party/hora/configs/train/AllegroHandHora.yaml',
       'tendonspin/baselines/reference_models.py','tendonspin/evaluation/rotation.py',
       'tendonspin/physics/coordinates.py','scripts/evaluate_boya_hora.py',
       'tendonspin/rl/termination.py','assets/grasp/rotation_workspace.json','third_party/hora/configs/task/AllegroHandHora.yaml',
       'tendonspin/rl/checkpoint.py','tendonspin/rl/native_video.py','tendonspin/rl/resource_guard.py','scripts/guarded_boya_entry.py',
       'tendonspin/physics/solver_profiles.py','tendonspin/rl/rotation_reward.py',
       'scripts/migrate_boya_pose_reward.py',args.protocol)
if args.scene_config:
    files+=('tendonspin/physics/learning_scene.py','tendonspin/physics/object_geometry.py',
            str(args.scene_config.resolve().relative_to(root)))
imported_identity=json.loads((root/'docs/data/isaac_boya_import_phases.json').read_text())['usd']
files+=(imported_identity['path'],)
record['imported_model_identity']=imported_identity
for name in files:
    source=root/name;content=source.read_bytes()
    snapshot=out/'sources'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
    record.setdefault('sources',[]).append(dict(path=name,sha256=hashlib.sha256(content).hexdigest()))
record['cache_sha256']=hashlib.sha256(args.cache.read_bytes()).hexdigest()

def save(status):
    active_env=globals().get('env')
    if active_env is not None and active_env.dr.first_force_readback is not None:
        record['randomization_force_readback']=active_env.dr.first_force_readback
    if active_env is not None:
        import torch
        record['torch_peak_allocated_MiB']=torch.cuda.max_memory_allocated()/2**20
        record['torch_peak_reserved_MiB']=torch.cuda.max_memory_reserved()/2**20
        record['host_peak_rss_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    record['status']=status;record['wall_s']=time.monotonic()-started
    if 'reward_branch_migration' in record:
        record['reward_branch_actions_executed']=(record['actions_executed']-
            record['reward_branch_migration']['reward_start_actions'])
    atomic_json(out/'result.json',record)
    print('BOYA_C '+json.dumps(dict(arm='C',reward_profile=PROFILE,status=status,updates=record['completed_updates'],
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
    from tendonspin.rl.isaac_spin_progress import SpinProgressBoyaEnv as HoraBoyaEnv
    from tendonspin.rl.domain_randomization import load_configuration
    from tendonspin.rl.checkpoint import save_checkpoint,restore_checkpoint
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    cfg=OmegaConf.create(dict(seed=args.seed,rl_device='cuda:0',test=False,checkpoint=None,
        task=dict(env=dict(numEnvs=args.num_envs)),
        train=OmegaConf.load(root/'third_party/hora/configs/train/AllegroHandHora.yaml')))
    cfg.train.ppo.priv_info=True
    cfg.train.ppo.minibatch_size=args.minibatch_size or args.num_envs*cfg.train.ppo.horizon_length
    if (cfg.train.ppo.horizon_length,cfg.train.ppo.mini_epochs) != (8,5):
        raise ValueError('C requires unchanged horizon8 and mini_epochs5')
    batch=args.num_envs*cfg.train.ppo.horizon_length
    target_updates=math.ceil(args.total_actions/batch) if args.total_actions else args.updates
    cfg.train.ppo.max_agent_steps=target_updates*batch
    record.update(requested_updates=target_updates,rounded_target_actions=target_updates*batch)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)),out/'config.yaml')
    ppo_contract=OmegaConf.to_container(cfg.train.ppo,resolve=True)
    for key in ('max_agent_steps','save_frequency','save_best_after','output_name'):ppo_contract.pop(key,None)
    contract=dict(arm='C',cache_sha256=record['cache_sha256'],num_envs=args.num_envs,
        ppo=ppo_contract,network=OmegaConf.to_container(cfg.train.network,resolve=True),
        sources={x['path']:x['sha256'] for x in record['sources'] if
                 x['path'].startswith(('tendonspin/physics/','tendonspin/baselines/','third_party/')) or
                 x['path'] in ('tendonspin/rl/isaac_hora.py','tendonspin/rl/isaac_hora_randomized.py',
                 'tendonspin/rl/isaac_spin_progress.py','tendonspin/rl/spin_progress_reward.py',
                 'assets/config/boya_spin_progress_reward.json','tendonspin/rl/domain_randomization.py','tendonspin/rl/rotation_reward.py','tendonspin/rl/termination.py','assets/grasp/rotation_workspace.json','tendonspin/interfaces.py','tendonspin/evaluation/rotation.py')})
    PPO=load_ppo();pulse.check('constructing scene')
    scene_configuration=None
    if args.scene_config:
        from tendonspin.physics.learning_scene import load_scene_configuration
        scene_configuration=load_scene_configuration(args.scene_config)
        if record['cache_sha256']!=scene_configuration['training_cache_sha256']:
            raise ValueError('Scene configuration requires its declared migrated cache')
    dr_cfg=load_configuration(args.randomization_config)
    declared=dr_cfg['training']
    if (declared.get('arm'),declared['reward_profile'],declared['num_envs']) != ('C',PROFILE,args.num_envs):
        raise ValueError('C config identity/environment count differs from the trainer')
    record['domain_randomization_configuration']=dr_cfg
    contract['domain_randomization']=dr_cfg
    if scene_configuration is None:raise ValueError('Randomized route requires its declared scene')
    env=HoraBoyaEnv(root,out,args.cache,args.num_envs,trace_mode=args.trace_mode,
        termination_profile=args.termination_profile,engine_profile=args.engine_profile,
        reward_profile=args.reward_profile,scene_configuration=scene_configuration,
        randomization_configuration=dr_cfg,training_randomization=True);env.stop_check=pulse.check
    if scene_configuration is not None:
        record['scene_configuration']=scene_configuration
        contract['scene_configuration']=scene_configuration
        record['nominal_initial_state_errors']=env.nominal_initial_state_errors
        record['geometry_application']=dict(before_physics_start=True,
            authored_body_count=len(env.geometry_application['authored']),geometry=env.geometry_application['geometry'],
            first_authored=env.geometry_application['authored'][0])
        record['diagnostic_nominal_axis']=env.rotation_axis[0].cpu().tolist()
    record['termination']=env.termination.spec.to_dict()
    contract['termination']=record['termination']
    record['engine_configuration']=env.physics.engine_configuration
    contract['engine_configuration']=record['engine_configuration']
    record['scene_parsing']=dict(replicate_physics=False,independently_sized_colliders=True)
    contract['scene_parsing']=record['scene_parsing']
    record['reward_configuration']=env.reward_configuration
    contract['reward_configuration']=record['reward_configuration']
    contract['runtime_versions']=record['runtime_versions']
    if args.engine_profile=='tgs16_velocity0':record['engine_audit']=env.physics.engine_audit
    record.update(cache_size=len(env.cache['q']),joint_names=env.physics.names,
        body_names=list(env.physics.hand.body_names),filter_names=env.physics.filter_names,reason_codes=env.reasons)
    if scene_configuration is not None and record['cache_size']!=scene_configuration['training_cache_size']:
        raise ValueError('Migrated cache size differs from scene configuration')
    if args.trace_mode=='full':record['origins']=env.physics.origins.cpu().tolist()
    agent=PPO(env,str(out/'training'),cfg)
    def diagnostic_policy(name):
        torch.save(dict(model=agent.model.state_dict(),running_mean_std=agent.running_mean_std.state_dict(),
            value_mean_std=agent.value_mean_std.state_dict(),arm='C',reward_profile=PROFILE,
            reward_configuration=env.reward_configuration,
            progress=dict(actions_executed=agent.agent_steps,completed_updates=agent.epoch_num,
                lineage_id=record['lineage_id']),resume_source=record['resume_from']),out/(name+'.pth'))
    if args.resume:
        previous=restore_checkpoint(agent,args.resume,contract=contract)
        base_actions=previous['actions_executed'];base_episodes=previous['completed_episodes']
        base_reason_counts=previous['episode_stop_counts']
        record.update(actions_executed=base_actions,completed_updates=previous['completed_updates'],
            completed_episodes=base_episodes,lineage_id=previous['lineage_id'],
            resumed_checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest())
        if 'reward_branch_migration' in previous:
            record['reward_branch_migration']=previous['reward_branch_migration']
        if agent.agent_steps!=base_actions or agent.epoch_num!=previous['completed_updates']:
            raise ValueError('Checkpoint is not a completed PPO-update boundary')
        if target_updates<=agent.epoch_num:raise ValueError('Cumulative target is already reached by checkpoint')
    if args.resume is None:
        if agent.agent_steps!=0 or agent.epoch_num!=0 or len(agent.optimizer.state)!=0:
            raise ValueError('Fresh randomized learner inherited state')
        record['fresh_initialization']=dict(actions=agent.agent_steps,updates=agent.epoch_num,
            adam_state_entries=len(agent.optimizer.state),learning_rate=agent.last_lr,
            seed=args.seed,network_initialization='new Hora ActorCritic, no load',
            observation_normalizer='new Hora RunningMeanStd',value_normalizer='new Hora RunningMeanStd')
    agent.obs=env.reset()
    diagnostic_policy('teacher_initial')
    record['randomization_readback']=dict(startup=env.dr.startup_readback,first_reset=env.dr.first_reset_readback)
    env.dr.snapshot(out/'initial_domain_properties.npz')
    atomic_json(out/'launch_identity.json',dict(fresh=record.get('fresh_initialization'),
        randomization=record['randomization_readback'],cache_counts=env.size_cache_counts.cpu().tolist(),
        static_seed=dr_cfg['static_seed'],replicate_physics=False))
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
        if 'reward_branch_migration' in record:
            progress['reward_branch_migration']=record['reward_branch_migration']
        save_checkpoint(agent,path,contract=contract,progress=progress)
        policy_path=out/(name+'_policy.pth')
        # The full learner remains the sole resume/evaluation source. The policy
        # export belongs to the same completed update and carries its C identity.
        policy=dict(model=agent.model.state_dict(),running_mean_std=agent.running_mean_std.state_dict(),
            value_mean_std=agent.value_mean_std.state_dict(),arm='C',reward_profile=PROFILE,reward_configuration=env.reward_configuration,
            progress=progress,learner_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        policy_tmp=policy_path.with_suffix('.pth.tmp')
        torch.save(policy,policy_tmp);policy_tmp.replace(policy_path)
        record['checkpoint']=str(path.relative_to(root))
        record['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        record['policy_checkpoint']=str(policy_path.relative_to(root))
        record['policy_checkpoint_sha256']=hashlib.sha256(policy_path.read_bytes()).hexdigest()
        atomic_json(out/'latest_checkpoint.json',dict(arm='C',reward_profile=PROFILE,
            path=str(path),sha256=record['checkpoint_sha256'],policy_path=str(policy_path),
            policy_sha256=record['policy_checkpoint_sha256'],**progress))

    for update in range(agent.epoch_num+1,target_updates+1):
        if stop_request['signal'] is not None:
            record['stop_reason']='signal requested stop at update boundary';break
        if args.wall_s>0 and time.monotonic()-started>args.wall_s:
            record['stop_reason']='wall budget';break
        torch.cuda.synchronize();update_started=time.monotonic()
        pulse.check('update boundary');env.begin_trace(agent.horizon_length);agent.epoch_num=update
        losses=agent.train_epoch()
        torch.cuda.synchronize();update_wall_s=time.monotonic()-update_started
        values={name:float(torch.stack(tensors).mean().detach()) for name,tensors in zip(
            ('actor_loss','critic_loss','bounds_loss','entropy','kl'),losses)}
        if not np.isfinite(list(values.values())).all():raise RuntimeError('Nonfinite PPO loss')
        agent.storage.data_dict=None
        stats=collect_episode_summary();processed_episodes+=stats['count']
        if len(record['updates'])==0:
            initial_model=torch.load(out/'teacher_initial.pth',weights_only=False)['model']
            record['first_update_finite']=True
            record['first_update_parameters_changed']=any(not torch.equal(v,initial_model[k]) for k,v in agent.model.state_dict().items())
            if not record['first_update_parameters_changed']:raise RuntimeError('First C update did not change model parameters')
        record['completed_episodes']=base_episodes+processed_episodes
        if args.trace_mode=='summary':env.completed.clear()
        env.flush_trace(out/f'rollout_{update:06d}.npz')
        record.update(actions_executed=base_actions+env.actions_executed,session_actions_executed=env.actions_executed,
            physics_steps=(base_actions+env.actions_executed)*env.physics.adapter.steps_per_control,
            session_physics_steps=env.physical_steps,completed_updates=update,
            training_actions_per_wall_s=env.actions_executed/(time.monotonic()-training_started),
            episode_stop_counts={k:base_reason_counts.get(k,0)+env.reason_counts.get(k,0) for k in env.reasons[1:]})
        record['updates'].append(dict(update=update,update_wall_s=update_wall_s,actions_per_s=batch/update_wall_s,actions=record['actions_executed'],learning_rate=agent.last_lr,
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
        record['updates'][-1]['completed_at_training_wall_s']=time.monotonic()-training_started
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
        diagnostic_policy('teacher_guard_stop')
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
