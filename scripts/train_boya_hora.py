# Reference: Hora v0.0.1 hora/algo/ppo/ppo.py (MIT), original PPO train_epoch.
# Only bounded run/provenance and Boya task interface added. No custom PPO losses.
"""Execute and save a nominal original-Hora teacher pilot on existing Isaac Lab."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import subprocess
import signal
import traceback
from tendonspin.rl.resource_guard import TrainingPulse,TrainingStopRequested

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--cache',type=Path,required=True)
parser.add_argument('--updates',type=int,default=16)
parser.add_argument('--num-envs',type=int,default=64)
parser.add_argument('--wall-s',type=float,default=240.)
parser.add_argument('--seed',type=int,default=43)
parser.add_argument('--save-every',type=int,default=4)
parser.add_argument('--minibatch-size',type=int,default=None)
parser.add_argument('--max-gpu-memory-mib',type=int,default=12288)
parser.add_argument('--stop-file',type=Path)
parser.add_argument('--heartbeat',type=Path)
parser.add_argument('--controller',default='hora_boya_nominal_teacher_v1')
parser.add_argument('--protocol',default='docs/experiments/2026-10-08-boya-hora-pilot/PROTOCOL.md')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
started=time.monotonic()
pulse=TrainingPulse(args.stop_file,args.heartbeat)
pulse.check('launching')
stop_request={'signal':None}
def request_stop(signum, frame):
    stop_request['signal']=signum
signal.signal(signal.SIGINT,request_stop)
signal.signal(signal.SIGTERM,request_stop)
record=dict(controller=args.controller,seed=args.seed,requested_updates=args.updates,
    requested_wall_s=args.wall_s,max_gpu_memory_mib=args.max_gpu_memory_mib,save_every=args.save_every,protocol=args.protocol,headless=True,enable_cameras=False,
    num_envs=args.num_envs,updates=[],actions_executed=0,physics_steps=0,
    stop_reason='update budget',full_hora_reproduction=False,benchmark_validated=False,
    domain_randomization=False,cache_path=str(args.cache),cache_size=None)
files=('scripts/train_boya_hora.py','tendonspin/rl/isaac_hora.py','tendonspin/baselines/hora_training.py',
       'tendonspin/physics/isaac_parallel.py','tendonspin/physics/isaac_boya.py',
       'tendonspin/physics/isaac_boya_sharpa.py','tendonspin/interfaces.py',
       'third_party/hora/hora/algo/ppo/ppo.py','third_party/hora/hora/algo/ppo/experience.py',
       'third_party/hora/hora/algo/models/models.py','third_party/hora/hora/algo/models/running_mean_std.py',
       'third_party/hora/hora/tasks/allegro_hand_hora.py',
       'tendonspin/baselines/reference_models.py','tendonspin/evaluation/rotation.py',
       'tendonspin/physics/coordinates.py','scripts/evaluate_boya_hora.py',
       'tendonspin/rl/resource_guard.py','scripts/guarded_boya_entry.py',args.protocol)
for name in files:
    source=root/name
    snapshot=out/'sources'/name;snapshot.parent.mkdir(parents=True,exist_ok=True)
    snapshot.write_bytes(source.read_bytes())
    record.setdefault('sources',[]).append(dict(path=name,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
record['cache_sha256']=hashlib.sha256(args.cache.read_bytes()).hexdigest()

def save(status):
    record['status']=status;record['wall_s']=time.monotonic()-started
    (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('BOYA_HORA '+json.dumps(dict(status=status,updates=len(record['updates']),
        actions=record['actions_executed'],wall_s=record['wall_s'],stop_reason=record['stop_reason'])),flush=True)

save('launching')
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False,device='cuda:0').app
failed=False;env=None;agent=None;update=0
try:
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from tendonspin.baselines.hora_training import load_ppo
    from tendonspin.rl.isaac_hora import HoraBoyaEnv
    torch.set_num_threads(4)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    cfg=OmegaConf.create(dict(seed=args.seed,rl_device='cuda:0',test=False,checkpoint=None,
        task=dict(env=dict(numEnvs=args.num_envs)),
        train=OmegaConf.load(root/'third_party/hora/configs/train/AllegroHandHora.yaml')))
    cfg.train.ppo.priv_info=True
    cfg.train.ppo.minibatch_size=args.minibatch_size or args.num_envs*cfg.train.ppo.horizon_length
    cfg.train.ppo.max_agent_steps=args.updates*args.num_envs*cfg.train.ppo.horizon_length
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)),out/'config.yaml')
    PPO=load_ppo()
    pulse.check('constructing scene')
    env=HoraBoyaEnv(root,out,args.cache,args.num_envs)
    env.stop_check=pulse.check
    record['cache_size']=len(env.cache['q'])
    record['joint_names']=env.physics.names
    record['body_names']=list(env.physics.hand.body_names)
    record['filter_names']=env.physics.filter_names
    record['reason_codes']=env.reasons
    record['origins']=env.physics.origins.cpu().tolist()
    agent=PPO(env,str(out/'training'),cfg)
    agent.obs=env.reset()
    agent.save(str(out/'teacher_initial'))
    save('training')
    training_started=time.monotonic()
    for update in range(1,args.updates+1):
        if stop_request['signal'] is not None:
            record['stop_reason']='signal requested stop at update boundary';break
        if time.monotonic()-started>args.wall_s:
            record['stop_reason']='wall budget';break
        pulse.check('update boundary')
        env.begin_trace(agent.horizon_length)
        agent.epoch_num=update
        losses=agent.train_epoch()
        values={name:float(torch.stack(tensors).mean().detach()) for name,tensors in zip(
            ('actor_loss','critic_loss','bounds_loss','entropy','kl'),losses)}
        if not np.isfinite(list(values.values())).all():
            raise RuntimeError('Nonfinite PPO loss')
        agent.storage.data_dict=None
        record['updates'].append(dict(update=update,actions=env.actions_executed,
            learning_rate=agent.last_lr,completed_episodes=len(env.completed),**values))
        env.flush_trace(out/f'rollout_{update:03d}.npz')
        record['actions_executed']=env.actions_executed
        record['physics_steps']=env.physical_steps
        record['training_actions_per_wall_s']=env.actions_executed/(time.monotonic()-training_started)
        record['episode_stop_counts']=dict(env.reason_counts)
        if update%args.save_every==0 or update==args.updates:
            agent.save(str(out/f'teacher_u{update:03d}'))
            torch.save(dict(optimizer=agent.optimizer.state_dict(),update=update,
                agent_steps=agent.agent_steps,learning_rate=agent.last_lr,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state(),environment_restore='future resume requires declared reset'),
                out/'optimizer_latest.pth')
        if update==1 or update%8==0:
            gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=5)
            if gpu.returncode==0:
                used=int(gpu.stdout.strip().splitlines()[0])
                record['observed_gpu_memory_used_MiB']=used
                record['max_observed_gpu_memory_used_MiB']=max(used,record.get('max_observed_gpu_memory_used_MiB',0))
        save('update completed')
        if record.get('observed_gpu_memory_used_MiB',0)>args.max_gpu_memory_mib:
            record['stop_reason']='device memory headroom limit';break
    agent.save(str(out/'teacher_final'))
    torch.save(dict(optimizer=agent.optimizer.state_dict(),update=len(record['updates']),
        agent_steps=agent.agent_steps,learning_rate=agent.last_lr,torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
        environment_restore='future resume requires declared reset'),out/'optimizer_final.pth')
    record['checkpoint']=str((out/'teacher_final.pth').relative_to(root))
    record['checkpoint_sha256']=hashlib.sha256((out/'teacher_final.pth').read_bytes()).hexdigest()
    initial=torch.load(out/'teacher_initial.pth',weights_only=False)['model']
    record['model_parameters_changed']=any(not torch.equal(v,initial[k]) for k,v in agent.model.state_dict().items())
except TrainingStopRequested as error:
    record['stop_reason']='resource guard requested stop'
    record['guard_reason']=str(error)
    if agent is not None:
        agent.save(str(out/'teacher_guard_stop'))
        torch.save(dict(optimizer=agent.optimizer.state_dict(),update=len(record['updates']),
            agent_steps=agent.agent_steps,learning_rate=agent.last_lr,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
            environment_restore='future resume requires declared reset'),out/'optimizer_guard_stop.pth')
        record['checkpoint']=str((out/'teacher_guard_stop.pth').relative_to(root))
        record['checkpoint_sha256']=hashlib.sha256((out/'teacher_guard_stop.pth').read_bytes()).hexdigest()
except BaseException as error:
    failed=True;record['stop_reason']='process error'
    record['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
finally:
    if env is not None:
        record['actions_executed']=env.actions_executed;record['physics_steps']=env.physical_steps
        record['episode_stop_counts']=dict(env.reason_counts)
        record['active_training_episodes_at_stop']=env.num_envs
        (out/'training_episodes.json').write_text(json.dumps(env.completed,indent=2)+'\n')
        if env.trace is not None:env.flush_trace(out/f'interrupted_rollout_{update:03d}.npz')
    if agent is not None:agent.writer.close()
    save('error' if failed else 'completed')
    app.close(exit_code=1 if failed else 0)
