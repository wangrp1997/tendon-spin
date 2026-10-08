# References: Hora v0.0.1 ActorCritic/RunningMeanStd (MIT), TendonSpin Boya
# physical runner and declared signed-prefix scoring. New frozen-policy runner;
# no learning, cache starts, controller switches or reset after restoration.
"""One original-state Isaac episode from one frozen Hora Boya checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--cache',type=Path,required=True,help='Task construction only; evaluation never calls cache reset')
parser.add_argument('--source-record',type=Path,required=True)
parser.add_argument('--video',action='store_true',help='Record actual evaluation with an Isaac RTX camera')
parser.add_argument('--seconds',type=float,default=120.)
parser.add_argument('--wall-s',type=float,default=1500.)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
started=time.monotonic()
record=dict(controller='frozen_hora_boya_cache28_teacher_v2',requested_s=args.seconds,
    physics_steps=0,valid_steps=0,actual_s=0.,valid_s=0.,episode_resets=0,
    controller_switches=0,training_actions=0,benchmark_validated=False,
    initial_state='original grasp44, not cache',headless=True,seed=43,enable_cameras=args.video,
    checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    stop_reason='time limit')
training=json.loads(args.source_record.read_text())
record['controller']='frozen_'+training['controller']
record['training_actions_in_checkpoint']=training['actions_executed']
record['sources']=training['sources']
for source in training['sources']:
    path=root/source['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:
        raise RuntimeError('Source changed since training; refuse silent substitution: '+source['path'])

def save(status):
    record['status']=status;record['wall_s']=time.monotonic()-started
    (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('BOYA_EVAL '+json.dumps({k:record.get(k) for k in ('status','actual_s','valid_s','net_deg','wall_s','stop_reason')}),flush=True)

save('launching')
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=args.video,device='cuda:0').app
failed=False;env=None;frames=[];angles=[];controls=[];chunks=0;video=None
try:
    import numpy as np
    import torch
    from tendonspin.rl.isaac_hora import HoraBoyaEnv,spin_increment
    from tendonspin.rl.termination import legacy_failure_codes
    from tendonspin.physics.isaac_parallel import tensor
    from tendonspin.baselines.reference_models import build_model,reference_module
    from tendonspin.evaluation.rotation import score_prefix
    torch.set_num_threads(4);torch.manual_seed(43);np.random.seed(43)
    if args.video:
        from tendonspin.rl.native_video import NativePolicyVideo
        video=NativePolicyVideo(out,record['training_actions_in_checkpoint'])
    env=HoraBoyaEnv(root,out,args.cache,num_envs=1,scene_setup=video.setup if video else None,
        termination_profile=training.get('termination',{}).get('profile','legacy_strict'))
    record['termination']=env.termination.spec.to_dict()
    if 'termination' in training and record['termination']!=training['termination']:
        raise ValueError('Evaluation termination differs from the training contract')
    p=env.physics
    p.reset_nominal()
    env.history[:]=env._frame(noise=False)[:,None,:]
    weights=torch.load(args.checkpoint,map_location=p.device,weights_only=False)
    model=build_model().to(p.device);model.load_state_dict(weights['model']);model.eval()
    norm=reference_module('hora_normalizer').RunningMeanStd((96,)).to(p.device)
    norm.load_state_dict(weights['running_mean_std']);norm.eval()
    record.update(joint_names=p.names,body_names=list(p.hand.body_names),filter_names=p.filter_names)
    initial=p.measure()
    np.savez(out/'initial_state.npz',**{k:v.cpu().numpy() for k,v in initial.items()},commands=p.adapter.commands.cpu().numpy())
    if video:
        video.attach(p);video.capture(0.,0.,float(initial['drift_mm'][0]),initial=True)
    prev=initial['object_state'][:,3:7].clone()
    net=0.;legacy_steps=0;legacy_stopped=False;legacy_reason='time limit'
    planned=round(args.seconds/p.dt)
    save('evaluating frozen original-state policy')
    for step in range(planned):
        if step%p.adapter.steps_per_control==0:
            if time.monotonic()-started>=args.wall_s:
                record['stop_reason']='wall budget';break
            obs=env.observe()
            with torch.no_grad():action=model.act_inference(dict(obs=norm(obs['obs']),priv_info=obs['priv_info'])).clamp(-1,1)
            controls.append(dict(step=step,obs=obs['obs'].cpu().numpy()[0],
                priv_info=obs['priv_info'].cpu().numpy()[0],action=action.cpu().numpy()[0]))
            p.adapter.set_action(action)
        q,v,effort=p.advance()
        m=p.measure()
        code=int(env.termination.failure_codes(m,p.origins,
            control_boundary=(step+1)%p.adapter.steps_per_control==0)[0])
        valid=code==0
        legacy_code=int(legacy_failure_codes(m)[0])
        if not legacy_stopped:
            if legacy_code:
                legacy_stopped=True;legacy_reason=env.reasons[legacy_code]
            else:legacy_steps+=1
        if valid:
            net+=float(torch.rad2deg(spin_increment(prev,m['object_state'][:,3:7]))[0])
            record['valid_steps']+=1
        prev=m['object_state'][:,3:7].clone()
        angles.append(net)
        frame={k:v.detach().cpu().numpy()[0].copy() for k,v in m.items()}
        frame.update(joint_pos_before=q.cpu().numpy()[0].copy(),joint_vel_before=v.cpu().numpy()[0].copy(),
            action=action.cpu().numpy()[0].copy(),commands=p.adapter.commands.cpu().numpy()[0].copy(),
            motor_effort_requested=effort.cpu().numpy()[0].copy(),
            actuator_effort_forwarded=tensor(p.hand.actuators.applied_effort).cpu().numpy()[0].copy(),
            valid=valid,failure_code=code,legacy_strict_code=legacy_code,
            legacy_strict_valid=not legacy_stopped,net_angle_deg=net,elapsed_s=(step+1)*p.dt)
        frames.append(frame)
        record.update(physics_steps=step+1,actual_s=(step+1)*p.dt,
            valid_s=record['valid_steps']*p.dt,net_deg=net)
        if len(frames)>=2000:
            np.savez(out/f'physics_{chunks:03d}.npz',**{k:np.asarray([f[k] for f in frames]) for k in frames[0]})
            frames=[];chunks+=1
        if not valid:
            record['stop_reason']=env.reasons[code];break
        if video and (step+1)%100==0:video.capture(record['actual_s'],net,float(m['drift_mm'][0]))
        if (step+1)%p.adapter.steps_per_control==0:
            env.history=torch.roll(env.history,-1,dims=1);env.history[:,-1]=env._frame()
        if (step+1)%20000==0:save('evaluating')
    if video:video.capture(record['actual_s'],net,float(m['drift_mm'][0]),reason=record['stop_reason'])
    record['metrics']={str(window):score_prefix(angles,p.dt,record['valid_steps'],window_s=window) for window in (30.,120.)}
    legacy_steps=min(legacy_steps,record['valid_steps'])
    record['legacy_strict_shadow']=dict(
        interpretation='Offline prefix of this same execution under old rules; not a second rollout or primary score',
        stop_reason=legacy_reason if legacy_stopped else record['stop_reason'],
        metrics={str(window):score_prefix(angles,p.dt,legacy_steps,window_s=window) for window in (30.,120.)})
except BaseException as error:
    failed=True;record['stop_reason']='process error'
    record['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
finally:
    if video is not None:
        record['video']=video.close()
        if record['video'] and record['video']['encoder_exit_code']:
            failed=True;record['video_error']='ffmpeg did not complete successfully'
    if frames:
        np.savez(out/f'physics_{chunks:03d}.npz',**{k:np.asarray([f[k] for f in frames]) for k in frames[0]})
        chunks+=1
    if controls:
        np.savez(out/'control_inputs.npz',**{k:np.asarray([f[k] for f in controls]) for k in controls[0]})
    if angles:np.save(out/'angles_deg.npy',np.asarray(angles))
    record['physics_trace_chunks']=chunks
    save('error' if failed else 'completed')
    app.close(exit_code=1 if failed else 0)
