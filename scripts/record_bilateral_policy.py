"""Record both hands controlling their own objects in one live physics scene."""
import argparse
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--training',type=Path,default=ROOT/'outputs/boya_touch_student_20261102_v1/training')
    parser.add_argument('--teacher',type=Path,default=ROOT/'outputs/boya_c_fresh1p5b_server_interrupted_recovery_20261028_v1/actions_1260000000/training/teacher_final.pth')
    parser.add_argument('--fixed-cases',type=Path,default=ROOT/'outputs/boya_touch_student_20261102_v1/evaluation/fixed_cases.npz')
    parser.add_argument('--case-index',type=int,default=9)
    parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--fps',type=int,default=40)
    parser.add_argument('--object-separation',type=float,default=.24)
    parser.add_argument('--view',choices=('oblique','top'),default='oblique')
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<=0 or args.fps<=0 or not .12<=args.object_separation<=.5:
        parser.error('Positive duration/fps and separation between .12 and .5 m required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();app=video=None
    status=dict(status='running',phase='starting',same_physics_environment=True,preview=args.preview,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},started_unix_s=time.time())
    def save_status():
        status['wall_s']=time.monotonic()-started
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save_status()
    try:
        from tendonspin.physics.physx_property_sync import install
        status['property_sync']=install(lambda name:None)
        os.environ['ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION']='0'
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=True,enable_cameras=True,device='cuda:0').app
        import torch
        from pxr import UsdPhysics
        from tendonspin.physics.bilateral import BilateralPhysics
        from tendonspin.physics.isaac_parallel import tensor
        from tendonspin.students.model import load_student
        from tendonspin.students.training import SELECTED_TEACHER_SHA256
        from tendonspin.students.observations import SensorHistory,SensorConfig,JOINT_NAMES,joint_limits,sensor_contract
        from tendonspin.rl.isaac_hora import spin_increment
        from tendonspin.rl.bilateral_video import BilateralCamera
        torch.set_num_threads(4)
        if digest(args.teacher)!=SELECTED_TEACHER_SHA256:raise ValueError('Wrong selected teacher')
        teacher=torch.load(args.teacher,map_location='cpu',weights_only=False)
        contract=teacher['contract'];scene=contract['scene_configuration'];dr=contract['domain_randomization']
        if contract['engine_configuration']['profile']!='original_tgs16_4':
            raise ValueError('Bilateral scene requires the original 16/4 physics contract')
        checkpoint=args.training/'proprio_best.pth'
        model,metadata=load_student(checkpoint,'cuda:0')
        if model.modality!='proprio' or metadata['teacher_sha256']!=SELECTED_TEACHER_SHA256:
            raise ValueError('Wrong student contract')
        with np.load(args.fixed_cases,allow_pickle=False) as saved:
            fixed={k:saved[k].copy() for k in saved.files}
        source=json.loads((args.fixed_cases.parent/'comparison.json').read_text())
        if source.get('hand_profile',{}).get('side','right')!='right' or digest(args.fixed_cases)!=source['fixed_cases_sha256']:
            raise ValueError('Require original verified right-hand fixed cases')
        case=args.case_index
        if not 0<=case<len(fixed['joint_pos']):raise ValueError('Case index out of range')
        cache=ROOT/scene['training_cache']
        if digest(cache)!=contract['cache_sha256']:raise ValueError('Cache identity mismatch')
        with np.load(cache,allow_pickle=False) as saved:
            fixed['object_state'][case,:3]=saved['object_state'][int(fixed['cache_index'][case]),:3]
            joint_names=saved['joint_names'].tolist()
        physics=BilateralPhysics(ROOT,out,fixed,case,scene['geometry'],dr['logical_nominal_scale'],args.object_separation)
        video=BilateralCamera(physics,out,args.fps,args.view,args.preview)
        physics.initialize();video.attach()
        controls=round(args.seconds/.05);sample_stride=round(1/(args.fps*physics.dt))
        if abs(controls*.05-args.seconds)>1e-9 or abs(sample_stride*physics.dt*args.fps-1)>1e-9:
            raise ValueError('Duration and frame sampling must align to physics steps')
        histories={};joint_ids={};initial_errors={}
        for side,p in physics.sides.items():
            if p.names!=p.hand_profile.runtime_names(joint_names):raise ValueError('Joint order differs')
            history=SensorHistory(joint_limits(p.contract),1,physics.device,SensorConfig(**metadata['observation_contract']['preprocessing']))
            if sensor_contract(history.limits,history.config)!=metadata['observation_contract']:
                raise ValueError('Sensor preprocessing differs from checkpoint')
            histories[side]=history
            joint_ids[side]=[p.joint_id(n) for n in JOINT_NAMES]
            initial_errors[side]=float((tensor(p.hand.data.joint_pos)[0]-torch.as_tensor(fixed['joint_pos'][case],device=physics.device)).abs().max())
        if max(initial_errors.values())>1e-6:raise ValueError('Joint initial readback failed')
        paths=[str(p.GetPath()) for p in physics.sim.stage.Traverse()]
        envs=[p for p in paths if p.startswith('/World/envs/env_') and p.count('/')==3]
        if envs!=['/World/envs/env_0']:raise ValueError('Expected exactly one physical environment')
        physics_scenes=[str(p.GetPath()) for p in physics.sim.stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(physics_scenes)!=1:raise ValueError('Expected one physics scene')
        status.update(phase='recording',scene=physics.configuration,environment_paths=envs,physics_scene_paths=physics_scenes,
            checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),fixed_cases_sha256=digest(args.fixed_cases),
            property_readback=physics.property_readback,initial_joint_errors=initial_errors,
            hand_profiles={side:p.hand_profile.identity() for side,p in physics.sides.items()},
            camera=video.camera_configuration,physics_steps=0,frames=0,playback_speed=1.,interpolated=False,capture_timing='right endpoint of each frame interval; no initial stale renderer pose',
            source_files={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__).resolve(),
                ROOT/'tendonspin/physics/bilateral.py',ROOT/'tendonspin/rl/bilateral_video.py',
                ROOT/'tendonspin/physics/hand_profiles.py',ROOT/'tendonspin/rl/termination.py']})
        # Keep the exact executed source alongside the recording.
        for relative in status['source_files']:
            target=out/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes((ROOT/relative).read_bytes())
        physics.sim.stage.GetRootLayer().Export(str(out/'scene.usda'))
        records=[];actions=[];forces=[];noises=[]
        net=torch.zeros(2,device=physics.device,dtype=torch.float64)
        failure=torch.zeros(2,device=physics.device,dtype=torch.long)
        rng=torch.Generator(device=physics.device).manual_seed(2026110205)
        force_rng=torch.Generator(device=physics.device).manual_seed(2026110204)
        canonical_force=torch.zeros((1,3),device=physics.device)
        measures={side:p.measure() for side,p in physics.sides.items()}
        min_gap=physics.cross_clearance(measures)
        if min_gap<=.005:raise ValueError(f'Insufficient initial cross-hand clearance: {min_gap} m')
        def capture(encode=True,save=None):
            elapsed=physics.physics_steps*physics.dt
            records.append(dict(elapsed_s=elapsed,
                body_pose=np.stack([tensor(measures[side]['body_pose'])[0].cpu().numpy() for side in ('left','right')]),
                object_state=np.stack([tensor(measures[side]['object_state'])[0].cpu().numpy() for side in ('left','right')]),
                net_deg=torch.rad2deg(net).cpu().numpy().copy(),failure_code=failure.cpu().numpy().copy()))
            video.capture(elapsed,torch.rad2deg(net).tolist(),failure.tolist(),encode=encode,save=save)
        # Capture the right endpoint of every 25 ms interval. PhysX does not
        # publish initial articulated link transforms to RTX until its first step.
        # The first image is therefore t=25 ms, with no stale zero-pose frame.
        for control in range(1 if args.preview else controls):
            if (out/'STOP').exists():raise InterruptedError('Stop requested')
            with torch.no_grad():
                noise=(torch.rand((1,20),generator=rng,device=physics.device)*2-1)*dr['joint_noise_rad']
                if control==0:noise[:,:16]=0
                noises.append(noise[0].cpu().numpy().copy())
                inputs=[]
                for side,p in physics.sides.items():
                    q=tensor(p.hand.data.joint_pos)[:,joint_ids[side]]+noise
                    # Same sensor-only path as proprio deployment: tactile
                    # channels are absent and masked by the frozen network.
                    force=torch.zeros((1,5),device=physics.device)
                    valid=torch.zeros((1,5),device=physics.device,dtype=torch.bool)
                    inputs.append(histories[side].push(q,p.adapter.commands[:,p.adapter.active],force,valid,torch.zeros_like(force)))
                action=model(torch.cat(inputs,dim=0))['action']
                if not torch.isfinite(action).all():raise FloatingPointError('Nonfinite action')
                actions.append(action.cpu().numpy().copy())
                if control==0:status['initial_action_max_difference']=float((action[0]-action[1]).abs().max())
                for i,p in enumerate(physics.sides.values()):p.adapter.set_action(action[i:i+1])
                canonical_force*=dr['force_decay']**(dr['force_decay_reference_dt_s']/dr['force_decay_interval_s'])
                if bool(torch.rand((),generator=force_rng,device=physics.device)<dr['force_probability_per_control']):
                    canonical_force=torch.randn((1,3),generator=force_rng,device=physics.device)*float(fixed['mass_kg'][case])*dr['force_scale']
                forces.append(canonical_force[0].cpu().numpy().copy())
                previous={side:m['object_state'][:,3:7].clone() for side,m in measures.items()}
                for substep in range(100):
                    physics.step(canonical_force)
                    measures={side:p.measure() for side,p in physics.sides.items()}
                    for i,(side,p) in enumerate(physics.sides.items()):
                        m=measures[side]
                        if not bool(m['finite'].all()):raise FloatingPointError('Nonfinite physics')
                        task_measure=dict(m,object_state=physics.canonical_state(side,m['object_state']))
                        codes=p.termination.failure_codes(task_measure,p.origins,control_boundary=substep==99)
                        failure[i]=torch.where(failure[i]==0,codes[0],failure[i])
                        inc=spin_increment(previous[side],m['object_state'][:,3:7])[0].double()
                        previous[side]=m['object_state'][:,3:7].clone()
                        net[i]+=torch.where(failure[i]==0,inc,0.)
                    min_gap=min(min_gap,physics.cross_clearance(measures))
                    if min_gap<=0:raise RuntimeError('Cross-hand bounding clearance exhausted; increase separation')
                    step=physics.physics_steps
                    terminal=step==(1 if args.preview else controls)*100 or bool(failure.any())
                    if step%sample_stride==0 or terminal:
                        save='initial.png' if step==sample_stride else 'final.png' if terminal else f'frame_{step:06d}.png' if step%10000==0 else None
                        capture(encode=not args.preview,save=save)
                    if bool(failure.any()):break
            status.update(elapsed_s=physics.physics_steps*physics.dt,physics_steps=physics.physics_steps,
                current_control=control+1,frames=video.frames,net_deg=torch.rad2deg(net).tolist(),
                failure_codes=failure.tolist(),minimum_cross_clearance_m=min_gap)
            if control%20==19 or bool(failure.any()):
                save_status();print('BILATERAL '+json.dumps({k:status[k] for k in
                    ('current_control','elapsed_s','net_deg','minimum_cross_clearance_m','wall_s')}),flush=True)
            if bool(failure.any()):break
        video.close()
        np.savez_compressed(out/'trajectory.npz',**{key:np.asarray([row[key] for row in records]) for key in records[0]},
            side=np.array(['left','right']),body_names=np.array([list(p.hand.body_names) for p in physics.sides.values()]),
            action=np.asarray(actions),canonical_force_n=np.asarray(forces),noise_rad=np.asarray(noises))
        status.update(status='completed',phase='completed',elapsed_s=physics.physics_steps*physics.dt,
            physics_steps=physics.physics_steps,frames=video.frames,fps=args.fps,duration_s=video.frames/args.fps,
            net_deg=torch.rad2deg(net).tolist(),failure_codes=failure.tolist(),minimum_cross_clearance_m=min_gap,
            trajectory_sha256=digest(out/'trajectory.npz'),new_training_updates=0,resets_within_recording=0,
            clearance_check='Conservative mesh AABB corners in measured rigid-body poses plus bounding spheres for both objects; positive separating X gap checked every physics step')
        if not args.preview:status['video_sha256']=digest(out/'bilateral_same_scene_1x.mp4')
    except BaseException as error:
        import traceback
        status.update(status='error',phase='error',error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if video is not None:video.close()
        save_status()
        if app is not None:app.close(exit_code=0 if status['status']=='completed' else 1)


if __name__=='__main__':main()
