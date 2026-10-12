# Reuse: TendonSpin6679f2b BoyaParallel original_tgs16_4/v4 clipped PD,
# settled-cache protocol at6089025, native cache named states, and HoraBoyaEnv
# zero-velocity cache reset. Coordinate port: WXYZ->XYZW; cache twist is WORLD.
# No PPO, new sampling, engine/controller substitutions, or training launch.
"""Transfer one existing native cache and screen it once in the training engine."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--native-cache',type=Path,required=True)
parser.add_argument('--grasp-profile',required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
started=time.monotonic()
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
spec=json.loads((ROOT/'assets/grasp/profiles.json').read_text())['profiles'][args.grasp_profile]
plan=json.loads((ROOT/spec['plan']).read_text())
record=dict(status='launching',source_native_cache=str(args.native_cache.resolve()),source_cache_sha256=digest(args.native_cache),
    grasp_profile=args.grasp_profile,object=plan['object'],object_reference=dict(pos=plan['pos'],quat_wxyz=plan['quat']),
    engine='IsaacSim6.1.0.0/Lab3.0.0rc1/GPUPhysX original_tgs16_4',
    collision='existing CAD convex/selfcollision/28excludes/friction0.5/contactoffset0.1mm/CCDoff; no supportfloor',
    actuators='16 finger actions, two held wrists, four passive1:1mimics, existing v4 clipped external PD; frozen source commands',
    observations='ground-truth states/contact for transfer and diagnostics; no learned policy',
    timing='dt0.0005s/control20Hz; each state ONE uninterrupted0.5s initialization+0.5s formal screen',
    budget='one batch of all existing53random states plus nominal anchor;2000batched steps;wall270s;no resampling/retry',
    reset_semantics='q/object pose/commands transferred by names; joint and object velocities zeroed EXACTLY as existing HoraBoyaEnv reset',
    native_solver_state_transferred=False,training_actions=0,learner_created=False,training_launch_authorized=False,
    acceptance='same cache gates:finite/drift<=5mm from fixed plan center/tilt<=15deg from fixed plan axis/speed<=100rad_s/linknormal<=12N/mimic<=.05rad;>=2groups in>=90%final0.1s;nonfinite initialization rejects;five-finger diagnostic only',
    task_shadow='unchanged B boya_workspace termination checked from first step, no initialization grace for this shadow',
    sources=[],stop_reason=None)
paths=[Path(__file__),args.native_cache.resolve(),ROOT/'assets/grasp/profiles.json',ROOT/spec['scene'],ROOT/spec['state'],ROOT/spec['plan'],ROOT/spec['evidence']]
paths += [ROOT/name for name in ('tendonspin/physics/isaac_parallel.py','tendonspin/physics/isaac_boya.py',
    'tendonspin/physics/isaac_boya_sharpa.py','tendonspin/physics/object_geometry.py','tendonspin/physics/coordinates.py',
    'tendonspin/physics/solver_profiles.py','tendonspin/interfaces.py','tendonspin/rl/isaac_hora.py','tendonspin/rl/termination.py',
    'assets/grasp/rotation_workspace.json','docs/experiments/2026-10-08-boya-settled-cache/PROTOCOL.md')]
for path in paths:
    record['sources'].append(dict(path=str(path),sha256=digest(path)))
    snap=out/'sources'/path.relative_to(ROOT);snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(path.read_bytes())
def save(status):
    record.update(status=status,wall_s=time.monotonic()-started);write(out/'result.json',record)
    print('CACHE_TRANSFER '+json.dumps({k:record.get(k) for k in ('status','wall_s','num_envs','physics_steps','accepted_random','nominal_passed','stop_reason')}),flush=True)
save('launching')
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False,device='cuda:0').app
failed=False;trace=None;executed=0
try:
    import numpy as np
    import torch
    from tendonspin.physics.isaac_parallel import BoyaParallel,tensor,axis_z
    from tendonspin.physics.coordinates import wxyz_to_xyzw,orientation_error_deg
    from tendonspin.physics.object_geometry import configure_object_scene
    from tendonspin.physics.solver_profiles import ORIGINAL
    from tendonspin.rl.isaac_hora import spin_increment
    from tendonspin.rl.termination import TaskTermination,make_termination_spec
    torch.set_num_threads(4)
    with np.load(args.native_cache,allow_pickle=False) as saved:source={k:saved[k].copy() for k in saved.files}
    if str(source['scene_sha256'])!=spec['scene_sha256'] or str(source['engine_sha256'])!=spec['engine_sha256']:
        raise RuntimeError('Native cache does not match the declared source profile/engine')
    if digest(ROOT/spec['scene'])!=spec['scene_sha256']:raise RuntimeError('Source scene identity changed')
    ids=source['candidate_id'].astype(int);N=len(ids)
    if N!=54 or np.sum(ids==0)!=1:raise ValueError('Expected existing53random states plus ONE nominal anchor')
    native_names=source['joint_names'].tolist();native_motors=source['motor_names'].tolist()
    geometry=dict(shape='cylinder',diameter_m=2*plan['object']['radius'],height_m=plan['object']['height'],mass_kg=plan['object']['mass'],dimensions_m=[2*plan['object']['radius']]*2+[plan['object']['height']])
    def setup(physics):record['geometry_application']=configure_object_scene(physics,geometry)
    env=BoyaParallel(ROOT,out,N,scene_setup=setup,engine_profile=ORIGINAL)
    motor_names=[x['joint'] for x in env.contract['actuators']]
    q_np=source['q'][:,[native_names.index(n) for n in env.names]]
    commands_np=source['commands'][:,[native_motors.index(n) for n in motor_names]]
    object_np=source['object_state_wxyz'].copy();object_np[:,3:7]=wxyz_to_xyzw(object_np[:,3:7]);object_np[:,7:]=0.
    q=torch.as_tensor(q_np,device=env.device,dtype=torch.float32);v=torch.zeros_like(q)
    commands=torch.as_tensor(commands_np,device=env.device,dtype=torch.float32)
    obj=torch.as_tensor(object_np,device=env.device,dtype=torch.float32);obj[:,:3]+=env.origins
    env._write_state(q,v,obj,commands);env.adapter.elapsed=0
    env.center=torch.tensor([plan['pos']],device=env.device,dtype=torch.float32).repeat(N,1)+env.origins
    env.axis0=axis_z(torch.tensor([wxyz_to_xyzw(plan['quat'])],device=env.device,dtype=torch.float32)).repeat(N,1)
    initial=env.measure();initial_np={k:x.cpu().numpy().copy() for k,x in initial.items()}
    actual_local=initial_np['object_state'].copy();actual_local[:,:3]-=env.origins.cpu().numpy()
    errors=dict(joint_rad=np.max(np.abs(initial_np['joint_pos']-q_np),axis=1),
        joint_velocity_rad_s=np.max(np.abs(initial_np['joint_vel']),axis=1),
        object_position_mm=np.linalg.norm(actual_local[:,:3]-object_np[:,:3],axis=1)*1000,
        object_orientation_deg=orientation_error_deg(actual_local[:,3:7],source['object_state_wxyz'][:,3:7]),
        object_velocity=np.max(np.abs(actual_local[:,7:]),axis=1),
        commands_rad=np.max(np.abs(env.adapter.commands.cpu().numpy()-commands_np),axis=1))
    limits=dict(joint_rad=1e-6,joint_velocity_rad_s=1e-6,object_position_mm=.05,object_orientation_deg=.05,object_velocity=1e-6,commands_rad=1e-6)
    transfer=np.logical_and.reduce([np.isfinite(x)&(x<=limits[k]) for k,x in errors.items()])
    np.savez_compressed(out/'initial_state.npz',**initial_np,commands=env.adapter.commands.cpu().numpy(),candidate_id=ids,origins=env.origins.cpu().numpy(),**{'transfer_error_'+k:x for k,x in errors.items()})
    record.update(num_envs=N,random_states=int((ids!=0).sum()),candidate_ids=ids.tolist(),joint_names=env.names,motor_names=motor_names,
        body_names=list(env.hand.body_names),filter_names=env.filter_names,origins=env.origins.cpu().tolist(),engine_configuration=env.engine_configuration,
        motor_parameters=env.parameters,collision_excludes=env.excludes,dt=env.dt,
        initial_transfer_passed=bool(transfer.all()),initial_transfer_tolerances=limits,
        initial_transfer_max_errors={k:float(x.max()) for k,x in errors.items()},velocity_zeroing_not_full_native_velocity_restore=True)
    if not transfer.all():raise RuntimeError('Named-state transfer readback failed; dependent physics stopped')
    task=TaskTermination(make_termination_spec(ROOT,'boya_workspace',env.contract['object']['pos'][2]));record['b_task_configuration']=task.spec.to_dict()
    first=torch.zeros(N,dtype=torch.long,device=env.device);first_step=torch.full_like(first,-1)
    init_first=torch.zeros_like(first);init_step=torch.full_like(first,-1);init_nonfinite=torch.zeros(N,dtype=torch.bool,device=env.device)
    task_first=torch.zeros_like(first);task_step=torch.full_like(first,-1)
    maxima=torch.zeros((N,5),device=env.device);init_maxima=torch.zeros_like(maxima);tail_support=torch.zeros_like(first)
    actions=torch.zeros((N,16),device=env.device);settle=1000;planned=2000;tail=200
    net=torch.zeros(N,device=env.device);peak=torch.zeros_like(net);backward=torch.zeros_like(net);valid_steps=torch.zeros_like(first)
    reasons=['completed','nonfinite','drift >5mm','tilt >15deg','speed >100rad/s','link normal >12N','mimic error >.05rad','insufficient observed support','wall budget']
    def violations(m):return (~m['finite'],m['drift_mm']>5.,m['tilt_deg']>15.,m['max_speed']>100.,m['max_normal']>12.,m['coupling_error']>.05)
    def mark(codes,indices,bads,step):
        for code,bad in enumerate(bads,1):
            fresh=(codes==0)&bad;codes[fresh]=code;indices[fresh]=step
    previous=initial['object_state'][:,3:7].clone()
    save('screening existing54 transferred states; no sampling or learner')
    for step in range(planned):
        if step==settle:
            formal=env.measure();np.savez_compressed(out/'formal_start.npz',**{k:x.cpu().numpy() for k,x in formal.items()},commands=env.adapter.commands.cpu().numpy())
            first[init_nonfinite]=1;first_step[init_nonfinite]=init_step[init_nonfinite]
            mark(first,first_step,violations(formal),step)
            maxima=torch.stack([formal[k] for k in ('drift_mm','tilt_deg','max_speed','max_normal','coupling_error')],dim=-1)
            previous=formal['object_state'][:,3:7].clone()
        if step%env.adapter.steps_per_control==0:
            if time.monotonic()-started>=270.:record['stop_reason']='wall budget';break
            env.adapter.set_action(actions)
        before_q,before_v,effort=env.advance();m=env.measure()
        values=torch.stack([m[k] for k in ('drift_mm','tilt_deg','max_speed','max_normal','coupling_error')],dim=-1)
        if step<settle:
            mark(init_first,init_step,violations(m),step+1);init_nonfinite|=~m['finite'];init_maxima=torch.maximum(init_maxima,values)
        else:
            mark(first,first_step,violations(m),step+1);maxima=torch.maximum(maxima,values)
            inc=spin_increment(previous,m['object_state'][:,3:7]);previous=m['object_state'][:,3:7].clone();valid=first==0
            net+=torch.where(valid,inc,0.);peak=torch.maximum(peak,net);backward+=torch.where(valid,(-inc).clamp_min(0),0.);valid_steps+=valid
        codes=task.failure_codes(m,env.origins,control_boundary=(step+1)%env.adapter.steps_per_control==0)
        fresh=(task_first==0)&(codes!=0);task_first[fresh]=codes[fresh];task_step[fresh]=step+1
        if step>=planned-tail:tail_support+=(m['support_groups']>=2)
        forces=torch.linalg.vector_norm(m['normal_force_matrix_w'],dim=-1)@env.groups
        frame=dict(m,joint_pos_before=before_q,joint_vel_before=before_v,action=actions,commands=env.adapter.commands,
            motor_effort_requested=effort,actuator_effort_forwarded=tensor(env.hand.actuators.applied_effort),
            group_normals_N=forces,first_failure=first,initialization_first_failure=init_first,b_task_first_failure=task_first,
            phase=torch.full_like(first,int(step>=settle)),net_angle_deg=torch.rad2deg(net),backward_deg=torch.rad2deg(backward))
        arrays={k:x.detach().cpu().numpy().copy() for k,x in frame.items()}
        if trace is None:trace={k:np.empty((planned,*x.shape),dtype=x.dtype) for k,x in arrays.items()}
        for k,x in arrays.items():trace[k][step]=x
        executed=step+1
    record.update(physics_steps=executed,actual_s=executed*env.dt,formal_s=max(0,executed-settle)*env.dt)
    if executed<planned:first[first==0]=8
    else:first[(first==0)&(tail_support<.9*tail)]=7
    accepted=first==0;random=torch.as_tensor(ids!=0,device=env.device);retained=accepted&random
    contacts=trace['group_normals_N'][settle:executed,:,:5]>1e-6 if executed>settle else None
    rows=[]
    for i,candidate_id in enumerate(ids):
        row=dict(candidate_id=int(candidate_id),env_id=i,accepted=bool(accepted[i]),stop_reason=reasons[int(first[i])],first_failure_step=int(first_step[i]),
            initialization_first_failure=reasons[int(init_first[i])],formal_maxima=maxima[i].cpu().tolist(),initialization_maxima=init_maxima[i].cpu().tolist(),
            b_task_first_code=int(task_first[i]),b_task_first_step=int(task_step[i]),valid_formal_s=float(valid_steps[i])*env.dt,
            signed_net_deg=float(torch.rad2deg(net[i])),peak_deg=float(torch.rad2deg(peak[i])),backward_deg=float(torch.rad2deg(backward[i])))
        if contacts is not None:row.update(formal_finger_contact_fractions=contacts[:,i].mean(0).tolist(),tail_min_finger_count=int((trace['group_normals_N'][executed-tail:executed,i,:5]>1e-6).sum(-1).min()))
        rows.append(row)
    record.update(candidates=rows,accepted_random=int(retained.sum()),nominal_passed=bool(accepted[np.flatnonzero(ids==0)[0]]),
        random_stop_reasons=dict(Counter(reasons[int(x)] for x in first[random].cpu().tolist())),
        b_task_failure_count=int((task_first[random]!=0).sum()),accepted_candidate_ids=ids[retained.cpu().numpy()].tolist(),
        accepted_random_tail_min_finger_count_distribution=dict(Counter(row['tail_min_finger_count'] for row in rows if row['candidate_id']!=0 and row['accepted'])),
        metric_order=['drift_mm','tilt_deg','speed_rad_s','linknormal_N','mimicerror_rad'])
    np.savez_compressed(out/'selection.npz',accepted=accepted.cpu().numpy(),candidate_id=ids,first_failure=first.cpu().numpy(),first_failure_step=first_step.cpu().numpy(),maxima=maxima.cpu().numpy(),b_task_first_failure=task_first.cpu().numpy(),b_task_first_failure_step=task_step.cpu().numpy())
    valid_ids=torch.where(retained)[0]
    if len(valid_ids):
        # Saved AFTER Isaac settling, in the existing training-cache array layout.
        cache=dict(q=m['joint_pos'][valid_ids].cpu().numpy(),v=m['joint_vel'][valid_ids].cpu().numpy(),
            object_state=m['object_state'][valid_ids].cpu().numpy(),commands=env.adapter.commands[valid_ids].cpu().numpy(),
            candidate_id=ids[valid_ids.cpu().numpy()],env_id=valid_ids.cpu().numpy(),batch_id=np.zeros(len(valid_ids),dtype=int),
            origins=env.origins[valid_ids].cpu().numpy(),joint_names=np.array(env.names),motor_names=np.array(motor_names),
            engine_profile=np.array(ORIGINAL),source_native_cache_sha256=np.array(record['source_cache_sha256']))
        cache['object_state'][:,:3]-=cache['origins']
        path=out/'grasp_cache.npz';np.savez_compressed(path,**cache);record.update(cache_path=str(path),cache_sha256=digest(path),distinct_accepted_states=len(np.unique(np.round(cache['q'],8),axis=0)))
    record['canonical_reference_for_future_configuration']=dict(pos=plan['pos'],quat_xyzw=wxyz_to_xyzw(plan['quat']).tolist(),geometry=geometry)
    record['limitation']='Migration screens are1s, not30s Isaac holding. Training entry must explicitly apply this geometry AND canonical reference; no training launched.'
    record['stop_reason']=record['stop_reason'] or 'existing54-state migration batch completed; awaiting user review before any training'
except BaseException as error:
    failed=True;record['stop_reason']='execution/source/transfer error; no retry';record['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
finally:
    if trace is not None and executed:np.savez_compressed(out/'execution.npz',**{k:x[:executed] for k,x in trace.items()})
    save('error' if failed else 'completed');app.close(exit_code=1 if failed else 0)
