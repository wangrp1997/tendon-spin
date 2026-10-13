# References: Hora v0.0.1 allegro_hand_grasp.py (MIT), size-matched grasp cache;
# TendonSpin generate_boya_grasps.py at8cca30e (MIT), two-phase admission,
# verify_boya_native_cache_isaac.py, zero-velocity named-state transfer.
# Port: ONE nine-size batch, same old gates and B workspace; streamed full traces.
"""Screen one bounded set of proposals in the actual training engine."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plans',type=Path,required=True)
parser.add_argument('--configuration',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
cfg=json.loads(args.configuration.read_text());plans=json.loads((args.plans/'result.json').read_text())
if plans['status']!='completed' or len(plans['scales'])!=len(cfg['scale_centers']):
    raise ValueError('All nine bounded proposals are required')
started=time.monotonic();record=dict(status='launching',configuration=cfg,
    training_actions=0,physics_steps=0,replicate_physics=False,engine_profile='original_tgs16_4',
    cache_states_are_not_30s_certified=True,auto_retry=False,stop_reason=None,sources=[])
for name in [str(Path(__file__).relative_to(ROOT)),str(args.configuration),str(args.plans/'result.json'),
        'tendonspin/physics/isaac_parallel_randomized.py','tendonspin/physics/isaac_parallel.py',
        'tendonspin/physics/object_geometry.py','tendonspin/physics/isaac_boya_sharpa.py',
        'tendonspin/physics/isaac_boya.py','tendonspin/rl/termination.py',
        'tendonspin/rl/isaac_hora.py','tendonspin/interfaces.py',
        'docs/experiments/2026-10-13-boya-randomized-b50m/PROTOCOL.md']:
    p=ROOT/name;record['sources'].append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    snap=out/'sources'/name;snap.parent.mkdir(parents=True,exist_ok=True);snap.write_bytes(p.read_bytes())
def save(status):
    record.update(status=status,wall_s=time.monotonic()-started)
    (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('SCALE_CACHE '+json.dumps({k:record.get(k) for k in ('status','physics_steps','wall_s','counts','stop_reason')}),flush=True)
save('launching')
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=False,device='cuda:0').app
failed=False;frames=[];chunks=0
try:
    import numpy as np
    import torch
    from pxr import UsdPhysics,UsdGeom
    from tendonspin.physics.isaac_parallel_randomized import BoyaRandomizedParallel
    from tendonspin.physics.isaac_parallel import tensor,axis_z
    from tendonspin.physics.object_geometry import author_primitive
    from tendonspin.rl.isaac_hora import spin_increment
    from tendonspin.rl.termination import TaskTermination,make_termination_spec,legacy_failure_codes
    from tendonspin.interfaces import ACTION_NAMES
    torch.set_num_threads(4)
    count=cfg['cache_preparation']['candidates_per_scale'];groups=len(cfg['scale_centers']);N=count*groups
    geometry=[];proposal=[]
    for row in plans['scales']:
        geo=dict(row['dimensions_m']);geo['dimensions_m']=[geo['diameter_m']]*2+[geo['height_m']]
        geometry.extend([geo]*count)
        with np.load(ROOT/row['proposal'],allow_pickle=False) as data:proposal.append({k:data[k].copy() for k in data.files})
    def setup(p):
        for i,geo in enumerate(geometry):
            body=f'/World/envs/env_{i}/Cylinder'
            prim=author_primitive(p.sim.stage,body+'/geometry/mesh',geo)
            if not prim.HasAPI(UsdPhysics.CollisionAPI):raise ValueError('Collision API lost')
            UsdPhysics.MassAPI(p.sim.stage.GetPrimAtPath(body)).CreateMassAttr(.05)
    p=BoyaRandomizedParallel(ROOT,out,N,scene_setup=setup)
    if p.scene.cfg.replicate_physics:raise RuntimeError('Individual size parsing not enabled')
    inertia=tensor(p.object.data.body_inertia).cpu().numpy().reshape(N,-1)
    centers=inertia[::count]
    record['inertia_per_scale_kg_m2']=centers.tolist()
    record['distinct_inertia_groups']=len(np.unique(np.round(centers,12),axis=0))
    if record['distinct_inertia_groups']!=groups or not np.isfinite(centers).all():
        raise RuntimeError('PhysX size-dependent inertia readback failed; refuse silent geometry replication')
    for i,geo in enumerate(geometry):
        shape=UsdGeom.Cylinder(p.sim.stage.GetPrimAtPath(f'/World/envs/env_{i}/Cylinder/geometry/mesh'))
        if abs(shape.GetRadiusAttr().Get()*2-geo['diameter_m'])>1e-9 or abs(shape.GetHeightAttr().Get()-geo['height_m'])>1e-9:
            raise RuntimeError('Authored size readback changed during startup')
    names=p.names;motors=[a['joint'] for a in p.contract['actuators']]
    q=[];obj=[];commands=[]
    rng=np.random.default_rng(cfg['cache_preparation']['seed'])
    joint_ids=[names.index(n) for n in ACTION_NAMES];motor_ids=[motors.index(n) for n in ACTION_NAMES]
    splay=[ACTION_NAMES.index(f+'J4') for f in ('FF','MF','RF','LF')]
    lower=p.adapter.control_limits[p.adapter.active,0].cpu().numpy();upper=p.adapter.control_limits[p.adapter.active,1].cpu().numpy()
    for data in proposal:
        base=data['q'][[data['joint_names'].tolist().index(n) for n in names]]
        target=data['commands'][[data['motor_names'].tolist().index(n) for n in motors]]
        qq=np.repeat(base[None],count,axis=0)
        perturb=rng.uniform(-cfg['cache_preparation']['joint_perturb_rad'],cfg['cache_preparation']['joint_perturb_rad'],(count,16))
        perturb[0]=0.;perturb[:,splay]=0.
        qq[:,joint_ids]=np.clip(qq[:,joint_ids]+perturb,lower,upper)
        for f in ('FF','MF','RF','LF'):qq[:,names.index(f+'J1')]=qq[:,names.index(f+'J2')]
        bias=target-base[[names.index(n) for n in motors]]
        cc=qq[:,[names.index(n) for n in motors]]+bias
        cc=np.clip(cc,p.adapter.control_limits[:,0].cpu().numpy(),p.adapter.control_limits[:,1].cpu().numpy())
        # Preserve the two held wrist commands from the proposal.
        cc[:,[i for i,n in enumerate(motors) if n not in ACTION_NAMES]]=target[[i for i,n in enumerate(motors) if n not in ACTION_NAMES]]
        q.append(qq);commands.append(cc);obj.append(np.repeat(data['object_state'][None],count,axis=0))
    q=torch.tensor(np.concatenate(q),device=p.device,dtype=torch.float32)
    commands=torch.tensor(np.concatenate(commands),device=p.device,dtype=torch.float32)
    obj=torch.tensor(np.concatenate(obj),device=p.device,dtype=torch.float32);obj[:,:3]+=p.origins;obj[:,7:]=0.
    p._write_state(q,torch.zeros_like(q),obj,commands)
    nominal=json.loads((ROOT/cfg['nominal_plan']).read_text())
    p.center=torch.tensor([nominal['pos']],device=p.device).repeat(N,1)+p.origins
    p.axis0=axis_z(torch.tensor([np.array(nominal['quat'])[[1,2,3,0]]],device=p.device,dtype=torch.float32)).repeat(N,1)
    m=p.measure()
    errors=dict(q=float((m['joint_pos']-q).abs().max()),v=float(m['joint_vel'].abs().max()),
        object=float((m['object_state']-obj).abs().max()),commands=float((p.adapter.commands-commands).abs().max()))
    record['initial_state_errors']=errors
    if max(errors.values())>1e-6:raise RuntimeError('Proposal state readback failed')
    np.savez(out/'initial_states.npz',q=q.cpu().numpy(),object_state=(obj-torch.nn.functional.pad(p.origins,(0,10))).cpu().numpy(),
        commands=commands.cpu().numpy(),scale_id=np.repeat(np.arange(groups),count),joint_names=np.array(names),motor_names=np.array(motors))
    task=TaskTermination(make_termination_spec(ROOT,'boya_workspace',p.contract['object']['pos'][2]))
    dt=p.dt;settle=round(cfg['cache_preparation']['initialization_s']/dt)
    planned=settle+round(cfg['cache_preparation']['formal_screen_s']/dt);tail=round(.1/dt)
    first=torch.zeros(N,dtype=torch.long,device=p.device);init_nonfinite=torch.zeros(N,dtype=torch.bool,device=p.device)
    task_first=torch.zeros_like(first);support=torch.zeros_like(first);net=torch.zeros(N,device=p.device)
    backward=torch.zeros_like(net);peak=torch.zeros_like(net);valid_steps=torch.zeros_like(first)
    maxima=torch.zeros((N,5),device=p.device);actions=torch.zeros((N,16),device=p.device)
    previous=m['object_state'][:,3:7].clone()
    def flush():
        global chunks
        if frames:
            np.savez(out/f'physics_{chunks:03d}.npz',**{k:np.stack([f[k] for f in frames]) for k in frames[0]})
            frames.clear();chunks+=1
    save('screening one batch')
    for step in range(planned):
        if step==settle:
            first[init_nonfinite]=1;m=p.measure();c=legacy_failure_codes(m);first=torch.where(first==0,c,first)
            previous=m['object_state'][:,3:7].clone()
        if step%p.adapter.steps_per_control==0:p.adapter.set_action(actions)
        before_q,before_v,effort=p.advance();m=p.measure()
        if step<settle:init_nonfinite|=~m['finite']
        else:
            c=legacy_failure_codes(m);first=torch.where(first==0,c,first)
            inc=spin_increment(previous,m['object_state'][:,3:7]);previous=m['object_state'][:,3:7].clone()
            valid=first==0;net+=torch.where(valid,inc,0.);peak=torch.maximum(peak,net)
            backward+=torch.where(valid,(-inc).clamp_min(0),0.);valid_steps+=valid
            maxima=torch.maximum(maxima,torch.stack([m[k] for k in ('drift_mm','tilt_deg','max_speed','max_normal','coupling_error')],dim=-1))
        c=task.failure_codes(m,p.origins,control_boundary=(step+1)%p.adapter.steps_per_control==0)
        task_first=torch.where(task_first==0,c,task_first)
        if step>=planned-tail:support+=(m['support_groups']>=2)
        frames.append({k:v.detach().cpu().numpy().copy() for k,v in dict(m,
            joint_pos_before=before_q,joint_vel_before=before_v,action=actions,commands=p.adapter.commands,
            motor_effort_requested=effort,actuator_effort_forwarded=tensor(p.hand.actuators.applied_effort),
            first_failure=first,workspace_first_failure=task_first,net_angle_rad=net,
            phase=torch.full_like(first,int(step>=settle))).items()})
        record['physics_steps']=step+1
        if len(frames)==100:flush()
        if (step+1)%500==0:save('screening')
    flush()
    accepted=(first==0)&(task_first==0)&(support>=.9*tail)
    scale_ids=np.repeat(np.arange(groups),count);candidate_ids=np.tile(np.arange(count),groups)
    retained=accepted.cpu().numpy();counts=[int(retained[scale_ids==i].sum()) for i in range(groups)]
    np.savez(out/'selection.npz',accepted=retained,scale_id=scale_ids,candidate_id=candidate_ids,
        first_failure=first.cpu().numpy(),workspace_first_failure=task_first.cpu().numpy(),
        maxima=maxima.cpu().numpy(),support_fraction=support.cpu().numpy()/tail,
        valid_s=valid_steps.cpu().numpy()*dt,net_deg=torch.rad2deg(net).cpu().numpy(),
        peak_deg=torch.rad2deg(peak).cpu().numpy(),backward_deg=torch.rad2deg(backward).cpu().numpy())
    final_obj=m['object_state'].cpu().numpy().copy();final_obj[:,:3]-=p.origins.cpu().numpy()
    path=out/'grasp_cache.npz'
    np.savez_compressed(path,q=m['joint_pos'].cpu().numpy()[retained],object_state=final_obj[retained],
        commands=p.adapter.commands.cpu().numpy()[retained],scale_id=scale_ids[retained],
        logical_scale=np.array(cfg['scale_centers'])[scale_ids[retained]],candidate_id=candidate_ids[retained],
        joint_names=np.array(names),motor_names=np.array(motors))
    record.update(counts=counts,accepted_total=sum(counts),cache_path=str(path.relative_to(ROOT)),
        cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scale_centers=cfg['scale_centers'],
        nominal_scene=cfg['nominal_scene'],trace_chunks=chunks,
        first_failure_counts=dict(Counter(first.cpu().tolist())),
        stop_reason='bounded nine-size batch completed' if all(counts) else 'at least one size has no stable cache; dependent training stopped')
    if not all(counts):failed=True
except BaseException as error:
    failed=True;record.update(stop_reason='execution/source/readback error; no retry',
        error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
finally:
    if frames:
        np.savez(out/f'physics_{chunks:03d}.npz',**{k:np.stack([f[k] for f in frames]) for k in frames[0]})
    save('error' if failed and 'error' in record else 'blocked' if failed else 'completed')
    app.close(exit_code=1 if failed else 0)
