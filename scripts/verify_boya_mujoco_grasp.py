# References: botyard-inhand/grasp/grasp_baseline.py at
# 204d197a9606fb7266e884f3b2e6110195be01cb (native named joints, contact forces,
# saved integration state); TendonSpin generate_boya_grasps.py at6089025
# (two-phase cache admission). Port: selectable grasp/object, five fingers,
# native MuJoCo contact groups, fixed kinematic preload, full diagnostic trace.
"""One parameterized native cache screen and conditional portable-state reload."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tendonspin.interfaces import ACTION_NAMES
from tendonspin.physics.metrics import SpinTracker
from tendonspin.physics.object_geometry import add_object_arguments,resolve_object_geometry
from tendonspin.physics.grasp_profiles import grasp_profile,load_native_grasp

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL='docs/experiments/2026-10-12-fivefinger-mujoco/PROTOCOL_CACHE28.md'
ENGINE_SHA='c0b309bba9a0913718f86fa375ed5003be09ca2131165cee51f9130fbafbf088'
FINGERS=('th','ff','mf','rf','lf')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def build_scene(directory,geometry):
    tree=ET.parse(ROOT/'assets/grasp/scene.xml');root=tree.getroot()
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file',os.path.relpath((ROOT/'assets/grasp'/mesh.get('file')).resolve(),directory))
    body=root.find("./worldbody/body[@name='cylinder']")
    radius,height,mass=geometry['diameter_m']/2,geometry['height_m'],geometry['mass_kg']
    inertial=body.find('inertial');inertial.set('quat','1 0 0 0')
    inertial.set('mass',str(mass));transverse=mass*(3*radius**2+height**2)/12
    inertial.set('diaginertia',f'{transverse:.17g} {transverse:.17g} {mass*radius**2/2:.17g}')
    body.find('geom').set('size',f'{radius:.17g} {height/2:.17g}')
    root.find("./equality/weld[@name='cylinder_support']").set('active','false')
    ET.indent(tree,space='  ');tree.write(directory/'scene.xml',encoding='unicode')
    return mujoco.MjModel.from_xml_path(str(directory/'scene.xml'))


def initialize(m,plan):
    d=mujoco.MjData(m)
    source=json.loads((ROOT/'docs/data/boya_native_contract.json').read_text())
    for j in source['joints']:d.qpos[m.joint(j['name']).qposadr[0]]=j['qpos']
    for name,value in zip(plan['joints'],plan['q'],strict=True):d.qpos[m.joint(name).qposadr[0]]=value
    for f in FINGERS[1:]:d.qpos[m.joint(f.upper()+'J1').qposadr[0]]=d.qpos[m.joint(f.upper()+'J2').qposadr[0]]
    qa=m.joint('cylinder_free').qposadr[0]
    d.qpos[qa:qa+3]=plan['pos'];d.qpos[qa+3:qa+7]=plan['quat']
    d.qvel[:]=0.;d.eq_active[m.equality('cylinder_support').id]=False
    mujoco.mj_forward(m,d)
    # Constant feedforward is encoded in position targets, not an online controller.
    torque=d.qfrc_bias.copy();jac=np.zeros((3,m.nv));jr=np.zeros_like(jac)
    contact_points=np.array(plan['contact_body_local']);forces=np.array(plan['forces'])
    plan_fingers=[f for f in FINGERS if f.upper()+'J4' in plan['joints']]
    for i,f in enumerate(plan_fingers):
        bid=m.body(f+'distal').id
        point=d.xpos[bid]+d.xmat[bid].reshape(3,3)@contact_points[i]
        mujoco.mj_jac(m,d,jac,jr,point,bid);torque+=jac.T@forces[i]
    for f in FINGERS[1:]:torque[m.joint(f.upper()+'J2').dofadr[0]]+=torque[m.joint(f.upper()+'J1').dofadr[0]]
    desired={};clipped={}
    for aid in range(m.nu):
        name=m.actuator(aid).name;jid=m.actuator_trnid[aid,0]
        q=d.qpos[m.jnt_qposadr[jid]]
        target=q+torque[m.jnt_dofadr[jid]]/m.actuator_gainprm[aid,0] if name in ACTION_NAMES else 0.
        desired[name]=float(target);d.ctrl[aid]=np.clip(target,*m.actuator_ctrlrange[aid])
        clipped[name]=float(d.ctrl[aid])
    mujoco.mj_forward(m,d)
    return d,dict(method='fixed q+(native gravity+contact J-transpose force)/kp; slave reflected; original wrists',
        desired_commands=desired,commands=clipped,torque_Nm=torque.tolist())


def owners(m):
    return {g:next((i for i,f in enumerate(FINGERS) if m.body(m.geom_bodyid[g]).name.startswith(f)),5)
            for g in range(m.ngeom) if m.geom_bodyid[g]!=m.body('cylinder').id and g!=m.geom('floor_collision').id}


def measure(m,d,plan,owner):
    obj=m.geom('cylinder_geom').id;qa=m.joint('cylinder_free').qposadr[0];da=m.joint('cylinder_free').dofadr[0]
    group_normals=np.zeros(6);group_force=np.zeros((6,3));link_normal=np.zeros((m.nbody,3))
    raw=[];floor=nonpad=0.;penetration=self_pen=0.
    pads={m.geom(f+'distal_collision').id for f in FINGERS}
    for k,c in enumerate(d.contact):
        cf=np.zeros(6);mujoco.mj_contactForce(m,d,k,cf)
        raw.append(np.r_[c.geom1,c.geom2,c.dist,c.pos,c.frame,cf])
        if c.geom1 in owner and c.geom2 in owner:self_pen=min(self_pen,float(c.dist))
        if obj not in (c.geom1,c.geom2):continue
        other=c.geom1 if c.geom2==obj else c.geom2;sign=1. if c.geom2==obj else -1.
        normal=sign*c.frame[:3]*cf[0];force=sign*c.frame.reshape(3,3).T@cf[:3]
        penetration=min(penetration,float(c.dist))
        if other in owner:
            link_normal[m.geom_bodyid[other]]+=normal
            group_force[owner[other]]+=force
            if other not in pads:nonpad+=np.linalg.norm(force)
        elif other==m.geom('floor_collision').id:floor+=np.linalg.norm(force)
    for bid in range(m.nbody):
        name=m.body(bid).name;group=next((i for i,f in enumerate(FINGERS) if name.startswith(f)),5)
        group_normals[group]+=np.linalg.norm(link_normal[bid])
    quat=d.qpos[qa+3:qa+7];axis=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()[:,2]
    initial_axis=Rotation.from_quat(np.array(plan['quat'])[[1,2,3,0]]).as_matrix()[:,2]
    drift=np.linalg.norm(d.qpos[qa:qa+3]-plan['pos'])*1000
    tilt=np.degrees(np.arccos(np.clip(axis@initial_axis,-1,1)))
    coupling=max(abs(d.qpos[m.joint(f.upper()+'J1').qposadr[0]]-d.qpos[m.joint(f.upper()+'J2').qposadr[0]]) for f in FINGERS[1:])
    finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(group_force).all())
    metrics=np.array([drift,tilt,np.max(abs(d.qvel[:da])),np.linalg.norm(link_normal,axis=1).max(),coupling])
    violations=np.r_[not finite,metrics>np.array([5.,15.,100.,12.,.05])]
    return dict(metrics=metrics,violations=violations,finite=finite,group_normals_N=group_normals,
        group_forces_N=group_force,contact_count=d.ncon,contacts=np.array(raw).reshape(-1,21),
        floor_force_N=floor,nonpad_force_N=nonpad,object_penetration_m=-penetration,self_penetration_m=-self_pen,
        solver_warnings=d.warning.number.copy(),support_active=bool(d.eq_active[m.equality('cylinder_support').id]),
        external_force=bool(np.any(d.xfrc_applied) or np.any(d.qfrc_applied)))


def screen(m,d,plan,out,label,*,seconds=1.,initialization_s=.5,quiet=False):
    if not 0<=initialization_s<seconds:raise ValueError('Initialization must be shorter than the episode')
    dt=m.opt.timestep;planned=round(seconds/dt);settle=round(initialization_s/dt);tail=round(.1/dt)
    owner=owners(m);reasons=['nonfinite','drift >5mm','axis tilt >15deg','speed >100rad/s','link normal >12N','mimic error >.05rad']
    qa=m.joint('cylinder_free').qposadr[0];records=[];raw=[];offsets=[0]
    initial_max=np.zeros(5);formal_max=np.zeros(5);init_bad=np.zeros(6,dtype=bool)
    tracker=None;peak=0.;valid_steps=0;support_steps=0;stop='time limit';first_failure=None
    formal_start=None
    formal_start_state=None
    start=time.monotonic()
    for step in range(planned):
        if step==settle:
            tracker=SpinTracker(d.qpos[qa+3:qa+7].copy())
            formal_start=measure(m,d,plan,owner)
            formal_start_state=dict(qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy())
            formal_max=np.maximum(formal_max,formal_start['metrics'])
            if formal_start['violations'].any():
                first_failure=reasons[int(np.flatnonzero(formal_start['violations'])[0])]
                stop='formal-start '+first_failure;break
        before=d.qpos.copy();mujoco.mj_step(m,d);v=measure(m,d,plan,owner)
        formal=step>=settle
        if formal:
            formal_max=np.maximum(formal_max,v['metrics'])
            if v['violations'].any():first_failure=reasons[int(np.flatnonzero(v['violations'])[0])]
            elif tracker is not None:
                tracker.update(d.qpos[qa+3:qa+7]);valid_steps+=1;peak=max(peak,float(np.degrees(tracker.net)))
        else:initial_max=np.maximum(initial_max,v['metrics']);init_bad|=v['violations']
        if step>=planned-tail:support_steps+=int((v['group_normals_N']>1e-6).sum()>=2)
        signature=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(m,signature));mujoco.mj_getState(m,d,state,signature)
        records.append(dict(time_s=float(d.time),phase=int(formal),qpos_before=before,qpos=d.qpos.copy(),qvel=d.qvel.copy(),
            ctrl=d.ctrl.copy(),actuator_force=d.actuator_force.copy(),integration_state=state,
            policy_action=np.zeros(16),metrics=v['metrics'],violations=v['violations'],
            group_normals_N=v['group_normals_N'],group_forces_N=v['group_forces_N'],
            support_active=v['support_active'],external_force=v['external_force'],solver_warnings=v['solver_warnings'],
            floor_force_N=v['floor_force_N'],nonpad_force_N=v['nonpad_force_N'],
            object_penetration_m=v['object_penetration_m'],self_penetration_m=v['self_penetration_m']))
        raw.append(v['contacts']);offsets.append(offsets[-1]+len(v['contacts']))
        if not v['finite']:stop='nonfinite initialization' if not formal else first_failure;break
        if formal and first_failure is not None:stop=first_failure;break
    completed=len(records)==planned
    support_fraction=support_steps/tail
    accepted=bool(completed and first_failure is None and not init_bad[0] and support_fraction>=.9)
    if completed and not accepted and first_failure is None:stop='insufficient observed support'
    if formal_start is not None:
        np.savez_compressed(out/f'{label}_formal_start.npz',
            **formal_start_state,
            metrics=formal_start['metrics'],violations=formal_start['violations'],
            group_normals_N=formal_start['group_normals_N'])
    arrays={k:np.array([r[k] for r in records]) for k in records[0]} if records else {}
    np.savez_compressed(out/f'{label}_execution.npz',**arrays,contact_offsets=np.array(offsets),
        contacts=np.concatenate(raw,axis=0) if raw else np.empty((0,21)))
    forces=arrays.get('group_normals_N',np.empty((0,6)));contacts=forces>1e-6;formal_contacts=contacts[settle:]
    result=dict(accepted=accepted,stop_reason=stop,physics_steps=len(records),actual_s=len(records)*dt,
        initialization_s=min(len(records),settle)*dt,formal_s=max(0,len(records)-settle)*dt,
        valid_formal_s=valid_steps*dt,initialization_maxima=initial_max.tolist(),formal_maxima=formal_max.tolist(),
        metric_order=['drift_mm','axis_tilt_deg','joint_speed_rad_s','link_normal_N','mimic_error_rad'],
        initialization_violations={r:bool(b) for r,b in zip(reasons,init_bad)},
        last_point1s_two_group_fraction=float(support_fraction),group_order=list(FINGERS)+['palm'],
        formal_finger_contact_fractions=formal_contacts[:,:5].mean(0).tolist() if len(formal_contacts) else None,
        formal_allfive_contact_fraction=float(formal_contacts[:,:5].all(1).mean()) if len(formal_contacts) else None,
        final_point1s_finger_contact_fractions=contacts[-tail:,:5].mean(0).tolist() if completed else None,
        final_point1s_allfive_contact_fraction=float(contacts[-tail:,:5].all(1).mean()) if completed else None,
        final_point1s_min_finger_count=int(contacts[-tail:,:5].sum(1).min()) if completed else None,
        formal_mean_group_normal_N=forces[settle:].mean(0).tolist() if len(formal_contacts) else None,
        formal_min_group_normal_N=forces[settle:].min(0).tolist() if len(formal_contacts) else None,
        net_deg=float(np.degrees(tracker.net)) if tracker else None,peak_deg=peak if tracker else None,
        backward_deg=float(np.degrees(tracker.backward)) if tracker else None,
        raw_trace=str(out/f'{label}_execution.npz'),wall_s=time.monotonic()-start,
        requested_s=seconds,requested_initialization_s=initialization_s,
        formal_start_violations=formal_start['violations'].tolist() if formal_start is not None else None,
        warning='nominal holding only; five-finger touch is diagnostic; raw contacts refer to mj_step pre-integration geometry')
    write(out/f'{label}_result.json',result)
    if not quiet:print(json.dumps(dict(phase=label,**result)),flush=True)
    return result


def portable(m,d,plan,scene_path):
    signature=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(m,signature))
    mujoco.mj_getState(m,d,state,signature);qa=m.joint('cylinder_free').qposadr[0];da=m.joint('cylinder_free').dofadr[0]
    return dict(schema='boya_named_mujoco_grasp_v1',scene_sha256=digest(scene_path),engine_sha256=ENGINE_SHA,
        mujoco_version=mujoco.__version__,validated=False,acceptance_profile='cache28',
        joint_pos={m.joint(i).name:float(d.qpos[m.jnt_qposadr[i]]) for i in range(m.njnt) if i!=m.joint('cylinder_free').id},
        joint_vel={m.joint(i).name:float(d.qvel[m.jnt_dofadr[i]]) for i in range(m.njnt) if i!=m.joint('cylinder_free').id},
        commands={m.actuator(i).name:float(d.ctrl[i]) for i in range(m.nu)},
        object_position=d.qpos[qa:qa+3].tolist(),object_quat_wxyz=d.qpos[qa+3:qa+7].tolist(),
        object_velocity_mujoco=d.qvel[da:da+6].tolist(),object=plan['object'],
        signature=int(signature),integration_state=state.tolist(),qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),ctrl=d.ctrl.tolist(),
        original_reference=dict(pos=plan['pos'],quat=plan['quat']),support_active=False,
        note='Exact full-state restore requires matching engine/model; named state is portable but transfer needs its own validation')


def main():
    p=argparse.ArgumentParser();add_object_arguments(p)
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--grasp-profile',choices=('boya_original','fivefinger_sharpa_cache28_v1'),default='boya_original')
    selection.add_argument('--grasp-plan',type=Path,help='Explicit geometric proposal; otherwise restore selected frozen grasp profile')
    p.add_argument('--acceptance-profile',choices=('cache28',),default='cache28')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    if mujoco.__version__!='3.13.0' or digest(Path(mujoco.__file__).parent/'libmujoco.so.3.13.0')!=ENGINE_SHA:raise RuntimeError('Declared native MuJoCo runtime required')
    geometry=resolve_object_geometry(ROOT,a)
    profile=grasp_profile(ROOT,a.grasp_profile) if a.grasp_plan is None else None
    plan_path=a.grasp_plan if a.grasp_plan is not None else ROOT/profile.get('plan','assets/grasp/source_plan.json')
    plan=json.loads(plan_path.read_text())
    if profile is not None and a.grasp_profile=='boya_original':
        original=json.loads((ROOT/'docs/data/boya_native_contract.json').read_text())['object']
        plan['pos']=original['pos'];plan['quat']=original['quat']
    if geometry['shape']!='cylinder':raise ValueError('This grasp verifier currently supports cylinders')
    expected=plan.get('object',dict(radius=.02,height=.032,mass=.05))
    plan.setdefault('object',expected)
    if not np.allclose([geometry['diameter_m']/2,geometry['height_m'],geometry['mass_kg']],
        [expected['radius'],expected['height'],expected['mass']],rtol=0,atol=1e-12):raise ValueError('Object profile differs from declared grasp plan')
    rec=dict(protocol=PROTOCOL,engine='native patched MuJoCo3.13.0',engine_sha256=ENGINE_SHA,
        object_geometry=geometry,grasp_profile=profile['name'] if profile else 'custom_plan',
        grasp_plan=str(plan_path),grasp_plan_sha256=digest(plan_path),
        training_actions=0,episode_resets=0,original_evaluation_modified=False,sources=[])
    for name in ('scripts/verify_boya_mujoco_grasp.py',PROTOCOL,'assets/grasp/scene.xml','tendonspin/interfaces.py',
        'tendonspin/physics/metrics.py','tendonspin/physics/object_geometry.py','tendonspin/physics/grasp_profiles.py'):
        rec['sources'].append(dict(path=name,sha256=digest(ROOT/name)))
        snapshot=out/'sources'/name;snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes((ROOT/name).read_bytes())
    if profile is None:
        m=build_scene(out,geometry);d,rec['preload']=initialize(m,plan);scene_path=out/'scene.xml'
    else:
        m,d,profile=load_native_grasp(ROOT,a.grasp_profile);scene_path=ROOT/profile['scene']
        rec['preload']=dict(method='restore frozen full integration state and motor targets; no new preload')
    rec['scene_sha256']=digest(scene_path);write(out/'initial_state.json',portable(m,d,plan,scene_path))
    rec['nominal']=screen(m,d,plan,out,'nominal')
    if rec['nominal']['accepted']:
        saved=portable(m,d,plan,scene_path);write(out/'handoff_state.json',saved)
        reload_model=mujoco.MjModel.from_xml_path(str(scene_path));reload_data=mujoco.MjData(reload_model)
        mujoco.mj_setState(reload_model,reload_data,np.array(saved['integration_state']),mujoco.mjtState(saved['signature']))
        # Restore exactly; do not run a warm-up forward that alters full state.
        actual=np.empty(mujoco.mj_stateSize(reload_model,mujoco.mjtState(saved['signature'])))
        mujoco.mj_getState(reload_model,reload_data,actual,mujoco.mjtState(saved['signature']))
        rec['reload_initial_state_max_error']=float(abs(actual-np.array(saved['integration_state'])).max())
        if rec['reload_initial_state_max_error']!=0:raise RuntimeError('Exact native handoff restoration failed')
        rec['reload']=screen(reload_model,reload_data,plan,out,'reload')
        saved['validated']=rec['reload']['accepted'];saved['validation']=dict(nominal=rec['nominal'],reload=rec['reload'])
        write(out/'handoff_state.json',saved)
    rec['accepted']=bool(rec['nominal']['accepted'] and rec.get('reload',{}).get('accepted',False))
    rec['status']='completed';write(out/'result.json',rec)


if __name__=='__main__':main()
