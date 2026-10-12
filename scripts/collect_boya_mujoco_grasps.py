# Reuse: TendonSpin generate_boya_grasps.py at6089025 (Hora-derived +/-0.25rad
# sampling, fixed preload bias, two-phase old cache gates); native optional
# grasp restore/measurement at0a610bf. Hora v0.0.1 grasp task is MIT licensed.
# Changes: native patchedMuJoCo execution, one30s antecedent hold, four CPU
# processes, explicit native-state cache and per-candidate raw physics archives.
# Additional port: parameterized small-angle sampling/initial-joint exclusions
# and explicit reuse of an already executed, identity-bound profile30s hold.
"""Verify one frozen30s grasp, then collect one bounded native grasp cache."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import mujoco
import numpy as np

from tendonspin.interfaces import ACTION_NAMES
from tendonspin.physics.grasp_profiles import grasp_profile,load_native_grasp
from verify_boya_mujoco_grasp import ROOT,ENGINE_SHA,digest,write,screen,portable

PROTOCOL='docs/experiments/2026-10-12-fivefinger-mujoco/PROTOCOL_HOLD30_CACHE.md'
_MODEL=None
_PLAN=None
_STATE=None
_BASE_Q=None
_BIAS=None


def worker_init(root,profile_name):
    global _MODEL,_PLAN,_STATE,_BASE_Q,_BIAS
    _MODEL,d,spec=load_native_grasp(root,profile_name)
    _PLAN=json.loads((Path(root)/spec['plan']).read_text())
    signature=mujoco.mjtState.mjSTATE_INTEGRATION
    _STATE=np.empty(mujoco.mj_stateSize(_MODEL,signature));mujoco.mj_getState(_MODEL,d,_STATE,signature)
    _BASE_Q=d.qpos.copy()
    _BIAS=d.ctrl-np.array([d.qpos[_MODEL.jnt_qposadr[j]] for j in _MODEL.actuator_trnid[:,0]])


def candidate(task):
    m=_MODEL;d=mujoco.MjData(m);signature=mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(m,d,_STATE,signature)
    active_qa=[m.joint(n).qposadr[0] for n in ACTION_NAMES]
    if not task['anchor']:
        active_a=[m.actuator(n).id for n in ACTION_NAMES]
        d.qpos[active_qa]=np.clip(_BASE_Q[active_qa]+np.array(task['perturb_rad']),
            m.actuator_ctrlrange[active_a,0],m.actuator_ctrlrange[active_a,1])
        for f in ('FF','MF','RF','LF'):d.qpos[m.joint(f+'J1').qposadr[0]]=d.qpos[m.joint(f+'J2').qposadr[0]]
        da=m.joint('cylinder_free').dofadr[0];d.qvel[:da]=0.;d.qacc_warmstart[:]=0.;d.time=0.
        qmotor=np.array([d.qpos[m.jnt_qposadr[j]] for j in m.actuator_trnid[:,0]])
        new_ctrl=np.clip(qmotor+_BIAS,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        for aid in active_a:d.ctrl[aid]=new_ctrl[aid]
        mujoco.mj_forward(m,d)
    else:
        mujoco.mj_forward(m,d);mujoco.mj_setState(m,d,_STATE,signature)
    out=Path(task['out']);out.mkdir(parents=True,exist_ok=False)
    initial=np.empty_like(_STATE);mujoco.mj_getState(m,d,initial,signature)
    np.savez_compressed(out/'initial_state.npz',qpos=d.qpos.copy(),qvel=d.qvel.copy(),ctrl=d.ctrl.copy(),
        integration_state=initial,perturb_rad=np.array(task['perturb_rad']),anchor=np.array(task['anchor']))
    result=screen(m,d,_PLAN,out,'episode',quiet=True)
    result.update(candidate_id=task['candidate_id'],batch_id=task['batch_id'],anchor=task['anchor'])
    write(out/'selection.json',result)
    payload=None
    if result['accepted'] and not task['anchor']:
        state=np.empty_like(_STATE);mujoco.mj_getState(m,d,state,signature)
        qa=m.joint('cylinder_free').qposadr[0];da=m.joint('cylinder_free').dofadr[0]
        q=d.qpos[qa+3:qa+7].copy()
        from scipy.spatial.transform import Rotation
        world_ang=Rotation.from_quat(q[[1,2,3,0]]).apply(d.qvel[da+3:da+6])
        payload=dict(q=d.qpos[:qa].copy(),v=d.qvel[:da].copy(),commands=d.ctrl.copy(),
            object_state_wxyz=np.r_[d.qpos[qa:qa+7],d.qvel[da:da+3],world_ang],
            integration_state=state,candidate_id=task['candidate_id'],batch_id=task['batch_id'])
    return result,payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--grasp-profile',default='boya_original')
    object_profiles=json.loads((ROOT/'assets/objects/profiles.json').read_text())['profiles']
    p.add_argument('--object-profile',choices=tuple(object_profiles),default='boya_original')
    p.add_argument('--batches',type=int,default=8);p.add_argument('--candidates-per-batch',type=int,default=64)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--hold-seconds',type=float,default=30.);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--perturb-degrees',type=float,default=float(np.degrees(.25)))
    p.add_argument('--fixed-initial-joints',nargs='*',choices=ACTION_NAMES,default=(),
        help='Exclude joints from INITIAL sampling only; their actuators remain active')
    p.add_argument('--reuse-profile-hold',action='store_true',
        help='Reuse an executed profile hold with matching exported-state/scene hashes; no additional hold episode')
    a=p.parse_args()
    if not np.isfinite(a.perturb_degrees) or a.perturb_degrees<=0:raise ValueError('Perturbation must be finite and positive')
    if a.batches<1 or a.candidates_per_batch<2 or a.workers<1:raise ValueError('Invalid fixed candidate budget/workers')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    spec=grasp_profile(ROOT,a.grasp_profile)
    if spec['object_profile']!=a.object_profile:raise ValueError('Explicit object and grasp profiles must agree')
    if 'plan' not in spec:raise ValueError('This collection route requires an explicit geometry plan')
    start=time.monotonic();plan=json.loads((ROOT/spec['plan']).read_text())
    rec=dict(protocol=PROTOCOL,status='starting',grasp_profile=a.grasp_profile,object_profile=a.object_profile,
        scene_sha256=spec['scene_sha256'],engine_sha256=ENGINE_SHA,state_sha256=digest(ROOT/spec['state']),
        seed=a.seed,workers=a.workers,batches=a.batches,candidates_per_batch=a.candidates_per_batch,
        requested_random=a.batches*(a.candidates_per_batch-1),requested_anchors=a.batches,
        perturb_degrees=a.perturb_degrees,fixed_initial_joints=list(a.fixed_initial_joints),
        reuse_profile_hold=a.reuse_profile_hold,
        protocol_overrides='Explicit runtime sampling parameters; optional identity-bound antecedent reuse. Physical gates/controller unchanged.',
        training_actions=0,sources=[],candidates=[],accepted_random=0,completed_candidates=0,
        original_evaluation_modified=False,stop_reason=None)
    sources=['scripts/collect_boya_mujoco_grasps.py','scripts/verify_boya_mujoco_grasp.py',
        'tendonspin/physics/grasp_profiles.py','tendonspin/physics/metrics.py','tendonspin/interfaces.py',
        PROTOCOL,'assets/grasp/profiles.json',spec['scene'],spec['state'],spec['plan']]
    if a.reuse_profile_hold:sources.append(spec['evidence'])
    for name in sources:
        path=ROOT/name;rec['sources'].append(dict(path=name,sha256=digest(path)))
        snapshot=out/'sources'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(path.read_bytes())
    def save(status):
        rec['status']=status;rec['wall_s']=time.monotonic()-start;write(out/'result.json',rec)
        print(json.dumps({k:rec.get(k) for k in ('status','completed_candidates','accepted_random','stop_reason','wall_s')}),flush=True)
    save('hold30 starting')
    m,d,_=load_native_grasp(ROOT,a.grasp_profile)
    if a.reuse_profile_hold:
        evidence=json.loads((ROOT/spec['evidence']).read_text());export=evidence['export'];hold=evidence['hold']
        if export['state_sha256']!=rec['state_sha256'] or export['scene_sha256']!=spec['scene_sha256']:
            raise RuntimeError('Prior hold evidence does not identify this exported state/scene')
        if not hold['accepted'] or hold['valid_formal_s']<a.hold_seconds:
            raise RuntimeError('Prior hold is not an accepted requested-duration antecedent')
        rec['hold30']=dict(hold,reused=True,evidence=spec['evidence'],new_antecedent_physics_steps=0)
    else:
        hold_dir=out/'hold30';hold_dir.mkdir();write(hold_dir/'initial_state.json',portable(m,d,plan,ROOT/spec['scene']))
        rec['hold30']=screen(m,d,plan,hold_dir,'hold30',seconds=a.hold_seconds,initialization_s=0.)
    if not rec['hold30']['accepted']:
        rec['stop_reason']='30s antecedent hold failed; collection not launched';save('completed');return
    save('hold30 passed; collecting fixed candidate budget')
    # Freeze source/config identity before starting dependent independent episodes.
    for item in rec['sources']:
        if digest(ROOT/item['path'])!=item['sha256']:raise RuntimeError('Source/config changed before collection')
    rng=np.random.default_rng(a.seed);tasks=[]
    for batch in range(a.batches):
        span=np.radians(a.perturb_degrees)
        perturb=rng.uniform(-span,span,(a.candidates_per_batch,16));perturb[0]=0.
        fixed=[ACTION_NAMES.index(name) for name in a.fixed_initial_joints]
        if fixed:perturb[:,fixed]=0.
        for local in range(a.candidates_per_batch):
            index=batch*a.candidates_per_batch+local
            tasks.append(dict(candidate_id=index,batch_id=batch,anchor=local==0,
                perturb_rad=perturb[local].tolist(),out=str(out/'candidates'/f'{index:04d}')))
    write(out/'candidate_plan.json',dict(seed=a.seed,tasks=tasks))
    accepted=[];pool=None;failed=False
    try:
        pool=ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=worker_init,initargs=(str(ROOT),a.grasp_profile))
        futures={pool.submit(candidate,task):task for task in tasks}
        for future in as_completed(futures):
            result,payload=future.result();rec['candidates'].append(result);rec['completed_candidates']+=1
            if payload is not None:accepted.append(payload);rec['accepted_random']+=1
            if rec['completed_candidates']%32==0:save('collecting')
    except BaseException as error:
        failed=True;rec['stop_reason']='execution/manual error; no retry'
        rec['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        if pool is not None:
            for f in futures:f.cancel()
    finally:
        if pool is not None:pool.shutdown(wait=True,cancel_futures=failed)
    rec['candidates'].sort(key=lambda x:x['candidate_id'])
    random_results=[r for r in rec['candidates'] if not r['anchor']]
    good=[r for r in random_results if r['accepted']]
    rec['random_stop_reasons']=dict(Counter(r['stop_reason'] for r in random_results))
    rec['nominal_anchors_passed']=sum(r['accepted'] for r in rec['candidates'] if r['anchor'])
    rec['accepted_min_finger_count_distribution']=dict(Counter(r['final_point1s_min_finger_count'] for r in good))
    rec['accepted_final_point1s_allfive_90percent_count']=sum(r['final_point1s_allfive_contact_fraction']>=.9 for r in good)
    if accepted:
        accepted.sort(key=lambda x:x['candidate_id'])
        joined={k:np.array([s[k] for s in accepted]) for k in accepted[0]}
        joined.update(joint_names=np.array([m.joint(i).name for i in range(m.njnt) if i!=m.joint('cylinder_free').id]),
            motor_names=np.array([m.actuator(i).name for i in range(m.nu)]),engine_sha256=np.array(ENGINE_SHA),
            scene_sha256=np.array(spec['scene_sha256']),state_signature=np.array(int(mujoco.mjtState.mjSTATE_INTEGRATION)),
            quaternion_convention=np.array('WXYZ; linear/angular velocities in world frame'),
            source_profile=np.array(a.grasp_profile))
        np.savez_compressed(out/'grasp_cache.npz',**joined)
        rec['cache_path']=str(out/'grasp_cache.npz');rec['cache_sha256']=digest(out/'grasp_cache.npz')
        rec['distinct_accepted_joint_states']=len(np.unique(np.round(joined['q'],10),axis=0))
        rec['native_cache_schema']='full native integration state; q/v named22, commands named18, object_state_wxyz; not Isaac cache schema'
    rec['stop_reason']=rec['stop_reason'] or 'fixed candidate budget completed'
    save('error' if failed else 'completed')


if __name__=='__main__':main()
