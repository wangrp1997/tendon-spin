# Reuse TendonSpin3a1390d verify_boya_mujoco_grasp.py and DecoupledPlanner force
# initializer, adapted from botyard-inhand204d197a9606fb7266e884f3b2e6110195be01cb
# (ssh://git@git.nrs-lab.com:10022/wangrenpeng25p/botyard-inhand.git).
# Change: explicit small object translation toward LF and small LF flexion,
# followed by exactly one frozen-target native hold; no automatic retries.
"""Apply one parameterized LF adjustment to an existing settled native grasp."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from tendonspin.interfaces import ACTION_NAMES
from plan_boya_fivefinger_decoupled import force_solve
from verify_boya_mujoco_grasp import (ROOT,ENGINE_SHA,build_scene,initialize,
    screen,portable,write,digest)

PROTOCOL='docs/experiments/2026-10-12-fivefinger-side/PROTOCOL_SHIFT.md'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--shift-mm',type=float,default=.15)
    p.add_argument('--lf-curl-deg',type=float,default=1.72)
    p.add_argument('--hold-seconds',type=float,default=30.)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    src=a.source.resolve();state=json.loads((src/'state.json').read_text())
    engine=Path(mujoco.__file__).parent/f'libmujoco.so.{mujoco.__version__}'
    if mujoco.__version__!='3.13.0' or digest(engine)!=ENGINE_SHA:
        raise RuntimeError('Declared native MuJoCo engine required')
    geometry=dict(shape='cylinder',diameter_m=.048,height_m=.06,mass_kg=.05)
    m=build_scene(out,geometry);d=mujoco.MjData(m)
    d.qpos[:]=state['qpos'];d.eq_active[m.equality('cylinder_support').id]=False
    mujoco.mj_fwdPosition(m,d)
    obj=m.geom('cylinder_geom').id;qa=m.joint('cylinder_free').qposadr[0]
    line=np.zeros(6);before_gap=mujoco.mj_geomDistance(m,d,m.geom('lfdistal_collision').id,obj,.2,line)
    axis=d.geom_xmat[obj].reshape(3,3)[:,2];direction=line[:3]-d.qpos[qa:qa+3]
    direction-=direction@axis*axis;direction/=np.linalg.norm(direction)
    d.qpos[qa:qa+3]+=direction*a.shift_mm/1000
    changes={}
    for name in ('LFJ3','LFJ2'):
        adr=m.joint(name).qposadr[0];old=d.qpos[adr]
        d.qpos[adr]=np.clip(old+np.radians(a.lf_curl_deg),*m.joint(name).range)
        changes[name]=float(np.degrees(d.qpos[adr]-old))
    for f in ('FF','MF','RF','LF'):
        d.qpos[m.joint(f+'J1').qposadr[0]]=d.qpos[m.joint(f+'J2').qposadr[0]]
    mujoco.mj_fwdPosition(m,d)
    points=[];local=[];gaps=[]
    for f in ('th','ff','mf','rf','lf'):
        g=m.geom(f+'distal_collision').id;line=np.zeros(6)
        gaps.append(mujoco.mj_geomDistance(m,d,g,obj,.2,line));points.append(line[:3].copy())
        bid=m.geom_bodyid[g];local.append(d.xmat[bid].reshape(3,3).T@(line[:3]-d.xpos[bid]))
    points=np.array(points);pos=d.qpos[qa:qa+3].copy()
    axial=(points-pos)@axis;radial=points-pos-axial[:,None]*axis
    normals=-radial/np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-10)
    force,diagnostic=force_solve(points,normals,pos,normals+np.array([0,0,.4905/5]))
    plan=dict(joints=list(ACTION_NAMES),q=[float(d.qpos[m.joint(n).qposadr[0]]) for n in ACTION_NAMES],
        pos=pos.tolist(),quat=d.qpos[qa+3:qa+7].tolist(),points=points.tolist(),
        normals=normals.tolist(),contact_body_local=np.array(local).tolist(),
        forces=force.reshape(5,3).tolist(),object=dict(shape='cylinder',radius=.024,height=.06,mass=.05),
        validated=False,force_is_admission_gate=False,geometric_gaps_are_admission_gate=False)
    write(out/'plan.json',plan);d,preload=initialize(m,plan)
    write(out/'initial_state.json',portable(m,d,plan,out/'scene.xml'))
    rec=dict(protocol=PROTOCOL,source_state_sha256=digest(src/'state.json'),source_scene_sha256=digest(src/'scene.xml'),
        engine_sha256=ENGINE_SHA,scene_sha256=digest(out/'scene.xml'),geometry=geometry,
        object_shift_mm=a.shift_mm,object_shift_world_mm=(direction*a.shift_mm).tolist(),
        LF_flexion_change_deg=changes,LF_gap_before_mm=before_gap*1000,
        planned_gaps_mm=(np.array(gaps)*1000).tolist(),preload=preload,
        force_initializer_diagnostic=diagnostic,force_is_admission_gate=False,
        episode_resets=0,controller_switches=0,training_actions=0,sources=[])
    for name in ('scripts/adjust_boya_smallfinger_grasp.py','scripts/verify_boya_mujoco_grasp.py',
        'scripts/plan_boya_fivefinger_decoupled.py','tendonspin/physics/metrics.py',PROTOCOL):
        path=ROOT/name;rec['sources'].append(dict(path=name,sha256=digest(path)))
        dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    print(json.dumps(dict(status='one hold starting',shift_mm=a.shift_mm,LF_flexion_change_deg=changes,
        planned_gaps_mm=rec['planned_gaps_mm'])),flush=True)
    rec['hold']=screen(m,d,plan,out,'hold30',seconds=.5+a.hold_seconds,initialization_s=.5)
    rec['observed_fivefinger_hold']=bool(rec['hold']['accepted'] and rec['hold']['formal_allfive_contact_fraction']>=.9)
    if rec['hold']['accepted']:
        saved=portable(m,d,plan,out/'scene.xml');saved['validated']=True
        saved['validation_scope']='One .5s initialization+30s formal native hold; old cache gates; contact fractions in result.json'
        write(out/'state.json',saved)
    write(out/'result.json',rec)
    print(json.dumps(dict(accepted=rec['hold']['accepted'],fivefinger=rec['observed_fivefinger_hold'],
        touch=rec['hold']['formal_finger_contact_fractions'],stop=rec['hold']['stop_reason'])),flush=True)


if __name__=='__main__':main()
