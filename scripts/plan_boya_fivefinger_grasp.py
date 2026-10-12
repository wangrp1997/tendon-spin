# Reference: botyard-inhand grasp/generate_grasp.py at
# 204d197a9606fb7266e884f3b2e6110195be01cb (CAD gaps, coupling, joint bounds,
# nonpad/inter-finger clearance). TendonSpin packaged source model at the same
# revision. Port changes: five fingers/16 motors; three bounded geometry-only
# solves, existing MuJoCo3.11 forward position with no dynamics; original axis.
"""Propose one five-finger grasp without running any physics integration."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

from tendonspin.interfaces import ACTION_NAMES


ROOT = Path(__file__).resolve().parents[1]
FINGERS = ('th', 'ff', 'mf', 'rf', 'lf')


class Geometry:
    def __init__(self):
        self.contract = json.loads((ROOT/'docs/data/boya_native_contract.json').read_text())
        self.model = mujoco.MjModel.from_xml_path(str(ROOT/'assets/grasp/scene.xml'))
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        for joint in self.contract['joints']:
            d.qpos[m.joint(joint['name']).qposadr[0]] = joint['qpos']
        self.qids = [m.joint(n).qposadr[0] for n in ACTION_NAMES]
        self.q0 = d.qpos[self.qids].copy()
        self.pos0 = np.array(self.contract['object']['pos'])
        self.quat = np.array(self.contract['object']['quat'])
        self.qa = m.joint('cylinder_free').qposadr[0]
        self.obj = m.geom('cylinder_geom').id
        self.pads = [m.geom(f+'distal_collision').id for f in FINGERS]
        self.bounds = np.array([m.joint(n).range for n in ACTION_NAMES], dtype=float)
        self.bounds[:,0] += .002
        self.bounds[:,1] -= .002
        active = [i for i in range(m.ngeom) if m.geom_contype[i] and i not in
                  (self.obj, m.geom('floor_collision').id)]
        self.nonpads = [g for g in active if g not in self.pads]
        self.pairs = []
        for index, a in enumerate(active):
            na = m.body(m.geom_bodyid[a]).name
            for b in active[index+1:]:
                nb = m.body(m.geom_bodyid[b]).name
                if na[:2] in FINGERS and nb[:2] in FINGERS and na[:2] != nb[:2]:
                    if not (na.endswith('base') and nb.endswith('base')):
                        self.pairs.append((a,b))
        self.calls = 0

    def distance(self, a, b, line=None):
        return mujoco.mj_geomDistance(self.model, self.data, a, b, .2, line)

    def measure(self, q, position, dimensions):
        self.calls += 1
        m, d = self.model, self.data
        d.qpos[self.qids] = q
        for f in FINGERS[1:]:
            d.qpos[m.joint(f.upper()+'J1').qposadr[0]] = d.qpos[m.joint(f.upper()+'J2').qposadr[0]]
        d.qpos[self.qa:self.qa+3] = position
        d.qpos[self.qa+3:self.qa+7] = self.quat
        m.geom_size[self.obj,:2] = np.array(dimensions)/2
        # mj_kinematics alone leaves moving mesh BVH bounds stale. Forward
        # position updates geometry/collision structures without a time step.
        mujoco.mj_fwdPosition(m,d)
        lines = np.zeros((5,6))
        gaps = np.array([self.distance(g,self.obj,lines[i]) for i,g in enumerate(self.pads)])
        local = []
        for i,g in enumerate(self.pads):
            body = m.geom_bodyid[g]
            local.append(d.xmat[body].reshape(3,3).T@(lines[i,:3]-d.xpos[body]))
        local = np.array(local)
        axis = d.geom_xmat[self.obj].reshape(3,3)[:,2]
        axial = (lines[:,:3]-position)@axis
        nonpad = np.array([self.distance(g,self.obj) for g in self.nonpads])
        clearance = np.array([self.distance(a,b) for a,b in self.pairs])
        return dict(gaps=gaps,lines=lines,local=local,axial=axial,
                    nonpad=nonpad,clearance=clearance)

    @staticmethod
    def acceptable(metrics, dimensions):
        local = metrics['local']
        sign = np.array([-1,1,1,1,1])
        return bool(np.max(np.abs(metrics['gaps']))<=.0002
            and np.min(metrics['gaps'])>=-.00015
            and min(metrics['nonpad'].min(),metrics['clearance'].min())>=-.0002
            and np.all(sign*local[:,1]>=0)
            and np.all((local[:,2]>=.006)&(local[:,2]<=.032))
            and np.all(np.abs(local[:,0])<=.009)
            and np.max(np.abs(metrics['axial']))<=dimensions[1]/2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    out = args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    started = time.monotonic();g = Geometry()
    original = g.measure(g.q0,g.pos0,[.04,.032])
    record = dict(kind='geometry proposal only',physics_steps=0,training_actions=0,
        mujoco_geometry_version=mujoco.__version__,stages=[],selected=None,
        original_gaps_mm=(original['gaps']*1000).tolist(),action_names=list(ACTION_NAMES))
    record['sources'] = []
    for name in ('scripts/plan_boya_fivefinger_grasp.py','docs/data/boya_native_contract.json',
                 'assets/grasp/scene.xml','tendonspin/interfaces.py',
                 'docs/experiments/2026-10-12-fivefinger-grasp/PROTOCOL.md'):
        content=(ROOT/name).read_bytes()
        record['sources'].append(dict(path=name,sha256=hashlib.sha256(content).hexdigest()))
    q = g.q0.copy();q[13:] = [0.,1.,.45]
    position = g.pos0.copy();dimensions = np.array([.04,.032])
    for stage in (1,2,3):
        calls = g.calls;stage_started=time.monotonic()
        if stage==1:
            x0=q[13:];lo=g.bounds[13:,0];hi=g.bounds[13:,1]
        else:
            x0=np.r_[q,(position-g.pos0)*1000]
            lo=np.r_[g.bounds[:,0],[-15.]*3];hi=np.r_[g.bounds[:,1],[15.]*3]
            if stage==3:
                x0=np.r_[x0,dimensions*1000];lo=np.r_[lo,[36.,32.]];hi=np.r_[hi,[56.,70.]]
        def unpack(x):
            if stage==1:
                pose=g.q0.copy();pose[13:]=x
                return pose,g.pos0.copy(),np.array([.04,.032])
            return x[:16],g.pos0+x[16:19]/1000,(x[19:21]/1000 if stage==3 else np.array([.04,.032]))
        def residual(x):
            pose,pos,dims=unpack(x);v=g.measure(pose,pos,dims);local=v['local']
            sign=np.array([-1,1,1,1,1])
            penalty=np.r_[np.maximum(.0003-v['nonpad'],0),np.maximum(.0003-v['clearance'],0),
                np.maximum(-sign*local[:,1],0),np.maximum(.006-local[:,2],0),
                np.maximum(local[:,2]-.032,0),np.maximum(np.abs(local[:,0])-.009,0),
                np.maximum(np.abs(v['axial'])-(dims[1]/2-.0005),0)]
            regularization=np.r_[.005*(pose-g.q0),.005*(pos-g.pos0)*1000,
                                 .005*(dims-np.array([.04,.032]))*1000]
            return np.r_[(v['gaps']+.0001)*1000,penalty*1000,regularization]
        x0=np.clip(x0,lo+1e-8,hi-1e-8)
        opt=least_squares(residual,x0,bounds=(lo,hi),max_nfev=180,
                          ftol=1e-9,xtol=1e-9,gtol=1e-9,diff_step=1e-5)
        q,position,dimensions=unpack(opt.x);metrics=g.measure(q,position,dimensions)
        entry=dict(stage=stage,optimizer_success=bool(opt.success),optimizer_message=opt.message,
            nfev=opt.nfev,geometry_calls=g.calls-calls,cost=float(opt.cost),wall_s=time.monotonic()-stage_started,
            q=q.tolist(),object_position=position.tolist(),object_quat_wxyz=g.quat.tolist(),
            diameter_mm=float(dimensions[0]*1000),height_mm=float(dimensions[1]*1000),mass_kg=.05,
            gaps_mm=(metrics['gaps']*1000).tolist(),nearest_points=metrics['lines'].tolist(),
            contact_body_local=metrics['local'].tolist(),
            minimum_nonpad_gap_mm=float(metrics['nonpad'].min()*1000),
            minimum_interfinger_gap_mm=float(metrics['clearance'].min()*1000),
            geometric_candidate=Geometry.acceptable(metrics,dimensions))
        record['stages'].append(entry)
        print(json.dumps({k:entry[k] for k in ('stage','gaps_mm','minimum_nonpad_gap_mm','minimum_interfinger_gap_mm','geometric_candidate','wall_s')}),flush=True)
        if entry['geometric_candidate']:
            record['selected']=entry;break
    record.update(status='completed',wall_s=time.monotonic()-started,geometry_calls=g.calls)
    (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
