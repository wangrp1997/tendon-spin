# Reuse: TendonSpin3a1390d Geometry/DecoupledPlanner, adapted from
# botyard-inhand204d197a9606fb7266e884f3b2e6110195be01cb grasp/generate_grasp.py.
# Source: ssh://git@git.nrs-lab.com:10022/wangrenpeng25p/botyard-inhand.git
# Changes: whole distal CAD nearest points including lateral pulp, original-pose
# regularization, contact spacing diagnostics; bounded geometry only, no dynamics.
"""One bounded original-pose five-finger side-contact proposal."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from plan_boya_fivefinger_decoupled import DecoupledPlanner, force_solve
from plan_boya_fivefinger_grasp import FINGERS, ROOT
from tendonspin.interfaces import ACTION_NAMES


class SidePlanner(DecoupledPlanner):
    def contacts(self, x):
        self.quat=(Rotation.from_rotvec(x[19:22])*self.base).as_quat()[[3,0,1,2]]
        v=self.measure(x[:16],self.pos0+x[16:19]*.04,self.dimensions)
        points=v['lines'][:,:3]
        pos=self.pos0+x[16:19]*.04
        axis=Rotation.from_quat(self.quat[[1,2,3,0]]).as_matrix()[:,2]
        axial=(points-pos)@axis
        radial=points-pos-axial[:,None]*axis
        v.update(points=points,pos=pos,quat=self.quat.copy(),axis=axis,
                 inward=-radial/np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-10))
        pairs=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
        v['pair_distances']=pairs[np.triu_indices(5,1)]
        return v

    def residual(self, x):
        v=self.contacts(x);local=v['local'];sign=np.array([-1,1,1,1,1])
        # Actual nearest points can move anywhere on the distal palmar/lateral
        # CAD half-surface. No central x-coordinate or frontal-normal target.
        penalties=np.r_[np.maximum(.0003-v['nonpad'],0),
            np.maximum(.0003-v['clearance'],0),
            np.maximum(-.0005-sign*local[:,1],0),
            np.maximum(.006-local[:,2],0),np.maximum(local[:,2]-.032,0),
            np.maximum(np.abs(v['axial'])-.029,0),
            np.maximum(.012-v['pair_distances'],0),
            max(0.,.020-np.ptp(v['axial']))]
        rotation_excess=max(0.,np.linalg.norm(x[19:22])-np.radians(12))
        return np.r_[(v['gaps']-.00005)*1000,penalties*1000,
            rotation_excess*100,.35*(x[:13]-self.q0[:13]),
            .3*x[16:19],1.0*x[19:22]]

    def report_side(self,x):
        v=self.contacts(x);local=v['local'];sign=np.array([-1,1,1,1,1])
        # These are declared geometry screens, not the old physical cache rules.
        gates=dict(gaps=bool(abs(v['gaps']).max()<=.0002 and v['gaps'].min()>=-.00015),
            clearance=bool(min(v['nonpad'].min(),v['clearance'].min())>=.0003-1e-6),
            distal_region=bool((sign*local[:,1]>=-.0005).all() and
                (local[:,2]>=.006).all() and (local[:,2]<=.032).all()),
            cylinder_side=bool(abs(v['axial']).max()<=.029+1e-6),
            spacing=bool(v['pair_distances'].min()>=.012-1e-6 and np.ptp(v['axial'])>=.020-1e-6),
            original_orientation=bool(np.linalg.norm(x[19:22])<=np.radians(12)+1e-5))
        return dict(geometry_pass=all(gates.values()),gates=gates,q=x[:16].tolist(),
            pos=v['pos'].tolist(),quat=v['quat'].tolist(),
            points=v['points'].tolist(),normals=v['inward'].tolist(),
            contact_body_local=local.tolist(),gaps_mm=(v['gaps']*1000).tolist(),
            axial_mm=(v['axial']*1000).tolist(),axial_span_mm=float(np.ptp(v['axial'])*1000),
            minimum_contact_spacing_mm=float(v['pair_distances'].min()*1000),
            minimum_nonpad_gap_mm=float(v['nonpad'].min()*1000),
            minimum_interfinger_gap_mm=float(v['clearance'].min()*1000),
            orientation_change_deg=float(np.degrees(np.linalg.norm(x[19:22]))),
            center_change_mm=float(np.linalg.norm(x[16:19]*.04)*1000),
            max_original13_joint_change_rad=float(abs(x[:13]-self.q0[:13]).max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-profile',choices=('sharpa_recommended',),required=True)
    parser.add_argument('--contact-profile',choices=('distal_palmar_side',),required=True)
    parser.add_argument('--max-nfev',type=int,default=120)
    parser.add_argument('--wall-seconds',type=float,default=120.)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();p=SidePlanner()
    engine=Path(mujoco.__file__).parent/f'libmujoco.so.{mujoco.__version__}'
    engine_sha=hashlib.sha256(engine.read_bytes()).hexdigest()
    if mujoco.__version__!='3.13.0' or engine_sha!='c0b309bba9a0913718f86fa375ed5003be09ca2131165cee51f9130fbafbf088':
        raise RuntimeError('Declared native patched MuJoCo runtime required')
    x=np.r_[p.q0,np.zeros(6)];x[13:16]=[.15,.98,.18]
    lower=np.r_[np.maximum(p.bounds[:13,0],p.q0[:13]-.35),p.bounds[13:,0],[-.3]*3,[-np.radians(12)]*3]
    upper=np.r_[np.minimum(p.bounds[:13,1],p.q0[:13]+.35),p.bounds[13:,1],[.3]*3,[np.radians(12)]*3]
    x=np.clip(x,lower+1e-7,upper-1e-7)
    initial=p.report_side(x);best=[x.copy(),np.linalg.norm(p.residual(x))];last=[start]
    def residual(z):
        if time.monotonic()-start>args.wall_seconds:raise TimeoutError('Declared planning wall budget reached')
        r=p.residual(z);score=np.linalg.norm(r)
        if score<best[1]:best[:]=[z.copy(),score]
        if time.monotonic()-last[0]>15:
            print(json.dumps(dict(status='solving',calls=p.calls,best_residual=float(best[1]),wall_s=time.monotonic()-start)),flush=True);last[0]=time.monotonic()
        return r
    def jac(z):
        baseline=residual(z);j=np.empty((len(baseline),len(z)))
        for k in range(len(z)):
            zz=z.copy();zz[k]+=1e-5;j[:,k]=(residual(zz)-baseline)/1e-5
        return j
    stop='optimizer completed';nfev=None
    try:
        opt=least_squares(residual,x,jac=jac,bounds=(lower,upper),max_nfev=args.max_nfev,
            ftol=1e-9,xtol=1e-9,gtol=1e-8)
        x=opt.x;nfev=opt.nfev;stop=str(opt.message)
    except TimeoutError as e:x=best[0];stop=str(e)
    r=p.report_side(x)
    # Reuse existing force initializer only after geometry; never an admission gate.
    if r['geometry_pass']:
        f,diagnostic=force_solve(np.array(r['points']),np.array(r['normals']),np.array(r['pos']),
            np.array(r['normals'])+np.array([0,0,.4905/5]))
        r.update(forces=f.reshape(5,3).tolist(),force_diagnostic=diagnostic)
    else:r['forces']=(np.array(r['normals'])+np.array([0,0,.4905/5])).tolist()
    rec=dict(status='completed',geometry_only=True,physics_steps=0,training_actions=0,
        object_profile=args.object_profile,contact_profile=args.contact_profile,engine_sha256=engine_sha,
        max_nfev=args.max_nfev,wall_budget_s=args.wall_seconds,nfev=nfev,stop_reason=stop,
        wall_s=time.monotonic()-start,initial=initial,result=r,sources=[])
    for name in ('scripts/plan_boya_fivefinger_side.py','scripts/plan_boya_fivefinger_decoupled.py',
        'scripts/plan_boya_fivefinger_surface.py','scripts/plan_boya_fivefinger_grasp.py',
        'assets/grasp/scene.xml','docs/data/boya_native_contract.json',
        'docs/experiments/2026-10-12-fivefinger-side/PROTOCOL.md'):
        source=ROOT/name;rec['sources'].append(dict(path=name,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        target=out/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
    plan={k:r[k] for k in ('q','pos','quat','points','normals','forces','contact_body_local')}
    plan.update(joints=list(ACTION_NAMES),object=dict(shape='cylinder',radius=.024,height=.06,mass=.05),
        validated=False,geometry_pass=r['geometry_pass'],contact_profile=args.contact_profile,
        force_is_admission_gate=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(dict(stop_reason=stop,wall_s=rec['wall_s'],**r)),flush=True)


if __name__=='__main__':main()
