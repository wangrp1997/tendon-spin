# Source: botyard-inhand/grasp/generate_grasp.py at204d197a9606fb7266e884f3b2e6110195be01cb
# (ssh://git@git.nrs-lab.com:10022/wangrenpeng25p/botyard-inhand.git), and
# TendonSpin6089025 surface planner. Changes: fixed-geometry convex force solve
# with analytic derivatives, conservative CAD face-axis separation for degenerate
# mesh distances, absolute finite differences, separately bounded geometry seeds.
"""Generate an explicitly selected five-finger proposal; original defaults untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from plan_boya_fivefinger_surface import SurfacePlanner
from plan_boya_fivefinger_grasp import Geometry, FINGERS, ROOT
from tendonspin.interfaces import ACTION_NAMES

PROTOCOL='docs/experiments/2026-10-12-fivefinger-mujoco/PROTOCOL.md'


def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def force_solve(points,normals,center,initial):
    """Fixed geometry: linear equilibrium and convex Coulomb cones."""
    count=len(points)
    equilibrium=np.vstack([np.tile(np.eye(3),(1,count)),np.hstack([skew(p-center)/.024 for p in points])])
    rhs=np.array([0.,0.,.4905,0.,0.,0.])
    normal_matrix=np.zeros((count,count*3))
    for i,n in enumerate(normals):normal_matrix[i,3*i:3*i+3]=n
    def cones(f):
        fs=f.reshape(count,3);fn=np.sum(fs*normals,axis=1)
        ft=fs-fn[:,None]*normals
        return .4*fn-np.linalg.norm(ft,axis=1)
    def cone_jac(f):
        fs=f.reshape(count,3);fn=np.sum(fs*normals,axis=1);ft=fs-fn[:,None]*normals
        jac=np.zeros((count,count*3))
        for i,n in enumerate(normals):jac[i,3*i:3*i+3]=.4*n-ft[i]/max(np.linalg.norm(ft[i]),1e-12)
        return jac
    opt=minimize(lambda f:.5*f@f,np.asarray(initial).ravel(),jac=lambda f:f,method='SLSQP',
        constraints=[dict(type='eq',fun=lambda f:equilibrium@f-rhs,jac=lambda f:equilibrium),
            dict(type='ineq',fun=lambda f:normal_matrix@f-.7,jac=lambda f:normal_matrix),
            dict(type='ineq',fun=lambda f:2.5-normal_matrix@f,jac=lambda f:-normal_matrix),
            dict(type='ineq',fun=cones,jac=cone_jac)],options=dict(maxiter=300,ftol=1e-11))
    f=opt.x;fn=normal_matrix@f;eq=equilibrium@f-rhs;cone=cones(f)
    report=dict(success=bool(abs(eq).max()<=.01 and fn.min()>=.7-1e-5
        and fn.max()<=2.5+1e-5 and cone.min()>=-1e-5),optimizer_success=bool(opt.success),
        message=opt.message,iterations=opt.nit,forces=f.reshape(count,3).tolist(),
        normal_forces_N=fn.tolist(),friction_margins_N=cone.tolist(),equilibrium_error_N=eq.tolist())
    return f,report


class DecoupledPlanner(SurfacePlanner):
    def __init__(self):
        self.hulls={};self.certificates={}
        super().__init__()

    def hull(self,g):
        if g not in self.hulls:
            name=self.model.body(self.model.geom_bodyid[g]).name
            mesh=trimesh.load_mesh(ROOT/'assets/hand/meshes'/f'{name}.stl',process=False)
            hull=ConvexHull(mesh.vertices)
            self.hulls[g]=(mesh.vertices[hull.vertices],hull.equations[:,:3])
        v,n=self.hulls[g];bid=self.model.geom_bodyid[g]
        rot=self.data.xmat[bid].reshape(3,3)
        return v@rot.T+self.data.xpos[bid],n@rot.T

    def distance(self,a,b,line=None):
        raw=super().distance(a,b,line)
        if (self.model.geom_type[a]==mujoco.mjtGeom.mjGEOM_MESH and
            self.model.geom_type[b]==mujoco.mjtGeom.mjGEOM_MESH and raw<.002):
            va,na=self.hull(a);vb,nb=self.hull(b);axes=np.vstack([na,nb])
            pa,pb=axes@va.T,axes@vb.T
            certified=float(np.maximum(pb.min(1)-pa.max(1),pa.min(1)-pb.max(1)).max())
            self.certificates[(a,b)]=(float(raw),certified)
            return certified
        return raw

    def geom_eq(self,x):
        return self.equality(x)[:-6]

    def geom_ineq(self,x):
        return self.inequality(x)[:-15]

    def from_record(self,entry):
        rot=Rotation.from_quat(np.array(entry['object_quat_wxyz'])[[1,2,3,0]])
        local=np.array(entry['planned_contact_body_local'])
        coords=np.column_stack([local[:,0]/.006,(local[:,2]-.019)/.008])
        return np.r_[entry['q'],(np.array(entry['object_position'])-self.pos0)/.04,
            (rot*self.base.inv()).as_rotvec(),coords.ravel(),np.array(entry['forces']).ravel()]

    def describe(self,x):
        self.last=None;g=self.geometry(x)
        q,pos,rot,_=self.unpack(x)
        force,fr=force_solve(g['points'],g['inward'],pos,x[32:]);x=x.copy();x[32:]=force
        r=super().report(x)
        all_geometry=bool(Geometry.acceptable(self.measure(q,pos,self.dimensions),self.dimensions)
            and g['nonpad'].min()>=.0003-1e-6 and g['clearance'].min()>=.0003-1e-6
            and g['align'].min()>=.95-1e-5 and abs(g['axial']).max()<=.029+1e-6
            and abs(self.geom_eq(x)).max()<=.001)
        r.update(geometry_pass=all_geometry,force_feasibility=fr,
            geometric_candidate=bool(all_geometry and fr['success']))
        r['near_mesh_pairs']=[dict(bodies=[self.model.body(self.model.geom_bodyid[a]).name,
            self.model.body(self.model.geom_bodyid[b]).name],engine_gap_mm=raw*1000,
            conservative_separation_mm=sat*1000) for (a,b),(raw,sat) in self.certificates.items()]
        return x,r


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--seed-plan',type=Path,required=True)
    p.add_argument('--object-profile',choices=('sharpa_recommended',),required=True)
    p.add_argument('--acceptance-profile',choices=('static_force','cache28'),default='static_force',
        help='cache28: force feasibility is diagnostic only; physical admission uses the old cache gates')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();g=DecoupledPlanner()
    if mujoco.__version__!='3.13.0':raise RuntimeError('Use the declared native MuJoCo3.13 runtime')
    library=Path(mujoco.__file__).parent/'libmujoco.so.3.13.0'
    digest=hashlib.sha256(library.read_bytes()).hexdigest()
    if digest!='c0b309bba9a0913718f86fa375ed5003be09ca2131165cee51f9130fbafbf088':raise RuntimeError('Native engine identity changed')
    seed_record=json.loads(a.seed_plan.read_text())['stages'][0];seed=g.from_record(seed_record)
    protocol=('docs/experiments/2026-10-12-fivefinger-mujoco/PROTOCOL_CACHE28.md'
              if a.acceptance_profile=='cache28' else PROTOCOL)
    rec=dict(protocol=protocol,acceptance_profile=a.acceptance_profile,object_profile=a.object_profile,mujoco_version=mujoco.__version__,
        engine_sha256=digest,physics_steps=0,training_actions=0,stages=[],selected=None,sources=[])
    for name in ('scripts/plan_boya_fivefinger_decoupled.py','scripts/plan_boya_fivefinger_surface.py',
        'scripts/plan_boya_fivefinger_grasp.py',protocol,'assets/grasp/scene.xml','tendonspin/interfaces.py'):
        rec['sources'].append(dict(path=name,sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()))
        (out/Path(name).name).write_bytes((ROOT/name).read_bytes())
    rec['seed_plan_sha256']=hashlib.sha256(a.seed_plan.read_bytes()).hexdigest()
    x,initial=g.describe(seed);rec['fixed_geometry_force_diagnosis']=initial
    print(json.dumps(dict(stage='fixed_geometry',geometry_pass=initial['geometry_pass'],force=initial['force_feasibility'],minimum_interfinger_gap_mm=initial['minimum_interfinger_gap_mm'])),flush=True)
    if initial['geometric_candidate'] or (a.acceptance_profile=='cache28' and initial['geometry_pass']):rec['selected']=initial
    lo=np.r_[g.bounds[:,0],[-1.]*3,[-np.pi]*3,[-1.]*10,[-3.]*15]
    hi=np.r_[g.bounds[:,1],[1.]*3,[np.pi]*3,[1.]*10,[3.]*15]
    for index in range(2):
        if rec['selected'] is not None:break
        t=time.monotonic();x=seed.copy()
        if index==1:x[19:22]*=.5
        x=np.clip(x,lo+1e-7,hi-1e-7)
        geom_residual=lambda z:np.r_[g.geom_eq(np.r_[z,x[32:]]),np.minimum(g.geom_ineq(np.r_[z,x[32:]]),0)]
        # Absolute step avoids effectively zero perturbations at near-zero joints.
        def jac(z):
            baseline=geom_residual(z);j=np.empty((len(baseline),len(z)))
            for k in range(len(z)):
                zz=z.copy();zz[k]+=1e-5;j[:,k]=(geom_residual(zz)-baseline)/1e-5
            return j
        ls=least_squares(geom_residual,x[:32],jac=jac,bounds=(lo[:32],hi[:32]),max_nfev=300,
            ftol=1e-10,xtol=1e-10,gtol=1e-9)
        x[:32]=ls.x;self_geom=g.geometry(x)
        x[32:],before_force=force_solve(self_geom['points'],self_geom['inward'],self_geom['pos'],x[32:])
        print(json.dumps(dict(seed=index,phase='geometry',ls_nfev=ls.nfev,residual=float(abs(geom_residual(x[:32])).max()),force=before_force['success'])),flush=True)
        if a.acceptance_profile=='cache28':
            x,r=g.describe(x)
            r.update(seed=index,phase='geometry',ls_nfev=ls.nfev,wall_s=time.monotonic()-t,
                force_is_admission_gate=False,geometric_candidate=r['geometry_pass'])
            rec['stages'].append(r)
            print(json.dumps(dict(seed=index,geometry_pass=r['geometry_pass'],gaps_mm=r['gaps_mm'],
                minimum_interfinger_gap_mm=r['minimum_interfinger_gap_mm'],force_is_admission_gate=False)),flush=True)
            if r['geometry_pass']:rec['selected']=r
            continue
        opt=minimize(lambda z:.03*np.sum(z[32:]**2)+.005*np.sum((z[:16]-seed[:16])**2),x,
            method='SLSQP',bounds=list(zip(lo,hi)),
            constraints=[dict(type='eq',fun=g.equality),dict(type='ineq',fun=g.inequality)],
            options=dict(maxiter=400,ftol=1e-10,eps=1e-5))
        candidates=[]
        for name,z in (('geometry',x),('coupled_refinement',opt.x)):
            z,r=g.describe(z);r.update(seed=index,phase=name,ls_nfev=ls.nfev,slsqp_nit=opt.nit,
                slsqp_message=opt.message,wall_s=time.monotonic()-t)
            candidates.append(r)
            if r['geometric_candidate'] and rec['selected'] is None:rec['selected']=r
        rec['stages'].extend(candidates)
        print(json.dumps([dict(seed=index,phase=r['phase'],geometry_pass=r['geometry_pass'],
            force_pass=r['force_feasibility']['success'],minimum_interfinger_gap_mm=r['minimum_interfinger_gap_mm'],
            gaps_mm=r['gaps_mm'],geometric_candidate=r['geometric_candidate']) for r in candidates]),flush=True)
    rec.update(status='completed',wall_s=time.monotonic()-start,geometry_calls=g.calls)
    (out/'result.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
    if rec['selected']:
        r=rec['selected'];plan=dict(joints=list(ACTION_NAMES),q=r['q'],pos=r['object_position'],
            quat=r['object_quat_wxyz'],points=r['points'],normals=r['normals'],forces=r['forces'],
            contact_body_local=r['planned_contact_body_local'],object=dict(shape='cylinder',radius=.024,height=.06,mass=.05),
            pad_geometry_version='cad_distal_surface_v3',planning_friction=.4,
            normal_bounds_N=[.7,2.5],validated=False,protocol=protocol,
            force_is_admission_gate=a.acceptance_profile!='cache28')
        (out/'plan.json').write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
