# Reference: botyard-inhand/grasp/generate_grasp.py at
# 204d197a9606fb7266e884f3b2e6110195be01cb, packaged source CAD, and its
# CAD surface-coordinate / static-force / SLSQP formulation. Original repository:
# ssh://git@git.nrs-lab.com:10022/wangrenpeng25p/botyard-inhand.git . Port: five fingers/16 motors,
# fixed Sharpa README48x60mm, free orientation, two deterministic seeds,
# halfspace-ray spline construction, updated BVHs, no dynamics or controller.
"""Bounded five-finger CAD contact planner; no physics integration."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares, minimize
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from plan_boya_fivefinger_grasp import Geometry, FINGERS, ROOT
from tendonspin.interfaces import ACTION_NAMES

PROTOCOL = 'docs/experiments/2026-10-12-fivefinger-sharpa-fixed/PROTOCOL.md'


class SurfacePlanner(Geometry):
    def __init__(self):
        super().__init__()
        self.dimensions = np.array([.048, .060])
        self.surfaces = []
        xs, zs = np.linspace(-.006,.006,25), np.linspace(.011,.027,33)
        for f in FINGERS:
            mesh = trimesh.load_mesh(ROOT/'assets/hand/meshes'/f'{f}distal.stl', process=False)
            planes = ConvexHull(mesh.vertices).equations
            sign = -1 if f == 'th' else 1
            planes = planes[sign*planes[:,1] > 1e-8]
            ys = -(planes[:,0,None,None]*xs[None,:,None]
                   +planes[:,2,None,None]*zs[None,None,:]+planes[:,3,None,None])/planes[:,1,None,None]
            values = ys.min(0) if sign == 1 else ys.max(0)
            self.surfaces.append(RectBivariateSpline(xs,zs,values,kx=3,ky=3,s=0))
        self.base = Rotation.from_quat(self.quat[[1,2,3,0]])
        self.last = None

    def geometry(self,x):
        if self.last is not None and np.array_equal(x[:32],self.last):
            return self.cached
        m,d = self.model,self.data
        q,pos,rot,coords = self.unpack(x)
        d.qpos[self.qids] = q
        for f in FINGERS[1:]:
            d.qpos[m.joint(f.upper()+'J1').qposadr[0]] = d.qpos[m.joint(f.upper()+'J2').qposadr[0]]
        self.quat = rot.as_quat()[[3,0,1,2]]
        d.qpos[self.qa:self.qa+3] = pos
        d.qpos[self.qa+3:self.qa+7] = self.quat
        m.geom_size[self.obj,:2] = self.dimensions/2
        mujoco.mj_fwdPosition(m,d)
        points, normals, local = [], [], []
        for i,(u,v) in enumerate(coords):
            xx,zz = .006*u,.019+.008*v
            surf = self.surfaces[i]
            yy = float(surf.ev(xx,zz))
            ln = (-1 if i==0 else 1)*np.array([-float(surf.ev(xx,zz,dx=1)),1.,-float(surf.ev(xx,zz,dy=1))])
            ln /= np.linalg.norm(ln)
            bid = m.geom_bodyid[self.pads[i]]
            matrix = d.xmat[bid].reshape(3,3)
            lp = np.array([xx,yy,zz])
            local.append(lp);points.append(d.xpos[bid]+matrix@lp);normals.append(matrix@ln)
        points,normals,local = map(np.asarray,(points,normals,local))
        axis = rot.as_matrix()[:,2]
        axial = (points-pos)@axis
        radial = points-pos-axial[:,None]*axis
        lengths = np.linalg.norm(radial,axis=1)
        inward = -radial/np.maximum(lengths[:,None],1e-10)
        gaps = lengths-.024
        align = np.einsum('ij,ij->i',normals,inward)
        # Exact convex distances are a clearance screen, not the contact objective.
        nonpad = np.array([self.distance(g,self.obj) for g in self.nonpads])
        clearance = np.array([self.distance(a,b) for a,b in self.pairs])
        mesh_gaps = np.array([self.distance(g,self.obj) for g in self.pads])
        self.calls += 1
        self.last=x[:32].copy()
        self.cached=dict(points=points,normals=normals,inward=inward,gaps=gaps,
            align=align,axis=axis,axial=axial,nonpad=nonpad,clearance=clearance,
            mesh_gaps=mesh_gaps,local=local,pos=pos,quat=self.quat.copy())
        return self.cached

    def unpack(self,x):
        return x[:16], self.pos0+x[16:19]*.04, Rotation.from_rotvec(x[19:22])*self.base, x[22:32].reshape(5,2)

    def equality(self,x):
        g=self.geometry(x);forces=x[32:].reshape(5,3)
        return np.r_[(g['gaps']-.00005)*100,
            g['normals']@g['axis'],
            np.einsum('ij,ij->i',g['normals'],np.cross(g['inward'],g['axis'])),
            forces.sum(0)-[0,0,.05*9.81],
            np.cross(g['points']-g['pos'],forces).sum(0)/.024]

    def inequality(self,x):
        g=self.geometry(x);forces=x[32:].reshape(5,3)
        fn=np.einsum('ij,ij->i',forces,g['inward']);ft=forces-fn[:,None]*g['inward']
        return np.r_[g['align']-.95,(.029-np.abs(g['axial']))*100,
            (g['mesh_gaps']+.00010)*100,
            (g['nonpad']-.0003)*100,(g['clearance']-.0003)*100,
            fn-.7,2.5-fn,(.4*fn)**2-np.sum(ft*ft,axis=1)]

    def seed(self,index):
        q=self.q0.copy();q[13:]=[.1,1.03,.2]
        if index==1:
            for offset in (4,7,10,13):q[offset:offset+3]=[0.,1.,.25]
        q=np.clip(q,self.bounds[:,0]+1e-6,self.bounds[:,1]-1e-6)
        x=np.r_[q,np.zeros(3),np.zeros(3),np.zeros(10),np.zeros(15)]
        if index==1:
            g=self.geometry(x)
            axis=g['points'][4]-g['points'][1];axis/=np.linalg.norm(axis)
            # Preserve cylinder-axis sign relative to the historical orientation.
            if axis@self.base.as_matrix()[:,2]<0:axis=-axis
            xaxis=np.cross([0.,0.,1.],axis)
            if np.linalg.norm(xaxis)<.1:xaxis=np.cross([0.,1.,0.],axis)
            xaxis/=np.linalg.norm(xaxis)
            rot=Rotation.from_matrix(np.column_stack([xaxis,np.cross(axis,xaxis),axis]))
            x[19:22]=(rot*self.base.inv()).as_rotvec()
            center=np.linalg.lstsq(g['normals'],np.einsum('ij,ij->i',g['normals'],g['points'])+.024,rcond=None)[0]
            x[16:19]=np.clip((center-self.pos0)/.04,-.95,.95)
        g=self.geometry(x)
        x[32:]=(g['inward']*1.+np.array([0,0,.05*9.81/5])).ravel()
        return x

    def report(self,x):
        g=self.geometry(x);q,pos,rot,_=self.unpack(x)
        measured=self.measure(q,pos,self.dimensions)
        eq,ineq=self.equality(x),self.inequality(x)
        forces=x[32:].reshape(5,3)
        fn=np.einsum('ij,ij->i',forces,g['inward'])
        ft=forces-fn[:,None]*g['inward']
        feasible=bool(Geometry.acceptable(measured,self.dimensions)
            and g['align'].min()>=.95-1e-5 and abs(g['axial']).max()<=.029+1e-6
            and abs(eq).max()<=.001 and fn.min()>=.7-1e-5 and fn.max()<=2.5+1e-5
            and (np.linalg.norm(ft,axis=1)-.4*fn).max()<=1e-5)
        return dict(q=q.tolist(),action_names=list(ACTION_NAMES),object_position=pos.tolist(),
            object_quat_wxyz=rot.as_quat()[[3,0,1,2]].tolist(),diameter_mm=48.,height_mm=60.,mass_kg=.05,
            gaps_mm=(measured['gaps']*1000).tolist(),contact_body_local=measured['local'].tolist(),
            planned_contact_body_local=g['local'].tolist(),points=g['points'].tolist(),
            normals=g['inward'].tolist(),forces=forces.tolist(),normal_forces_N=fn.tolist(),
            contact_alignment=g['align'].tolist(),axial_mm=(g['axial']*1000).tolist(),
            minimum_nonpad_gap_mm=float(measured['nonpad'].min()*1000),
            minimum_interfinger_gap_mm=float(measured['clearance'].min()*1000),
            equality_max_abs=float(abs(eq).max()),inequality_min=float(ineq.min()),
            force_balance_error_N=float(abs(eq[-6:-3]).max()),
            normalized_torque_error_N=float(abs(eq[-3:]).max()),geometric_candidate=feasible)

    def preload(self,x):
        g=self.geometry(x);m,d=self.model,self.data
        jac=np.zeros((3,m.nv));jr=np.zeros_like(jac);torque=np.zeros(m.nv)
        for i,geom in enumerate(self.pads):
            mujoco.mj_jac(m,d,jac,jr,g['points'][i],m.geom_bodyid[geom])
            torque+=jac.T@x[32:].reshape(5,3)[i]
        for f in FINGERS[1:]:
            torque[m.joint(f.upper()+'J2').dofadr[0]]+=torque[m.joint(f.upper()+'J1').dofadr[0]]
        desired={};clipped={};bias={}
        for a in self.contract['actuators']:
            name=a['joint'];q=float(d.qpos[m.joint(name).qposadr[0]])
            command=q+torque[m.joint(name).dofadr[0]]/a['gain'] if name in ACTION_NAMES else a['initial_ctrl']
            desired[name]=float(command);clipped[name]=float(np.clip(command,*a['control_range']))
            bias[name]=float(clipped[name]-q)
        return dict(desired_commands=desired,commands=clipped,bias_rad=bias,
            generalized_contact_torque_Nm=torque.tolist(),method='fixed static contact Jacobian transpose / kp; slave reflected; original wrists')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    planner=SurfacePlanner()
    rec=dict(protocol=PROTOCOL,physics_steps=0,training_actions=0,mujoco_geometry_version=mujoco.__version__,
        status='solving',stages=[],selected=None,sources=[])
    for name in ('scripts/plan_boya_fivefinger_surface.py','scripts/plan_boya_fivefinger_grasp.py',PROTOCOL,
                 'assets/grasp/scene.xml','docs/data/boya_native_contract.json','tendonspin/interfaces.py'):
        rec['sources'].append(dict(path=name,sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()))
    lo=np.r_[planner.bounds[:,0],[-1.]*3,[-np.pi]*3,[-1.]*10,[-3.]*15]
    hi=np.r_[planner.bounds[:,1],[1.]*3,[np.pi]*3,[1.]*10,[3.]*15]
    violation=lambda x:float(abs(planner.equality(x)).max()+max(0,-planner.inequality(x).min()))
    for index in range(2):
        t=time.monotonic();x=np.clip(planner.seed(index),lo+1e-7,hi-1e-7)
        residual=lambda z:np.r_[planner.equality(z),np.minimum(planner.inequality(z),0)]
        ls=least_squares(residual,x,bounds=(lo,hi),diff_step=1e-4,max_nfev=300,
            ftol=1e-9,xtol=1e-9,gtol=1e-8)
        print(json.dumps(dict(seed=index,phase='least_squares',nfev=ls.nfev,violation=violation(ls.x))),flush=True)
        objective=lambda z:.005*np.sum((z[:16]-x[:16])**2)+.03*np.sum(z[32:]**2)+.01*np.sum(z[22:32]**2)
        opt=minimize(objective,ls.x,method='SLSQP',bounds=list(zip(lo,hi)),
            constraints=[dict(type='eq',fun=planner.equality),dict(type='ineq',fun=planner.inequality)],
            options=dict(maxiter=200,ftol=1e-8,eps=1e-5))
        chosen=min((ls.x,opt.x),key=violation)
        report=planner.report(chosen)
        report.update(seed=index,wall_s=time.monotonic()-t,ls_nfev=ls.nfev,slsqp_nit=opt.nit,
            ls_message=ls.message,slsqp_message=opt.message)
        rec['stages'].append(report)
        print(json.dumps(report),flush=True)
        if report['geometric_candidate']:
            report['preload']=planner.preload(chosen);rec['selected']=report;break
    rec.update(status='completed',wall_s=time.monotonic()-start,geometry_calls=planner.calls)
    (out/'result.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
