"""Self-contained original native engine, packaged meshes and strict demo gates.

Mesh path relocation does not change physical parameters or the saved state.
Migration is checked against the original complete executed arrays separately.
"""
from dataclasses import dataclass
import ctypes as ct
import hashlib
from pathlib import Path
import subprocess
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from tendonspin.rl.config import ROOT, RUNTIME, digest
from .metrics import SpinTracker

SIGNATURE = mujoco.mjtState.mjSTATE_INTEGRATION
ENGINE_SHA = 'c0b309bba9a0913718f86fa375ed5003be09ca2131165cee51f9130fbafbf088'
FLAGS = ('position_drift','axis_tilt','excess_force','lost_support','object_penetration',
         'self_penetration','outside_support','floor_contact','external_support','numerical_warning','nonfinite')
FINGERS = ('TH','FF','MF','RF')
GROUPS = FINGERS + ('LF','PALM')
ACTION_NAMES = tuple(f + 'J' + str(j) for f in FINGERS for j in ((4,3,2,1) if f == 'TH' else (4,3,2)))
D = ct.POINTER(ct.c_double)
I = ct.POINTER(ct.c_int)


def verify_runtime():
    package = Path(mujoco.__file__).resolve().parent
    assert package == (RUNTIME / 'mujoco').resolve(), str(package)
    assert mujoco.__version__ == '3.13.0'
    assert digest(package / 'libmujoco.so.3.13.0') == ENGINE_SHA


def state(model, data):
    saved = np.empty(mujoco.mj_stateSize(model, SIGNATURE))
    mujoco.mj_getState(model, data, saved, SIGNATURE)
    return saved


@dataclass
class Scene:
    model: mujoco.MjModel

    def __post_init__(self):
        m = self.model
        self.joints = {m.joint(i).name:i for i in range(m.njnt)}
        self.acts = {m.actuator(i).name:i for i in range(m.nu)}
        self.obj = m.body('cylinder').id
        self.geom = m.geom('cylinder_geom').id
        self.floor = m.geom('floor_collision').id
        self.support = m.equality('cylinder_support').id
        self.qa = m.jnt_qposadr[self.joints['cylinder_free']]
        self.da = m.jnt_dofadr[self.joints['cylinder_free']]


def active_ids(scene):
    return np.array([scene.acts[name] for name in ACTION_NAMES])


def owners(scene):
    m = scene.model
    palm = m.body('palm').id
    bases = {m.body(f.lower()+'base').id:i for i,f in enumerate(GROUPS[:5])}
    result = {}
    for geom in range(m.ngeom):
        body = int(m.geom_bodyid[geom])
        while body > 0:
            if body in bases:
                result[geom] = bases[body]
                break
            if body == palm:
                result[geom] = 5
                break
            body = int(m.body_parentid[body])
    return result


def scene():
    verify_runtime()
    directory = ROOT / 'assets/grasp'
    import json
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert digest(directory / 'scene.xml') == manifest['scene_sha256']
    for name, expected in manifest['meshes'].items():
        assert digest(directory / name) == expected
    spec = mujoco.MjSpec.from_file(str(directory / 'scene.xml'))
    # Same sensor-only instrumentation and ordering as the source scene loader.
    for f in FINGERS:
        spec.add_sensor(name=f+'_contacts', type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=mujoco.mjtObj.mjOBJ_GEOM, objname=f.lower()+'distal_collision',
            reftype=mujoco.mjtObj.mjOBJ_GEOM, refname='cylinder_geom', intprm=[1|2|8|16|32,0,4])
        for suffix,kind in [('pos',mujoco.mjtSensor.mjSENS_FRAMEPOS),('quat',mujoco.mjtSensor.mjSENS_FRAMEQUAT)]:
            spec.add_sensor(name=f+'_'+suffix,type=kind,objtype=mujoco.mjtObj.mjOBJ_XBODY,objname=f.lower()+'distal')
    spec.add_sensor(name='object_contacts',type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_GEOM,objname='cylinder_geom',intprm=[1|2,0,24])
    for name,kind,body in [(f,mujoco.mjtObj.mjOBJ_XBODY,f.lower()+'base') for f in GROUPS[:5]]+[
        ('PALM',mujoco.mjtObj.mjOBJ_BODY,'palm'),('HAND',mujoco.mjtObj.mjOBJ_XBODY,'palm')]:
        spec.add_sensor(name=name+'_native',type=mujoco.mjtSensor.mjSENS_CONTACT,
            objtype=kind,objname=body,reftype=mujoco.mjtObj.mjOBJ_GEOM,refname='cylinder_geom',intprm=[1|2|8|32|64,0,24])
    spec.add_sensor(name='OBJECT_native',type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_GEOM,objname='cylinder_geom',intprm=[1|2|8,0,24])
    spec.add_sensor(name='HAND_self_native',type=mujoco.mjtSensor.mjSENS_CONTACT,
        objtype=mujoco.mjtObj.mjOBJ_XBODY,objname='palm',reftype=mujoco.mjtObj.mjOBJ_XBODY,
        refname='palm',intprm=[1|2|8,0,64])
    for f in FINGERS:
        for part in ('proximal','middle','distal'):
            spec.add_sensor(name=f'{f}_{part}_object_distance',type=mujoco.mjtSensor.mjSENS_GEOMDIST,
                objtype=mujoco.mjtObj.mjOBJ_BODY,objname=f.lower()+part,
                reftype=mujoco.mjtObj.mjOBJ_GEOM,refname='cylinder_geom',cutoff=.06)
    m = spec.compile()
    s, d = Scene(m), mujoco.MjData(m)
    with np.load(directory / 'stable_state.npz', allow_pickle=False) as saved:
        assert str(saved['model_sha256']) == manifest['source_scene_sha256']
        initial = saved['state'].copy()
        assert int(saved['signature']) == int(SIGNATURE)
    mujoco.mj_setState(m,d,initial,SIGNATURE)
    mujoco.mj_forward(m,d)
    mujoco.mj_setState(m,d,initial,SIGNATURE)
    m.eq_active0[:] = d.eq_active
    np.testing.assert_array_equal(state(m,d), initial)
    assert not d.eq_active[s.support] and not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
    assert m.geom_size[s.geom,0] == .02 and m.geom_size[s.geom,1] == .016 and m.body_mass[s.obj] == .05
    assert m.opt.timestep == .0005 and np.array_equal(m.opt.gravity,[0,0,-9.81])
    return s,d,dict(scene_sha256=manifest['scene_sha256'],source_scene_sha256=manifest['source_scene_sha256'],manifest=manifest)


def spin_increments(quaternions, initial, direction):
    batch,steps = quaternions.shape[:2]
    previous = np.concatenate((np.broadcast_to(initial,(batch,1,4)),quaternions[:,:-1]),axis=1)
    now = Rotation.from_quat(quaternions[..., [1,2,3,0]].reshape(-1,4))
    before = Rotation.from_quat(previous[..., [1,2,3,0]].reshape(-1,4))
    axes = now.as_matrix()[:,:,2]+before.as_matrix()[:,:,2]
    axes /= np.linalg.norm(axes,axis=1)[:,None]
    return np.degrees((direction*np.einsum('ij,ij->i',(now*before.inv()).as_rotvec(),axes)).reshape(batch,steps))


def measure(scene, data, tracker, center, owner):
    m,s,d = scene.model,scene,data
    forces = np.zeros((6,3)); normal = np.zeros(6)
    outside=floor=obj_pen=self_pen=self_force=0.
    for k,c in enumerate(d.contact):
        cf=np.zeros(6);mujoco.mj_contactForce(m,d,k,cf)
        if c.geom1 in owner and c.geom2 in owner:
            self_force+=cf[0];self_pen=min(self_pen,c.dist)
        if s.geom not in (c.geom1,c.geom2):continue
        other=c.geom1 if c.geom2==s.geom else c.geom2
        sign=1. if c.geom2==s.geom else -1.
        world=sign*c.frame.reshape(3,3).T@cf[:3]
        obj_pen=min(obj_pen,c.dist)
        if other in owner:
            forces[owner[other]]+=world;normal[owner[other]]+=cf[0]
        else:
            outside+=np.linalg.norm(world)
            if other==s.floor:floor+=np.linalg.norm(world)
    drift=float(np.linalg.norm(d.qpos[s.qa:s.qa+3]-center)*1000)
    tilt=float(np.degrees(tracker.tilt));maximum=float(np.linalg.norm(forces,axis=1).max())
    conditions=(drift>5,tilt>15,maximum>12,(normal[:5]>1e-6).sum()<2,-obj_pen>.0015,-self_pen>.0015,
                outside>.05,floor>.05,bool(d.eq_active[s.support] or np.any(d.xfrc_applied) or np.any(d.qfrc_applied)),
                bool(np.any(d.warning.number)),not(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()))
    flags=[name for name,condition in zip(FLAGS,conditions) if condition]
    info=dict(drift_mm=drift,tilt_deg=tilt,normal_N=normal,maximum_object_force_N=maximum,
              self_normal_N=float(self_force),object_penetration_mm=-obj_pen*1000,
              self_penetration_mm=-self_pen*1000,outside_N=float(outside),floor_N=float(floor))
    return flags,info


class Runner:
    def __init__(self,s,original):
        self.s,self.m=s,s.model
        self.owner=np.full(self.m.ngeom,-1,dtype=np.int32)
        self.owner_dict=owners(s)
        for geom,owner in self.owner_dict.items(): self.owner[geom]=owner
        self.center=original.qpos[s.qa:s.qa+3].copy()
        self.axis=Rotation.from_quat(original.qpos[s.qa+3:s.qa+7][[1,2,3,0]]).as_matrix()[:,2].copy()
        package=Path(mujoco.__file__).parent
        runtime=list(package.glob('libmujoco.so.*'))
        assert len(runtime)==1
        self.runtime=runtime[0]
        include=package/'include'
        if not (include/'mujoco/mujoco.h').exists():
            include=RUNTIME/'mujoco/include'
        cfile=Path(__file__).parent/'replay.c'
        headers=b''.join(p.read_bytes() for p in sorted((include/'mujoco').glob('*.h')))
        digest=hashlib.sha256(cfile.read_bytes()+self.runtime.read_bytes()+headers).hexdigest()[:20]
        cache=Path(tempfile.gettempdir())/f'tendonspin-native-replay-{digest}'
        cache.mkdir(exist_ok=True)
        binary=cache/'replay.so'
        if not binary.exists():
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                target=Path(tmp)/'replay.so'
                p=subprocess.run(['cc','-O3','-fPIC','-shared','-ffp-contract=off',str(cfile),
                    '-I',str(include),str(self.runtime),f'-Wl,-rpath,{package}','-lm','-o',str(target)],
                    capture_output=True,text=True)
                if p.returncode: raise RuntimeError(p.stderr)
                target.replace(binary)
        self.lib=ct.CDLL(str(binary))
        self.lib.replay_version.restype=ct.c_int
        assert self.lib.replay_version()==mujoco.mj_version()
        self.lib.replay.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,D,I]+[ct.c_int]*4+[D]*7+[I,ct.c_int]
        self.lib.replay.restype=ct.c_int

    def run(self,d,controls,*,sensors=False,stop=True):
        m=self.m; controls=np.ascontiguousarray(controls,dtype=np.float64)
        assert controls.ndim==2 and controls.shape[1]==m.nu and np.isfinite(controls).all()
        assert m.nflex==0
        n=len(controls)
        q=np.empty((n,m.nq));v=np.empty((n,m.nv));motor=np.empty((n,m.nu))
        metrics=np.empty((n,14));flags=np.empty(n,dtype=np.int32)
        sens=np.empty((n,m.nsensordata)) if sensors else None
        ptr=lambda a: a.ctypes.data_as(D)
        count=self.lib.replay(m._address,d._address,n,ptr(controls),self.owner.ctypes.data_as(I),
            self.s.geom,self.s.qa,self.s.floor,self.s.support,ptr(self.center),ptr(self.axis),
            ptr(q),ptr(v),ptr(motor),ptr(sens) if sens is not None else D(),ptr(metrics),
            flags.ctypes.data_as(I),int(stop))
        assert 0<count<=n
        result={k:a[:count].copy() for k,a in dict(ctrl=controls,qpos=q,qvel=v,motor=motor,
                                                  metrics=metrics,flags=flags).items()}
        if sens is not None: result['sensors']=sens[:count].copy()
        result['final_state']=state(m,d)
        return result

    def check_physical(self,d,record):
        tracker=SpinTracker(self.original_quaternion,-1)
        tracker.update(d.qpos[self.s.qa+3:self.s.qa+7])
        names,info=measure(self.s,d,tracker,self.center,self.owner_dict)
        code=sum(1<<i for i,name in enumerate(FLAGS) if name in names)
        assert code==int(record['flags'][-1]),(code,record['flags'][-1],names)
        z=record['metrics'][-1]
        values=np.r_[info['drift_mm'],info['tilt_deg'],info['normal_N'],
            info['maximum_object_force_N'],info['self_normal_N'],info['object_penetration_mm'],
            info['self_penetration_mm'],info['outside_N'],info['floor_N']]
        # acos is ill-conditioned near zero tilt: algebraically equivalent
        # quaternion-to-matrix implementations differ by roundoff in the dot
        # product. Compare that dot product; keep the physical flags exact.
        other=np.arange(len(z))!=1
        assert np.max(abs(values[other]-z[other]))<1e-8,(values,z)
        assert abs(np.cos(np.radians(values[1]))-np.cos(np.radians(z[1])))<1e-12,(values,z)

