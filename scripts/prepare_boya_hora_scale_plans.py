# References: HaozhiQi/hora v0.0.1 allegro_hand_grasp.py (MIT), scale caches;
# TendonSpin plan_boya_fivefinger_grasp.py/verify_boya_mujoco_grasp.py (MIT),
# CAD nearest-distance fit and fixed position-target preload. Source CAD port:
# botyard-inhand204d197a9606fb7266e884f3b2e6110195be01cb.
# New port: ONE fixed, bounded fit per declared size; nominal orientation/center,
# long-finger splay within +/-2deg. Geometry proposals require Isaac screening.
"""Prepare nine CAD grasp proposals without integrating dynamics."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from plan_boya_fivefinger_grasp import Geometry, FINGERS, ROOT
from plan_boya_fivefinger_decoupled import force_solve
from verify_boya_mujoco_grasp import build_scene, initialize, ENGINE_SHA
from tendonspin.interfaces import ACTION_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.configuration.read_text())
    scene = json.loads((ROOT/cfg['nominal_scene']).read_text())
    nominal = json.loads((ROOT/cfg['nominal_plan']).read_text())
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    engine = Path(mujoco.__file__).parent/f'libmujoco.so.{mujoco.__version__}'
    if mujoco.__version__ != '3.13.0' or hashlib.sha256(engine.read_bytes()).hexdigest() != ENGINE_SHA:
        raise RuntimeError('Declared local native geometry runtime required')
    g = Geometry()
    for n, q in zip(scene['nominal_state']['joint_names'], scene['nominal_state']['joint_pos'], strict=True):
        g.data.qpos[g.model.joint(n).qposadr[0]] = q
    g.quat = np.array(nominal['quat'])
    q0 = np.array(nominal['q']); pos = np.array(nominal['pos'])
    base_dims = np.array([scene['geometry']['diameter_m'], scene['geometry']['height_m']])
    axis = Rotation.from_quat(g.quat[[1,2,3,0]]).as_matrix()[:,2]
    splay_ids = [ACTION_NAMES.index(f+'J4') for f in ('FF','MF','RF','LF')]
    lo = np.maximum(g.bounds[:,0], q0-.35)
    hi = np.minimum(g.bounds[:,1], q0+.35)
    bound = cfg['cache_preparation']['long_finger_splay_limit_rad']
    lo[splay_ids] = -bound; hi[splay_ids] = bound
    records = []
    sources = [args.configuration, Path(__file__), ROOT/cfg['nominal_scene'], ROOT/cfg['nominal_plan'],
        ROOT/'scripts/plan_boya_fivefinger_grasp.py', ROOT/'scripts/plan_boya_fivefinger_decoupled.py',
        ROOT/'scripts/verify_boya_mujoco_grasp.py', ROOT/'assets/grasp/scene.xml']
    pins = [dict(path=str(p.resolve().relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]
    for scale_id, scale in enumerate(cfg['scale_centers']):
        dims = base_dims*scale/cfg['logical_nominal_scale']
        def residual(q):
            if time.monotonic()-started > cfg['cache_preparation']['geometry_wall_budget_s']:
                raise TimeoutError('Fixed nine-size geometry budget exhausted; no retry')
            v = g.measure(q, pos, dims)
            penalties = np.r_[np.maximum(-.00015-v['nonpad'],0),
                np.maximum(.0001-v['clearance'],0), np.maximum(.006-v['local'][:,2],0),
                np.maximum(v['local'][:,2]-.032,0),
                np.maximum(np.abs(v['local'][:,0])-.009,0)]
            return np.r_[(v['gaps']-.00005)*1000, penalties*1000, .05*(q-q0)]
        fit = least_squares(residual,np.clip(q0,lo+1e-8,hi-1e-8),bounds=(lo,hi),
            max_nfev=cfg['cache_preparation']['geometry_max_nfev_per_scale'],
            ftol=1e-8,xtol=1e-8,gtol=1e-8,diff_step=1e-5)
        v = g.measure(fit.x,pos,dims); points = v['lines'][:,:3]
        radial = points-pos-((points-pos)@axis)[:,None]*axis
        inward = -radial/np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-9)
        force, force_diagnostic = force_solve(points,inward,pos,np.array(nominal['forces']))
        plan = dict(joints=list(ACTION_NAMES),q=fit.x.tolist(),pos=pos.tolist(),quat=g.quat.tolist(),
            object=dict(shape='cylinder',radius=float(dims[0]/2),height=float(dims[1]),mass=.05),
            contact_body_local=v['local'].tolist(),forces=force.reshape(5,3).tolist(),validated=False)
        folder=out/f'scale_{scale_id:02d}';folder.mkdir()
        geometry=dict(shape='cylinder',diameter_m=float(dims[0]),height_m=float(dims[1]),mass_kg=.05)
        model=build_scene(folder,geometry);data,preload=initialize(model,plan)
        names=scene['nominal_state']['joint_names'];motors=scene['nominal_state']['motor_names']
        q=np.array([data.qpos[model.joint(n).qposadr[0]] for n in names])
        commands=np.array([data.ctrl[model.actuator(n).id] for n in motors])
        obj=np.r_[pos,g.quat[[1,2,3,0]],np.zeros(6)]
        np.savez(folder/'proposal.npz',q=q,object_state=obj,commands=commands,
            joint_names=np.array(names),motor_names=np.array(motors),scale_id=scale_id,logical_scale=scale)
        row=dict(scale_id=scale_id,logical_scale=scale,dimensions_m=geometry,
            gaps_mm=(v['gaps']*1000).tolist(),nfev=fit.nfev,cost=float(fit.cost),
            force_diagnostic=force_diagnostic,preload=preload,plan=plan,
            proposal=str((folder/'proposal.npz').relative_to(ROOT)),
            physics_steps=0,geometry_gate_used=False,physical_acceptance='pending Isaac screen')
        (folder/'plan.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
        records.append(row)
        (out/'result.json').write_text(json.dumps(dict(status='preparing' if len(records)<9 else 'completed',
            scales=records,configuration=cfg,sources=pins,wall_s=time.monotonic()-started,
            physics_steps=0,training_actions=0),indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(scale_id=scale_id,scale=scale,gaps_mm=row['gaps_mm'],
            nfev=fit.nfev,wall_s=time.monotonic()-started)),flush=True)


if __name__ == '__main__':main()
