# Sources: existing TendonSpin Boya/Isaac trajectories and current installed
# Isaac Lab3.0.0rc1/PhysX Python sources (BSD-3-Clause), Isaac Sim6.1.0.0.
# Independent rotation calculation: scipy.spatial.transform.Rotation. No source
# code copied, simulator imports, physics execution, learner updates or rule edits.
"""Localize the observed angular-velocity/pose mismatch using archived states."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
SITE = Path('/home/rw/miniconda3/envs/env_isaaclab/lib/python3.12/site-packages')
DT = .0005


def identity(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    sources = {}
    rows = []
    for milestone, start, end in [(20, 20, 30), (30, 20, 30), (50, 20, 30), (50, 35.73, 35.8495)]:
        folder = ROOT/f'outputs/boya_hora1024_workspace50m_v1/actions_{milestone*1000000:09d}/evaluation'
        record = json.loads((folder/'result.json').read_text())
        assert record['episode_resets'] == record['controller_switches'] == record['training_actions'] == 0
        arrays = {k: [] for k in ('object_state', 'elapsed_s', 'valid', 'net_angle_deg', 'support_groups')}
        for i in range(max(0, int(start)-1), int(end)+1):
            path = folder/f'physics_{i:03d}.npz'
            if not path.exists():
                continue
            sources[str(path)] = identity(path)
            with np.load(path) as f:
                for key in arrays:
                    arrays[key].append(f[key])
        d = {key: np.concatenate(values) for key, values in arrays.items()}
        assert np.allclose(np.diff(d['elapsed_s']), DT, rtol=0, atol=1e-10)
        # SciPy uses XYZW. Reconstruct the full WORLD rotation vector, without
        # reusing TendonSpin's quaternion integration or axis implementation.
        rotations = Rotation.from_quat(d['object_state'][:, 3:7])
        previous, current = rotations[:-1], rotations[1:]
        pose_omega = (current*previous.inv()).as_rotvec()/DT
        axes = previous.apply([0, 0, 1])+current.apply([0, 0, 1])
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        signed_pose_spin = -(pose_omega*axes).sum(-1)
        omega = d['object_state'][1:, 10:13].astype(float)
        m = (d['elapsed_s'][1:] > start+1e-10) & (d['elapsed_s'][1:] <= end+1e-10) & d['valid'][1:]
        norm = lambda x: np.linalg.norm(x, axis=-1)
        recomputed = float(signed_pose_spin[m].sum()*DT*180/np.pi)
        archived = float(np.diff(d['net_angle_deg'])[m].sum())
        assert abs(recomputed-archived) < .001
        row = dict(milestone_M=milestone, window_s=[start, end], samples=int(m.sum()),
            sensor_visible_hand_contact_fraction=float((d['support_groups'][1:][m] > 0).mean()),
            geometric_endpoint_deg=recomputed, archived_endpoint_deg=archived,
            independent_angle_error_deg=recomputed-archived,
            geometric_spin_mean_rad_s=float(signed_pose_spin[m].mean()),
            reported_world_omega_mean_rad_s=omega[m].mean(0).tolist(),
            geometric_world_omega_mean_rad_s=pose_omega[m].mean(0).tolist(),
            reported_omega_norm_mean_rad_s=float(norm(omega[m]).mean()),
            geometric_omega_norm_mean_rad_s=float(norm(pose_omega[m]).mean()),
            omega_pose_vector_rmse_rad_s=float(np.sqrt(np.mean(norm(pose_omega[m]-omega[m])**2))),
            one_step_shift_rmse_rad_s={})
        # Exactly +/- one physics step checks a simple read-order offset; this
        # is not a parameter search or a resimulated counterfactual trajectory.
        for shift in (-1, 0, 1):
            idx = np.flatnonzero(m)+1
            shifted = idx+shift
            ok = (shifted >= 0) & (shifted < len(d['object_state']))
            row['one_step_shift_rmse_rad_s'][str(shift)] = float(np.sqrt(np.mean(
                norm(pose_omega[idx[ok]-1]-d['object_state'][shifted[ok], 10:13])**2)))
        rows.append(row)
    runtime = [
        SITE/'isaaclab/source/isaaclab/isaaclab/assets/rigid_object/base_rigid_object_data.py',
        SITE/'isaaclab/source/isaaclab/isaaclab/scene/interactive_scene.py',
        SITE/'isaaclab/source/isaaclab/isaaclab/sim/simulation_context.py',
        SITE/'isaaclab_physx/assets/rigid_object/rigid_object_data.py',
        SITE/'isaaclab_physx/assets/rigid_object/rigid_object.py',
        SITE/'isaaclab_physx/assets/kernels.py',
        SITE/'isaaclab_physx/physics/physx_manager.py',
        SITE/'isaaclab_physx/physics/physx_manager_cfg.py',
        SITE/'isaacsim/extscache/omni.physics.tensors-110.3.2+110.3.0.lx64.r.cp312.u7f4/omni/physics/tensors/api.py',
    ]
    local = [ROOT/p for p in ('tendonspin/physics/coordinates.py', 'tendonspin/physics/isaac_parallel.py',
        'tendonspin/rl/isaac_hora.py', 'scripts/evaluate_boya_hora.py', 'third_party/hora/hora/tasks/allegro_hand_hora.py')]
    result = dict(date='2026-10-10', new_physics_steps=0, new_training_actions=0,
        method='Independent float64 SciPy world rotation vectors; same archived physics samples, no smoothing',
        findings='Mismatch is large during hand contact and small during the observed final contact-free interval; no causal solver intervention',
        source_chain='PhysX get_transforms/get_velocities -> timestamped Lab buffers -> root_state_w concatenation -> Boya measure -> unchanged original Hora reward function',
        limits='Current installed runtime source audit; runtime file hashes collected now are not a retroactive hash pin of training-time dependencies. Solver internals/cause not established.',
        rows=rows, archived_sources=list(sources.values()), local_sources=[identity(p) for p in local],
        installed_runtime_sources=[identity(p) for p in runtime], script=identity(Path(__file__)),
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__))
    target = Path(__file__).with_name('source_chain.json')
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == '__main__':
    main()
