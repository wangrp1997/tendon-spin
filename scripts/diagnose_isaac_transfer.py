# Sources: archived TendonSpin Boya-PD-v1 execution; Isaac Lab3 AssetBaseCfg
# documents XYZW, original Boya/MuJoCo contract stores WXYZ. New offline audit.
"""Diagnose the failed physical port without another engine step."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

root=Path(__file__).resolve().parents[1]
phase_path=root/'outputs/isaac_boya_physical_audit/phases_raw.json'
phase=json.loads(phase_path.read_text())
contract_path=root/phase['contract']['path']
assert hashlib.sha256(contract_path.read_bytes()).hexdigest()==phase['contract']['sha256']
contract=json.loads(contract_path.read_text())
execution=root/phase['execution']['path']
assert hashlib.sha256(execution.read_bytes()).hexdigest()==phase['execution']['sha256']
with np.load(execution,allow_pickle=False) as raw:
    fields={}
    for name in ('joint_pos','joint_vel','object_state','commands','efforts','body_pose','contact_force_matrix'):
        array=raw[name];flat=array.reshape(len(array),-1)
        finite=np.isfinite(flat).all(1)
        bad=np.flatnonzero(~finite)
        fields[name]=dict(shape=list(array.shape),first_nonfinite_step=int(bad[0]+1) if len(bad) else None,
            maximum_finite_absolute_value=float(np.abs(flat[np.isfinite(flat)]).max()) if flat.size else None)
    expected={body['name']:body for body in contract['body_frames']}
    frames=[]
    for index,name in enumerate(phase['body_names']):
        if name not in expected: continue
        actual=raw['body_pose'][0,index];source=expected[name]
        ref=np.array(source['quat'])[[1,2,3,0]]
        orientation=(Rotation.from_quat(actual[3:7])*Rotation.from_quat(ref).inv()).magnitude()
        frames.append(dict(body=name,first_step_position_difference_mm=float(np.linalg.norm(actual[:3]-source['pos'])*1000),
            first_step_orientation_difference_deg=float(np.degrees(orientation)),
            isaac_pos=actual[:3].tolist(),source_native_pos=source['pos']))
    finite_rows=np.flatnonzero(np.isfinite(raw['joint_pos']).all(1)&np.isfinite(raw['joint_vel']).all(1))
    coupling={}
    names=phase['joint_names']
    for c in contract['couplings']:
        residual=raw['joint_pos'][finite_rows,names.index(c['slave'])]-raw['joint_pos'][finite_rows,names.index(c['master'])]
        coupling[c['finger']]=dict(max_abs_on_finite_records_rad=float(np.abs(residual).max()),
            last_finite_record_rad=float(residual[-1]),finite_records=int(len(residual)),
            note='finite does not mean physically valid; original pose was not preserved')
    control_error=float(np.abs(raw['commands'][0]-[a['initial_ctrl'] for a in contract['actuators']]).max())
record=dict(kind='offline transfer audit; no new integration',new_physics_steps=0,
    original_physical_state_verified=False,benchmark_eligible=False,
    source_phase=dict(path=str(phase_path.relative_to(root)),sha256=hashlib.sha256(phase_path.read_bytes()).hexdigest()),
    execution=phase['execution'],requested_controller=phase['controller'],logged_steps=phase['physics_steps'],
    integration_elapsed_s=phase['diagnostic_elapsed_s'],actual_diagnostic_stop_reason=phase['stop_reason'],
    first_nonfinite_by_field=fields,first_step_body_comparison=frames,finite_record_coupling_residuals=coupling,
    initial_command_max_error_rad=control_error,
    confirmed_bug='Native WXYZ quaternion was passed directly to Isaac Lab3 InitialStateCfg.rot requiring XYZW; output quaternions were also interpreted with the wrong order. Declared grasp/orientation did not match actual execution.',
    confirmed_api_reference=dict(path='/home/rw/miniconda3/envs/env_isaaclab/lib/python3.12/site-packages/isaaclab/source/isaaclab/isaaclab/assets/asset_base_cfg.py',lines=[37,38,39]),
    numerical_issue='Joint speeds alternate/grow to1.137e22rad/s and first become nonfinite atstep34. Explicit viscous damping in the adapter is an additional numerical-risk hypothesis; its causal contribution is not isolated from wrong pose/import/contact issues.',
    reporting_corrections=['No eligible original-state controller result','raw efforts are requested combined motor/passive efforts, not certified actually applied joint torque; the forwarding actuator also has a declared1e6 clamp','Configured CCD was disabled by the runtime for GPU dynamics','Contact filters warn about fixed tip/central colliders; all-zero sensor matrix is not proof of actual contact loss','max tilt0 was computed with the wrong quaternion order and is not a valid physical axis estimate','The first33 finite records are not declared a physically valid prefix'],
    action='Stop dependent Isaac grasp/cache/PPO work pending user choice; preserve paper method and original configuration')
p=root/'docs/experiments/2026-10-08-boya-isaac-physical/data/failure_diagnosis.json'
p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(logged_steps=phase['physics_steps'],original_state_verified=False,confirmed_bug=record['confirmed_bug'],first_step_root=next(f for f in frames if f['body']=='fabase'))))
