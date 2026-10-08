# Sources: native Boya initial-state/PD contract, NVIDIA Isaac Lab 3 APIs.
# TendonSpin physical transfer v2; not a Hora/AnyRotate/Sharpa policy or score.
"""Verify initial transfer, then log one .5s source-PD holding diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
archive = root/'outputs/isaac_boya_physical_v2'
started = time.monotonic()
record = dict(kind='Boya Isaac physical transfer diagnostic v2',
    controller='boya_source_pd_hold_v2', physics_steps=0, training_actions=0,
    benchmark_validated=False, grasp_validated=False, complete=False,
    original_physical_state_verified=False, episode_resets=0, controller_switches=0,
    eula_acceptance_supplied=os.environ.get('OMNI_KIT_ACCEPT_EULA')=='YES')
frames = []


def file_record(path):
    return dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(phase):
    record['phase'] = phase
    record['wall_s_at_phase'] = time.monotonic()-started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print('TENDONSPIN_PHYSICS_V2 '+json.dumps(dict(phase=phase, physics_steps=record['physics_steps'],
        wall_s=record['wall_s_at_phase'], stop_reason=record.get('stop_reason'))), flush=True)


save('before official launcher')
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=False, device='cuda:0')
app = launcher.app
failed = False
try:
    import numpy as np
    import torch
    from tendonspin.physics.isaac_boya import make_scene, SourcePositionAdapter
    from tendonspin.physics.coordinates import audit_initial_state, axis_z_xyzw, wxyz_to_xyzw

    def to_np(value):
        if hasattr(value, 'torch'):
            value = value.torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().copy()
        if hasattr(value, 'numpy'):
            return value.numpy().copy()
        return np.asarray(value).copy()

    def snapshot():
        return dict(joint_pos=to_np(hand.data.joint_pos)[0],
            joint_vel=to_np(hand.data.joint_vel)[0],
            object_state=to_np(obj.data.root_state_w)[0],
            body_pose=to_np(hand.data.body_link_pose_w)[0])

    contract_path = root/'docs/data/boya_native_contract.json'
    contract = json.loads(contract_path.read_text())
    record['contract'] = file_record(contract_path)
    imported = json.loads((root/'docs/data/isaac_boya_import_phases.json').read_text())
    usd = root/imported['usd']['path']
    if file_record(usd)['sha256'] != imported['usd']['sha256']:
        raise ValueError('Imported USD changed')
    record['usd'] = file_record(usd)
    save('before scene construction/reset')
    sim, hand, obj, sensor, filters, cfg = make_scene(root, contract, usd)
    adapter = SourcePositionAdapter(hand, contract)
    record['configuration'] = cfg.to_dict()
    record['joint_names'] = list(hand.joint_names)
    record['body_names'] = list(hand.body_names)
    record['backend_joint_names'] = list(hand.backend_joint_names)
    record['runtime_body_paths'] = list(hand.root_view.link_paths[0])
    record['contact_filters'] = filters
    record['effective_settings'] = dict(
        backend=type(sim.physics_manager).__name__,
        quaternion_input_output='XYZW; native contract WXYZ explicitly converted',
        ccd_requested=True, ccd_effective=None,
        ccd_evidence='Requested flag is not proof of effective CCD; inspect this launch log',
        damping_implementation='solver-side viscous joint friction; no explicit damping effort',
        self_collision_requested=True,
        friction_properties_backend_order=to_np(hand.root_view.get_dof_friction_properties()).tolist(),
        drive_stiffness_backend_order=to_np(hand.root_view.get_dof_stiffnesses()).tolist(),
        drive_damping_backend_order=to_np(hand.root_view.get_dof_dampings()).tolist(),
        actuator_forwarder_effort_limit_Nm=1e6,
        source_motor_caps_Nm=[a['force_range'] for a in contract['actuators']])
    friction = np.asarray(record['effective_settings']['friction_properties_backend_order'])[0]
    by_name = {j['name']: j for j in contract['joints']}
    damping_error = float(np.max(np.abs(friction[:,2]-[by_name[n]['damping'] for n in hand.backend_joint_names])))
    record['effective_settings']['damping_max_readback_error'] = damping_error
    record['effective_settings']['damping_readback_passed'] = damping_error <= 1e-6
    state0 = snapshot()
    state0['commands'] = to_np(adapter.commands)[0]
    archive.mkdir(parents=True, exist_ok=True)
    initial_path = archive/'initial_state.npz'
    np.savez_compressed(initial_path, **state0)
    record['initial_state'] = file_record(initial_path)
    initial_check = audit_initial_state(contract, hand.joint_names, hand.body_names, state0)
    command_error = float(np.max(np.abs(state0['commands']-[a['initial_ctrl'] for a in contract['actuators']])))
    initial_check['command_max_error_rad'] = command_error
    initial_check['checks']['motor_commands'] = command_error <= 1e-6
    initial_check['checks']['solver_damping_readback'] = damping_error <= 1e-6
    initial_check['checks']['runtime_contact_filters'] = set(filters)==set(record['runtime_body_paths'])
    initial_check['passed'] = all(initial_check['checks'].values())
    record['initialization_audit'] = initial_check
    record['original_physical_state_verified'] = initial_check['passed']
    save('initial physical state and damping audited before execution')
    if not initial_check['passed']:
        record['stop_reason'] = 'initial transfer identity check failed; no logged physics execution'
    else:
        pose0 = np.array(contract['object']['pos'])
        axis0 = axis_z_xyzw(wxyz_to_xyzw(contract['object']['quat']))
        backend_to_user = [list(hand.backend_joint_names).index(n) for n in hand.joint_names]
        record['effort_semantics'] = dict(
            motor_effort_requested='external clipped source proportional motor effort; excludes passive damping',
            actuator_effort_forwarded='actuator collection applied effort after model clipping; excludes solver passive/contact forces',
            solver_actuation_effort_readback='PhysX get_dof_actuation_forces; not total joint reaction or passive damping force',
            state_alignment='joint state before action, command/effort, then resulting post-step state')
        stop = 'diagnostic time limit'
        for step in range(1000):
            if step % adapter.steps_per_control == 0:
                adapter.set_action(torch.zeros((1,13), device=hand.device))
            before_q = to_np(hand.data.joint_pos)[0]
            before_v = to_np(hand.data.joint_vel)[0]
            effort = to_np(adapter.apply())[0]
            forwarded = to_np(hand.actuators.applied_effort)[0]
            obj.write_data_to_sim()
            sim.step(render=False)
            hand.update(cfg.dt); obj.update(cfg.dt); sensor.update(cfg.dt, force_recompute=True)
            state = snapshot()
            pose = state['object_state']
            drift = float(np.linalg.norm(pose[:3]-pose0)*1000)
            tilt = float(np.degrees(np.arccos(np.clip(axis_z_xyzw(pose[3:7])@axis0,-1.,1.))))
            force = to_np(sensor.data.normal_force_matrix_w)[0,0]
            friction_force = to_np(sensor.data.friction_force_matrix_w)[0,0]
            contact_pos = to_np(sensor.data.contact_pos_w)[0,0]
            readback = to_np(hand.root_view.get_dof_actuation_forces())[0,backend_to_user]
            state.update(joint_pos_before=before_q, joint_vel_before=before_v,
                action=np.zeros(13), commands=to_np(adapter.commands)[0],
                motor_effort_requested=effort, actuator_effort_forwarded=forwarded,
                solver_actuation_effort_readback=readback,
                normal_force_matrix_w=force, friction_force_matrix_w=friction_force,
                contact_pos_w=contact_pos, drift_mm=drift, tilt_deg=tilt,
                elapsed_s=(step+1)*cfg.dt)
            # Contact positions may be NaN for absent contacts. Force/state must be finite.
            finite = all(np.isfinite(state[k]).all() for k in (
                'joint_pos','joint_vel','object_state','body_pose','normal_force_matrix_w',
                'friction_force_matrix_w','motor_effort_requested','actuator_effort_forwarded',
                'solver_actuation_effort_readback'))
            state['state_force_finite'] = finite
            frames.append(state)
            record['physics_steps'] = step+1
            if not finite:
                stop = 'nonfinite state or effort/force'; break
            if drift>5.:
                stop = 'position drift >5mm'; break
            if tilt>15.:
                stop = 'axis tilt >15deg'; break
        record['stop_reason'] = stop
    record['complete'] = True
except BaseException as error:
    failed = True
    record['stop_reason'] = 'process error'
    record['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
finally:
    if frames:
        raw = archive/'execution.npz'
        np.savez_compressed(raw, **{k:np.array([f[k] for f in frames]) for k in frames[0]}, physics_dt=cfg.dt)
        record['execution'] = file_record(raw)
        q = np.array([f['joint_pos'] for f in frames])
        finite_rows = np.isfinite(q).all(axis=1)
        names = record['joint_names']
        coupling = {}
        for c in contract['couplings']:
            residual = q[finite_rows,names.index(c['slave'])]-q[finite_rows,names.index(c['master'])]
            coupling[c['finger']] = dict(max_abs_finite_rad=float(np.max(np.abs(residual))) if len(residual) else None,
                last_finite_rad=float(residual[-1]) if len(residual) else None)
        record['coupling_residuals'] = coupling
        for field in ('drift_mm','tilt_deg'):
            values = np.asarray([f[field] for f in frames])
            values = values[np.isfinite(values)]
            record['max_'+field] = float(values.max()) if len(values) else None
        record['contact_force_shape'] = list(frames[-1]['normal_force_matrix_w'].shape)
        record['max_contact_normal_force_N'] = float(max(np.linalg.norm(f['normal_force_matrix_w'],axis=-1).max() for f in frames))
        record['max_abs_joint_speed_rad_s'] = float(max(np.abs(f['joint_vel'][np.isfinite(f['joint_vel'])]).max() for f in frames))
        record['forwarding_max_error_Nm'] = float(max(np.abs(f['actuator_effort_forwarded']-f['motor_effort_requested']).max() for f in frames))
        record['actuation_readback_max_error_Nm'] = float(max(np.abs(f['solver_actuation_effort_readback']-f['motor_effort_requested']).max() for f in frames))
    record['diagnostic_elapsed_s'] = record['physics_steps']*record.get('configuration',{}).get('dt',.0005)
    record['remaining_gates'] = ['certify per-link contact/support interpretation','object/self penetration',
        'all original force/external/numerical gates','long-duration unsupported holding','large-batch throughput']
    save('before close')
    app.close(exit_code=1 if failed else 0)
