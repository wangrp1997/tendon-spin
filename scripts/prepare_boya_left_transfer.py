"""Prepare and statically calibrate the left-hand transfer candidate.

The output is explicitly a preflight candidate: MuJoCo contact geometry is used
to fit the initial grasps, but no Isaac or hardware run is claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from tendonspin.physics.hand_asset_audit import audit_hand_asset
from tendonspin.physics.hand_profiles import LEFT_MODEL_SOURCE, resolve_hand_profile
from tendonspin.physics.left_grasp import LeftGraspCalibrator
from tendonspin.interfaces import ACTION_NAMES
from tendonspin.students.model import sha256
from tendonspin.students.training import SELECTED_TEACHER_SHA256


def json_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def active_limits(contract):
    return {a['joint']: a['control_range'] for a in contract['actuators']
            if a['joint'] in ACTION_NAMES}


def q_limit_errors(names, q, limits):
    errors = {}
    q = np.asarray(q)
    if q.ndim == 1:
        q = q[None]
    for index, name in enumerate(names):
        if name not in limits:
            continue
        lower, upper = limits[name]
        low = float(np.min(q[:, index]))
        high = float(np.max(q[:, index]))
        if low < lower-1e-6 or high > upper+1e-6:
            errors[name] = dict(min=low, max=high, limits=[lower, upper])
    return errors


def shift_commands(commands, original_q, calibrated_q, names, motors, contract, limits,
                   mode='bias'):
    result = np.asarray(commands, dtype=np.float32).copy()
    by_joint = {a['joint']: a for a in contract['actuators']}
    for index, motor in enumerate(motors):
        joint = by_joint[motor]['joint']
        if joint not in limits:
            continue
        if mode == 'position':
            target = calibrated_q[names.index(joint)]
        else:
            delta = calibrated_q[names.index(joint)]-original_q[names.index(joint)]
            target = result[index]+delta
        result[index] = float(np.clip(target, *limits[joint]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--left-asset', type=Path,
        help='V3.2 left_hand.usd override; defaults to the bundled snapshot')
    parser.add_argument('--teacher', type=Path, default=Path(
        'outputs/boya_c_fresh1p5b_server_interrupted_recovery_20261028_v1/'
        'actions_1260000000/training/teacher_final.pth'))
    parser.add_argument('--target-distance-m', type=float, default=-1e-4,
        help='Signed static distal-cylinder target; positive values leave clearance')
    parser.add_argument('--command-mode', choices=('bias', 'position'), default='position',
        help='Preserve source command bias or start with command equal to calibrated position')
    args = parser.parse_args()
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    teacher_path = args.teacher if args.teacher.is_absolute() else root/args.teacher
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    if teacher.get('contract', {}).get('arm') != 'C':
        raise ValueError('Left transfer requires the selected C teacher contract')
    if sha256(teacher_path) != SELECTED_TEACHER_SHA256:
        raise ValueError('Teacher identity differs from selected C1260M')

    profile = resolve_hand_profile(root, 'left', args.left_asset)
    contract = profile.adapt_contract(json.loads(profile.contract_path.read_text()))
    hand_quat = np.asarray(contract['hand_root']['quat'], dtype=float)
    teacher_scene = teacher['contract']['scene_configuration']
    scene = profile.adapt_scene_configuration(teacher_scene)
    source_cache = root/teacher_scene['training_cache']
    with np.load(source_cache, allow_pickle=False) as saved:
        cache = {name: saved[name].copy() for name in saved.files}
    names = cache['joint_names'].astype(str).tolist()
    motors = cache['motor_names'].astype(str).tolist()
    contract_joints = [joint['name'] for joint in contract['joints']]
    if len(names) != len(set(names)) or set(names) != set(contract_joints):
        raise ValueError('Source cache joint names differ from contract')
    if motors != [actuator['joint'] for actuator in contract['actuators']]:
        raise ValueError('Source cache motor order differs from contract')
    if not profile.validation_scene:
        raise FileNotFoundError('Left MuJoCo validation scene is unavailable')

    audit = audit_hand_asset(profile, contract)
    write_json(out/'asset_audit.json', audit)
    if not audit['all_pass']:
        raise ValueError('Left USD contract audit failed: '+str(audit['errors']))
    limits = active_limits(contract)
    source_q_errors = q_limit_errors(names, cache['q'], limits)
    if source_q_errors:
        raise ValueError('Source grasp cache exceeds left joint control ranges: '
                         +str(source_q_errors))

    if profile.model_source == LEFT_MODEL_SOURCE:
        # The strict mirror preserves every right-hand physical parameter and
        # joint coordinate. Re-fitting q would move the frozen teacher out of
        # its training distribution, so preserve the complete source cache.
        output_cache = {name: value.copy() for name, value in cache.items()}
        group_counts = np.bincount(
            output_cache['scale_id'], minlength=9).tolist()
        if any(count == 0 for count in group_counts):
            raise RuntimeError(
                f'Mirrored cache does not cover all nine size groups: {group_counts}')
        cache_path = out/'grasp_cache_left_candidate_v1.npz'
        np.savez_compressed(cache_path, **output_cache)
        scene['name'] += '_strict_right_mirror_v1'
        scene['training_cache'] = str(cache_path)
        scene['training_cache_size'] = int(len(output_cache['q']))
        scene['training_cache_sha256'] = sha256(cache_path)
        scene['scale_cache_counts'] = group_counts
        scene['configuration_sha256'] = json_sha256(scene)
        write_json(out/'scene_configuration_left_candidate.json', scene)
        write_json(out/'hand_profile.json', profile.identity())
        write_json(out/'calibration.json', dict(
            schema='tendonspin-left-grasp-preflight-v2',
            method='strict right-mirror cache preservation',
            source_cache=str(source_cache), source_cache_sha256=sha256(source_cache),
            output_cache=str(cache_path), output_cache_sha256=sha256(cache_path),
            source_size=len(cache['q']), output_size=len(output_cache['q']),
            scale_cache_counts=group_counts,
            joint_limit_errors=source_q_errors,
            validation=('Exact mirrored right URDF/MJCF; source q, commands and '
                        'object state retained without recalibration.'),
            samples=[]))
        write_json(out/'status.json', dict(status='completed',
            wall_s=time.monotonic()-started, all_isaac_closed_loop_validated=False,
            hardware_validated=False, included=len(output_cache['q']), excluded=0,
            output_cache=str(cache_path)))
        print(json.dumps(dict(status='completed', included=len(output_cache['q']),
            excluded=0, groups=group_counts, mode='strict_mirror_cache_copy')), flush=True)
        return

    calibrator = LeftGraspCalibrator(profile.validation_scene, names)
    nominal_scale = float(
        teacher['contract']['domain_randomization']['logical_nominal_scale'])
    base_diameter = float(scene['geometry']['diameter_m'])
    base_height = float(scene['geometry']['height_m'])
    records = []
    calibrated_q = []
    calibrated_commands = []
    included = []
    reasons = {}
    for index, (q, commands, state, logical_scale) in enumerate(zip(
            cache['q'], cache['commands'], cache['object_state'],
            cache['logical_scale'])):
        ratio = float(logical_scale)/nominal_scale
        fit = calibrator.calibrate(q, state, hand_quat,
            base_diameter*ratio, base_height*ratio,
            target_distance=args.target_distance_m, joint_limits=limits)
        output_commands = shift_commands(commands, q, fit['joint_pos'],
            names, motors, contract, limits, mode=args.command_mode)
        limit_errors = q_limit_errors(names, fit['joint_pos'][None], limits)
        passed = fit['all_fingers_contact'] and not limit_errors
        records.append(dict(
            candidate_id=int(cache['candidate_id'][index]),
            logical_scale=float(logical_scale),
            scale_id=int(cache['scale_id'][index]), passed=bool(passed),
            max_abs_contact_distance_m=fit['max_abs_distance_m'],
            contact_distance_m=fit['contact_distance_m'],
            finger_delta_rad=fit['finger_delta_rad'],
            object_position_local_left=fit['object_position_local_left'].tolist(),
            object_quat_wxyz_local_left=fit['object_quat_wxyz_local_left'].tolist(),
            joint_limit_errors=limit_errors))
        if passed:
            included.append(index)
            calibrated_q.append(fit['joint_pos'].astype(np.float32))
            calibrated_commands.append(output_commands)
        else:
            reason = 'joint limit' if limit_errors else 'finger contact'
            reasons[reason] = reasons.get(reason, 0)+1
        if (index+1) % 20 == 0:
            print(json.dumps(dict(phase='calibrating', completed=index+1,
                included=len(included))), flush=True)
    if not included:
        raise RuntimeError('No static left-hand candidates passed')
    included = np.asarray(included, dtype=np.int64)
    output_cache = dict(
        q=np.asarray(calibrated_q, dtype=np.float32),
        object_state=cache['object_state'][included],
        commands=np.asarray(calibrated_commands, dtype=np.float32),
        scale_id=cache['scale_id'][included],
        logical_scale=cache['logical_scale'][included],
        candidate_id=cache['candidate_id'][included],
        joint_names=cache['joint_names'], motor_names=cache['motor_names'])
    group_counts = np.bincount(output_cache['scale_id'], minlength=9).tolist()
    if any(count == 0 for count in group_counts):
        raise RuntimeError(
            f'Calibrated cache does not cover all nine size groups: {group_counts}')
    cache_path = out/'grasp_cache_left_candidate_v1.npz'
    np.savez_compressed(cache_path, **output_cache)

    nominal = scene['nominal_state']
    source_nominal = teacher_scene['nominal_state']
    nominal_q = np.asarray(source_nominal['joint_pos'], dtype=float)
    nominal_state = np.asarray(
        source_nominal['object_state_xyzw_world_twist'], dtype=float)
    nominal_fit = calibrator.calibrate(nominal_q, nominal_state, hand_quat,
        base_diameter, base_height,
        target_distance=args.target_distance_m, joint_limits=limits)
    nominal_commands = shift_commands(
        source_nominal['commands'], nominal_q, nominal_fit['joint_pos'],
        names, motors, contract, limits, mode=args.command_mode)
    nominal['joint_pos'] = nominal_fit['joint_pos'].tolist()
    nominal['commands'] = nominal_commands.tolist()
    scene['name'] += '_mujoco_contact_calibrated_v1'
    scene['training_cache'] = str(cache_path)
    scene['training_cache_size'] = int(len(output_cache['q']))
    scene['training_cache_sha256'] = sha256(cache_path)
    scene['scale_cache_counts'] = group_counts
    scene['configuration_sha256'] = json_sha256(scene)

    write_json(out/'scene_configuration_left_candidate.json', scene)
    write_json(out/'hand_profile.json', profile.identity())
    write_json(out/'calibration.json', dict(
        schema='tendonspin-left-grasp-preflight-v1',
        source_cache=str(source_cache), source_cache_sha256=sha256(source_cache),
        output_cache=str(cache_path), output_cache_sha256=sha256(cache_path),
        source_size=len(cache['q']), output_size=len(output_cache['q']),
        scale_cache_counts=group_counts, failure_reasons=reasons,
        nominal=dict(passed=nominal_fit['all_fingers_contact'],
            max_abs_contact_distance_m=nominal_fit['max_abs_distance_m'],
            contact_distance_m=nominal_fit['contact_distance_m'],
            finger_delta_rad=nominal_fit['finger_delta_rad']),
        target_distance_m=args.target_distance_m,
        command_mode=args.command_mode,
        validation='MuJoCo static distal-cylinder distance fit; no Isaac or hardware claim',
        samples=records))
    write_json(out/'status.json', dict(status='completed',
        wall_s=time.monotonic()-started, all_isaac_closed_loop_validated=False,
        hardware_validated=False, included=len(output_cache['q']),
        excluded=len(cache['q'])-len(output_cache['q']),
        output_cache=str(cache_path)))
    print(json.dumps(dict(status='completed', included=len(output_cache['q']),
        excluded=len(cache['q'])-len(output_cache['q']),
        groups=group_counts, output_cache=str(cache_path))), flush=True)


if __name__ == '__main__':
    main()
