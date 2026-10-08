# Sources: native Boya/MuJoCo WXYZ contract and Isaac Lab 3 XYZW API.
# New TendonSpin conversion and initialization audit; no physics integration.
"""Explicit quaternion boundaries and an original-state transfer check."""
import numpy as np


def wxyz_to_xyzw(quaternion):
    value = np.asarray(quaternion, dtype=float)
    if value.shape[-1:] != (4,):
        raise ValueError("Quaternion must have four components")
    return value[..., [1, 2, 3, 0]].copy()


def xyzw_to_wxyz(quaternion):
    value = np.asarray(quaternion, dtype=float)
    if value.shape[-1:] != (4,):
        raise ValueError("Quaternion must have four components")
    return value[..., [3, 0, 1, 2]].copy()


def axis_z_xyzw(quaternion):
    value = np.asarray(quaternion, dtype=float)
    value = value / np.linalg.norm(value, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(value, -1, 0)
    return np.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), axis=-1)


def orientation_error_deg(actual_xyzw, expected_wxyz):
    actual = np.asarray(actual_xyzw, dtype=float)
    expected = wxyz_to_xyzw(expected_wxyz)
    actual = actual / np.linalg.norm(actual, axis=-1, keepdims=True)
    expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
    cosine = np.abs(np.sum(actual*expected, axis=-1))
    return np.degrees(2*np.arccos(np.clip(cosine, 0., 1.)))


def audit_initial_state(contract, joint_names, body_names, state):
    """Numerical transfer tolerances, distinct from task physical-validity gates."""
    limits = dict(position_mm=.05, orientation_deg=.05, joint_rad=1e-6,
                  joint_speed_rad_s=1e-6, object_speed=1e-6)
    expected = {b['name']: b for b in contract['body_frames']}
    rows = []
    for name, pose in zip(body_names, state['body_pose'], strict=True):
        source = expected[name]
        rows.append(dict(body=name,
            position_mm=float(np.linalg.norm(pose[:3]-source['pos'])*1000),
            orientation_deg=float(orientation_error_deg(pose[3:7], source['quat']))))
    joint_by_name = {j['name']: j for j in contract['joints']}
    object_state = state['object_state']
    obj = contract['object']
    errors = dict(
        max_body_position_mm=max(r['position_mm'] for r in rows),
        max_body_orientation_deg=max(r['orientation_deg'] for r in rows),
        object_position_mm=float(np.linalg.norm(object_state[:3]-obj['pos'])*1000),
        object_orientation_deg=float(orientation_error_deg(object_state[3:7], obj['quat'])),
        max_joint_rad=float(np.max(np.abs(state['joint_pos']-[joint_by_name[n]['qpos'] for n in joint_names]))),
        max_joint_speed_rad_s=float(np.max(np.abs(state['joint_vel']-[joint_by_name[n]['qvel'] for n in joint_names]))),
        max_object_speed_error=float(np.max(np.abs(object_state[7:13]-(obj['lin_vel_world']+obj['ang_vel_world'])))))
    finite = all(np.isfinite(state[k]).all() for k in ('body_pose', 'object_state', 'joint_pos', 'joint_vel'))
    checks = dict(all_state_finite=bool(finite),
        body_positions=errors['max_body_position_mm']<=limits['position_mm'],
        body_orientations=errors['max_body_orientation_deg']<=limits['orientation_deg'],
        object_position=errors['object_position_mm']<=limits['position_mm'],
        object_orientation=errors['object_orientation_deg']<=limits['orientation_deg'],
        joint_positions=errors['max_joint_rad']<=limits['joint_rad'],
        joint_velocities=errors['max_joint_speed_rad_s']<=limits['joint_speed_rad_s'],
        object_velocities=errors['max_object_speed_error']<=limits['object_speed'])
    return dict(passed=all(checks.values()), tolerances=limits, checks=checks,
                errors=errors, body_comparison=rows)
