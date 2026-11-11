"""Static left-hand grasp fitting against the task cylinder.

This is a MuJoCo preflight, not an Isaac or hardware acceptance result.
"""
import mujoco
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


FINGERS = ('TH', 'FF', 'MF', 'RF', 'LF')
FINGER_JOINTS = {
    'TH': ('THJ3', 'THJ2', 'THJ1'),
    **{finger: (f'{finger}J3', f'{finger}J2')
       for finger in ('FF', 'MF', 'RF', 'LF')},
}


def left_local_object_state(hand_quat_wxyz, state_xyzw_world_twist):
    """Transform canonical right-world xyz state into the left hand frame."""
    w, x, y, z = hand_quat_wxyz
    root = Rotation.from_quat([x, y, z, w]).as_matrix()
    mirror = np.diag([1., -1., 1.])
    state = np.asarray(state_xyzw_world_twist, dtype=float)
    position = root.T @ (mirror @ state[:3])
    orientation = root.T @ mirror @ Rotation.from_quat(
        state[3:7]).as_matrix() @ mirror
    quat_xyzw = Rotation.from_matrix(orientation).as_quat()
    return position, np.r_[quat_xyzw[3], quat_xyzw[:3]]


class LeftGraspCalibrator:
    def __init__(self, scene_path, joint_names, object_body='object'):
        self.spec = mujoco.MjSpec.from_file(str(scene_path))
        self.object_body = self.spec.body(object_body)
        if self.object_body is None or len(self.object_body.geoms) != 1:
            raise ValueError('Validation scene requires one object proxy geom')
        self.object_body.geoms[0].type = mujoco.mjtGeom.mjGEOM_CYLINDER
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.joint_names = tuple(joint_names)
        self.object_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        self.object_joint_id = self.model.body_jntadr[self.object_id]
        self.object_qpos = self.model.jnt_qposadr[self.object_joint_id]
        self.object_geom = self.model.body_geomadr[self.object_id]
        self.distal_geoms = {}
        for finger in FINGERS:
            body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f'{finger.lower()}distal')
            if body < 0:
                raise ValueError(f'Missing distal body for {finger}')
            self.distal_geoms[finger] = self.model.body_geomadr[body]

    def _write(self, q, object_position, object_quat_wxyz, diameter, height):
        self.model.geom_size[self.object_geom] = [
            diameter/2, height/2, 0.]
        for name, value in zip(self.joint_names, q):
            joint = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint >= 0:
                self.data.qpos[self.model.jnt_qposadr[joint]] = value
        self.data.qpos[self.object_qpos:self.object_qpos+3] = object_position
        self.data.qpos[self.object_qpos+3:self.object_qpos+7] = object_quat_wxyz
        mujoco.mj_forward(self.model, self.data)

    def _distance(self, finger):
        return float(mujoco.mj_geomDistance(
            self.model, self.data, self.distal_geoms[finger],
            self.object_geom, .2, np.zeros(6)))

    @staticmethod
    def _delta_bounds(joint_names, q, names, limits):
        lower, upper = -1., 1.
        for name in names:
            if name not in limits or name not in joint_names:
                continue
            base = q[joint_names.index(name)]
            lower = max(lower, float(limits[name][0])-base)
            upper = min(upper, float(limits[name][1])-base)
        return lower, upper

    def calibrate(self, q, object_state_xyzw_world_twist, hand_quat_wxyz,
                  diameter, height, target_distance=-1e-4, joint_limits=None):
        q = np.asarray(q, dtype=float).copy()
        joint_limits = joint_limits or {}
        position, quat = left_local_object_state(
            hand_quat_wxyz, object_state_xyzw_world_twist)
        deltas = {}
        for finger, variables in FINGER_JOINTS.items():
            if any(name not in self.joint_names for name in variables):
                raise ValueError(f'Missing fit joint for {finger}: {variables}')
            def candidate_for(flex_delta, abduct_delta=0.):
                candidate = q.copy()
                candidate[self.joint_names.index(finger+'J4')] += abduct_delta
                for name in variables:
                    candidate[self.joint_names.index(name)] += flex_delta
                if finger != 'TH':
                    candidate[self.joint_names.index(finger+'J1')] = \
                        candidate[self.joint_names.index(finger+'J2')]
                return candidate

            def residual(flex_delta, abduct_delta=0.):
                candidate = candidate_for(flex_delta, abduct_delta)
                self._write(candidate, position, quat, diameter, height)
                return self._distance(finger)-target_distance

            flex_lower, flex_upper = self._delta_bounds(
                self.joint_names, q, variables, joint_limits)
            if flex_lower > flex_upper:
                raise ValueError(f'No feasible flexion range for {finger}')

            def solve_flex(abduct_delta):
                grid = np.linspace(flex_lower, flex_upper, 81)
                values = np.asarray([
                    residual(float(value), abduct_delta) for value in grid])
                for left, right, a, b in zip(
                        grid[:-1], grid[1:], values[:-1], values[1:]):
                    if a == 0:
                        return float(left), 0.
                    if a*b < 0:
                        delta = float(brentq(
                            lambda value: residual(value, abduct_delta),
                            float(left), float(right), xtol=1e-6))
                        return delta, residual(delta, abduct_delta)
                index = int(np.argmin(np.abs(values)))
                return float(grid[index]), float(values[index])

            flex_delta, best_error = solve_flex(0.)
            abduct_delta = 0.
            if abs(best_error) > 1e-4:
                abduct_lower, abduct_upper = self._delta_bounds(
                    self.joint_names, q, (finger+'J4',), joint_limits)
                trials = [
                    value for value in np.arange(-.3, .301, .05)
                    if value and abduct_lower <= value <= abduct_upper]
                for trial in sorted(trials, key=abs):
                    trial_flex, trial_error = solve_flex(float(trial))
                    if abs(trial_error) < abs(best_error):
                        flex_delta, best_error = trial_flex, trial_error
                        abduct_delta = float(trial)
            q = candidate_for(flex_delta, abduct_delta)
            deltas[finger] = dict(J4=abduct_delta, flexor=float(flex_delta))

        self._write(q, position, quat, diameter, height)
        distances = {finger: self._distance(finger) for finger in FINGERS}
        return dict(joint_pos=q, finger_delta_rad=deltas,
            contact_distance_m=distances, max_abs_distance_m=max(
                abs(value) for value in distances.values()),
            all_fingers_contact=all(abs(value-target_distance) <= 1e-4
                                    for value in distances.values()),
            object_position_local_left=position,
            object_quat_wxyz_local_left=quat,
            target_distance_m=target_distance)
