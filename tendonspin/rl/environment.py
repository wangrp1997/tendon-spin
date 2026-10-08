"""Native privileged rotation task with original per-physics-step demo gates."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .config import Config, ROOT, REVISION, digest, write
from tendonspin.physics import native as audit
from tendonspin.physics.native import SIGNATURE, active_ids, state, spin_increments
from tendonspin.physics.metrics import SpinTracker
AUDIT = ROOT / 'tendonspin/physics'



class RotationEnv:
    def __init__(self, config: Config):
        audit.verify_runtime()
        self.c = config
        self.s, self.original, self.scene_metadata = audit.scene()
        self.m = self.s.model
        self.runner = audit.Runner(self.s, self.original)
        self.initial = state(self.m, self.original)
        self.ids = active_ids(self.s)
        names = [f + 'J' + str(j) for f in ('TH', 'FF', 'MF', 'RF') for j in (4, 3, 2, 1)]
        self.qids = np.array([self.m.jnt_qposadr[self.s.joints[n]] for n in names])
        self.vids = np.array([self.m.jnt_dofadr[self.s.joints[n]] for n in names])
        self.jids = np.array([self.s.joints[n] for n in names])
        self.control_qids = np.array([self.m.jnt_qposadr[self.m.actuator_trnid[i, 0]]
                                    for i in self.ids])
        self.qmid = self.m.jnt_range[self.jids].mean(1)
        self.qhalf = np.maximum(np.diff(self.m.jnt_range[self.jids], axis=1)[:, 0] / 2, .01)
        self.lo, self.hi = self.m.actuator_ctrlrange[self.ids].T
        self.center = self.original.qpos[self.s.qa:self.s.qa + 3].copy()
        self.quaternion = self.original.qpos[self.s.qa + 3:self.s.qa + 7].copy()
        self.R0 = Rotation.from_quat(self.quaternion[[1, 2, 3, 0]]).as_matrix()
        self.stride = round(config.control_dt / self.m.opt.timestep)
        assert self.stride > 0 and abs(self.stride * self.m.opt.timestep - config.control_dt) < 1e-12
        assert len(self.ids) == 13
        # In this model the position actuators use joint transmissions. Any
        # future change to tendon transmissions requires its own observation map.
        assert np.all(self.m.actuator_trntype[self.ids] == mujoco.mjtTrn.mjTRN_JOINT)
        self.directory = None
        self.reset()

    def identity(self):
        sources = {Path(__file__), AUDIT / 'replay.c'}
        for module in list(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if filename:
                path = Path(filename).resolve()
                if path.is_relative_to(ROOT) and path.suffix == '.py' and 'outputs' not in path.parts:
                    sources.add(path)
        return dict(controller=REVISION, scene=self.scene_metadata,
                    engine_sha256=digest(self.runner.runtime), mujoco_version=mujoco.__version__,
                    initial_state_sha256=hashlib.sha256(self.initial.tobytes()).hexdigest(),
                    physics_dt=self.m.opt.timestep, control_dt=self.c.control_dt,
                    actuator_names=[self.m.actuator(int(i)).name for i in self.ids],
                    action_dim=len(self.ids), observation_dim=len(self.observe()),
                    sources=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for p in sorted(sources)])

    def loads(self):
        result = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            if self.s.geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom1 if contact.geom2 == self.s.geom else contact.geom2
            group = self.runner.owner_dict.get(other)
            if group is not None:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.m, self.data, i, force)
                result[group] += force[0]
        return result

    def observe(self):
        d, s = self.data, self.s
        R = Rotation.from_quat(d.qpos[s.qa + 3:s.qa + 7][[1, 2, 3, 0]]).as_matrix()
        obs = np.r_[(d.qpos[self.qids] - self.qmid) / self.qhalf,
                    d.qvel[self.vids] / 2.,
                    2 * (d.ctrl[self.ids] - self.lo) / (self.hi - self.lo) - 1,
                    d.actuator_force[self.ids] / .1,
                    (d.qpos[s.qa:s.qa + 3] - self.center) / .005,
                    (self.R0.T @ R).reshape(-1),
                    d.qvel[s.da:s.da + 3] / .05, d.qvel[s.da + 3:s.da + 6],
                    self.loads() / 2., self.previous_action]
        assert len(obs) == 95 and np.isfinite(obs).all()
        return np.clip(obs, -10., 10.).astype(np.float32)

    def reset(self, max_seconds=None, directory=None, attribution=None):
        self.data = mujoco.MjData(self.m)
        mujoco.mj_copyData(self.data, self.m, self.original)
        np.testing.assert_array_equal(state(self.m, self.data), self.initial)
        self.previous_action = np.zeros(13)
        self.total_steps = self.valid_steps = 0
        self.net = self.forward = self.backward = self.peak = self.peak_seconds = 0.
        self.max_drift = self.max_tilt = self.max_force = self.max_object_pen = self.max_self_pen = 0.
        self.min_loaded = 5
        self.return_ = 0.
        self.done = False
        seconds = self.c.train_episode_s if max_seconds is None else max_seconds
        self.max_steps = round(seconds / self.m.opt.timestep)
        assert self.max_steps > 0
        self.records = []
        self.directory = Path(directory).resolve() if directory is not None else None
        if self.directory:
            assert self.directory.is_relative_to(ROOT / 'outputs')
            self.directory.mkdir(parents=True, exist_ok=False)
            self.attribution = attribution or {}
            self.manifest = self.identity()
            self.manifest.update(config=asdict(self.c), max_seconds=seconds,
                                 attribution=self.attribution, original_state=True,
                                 episode_resets=0, controller_switches=0, learning_during_evaluation=False)
            write(self.directory / 'protocol.json', self.manifest)
            for record in self.manifest['sources']:
                target = self.directory / 'source' / record['path']
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / record['path']).read_bytes())
        return self.observe()

    def step(self, action):
        if self.done:
            raise RuntimeError('Explicit reset required after termination')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (13,) or not np.isfinite(action).all() or np.max(abs(action)) > 1 + 1e-6:
            raise ValueError('13 finite bounded position-increment actions required')
        action = np.clip(action, -1., 1.)
        old = self.data.ctrl.copy()
        target = old.copy()
        target[self.ids] = np.clip(old[self.ids] + action * self.c.command_speed_rad_s * self.c.control_dt,
                                   self.lo, self.hi)
        count = min(self.stride, self.max_steps - self.total_steps)
        fraction = np.arange(1, count + 1) / self.stride
        controls = old + fraction[:, None] * (target - old)
        q0 = self.data.qpos[self.s.qa + 3:self.s.qa + 7].copy()
        record = self.runner.run(self.data, controls, stop=True)
        n = len(record['ctrl'])
        failed = bool(record['flags'][-1])
        valid = n - int(failed)
        increments = spin_increments(record['qpos'][None, :, self.s.qa + 3:self.s.qa + 7], q0, -1)[0]
        angles = self.net + np.cumsum(increments)
        if valid:
            metrics = record['metrics'][:valid]
            peak = int(np.argmax(angles[:valid]))
            if angles[peak] > self.peak:
                self.peak, self.peak_seconds = float(angles[peak]), (self.total_steps + peak + 1) * self.m.opt.timestep
            self.net = float(angles[valid - 1])
            self.forward += float(np.maximum(increments[:valid], 0).sum())
            self.backward += float(-np.minimum(increments[:valid], 0).sum())
            self.max_drift = max(self.max_drift, float(metrics[:, 0].max()))
            self.max_tilt = max(self.max_tilt, float(metrics[:, 1].max()))
            self.max_force = max(self.max_force, float(metrics[:, 8].max()))
            self.max_object_pen = max(self.max_object_pen, float(metrics[:, 10].max()))
            self.max_self_pen = max(self.max_self_pen, float(metrics[:, 11].max()))
            self.min_loaded = min(self.min_loaded, int((metrics[:, 2:7] > 1e-6).sum(1).min()))
            duration = valid * self.m.opt.timestep
            # Saturation affects training reward only. The demo score always
            # integrates uncapped, signed rotation at every physics step.
            spin_rate = np.radians(increments[:valid].sum()) / duration
            reward = self.c.rotation_reward_per_rad * duration * np.clip(
                spin_rate, -self.c.reward_spin_cap_rad_s, self.c.reward_spin_cap_rad_s)
            reward -= duration * (self.c.position_cost_per_s * np.mean((metrics[:, 0] / 3.) ** 2)
                                  + self.c.tilt_cost_per_s * np.mean((metrics[:, 1] / 10.) ** 2)
                                  + self.c.tracking_cost_per_s * np.mean(
                                      ((record['ctrl'][:valid, self.ids] - record['qpos'][:valid, self.control_qids]) / .1) ** 2)
                                  + self.c.action_cost_per_s * np.mean(action ** 2))
        else:
            reward = 0.
        if failed:
            reward -= self.c.failure_penalty
        self.return_ += float(reward)
        self.total_steps += n
        self.valid_steps += valid
        self.previous_action = action.copy()
        truncated = self.total_steps >= self.max_steps and not failed
        self.done = failed or truncated
        reason = '+'.join(name for i, name in enumerate(audit.FLAGS)
                          if int(record['flags'][-1]) & (1 << i)) if failed else 'time_limit'
        if self.directory:
            record['angle_deg'] = angles
            self.records.append(record)
        info = None
        if self.done:
            info = dict(controller=REVISION, valid_seconds=self.valid_steps * self.m.opt.timestep,
                        total_seconds=self.total_steps * self.m.opt.timestep,
                        net_deg=self.net, peak_net_deg=self.peak, peak_seconds=self.peak_seconds,
                        forward_deg=self.forward, backward_deg=self.backward,
                        reason=reason, physical_failure=failed, return_=self.return_,
                        max_drift_mm=self.max_drift, max_tilt_deg=self.max_tilt,
                        max_group_force_N=self.max_force, min_loaded_fingers=self.min_loaded,
                        max_object_penetration_mm=self.max_object_pen, max_self_penetration_mm=self.max_self_pen,
                        failure_frame=record['metrics'][-1].tolist() if failed else None,
                        failure_flags=int(record['flags'][-1]) if failed else 0,
                        valid_steps=self.valid_steps, total_steps=self.total_steps,
                        original_state=True, episode_resets=0, controller_switches=0,
                        privileged=True, indefinite_rotation_proven=False, hardware_robustness_proven=False)
            if self.directory:
                info.update(self.finish(info))
        return self.observe(), float(reward), failed, truncated, info

    def finish(self, info):
        keys = ('ctrl', 'qpos', 'qvel', 'motor', 'metrics', 'flags', 'angle_deg')
        saved = {k: np.concatenate([r[k] for r in self.records]) for k in keys}
        saved.update(initial_state=self.initial, final_state=state(self.m, self.data),
                     physics_dt=np.array(self.m.opt.timestep), valid_steps=np.array(self.valid_steps))
        raw = self.directory / 'execution.npz'
        np.savez_compressed(raw, **saved)
        replay = mujoco.MjData(self.m)
        mujoco.mj_copyData(replay, self.m, self.original)
        check = self.runner.run(replay, saved['ctrl'], stop=True)
        self.runner.original_quaternion = self.quaternion
        self.runner.check_physical(replay, check)
        errors = {k: float(abs(check[k] - saved[k]).max())
                  for k in ('ctrl', 'qpos', 'qvel', 'motor', 'metrics', 'flags', 'final_state')}
        assert not any(errors.values()), errors
        tracker = SpinTracker(self.quaternion, -1)
        for q in saved['qpos'][:self.valid_steps, self.s.qa + 3:self.s.qa + 7]:
            tracker.update(q)
        assert abs(np.degrees(tracker.net) - self.net) < 1e-7
        assert abs(self.forward - self.backward - self.net) < 1e-7
        assert all(digest(ROOT / r['path']) == r['sha256'] for r in self.manifest['sources'])
        result = dict(info, attribution=self.attribution, replay_errors=errors,
                      spin_tracker_error_deg=float(abs(np.degrees(tracker.net) - self.net)),
                      sources_unchanged=True,
                      execution=dict(path=str(raw.relative_to(ROOT)), sha256=digest(raw)),
                      protocol=dict(path=str((self.directory / 'protocol.json').relative_to(ROOT)),
                                    sha256=digest(self.directory / 'protocol.json')))
        write(self.directory / 'results.json', result)
        self.records = []
        return result
