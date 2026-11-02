"""Adapt the unchanged C scene to real-measurable student channels.

Tactile v1 is an ideal resultant on each distal rigid body, not a pressure-array
simulation or a calibrated pad ROI. Normal plus friction forces are included.
"""
import torch

from tendonspin.physics.isaac_parallel import tensor
from .observations import JOINT_NAMES, SensorConfig, SensorHistory, joint_limits


class IsaacStudentSensors:
    def __init__(self, env, config=SensorConfig()):
        self.env = env
        p = env.physics
        self.joint_ids = [p.names.index(n) for n in JOINT_NAMES]
        self.distal_names = [f'{f}distal' for f in ('th', 'ff', 'mf', 'rf', 'lf')]
        self.force_ids = [p.filter_names.index(n) for n in self.distal_names]
        self.history = SensorHistory(joint_limits(p.contract), env.num_envs, p.device, config)
        expected = torch.stack((env.lower, env.upper), dim=-1)
        if not torch.equal(expected, self.history.limits[:16]):
            raise ValueError('Student active scaling differs from C teacher')

    def reset(self, ids=None):
        self.history.reset(ids)

    @torch.no_grad()
    def observe(self, observation, reset_ids=None):
        p = self.env.physics
        if reset_ids is not None and len(reset_ids):
            self.reset(reset_ids)
        q = tensor(p.hand.data.joint_pos)[:, self.joint_ids].clone()
        lower, upper = self.history.limits.unbind(-1)
        core = observation['proprio_hist'][:, -1]
        # Reuse the *same* active encoder sample as the teacher, including its
        # existing noise. Passive J1 is sampled independently, never from J2.
        q[:, :16] = (core[:, :16]+1)*(upper[:16]-lower[:16])/2+lower[:16]
        if self.env.training_randomization:
            q[:, 16:] += (torch.rand_like(q[:, 16:])*2-1)*self.env.dr.cfg['joint_noise_rad']
        normal = tensor(p.contact.data.normal_force_matrix_w)[:, 0, self.force_ids]
        friction = tensor(p.contact.data.friction_force_matrix_w)[:, 0, self.force_ids]
        force = (normal+friction).norm(dim=-1)
        valid = torch.ones_like(force, dtype=torch.bool)
        # Contact buffers immediately after a reset can refer to pre-reset
        # physics. They become valid after the first new physical control.
        if reset_ids is None:
            valid[~self.history.initialized] = False
        elif len(reset_ids):
            valid[reset_ids] = False
        frames = self.history.push(q, core[:, 16:], force, valid, torch.zeros_like(force))
        if not torch.allclose(frames[:, -1, :32], core.clamp(-5, 5), atol=1e-6, rtol=0):
            raise RuntimeError('Student and teacher encoder/target sample mismatch')
        return frames
