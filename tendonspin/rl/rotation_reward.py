# Sources: TendonSpin isaac_hora.py::spin_increment, commit0c48392 (world XYZW
# quaternion delta), and HaozhiQi/hora v0.0.1 compute_hand_reward (MIT).
# Repositories: github.com/wangrp1997/tendon-spin; github.com/HaozhiQi/hora.
# Paper: In-Hand Object Rotation via Rapid Motor Adaptation, arXiv:2210.04887.
# Boya port: replace the reported angular-velocity input with the mean of actual
# per-physics-step world rotation vectors. Keep Hora's fixed target axis,
# clipping, scaling and other reward terms. This is not original Hora reward input.
"""Explicit, shared rotation-reward signals for training and frozen evaluation."""
import math

REPORTED = 'hora_reported_velocity'
POSE_DELTA = 'hora_pose_delta'
PROFILES = (REPORTED, POSE_DELTA)


def configuration(profile, physics_dt, steps_per_control):
    if profile not in PROFILES:
        raise ValueError('Unknown rotation reward profile: '+str(profile))
    if not math.isfinite(physics_dt) or physics_dt <= 0 or steps_per_control < 1:
        raise ValueError('Positive physics interval and control step count required')
    return dict(profile=profile, physics_dt=physics_dt, steps_per_control=steps_per_control,
        control_dt=physics_dt*steps_per_control, frame='world',
        target_axis='negative original-grasp cylinder Z; fixed, unchanged from the Hora port',
        velocity_source='measured XYZW pose increments' if profile == POSE_DELTA else 'engine-reported angular velocity',
        aggregation='sum shortest world rotation vectors / control_dt' if profile == POSE_DELTA else 'last physics sample of control',
        pose_arithmetic='float64; cast velocity to native tensor dtype before Hora reward' if profile == POSE_DELTA else None,
        rotation_scale=1., angular_velocity_clip=[-.5, .5],
        remaining_terms='original Hora linear/pose/torque/work penalties, unchanged')


def world_rotation_increment_xyzw(previous, current):
    """Shortest world rotation vector for current * inverse(previous), radians.

    Double arithmetic limits cancellation between near-identical float32 poses.
    Each PHYSICS increment must be below pi; a complete control interval may
    exceed pi because its increments are summed rather than endpoint-wrapped.
    Quaternion signs are equivalent and do not create a turn.
    """
    import torch
    previous=previous.to(torch.float64)
    current=current.to(torch.float64)
    previous=previous/previous.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    current=current/current.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    a,b=previous[...,:3],current[...,:3]
    vector=previous[...,3:]*b-current[...,3:]*a-torch.cross(b,a,dim=-1)
    scalar=(previous*current).sum(-1)
    sign=torch.where(scalar<0,-1.,1.)
    vector=vector*sign[...,None];scalar=scalar*sign
    norm=vector.norm(dim=-1)
    return vector*(2*torch.atan2(norm,scalar)/norm.clamp_min(1e-12))[...,None]


class RotationRewardSignal:
    """One control interval, seeded AFTER any environment reset.

    No state is carried across control intervals or reset poses. Training and
    evaluation use this same accumulator; no contact or success masks are added.
    """
    def __init__(self, profile, physics_dt, steps_per_control):
        self.configuration=configuration(profile,physics_dt,steps_per_control)
        self.profile=profile
        self.steps_per_control=steps_per_control
        self.steps=0
        self.previous=None
        self.total_rotation=None

    def begin(self, orientation):
        self.steps=0
        if self.profile == POSE_DELTA:
            import torch
            self.previous=orientation.clone()
            self.total_rotation=torch.zeros_like(orientation[...,:3],dtype=torch.float64)

    def advance(self, orientation):
        if self.profile == POSE_DELTA:
            if self.previous is None:
                raise RuntimeError('Begin a control interval before recording poses')
            self.total_rotation+=world_rotation_increment_xyzw(self.previous,orientation)
            self.previous=orientation.clone()
        self.steps+=1

    def angular_velocity(self, reported_velocity):
        if self.steps != self.steps_per_control:
            raise RuntimeError('Rotation reward requires one complete control interval')
        if self.profile == REPORTED:
            return reported_velocity
        return (self.total_rotation/self.configuration['control_dt']).to(reported_velocity.dtype)


def require_checkpoint_configuration(contract, expected):
    """Never silently label an old checkpoint as trained with a new reward."""
    saved=contract.get('reward_configuration')
    if saved is None:
        if expected['profile'] != REPORTED:
            raise ValueError('Legacy checkpoint lacks pose-delta reward identity')
    elif saved != expected:
        raise ValueError('Checkpoint reward configuration differs from evaluation')
