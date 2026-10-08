# Reference: sharpa-rl-lab 5accf024d376685eaa17da7aa4614498217eab4d,
# rl_isaaclab/tasks/inhand_rotate/sharpa_wave_env.py::_apply_action (BSD-3-Clause).
# Reuse: TendonSpin SourcePositionAdapter, named Boya finger inputs/18 motors.
# Port: bounded external PD, declared prototype armature, passive tendon damping
# reflected into the master motor, original named collision exclusions restored.
"""Sharpa-style motor organization for Boya; nominal simulation parameters only."""
import xml.etree.ElementTree as ET

import torch
from isaaclab.actuators import IdealPDActuatorCfg
from .isaac_boya import SourcePositionAdapter
from tendonspin.interfaces import finger_action_contract

CONTROLLER = 'boya_sharpa_fingers16_pd_v4'


def motor_parameters(contract):
    damping = {j['name']: j['damping'] for j in contract['joints']}
    # For q_slave = ratio*q_master, dissipated power maps D_slave*ratio**2
    # onto the master. The passive joints receive no independent motor effort.
    for c in contract['couplings']:
        damping[c['master']] += damping[c['slave']] * c['multiplier'] ** 2
    motors = {a['joint']: a for a in contract['actuators']}
    caps = {j['name']: max(abs(v) for v in motors[j['name']]['force_range'])
            if j['name'] in motors else 0. for j in contract['joints']}
    # Numerical prototype, NOT measured Boya rotor inertia. Set D*dt/A=0.25
    # using the declared integration interval, rather than copying Sharpa values.
    armature = {name: 4 * contract['physics_dt'] * damping[name] if name in motors else 0.
                for name in damping}
    return dict(damping=damping, armature=armature, caps=caps,
                armature_semantics='unidentified nominal simulation assumption: A=4*D_effective*dt on 18 motors')


def actuator_configuration(parameters):
    return {'motor_forwarder': IdealPDActuatorCfg(
        joint_names_expr=['.*'], stiffness=0., damping=0.,
        actuator_effort_limit=parameters['caps'], joint_effort_limit=parameters['caps'],
        velocity_limit_sim=100., armature=parameters['armature'],
        friction=0., dynamic_friction=0., viscous_friction=0.)}


def configure_constraints(stage, root, contract, hand_prefix='/World/Hand'):
    from pxr import UsdPhysics, Sdf
    bodies = {p.GetName(): p for p in stage.Traverse()
              if str(p.GetPath()).startswith(hand_prefix+'/') and p.HasAPI(UsdPhysics.RigidBodyAPI)}
    excluded = []
    for entry in ET.parse(root/'assets/grasp/scene.xml').getroot().findall('./contact/exclude'):
        a, b = entry.attrib['body1'], entry.attrib['body2']
        UsdPhysics.FilteredPairsAPI.Apply(bodies[a]).CreateFilteredPairsRel().AddTarget(bodies[b].GetPath())
        excluded.append([a, b])
    joints = {p.GetName(): p for p in stage.Traverse()
              if str(p.GetPath()).startswith(hand_prefix+'/') and p.IsA(UsdPhysics.RevoluteJoint)}
    # Preserve the currently documented NewtonMimicAPI, make its coefficients
    # explicit. This alone does not claim equivalence to MuJoCo solref/solimp.
    for c in contract['couplings']:
        prim = joints[c['slave']]
        prim.CreateAttribute('newton:mimicEnabled', Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute('newton:mimicCoef0', Sdf.ValueTypeNames.Float).Set(c['offset'])
        prim.CreateAttribute('newton:mimicCoef1', Sdf.ValueTypeNames.Float).Set(c['multiplier'])
        prim.CreateRelationship('newton:mimicJoint').SetTargets([joints[c['master']].GetPath()])
    return excluded


class SharpaPositionAdapter(SourcePositionAdapter):
    """One fixed PD implementation; zero actions hold the original position targets."""
    def __init__(self, hand, contract, parameters):
        contract = finger_action_contract(contract)
        super().__init__(hand, contract)
        self.motor_damping = torch.tensor([parameters['damping'][a['joint']] for a in contract['actuators']],
                                          device=self.device, dtype=torch.float32)

    def apply(self):
        self.elapsed += 1
        alpha = min(1., self.elapsed / self.steps_per_control)
        self.commands = self.old + alpha * (self.next - self.old)
        q = self.hand.data.joint_pos.torch
        v = self.hand.data.joint_vel.torch
        effort = torch.zeros_like(v)
        # Same external PD -> actuator clipping organization as Sharpa.
        motor = self.kp * (self.commands - q[:, self.ids]) - self.motor_damping * v[:, self.ids]
        effort[:, self.ids] = torch.clamp(motor, min=self.force_limits[:, 0], max=self.force_limits[:, 1])
        self.hand.set_joint_effort_target(effort)
        self.hand.write_data_to_sim()
        return effort
