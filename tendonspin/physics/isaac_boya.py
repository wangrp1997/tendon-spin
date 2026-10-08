# Source: original Boya CAD/position actuator contract from botyard-inhand
# 204d197a9606fb7266e884f3b2e6110195be01cb; NVIDIA Isaac Lab3 API (BSD-3-Clause).
# New declared engine adapter. Does not implement or claim Hora/AnyRotate/Sharpa.
"""Build a single Boya scene and apply the original motor-position contract."""
from pathlib import Path

from tendonspin.physics.coordinates import wxyz_to_xyzw
from tendonspin.interfaces import action_motor_indices

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab_physx.physics import PhysxCfg


def make_scene(root, contract, usd_path, *, actuator_cfg=None, before_reset=None):
    """Call only after AppLauncher. This separately declared engine is PhysX."""
    cfg=sim_utils.SimulationCfg(dt=contract['physics_dt'],device='cuda:0',
        gravity=tuple(contract['gravity']),visualizer_cfgs=[],use_newton_actuators=False,
        physics=PhysxCfg(solver_type=1,min_position_iteration_count=16,
            min_velocity_iteration_count=4,enable_ccd=True))
    sim=sim_utils.SimulationContext(cfg)
    material=sim_utils.RigidBodyMaterialCfg(static_friction=.5,dynamic_friction=.5,restitution=0.)
    collision=sim_utils.CollisionPropertiesCfg(contact_offset=.0001,rest_offset=0.)
    joints={j['name']:j['qpos'] for j in contract['joints']}
    velocities={j['name']:j['qvel'] for j in contract['joints']}
    base=contract['hand_root']
    hand=Articulation(ArticulationCfg(prim_path='/World/Hand',
        spawn=sim_utils.UsdFileCfg(usd_path=str(usd_path),activate_contact_sensors=True,
            collision_props=collision,physics_material=material,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,solver_position_iteration_count=16,
                solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(base['pos']),rot=tuple(wxyz_to_xyzw(base['quat'])),
            joint_pos=joints,joint_vel=velocities),
        # Zero-gain forwarding actuator: all efforts are derived by the source
        # position motor adapter below; no independent passive-joint actions.
        actuators=actuator_cfg if actuator_cfg is not None else {'source_effort_forwarder':IdealPDActuatorCfg(joint_names_expr=['.*'],
            stiffness=0.,damping=0.,effort_limit=1000000.,effort_limit_sim=1000000.,
            velocity_limit_sim=100.,
            viscous_friction={j['name']:j['damping'] for j in contract['joints']})}))
    obj=contract['object']
    cylinder=RigidObject(RigidObjectCfg(prim_path='/World/Cylinder',
        spawn=sim_utils.CylinderCfg(radius=obj['diameter_m']/2,height=obj['length_m'],
            mass_props=sim_utils.MassPropertiesCfg(mass=obj['mass_kg']),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,
                enable_gyroscopic_forces=True),collision_props=collision,physics_material=material,
            activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(obj['pos']),rot=tuple(wxyz_to_xyzw(obj['quat'])),
            lin_vel=tuple(obj['lin_vel_world']),ang_vel=tuple(obj['ang_vel_world']))))
    # Imported hierarchy is nested. Only genuine rigid bodies may be GPU filters;
    # fixed tip/central geometry belongs to its parent rigid body.
    from pxr import UsdPhysics
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    filters=[str(p.GetPath()) for p in stage.Traverse()
             if str(p.GetPath()).startswith('/World/Hand/')
             and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not filters:
        raise RuntimeError('No rigid-body contact filters resolved')
    sensor=ContactSensor(ContactSensorCfg(prim_path='/World/Cylinder',update_period=0.,
        filter_prim_paths_expr=filters,max_contact_data_count_per_prim=64,
        track_contact_points=True,track_friction_forces=True))
    if before_reset is not None:
        before_reset(stage, hand, cylinder)
    sim.reset()
    runtime_paths=list(hand.root_view.link_paths[0])
    if set(filters)!=set(runtime_paths):
        raise RuntimeError('Contact paths do not match runtime articulation links')
    # Solver reset may advance internal setup; explicitly restore the declared
    # physical q/velocity initial condition before the logged diagnostic prefix.
    hand.write_root_pose_to_sim(hand.data.default_root_state[:,:7].clone())
    hand.write_joint_state_to_sim(hand.data.default_joint_pos.clone(),hand.data.default_joint_vel.clone())
    cylinder.write_root_state_to_sim(cylinder.data.default_root_state.clone())
    hand.reset();cylinder.reset();sensor.reset()
    sim.forward()  # FK only: no new physics step after state restoration.
    hand.update(cfg.dt);cylinder.update(cfg.dt);sensor.update(cfg.dt,force_recompute=True)
    return sim,hand,cylinder,sensor,filters,cfg


class SourcePositionAdapter:
    """Named finger position inputs,18 source motors,4 passive solver-damped joints.

    Motor effort is clipped independently. The same passive damping coefficients
    are assigned as solver-side viscous friction, avoiding explicit -D*qdot
    injection on tiny inertias. Solver/contact/coupling physics remain distinct.
    """
    def __init__(self,hand,contract):
        self.hand=hand;self.contract=contract;self.device=hand.device
        names=list(hand.joint_names)
        self.ids=torch.tensor([names.index(a['joint']) for a in contract['actuators']],device=self.device)
        self.num_envs=hand.num_instances
        self.commands=torch.tensor([a['initial_ctrl'] for a in contract['actuators']],device=self.device,dtype=torch.float32)[None].repeat(self.num_envs,1)
        self.kp=torch.tensor([a['gain'] for a in contract['actuators']],device=self.device,dtype=torch.float32)
        self.force_limits=torch.tensor([a['force_range'] for a in contract['actuators']],device=self.device,dtype=torch.float32)
        self.control_limits=torch.tensor([a['control_range'] for a in contract['actuators']],device=self.device,dtype=torch.float32)
        self.action_names=tuple(contract['action_names'])
        self.active=torch.tensor(action_motor_indices(contract,self.action_names),device=self.device)
        by_name={j['name']:j for j in contract['joints']}
        self.damping=torch.tensor([by_name[n]['damping'] for n in names],device=self.device,dtype=torch.float32)
        self.action_dim=len(self.active)
        assert self.action_dim in (13,16) and len(names)==22
        self.old=self.commands.clone();self.next=self.commands.clone()
        self.physics_dt=contract['physics_dt'];self.control_dt=contract['control_dt']
        self.steps_per_control=round(self.control_dt/self.physics_dt)
        self.elapsed=0

    def set_action(self,action):
        action=torch.as_tensor(action,device=self.device,dtype=torch.float32)
        if action.shape!=(self.num_envs,self.action_dim) or not torch.isfinite(action).all() or (action.abs()>1.).any():
            raise ValueError(f'Expected ({self.num_envs},{self.action_dim}) bounded position inputs')
        self.old=self.commands.clone();self.next=self.old.clone();self.elapsed=0
        self.next[:,self.active]=torch.clamp(self.old[:,self.active]+action*.35*self.control_dt,
            min=self.control_limits[self.active,0],max=self.control_limits[self.active,1])

    def apply(self):
        self.elapsed+=1
        alpha=min(1.,self.elapsed/self.steps_per_control)
        self.commands=self.old+alpha*(self.next-self.old)
        effort=torch.zeros_like(self.hand.data.joint_vel.torch)
        motor=torch.clamp(self.kp*(self.commands-self.hand.data.joint_pos[:,self.ids]),
            min=self.force_limits[:,0],max=self.force_limits[:,1])
        effort[:,self.ids]+=motor
        self.hand.set_joint_effort_target(effort)
        self.hand.write_data_to_sim()
        return effort
