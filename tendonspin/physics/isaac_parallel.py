# References: Sharpa RL Lab 5accf024d376685eaa17da7aa4614498217eab4d
# sharpa_wave_env.py::_setup_scene/reset and Isaac Lab3 InteractiveScene (BSD-3-Clause).
# Reuse: TendonSpin Boya v3 dynamics and v4 16-input adapter. New batched scene,
# explicit source excludes in a derived USD layer, no original asset mutations.
"""Batched original-size Boya hands with independent environments and 16 inputs."""
from pathlib import Path
import json
import numpy as np
import torch
import warp as wp
from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from tendonspin.interfaces import finger_action_contract, ACTION_NAMES
from .coordinates import wxyz_to_xyzw
from .isaac_boya_sharpa import motor_parameters, actuator_configuration, configure_constraints, SharpaPositionAdapter


def tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, 'torch'):
        return value.torch
    return wp.to_torch(value)


def axis_z(quat):
    q = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    x,y,z,w = q.unbind(-1)
    return torch.stack((2*(x*z+w*y),2*(y*z-w*x),1-2*(x*x+y*y)),dim=-1)


class BoyaParallel:
    def __init__(self, root, out, num_envs=64):
        self.root, self.out = Path(root), Path(out)
        self.contract = finger_action_contract(json.loads((self.root/'docs/data/boya_native_contract.json').read_text()))
        c = self.contract
        self.num_envs = num_envs
        self.dt = c['physics_dt']
        self.device = 'cuda:0'
        imported = json.loads((self.root/'docs/data/isaac_boya_import_phases.json').read_text())
        source = self.root/imported['usd']['path']
        original = Usd.Stage.Open(str(source))
        self.asset_path = self.out/'boya_parallel.usda'
        stage = Usd.Stage.CreateNew(str(self.asset_path))
        stage.GetRootLayer().subLayerPaths = [str(source)]
        default_path = str(original.GetDefaultPrim().GetPath())
        stage.SetDefaultPrim(stage.GetPrimAtPath(default_path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        self.excludes = configure_constraints(stage,self.root,c,hand_prefix=default_path)
        paths = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self.filter_names = [p.rsplit('/',1)[-1] for p in paths]
        filters = ['{ENV_REGEX_NS}/Hand'+p[len(default_path):] for p in paths]
        stage.GetRootLayer().Save()
        del stage, original
        self.parameters = motor_parameters(c)
        self.cfg = sim_utils.SimulationCfg(dt=self.dt,device=self.device,gravity=tuple(c['gravity']),
            visualizer_cfgs=[],use_newton_actuators=False,
            physics=PhysxCfg(solver_type=1,min_position_iteration_count=16,min_velocity_iteration_count=4,
                             enable_ccd=False))
        self.sim = sim_utils.SimulationContext(self.cfg)
        material = sim_utils.RigidBodyMaterialCfg(static_friction=.5,dynamic_friction=.5,restitution=0.)
        collision = sim_utils.CollisionPropertiesCfg(contact_offset=.0001,rest_offset=0.)
        hand_cfg = ArticulationCfg(prim_path='{ENV_REGEX_NS}/Hand',
            spawn=sim_utils.UsdFileCfg(usd_path=str(self.asset_path),activate_contact_sensors=True,
                collision_props=collision,physics_material=material,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                    solver_position_iteration_count=16,solver_velocity_iteration_count=4)),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(c['hand_root']['pos']),
                rot=tuple(wxyz_to_xyzw(c['hand_root']['quat'])),
                joint_pos={j['name']:j['qpos'] for j in c['joints']},
                joint_vel={j['name']:j['qvel'] for j in c['joints']}),
            actuators=actuator_configuration(self.parameters))
        o = c['object']
        object_cfg = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Cylinder',
            spawn=sim_utils.CylinderCfg(radius=o['diameter_m']/2,height=o['length_m'],
                mass_props=sim_utils.MassPropertiesCfg(mass=o['mass_kg']),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,enable_gyroscopic_forces=True),
                collision_props=collision,physics_material=material,activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(o['pos']),rot=tuple(wxyz_to_xyzw(o['quat'])),
                lin_vel=tuple(o['lin_vel_world']),ang_vel=tuple(o['ang_vel_world'])))
        scene_cfg = InteractiveSceneCfg(num_envs=num_envs,env_spacing=1.,replicate_physics=True,filter_collisions=True)
        scene_cfg.hand = hand_cfg
        scene_cfg.cylinder = object_cfg
        scene_cfg.contact = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Cylinder',update_period=0.,
            filter_prim_paths_expr=filters,max_contact_data_count_per_prim=64,
            track_contact_points=False,track_friction_forces=True)
        self.scene = InteractiveScene(scene_cfg)
        self.hand = self.scene.articulations['hand']
        self.object = self.scene.rigid_objects['cylinder']
        self.contact = self.scene.sensors['contact']
        self.sim.reset()
        self.origins = tensor(self.scene.env_origins)
        self.adapter = SharpaPositionAdapter(self.hand,c,self.parameters)
        self.names = list(self.hand.joint_names)
        self.active_joint_ids = torch.tensor([self.names.index(n) for n in ACTION_NAMES],device=self.device)
        self.slaves = torch.tensor([self.names.index(p['slave']) for p in c['couplings']],device=self.device)
        self.masters = torch.tensor([self.names.index(p['master']) for p in c['couplings']],device=self.device)
        self.q0 = tensor(self.hand.data.default_joint_pos).clone()
        self.v0 = tensor(self.hand.data.default_joint_vel).clone()
        self.commands0 = self.adapter.commands.clone()
        self.object0 = tensor(self.object.data.default_root_state).clone()
        self.object0[:,:3] += self.origins
        self.center = self.object0[:,:3].clone()
        self.axis0 = axis_z(self.object0[:,3:7])
        self.motor_bias = self.commands0-self.q0[:,self.adapter.ids]
        group_ids = [next((i for i,f in enumerate(('th','ff','mf','rf','lf')) if n.startswith(f)),5)
                     for n in self.filter_names]
        self.groups = torch.nn.functional.one_hot(torch.tensor(group_ids,device=self.device),6).float()
        self.reset_nominal()

    def _write_state(self,q,v,object_state,commands,env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs,device=self.device)
        hand_root = tensor(self.hand.data.default_root_state)[env_ids].clone()
        hand_root[:,:3] += self.origins[env_ids]
        self.hand.write_root_pose_to_sim(hand_root[:,:7],env_ids=env_ids)
        self.hand.write_joint_state_to_sim(q,v,env_ids=env_ids)
        self.object.write_root_state_to_sim(object_state,env_ids=env_ids)
        self.adapter.commands[env_ids] = commands
        self.adapter.old[env_ids] = commands
        self.adapter.next[env_ids] = commands
        self.hand.set_joint_effort_target(torch.zeros_like(tensor(self.hand.data.joint_vel)))
        self.hand.write_data_to_sim()
        self.scene.reset(env_ids)
        self.sim.forward()
        self.scene.update(self.dt)
        self.contact.update(self.dt,force_recompute=True)

    def reset_nominal(self):
        self._write_state(self.q0,self.v0,self.object0,self.commands0)

    def sample_grasps(self,generator,noise=.25):
        q = self.q0.clone()
        perturb = (2*torch.rand((self.num_envs,16),device=self.device,generator=generator)-1)*noise
        perturb[0] = 0.  # Declared nominal anchor; excluded from random cache counts.
        active = self.adapter.active
        q[:,self.active_joint_ids] = torch.clamp(q[:,self.active_joint_ids]+perturb,
                   min=self.adapter.control_limits[active,0],max=self.adapter.control_limits[active,1])
        q[:,self.slaves] = q[:,self.masters]
        q[0] = self.q0[0]
        commands = q[:,self.adapter.ids]+self.motor_bias
        commands = torch.clamp(commands,min=self.adapter.control_limits[:,0],max=self.adapter.control_limits[:,1])
        commands[0] = self.commands0[0]
        self._write_state(q,torch.zeros_like(q),self.object0.clone(),commands)
        return dict(q=q.clone(),object_state=self.object0.clone(),commands=commands.clone(),perturb_rad=perturb)

    def advance(self):
        before_q = tensor(self.hand.data.joint_pos).clone()
        before_v = tensor(self.hand.data.joint_vel).clone()
        effort = self.adapter.apply()
        self.object.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.dt)
        self.contact.update(self.dt,force_recompute=True)
        return before_q,before_v,effort

    def measure(self):
        q = tensor(self.hand.data.joint_pos)
        v = tensor(self.hand.data.joint_vel)
        obj = tensor(self.object.data.root_state_w)
        force = tensor(self.contact.data.normal_force_matrix_w)[:,0]
        friction = tensor(self.contact.data.friction_force_matrix_w)[:,0]
        drift = torch.linalg.vector_norm(obj[:,:3]-self.center,dim=-1)*1000
        tilt = torch.rad2deg(torch.acos((axis_z(obj[:,3:7])*self.axis0).sum(-1).clamp(-1,1)))
        normal = torch.linalg.vector_norm(force,dim=-1)
        support = ((normal@self.groups)>1e-6).sum(-1)
        coupling = (q[:,self.slaves]-q[:,self.masters]).abs().amax(-1)
        finite = torch.isfinite(torch.cat((q,v,obj,force.flatten(1),friction.flatten(1)),dim=-1)).all(-1)
        return dict(joint_pos=q,joint_vel=v,object_state=obj,body_pose=tensor(self.hand.data.body_link_pose_w),
            normal_force_matrix_w=force,friction_force_matrix_w=friction,drift_mm=drift,tilt_deg=tilt,
            max_speed=v.abs().amax(-1),max_normal=normal.amax(-1),coupling_error=coupling,
            support_groups=support,finite=finite)
