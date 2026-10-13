# Derived from TendonSpin isaac_parallel.py at8cca30e (MIT); same Boya
# CAD/solver/actuator/timing. Only scene replication is disabled, as required by
# Isaac Lab3 randomize_rigid_body_scale for per-environment geometry.
# Existing driver and historical checkpoint source identity remain unchanged.
# References: Sharpa RL Lab 5accf024d376685eaa17da7aa4614498217eab4d
# sharpa_wave_env.py::_setup_scene/reset and Isaac Lab3 InteractiveScene (BSD-3-Clause).
# Reuse: TendonSpin Boya v3 dynamics and v4 16-input adapter. New batched scene,
# explicit source excludes in a derived USD layer, no original asset mutations.
# Optional solver profile reuses the declared effective-zero diagnostic86cf44f;
# default original16/4 dynamics are retained, and0 is separately recorded.
"""Boya hands with independently parsed sized objects and the same16 inputs."""
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
from .isaac_parallel import BoyaParallel
from .isaac_boya_sharpa import motor_parameters, actuator_configuration, configure_constraints, SharpaPositionAdapter
from .solver_profiles import ORIGINAL, VELOCITY_ZERO, configuration, author_zero_velocity, audit_zero_velocity


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


class BoyaRandomizedParallel(BoyaParallel):
    def __init__(self, root, out, num_envs=64, scene_setup=None, engine_profile=ORIGINAL):
        self.root, self.out = Path(root), Path(out)
        self.engine_configuration = configuration(engine_profile)
        zero_velocity = engine_profile == VELOCITY_ZERO
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
            physics=PhysxCfg(solver_type=1,min_position_iteration_count=16,
                             min_velocity_iteration_count=0 if zero_velocity else 4,
                             max_velocity_iteration_count=0 if zero_velocity else 255,
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
        scene_cfg = InteractiveSceneCfg(num_envs=num_envs,env_spacing=1.,replicate_physics=False,filter_collisions=True)
        scene_cfg.hand = hand_cfg
        scene_cfg.cylinder = object_cfg
        scene_cfg.contact = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Cylinder',update_period=0.,
            filter_prim_paths_expr=filters,max_contact_data_count_per_prim=64,
            track_contact_points=False,track_friction_forces=True)
        self.scene = InteractiveScene(scene_cfg)
        self.hand = self.scene.articulations['hand']
        self.object = self.scene.rigid_objects['cylinder']
        self.contact = self.scene.sensors['contact']
        if zero_velocity:author_zero_velocity(self.sim.stage)
        if scene_setup is not None:scene_setup(self)
        if zero_velocity:
            before = audit_zero_velocity(self.sim.stage,self.cfg.physics_prim_path,num_envs)
        self.sim.reset()
        if zero_velocity:
            after = audit_zero_velocity(self.sim.stage,self.cfg.physics_prim_path,num_envs)
            if any(before[k]!=after[k] for k in before if k!='scene_physics_attributes'):
                raise RuntimeError('Actor solver requests changed during PhysX initialization')
            self.engine_audit = dict(configuration=self.engine_configuration,before_reset=before,after_reset=after)
            (self.out/'solver_configuration.json').write_text(json.dumps(self.engine_audit,indent=2,allow_nan=False)+'\n')
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

    def advance(self):
        if hasattr(self,'domain_randomization'):
            dr=self.domain_randomization
            dr.before_physics_step()
            if dr.enabled and dr.first_force_readback is None:
                # One startup read before integration, no additional step.
                dr.verify_forwarded_force()
        return super().advance()

    def measure(self):
        values=super().measure()
        if hasattr(self,'domain_randomization'):
            dr=self.domain_randomization
            values.update(external_force_requested_w=dr.force,object_mass_kg=dr.mass,
                object_com_b_m=dr.com,object_friction=dr.friction,object_logical_scale=dr.scale,
                p_gain_ratio=dr.p_ratio,d_gain_ratio=dr.d_ratio)
        return values

