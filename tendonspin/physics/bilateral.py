"""Two articulated hands and two objects in ONE Isaac physics environment.

Only a rigid Z rotation/translation positions the hands for the camera. Both
controllers' forces are applied before the single shared physics step.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab_physx.physics import PhysxCfg

from tendonspin.interfaces import ACTION_NAMES, finger_action_contract
from .coordinates import wxyz_to_xyzw
from .hand_profiles import resolve_hand_profile
from .isaac_boya_sharpa import motor_parameters, actuator_configuration, configure_constraints, SharpaPositionAdapter
from .isaac_parallel import BoyaParallel, tensor, axis_z
from tendonspin.rl.termination import TaskTermination, make_termination_spec


class BilateralPhysics:
    def __init__(self, root, out, fixed, case, geometry, logical_scale, separation=.24):
        self.device = 'cuda:0'
        self.dt = .0005
        self.root, self.out = Path(root), Path(out)
        self.sides = {}
        self.layout = {}
        self.geometry = dict(geometry)
        ratio = float(fixed['scale'][case])/logical_scale
        self.geometry['diameter_m'] *= ratio
        self.geometry['height_m'] *= ratio
        self.geometry['dimensions_m'] = [self.geometry['diameter_m']]*2+[self.geometry['height_m']]
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.dt, device=self.device,
            gravity=(0.,0.,-9.81), visualizer_cfgs=[], use_newton_actuators=False,
            physics=PhysxCfg(solver_type=1, min_position_iteration_count=16,
                min_velocity_iteration_count=4, max_velocity_iteration_count=255, enable_ccd=False)))
        cfg = InteractiveSceneCfg(num_envs=1, env_spacing=1., replicate_physics=False, filter_collisions=False)
        # fixed['object_state'] has already had the source environment origin removed.
        for side, sign in (('left', -1), ('right', 1)):
            profile = resolve_hand_profile(root, side)
            c = finger_action_contract(profile.adapt_contract(json.loads(profile.contract_path.read_text())))
            parameters = motor_parameters(c)
            asset = self.out/profile.parallel_asset_name
            layer = Usd.Stage.CreateNew(str(asset))
            layer.GetRootLayer().subLayerPaths = [str(profile.asset_path)]
            layer.SetDefaultPrim(layer.GetPrimAtPath(profile.source_prim))
            UsdGeom.SetStageMetersPerUnit(layer, 1.)
            UsdGeom.SetStageUpAxis(layer, 'Z')
            excludes = configure_constraints(layer, root, c, hand_prefix=profile.source_prim,
                contact_scene=profile.contact_scene, body_prefix=profile.body_prefix)
            paths = [str(p.GetPath()) for p in layer.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            layer.GetRootLayer().Save()
            local_obj = profile.adapt_object_state(fixed['object_state'][case].copy())
            pivot = local_obj[:3].copy()
            target = np.array([sign*separation/2, 0., pivot[2]])
            yaw = Rotation.from_euler('z', 180. if side == 'left' else 0., degrees=True)
            self.layout[side] = (pivot, target, yaw)
            root_pos = target+yaw.apply(np.asarray(c['hand_root']['pos'])-pivot)
            root_quat = (yaw*Rotation.from_quat(wxyz_to_xyzw(c['hand_root']['quat']))).as_quat()
            local_obj[:3] = target
            local_obj[3:7] = (yaw*Rotation.from_quat(local_obj[3:7])).as_quat()
            local_obj[7:10] = yaw.apply(local_obj[7:10])
            local_obj[10:13] = yaw.apply(local_obj[10:13])
            material = sim_utils.RigidBodyMaterialCfg(static_friction=float(fixed['friction'][case]),
                dynamic_friction=float(fixed['friction'][case]), restitution=0.)
            collision = sim_utils.CollisionPropertiesCfg(contact_offset=.0001, rest_offset=0.)
            hand_path = '{ENV_REGEX_NS}/'+side.title()+'Hand'
            object_path = '{ENV_REGEX_NS}/'+side.title()+'Object'
            setattr(cfg, side+'_hand', ArticulationCfg(prim_path=hand_path,
                spawn=sim_utils.UsdFileCfg(usd_path=str(asset), activate_contact_sensors=True,
                    collision_props=collision, physics_material=material,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                        solver_position_iteration_count=16, solver_velocity_iteration_count=4)),
                init_state=ArticulationCfg.InitialStateCfg(pos=tuple(root_pos), rot=tuple(root_quat),
                    joint_pos={profile.runtime_joint_name(j['name']): j['qpos'] for j in c['joints']},
                    joint_vel={profile.runtime_joint_name(j['name']): 0. for j in c['joints']}),
                actuators=actuator_configuration(parameters, joint_prefix=profile.joint_prefix)))
            setattr(cfg, side+'_object', RigidObjectCfg(prim_path=object_path,
                spawn=sim_utils.CylinderCfg(radius=self.geometry['diameter_m']/2, height=self.geometry['height_m'],
                    mass_props=sim_utils.MassPropertiesCfg(mass=geometry['mass_kg']),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, enable_gyroscopic_forces=True),
                    collision_props=collision, physics_material=material, activate_contact_sensors=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(local_obj[:3]), rot=tuple(local_obj[3:7]))))
            setattr(cfg, side+'_contact', ContactSensorCfg(prim_path=object_path, update_period=0.,
                filter_prim_paths_expr=[hand_path+path[len(profile.source_prim):] for path in paths],
                max_contact_data_count_per_prim=64, track_contact_points=False, track_friction_forces=True))
            p = SimpleNamespace(hand_profile=profile, contract=c, parameters=parameters, num_envs=1,
                device=self.device, dt=self.dt, sim=self.sim,
                filter_names=[profile.canonical_body_name(path.rsplit('/',1)[-1]) for path in paths],
                initial_object=local_obj, excludes=excludes)
            self.sides[side] = p
        self.scene = InteractiveScene(cfg)
        self.case, self.fixed = case, fixed
        self.configuration = dict(num_envs=1, physics_scenes=1, articulations=2, objects=2,
            filter_collisions=False, object_separation_m=separation, physics_dt_s=self.dt,
            controller_hz=20, shared_step=True, independently_inferred_actions=True)

    def initialize(self):
        """Call after the camera is authored, so one reset initializes everything."""
        self.sim.reset()
        self.property_readback = {}
        for side, p in self.sides.items():
            p.scene = self.scene
            p.hand = self.scene.articulations[side+'_hand']
            p.object = self.scene.rigid_objects[side+'_object']
            p.contact = self.scene.sensors[side+'_contact']
            p.names = list(p.hand.joint_names)
            bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
            prims = {prim.GetName(): prim for prim in self.sim.stage.Traverse()
                if str(prim.GetPath()).startswith(f'/World/envs/env_0/{side.title()}Hand/')
                and prim.HasAPI(UsdPhysics.RigidBodyAPI)}
            corners = []
            for name in p.hand.body_names:
                bounds = bbox.ComputeUntransformedBound(prims[name]).ComputeAlignedRange()
                lo, hi = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
                if not np.isfinite([lo, hi]).all() or np.any(hi < lo):
                    raise ValueError('Missing mesh bounds: '+name)
                corners.append([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
            p.bounds_corners = torch.tensor(corners,device=self.device,dtype=torch.float32)
            p.joint_id = lambda name, p=p: p.names.index(p.hand_profile.runtime_joint_name(name))
            p.adapter = SharpaPositionAdapter(p.hand, p.contract, p.parameters, joint_prefix=p.hand_profile.joint_prefix)
            p.active_joint_ids = torch.tensor([p.joint_id(n) for n in ACTION_NAMES], device=self.device)
            p.slaves = torch.tensor([p.joint_id(c['slave']) for c in p.contract['couplings']], device=self.device)
            p.masters = torch.tensor([p.joint_id(c['master']) for c in p.contract['couplings']], device=self.device)
            group_ids = [next((i for i,f in enumerate(('th','ff','mf','rf','lf')) if n.startswith(f)),5) for n in p.filter_names]
            p.groups = torch.nn.functional.one_hot(torch.tensor(group_ids,device=self.device),6).float()
            p.origins = torch.zeros((1,3),device=self.device)
            q = torch.as_tensor(self.fixed['joint_pos'][self.case], device=self.device)[None]
            commands = torch.as_tensor(self.fixed['commands'][self.case],device=self.device)[None]
            obj = torch.as_tensor(p.initial_object, dtype=torch.float32,device=self.device)[None]
            p.hand.write_root_pose_to_sim(tensor(p.hand.data.default_root_state)[:,:7].clone())
            p.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
            p.object.write_root_state_to_sim(obj)
            p.adapter.commands = commands.clone()
            p.adapter.old = commands.clone()
            p.adapter.next = commands.clone()
            active = p.adapter.active
            p.adapter.kp[active] *= torch.as_tensor(self.fixed['p_ratio'][self.case],device=self.device)
            p.adapter.motor_damping[active] *= torch.as_tensor(self.fixed['d_ratio'][self.case],device=self.device)
            mass = torch.tensor([[self.fixed['mass_kg'][self.case]]],device=self.device,dtype=torch.float32)
            ids = torch.zeros(1,device=self.device,dtype=torch.long)
            p.object.set_masses_index(masses=mass,env_ids=ids)
            com = p.hand_profile.adapt_policy_position(self.fixed['com_m'][self.case])
            coms = tensor(p.object.data.body_com_pose_b).clone()
            coms[..., :3] = torch.as_tensor(com,device=self.device)
            p.object.set_coms_index(coms=coms,env_ids=ids)
            errors = dict(mass=float((tensor(p.object.root_view.get_masses()).to(self.device)-mass).abs().max()),
                com=float((tensor(p.object.root_view.get_coms()).to(self.device).reshape(1,1,7)[...,:3]-coms[...,:3]).abs().max()))
            for name, asset in (('hand',p.hand),('object',p.object)):
                mu = tensor(asset.root_view.get_material_properties())[...,:2]
                errors[name+'_friction'] = float((mu-float(self.fixed['friction'][self.case])).abs().max())
            if max(errors.values()) > 1e-6:
                raise RuntimeError('Property mismatch: '+str(errors))
            self.property_readback[side] = errors
            p.center = obj[:,:3].clone()
            p.axis0 = axis_z(obj[:,3:7])
            p.hand.set_joint_effort_target(torch.zeros_like(tensor(p.hand.data.joint_vel)))
            p.hand.write_data_to_sim()
            # The proprioceptive policy has no tactile input. Avoid forcing
            # unused contact/tilt/support reductions at every 2 kHz step.
            p.measure = lambda p=p: self.measure_side(p)
            p.termination = TaskTermination(make_termination_spec(self.root, 'boya_workspace',
                p.contract['object']['pos'][2]), mirror_world_y=p.hand_profile.mirror_world_y)
        self.scene.reset()
        self.sim.forward()
        self.scene.update(self.dt)
        self.physics_steps = 0

    @staticmethod
    def measure_side(p):
        q = tensor(p.hand.data.joint_pos)
        v = tensor(p.hand.data.joint_vel)
        obj = tensor(p.object.data.root_state_w)
        return dict(joint_pos=q, joint_vel=v, object_state=obj,
            body_pose=tensor(p.hand.data.body_link_pose_w),
            finite=torch.isfinite(torch.cat((q, v, obj), dim=-1)).all(-1),
            max_speed=v.abs().amax(-1),
            coupling_error=(q[:, p.slaves]-q[:, p.masters]).abs().amax(-1))

    def cross_clearance(self, measures):
        """Conservative separating-plane proof including all moving mesh bounds.

        A positive gap separates the entire left hand/object from the entire
        right hand/object, so neither cross-hand nor cross-object collision is
        possible. Intra-hand grasp contacts are intentionally retained.
        """
        extents = {}
        radius = float(np.hypot(self.geometry['diameter_m']/2, self.geometry['height_m']/2))
        for side, p in self.sides.items():
            poses = measures[side]['body_pose'][0]
            q = poses[:, None, 3:7]
            xyz = q[..., :3].expand_as(p.bounds_corners)
            points = p.bounds_corners
            rotated = points+2*torch.cross(xyz, torch.cross(xyz,points,dim=-1)+q[...,3:]*points, dim=-1)
            world_x = rotated[...,0]+poses[:,None,0]
            object_x = measures[side]['object_state'][0,0]
            extents[side] = (torch.minimum(world_x.min(),object_x-radius),
                             torch.maximum(world_x.max(),object_x+radius))
        return float(extents['right'][0]-extents['left'][1])

    def canonical_state(self, side, state):
        pivot, target, yaw = self.layout[side]
        result = state.clone()
        rotation = torch.as_tensor(yaw.as_matrix(), device=self.device, dtype=state.dtype)
        # inverse of row-vector rigid layout; orientation is irrelevant to bounds.
        result[:,:3] = (state[:,:3]-state.new_tensor(target))@rotation+state.new_tensor(pivot)
        return result

    def world_force(self, side, canonical_force):
        _, _, yaw = self.layout[side]
        force = self.sides[side].hand_profile.adapt_policy_position(canonical_force)
        rotation = torch.as_tensor(yaw.as_matrix(), device=self.device, dtype=force.dtype)
        return force@rotation.T

    def step(self, canonical_force):
        for side, p in self.sides.items():
            p.adapter.apply()
            force = self.world_force(side, canonical_force)
            p.object.instantaneous_wrench_composer.set_forces_and_torques_index(
                forces=force[:,None,:], torques=torch.zeros_like(force[:,None,:]), is_global=True)
            p.object.write_data_to_sim()
        self.sim.step(render=False)  # Exactly ONE integration step for BOTH hands.
        self.scene.update(self.dt)
        self.physics_steps += 1
