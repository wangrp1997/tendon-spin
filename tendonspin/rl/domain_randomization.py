# Reference: HaozhiQi/hora v0.0.1 AllegroHandHora._create_envs/reset_idx/
# pre_physics_step/_update_priv_buf and scripts/train_s1.sh (MIT),
# https://github.com/HaozhiQi/hora; arXiv:2210.04887.
# Isaac port: installed Isaac Lab3 PhysX mass/COM/material/wrench interfaces
# (BSD-3-Clause). Boya gain ratios, independent collision parsing and fixed
# nominal inertia after mass changes are explicitly recorded port choices.
"""Hora teacher randomization mechanisms with actual Boya property readback."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch


def load_configuration(path):
    cfg=json.loads(Path(path).read_text())
    if cfg['schema']!='tendonspin-hora-domain-randomization-v1':raise ValueError('Unknown DR schema')
    if cfg['replicate_physics'] or len(cfg['scale_centers'])!=9:raise ValueError('Independent nine-size parsing required')
    return cfg


def static_samples(cfg,num_envs,enabled=True):
    rng=np.random.default_rng(cfg['static_seed'])
    if enabled:
        groups=np.arange(num_envs)%len(cfg['scale_centers'])
        scale=np.asarray(cfg['scale_centers'])[groups]+rng.uniform(-cfg['scale_jitter'],cfg['scale_jitter'],num_envs)
        friction=rng.uniform(*cfg['friction'],num_envs)
        com=rng.uniform(*cfg['com_m'],(num_envs,3))
    else:
        groups=np.full(num_envs,cfg['scale_centers'].index(cfg['logical_nominal_scale']),dtype=int)
        scale=np.full(num_envs,cfg['logical_nominal_scale']);friction=np.full(num_envs,.5);com=np.zeros((num_envs,3))
    return dict(scale_id=groups,scale=scale,friction=friction,com=com)


def configure_sizes(p,geometry,cfg,samples):
    from pxr import UsdGeom,UsdPhysics
    from tendonspin.physics.object_geometry import author_primitive
    if p.sim.is_playing():raise RuntimeError('Size changes require pre-startup authoring')
    for i,scale in enumerate(samples['scale']):
        geo=deepcopy(geometry);ratio=float(scale/cfg['logical_nominal_scale'])
        geo['diameter_m']*=ratio;geo['height_m']*=ratio
        geo['dimensions_m']=[geo['diameter_m']]*2+[geo['height_m']]
        body=f'/World/envs/env_{i}/Cylinder';prim=author_primitive(p.sim.stage,body+'/geometry/mesh',geo)
        if not prim.HasAPI(UsdPhysics.CollisionAPI):raise RuntimeError('Sized collider API lost')
        UsdPhysics.MassAPI(p.sim.stage.GetPrimAtPath(body)).CreateMassAttr(geometry['mass_kg'])
    p.object_geometry=dict(nominal=deepcopy(geometry),logical_scale=samples['scale'].tolist(),replicate_physics=False)


class HoraDomainRandomization:
    def __init__(self,p,cfg,samples,enabled=True):
        import warp as wp
        from tendonspin.physics.isaac_parallel import tensor
        self.p=p;self.cfg=cfg;self.enabled=enabled;self.wp=wp;self.tensor=tensor
        self.scale=torch.tensor(samples['scale'],device=p.device,dtype=torch.float32)
        self.scale_id=torch.tensor(samples['scale_id'],device=p.device,dtype=torch.long)
        self.friction=torch.tensor(samples['friction'],device=p.device,dtype=torch.float32)
        self.com=torch.tensor(samples['com'],device=p.device,dtype=torch.float32)
        self.mass=tensor(p.object.data.body_mass).reshape(p.num_envs).clone()
        self.nominal_inertia=tensor(p.object.data.body_inertia).clone()
        self.base_kp=p.adapter.kp.clone();self.base_damping=p.adapter.motor_damping.clone()
        p.adapter.kp=self.base_kp[None].repeat(p.num_envs,1)
        p.adapter.motor_damping=self.base_damping[None].repeat(p.num_envs,1)
        self.force=torch.zeros((p.num_envs,3),device=p.device)
        self.p_ratio=torch.ones((p.num_envs,16),device=p.device)
        self.d_ratio=torch.ones_like(self.p_ratio)
        ids_cpu=torch.arange(p.num_envs,dtype=torch.int32)
        for asset in (p.hand,p.object):
            materials=tensor(asset.root_view.get_material_properties()).clone()
            mu=self.friction.cpu().reshape(p.num_envs,1)
            materials[...,0]=mu;materials[...,1]=mu
            asset.root_view.set_material_properties(wp.from_torch(materials.contiguous(),dtype=wp.float32),
                wp.from_torch(ids_cpu,dtype=wp.int32))
        coms=tensor(p.object.data.body_com_pose_b).clone()
        coms[..., :3]=self.com[:,None,:]
        p.object.set_coms_index(coms=coms,env_ids=torch.arange(p.num_envs,device=p.device))
        self.first_reset_readback=None
        self.first_force_readback=None
        if getattr(p.object,'__backend_name__',None)!='physx' or not type(p.object).__module__.startswith('isaaclab_physx.'):
            raise RuntimeError('This branch requires the original isaaclab_physx backend')
        self.startup_readback=self.verify_properties()

    def verify_properties(self):
        p=self.p
        mass=self.tensor(p.object.root_view.get_masses()).to(p.device).reshape(p.num_envs)
        com=self.tensor(p.object.root_view.get_coms()).to(p.device).reshape(p.num_envs,1,7)[...,:3].squeeze(1)
        inertia=self.tensor(p.object.root_view.get_inertias()).to(p.device).reshape_as(self.nominal_inertia)
        errors=dict(mass=float((mass-self.mass).abs().max()),com=float((com-self.com).abs().max()),
            inertia=float((inertia-self.nominal_inertia).abs().max()))
        for name,asset in (('hand',p.hand),('object',p.object)):
            mu=self.tensor(asset.root_view.get_material_properties())[...,:2]
            errors[name+'_friction']=float((mu-self.friction.to(mu.device)[:,None,None]).abs().max())
        if max(errors.values())>1e-6:raise RuntimeError('Physical randomized property readback mismatch: '+str(errors))
        return dict(errors=errors,mass_kg=[float(mass.min()),float(mass.max())],
            backend=type(p.object).__module__,
            friction=[float(self.friction.min()),float(self.friction.max())],
            com_range_m=[float(com.min()),float(com.max())],
            scale_range=[float(self.scale.min()),float(self.scale.max())],
            physical_inertia='retained from each sized nominal50g collider; mass-only reset')

    def reset(self,ids):
        p=self.p
        if self.enabled:
            def sample(bounds,shape):return torch.rand(shape,device=p.device)*(bounds[1]-bounds[0])+bounds[0]
            self.mass[ids]=sample(self.cfg['mass_kg'],(len(ids),))
            self.p_ratio[ids]=sample(self.cfg['p_gain_ratio'],(len(ids),16))
            self.d_ratio[ids]=sample(self.cfg['d_gain_ratio'],(len(ids),16))
        else:
            self.mass[ids]=.05;self.p_ratio[ids]=1.;self.d_ratio[ids]=1.
        p.object.set_masses_index(masses=self.mass[ids,None],env_ids=ids)
        active=p.adapter.active
        p.adapter.kp[ids[:,None],active]=self.base_kp[active][None]*self.p_ratio[ids]
        p.adapter.motor_damping[ids[:,None],active]=self.base_damping[active][None]*self.d_ratio[ids]
        self.force[ids]=0.
        if self.first_reset_readback is None:
            self.first_reset_readback=self.verify_properties()
            self.first_reset_readback['p_ratio']=[float(self.p_ratio.min()),float(self.p_ratio.max())]
            self.first_reset_readback['d_ratio']=[float(self.d_ratio.min()),float(self.d_ratio.max())]

    def begin_control(self):
        if not self.enabled:return
        cfg=self.cfg
        self.force*=cfg['force_decay']**(cfg['force_decay_reference_dt_s']/cfg['force_decay_interval_s'])
        ids=torch.where(torch.rand(self.p.num_envs,device=self.p.device)<cfg['force_probability_per_control'])[0]
        self.force[ids]=torch.randn((len(ids),3),device=self.p.device)*self.mass[ids,None]*cfg['force_scale']

    def before_physics_step(self):
        if self.enabled:
            # Actual backend isaaclab_physx writes this composer through
            # root_view.apply_forces_and_torques_at_position(position_data=None),
            # hence COM force. Recompose each physics step as the body rotates.
            self.p.object.instantaneous_wrench_composer.set_forces_and_torques_index(
                forces=self.force[:,None,:],torques=torch.zeros_like(self.force[:,None,:]),is_global=True)

    def verify_forwarded_force(self):
        composer=self.p.object.instantaneous_wrench_composer
        local=self.tensor(composer.out_force_b).reshape(self.p.num_envs,3)
        local_torque=self.tensor(composer.out_torque_b).reshape(self.p.num_envs,3)
        q=self.tensor(self.p.object.data.body_link_quat_w).reshape(self.p.num_envs,4)
        xyz=q[:,:3];w=q[:,3:]
        actual=local+2*torch.cross(xyz,torch.cross(xyz,local,dim=-1)+w*local,dim=-1)
        error=float((actual-self.force).abs().max())
        torque=float(local_torque.abs().max())
        if error>1e-5 or torque>1e-5:
            raise RuntimeError('Prepared world COM force differs: '+str((error,torque)))
        self.first_force_readback=dict(force_max_error_N=error,torque_max_Nm=torque,
            verification='coordinate check of actual source-API composer buffers; physical force measurement unavailable',
            nonzero_force_envs=int((self.force.norm(dim=-1)>0).sum()),
            requested_sample_N=self.force[:8].cpu().tolist(),prepared_sample_N=actual[:8].cpu().tolist())

    def privileged(self,position):
        # Exact source normalization; mass/friction can legitimately exceed +1.
        return torch.cat((position,(2*self.scale-.9-.6)[:,None]/.3,
            (2*self.mass-.2)[:,None]/.2,(2*self.friction-1.5)[:,None]/1.5,
            self.com/.02),dim=1)

    def snapshot(self,path):
        np.savez_compressed(path,scale=self.scale.cpu().numpy(),scale_id=self.scale_id.cpu().numpy(),
            friction=self.friction.cpu().numpy(),com_m=self.com.cpu().numpy(),mass_kg=self.mass.cpu().numpy(),
            p_ratio=self.p_ratio.cpu().numpy(),d_ratio=self.d_ratio.cpu().numpy(),
            nominal_inertia=self.nominal_inertia.cpu().numpy())
