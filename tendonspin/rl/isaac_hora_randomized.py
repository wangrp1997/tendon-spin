# Derived from TendonSpin isaac_hora.py at8cca30e (MIT), Hora v0.0.1
# AllegroHandHora (MIT), https://github.com/HaozhiQi/hora; arXiv:2210.04887.
# Reuses original Boya step/reward/history/termination implementation by inheritance.
# New: physical DR from first reset, matching scale-group caches, true privileges.
# Historical task/physics/train/eval files remain unchanged.
"""Separate randomized Boya teacher route; inherited PPO task dynamics."""
from collections import Counter
import numpy as np
import torch
from gymnasium.spaces import Box
from tendonspin.rl.isaac_hora import HoraBoyaEnv
from tendonspin.physics.isaac_parallel import tensor
from tendonspin.physics.isaac_parallel_randomized import BoyaRandomizedParallel
from tendonspin.baselines.hora_training import load_reward
from tendonspin.rl.termination import TaskTermination,make_termination_spec
from tendonspin.rl.rotation_reward import RotationRewardSignal,REPORTED,POSE_DROP
from tendonspin.rl.domain_randomization import static_samples,configure_sizes,HoraDomainRandomization


class RandomizedHoraBoyaEnv(HoraBoyaEnv):
    def __init__(self,root,out,cache,num_envs=64,trace_mode="full",scene_setup=None,
                 termination_profile="legacy_strict",engine_profile="original_tgs16_4",
                 reward_profile=REPORTED,scene_configuration=None,
                 randomization_configuration=None,training_randomization=True):
        if trace_mode not in ("full","summary"):raise ValueError(trace_mode)
        if reward_profile==POSE_DROP and termination_profile!='boya_workspace':
            raise ValueError('Drop16 reward requires the existing Boya workspace failure rules')
        self.dr_configuration=randomization_configuration
        self.training_randomization=training_randomization
        self.static_dr_samples=static_samples(randomization_configuration,num_envs,training_randomization)
        callback=scene_setup
        def randomized_setup(physics):
            configure_sizes(physics,scene_configuration['geometry'],randomization_configuration,self.static_dr_samples)
            if callback is not None:callback(physics)
        scene_setup=randomized_setup
        self.trace_mode=trace_mode
        setup=scene_setup
        if scene_configuration is not None:
            from tendonspin.physics.object_geometry import configure_object_scene
            def setup(physics):
                self.geometry_application=configure_object_scene(physics,scene_configuration['geometry'])
                if scene_setup is not None:scene_setup(physics)
        self.physics=BoyaRandomizedParallel(root,out,num_envs,scene_setup=setup,engine_profile=engine_profile)
        p=self.physics
        if scene_configuration is not None:
            from tendonspin.physics.learning_scene import configure_nominal
            self.nominal_initial_state_errors=configure_nominal(p,scene_configuration)
            with np.load(cache,allow_pickle=False) as saved:
                if saved['joint_names'].tolist()!=p.names or saved['motor_names'].tolist()!=scene_configuration['nominal_state']['motor_names']:
                    raise ValueError('Migrated cache joint/motor order differs')
        self.termination=TaskTermination(make_termination_spec(root,termination_profile,p.contract['object']['pos'][2]))
        self.device=p.device;self.num_envs=num_envs
        self.action_space=Box(-1.,1.,shape=(16,),dtype=np.float32)
        self.observation_space=Box(-5.,5.,shape=(96,),dtype=np.float32)
        with np.load(cache) as saved:
            self.cache={k:torch.as_tensor(saved[k],device=self.device) for k in ('q','object_state','commands')}
        self.reward_fn=load_reward()
        self.rotation_reward_signal=RotationRewardSignal(reward_profile,p.dt,p.adapter.steps_per_control)
        self.reward_configuration=self.rotation_reward_signal.configuration
        self.history=torch.zeros((num_envs,30,32),device=self.device)
        self.progress=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.net=torch.zeros(num_envs,device=self.device)
        self.peak=torch.zeros_like(self.net);self.backward=torch.zeros_like(self.net)
        self.valid_steps=torch.zeros_like(self.progress)
        self.diagnostic_max={k:torch.zeros_like(self.net) for k in ('drift_mm','tilt_deg','max_normal')}
        self.legacy_failed=torch.zeros(num_envs,dtype=torch.bool,device=self.device)
        self.legacy_net=torch.zeros_like(self.net)
        self.legacy_valid_steps=torch.zeros_like(self.progress)
        self.episode_ids=torch.arange(num_envs,device=self.device)
        self.next_episode_id=num_envs
        self.cache_ids=torch.full((num_envs,),-1,dtype=torch.long,device=self.device)
        self.completed=[];self.reason_counts=Counter()
        self.init_q=p.q0[:,p.active_joint_ids].clone()
        self.lower=p.adapter.control_limits[p.adapter.active,0]
        self.upper=p.adapter.control_limits[p.adapter.active,1]
        self.rotation_axis=-p.axis0
        self.stop_check=None
        self.physical_steps=0;self.actions_executed=0
        self.trace=None;self.trace_count=0;self.control_trace=[]

        self.dr=HoraDomainRandomization(p,randomization_configuration,self.static_dr_samples,training_randomization)
        p.domain_randomization=self.dr
        counts=[];indices=[]
        with np.load(cache,allow_pickle=False) as saved:
            if saved['joint_names'].tolist()!=p.names:raise ValueError('Size cache joint order differs')
            groups=saved['scale_id']
            for i in range(len(randomization_configuration['scale_centers'])):
                ids=np.flatnonzero(groups==i)
                if not len(ids):raise ValueError('A size cache is empty')
                counts.append(len(ids));indices.append(ids)
        padded=np.zeros((len(counts),max(counts)),dtype=int)
        for i,ids in enumerate(indices):padded[i,:len(ids)]=ids
        self.size_cache_counts=torch.tensor(counts,device=self.device)
        self.size_cache_indices=torch.tensor(padded,device=self.device)

    def _reset_ids(self,ids):
        p=self.physics
        group=self.dr.scale_id[ids]
        choices=(torch.rand(len(ids),device=self.device)*self.size_cache_counts[group]).long()
        samples=self.size_cache_indices[group,choices]
        self.dr.reset(ids)
        self.cache_ids[ids]=samples
        q=self.cache['q'][samples].clone()
        obj=self.cache['object_state'][samples].clone()
        obj[:,:3]+=p.origins[ids];obj[:,7:]=0.
        commands=self.cache['commands'][samples].clone()
        p._write_state(q,torch.zeros_like(q),obj,commands,ids)
        self.init_q[ids]=q[:,p.active_joint_ids]
        self.termination.reset(ids)
        self.progress[ids]=0;self.net[ids]=0.;self.peak[ids]=0.;self.backward[ids]=0.
        self.valid_steps[ids]=0
        for values in self.diagnostic_max.values():values[ids]=0.
        self.legacy_failed[ids]=False;self.legacy_net[ids]=0.;self.legacy_valid_steps[ids]=0
        self.history[ids]=self._frame(noise=False)[ids,None,:]

    def observe(self):
        obs=super().observe()
        obs['priv_info']=self.dr.privileged(tensor(self.physics.object.data.root_state_w)[:,:3]-self.physics.origins)
        return obs

    def step(self,actions):
        self.dr.begin_control()
        return super().step(actions)
