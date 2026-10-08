# References: Hora v0.0.1 allegro_hand_hora.py (MIT): obs/history, cache reset,
# reward, action-target structure. Boya physics/rate limits from native contract.
# Rotation increment: TendonSpin SpinTracker, moving-axis quaternion integration.
# Port differences are declared in docs/experiments/2026-10-08-boya-hora-pilot/PROTOCOL.md.
"""Nominal Boya task for the original Hora teacher PPO; no robustness claim."""
from collections import Counter
import numpy as np
import torch
from gymnasium.spaces import Box
from tendonspin.physics.isaac_parallel import BoyaParallel, tensor, axis_z
from tendonspin.baselines.hora_training import load_reward


def spin_increment(previous,current):
    previous=previous/previous.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    current=current/current.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    a,b=previous[:,:3],current[:,:3]
    vector=previous[:,3:]*b-current[:,3:]*a-torch.cross(b,a,dim=-1)
    scalar=(previous*current).sum(-1)
    sign=torch.where(scalar<0,-1.,1.)
    vector=vector*sign[:,None];scalar=scalar*sign
    norm=vector.norm(dim=-1)
    rv=vector*(2*torch.atan2(norm,scalar)/norm.clamp_min(1e-12))[:,None]
    axis=axis_z(previous)+axis_z(current)
    axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    return -(rv*axis).sum(-1)


class HoraBoyaEnv:
    reasons=('active','nonfinite','drift >5mm','tilt >15deg','speed >100rad/s',
             'link normal >12N','mimic error >.05rad','episode time limit')

    def __init__(self,root,out,cache,num_envs=64):
        self.physics=BoyaParallel(root,out,num_envs)
        p=self.physics
        self.device=p.device;self.num_envs=num_envs
        self.action_space=Box(-1.,1.,shape=(16,),dtype=np.float32)
        self.observation_space=Box(-5.,5.,shape=(96,),dtype=np.float32)
        with np.load(cache) as saved:
            self.cache={k:torch.as_tensor(saved[k],device=self.device) for k in ('q','object_state','commands')}
        self.reward_fn=load_reward()
        self.history=torch.zeros((num_envs,30,32),device=self.device)
        self.progress=torch.zeros(num_envs,dtype=torch.long,device=self.device)
        self.net=torch.zeros(num_envs,device=self.device)
        self.peak=torch.zeros_like(self.net);self.backward=torch.zeros_like(self.net)
        self.valid_steps=torch.zeros_like(self.progress)
        self.episode_ids=torch.arange(num_envs,device=self.device)
        self.next_episode_id=num_envs
        self.cache_ids=torch.full((num_envs,),-1,dtype=torch.long,device=self.device)
        self.completed=[];self.reason_counts=Counter()
        self.init_q=p.q0[:,p.active_joint_ids].clone()
        self.lower=p.adapter.control_limits[p.adapter.active,0]
        self.upper=p.adapter.control_limits[p.adapter.active,1]
        self.rotation_axis=-p.axis0
        self.physical_steps=0;self.actions_executed=0
        self.trace=None;self.trace_count=0;self.control_trace=[]

    def _frame(self,noise=True):
        p=self.physics
        q=tensor(p.hand.data.joint_pos)[:,p.active_joint_ids]
        if noise:q=q+(torch.rand_like(q)*2-1)*.02
        q=2*(q-self.lower)/(self.upper-self.lower)-1
        return torch.cat((q,p.adapter.commands[:,p.adapter.active]),dim=-1)

    def observe(self):
        p=self.physics
        priv=torch.zeros((self.num_envs,9),device=self.device)
        priv[:,:3]=tensor(p.object.data.root_state_w)[:,:3]-p.origins
        priv[:,3]=1.;priv[:,4]=-.5;priv[:,5]=0.
        return dict(obs=self.history[:,-3:].reshape(self.num_envs,96).clamp(-5,5).clone(),
                    proprio_hist=self.history.clone(),priv_info=priv)

    def _reset_ids(self,ids):
        p=self.physics
        samples=torch.randint(len(self.cache['q']),(len(ids),),device=self.device)
        self.cache_ids[ids]=samples
        q=self.cache['q'][samples].clone()
        obj=self.cache['object_state'][samples].clone()
        obj[:,:3]+=p.origins[ids];obj[:,7:]=0.
        commands=self.cache['commands'][samples].clone()
        p._write_state(q,torch.zeros_like(q),obj,commands,ids)
        self.init_q[ids]=q[:,p.active_joint_ids]
        self.progress[ids]=0;self.net[ids]=0.;self.peak[ids]=0.;self.backward[ids]=0.
        self.valid_steps[ids]=0
        self.history[ids]=self._frame(noise=False)[ids,None,:]

    def reset(self):
        self._reset_ids(torch.arange(self.num_envs,device=self.device))
        return self.observe()

    def begin_trace(self,controls):
        self.trace_capacity=controls*self.physics.adapter.steps_per_control
        self.trace=None;self.trace_count=0;self.control_trace=[]

    def flush_trace(self,path):
        if self.trace is not None:
            np.savez(path,**{k:v[:self.trace_count] for k,v in self.trace.items()},
                     **{'control_'+k:np.stack([r[k] for r in self.control_trace]) for k in self.control_trace[0]})
        self.trace=None;self.control_trace=[]

    @torch.no_grad()
    def step(self,actions):
        p=self.physics
        self.control_trace.append({k:v.cpu().numpy() for k,v in self.observe().items() if k!='proprio_hist'})
        p.adapter.set_action(actions)
        first=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        prev=tensor(p.object.data.root_state_w)[:,3:7].clone()
        for substep in range(p.adapter.steps_per_control):
            q,v,effort=p.advance()
            m=p.measure()
            bads=(~m['finite'],m['drift_mm']>5.,m['tilt_deg']>15.,m['max_speed']>100.,
                  m['max_normal']>12.,m['coupling_error']>.05)
            for code,bad in enumerate(bads,1):first[(first==0)&bad]=code
            valid=first==0
            inc=spin_increment(prev,m['object_state'][:,3:7]);prev=m['object_state'][:,3:7].clone()
            self.net+=torch.where(valid,inc,0.)
            self.backward+=torch.where(valid,(-inc).clamp_min(0),0.)
            self.peak=torch.maximum(self.peak,self.net)
            self.valid_steps+=valid
            frame=dict(m,joint_pos_before=q,joint_vel_before=v,action=actions,
                commands=p.adapter.commands,motor_effort_requested=effort,
                actuator_effort_forwarded=tensor(p.hand.actuators.applied_effort),
                first_failure=first,episode_id=self.episode_ids,cache_index=self.cache_ids,valid_prefix=valid,net_angle_rad=self.net)
            if self.trace is None:
                self.trace={k:np.empty((self.trace_capacity,*val.shape),dtype=val.cpu().numpy().dtype) for k,val in frame.items()}
            for k,val in frame.items():self.trace[k][self.trace_count]=val.cpu().numpy()
            self.trace_count+=1
            self.physical_steps+=self.num_envs
        active=p.active_joint_ids
        tau=effort[:,active]
        velocity=(m['joint_pos'][:,active]-q[:,active])/p.dt
        pose=((m['joint_pos'][:,active]-self.init_q)**2).sum(-1)
        torque=(tau**2).sum(-1)
        work=((tau*velocity).sum(-1))**2
        reward,rot,lin=self.reward_fn(m['object_state'][:,7:10],-.3,m['object_state'][:,10:13],
            self.rotation_axis,1.,.5,-.5,pose,-.3,torque,-.1,work,-2.)
        # Nonfinite physics cannot enter the network/optimizer; it is a hard blocker.
        if not m['finite'].all() or not torch.isfinite(reward).all():
            raise RuntimeError('Nonfinite physics or reward; preserve trace and stop nominal training')
        self.progress+=1;self.actions_executed+=self.num_envs
        timeout=(self.progress>=400)&(first==0)
        first[timeout]=7
        done=first!=0
        self.control_trace[-1].update(reward=reward.cpu().numpy(),done=done.cpu().numpy(),
            rotation_reward=rot.cpu().numpy(),linear_penalty=lin.cpu().numpy(),
            pose_penalty=pose.cpu().numpy(),torque_penalty=torque.cpu().numpy(),work_penalty=work.cpu().numpy())
        info=dict(time_outs=timeout,rotation_reward=float(rot.mean()),object_linvel_penalty=float(lin.mean()))
        ids=torch.where(done)[0]
        for i in ids.cpu().tolist():
            reason=self.reasons[int(first[i])]
            self.reason_counts[reason]+=1
            self.completed.append(dict(episode_id=int(self.episode_ids[i]),env=i,cache_index=int(self.cache_ids[i]),
                actions=int(self.progress[i]),valid_prefix_s=float(self.valid_steps[i]*p.dt),
                net_deg=float(torch.rad2deg(self.net[i])),peak_deg=float(torch.rad2deg(self.peak[i])),
                backward_deg=float(torch.rad2deg(self.backward[i])),stop_reason=reason))
        self.history=torch.roll(self.history,-1,dims=1)
        self.history[:,-1]=self._frame()
        if len(ids):
            self._reset_ids(ids)
            self.episode_ids[ids]=torch.arange(self.next_episode_id,self.next_episode_id+len(ids),device=self.device)
            self.next_episode_id+=len(ids)
        return self.observe(),reward,done.to(torch.uint8),info
