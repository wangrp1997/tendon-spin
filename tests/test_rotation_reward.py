"""Check physical rotation semantics without launching Isaac or learning a policy."""
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tendonspin.baselines.hora_training import load_reward
from tendonspin.rl.rotation_reward import (
    POSE_DELTA, POSE_DROP, REPORTED, RotationRewardSignal, configuration,
    require_checkpoint_configuration,
)


class RotationRewardTests(unittest.TestCase):
    dt=.0005
    steps=100

    def test_drop_variant_keeps_pose_signal_and_only_costs_workspace_termination(self):
        signal=RotationRewardSignal(POSE_DROP,self.dt,self.steps)
        q=torch.tensor([[0.,0.,0.,1.]])
        signal.begin(q)
        for i in range(1,self.steps+1):
            q=torch.tensor(Rotation.from_rotvec([0.,0.,.2*self.dt*i]).as_quat(),dtype=torch.float32)[None]
            signal.advance(q)
        torch.testing.assert_close(signal.angular_velocity(torch.ones(1,3)*99),
            torch.tensor([[0.,0.,.2]]),rtol=1e-6,atol=1e-6)
        codes=torch.tensor([0,7,9,10,5,6])
        base=torch.full((6,),.25)
        cost=signal.terminal_cost(codes,base)
        torch.testing.assert_close(cost,torch.tensor([0.,0.,-16.,-16.,0.,0.]),rtol=0,atol=0)
        torch.testing.assert_close(base+cost,torch.tensor([.25,.25,-15.75,-15.75,.25,.25]),rtol=0,atol=0)
        # The following control is from reset; no cost carries into its valid start.
        torch.testing.assert_close(signal.terminal_cost(torch.zeros_like(codes),base),torch.zeros_like(base))
        for profile in (REPORTED,POSE_DELTA):
            torch.testing.assert_close(RotationRewardSignal(profile,self.dt,self.steps).terminal_cost(codes,base),
                torch.zeros_like(base),rtol=0,atol=0)
        require_checkpoint_configuration({'reward_configuration':signal.configuration},signal.configuration)
        with self.assertRaisesRegex(ValueError,'differs'):
            require_checkpoint_configuration({'reward_configuration':configuration(POSE_DELTA,self.dt,self.steps)},
                signal.configuration)

    def test_stationary_pose_ignores_reported_spin_and_reset_jump(self):
        q=torch.tensor(Rotation.from_euler('xyz',[.4,-.7,1.2]).as_quat(),dtype=torch.float32)[None]
        axis=torch.tensor([[0.,0.,1.]])
        reported=axis*.35
        new=RotationRewardSignal(POSE_DELTA,self.dt,self.steps)
        old=RotationRewardSignal(REPORTED,self.dt,self.steps)
        # Different interval-start poses model a cache reset. Neither jump counts.
        for initial in (q,torch.tensor([[0.,0.,0.,1.]])):
            new.begin(initial);old.begin(initial)
            for i in range(self.steps):
                sample=initial if i%2 else -initial
                new.advance(sample);old.advance(sample)
            omega=new.angular_velocity(reported)
            torch.testing.assert_close(omega,torch.zeros_like(reported),rtol=0,atol=1e-12)
            torch.testing.assert_close(old.angular_velocity(reported),reported,rtol=0,atol=0)
            zeros=torch.zeros(1)
            reward,rotation,_=load_reward()(torch.zeros(1,3),-.3,omega,axis,
                1.,.5,-.5,zeros,-.3,zeros,-.1,zeros,-2.)
            self.assertAlmostEqual(float(rotation[0]),0.,places=12)
            self.assertAlmostEqual(float(reward[0]),0.,places=12)

    def test_world_sign_axis_and_more_than_pi_per_control(self):
        axis=np.array([.3,-.4,np.sqrt(.75)])
        initial=Rotation.from_euler('xyz',[.7,-.2,1.1])
        speeds=np.array([80.,-80.])  # +/-4 radians in one control; no endpoint alias.
        signal=RotationRewardSignal(POSE_DELTA,self.dt,self.steps)
        signal.begin(torch.tensor(np.tile(initial.as_quat(),(2,1)),dtype=torch.float32))
        for i in range(1,self.steps+1):
            rotation=Rotation.from_rotvec(speeds[:,None]*axis*self.dt*i)*initial
            q=rotation.as_quat()
            if i%3==0:q=-q
            signal.advance(torch.tensor(q,dtype=torch.float32))
        result=signal.angular_velocity(torch.zeros(2,3))
        torch.testing.assert_close(result,torch.tensor(speeds[:,None]*axis,dtype=torch.float32),rtol=1e-6,atol=1e-5)
        zeros=torch.zeros(2)
        _,rot,_=load_reward()(torch.zeros(2,3),-.3,result,
            torch.tensor(np.tile(axis,(2,1)),dtype=torch.float32),1.,.5,-.5,
            zeros,-.3,zeros,-.1,zeros,-2.)
        torch.testing.assert_close(rot,torch.tensor([.5,-.5]),rtol=0,atol=0)

    def test_incomplete_control_and_checkpoint_identity(self):
        signal=RotationRewardSignal(POSE_DELTA,self.dt,self.steps)
        q=torch.tensor([[0.,0.,0.,1.]])
        signal.begin(q);signal.advance(q)
        with self.assertRaisesRegex(RuntimeError,'complete control'):
            signal.angular_velocity(torch.zeros(1,3))
        pose=configuration(POSE_DELTA,self.dt,self.steps)
        old=configuration(REPORTED,self.dt,self.steps)
        require_checkpoint_configuration({'reward_configuration':pose},pose)
        require_checkpoint_configuration({},old)
        with self.assertRaisesRegex(ValueError,'lacks pose-delta'):
            require_checkpoint_configuration({},pose)
        with self.assertRaisesRegex(ValueError,'differs'):
            require_checkpoint_configuration({'reward_configuration':old},pose)


if __name__=='__main__':unittest.main()
