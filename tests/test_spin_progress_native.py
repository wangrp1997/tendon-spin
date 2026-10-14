"""Exercise the real control loop with prescribed poses, without loading Isaac."""
import ast
from collections import Counter
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from tendonspin.baselines.hora_training import load_reward
from tendonspin.rl.rotation_reward import POSE_DROP, RotationRewardSignal
from tendonspin.rl.spin_progress_reward import SpinProgressRewardSignal
from tendonspin.rl.termination import REASONS

ROOT = Path(__file__).resolve().parents[1]


def functions(path, class_name, names, namespace):
    tree = ast.parse((ROOT/path).read_text())
    body = tree.body if class_name is None else next(n.body for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == class_name)
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def prescribed_env(c_reward=True):
    ns = dict(torch=torch, np=np, tensor=lambda t:t,
        legacy_failure_codes=lambda m:torch.zeros(5, dtype=torch.long))
    functions('tendonspin/physics/isaac_parallel.py', None, ('axis_z',), ns)
    functions('tendonspin/rl/isaac_hora.py', None, ('spin_increment',), ns)
    functions('tendonspin/rl/isaac_hora.py', 'HoraBoyaEnv', ('step','control_reward'), ns)
    step, old_reward = ns['step'], ns['control_reward']
    functions('tendonspin/rl/isaac_spin_progress.py', 'SpinProgressBoyaEnv', ('control_reward',), ns)
    state = torch.zeros(5,13)
    state[:,6] = 1
    q = torch.zeros(5,16)
    adapter = SimpleNamespace(steps_per_control=100, set_action=lambda a:None)
    p = SimpleNamespace(dt=.0005, adapter=adapter, active_joint_ids=torch.arange(16),
        object=SimpleNamespace(data=SimpleNamespace(root_state_w=state)),
        origins=torch.zeros(5,3), speeds=np.array([.4,-2.,1.,0.,0.]), substep=0)

    def advance():
        if p.substep % 100 == 0:
            p.start = Rotation.from_quat(state[:,3:7].numpy())
        p.substep += 1
        local = np.zeros((5,3))
        local[:,2] = -p.speeds*((p.substep-1)%100+1)*p.dt
        state[:,3:7] = torch.as_tensor((p.start*Rotation.from_rotvec(local)).as_quat(), dtype=torch.float32)
        return q, q, q

    p.advance = advance
    p.measure = lambda:dict(object_state=state.clone(), joint_pos=q,
        finite=torch.ones(5,dtype=torch.bool), drift_mm=torch.zeros(5),
        tilt_deg=torch.zeros(5), max_normal=torch.zeros(5))
    rule = SimpleNamespace(failures=torch.tensor([0,0,0,9,0]), timeout=torch.tensor([0,0,0,0,1],dtype=torch.bool))
    rule.failure_codes = lambda m,origins,control_boundary: rule.failures.clone() if control_boundary else torch.zeros(5,dtype=torch.long)
    rule.timeouts = lambda progress,first:rule.timeout & (first==0)
    env = SimpleNamespace(physics=p, num_envs=5, device='cpu', stop_check=None,
        trace_mode='summary', termination=rule, reward_fn=load_reward(),
        rotation_axis=torch.tensor([[0.,0.,-1.]]*5),
        rotation_reward_signal=SpinProgressRewardSignal() if c_reward else RotationRewardSignal(POSE_DROP,.0005,100),
        diagnostic_max={k:torch.zeros(5) for k in ('drift_mm','tilt_deg','max_normal')},
        legacy_failed=torch.zeros(5,dtype=torch.bool), net=torch.zeros(5),
        legacy_net=torch.zeros(5), legacy_valid_steps=torch.zeros(5,dtype=torch.long),
        backward=torch.zeros(5), peak=torch.zeros(5), valid_steps=torch.zeros(5,dtype=torch.long),
        physical_steps=0, actions_executed=0, init_q=q.clone(), progress=torch.zeros(5,dtype=torch.long),
        episode_ids=torch.arange(5), cache_ids=torch.zeros(5,dtype=torch.long),
        reasons=REASONS, reason_counts=Counter(), completed=[], history=torch.zeros(5,30,32),
        next_episode_id=5, reset_ids=[])
    env.control_reward=MethodType(ns['control_reward'] if c_reward else old_reward,env)
    env.observe=lambda:{}
    env._frame=lambda:torch.zeros(5,32)

    def reset(ids):
        env.reset_ids.append(ids.clone())
        # A discontinuous reset pose must not contribute to the next reward.
        state[ids,3:7]=torch.tensor(Rotation.from_euler('xyz',[1.4,.2,-.8]).as_quat(),dtype=torch.float32)

    env._reset_ids=reset
    return env, lambda:step(env,torch.zeros(5,16))


class NativeSpinProgressTests(unittest.TestCase):
    def test_native_loop_keeps_reverse_penalty_single_failure_cost_and_free_timeout(self):
        env, step = prescribed_env()
        _, reward, done, info = step()
        torch.testing.assert_close(reward,torch.tensor([.4,-2.,.5,-16.,0.]),atol=1e-5,rtol=0)
        torch.testing.assert_close(done,torch.tensor([0,0,0,1,1],dtype=torch.uint8))
        self.assertEqual(env.rotation_reward_signal.steps,100)
        self.assertEqual((env.physical_steps,env.actions_executed),(500,5))
        self.assertAlmostEqual(info['terminal_failure_cost'],-16/5,places=5)
        torch.testing.assert_close(env.reset_ids[0],torch.tensor([3,4]))
        env.physics.speeds[:]=0
        env.termination.failures.zero_();env.termination.timeout.zero_()
        _, reward, _, _=step()
        torch.testing.assert_close(reward,torch.zeros(5),atol=1e-6,rtol=0)

    def test_shared_boundary_preserves_b_bidirectional_cap_and_terminal_cost(self):
        _, step = prescribed_env(c_reward=False)
        _, reward, _, _ = step()
        torch.testing.assert_close(reward,torch.tensor([.4,-.5,.5,-16.,0.]),atol=1e-5,rtol=0)


if __name__ == '__main__':
    unittest.main()
