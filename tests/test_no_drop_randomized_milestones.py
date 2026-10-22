import ast
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from tendonspin.rl.spin_progress_reward import SpinProgressRewardSignal
from tendonspin.rl.spin_progress_no_drop import SpinProgressNoDropRewardSignal, configuration
from scripts.run_boya_randomized_milestones import validate_lineage, validate_evaluation

ROOT=Path(__file__).resolve().parents[1]


class NoDropTests(unittest.TestCase):
    def test_native_reward_differs_only_on_original_workspace_failures(self):
        c,d=SpinProgressRewardSignal(),SpinProgressNoDropRewardSignal()
        # Include saturated forward, slow forward, reverse and still trajectories.
        speeds=torch.tensor([2.,.2,-1.3,0.,.7,-.3])
        codes=torch.tensor([0,9,10,8,1,0])
        for signal in (c,d):
            signal.begin(torch.tensor([[0.,0.,0.,1.]]).repeat(6,1))
            for step in range(1,101):
                angle=-speeds*step*.0005
                q=torch.stack((torch.zeros_like(angle),torch.zeros_like(angle),
                    torch.sin(angle/2),torch.cos(angle/2)),dim=-1)
                signal.advance(q)
        args=(torch.arange(18,dtype=torch.float32).reshape(6,3)*.01,
            torch.arange(6,dtype=torch.float32)*.1,torch.arange(6,dtype=torch.float32)*.03,
            torch.arange(6,dtype=torch.float32)*.02,codes)
        left,right=c.compute_reward(*args),d.compute_reward(*args)
        for key in ('rotation','nonrotation_reward','linear_penalty'):
            torch.testing.assert_close(left[key],right[key],rtol=0,atol=0)
        torch.testing.assert_close(right['terminal_failure_cost'],torch.zeros(6),rtol=0,atol=0)
        torch.testing.assert_close(right['reward']-left['reward'],
            torch.tensor([0.,16.,16.,0.,0.,0.]),rtol=0,atol=1e-6)
        torch.testing.assert_close(right['rotation'],torch.tensor([.5,.2,-1.3,0.,.5,-.3]),atol=2e-6,rtol=0)

    def test_configuration_keeps_other_numerical_reward_settings(self):
        c=json.loads((ROOT/'assets/config/boya_spin_progress_reward.json').read_text())
        d=configuration()
        for key in ('positive_speed_cap_rad_s','negative_speed_floor_rad_s','rotation_scale',
                    'physics_dt_s','steps_per_control','terminal_failure_codes',
                    'ordinary_timeout_cost','other_penalty_scales'):
            self.assertEqual(c[key],d[key])

    def test_original_ppo_training_loop_is_preserved_in_d(self):
        trees=[ast.parse((ROOT/('scripts/train_boya_spin_progress'+suffix+'.py')).read_text())
            for suffix in ('','_no_drop')]
        loops=[next(n for n in ast.walk(t) if isinstance(n,ast.For)
            and isinstance(n.target,ast.Name) and n.target.id=='update') for t in trees]
        class Identity(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value=='C':node.value='D'
                if node.value=='First C update did not change model parameters':
                    node.value='First D update did not change model parameters'
                return node
        self.assertEqual(ast.dump(Identity().visit(loops[0])),ast.dump(loops[1]))

    def test_resume_chain_rejects_other_lineage_and_wrong_action_delta(self):
        previous=dict(checkpoint_sha256='own',lineage_id='own-lineage',actions_executed=10)
        record=dict(resumed_checkpoint_sha256='own',lineage_id='own-lineage',
            actions_executed=20,session_actions_executed=10)
        target={'rounded_actions':20}
        with patch('scripts.run_boya_randomized_milestones.validate_resume'):
            validate_lineage(record,previous,target,{})
            for field,value in [('resumed_checkpoint_sha256','other'),('lineage_id','other'),
                                ('session_actions_executed',20)]:
                bad=dict(record,**{field:value})
                with self.assertRaises(ValueError):validate_lineage(bad,previous,target,{})
            with self.assertRaises(ValueError):
                validate_lineage(dict(record,resume_from='C.pth'),None,{'rounded_actions':20},{})

    def test_randomized_validation_rejects_partial_population_or_wrong_drop_cost(self):
        record=dict(arm='D',actions_executed=20,checkpoint_sha256='same',
            reward_configuration=dict(terminal_failure_cost=0.))
        result=dict(status='completed',wall_budget_exhausted=False,inference_state_unchanged={'model':'same'},
            num_envs=492,requested_s=30.,training_actions=0,episode_resets=0,arm='D',
            training_actions_in_checkpoint=20,checkpoint=dict(checkpoint_sha256='same'))
        rows=[dict(case_index=i,first_failure_code=9,terminal_failure_cost_sum=0.,stop_reason='physical failure')
            for i in range(492)]
        def read(path):return result if path.name=='result.json' else rows
        with patch('scripts.run_boya_randomized_milestones.read_json',side_effect=read):
            validate_evaluation(Path('/tmp/mock'),record)
            rows[0]['terminal_failure_cost_sum']=-16.
            with self.assertRaises(ValueError):validate_evaluation(Path('/tmp/mock'),record)
            rows[0]['terminal_failure_cost_sum']=0.
            rows.pop()
            with self.assertRaises(ValueError):validate_evaluation(Path('/tmp/mock'),record)


if __name__=='__main__':unittest.main()
