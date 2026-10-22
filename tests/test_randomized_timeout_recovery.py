import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.resume_boya_randomized_timeout import (
    TIMEOUT_MESSAGE, recovered_plan, require_stopped_source, validate_timeout_record,
)
from scripts.serve_boya_training_dashboard import FIXTURES, evaluation_history

ROOT=Path(__file__).resolve().parents[1]


class TimeoutRecoveryTests(unittest.TestCase):
    def records(self):
        training=dict(actions_executed=840040448,checkpoint_sha256='own',policy_checkpoint_sha256='policy')
        failed=dict(status='error',wall_budget_exhausted=True,error=dict(message=TIMEOUT_MESSAGE),
            arm='C',training_actions_in_checkpoint=840040448,
            checkpoint=dict(checkpoint_sha256='own',policy_sha256='policy'),
            num_envs=492,requested_s=30.,normalizer_updates=0)
        return failed,training

    def test_only_reviewed_timeout_can_be_recovered(self):
        failed,training=self.records();validate_timeout_record(failed,training)
        for change in (dict(status='completed'),dict(wall_budget_exhausted=False),
                dict(error=dict(message='Nonfinite physics; stop without retry'))):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_timeout_record(dict(failed,**change),training)

    def test_wrong_learner_policy_or_arm_is_rejected(self):
        failed,training=self.records()
        for change in (dict(arm='D'),dict(training_actions_in_checkpoint=830078976),
                dict(checkpoint=dict(checkpoint_sha256='foreign',policy_sha256='policy')),
                dict(checkpoint=dict(checkpoint_sha256='own',policy_sha256='foreign'))):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_timeout_record(dict(failed,**change),training)

    def test_original_population_window_and_freeze_are_required(self):
        failed,training=self.records()
        for change in (dict(num_envs=1),dict(requested_s=17.1),dict(normalizer_updates=1)):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_timeout_record(dict(failed,**change),training)

    def test_live_supervisor_or_worker_blocks_duplicate_recovery(self):
        state=dict(phase='error',arm='C',pid=11,child_pid=22)
        with patch('scripts.resume_boya_randomized_timeout.process_alive',return_value=False):
            require_stopped_source(state)
            for change in (dict(arm='D'),dict(phase='training')):
                with self.assertRaises(ValueError):require_stopped_source(dict(state,**change))
        for live_pid in (11,22):
            with patch('scripts.resume_boya_randomized_timeout.process_alive',side_effect=lambda pid:pid==live_pid):
                with self.assertRaises(ValueError):require_stopped_source(state)

    def test_plan_preserves_learning_budget_and_excludes_completed_training(self):
        _,training=self.records();target=dict(requested_actions=840000000,rounded_actions=840040448)
        original=dict(arm='C',num_envs=16384,seed=43,requested_total_actions=1500000000,
            reward_profile='C-drop16',scene_configuration={'same':True},
            targets=[dict(requested_actions=830000000,rounded_actions=830078976),target,
                dict(requested_actions=850000000,rounded_actions=850001920),
                dict(requested_actions=1500000000,rounded_actions=1500119040)])
        before=copy.deepcopy(original)
        result=recovered_plan(original,Path('/old'),Path('/old/actions_840000000'),training,target,21600,{})
        self.assertEqual(original,before)
        for key in ('arm','num_envs','seed','requested_total_actions','reward_profile','scene_configuration','targets'):
            self.assertEqual(result[key],original[key])
        self.assertEqual(result['start_actions'],840040448)
        self.assertEqual(result['remaining_targets'][0]['requested_actions'],850000000)
        self.assertEqual(result['initial_evaluation_target'],target)
        self.assertEqual(result['evaluation_wall_s'],21600)
        for budget in (0,7200,-1):
            with self.assertRaises(ValueError):recovered_plan(original,Path('/old'),Path('/stage'),training,target,budget,{})

    def test_supervision_loop_differs_only_in_operational_wall_budget(self):
        def function(path):
            tree=ast.parse(path.read_text())
            return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        old=function(ROOT/'scripts/run_boya_randomized_milestones.py')
        new=function(ROOT/'scripts/resume_boya_randomized_timeout.py')
        class WallBudget(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value=='7200':return ast.parse("str(plan['evaluation_wall_s'])",mode='eval').body
                return node
        self.assertEqual(ast.dump(WallBudget().visit(old)),ast.dump(new))

    def test_dashboard_keeps_valid_history_and_excludes_failed_or_foreign_results(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs=Path(directory)/'outputs'
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            def node(run,actions,status='completed',lineage='own',checkpoint='paired'):
                stage=outputs/run/f'actions_{actions:09d}'
                write(stage/'training/result.json',dict(status='completed',arm='C',lineage_id=lineage,
                    actions_executed=actions,checkpoint_sha256='paired'))
                write(stage/'evaluation/result.json',dict(status=status,wall_budget_exhausted=status!='completed',
                    num_envs=492,requested_s=30.,domain_randomization=True,inference_state_unchanged={'model':'same'},
                    manifest_sha256=dict(FIXTURES),arm='C',training_actions_in_checkpoint=actions,
                    checkpoint=dict(checkpoint_sha256=checkpoint)))
                stat=dict(mean=1.,median=1.,p10=1.)
                write(stage/'evaluation/summary.json',dict(cases=492,infrastructure_incomplete=0,
                    cache_weighted=dict(net_deg=stat,last10s_net_deg=stat,valid_s=stat),complete_window=400,
                    complete_window_fraction=400/492,scale_macro_net_deg=1.,scale_macro_complete_fraction=.8,
                    by_scale={},stop_reasons={}))
            node('old',10);node('old',20,status='error');node('new',20)
            node('new',30,lineage='foreign');node('new',40,checkpoint='foreign')
            rows=evaluation_history(outputs,['old','new'],'C','own')
            self.assertEqual([r['actions'] for r in rows],[10,20])
            self.assertIn('/new/',rows[1]['source'])
            node('new',10)
            rows=evaluation_history(outputs,['old','new'],'C','own')
            self.assertEqual(len(rows),2);self.assertIn('/new/',rows[0]['source'])


if __name__=='__main__':unittest.main()
