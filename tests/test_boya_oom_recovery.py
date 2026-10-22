import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from scripts.resume_boya_oom import ROOT, recovered_plan, validate_oom_record, wait_for_memory


class OomRecoveryTests(unittest.TestCase):
    def test_only_reviewed_preupdate_oom_can_recover_own_checkpoint(self):
        training = dict(checkpoint='outputs/own/teacher_final.pth')
        failure = dict(arm='C', status='error', stop_reason='process error',
            actions_executed=0, completed_updates=0, session_actions_executed=0,
            resume_from=str(ROOT/training['checkpoint']),
            error=dict(message="Failed to allocate 88 bytes on device 'cuda:0'"))
        validate_oom_record(failure, 'CUDA error: out of memory', training)
        for change in [dict(arm='D'), dict(status='completed'), dict(actions_executed=1),
                       dict(completed_updates=1), dict(session_actions_executed=1),
                       dict(resume_from=str(ROOT/'outputs/other.pth')),
                       dict(error=dict(message='nonfinite model'))]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_oom_record(dict(failure, **change), 'CUDA error: out of memory', training)
        with self.assertRaises(ValueError):
            validate_oom_record(failure, 'unrelated allocation failure', training)

    def test_memory_admission_requires_consecutive_samples_and_does_not_launch(self):
        plan = dict(resource_admission=dict(minimum_free_mib=15141,
            consecutive_samples=3, poll_seconds=5))
        state = {}; save = Mock(); check = Mock()
        samples = [dict(free_mib=n) for n in [17288, 14000, 17288, 17288, 17288]]
        with patch('scripts.resume_boya_oom.gpu_snapshot', side_effect=samples), \
                patch('scripts.resume_boya_oom.time.sleep') as sleep, \
                patch('scripts.resume_boya_oom.subprocess.Popen') as launch:
            wait_for_memory(plan, state, save, check, 'training')
        self.assertEqual(check.call_count, 5)
        self.assertEqual(save.call_count, 5)
        self.assertEqual(sleep.call_count, 4)
        self.assertEqual(state['gpu_admission']['consecutive_passes'], 3)
        self.assertNotIn('pending_phase', state)
        launch.assert_not_called()

    def test_waiting_stops_on_user_or_source_error_without_launch(self):
        plan = dict(resource_admission=dict(minimum_free_mib=15141,
            consecutive_samples=3, poll_seconds=5))
        with patch('scripts.resume_boya_oom.subprocess.Popen') as launch, \
                self.assertRaisesRegex(RuntimeError, 'stop'):
            wait_for_memory(plan, {}, Mock(), Mock(side_effect=RuntimeError('stop')), 'training')
        launch.assert_not_called()

    def test_plan_preserves_training_budget_and_only_runs_later_targets(self):
        original = dict(targets=[dict(requested_actions=n, rounded_actions=n)
            for n in [910, 920, 930, 940]], requested_total_actions=1500000000,
            num_envs=16384, reward_profile='C16', evaluation_wall_s=21600,
            physical_contract=dict(original=True))
        before = copy.deepcopy(original)
        plan = recovered_plan(original, Path('/source'), Path('/source/920'),
            dict(actions_executed=920), original['targets'][1], {})
        self.assertEqual(original, before)
        self.assertEqual([t['rounded_actions'] for t in plan['remaining_targets']], [930, 940])
        for key in ['requested_total_actions', 'num_envs', 'reward_profile',
                    'evaluation_wall_s', 'physical_contract']:
            self.assertEqual(plan[key], original[key])
        self.assertEqual(plan['new_frozen_episodes'], 2)

    def test_supervision_loop_unchanged_except_admission_and_completed_bootstrap(self):
        def load(name):
            tree = ast.parse((ROOT/'scripts'/name).read_text())
            return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        old, new = load('resume_boya_randomized_timeout.py'), load('resume_boya_oom.py')
        def strip_bootstrap(function):
            block = next(n for n in function.body if isinstance(n, ast.Try)).body
            target = next(i for i,n in enumerate(block) if isinstance(n, ast.For)
                          and ast.unparse(n.iter) == "plan['remaining_targets']")
            block[1:target] = []
        strip_bootstrap(old); strip_bootstrap(new)
        execute = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
        admission = [n for n in execute.body if isinstance(n, ast.Expr)
                     and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                     and n.value.func.id == 'wait_for_memory']
        self.assertEqual(len(admission), 1)
        execute.body.remove(admission[0])
        self.assertEqual(ast.dump(old), ast.dump(new))


if __name__ == '__main__':
    unittest.main()
