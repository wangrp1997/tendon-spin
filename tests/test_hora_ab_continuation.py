"""Necessary own-arm/resume/schedule checks only; no physics or GPU work."""
import unittest
from scripts.run_boya_hora_ab_continuation import (
    ARM_PROFILES, CACHE_SHA256, START_ACTIONS, command_for,
    continuation_targets, validate_origin)


class ABContinuationTests(unittest.TestCase):
    def test_fixed_cumulative_schedule_resumes_each_own_arm(self):
        targets = continuation_targets()
        self.assertEqual([t['requested_actions'] for t in targets],
                         [20000000, 30000000, 40000000, 50000000])
        self.assertEqual([t['rounded_actions'] for t in targets],
                         [20004864, 30007296, 40001536, 50003968])
        for arm in ARM_PROFILES:
            command = command_for(arm, '/tmp/next', targets[0], 'continuation.md',
                                  {'checkpoint': '/tmp/' + arm + '_10M.pth'})
            self.assertEqual(command[command.index('--resume') + 1], '/tmp/' + arm + '_10M.pth')
            self.assertEqual(command[command.index('--reward-profile') + 1], ARM_PROFILES[arm])
            self.assertEqual(command[command.index('--eval-seconds') + 1], '30')

    def test_rejects_wrong_arm_or_partial_origin(self):
        for arm in ARM_PROFILES:
            origin = dict(status='completed', stop_reason='update budget',
                actions_executed=START_ACTIONS, completed_updates=1221, seed=43,
                num_envs=1024, engine_profile='original_tgs16_4', reward_profile=ARM_PROFILES[arm],
                controller='hora_boya_pose_ab_fresh10m_' + arm.lower(),
                termination={'profile': 'boya_workspace'}, cache_sha256=CACHE_SHA256)
            validate_origin(arm, origin)
            for key, value in [('actions_executed', 8192), ('completed_updates', 0),
                    ('stop_reason', 'manual stop'), ('num_envs', 512),
                    ('reward_profile', ARM_PROFILES['B' if arm == 'A' else 'A']),
                    ('reward_branch_migration', {})]:
                with self.assertRaises(ValueError):
                    validate_origin(arm, dict(origin, **{key: value}))


if __name__ == '__main__':
    unittest.main()
