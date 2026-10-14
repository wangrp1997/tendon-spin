"""C capacity headroom and dynamic milestone commands; no simulation."""
from pathlib import Path
import unittest

from scripts.benchmark_boya_spin_progress import select_trial, select_total_time_trial, projected_run_wall
from scripts.run_boya_hora_milestones import milestones
from scripts.run_boya_spin_progress_milestones import command_for
from tendonspin.rl.spin_progress_reward import PROFILE


class CTrainingScheduleTests(unittest.TestCase):
    def test_dynamic_rollout_budget_and_own_checkpoint_reach_the_trainer(self):
        for count in (1024,16384,32768):
            targets=milestones(0,50000000,10000000,batch=count*8)
            self.assertEqual(len(targets),5)
            for target in targets:
                self.assertEqual(target['rounded_actions']%(count*8),0)
                self.assertLess(target['rounded_actions']-target['requested_actions'],count*8)
            first=command_for(Path('/tmp/c-run'),targets[0],'protocol.md',
                Path('/tmp/cache'),Path('/tmp/scene'),Path('/tmp/config'),num_envs=count)
            self.assertNotIn('--resume',first)
            self.assertEqual(first[first.index('--num-envs')+1],str(count))
            self.assertEqual(first[first.index('--reward-profile')+1],PROFILE)
            later=command_for(Path('/tmp/c-run2'),targets[1],'protocol.md',
                Path('/tmp/cache'),Path('/tmp/scene'),Path('/tmp/config'),
                previous=dict(checkpoint='/tmp/own-c-learner.pth'),num_envs=count)
            self.assertEqual(later[later.index('--resume')+1],'/tmp/own-c-learner.pth')
        self.assertEqual(targets[-1]['rounded_actions'],50069504)

    def test_capacity_failure_and_missing_save_headroom_cannot_win_selection(self):
        rows=[dict(status='completed',num_envs=1024,peak_device_MiB=3000,actions_per_s=500),
            dict(status='completed',num_envs=16384,peak_device_MiB=89000,actions_per_s=4000),
            dict(status='completed',num_envs=32768,peak_device_MiB=96000,actions_per_s=4500),
            dict(status='capacity failure',num_envs=65536)]
        self.assertEqual(select_trial(rows,97887,4894)['num_envs'],16384)
        with self.assertRaises(RuntimeError):
            select_trial(rows[2:],97887,4894)

    def test_repeated_scene_startup_can_outweigh_higher_steady_throughput(self):
        rows=[dict(status='completed',num_envs=8192,peak_device_MiB=7000,
                actions_per_s=4000,initialization_and_shutdown_s=250),
            dict(status='completed',num_envs=16384,peak_device_MiB=10000,
                actions_per_s=5500,initialization_and_shutdown_s=500),
            dict(status='completed',num_envs=32768,peak_device_MiB=16000,
                actions_per_s=6500,initialization_and_shutdown_s=1100)]
        self.assertEqual(select_trial(rows,97887,4894)['num_envs'],32768)
        self.assertEqual(select_total_time_trial(rows,97887,4894,50000000,10000000)['num_envs'],16384)
        # Keeping a 10M evaluation interval also keeps the per-segment restart
        # penalty for a longer budget; 1.5B requires 150 starts, not five.
        expected=(-(-1500000000//131072)*131072)/5500+150*500
        self.assertAlmostEqual(projected_run_wall(rows[1],1500000000,10000000),expected)

    def test_original_hora_budget_keeps_all_150_evaluations_and_stops_at_the_rounded_limit(self):
        for count in (16384,24576):
            targets=milestones(0,1500000000,10000000,batch=count*8)
            self.assertEqual(len(targets),150)
            self.assertEqual(targets[-1]['requested_actions'],1500000000)
            self.assertEqual(targets[-1]['rounded_actions'],1500119040)
            command=command_for(Path('/tmp/c1p5b'),targets[-1],'protocol.md',
                Path('/tmp/cache'),Path('/tmp/scene'),Path('/tmp/config'),
                previous=dict(checkpoint='/tmp/own-c-learner.pth'),num_envs=count,
                controller='c_fresh1500000000_actions')
            self.assertEqual(command[command.index('--total-actions')+1],'1500000000')
            self.assertEqual(command[command.index('--controller')+1],'c_fresh1500000000_actions')


if __name__ == '__main__':
    unittest.main()
