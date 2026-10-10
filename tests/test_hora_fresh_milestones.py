"""Necessary fixed-budget/own-arm resume/error propagation checks; no simulation."""
import copy
import unittest
from scripts.run_boya_hora_fresh_milestones import command_for, require_stage, targets


class FreshMilestoneTests(unittest.TestCase):
    def test_each_arm_starts_fresh_and_resumes_only_its_own_checkpoint(self):
        self.assertEqual([t['rounded_actions'] for t in targets()],[1007616,3006464,5005312,10002432])
        for arm,profile in [('A','hora_pose_delta'),('B','hora_pose_delta_drop16')]:
            first=command_for(arm,'/tmp/first',targets()[0],'protocol.md')
            self.assertNotIn('--resume',first)
            self.assertEqual(first[first.index('--reward-profile')+1],profile)
            later=command_for(arm,'/tmp/later',targets()[1],'protocol.md',{'checkpoint':'/tmp/'+arm+'.pth'})
            self.assertEqual(later[later.index('--resume')+1],'/tmp/'+arm+'.pth')
            self.assertEqual(later[later.index('--total-actions')+1],'3000000')

    def test_failed_or_partial_stage_stops_chain_but_physical_early_stop_is_a_result(self):
        stage=dict(phase='completed',training=dict(exit_code=0,status='completed',stop_reason='update budget',
            actions_executed=1007616),evaluation=dict(exit_code=0,status='completed',
            stop_reason='object below Boya manipulation region'),analysis=dict(exit_code=0,status='completed'))
        require_stage(stage,1007616)
        for key,value in [('phase','stopped'),('phase','error')]:
            bad=copy.deepcopy(stage);bad[key]=value
            with self.assertRaises(RuntimeError):require_stage(bad,1007616)
        for name,key,value in [('training','actions_executed',8192),('training','stop_reason','manual stop'),
                               ('evaluation','exit_code',1),('analysis','status','error')]:
            bad=copy.deepcopy(stage);bad[name][key]=value
            with self.assertRaises(RuntimeError):require_stage(bad,1007616)


if __name__=='__main__': unittest.main()
