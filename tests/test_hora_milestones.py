"""Cumulative scheduling and stop propagation; no new robot simulation."""
import copy
import unittest
from scripts.run_boya_hora_milestones import milestones, require_completed_stage


class MilestoneTests(unittest.TestCase):
    def test_resume_adds_only_remaining_budget_and_evaluates_each_boundary(self):
        targets = milestones(10002432, 50000000, 10000000)
        self.assertEqual([x['requested_actions'] for x in targets], [20000000,30000000,40000000,50000000])
        self.assertEqual([x['rounded_actions'] for x in targets], [20004864,30007296,40001536,50003968])
        self.assertEqual(targets[-1]['rounded_actions'] - 10002432, 40001536)
        self.assertEqual(milestones(30007296,50000000,10000000),targets[2:])

    def test_interruption_or_partial_budget_cannot_start_next_stage(self):
        good={'phase':'completed','training':{'exit_code':0,'status':'completed',
              'stop_reason':'update budget','actions_executed':20004864},
              'evaluation':{'exit_code':0,'status':'completed'}}
        require_completed_stage(good,20004864)
        for key,value in [('phase','stopped'),('phase','error')]:
            bad=copy.deepcopy(good);bad[key]=value
            with self.assertRaises(RuntimeError):require_completed_stage(bad,20004864)
        for target,key,value in [('training','stop_reason','signal requested stop'),
                                 ('training','actions_executed',10002432),
                                 ('evaluation','exit_code',1),('evaluation','status','error')]:
            bad=copy.deepcopy(good);bad[target][key]=value
            with self.assertRaises(RuntimeError):require_completed_stage(bad,20004864)

    def test_nonround_final_budget_is_evaluated_once_and_invalid_budget_rejected(self):
        self.assertEqual([x['requested_actions'] for x in milestones(10002432,25000000,10000000)],
                         [20000000,25000000])
        for start,total,interval in [(10002432,10000000,10000000),(10002432,50000000,0)]:
            with self.assertRaises(ValueError):milestones(start,total,interval)


if __name__=='__main__':unittest.main()
