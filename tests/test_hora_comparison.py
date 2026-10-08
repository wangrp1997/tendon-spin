"""Comparison integrity with synthetic CPU fixtures; no simulated rotation scores."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tendonspin.evaluation.comparison import (
    validate_result, compare_initial_states, write_report, make_old_execution_contract,
    verify_checkpoint, digest, dependency_completed, diagnostics, contact_diagnostics_only)
from tendonspin.evaluation.rotation import score_prefix


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.plan=dict(requested_s=120.,seed=43,initial_state='original grasp44, not cache',
            termination={'profile':'hora_height','reset_height_m':.101},
            expected_training_actions=10002432,runtime_sources=[{'path':'physics.py','sha256':'fixed'}],
            old_execution_controller='matched_old',old_run='/old',old_training_record_sha256='old-record',
            old_checkpoint_sha256='old-weights')
        self.result=dict(status='completed',requested_s=120.,seed=43,episode_resets=0,
            controller_switches=0,training_actions=0,initial_state='original grasp44, not cache',
            headless=True,enable_cameras=False,checkpoint_sha256='old-weights',
            termination=copy.deepcopy(self.plan['termination']),training_actions_in_checkpoint=10002432,
            controller='frozen_matched_old',sources=copy.deepcopy(self.plan['runtime_sources']),
            stop_reason='object below Hora height',
            metrics={str(w):score_prefix([10.,20.,17.],.1,3,window_s=w) for w in (30.,120.)})

    def test_only_completed_dependencies_can_start_gpu_evaluation(self):
        self.assertFalse(dependency_completed({'phase':'training'}))
        for phase in ('error','stopped'):
            with self.assertRaises(ValueError):dependency_completed({'phase':phase})
        stage={'phase':'completed','training':{'exit_code':0,'status':'completed','stop_reason':'update budget'},
               'evaluation':{'exit_code':0,'status':'completed'}}
        self.assertTrue(dependency_completed(stage))
        stage['training']['stop_reason']='signal requested stop'
        with self.assertRaises(ValueError):dependency_completed(stage)

    def test_historical_strict_result_cannot_enter_new_ranking(self):
        validate_result(self.result,self.plan,'old-weights','frozen_matched_old')
        self.result['termination']={'profile':'legacy_strict'}
        with self.assertRaisesRegex(ValueError,'termination'):
            validate_result(self.result,self.plan,'old-weights','frozen_matched_old')

    def test_checkpoint_budget_and_controller_substitutions_are_rejected(self):
        changes={'checkpoint_sha256':'other','training_actions_in_checkpoint':9000,
                 'episode_resets':1,'controller_switches':1,'sources':[{'path':'physics.py','sha256':'changed'}]}
        for key,value in changes.items():
            bad=copy.deepcopy(self.result);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_result(bad,self.plan,'old-weights','frozen_matched_old')

    def test_initial_state_difference_is_not_hidden_by_equal_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';a.mkdir();b.mkdir()
            fields={k:np.zeros((1,3)) for k in ('joint_pos','joint_vel','object_state','commands')}
            np.savez(a/'initial_state.npz',**fields);np.savez(b/'initial_state.npz',**fields)
            self.assertTrue(compare_initial_states(a,b)['independently_initialized'])
            fields['object_state'][0,0]=.001
            np.savez(b/'initial_state.npz',**fields)
            with self.assertRaises(AssertionError):compare_initial_states(a,b)

    def test_migration_keeps_original_training_identity_and_normalizer_contract(self):
        old={'controller':'old_teacher','sources':[{'path':'old.py','sha256':'old'}]}
        out=make_old_execution_contract(self.plan,old)
        self.assertEqual(out['record_type'],'evaluation_execution_contract_not_training_record')
        self.assertEqual(out['training_origin']['sources'],old['sources'])
        self.assertEqual(out['training_origin']['original_termination']['profile'],'legacy_strict')
        self.assertEqual(out['termination']['profile'],'hora_height')
        self.assertEqual(out['sources'],self.plan['runtime_sources'])

    def test_checkpoint_count_is_read_from_artifact_not_just_result_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'teacher.pth'
            torch.save({'schema':'tendonspin-hora-resume-v1','progress':{'actions_executed':32,'lineage_id':'A'},
                        'agent_steps':32,'model':{'weight':torch.tensor(1.)},
                        'running_mean_std':{'count':torch.tensor(1.)}},file)
            training={'checkpoint_sha256':digest(file),'actions_executed':64,'lineage_id':'A'}
            with self.assertRaisesRegex(ValueError,'sample count'):
                verify_checkpoint(Path(tmp),training,file,64)

    def test_report_retains_early_failure_and_does_not_pool_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);plan=dict(self.plan,reuse_new_evaluation=str(out/'new_evaluation'))
            report=write_report(out,plan,self.result,self.result,{'independently_initialized':True})
            self.assertEqual(len(report['rows']),4)
            self.assertTrue(all(r['net_deg']==17. for r in report['rows']))
            self.assertTrue(all(not r['complete_window'] for r in report['rows']))
            self.assertEqual(report['independent_episodes'],2)
            self.assertFalse(report['multiple_seed_benchmark'])
            saved=json.loads((out/'comparison.json').read_text())
            self.assertEqual(saved['rows'][0]['stop_reason'],'object below Hora height')

    def test_window_diagnostics_exclude_later_and_invalid_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp)/'physics_000.npz',valid=[True,True,False],elapsed_s=[.1,40.,40.5],
                     drift_mm=[1.,9.,100.],tilt_deg=[2.,45.,100.],max_normal=[3.,12.,300.])
            values=diagnostics(tmp)
            self.assertEqual(values['30.0']['drift_mm'],1.)
            self.assertEqual(values['120.0']['drift_mm'],9.)
            self.assertEqual(values['120.0']['max_normal'],12.)

    def test_late_failure_does_not_mark_completed_30s_window_as_failed(self):
        record=copy.deepcopy(self.result)
        record['metrics']={str(w):score_prefix([1.,2.,3.,4.,5.,6.,7.],10.,7,window_s=w)
                           for w in (30.,120.)}
        with tempfile.TemporaryDirectory() as tmp:
            plan=dict(self.plan,reuse_new_evaluation=str(Path(tmp)/'new_evaluation'))
            rows=write_report(tmp,plan,record,record,{})['rows']
            self.assertEqual(rows[0]['stop_reason'],'window completed')
            self.assertEqual(rows[0]['source_episode_stop_reason'],'object below Hora height')
            self.assertEqual(rows[1]['stop_reason'],'object below Hora height')


    def test_workspace_rule_is_reported_and_old_height_score_is_rejected(self):
        plan=copy.deepcopy(self.plan)
        plan['termination']={'profile':'boya_workspace','reset_height_m':.0668}
        with self.assertRaisesRegex(ValueError,'termination'):
            validate_result(self.result,plan,'old-weights','frozen_matched_old')
        result=copy.deepcopy(self.result);result['termination']=copy.deepcopy(plan['termination'])
        validate_result(result,plan,'old-weights','frozen_matched_old')
        with tempfile.TemporaryDirectory() as tmp:
            plan['reuse_new_evaluation']=str(Path(tmp)/'new')
            write_report(tmp,plan,result,result,{})
            report=(Path(tmp)/'comparison.md').read_text()
            self.assertIn('boya_workspace',report)
            self.assertNotIn('hora_height',report)

    def test_contact_diagnostics_exception_rejects_other_physics_changes(self):
        old=("class Physics:\n"
             "    def advance(self):\n        dt = .0005\n        return dt\n"
             "    def measure(self):\n"
             "        support = ((normal@self.groups)>1e-6).sum(-1)\n"
             "        return dict(\n            support_groups=support,finite=finite)\n")
        new=old.replace('        support = ((normal@self.groups)>1e-6).sum(-1)',
                        '        group_contact = (normal@self.groups)>1e-6\n        support = group_contact.sum(-1)')
        new=new.replace('            support_groups=support,finite=finite)',
                        '            support_groups=support,finger_contact_count=group_contact[:,:5].sum(-1),\n            palm_contact=group_contact[:,5],finite=finite)')
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'old.py',Path(tmp)/'new.py';a.write_text(old);b.write_text(new)
            self.assertTrue(contact_diagnostics_only(a,b))
            for bad in (new.replace('.0005','.001'),new.replace('1e-6','1e-2'),new+'\nhidden_change = True\n'):
                b.write_text(bad)
                self.assertFalse(contact_diagnostics_only(a,b))


if __name__=='__main__':unittest.main()
