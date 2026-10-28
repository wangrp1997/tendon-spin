import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from scripts import resume_boya_interrupted as recovery


class InterruptedRecoveryTests(unittest.TestCase):
    def test_reject_live_workers_and_execution_failures(self):
        state=dict(arm='C',phase='training',pid=1,child_pid=2)
        record=dict(arm='C',status='update completed',stop_reason=None)
        with patch.object(recovery,'process_alive',return_value=False):
            recovery.require_interrupted(state,record)
            for change in [dict(arm='D'),dict(status='error'),dict(stop_reason='process error'),
                           dict(error=dict(message='OOM'))]:
                with self.subTest(change=change),self.assertRaises(ValueError):
                    recovery.require_interrupted(state,dict(record,**change))
        with patch.object(recovery,'process_alive',return_value=True),self.assertRaises(ValueError):
            recovery.require_interrupted(state,record)

    def test_periodic_boundary_keeps_unfinished_target_and_original_contract(self):
        original=dict(arm='D',targets=[dict(requested_actions=n,rounded_actions=n)
            for n in [730,740,750]],evaluation_protocol={'cases':492},
            requested_total_actions=1500000000,num_envs=16384,reward_profile='D0')
        before=copy.deepcopy(original)
        plan=recovery.continued_plan(original,Path('/old'),Path('/old/740'),
            dict(actions_executed=738),{})
        self.assertEqual(original,before)
        self.assertEqual([t['rounded_actions'] for t in plan['remaining_targets']],[740,750])
        self.assertEqual(plan['evaluation_wall_s'],7200)
        self.assertEqual(plan['evaluation_protocol'],original['evaluation_protocol'])
        self.assertEqual(plan['requested_total_actions'],1500000000)
        self.assertEqual(recovery.continued_plan(dict(original,evaluation_wall_s=21600),
            Path('/old'),Path('/old/740'),dict(actions_executed=738),{})['evaluation_wall_s'],21600)

    def test_worker_commands_and_evaluation_loop_match_existing_supervisor(self):
        def load(path):
            tree=ast.parse(path.read_text())
            f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
            block=next(n for n in f.body if isinstance(n,ast.Try)).body
            target=next(i for i,n in enumerate(block) if isinstance(n,ast.For)
                and ast.unparse(n.iter)=="plan['remaining_targets']")
            block[1:target]=[]
            return ast.dump(f,include_attributes=False)
        root=Path(recovery.__file__).parent
        self.assertEqual(load(root/'resume_boya_interrupted.py'),load(root/'resume_boya_oom.py'))

    def test_cpu_snapshot_audit_accepts_full_state_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'outputs/old';directory=source/'actions_1000000/training'
            directory.mkdir(parents=True)
            def write(path,value):path.write_text(json.dumps(value))
            def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
            backend=root/'backend';backend.write_text('fixed')
            cache=root/'cache';cache.write_text('fixed')
            contract=dict(arm='C',num_envs=16384)
            progress=dict(actions_executed=5*131072,completed_updates=5,completed_episodes=1,
                episode_stop_counts={},lineage_id='own')
            full=dict(schema='tendonspin-hora-resume-v1',progress=progress,agent_steps=5*131072,
                epoch_num=5,contract=contract,optimizer={'state':{0:{'step':torch.tensor(6400.),
                'exp_avg':torch.zeros(1)}}},learning_rate=.001,scheduler={},torch_rng=None,
                cuda_rng=[],numpy_rng=None,python_rng=None,
                **{k:{'weight':torch.ones(1)} for k in ('model','running_mean_std','value_mean_std')})
            prior=root/'prior.pth';torch.save(dict(contract=contract,agent_steps=4*131072,
                progress=dict(lineage_id='own')),prior)
            path=directory/'teacher_u000005.pth';policy_path=directory/'teacher_u000005_policy.pth'
            plan=dict(arm='C',gpu_index=1,num_envs=16384,all_sources=[],backend_source=str(backend),
                backend_source_sha256=sha(backend),scene_configuration={'training_cache':'cache'},cache_sha256=sha(cache))
            write(source/'plan.json',plan)
            write(source/'status.json',dict(arm='C',phase='training',current_run=str(directory.parent),
                current_target={'rounded_actions':8*131072}))
            record=dict(arm='C',status='update completed',stop_reason=None,sources=[],
                checkpoint=str(path.relative_to(root)),policy_checkpoint=str(policy_path.relative_to(root)),
                resume_from=str(prior),lineage_id='own',actions_executed=7*131072,completed_updates=7)
            write(directory/'result.json',record)
            def save(state):
                torch.save(state,path)
                policy={k:copy.deepcopy(state[k]) for k in ('model','running_mean_std','value_mean_std','progress')}
                policy.update(arm='C',learner_sha256=sha(path));torch.save(policy,policy_path)
                write(directory/'latest_checkpoint.json',dict(arm='C',path=str(path),policy_path=str(policy_path),
                    sha256=sha(path),policy_sha256=sha(policy_path),**state['progress']))
            with patch.object(recovery,'ROOT',root),patch.object(recovery,'process_alive',return_value=False):
                save(full);descriptor,audit=recovery.audit_periodic(source)
                self.assertEqual(descriptor['actions_executed'],5*131072)
                self.assertEqual(audit['discarded_unsaved_updates'],2)
                for change in ['foreign_contract','nonfinite_adam','wrong_step','missing_rng','stale_policy']:
                    bad=copy.deepcopy(full)
                    if change=='foreign_contract':bad['contract']['arm']='D'
                    elif change=='nonfinite_adam':bad['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
                    elif change=='wrong_step':bad['optimizer']['state'][0]['step'].fill_(1)
                    elif change=='missing_rng':del bad['numpy_rng']
                    save(bad)
                    if change=='stale_policy':
                        policy=torch.load(policy_path,weights_only=False);policy['learner_sha256']='other'
                        torch.save(policy,policy_path)
                        meta=json.loads((directory/'latest_checkpoint.json').read_text())
                        meta['policy_sha256']=sha(policy_path);write(directory/'latest_checkpoint.json',meta)
                    with self.subTest(change=change),self.assertRaises(ValueError):recovery.audit_periodic(source)


if __name__=='__main__':unittest.main()
