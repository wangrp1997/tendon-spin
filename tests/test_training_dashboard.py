import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.serve_boya_training_dashboard import (FIXTURES, WM_ATTACHMENT, WM_EVALUATION, WM_LAUNCH,
    WM_RECOVERIES, WorldModel, combine_wm_evaluations, comparison, display_lineage,
    evaluation, evaluation_history, merge_wm_series, series, verified_wm_continuation, weighted_fps,
    verified_wm_recovery, wm_evaluation, wm_series, training_history, wm_training_resume_evidence)


class DashboardTests(unittest.TestCase):
    def test_wm_keeps_following_original_writer_after_auxiliary_event_file_appears(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'outputs/own';logs=output/'runs/summaries'
            training=EventFileWriter(str(logs),filename_suffix='.training')
            auxiliary=None
            def scalar(writer,tag,step,value,wall_time):
                writer.add_event(Event(wall_time=wall_time,step=step,
                    summary=Summary(value=[Summary.Value(tag=tag,simple_value=value)])))
            try:
                scalar(training,'info/epochs',0,1,100)
                scalar(training,'rewards/iter',1,10,100);training.flush()
                model=WorldModel(root)
                self.assertEqual([p['epoch'] for p in model.points(output,16384,.05)],[1])
                auxiliary=EventFileWriter(str(logs),filename_suffix='.physical_metrics')
                scalar(auxiliary,'physical_metrics/net_deg',16384,1000,110);auxiliary.flush()
                self.assertEqual([p['epoch'] for p in model.points(output,16384,.05)],[1])
                scalar(training,'info/epochs',16384,2,120)
                scalar(training,'rewards/iter',2,20,120);training.flush()
                points=model.points(output,16384,.05)
                self.assertEqual([p['epoch'] for p in points],[1,2])
                self.assertEqual([p['reward'] for p in points],[10,20])
                self.assertEqual(points[-1]['wall_time'],120)
            finally:
                training.close()
                if auxiliary is not None:auxiliary.close()

    def test_periodic_resume_excludes_unsaved_predecessor_training_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs=Path(tmp)
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            write(outputs/'old/actions_100/training/result.json',dict(arm='C',lineage_id='own',
                updates=[dict(actions=n) for n in [10,20,30]]))
            write(outputs/'new/plan.json',dict(arm='C',start_actions=20,interruption_recovery=dict(
                audit=dict(exact_saved_learning_state_verified=True,restored_actions=20))))
            write(outputs/'new/actions_100/training/result.json',dict(arm='C',lineage_id='own',
                updates=[dict(actions=21)]))
            records=training_history(outputs,['old','new'],'C')
            self.assertEqual([p['actions'] for p in series(records,'C','own')],[10,20,21])
            old=json.loads((outputs/'old/actions_100/training/result.json').read_text())
            self.assertEqual([p['actions'] for p in old['updates']],[10,20,30])
            write(outputs/'new/plan.json',dict(arm='C',start_actions=30,interruption_recovery=dict(
                audit=dict(exact_saved_learning_state_verified=True,restored_actions=20))))
            with self.assertRaises(ValueError):training_history(outputs,['old','new'],'C')

    def test_wm_training_resume_requires_own_checkpoint_command_and_full_verification(self):
        predecessor=Path('/tmp/wm-predecessor');checkpoint=str(predecessor/'runs/nn/best.pth')
        launch=dict(physical_gpu=1,planned_interactions=329318400,num_envs=1024)
        verification=dict(status='passed',exact_learning_state_comparison=True,
            frame=76152832,epoch=4648,checkpoint_sha256='own-sha',identity='own',checkpoint=checkpoint)
        recovery=dict(out='/tmp/wm-new',checkpoint=checkpoint,checkpoint_sha256='own-sha',
            physical_gpu=1,training_budget=329318400,num_envs=1024,horizon=16,max_epochs=20100,
            evaluation_interval_actions=10000000,subprocess_timeout_seconds=21600,
            core_code_and_learning_config_unchanged=True,restored_epoch=4648,
            restored_training_actions=76152832,resume_verification=verification,
            command=['python','train_wm.py','--checkpoint',checkpoint,'--out','/tmp/wm-new'])
        normalized=wm_training_resume_evidence(recovery,predecessor,launch)
        verified_wm_recovery(normalized,verification,'own',16384,predecessor,launch)
        for change in [dict(checkpoint='/tmp/foreign/learner.pth'),dict(horizon=32),
                dict(core_code_and_learning_config_unchanged=False),dict(restored_epoch=0),
                dict(command=['python','train_wm.py','--out','/tmp/wm-new'])]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                wm_training_resume_evidence(dict(recovery,**change),predecessor,launch)
        for change in [dict(restored_training_actions=1),dict(checkpoint_sha256='foreign'),
                dict(resume_verification=dict(verification,exact_learning_state_comparison=False))]:
            bad=wm_training_resume_evidence(dict(recovery,**change),predecessor,launch)
            with self.subTest(change=change),self.assertRaises(ValueError):
                verified_wm_recovery(bad,verification,'own',16384,predecessor,launch)

    def test_unrestored_worker_keeps_own_previous_history(self):
        previous=dict(arm='C',status='completed',checkpoint='outputs/own/teacher_final.pth',lineage_id='own')
        current=dict(arm='C',resume_from=str(Path('outputs/own/teacher_final.pth').resolve()),
            actions_executed=0,completed_updates=0,lineage_id='temporary-initialization')
        self.assertEqual(display_lineage(current,[previous],'C'),'own')
        self.assertIsNone(display_lineage(current,[dict(previous,arm='D')],'C'))
        self.assertIsNone(display_lineage(current,[dict(previous,status='error')],'C'))
        self.assertIsNone(display_lineage(current,[dict(previous,checkpoint='outputs/foreign.pth')],'C'))
        self.assertEqual(display_lineage(dict(current,actions_executed=123,lineage_id='restored'),[previous],'C'),'restored')

    def test_continuous_series_deduplicates_and_excludes_other_lineages(self):
        def record(arm,lineage,actions,reward):
            return dict(arm=arm,lineage_id=lineage,updates=[dict(actions=actions,update=actions//8,
                mean_recent_episode_reward=reward,actions_per_s=100.,update_wall_s=1.)])
        rows=[record('C','own',8,1.),record('C','other',16,999.),
            record('C','own',16,2.),record('C','own',8,3.),record('D','own',24,999.)]
        points=series(rows,'C','own')
        self.assertEqual([r['actions']for r in points],[8,16])
        self.assertEqual([r['reward']for r in points],[3.,2.])

    def test_throughput_uses_elapsed_time_not_mean_of_rates(self):
        self.assertEqual(weighted_fps([dict(update_wall_s=1),dict(update_wall_s=3)],64),256.)
        self.assertIsNone(weighted_fps([],64))

    def test_only_complete_matching_randomized_evaluations_appear(self):
        result=dict(status='completed',wall_budget_exhausted=False,num_envs=492,
            requested_s=30.,domain_randomization=True,inference_state_unchanged={'model':'same'},
            manifest_sha256=dict(FIXTURES),arm='C',training_actions_in_checkpoint=100)
        stat=dict(mean=1.,median=1.,p10=1.)
        summary=dict(cases=492,infrastructure_incomplete=0,cache_weighted=dict(
            net_deg=stat,last10s_net_deg=stat,valid_s=stat),complete_window=300,
            complete_window_fraction=300/492,scale_macro_net_deg=1.,scale_macro_complete_fraction=.5,
            by_scale={},stop_reasons={})
        self.assertIsNotNone(evaluation(result,summary,'C','model','result.json'))
        for key,value in [('status','evaluating'),('num_envs',1),('requested_s',1.),
                          ('domain_randomization',False),('wall_budget_exhausted',True),
                          ('manifest_sha256',{}),('inference_state_unchanged',{}),('arm','D')]:
            with self.subTest(key=key):
                self.assertIsNone(evaluation(dict(result,**{key:value}),summary,'C','model','result.json'))
        self.assertIsNone(evaluation(result,dict(summary,infrastructure_incomplete=1),'C','model','result.json'))

    def test_legacy_nominal_checkpoint_path_is_skipped_by_recovery_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            outputs=Path(tmp)/'outputs';stage=outputs/'old/actions_010000000'
            for relative,value in (
                ('training/result.json',dict(status='completed',arm='C',lineage_id='own',
                    actions_executed=10000000,checkpoint_sha256='own')),
                ('evaluation/result.json',dict(status='completed',checkpoint='teacher_final.pth',
                    training_actions_in_checkpoint=10000000)),
                ('evaluation/summary.json',{}),
            ):
                path=stage/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            self.assertEqual(evaluation_history(outputs,['old'],'C','own'),[])

    def test_no_fabricated_evaluation_when_first_d_node_is_pending(self):
        self.assertIsNone(evaluation({}, {},'D','10M','result.json'))

    def test_comparison_never_pairs_d_fresh_budget_with_c_continuation(self):
        arms={'C':dict(actions=410000000),'D':dict(actions=11000000)}
        rows=[dict(arm='C',actions=70000000),dict(arm='C',actions=410000000),
            dict(arm='D',actions=10000000)]
        result=comparison(arms,rows)
        self.assertEqual(result['training_end_actions'],11000000)
        self.assertEqual(result['evaluation_pairs'],0)
        rows.append(dict(arm='D',actions=70000000))
        self.assertEqual(comparison(arms,rows)['evaluation_actions'],[70000000])

    def test_wm_corrects_preincrement_frames_and_excludes_unfinished_epoch(self):
        def scalar(step,value):return dict(step=step,value=value,wall_time=1234.)
        scalars={'info/epochs':[scalar(0,1.),scalar(16384,2.)],
            'performance/step_inference_rl_update_fps':[scalar(0,100.),scalar(16384,200.)],
            'rewards/iter':[scalar(1,10.),scalar(2,20.)],
            'episode_lengths/iter':[scalar(2,40.)],
            'world_model/model_loss':[scalar(2,800.),scalar(3,200.)]}
        rows=wm_series(scalars,16384,.05)
        self.assertEqual([r['actions'] for r in rows],[16384,32768])
        self.assertEqual([r['fps'] for r in rows],[100.,200.])
        self.assertEqual([r['reward'] for r in rows],[10.,20.])
        self.assertIsNone(rows[0]['wm_loss'])
        self.assertEqual(rows[1]['wm_loss'],800.)
        self.assertEqual(rows[1]['episode_length_s'],2.)
        with self.assertRaisesRegex(ValueError,'counters disagree'):
            wm_series(dict(scalars,**{'info/epochs':[scalar(16384,1.)]}),16384,.05)

    def test_missing_wm_source_is_unavailable_without_breaking_cd_dashboard(self):
        self.assertFalse(WorldModel('/tmp/absent-boya-dashboard-project').collect()['available'])

    def test_wm_handover_requires_own_exact_learning_state(self):
        attachment=dict(restart_from_zero=False,restored_training_actions=1638400,restored_epoch=100,
            checkpoint_sha256='same',original_task_identity='own')
        verification=dict(status='passed',exact_learning_state_comparison=True,
            frame=1638400,epoch=100,checkpoint_sha256='same',identity='own')
        verified_wm_continuation(attachment,verification,'own',16384)
        for change in [dict(frame=0),dict(identity='other'),dict(checkpoint_sha256='other'),
                dict(exact_learning_state_comparison=False)]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                verified_wm_continuation(attachment,dict(verification,**change),'own',16384)

    def test_wm_history_excludes_unretained_predecessor_work(self):
        old=[dict(epoch=e,actions=e*16384,reward=e) for e in [99,100,101]]
        new=[dict(epoch=101,actions=101*16384,reward=999)]
        rows=merge_wm_series(old,new,100)
        self.assertEqual([r['epoch'] for r in rows],[99,100,101])
        self.assertEqual(rows[-1]['reward'],999)
        self.assertEqual([r['epoch'] for r in merge_wm_series(old,[],100)],[99,100])

    def test_wm_recovery_requires_matching_source_and_full_restore(self):
        source=Path('/tmp/wm-source')
        launch=dict(physical_gpu=1,planned_interactions=329318400,num_envs=1024)
        verification=dict(status='passed',exact_learning_state_comparison=True,frame=70008832,
            epoch=4273,checkpoint_sha256='same',identity='own',checkpoint=str(source/'learner.pth'))
        recovery=dict(predecessor_out=str(source),physical_gpu=1,training_budget=329318400,
            num_envs=1024,checkpoint=verification['checkpoint'],restart_from_zero=False,
            restored_training_actions=70008832,restored_epoch=4273,checkpoint_sha256='same',
            resume_verification=dict(verification))
        verified_wm_recovery(recovery,verification,'own',16384,source,launch)
        for change in [dict(predecessor_out='/tmp/other'),dict(physical_gpu=0),dict(num_envs=4096),
                       dict(training_budget=1),dict(restart_from_zero=True),dict(checkpoint='other'),
                       dict(checkpoint_sha256='other'),dict(restored_training_actions=70008833),
                       dict(resume_verification=dict(verification,exact_learning_state_comparison=False))]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                verified_wm_recovery(dict(recovery,**change),verification,'own',16384,source,launch)

    def test_wm_old_failed_evaluation_does_not_override_recovered_status(self):
        old=dict(status='evaluation_failed',rows=[dict(actions=n,mode='randomized',origin='old')
            for n in [10,60,80]],rejected=[],active=dict(actions=70,label='70M',error={'message':'timeout'}))
        new=dict(status='training',rows=[dict(actions=80,mode='randomized',origin='new')],
            rejected=[],active=None,next_target_actions=90)
        result=combine_wm_evaluations([(old,70),(new,None)])
        self.assertEqual([r['actions'] for r in result['rows']],[10,60,80])
        self.assertEqual(result['rows'][-1]['origin'],'new')
        self.assertEqual(result['status'],'training')
        self.assertIsNone(result['active'])
        self.assertEqual(len(result['historical_failures']),1)
        self.assertEqual(len(old['rows']),3)

    def test_wm_second_recovery_follows_new_process_and_archives_old_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fresh=root/'outputs/fresh';old=root/'outputs/old';new=root/'outputs/new'
            def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            record=dict(num_envs=1024,configuration_ids=['one']*1024,
                learning_config=dict(params=dict(config=dict(horizon_length=16,max_epochs=20100))))
            for out in (fresh,old,new):
                write(out/'result.json',dict(record,**({'error':{'message':'old timeout'}} if out==old else {})))
                write(out/'task_identity.json',dict(checkpoint_identity='own',configuration_ids=['one']*1024))
            write(root/WM_LAUNCH,dict(out=str(fresh),num_envs=1024,horizon=16,physical_gpu=1,
                planned_interactions=329318400,pid=99999999))
            write(root/WM_ATTACHMENT,dict(predecessor_out=str(fresh),new_out=str(old),physical_gpu=1,
                training_budget=329318400,restart_from_zero=False,restored_training_actions=1638400,
                restored_epoch=100,checkpoint_sha256='first',original_task_identity='own',continuation_pid=99999999))
            write(old/'resume_verified.json',dict(status='passed',exact_learning_state_comparison=True,
                frame=1638400,epoch=100,checkpoint_sha256='first',identity='own'))
            protocol=dict(interval_actions=10000000,episode_seconds=20.,nominal_seed=1700,
                randomized_seeds=[1701,1702,1703],gravity_cone_deg=4.,subprocess_timeout_seconds=7200)
            write(root/WM_EVALUATION,dict(protocol=protocol))
            checkpoint=old/'evaluations/actions_070008832/learner.pth'
            checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(b'own full learner')
            sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            verification=dict(status='passed',exact_learning_state_comparison=True,frame=70008832,
                epoch=4273,checkpoint_sha256=sha,identity='own',checkpoint=str(checkpoint))
            write(new/'resume_verified.json',verification)
            recovery=dict(predecessor_out=str(old),out=str(new),physical_gpu=1,training_budget=329318400,
                num_envs=1024,checkpoint=str(checkpoint),restart_from_zero=False,
                restored_training_actions=70008832,restored_epoch=4273,checkpoint_sha256=sha,
                resume_verification=verification,pid=99999999,repair=dict(subprocess_timeout_seconds=21600))
            write(root/WM_RECOVERIES[0],recovery)
            write(checkpoint.parent/'snapshot.json',dict(training_frame=70008832))
            write(checkpoint.parent/'error.json',dict(message='old evaluation timeout'))
            write(old/'periodic_evaluation.json',dict(status='evaluation_failed',next_target_actions=70000000))
            write(new/'periodic_evaluation.json',dict(status='training',next_target_actions=80000000))
            model=WorldModel(root)
            model.points=lambda output,batch,dt:[dict(epoch=e,actions=e*16384,wall_time=1,wall_s=200)
                for e in ([99] if output==fresh else [4273] if output==old else [4274])]
            result=model.collect()
            self.assertTrue(result['available'],result)
            self.assertEqual(result['output'],str(new))
            self.assertEqual(result['actions'],70025216)
            self.assertIsNone(result['error'])
            self.assertEqual([p['epoch'] for p in result['series']],[99,4273,4274])
            self.assertEqual(result['evaluation']['next_target_actions'],80000000)
            self.assertEqual(result['evaluation']['protocol']['subprocess_timeout_seconds'],21600)
            self.assertEqual(len(result['evaluation']['historical_failures']),1)
            checkpoint.write_bytes(b'changed learner')
            self.assertFalse(model.collect()['available'])

    def test_wm_checkpoint_counter_survives_unflushed_final_log_and_initial_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'outputs/old';new=root/'outputs/new'
            def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            record=dict(num_envs=1024,configuration_ids=['one']*1024,
                learning_config=dict(params=dict(config=dict(horizon_length=16,max_epochs=20100))))
            for out in (old,new):write(out/'result.json',record);write(out/'task_identity.json',dict(checkpoint_identity='own'))
            write(root/WM_LAUNCH,dict(out=str(old),num_envs=1024,horizon=16,physical_gpu=1,
                planned_interactions=329318400,pid=99999999))
            write(root/WM_ATTACHMENT,dict(predecessor_out=str(old),new_out=str(new),physical_gpu=1,
                training_budget=329318400,restart_from_zero=False,restored_training_actions=1638400,
                restored_epoch=100,checkpoint_sha256='same',original_task_identity='own',continuation_pid=99999999))
            write(new/'resume_verified.json',dict(status='passed',exact_learning_state_comparison=True,
                frame=1638400,epoch=100,checkpoint_sha256='same',identity='own'))
            model=WorldModel(root)
            model.points=lambda output,batch,dt:[dict(epoch=99,actions=99*16384,wall_time=1,wall_s=100)] if output==old else []
            result=model.collect()
            self.assertTrue(result['available'],result)
            self.assertEqual(result['actions'],1638400)
            self.assertEqual(result['epoch'],100)
            self.assertEqual(result['logged_epoch'],99)
            self.assertEqual(len(result['series']),1)

    def make_wm_suite(self,root):
        configurations=[f'config{i:02d}' for i in range(31)]
        catalog={key:f'asset{max(0,i-1):02d}' for i,key in enumerate(configurations)}
        identity=dict(checkpoint_identity='own',configuration_ids=[configurations[i%31] for i in range(1024)])
        protocol=dict(interval_actions=10000000,episode_seconds=20.,nominal_seed=1700,
            randomized_seeds=[1701,1702,1703],gravity_cone_deg=4.,subprocess_timeout_seconds=7200)
        frame=1638400;sha='f'*64
        def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        write(root/'snapshot.json',dict(training_frame=frame,task_identity=identity,protocol=protocol,checkpoint_sha256=sha))
        summaries={}
        fields=('net_rotation_rad','backward_rotation_rad','valid_seconds','max_drift_m',
            'mean_off_axis_rad','retained','dropped','sustained_positive_spin')
        def average(rows):return {k:sum(float(r[k]) for r in rows)/len(rows) for k in fields}
        groups={'nominal':[],'randomized':[]}
        for mode,seed in [('nominal',1700),('randomized',1701),('randomized',1702),('randomized',1703)]:
            cases=[]
            for i,cid in enumerate(configurations):
                cases.append(dict(configuration_id=cid,asset_id=catalog[cid],mode=mode,seed=seed,
                    net_rotation_rad=math.pi if i else 0.,backward_rotation_rad=0.,
                    valid_seconds=20. if i else 0.,max_drift_m=.01,mean_off_axis_rad=0.,
                    retained=bool(i),dropped=not bool(i),sustained_positive_spin=bool(i),
                    initial_failures_included=True))
            base=root/f'{mode}_{seed}';write(base/'cases.json',cases)
            write(base/'result.json',dict(status='passed',mode=mode,seed=seed,cases=31,
                configuration_ids=configurations,checkpoint_sha256=sha,training_frame=frame,physical_gpu=1,
                optimizer_steps=0,deterministic_actor=True,frozen_WM_weights=True,frozen_normalizers=True,
                no_early_failure_filter=True,evaluation_actions_count_toward_training_budget=False,
                model_hash_before=['actor','wm'],model_hash_after=['actor','wm'],
                paired_initial_conditions_verified=False))
            groups[mode]+=cases
        for mode,cases in groups.items():
            by_config={c:average([r for r in cases if r['configuration_id']==c]) for c in configurations}
            by_asset={a:average([by_config[c] for c in configurations if catalog[c]==a]) for a in set(catalog.values())}
            summaries[mode]=dict(cases=len(cases),configurations=31,assets=30,pooled=average(cases),
                asset_equal_weight=average(list(by_asset.values())),per_configuration=by_config,per_asset=by_asset)
        write(root/'summary.json',dict(status='passed',training_frame=frame,training_epoch=100,
            checkpoint_sha256=sha,evaluation_actions_count_toward_training_budget=False,
            requested_target_actions=frame,summaries=summaries))
        return identity,protocol,catalog,frame

    def test_wm_full_suite_keeps_early_failures_and_separates_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);identity,protocol,catalog,frame=self.make_wm_suite(root)
            rows=wm_evaluation(root,identity,protocol,catalog,frame)
            self.assertEqual([r['cases'] for r in rows],[31,93])
            self.assertEqual([r['mode'] for r in rows],['nominal','randomized'])
            self.assertAlmostEqual(rows[0]['pooled']['retention'],30/31)
            self.assertAlmostEqual(rows[0]['pooled']['mean_net_deg'],180*30/31)
            self.assertAlmostEqual(rows[0]['asset_equal_weight']['mean_net_deg'],180*29.5/30)
            self.assertEqual(rows[0]['per_configuration']['config00']['mean_valid_s'],0.)

    def test_wm_recovery_timeout_does_not_relax_scientific_protocol_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);identity,protocol,catalog,frame=self.make_wm_suite(root)
            path=root/'snapshot.json';snapshot=json.loads(path.read_text())
            updated=dict(protocol,subprocess_timeout_seconds=21600)
            snapshot['protocol']=updated;path.write_text(json.dumps(snapshot))
            self.assertEqual(len(wm_evaluation(root,identity,updated,catalog,frame)),2)
            with self.assertRaises(ValueError):wm_evaluation(root,identity,protocol,catalog,frame)
            snapshot['protocol']=dict(updated,gravity_cone_deg=5.);path.write_text(json.dumps(snapshot))
            with self.assertRaises(ValueError):wm_evaluation(root,identity,updated,catalog,frame)

    def test_wm_suite_rejects_partial_wrong_checkpoint_and_changed_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);identity,protocol,catalog,frame=self.make_wm_suite(root)
            path=root/'randomized_1703/result.json';original=json.loads(path.read_text())
            for change in [dict(status='initializing'),dict(checkpoint_sha256='other'),
                    dict(model_hash_after=['changed']),dict(optimizer_steps=1),dict(no_early_failure_filter=False)]:
                path.write_text(json.dumps(dict(original,**change)))
                with self.subTest(change=change),self.assertRaises(ValueError):
                    wm_evaluation(root,identity,protocol,catalog,frame)
            path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):wm_evaluation(root,identity,protocol,catalog,frame-16384)
            cases=root/'randomized_1703/cases.json';cases.write_text(json.dumps(json.loads(cases.read_text())[1:]))
            with self.assertRaises(ValueError):wm_evaluation(root,identity,protocol,catalog,frame)

    def test_wm_initializing_suite_has_no_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(wm_evaluation(Path(tmp),{}, {},{},0),[])


if __name__=='__main__':unittest.main()
