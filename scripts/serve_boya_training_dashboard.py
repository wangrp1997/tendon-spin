"""Read-only C/D + BOTYARD WM dashboard. Never imports a GPU runtime."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
HISTORY={
    'C': ['boya_c_fresh1p5b_server_v1','boya_c_fresh1p5b_server_recovery_sync_v1',
          'boya_c_fresh1p5b_server_dr_eval_v1',
          'boya_c_fresh1p5b_server_dr_eval_timeout_recovery_v1',
          'boya_c_fresh1p5b_server_oom_recovery_v1'],
    'D': ['boya_d_fresh1p5b_server_dr_eval_v1'],
}
ACTIVE={arm:names[-1] for arm,names in HISTORY.items()}
REFERENCE='boya_c_paired_dr_eval_20261016_gpu0_v3'
FIXTURES={'cases.json':'b45ac48c9632dba7828f1403ef991b985e2ab4af3e1ae045dc21a09b5d259b85',
          'stimuli.npz':'09a16d13de8b952afb07261b8fb0fbe0012c36ba84b100150f2d4f8830fb5b26'}
WM_ROOT=ROOT.parent/'boya-inhand-wm'
WM_LAUNCH='evidence/wm_server_1024_launch_20261016.json'
WM_ATTACHMENT='evidence/wm_periodic_evaluation_attachment_20261016.json'
WM_EVALUATION='evidence/wm_periodic_evaluation_integration_20261016.json'
WM_RECOVERIES=('evidence/wm_evaluation_timeout_recovery_20261022.json',)
WM_TAGS=('info/epochs','performance/step_inference_rl_update_fps','rewards/iter',
    'episode_lengths/iter','losses/a_loss','losses/c_loss','world_model/model_loss',
    'world_model/image_loss','world_model/wm_pose_loss','world_model/wm_tac_loss')


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError,ValueError):
        return default


def finite(value):
    return value if isinstance(value,(int,float)) and math.isfinite(value) else None


def series(records, arm, lineage):
    """Later continuation copies replace duplicates; never combine other lineages."""
    points={}
    for record in records:
        if record.get('arm')!=arm or record.get('lineage_id')!=lineage:
            continue
        for row in record.get('updates',[]):
            actions=row.get('actions')
            if not isinstance(actions,int) or actions<=0:
                continue
            points[actions]=dict(actions=actions,update=row.get('update'),
                reward=finite(row.get('mean_recent_episode_reward')),
                fps=finite(row.get('actions_per_s')),wall_s=finite(row.get('update_wall_s')),
                learning_rate=finite(row.get('learning_rate')))
    return [points[key] for key in sorted(points)]


def weighted_fps(updates,num_envs):
    valid=[r for r in updates if finite(r.get('update_wall_s')) is not None and r['update_wall_s']>0]
    return len(valid)*num_envs*8/sum(r['update_wall_s']for r in valid) if valid else None


def display_lineage(current,records,arm):
    """Before resume completes, use the retained learner's known history."""
    if (current.get('resume_from') and not current.get('actions_executed')
            and not current.get('completed_updates')):
        source=Path(current['resume_from']).resolve()
        for previous in reversed(records):
            if (previous.get('arm')==arm and previous.get('status')=='completed'
                    and previous.get('checkpoint')
                    and (ROOT/previous['checkpoint']).resolve()==source):
                return previous.get('lineage_id')
        return None
    return current.get('lineage_id')


def process_alive(pid):
    try:
        return (Path('/proc')/str(int(pid))/'stat').read_text().split()[2] not in ('Z','X')
    except (OSError,ValueError,TypeError,IndexError):
        return False


def evaluation(result,summary,arm,label,location):
    if (result.get('status')!='completed' or result.get('wall_budget_exhausted')
            or result.get('num_envs')!=492 or result.get('requested_s')!=30.
            or not result.get('domain_randomization') or not result.get('inference_state_unchanged')
            or result.get('manifest_sha256')!=FIXTURES
            or result.get('arm')!=arm or summary.get('cases')!=492
            or summary.get('infrastructure_incomplete',1)!=0):
        return None
    weighted=summary['cache_weighted']
    return dict(arm=arm,label=label,actions=result['training_actions_in_checkpoint'],
        mean_net_deg=weighted['net_deg']['mean'],median_net_deg=weighted['net_deg']['median'],
        p10_net_deg=weighted['net_deg']['p10'],
        complete_window=summary['complete_window'],retention=summary['complete_window_fraction'],
        scale_macro_net_deg=summary['scale_macro_net_deg'],
        scale_macro_retention=summary['scale_macro_complete_fraction'],
        last10s_net_deg=weighted['last10s_net_deg']['mean'],
        mean_valid_s=weighted['valid_s']['mean'],
        by_scale=summary['by_scale'],stop_reasons=summary['stop_reasons'],
        wall_s=result.get('wall_s'),source=location)


def evaluation_history(outputs,names,arm,lineage):
    """Keep valid pre-recovery evaluations, and prefer later validated copies."""
    rows={}
    for name in names:
        for path in sorted((outputs/name).glob('actions_*/evaluation/result.json')):
            if not re.fullmatch(r'actions_[0-9]+',path.parent.parent.name):continue
            training=read(path.parent.parent/'training/result.json',{})
            result=read(path,{});summary=read(path.parent/'summary.json',{})
            checkpoint=result.get('checkpoint')
            if not isinstance(checkpoint,dict):continue
            if (not lineage or training.get('lineage_id')!=lineage or training.get('arm')!=arm
                    or training.get('status')!='completed'
                    or training.get('actions_executed')!=result.get('training_actions_in_checkpoint')
                    or not training.get('checkpoint_sha256')
                    or training['checkpoint_sha256']!=checkpoint.get('checkpoint_sha256')):
                continue
            row=evaluation(result,summary,arm,path.parent.parent.name,str(path.relative_to(outputs.parent)))
            if row:rows[row['actions']]=row
    return [rows[key] for key in sorted(rows)]


def comparison(arms,evaluations):
    """Pair only identical training budgets; never align C's continuation to D's zero."""
    common=set(r['actions'] for r in evaluations if r['arm']=='C') & set(
        r['actions'] for r in evaluations if r['arm']=='D')
    return dict(training_end_actions=min(a['actions'] for a in arms.values()),
        evaluation_actions=sorted(common),evaluation_pairs=len(common),
        evaluation_default='separate',training_default='matched',
        c_dr_transition_actions=410124288)


def wm_series(scalars,batch,control_dt):
    """WM logs frames before increment, but epochs/reward/WM losses after increment."""
    epochs={}
    for row in scalars.get('info/epochs',[]):
        value=finite(row['value'])
        if value is None or value<1 or not float(value).is_integer():
            raise ValueError('Invalid WM epoch counter')
        epoch=int(value)
        if row['step']!=(epoch-1)*batch:
            raise ValueError('WM fresh-run epoch/frame counters disagree')
        epochs[epoch]=row
    by_tag={tag:{row['step']:row for row in rows} for tag,rows in scalars.items()}
    fields={'fps':('performance/step_inference_rl_update_fps',False),
        'reward':('rewards/iter',True),'episode_length_s':('episode_lengths/iter',True),
        'wm_loss':('world_model/model_loss',True),'image_loss':('world_model/image_loss',True),
        'pose_loss':('world_model/wm_pose_loss',True),'tactile_loss':('world_model/wm_tac_loss',True),
        'actor_loss':('losses/a_loss',False),'critic_loss':('losses/c_loss',False)}
    points=[]
    for epoch,row in sorted(epochs.items()):
        point=dict(epoch=epoch,actions=epoch*batch,wall_time=row['wall_time'])
        for field,(tag,index_is_epoch) in fields.items():
            value=finite(by_tag.get(tag,{}).get(epoch if index_is_epoch else row['step'],{}).get('value'))
            point[field]=value*control_dt if field=='episode_length_s' and value is not None else value
        point['wall_s']=batch/point['fps'] if point['fps'] is not None and point['fps']>0 else None
        points.append(point)
    return points


def verified_wm_continuation(attachment,verification,identity,batch):
    if (attachment.get('restart_from_zero') is not False
            or verification.get('status')!='passed'
            or verification.get('exact_learning_state_comparison') is not True
            or verification.get('frame')!=attachment.get('restored_training_actions')
            or verification.get('epoch')!=attachment.get('restored_epoch')
            or verification.get('frame')!=verification.get('epoch',-1)*batch
            or not verification.get('checkpoint_sha256')
            or verification['checkpoint_sha256']!=attachment.get('checkpoint_sha256')
            or verification.get('identity')!=identity
            or attachment.get('original_task_identity')!=identity):
        raise ValueError('WM continuation learning-state verification differs')


def verified_wm_recovery(recovery,verification,identity,batch,predecessor,launch):
    if (Path(recovery['predecessor_out']).resolve()!=predecessor.resolve()
            or recovery.get('physical_gpu')!=launch['physical_gpu']
            or recovery.get('training_budget')!=launch['planned_interactions']
            or recovery.get('num_envs')!=launch['num_envs']
            or recovery.get('checkpoint')!=verification.get('checkpoint')):
        raise ValueError('WM recovery source or training configuration differs')
    verified_wm_continuation(dict(recovery,original_task_identity=identity),verification,identity,batch)
    declared=recovery.get('resume_verification',{})
    for key in ('checkpoint_sha256','epoch','frame','exact_learning_state_comparison','identity'):
        if declared.get(key)!=verification.get(key):
            raise ValueError('WM recovery evidence differs: '+key)


def combine_wm_evaluations(segments):
    """Retain completed history; only the latest run controls current failures."""
    result=dict(segments[-1][0]);rows={};rejected=[];historical=[]
    for index,(state,retained_actions) in enumerate(segments):
        for row in state.get('rows',[]):
            if retained_actions is None or row['actions']<=retained_actions:
                rows[(row['actions'],row['mode'])]=row
        rejected+=state.get('rejected',[])
        active=state.get('active')or {}
        if index<len(segments)-1 and active.get('error'):
            historical.append(dict(label=active.get('label'),actions=active.get('actions'),
                error=active['error']))
    result.update(rows=sorted(rows.values(),key=lambda r:(r['actions'],r['mode'])),
        rejected=rejected,historical_failures=historical)
    result['last_evaluation']=result['rows'][-2:]
    return result


def merge_wm_series(predecessor,continuation,restored_epoch):
    points={r['epoch']:r for r in predecessor if r['epoch']<=restored_epoch}
    points.update({r['epoch']:r for r in continuation if r['epoch']>restored_epoch})
    return [points[e] for e in sorted(points)]


def wm_metrics(values):
    keys=('net_rotation_rad','backward_rotation_rad','valid_seconds','max_drift_m',
        'mean_off_axis_rad','retained','dropped','sustained_positive_spin')
    if any(finite(values.get(k)) is None for k in keys):
        raise ValueError('WM evaluation metrics are missing or nonfinite')
    if any(not 0<=values[k]<=1 for k in ('retained','dropped','sustained_positive_spin')):
        raise ValueError('WM evaluation rates are outside [0,1]')
    return dict(mean_net_deg=math.degrees(values['net_rotation_rad']),
        mean_backward_deg=math.degrees(values['backward_rotation_rad']),
        retention=values['retained'],drop_fraction=values['dropped'],
        sustained_spin=values['sustained_positive_spin'],mean_valid_s=values['valid_seconds'],
        mean_max_drift_m=values['max_drift_m'],mean_off_axis_deg=math.degrees(values['mean_off_axis_rad']))


def wm_evaluation(directory,identity,protocol,catalog,restored_frame):
    """Expose only a full trained-checkpoint suite, with all first failures retained."""
    summary=read(directory/'summary.json',{})
    if summary.get('status')!='passed':return []
    snapshot=read(directory/'snapshot.json',{})
    frame=summary.get('training_frame');sha=summary.get('checkpoint_sha256')
    configurations=set(identity['configuration_ids']);assets={catalog[c] for c in configurations}
    if (not isinstance(frame,int) or frame<=0 or frame<restored_frame
            or frame!=summary.get('training_epoch',-1)*len(identity['configuration_ids'])*16
            or snapshot.get('training_frame')!=frame
            or snapshot.get('task_identity',{}).get('checkpoint_identity')!=identity['checkpoint_identity']
            or snapshot.get('protocol')!=protocol or not sha
            or snapshot.get('checkpoint_sha256')!=sha
            or summary.get('evaluation_actions_count_toward_training_budget') is not False):
        raise ValueError('WM evaluation checkpoint, protocol or lineage differs')
    groups={'nominal':[], 'randomized':[]}
    jobs=[('nominal',protocol['nominal_seed'])]+[('randomized',s) for s in protocol['randomized_seeds']]
    for mode,seed in jobs:
        base=directory/f'{mode}_{seed}';result=read(base/'result.json',{});cases=read(base/'cases.json',[])
        if (result.get('status')!='passed' or result.get('mode')!=mode or result.get('seed')!=seed
                or result.get('checkpoint_sha256')!=sha or result.get('training_frame')!=frame
                or result.get('physical_gpu')!=1 or result.get('cases')!=len(configurations)
                or len(cases)!=len(configurations) or {r.get('configuration_id') for r in cases}!=configurations
                or set(result.get('configuration_ids',[]))!=configurations
                or result.get('optimizer_steps')!=0
                or any(result.get(k) is not True for k in ('deterministic_actor','frozen_WM_weights',
                    'frozen_normalizers','no_early_failure_filter'))
                or result.get('evaluation_actions_count_toward_training_budget') is not False
                or not result.get('model_hash_before')
                or result['model_hash_before']!=result.get('model_hash_after')
                or (frame>restored_frame and result.get('paired_initial_conditions_verified') is not True)):
            raise ValueError('WM evaluation is incomplete, unpaired or not frozen')
        for case in cases:
            wm_metrics(case)
            if (case.get('mode')!=mode or case.get('seed')!=seed
                    or case.get('asset_id')!=catalog[case['configuration_id']]
                    or case.get('initial_failures_included') is not True
                    or not 0<=case['valid_seconds']<=protocol['episode_seconds']+1e-6):
                raise ValueError('WM evaluation case coverage or window differs')
        groups[mode]+=cases
    rows=[]
    for mode,cases in groups.items():
        stats=summary.get('summaries',{}).get(mode,{})
        if (stats.get('cases')!=len(cases) or stats.get('configurations')!=len(configurations)
                or stats.get('assets')!=len(assets)
                or set(stats.get('per_configuration',{}))!=configurations
                or set(stats.get('per_asset',{}))!=assets):
            raise ValueError('WM evaluation summary coverage differs')
        rows.append(dict(mode=mode,actions=frame,epoch=summary['training_epoch'],
            label=directory.name,cases=len(cases),configurations=len(configurations),assets=len(assets),
            seconds=protocol['episode_seconds'],requested_target_actions=summary['requested_target_actions'],
            checkpoint_sha256=sha,source=str(directory/'summary.json'),
            asset_equal_weight=wm_metrics(stats['asset_equal_weight']),pooled=wm_metrics(stats['pooled']),
            per_asset={key:wm_metrics(v) for key,v in stats['per_asset'].items()},
            per_configuration={key:wm_metrics(v) for key,v in stats['per_configuration'].items()}))
    return rows


def wm_evaluation_state(root,output,identity,restored_frame,protocol_override=None):
    definition=read(root/WM_EVALUATION,{})
    protocol=protocol_override or definition.get('protocol');state=read(output/'periodic_evaluation.json',{})
    if not protocol:return dict(enabled=False,rows=[],reason='尚无评估协议记录')
    enabled=bool(state);rows=[];rejected=[];active=[]
    catalog={r['configuration_id']:r['asset_id'] for r in read(root/'assets/catalog.json',{}).get('configurations',[])}
    for directory in (output/'evaluations').glob('*'):
        if not directory.is_dir():continue
        if (directory/'summary.json').exists():
            try:rows+=wm_evaluation(directory,identity,protocol,catalog,restored_frame)
            except (ValueError,KeyError,TypeError) as error:
                rejected.append(dict(directory=directory.name,reason=str(error)))
        else:
            snapshot=read(directory/'snapshot.json',{})
            jobs=[]
            for mode,seed in [('nominal',protocol['nominal_seed'])]+[
                    ('randomized',s) for s in protocol['randomized_seeds']]:
                base=directory/f'{mode}_{seed}';result=read(base/'result.json',{});progress=read(base/'progress.json',{})
                status=result.get('status','pending')
                if status=='initializing' and progress:status='running'
                jobs.append(dict(mode=mode,seed=seed,status=status,
                    simulation_s=progress.get('control_steps',0)*.05,
                    active_cases=progress.get('active_cases'),max_s=protocol['episode_seconds']))
            active.append(dict(label=directory.name,actions=snapshot.get('training_frame'),jobs=jobs,
                error=read(directory/'error.json')))
    rows.sort(key=lambda r:(r['actions'],r['mode']))
    return dict(enabled=enabled,status=state.get('status'),protocol=protocol,
        configurations=31,assets=30,total_cases=31*(1+len(protocol['randomized_seeds'])),
        rows=rows,rejected=rejected,next_target_actions=state.get('next_target_actions'),
        active=sorted(active,key=lambda r:r.get('actions') or 0)[-1] if active else None,
        last_evaluation=rows[-2:] if rows else [])


class WorldModel:
    """Only read this declared run's metadata and incremental CPU TensorBoard scalars."""
    def __init__(self,root=WM_ROOT):
        self.root=Path(root);self.accumulators={};self.checkpoint_hashes={}

    def checkpoint_digest(self,path):
        stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
        if key not in self.checkpoint_hashes:
            with path.open('rb') as stream:
                self.checkpoint_hashes[key]=hashlib.file_digest(stream,'sha256').hexdigest()
        return self.checkpoint_hashes[key]

    def points(self,output,batch,dt):
        event_path=output/'runs/summaries'
        if not event_path.exists():return []
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        if event_path not in self.accumulators:
            self.accumulators[event_path]=EventAccumulator(str(event_path),size_guidance={'scalars':0})
        accumulator=self.accumulators[event_path];accumulator.Reload();tags=set(accumulator.Tags()['scalars'])
        scalars={tag:[dict(step=r.step,value=r.value,wall_time=r.wall_time)
            for r in accumulator.Scalars(tag)] for tag in WM_TAGS if tag in tags}
        return wm_series(scalars,batch,dt)

    def collect(self):
        launch=read(self.root/WM_LAUNCH,{})
        if not launch:return dict(available=False,label='BOTYARD WM',reason='未找到任务启动记录')
        try:
            output=Path(launch['out']).resolve()
            if not output.is_relative_to((self.root/'outputs').resolve()):
                raise ValueError('WM output is outside the declared project outputs')
            record=read(output/'result.json',{})
            config=record.get('learning_config',{}).get('params',{}).get('config',{})
            num_envs=int(record['num_envs']);horizon=int(config['horizon_length'])
            if num_envs!=launch['num_envs'] or horizon!=launch['horizon']:
                raise ValueError('WM actual environment/horizon differs from launch')
            batch=num_envs*horizon;dt=record.get('task',{}).get('control_dt',.05)
            points=self.points(output,batch,dt);pid=launch.get('pid');restored_frame=0
            identity=read(output/'task_identity.json',{})
            attachment=read(self.root/WM_ATTACHMENT,{})
            attaching=False;continuation_verified=False;history=[str(output)]
            if attachment:
                if (Path(attachment['predecessor_out']).resolve()!=output
                        or attachment.get('physical_gpu')!=launch['physical_gpu']
                        or attachment.get('training_budget')!=launch['planned_interactions']):
                    raise ValueError('WM evaluation attachment belongs to a different run')
                candidate=Path(attachment['new_out']).resolve()
                if not candidate.is_relative_to((self.root/'outputs').resolve()):
                    raise ValueError('WM continuation is outside the declared outputs')
                verification=read(candidate/'resume_verified.json')
                if verification:
                    verified_wm_continuation(attachment,verification,identity['checkpoint_identity'],batch)
                    new_identity=read(candidate/'task_identity.json',{})
                    if new_identity.get('checkpoint_identity')!=identity['checkpoint_identity']:
                        raise ValueError('WM continuation task identity differs')
                    restored_frame=verification['frame'];history.append(str(candidate))
                    points=merge_wm_series(points,self.points(candidate,batch,dt),verification['epoch'])
                    output=candidate;record=read(output/'result.json',{})
                    pid=attachment['continuation_pid'];continuation_verified=True
                else:attaching=attachment.get('status')!='attachment_failed'
            evaluation_sources=[(output,restored_frame,None)]
            definition=read(self.root/WM_EVALUATION,{})
            current_protocol=definition.get('protocol')
            for relative in WM_RECOVERIES:
                recovery=read(self.root/relative)
                if not recovery:continue
                candidate=Path(recovery['out']).resolve()
                if candidate.parent!=(self.root/'outputs').resolve():
                    raise ValueError('WM recovery is outside declared project outputs')
                verification=read(candidate/'resume_verified.json')
                if not verification:
                    attaching=True;continue
                verified_wm_recovery(recovery,verification,identity['checkpoint_identity'],batch,output,launch)
                checkpoint=Path(recovery['checkpoint']).resolve()
                if not checkpoint.is_relative_to(output.resolve()):
                    raise ValueError('WM recovery checkpoint is outside its predecessor')
                if self.checkpoint_digest(checkpoint)!=recovery['checkpoint_sha256']:
                    raise ValueError('WM recovery checkpoint hash differs')
                new_identity=read(candidate/'task_identity.json',{})
                new_record=read(candidate/'result.json',{})
                new_config=new_record.get('learning_config',{}).get('params',{}).get('config',{})
                if (new_identity.get('checkpoint_identity')!=identity['checkpoint_identity']
                        or new_record.get('num_envs')!=num_envs
                        or new_config.get('horizon_length')!=horizon
                        or new_config.get('max_epochs')!=config['max_epochs']):
                    raise ValueError('WM recovered task/learning identity differs')
                timeout=recovery.get('repair',{}).get('subprocess_timeout_seconds')
                if not isinstance(timeout,int) or timeout<=0 or not current_protocol:
                    raise ValueError('WM recovery evaluation timeout is not declared')
                current_protocol=dict(current_protocol,subprocess_timeout_seconds=timeout)
                restored_frame=verification['frame'];history.append(str(candidate))
                points=merge_wm_series(points,self.points(candidate,batch,dt),verification['epoch'])
                output=candidate;record=new_record;pid=recovery['pid']
                continuation_verified=True;attaching=False
                evaluation_sources.append((output,restored_frame,current_protocol))
            segments=[]
            for index,(directory,floor,protocol) in enumerate(evaluation_sources):
                evaluated=wm_evaluation_state(self.root,directory,identity,floor,protocol)
                ceiling=evaluation_sources[index+1][1] if index+1<len(evaluation_sources) else None
                segments.append((evaluated,ceiling))
            evaluation=combine_wm_evaluations(segments)
            live=process_alive(pid);paused=False
            if live:
                argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split('\0')
                identity=(str(self.root/'scripts/train_wm.py') in argv and '--out' in argv
                    and (self.root/argv[argv.index('--out')+1]).resolve()==output)
                if not identity:raise ValueError('WM launch PID belongs to a different process')
                paused=(Path('/proc')/str(pid)/'stat').read_text().split()[2] in ('T','t')
            last=points[-1] if points else {};now=time.time()
            age=max(0.,now-last['wall_time']) if points else None
            recent=[p for p in points[-5:] if p['wall_s'] is not None]
            fps=len(recent)*num_envs*horizon/sum(p['wall_s'] for p in recent) if recent else None
            error=record.get('error')or (evaluation.get('active')or {}).get('error')
            if not error and 'failed' in (evaluation.get('status')or ''):
                error=dict(message=evaluation['status'])
            completed=(record.get('status') in ('completed','training stopped')
                and last.get('epoch',0)>=config['max_epochs'])
            stale=age is not None and age>max(300.,5*num_envs*horizon/fps if fps else 300.)
            phase=('error' if error else 'completed' if completed else 'stopped' if not live
                else 'attaching' if attaching else 'evaluation' if evaluation.get('status')=='evaluating'
                else 'paused' if paused else 'waiting' if stale or not points else 'training')
            return dict(available=True,label='BOTYARD WM',phase=phase,
                phase_label={'training':'训练中','waiting':'等待新训练日志','paused':'已暂停',
                    'stopped':'进程已退出','completed':'已完成','error':'出现错误',
                    'attaching':'正在接入定期评估','evaluation':'冻结策略评估中'}[phase],
                process_alive=live,pid=pid,error=error,actions=max(last.get('actions',0),restored_frame),
                budget=launch['planned_interactions'],epoch=max(last.get('epoch',0),restored_frame//batch),
                logged_epoch=last.get('epoch',0),max_epochs=config['max_epochs'],
                num_envs=num_envs,horizon=horizon,gpu_index=launch['physical_gpu'],
                configurations=len(set(record.get('configuration_ids',[]))),
                domain_randomization=record.get('physical_domain_randomization'),
                fps=fps if phase=='training' else None,last_metrics=last,series=points,
                last_update_unix_s=last.get('wall_time'),log_age_s=age,
                output=str(output),launch_source=str(self.root/WM_LAUNCH),
                history=history,continuation_verified=continuation_verified,
                attachment_status=attachment.get('status'),evaluation=evaluation,
                evaluation_available=bool(evaluation['rows']))
        except Exception as error:
            return dict(available=False,label='BOTYARD WM',reason=str(error))


class Dashboard:
    def __init__(self,root=ROOT,wm_root=WM_ROOT):
        self.root=Path(root);self.lock=threading.Lock();self.cached=None;self.cached_at=0.
        self.gpu=None;self.gpu_at=0.
        self.wm=WorldModel(wm_root)

    def gpu_state(self):
        if time.monotonic()-self.gpu_at<20:
            return self.gpu
        try:
            output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True,timeout=3,stderr=subprocess.DEVNULL)
            self.gpu=[dict(index=int(p[0]),used_mib=int(p[1]),free_mib=int(p[2]),utilization=int(p[3]))
                for line in output.splitlines() if len(p:=[v.strip()for v in line.split(',')])==4]
        except (OSError,ValueError,subprocess.SubprocessError):
            self.gpu=None
        self.gpu_at=time.monotonic()
        return self.gpu

    def collect(self):
        outputs=self.root/'outputs';now=time.time();arms={};evaluations=[]
        for arm,names in HISTORY.items():
            base=outputs/ACTIVE[arm];state=read(base/'status.json',{});plan=read(base/'plan.json',{})
            current_path=Path(state['current_run']) if state.get('current_run') else base/'not-started'
            current=read(current_path/'training/result.json',{})
            records=[]
            for name in names:
                for path in sorted((outputs/name).glob('actions_*/training/result.json')):
                    if re.fullmatch(r'actions_[0-9]+',path.parent.parent.name):
                        record=read(path)
                        if record:records.append(record)
            lineage=display_lineage(current,records,arm)
            if not lineage and plan.get('source_run'):
                lineage=read(Path(plan['source_run'])/'training/result.json',{}).get('lineage_id')
            if not lineage:
                known={r.get('lineage_id') for r in records if r.get('arm')==arm and r.get('lineage_id')}
                if len(known)==1:lineage=known.pop()
            points=series(records,arm,lineage)
            actions=max([plan.get('start_actions',0),current.get('actions_executed',0),
                state.get('last_completed_actions',0)or 0,*([points[-1]['actions']]if points else [])])
            age=max(0.,now-state.get('updated_unix_s',0)) if state.get('updated_unix_s') else None
            phase=state.get('phase','unknown')
            phase_label={'prepared':'待启动','starting':'启动中','training':'训练中',
                'randomized_evaluation':'领域随机化评估中','completed':'已完成','stopped':'已停止',
                'waiting_for_gpu_memory':'等待GPU显存','error':'出现错误'}.get(phase,phase)
            initializing=phase=='training' and current.get('status')=='launching'
            if initializing:phase_label='构建训练场景'
            live=process_alive(state.get('pid'))
            error=state.get('error')or current.get('error')
            if not live and phase not in ('prepared','completed','error','stopped'):
                phase_label='监督进程已退出'
            eval_result=read(current_path/'evaluation/result.json',{}) if phase=='randomized_evaluation' else {}
            arms[arm]=dict(arm=arm,phase=phase,phase_label=phase_label,initializing=initializing,
                process_alive=live,heartbeat_age_s=age,supervisor_pid=state.get('pid'),worker_pid=state.get('child_pid'),
                actions=actions,budget=plan.get('requested_total_actions',1500000000),
                num_envs=plan.get('num_envs',16384),gpu_index=plan.get('gpu_index',1),
                fps=weighted_fps(current.get('updates',[])[-5:],plan.get('num_envs',16384))
                    if phase=='training' and not initializing else None,
                target_actions=state.get('current_target',{}).get('requested_actions'),
                drop_cost=16 if arm=='C' else 0,series=points,error=error,
                output=str(base.relative_to(self.root)),
                evaluation_progress=dict(simulation_s=eval_result.get('simulation_s'),
                    fps=eval_result.get('environment_actions_per_s'),live_cases=eval_result.get('live_cases'))
                    if eval_result else None)
            evaluations+=evaluation_history(outputs,names,arm,lineage)
        reference=outputs/REFERENCE
        for model in ('70M','230M','300M'):
            path=reference/model
            row=evaluation(read(path/'result.json',{}),read(path/'summary.json',{}),'C',model,
                str((path/'result.json').relative_to(self.root)))
            if row:evaluations.append(row)
        unique={(r['arm'],r['actions']):r for r in evaluations}
        evaluations=sorted(unique.values(),key=lambda r:(r['actions'],r['arm']))
        for arm in arms:
            complete=[r for r in evaluations if r['arm']==arm]
            arms[arm]['last_evaluation']=complete[-1] if complete else None
        return dict(updated_unix_s=now,refresh_s=15,arms=arms,evaluations=evaluations,gpus=self.gpu_state(),
            comparison=comparison(arms,evaluations),world_model=self.wm.collect(),
            protocol=dict(every_actions=10000000,cases=492,grasps=164,conditions_per_grasp=3,
                seconds=30,training_domain_randomization=True,paired_fixture=True,
                scope='训练缓存内抓取 × 新的随机物理参数、观测噪声和外力组合（ID）'),
            training_unchanged=True,mode='read-only')

    def payload(self):
        with self.lock:
            if self.cached is None or time.monotonic()-self.cached_at>=5:
                self.cached=json.dumps(self.collect(),ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
                self.cached_at=time.monotonic()
            return self.cached


def make_handler(dashboard,html):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlsplit(self.path).path
            if path in ('/','/index.html'):
                body=html.read_bytes();kind='text/html; charset=utf-8'
            elif path=='/api/status':
                try:body=dashboard.payload()
                except Exception as error:
                    self.log_error('Dashboard read failed: %r',error)
                    self.send_error(503,'Training data temporarily unavailable');return
                kind='application/json; charset=utf-8'
            elif path=='/healthz':body=b'OK\n';kind='text/plain; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
            self.wfile.write(body)

        def log_message(self,fmt,*args):
            if '200' not in str(args):super().log_message(fmt,*args)
    return Handler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--wm-root',type=Path,default=WM_ROOT)
    args=parser.parse_args()
    html=ROOT/'assets/dashboard/boya_training.html'
    server=ThreadingHTTPServer((args.host,args.port),make_handler(Dashboard(wm_root=args.wm_root),html))
    print(json.dumps(dict(status='serving',pid=os.getpid(),host=args.host,port=args.port,
        url=f'http://{args.host}:{args.port}/',mode='read-only',gpu_work=False)),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
