# Reference: Hora v0.0.1 hora/tasks/allegro_hand_grasp.py (MIT): uniform
# canonical ±.25rad sampling and .5s holding. Isaac scene/PD from the attributed
# TendonSpin Boya adapter. Port differences declared in the accompanying protocol.
"""Bounded, independently screened Boya grasp-cache collection on Isaac GPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--num-envs', type=int, default=64)
parser.add_argument('--batches', type=int, default=8)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=False)
start = time.monotonic()
record = dict(controller='boya_fingers16_grasp_cache_v1', num_envs=args.num_envs,
    requested_batches=args.batches, seed=args.seed, batches=[], training_actions=0,
    benchmark_validated=False, controller_switches=0, accepted_random=0,
    random_candidates_executed=0, nominal_anchors_passed=0,
    stop_reason='requested batch budget')
for name in ('scripts/generate_boya_grasps.py','tendonspin/physics/isaac_parallel.py',
             'tendonspin/physics/isaac_boya.py','tendonspin/physics/isaac_boya_sharpa.py',
             'tendonspin/interfaces.py','docs/data/boya_native_contract.json',
             'docs/experiments/2026-10-08-boya-parallel-cache/PROTOCOL.md'):
    source = root/name
    (out/source.name).write_bytes(source.read_bytes())
    record.setdefault('sources',[]).append(dict(path=name,sha256=hashlib.sha256(source.read_bytes()).hexdigest()))

def save(status):
    record['status'] = status
    record['wall_s'] = time.monotonic()-start
    (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('BOYA_CACHE '+json.dumps({k:record[k] for k in ('status','wall_s','accepted_random','random_candidates_executed','nominal_anchors_passed','stop_reason')}),flush=True)

save('launching')
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True,enable_cameras=False,device='cuda:0').app
failed = False
try:
    import numpy as np
    import torch
    from tendonspin.physics.isaac_parallel import BoyaParallel, tensor
    env = BoyaParallel(root,out,args.num_envs)
    record.update(joint_names=env.names,body_names=list(env.hand.body_names),
        filter_names=env.filter_names,motor_parameters=env.parameters,
        collision_excludes=env.excludes,dt=env.dt,action_names=list(env.adapter.action_names),
        origins=env.origins.cpu().tolist())
    generator = torch.Generator(device=env.device).manual_seed(args.seed)
    actions = torch.zeros((args.num_envs,16),device=env.device)
    planned = round(.5/env.dt)
    tail_steps = round(.1/env.dt)
    reasons = ['completed','nonfinite','drift >5mm','tilt >15deg','speed >100rad/s',
               'link normal >12N','mimic error >.05rad','insufficient observed support','wall budget']
    record['reason_codes'] = reasons
    cache = []
    all_counts = Counter()
    save('collecting')
    for batch in range(args.batches):
        if time.monotonic()-start >270:
            record['stop_reason']='wall budget'; break
        initial = env.sample_grasps(generator)
        np.savez(out/f'batch_{batch:03d}_initial.npz',**{k:v.cpu().numpy() for k,v in initial.items()})
        first = torch.zeros(args.num_envs,dtype=torch.long,device=env.device)
        first_step = torch.full_like(first,-1)
        tail_support = torch.zeros_like(first)
        trace = None
        maxima = torch.zeros((args.num_envs,5),device=env.device)
        executed = 0
        for step in range(planned):
            if step%env.adapter.steps_per_control==0:
                if time.monotonic()-start >270:
                    record['stop_reason']='wall budget'; break
                env.adapter.set_action(actions)
            q,v,effort = env.advance()
            m = env.measure()
            bads = (~m['finite'],m['drift_mm']>5.,m['tilt_deg']>15.,m['max_speed']>100.,
                    m['max_normal']>12.,m['coupling_error']>.05)
            for code,bad in enumerate(bads,1):
                new = (first==0)&bad
                first[new]=code; first_step[new]=step+1
            if step>=planned-tail_steps:
                tail_support += (m['support_groups']>=2)
            maxima = torch.maximum(maxima,torch.stack([m[k] for k in
                ('drift_mm','tilt_deg','max_speed','max_normal','coupling_error')],dim=-1))
            frame = dict(m, joint_pos_before=q,joint_vel_before=v,action=actions,
                commands=env.adapter.commands,motor_effort_requested=effort,
                actuator_effort_forwarded=tensor(env.hand.actuators.applied_effort),first_failure=first)
            if trace is None:
                trace = {k:np.empty((planned,*value.shape),dtype=value.cpu().numpy().dtype) for k,value in frame.items()}
            for k,value in frame.items():
                trace[k][step] = value.detach().cpu().numpy()
            executed=step+1
        if executed<planned:
            first[first==0]=8
        else:
            first[(first==0)&(tail_support<.9*tail_steps)]=7
        accepted = (first==0)
        accepted[0]=False
        ids = torch.where(accepted)[0]
        if len(ids):
            cache.append(dict(q=m['joint_pos'][ids].clone(),v=m['joint_vel'][ids].clone(),
                object_state=m['object_state'][ids].clone(),commands=env.adapter.commands[ids].clone(),
                env_id=ids.clone(),batch_id=torch.full_like(ids,batch),origins=env.origins[ids].clone()))
        counts = Counter(reasons[i] for i in first[1:].cpu().tolist())
        all_counts.update(counts)
        summary=dict(batch=batch,physics_steps=executed,simulated_s=executed*env.dt,
            random_accepted=int(accepted.sum()),random_reasons=dict(counts),
            anchor_reason=reasons[int(first[0])],anchor_maxima=maxima[0].cpu().tolist(),
            anchor_tail_observed_support_fraction=float(tail_support[0]/tail_steps))
        record['batches'].append(summary)
        record['accepted_random'] += summary['random_accepted']
        record['random_candidates_executed'] += args.num_envs-1
        record['nominal_anchors_passed'] += int(first[0]==0)
        record['random_reason_counts']=dict(all_counts)
        if trace is not None:
            np.savez(out/f'batch_{batch:03d}_execution.npz',**{k:v[:executed] for k,v in trace.items()})
        np.savez(out/f'batch_{batch:03d}_selection.npz',first_failure=first.cpu().numpy(),
            first_failure_step=first_step.cpu().numpy(),maxima=maxima.cpu().numpy(),
            tail_support_steps=tail_support.cpu().numpy(),accepted=accepted.cpu().numpy())
        del trace
        save('batch completed')
        if executed<planned: break
    if cache:
        joined={k:torch.cat([c[k] for c in cache]).cpu().numpy() for k in cache[0]}
        joined['object_state'][:,:3]-=joined['origins']
        np.savez_compressed(out/'grasp_cache.npz',**joined)
        record['cache_path']=str((out/'grasp_cache.npz').relative_to(root))
    record['cuda_peak_allocated_MB']=torch.cuda.max_memory_allocated()/1e6
except BaseException as error:
    failed=True
    record['stop_reason']='process error'
    record['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
finally:
    save('error' if failed else 'completed')
    app.close(exit_code=1 if failed else 0)
