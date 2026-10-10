# Sources: TendonSpin analyze_boya_velocity_probe.py, commit86cf44f; SciPy
# float64 world rotation vectors. Hora v0.0.1 reward reconstructed by the evaluator.
# This analyzes ONE new frozen episode; prior20M/4 results are not matched-budget
# learning controls. No simulation, training, plotting or modified primary score.
"""Report actual rotation/retention and reported-velocity agreement after1M."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def identity(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    folder=args.evaluation.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    record=json.loads((folder/'result.json').read_text())
    training=json.loads(args.training.read_text())
    assert record['status']=='completed'
    assert record['episode_resets']==record['controller_switches']==record['training_actions']==0
    assert record['engine_configuration']==training['engine_configuration']
    assert max(record['initial_state_max_errors'].values())==0
    keys=('elapsed_s','valid','object_state','support_groups','net_angle_deg','drift_mm','tilt_deg','max_normal')
    blocks={k:[] for k in keys};sources=[]
    for path in sorted(folder.glob('physics_*.npz')):
        sources.append(identity(path))
        with np.load(path) as data:
            for k in keys:blocks[k].append(data[k])
    d={k:np.concatenate(v) for k,v in blocks.items()}
    assert len(d['valid'])==record['physics_steps']
    with np.load(folder/'initial_state.npz') as data:initial=data['object_state'][0].copy()
    dt=.0005
    rotation=Rotation.from_quat(d['object_state'][:,3:7])
    previous=Rotation.concatenate([Rotation.from_quat(initial[3:7]),rotation[:-1]])
    omega_pose=(rotation*previous.inv()).as_rotvec()/dt
    axes=rotation.apply([0,0,1])+previous.apply([0,0,1]);axes/=np.linalg.norm(axes,axis=1)[:,None]
    axis0=-Rotation.from_quat(initial[3:7]).apply([0,0,1])
    reported=d['object_state'][:,10:13]
    error=np.linalg.norm(reported-omega_pose,axis=1)
    moving=-(omega_pose*axes).sum(-1)
    fixed_reported=reported@axis0;fixed_pose=omega_pose@axis0
    with np.load(folder/'reconstructed_control_reward.npz') as data:
        rewards={k:data[k].copy() for k in data.files}
    valid=d['valid'];windows={}
    for label,start,end in [('first5s',0.,5.),('5to20s',5.,20.),('20to30s',20.,30.),
                             ('whole_valid_prefix',0.,record['valid_s'])]:
        mask=valid & (d['elapsed_s']>start+1e-10) & (d['elapsed_s']<=end+1e-10)
        entry=None
        if mask.any():
            contact=mask & (d['support_groups']>0)
            control=rewards['valid'] & (rewards['elapsed_s']>start+1e-10) & (rewards['elapsed_s']<=end+1e-10)
            entry=dict(start_s=start,requested_end_s=end,actual_end_s=float(d['elapsed_s'][mask][-1]),
                complete_window_observed=bool(d['elapsed_s'][mask][-1]>=end-1e-10),
                actual_pose_net_deg=float(moving[mask].sum()*dt*180/np.pi),
                visible_hand_contact_fraction=float((d['support_groups'][mask]>0).mean()),
                omega_pose_vector_rmse_rad_s=float(np.sqrt(np.mean(error[mask]**2))),
                contact_omega_pose_vector_rmse_rad_s=float(np.sqrt(np.mean(error[contact]**2))) if contact.any() else None,
                reported_fixed_axis_mean_rad_s=float(fixed_reported[mask].mean()),
                pose_fixed_axis_mean_rad_s=float(fixed_pose[mask].mean()),
                pose_moving_axis_mean_rad_s=float(moving[mask].mean()),
                original_reward_means={k:float(rewards[k][control].mean()) for k in
                    ('rotation','linear_cost','pose_cost','torque_cost','work_cost','reward')} if control.any() else None)
        windows[label]=entry
    terminal=None
    if valid.any():
        end=record['valid_s'];tail=valid & (d['elapsed_s']>end-.2+1e-10)
        before=np.searchsorted(d['elapsed_s'],max(0.,end-.2)+1e-10,side='right')-1
        terminal=dict(purpose='Posthoc endpoint interpretation; no primary angle filtering',
            last_0p2s_net_deg=float(record['net_deg']-(d['net_angle_deg'][before] if before>=0 else 0.)),
            last_0p2s_visible_hand_contact_fraction=float((d['support_groups'][tail]>0).mean()))
    result=dict(status='completed',training_actions=training['actions_executed'],new_frozen_episodes=1,
        engine_configuration=record['engine_configuration'],evaluation=record,windows=windows,
        primary_30s_metrics=record['metrics']['30.0'],actual_stop_reason=record['stop_reason'],
        valid_prefix_maxima={key:float(d[key][valid].max()) if valid.any() else None
            for key in ('drift_mm','tilt_deg','max_normal')},posthoc_endpoint_motion=terminal,
        interpretation='Single fresh1M feasibility trial; no matched-budget original4 evaluation, superiority, stable rotation or hardware claim',
        sources=sources+[identity(path) for path in (args.training,folder/'result.json',folder/'initial_state.npz',
            folder/'reconstructed_control_reward.npz',Path(__file__))])
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='completed',actions=result['training_actions'],metrics=result['primary_30s_metrics'],
        stop_reason=result['actual_stop_reason'],windows=windows),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
