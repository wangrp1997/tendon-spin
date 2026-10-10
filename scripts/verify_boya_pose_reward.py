# Sources: TendonSpin analyze_boya_small_budget.py, commit0c48392, SciPy world
# rotvec cross-check, and Hora v0.0.1 compute_hand_reward (MIT).
# Offline only: rescore EXISTING poses with the deployed pose-delta accumulator.
# No policy execution, physics replay, learning or replacement of historical scores.
"""Verify the new rotation reward on one archived original-state evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tendonspin.baselines.hora_training import load_reward
from tendonspin.physics.coordinates import axis_z_xyzw
from tendonspin.rl.rotation_reward import POSE_DELTA,RotationRewardSignal


def identity(path):
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    folder=args.evaluation.resolve();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    record=json.loads((folder/'result.json').read_text())
    assert record['status']=='completed'
    assert record['episode_resets']==record['controller_switches']==record['training_actions']==0
    paths=sorted(folder.glob('physics_*.npz'))
    blocks={k:[] for k in ('object_state','valid','elapsed_s')}
    for path in paths:
        with np.load(path) as data:
            for key in blocks:blocks[key].append(data[key])
    d={k:np.concatenate(values) for k,values in blocks.items()}
    assert len(d['valid'])==record['physics_steps']
    with np.load(folder/'initial_state.npz') as data:initial=data['object_state'][0].copy()
    with np.load(folder/'reconstructed_control_reward.npz') as data:
        rewards={k:data[k].copy() for k in data.files}
    dt=.0005;steps=100;control_dt=dt*steps
    count=len(rewards['elapsed_s'])
    endpoints=np.rint(rewards['elapsed_s']/dt).astype(int)-1
    assert np.array_equal(endpoints,np.arange(1,count+1)*steps-1)
    assert np.allclose(d['elapsed_s'][endpoints],rewards['elapsed_s'],rtol=0,atol=1e-10)
    assert np.array_equal(d['valid'][endpoints],rewards['valid'])
    q=d['object_state'][:count*steps,3:7]
    previous=np.concatenate((initial[None,3:7],q[:-1]))
    signal=RotationRewardSignal(POSE_DELTA,dt,steps)
    signal.begin(torch.as_tensor(previous[::steps]))
    grouped=q.reshape(count,steps,4)
    for i in range(steps):signal.advance(torch.as_tensor(grouped[:,i]))
    reported=torch.as_tensor(d['object_state'][endpoints,10:13])
    actual=signal.angular_velocity(reported)
    independent=(Rotation.from_quat(q)*Rotation.from_quat(previous).inv()).as_rotvec()
    independent=independent.reshape(count,steps,3).sum(axis=1)/control_dt
    error=float(np.max(np.abs(actual.numpy()-independent)))
    assert error<1e-5,('Torch/SciPy control-rate mismatch',error)
    axis=torch.tensor(-axis_z_xyzw(initial[3:7]),dtype=reported.dtype).expand(count,3)
    zeros=torch.zeros(count,dtype=reported.dtype)
    function=load_reward()
    _,old_rotation,_=function(torch.zeros_like(reported),-.3,reported,axis,
        1.,.5,-.5,zeros,-.3,zeros,-.1,zeros,-2.)
    _,new_rotation,_=function(torch.zeros_like(reported),-.3,actual,axis,
        1.,.5,-.5,zeros,-.3,zeros,-.1,zeros,-2.)
    original_error=float(np.max(np.abs(old_rotation.numpy()-rewards['rotation'])))
    assert original_error<1e-5,('Captured original reward not reproduced',original_error)
    # Keep the captured nonrotation terms exactly; only replace rotation.
    total=rewards['reward']-rewards['rotation']+new_rotation.numpy()
    np.savez(out/'rescored_control_reward.npz',elapsed_s=rewards['elapsed_s'],valid=rewards['valid'],
        reported_velocity=reported.numpy(),pose_velocity=actual.numpy(),
        original_rotation=rewards['rotation'],pose_rotation=new_rotation.numpy(),
        original_reward=rewards['reward'],pose_reward=total)
    windows={}
    valid_end=float(d['elapsed_s'][d['valid']][-1])
    for label,start,end in (('first5s',0.,5.),('5to20s',5.,20.),('20to30s',20.,30.),('whole_valid_prefix',0.,valid_end)):
        mask=rewards['valid'] & (rewards['elapsed_s']>start+1e-10) & (rewards['elapsed_s']<=end+1e-10)
        row=None
        if mask.any():
            row=dict(start_s=start,requested_end_s=end,observed_last_control_s=float(rewards['elapsed_s'][mask][-1]),
                complete_window_observed=bool(valid_end>=end-1e-10),controls=int(mask.sum()),
                reported_target_axis_rate_mean_rad_s=float((reported*axis).sum(-1).numpy()[mask].mean()),
                pose_target_axis_rate_mean_rad_s=float((actual*axis).sum(-1).numpy()[mask].mean()),
                original_rotation_reward_mean=float(rewards['rotation'][mask].mean()),
                pose_rotation_reward_mean=float(new_rotation.numpy()[mask].mean()),
                original_total_reward_mean=float(rewards['reward'][mask].mean()),
                pose_total_reward_mean=float(total[mask].mean()))
        windows[label]=row
    result=dict(status='completed',new_physics_steps=0,new_training_actions=0,new_policy_episodes=0,
        source_evaluation=str(folder),source_checkpoint_sha256=record.get('checkpoint_sha256'),
        source_actual_s=record['actual_s'],source_valid_s=record['valid_s'],
        reward_configuration=signal.configuration,controls_rescored=count,
        max_control_velocity_error_vs_scipy_rad_s=error,max_original_rotation_reward_reproduction_error=original_error,
        windows=windows,source_primary_metrics_unchanged=record.get('metrics'),
        interpretation='Offline reward-input verification only. Fixed original target axis is retained; primary moving-cylinder-axis evaluation metrics remain distinct. Clipping is unchanged, so mean clipped reward is not identical to net angular progress. No learned improvement is established.',
        sources=[identity(path) for path in paths+[folder/'result.json',folder/'initial_state.npz',folder/'reconstructed_control_reward.npz',Path(__file__),Path('tendonspin/rl/rotation_reward.py'),Path('third_party/hora/hora/tasks/allegro_hand_hora.py')]])
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','controls_rescored','max_control_velocity_error_vs_scipy_rad_s','max_original_rotation_reward_reproduction_error','windows')},ensure_ascii=False))


if __name__=='__main__':main()
