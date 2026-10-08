# References: Sharpa RL Lab 5accf024... external PD; TendonSpin physical-v2
# probe/runner logging; NVIDIA Isaac Lab Camera/AppLauncher APIs (BSD-3-Clause).
# Separate v3 controller and original-state run; native Isaac RTX images only.
"""Run the declared five-second Boya Sharpa-style holding adaptation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--seconds', type=float, default=5.)
parser.add_argument('--no-video', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
args.out = args.out.resolve()
args.out.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
record = dict(controller='boya_sharpa_pd_hold_v3', requested_s=args.seconds,
              physics_steps=0, actual_s=0., training_actions=0, controller_switches=0,
              episode_resets=0, benchmark_validated=False, hold_completed=False,
              rendering_engine='Isaac Sim RTX native camera' if not args.no_video else None,
              armature_calibrated_on_hardware=False)
frames = []
encoder = None
camera = None
video_count = 0


def save(status):
    record['status'] = status
    record['wall_s'] = time.monotonic() - started
    (args.out/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print('BOYA_SHARPA_HOLD ' + json.dumps({k: record.get(k) for k in
          ('status','physics_steps','actual_s','wall_s','stop_reason','hold_completed')}), flush=True)


for name in ('tendonspin/physics/isaac_boya_sharpa.py','tendonspin/physics/isaac_boya.py',
             'scripts/run_boya_sharpa_hold.py'):
    source = root/name
    snapshot = args.out/source.name
    snapshot.write_bytes(source.read_bytes())
    record.setdefault('sources', []).append(dict(path=name, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
save('launching')
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=not args.no_video, device='cuda:0')
app = launcher.app
failed = False
try:
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from tendonspin.physics.isaac_boya import make_scene
    from tendonspin.physics.isaac_boya_sharpa import motor_parameters, actuator_configuration, configure_constraints, SharpaPositionAdapter
    from tendonspin.physics.coordinates import axis_z_xyzw, wxyz_to_xyzw

    def to_np(value):
        if hasattr(value, 'torch'):
            value = value.torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().copy()
        if hasattr(value, 'numpy'):
            return value.numpy().copy()
        return np.asarray(value).copy()

    def snapshot():
        return dict(joint_pos=to_np(hand.data.joint_pos)[0], joint_vel=to_np(hand.data.joint_vel)[0],
                    object_state=to_np(obj.data.root_state_w)[0], body_pose=to_np(hand.data.body_link_pose_w)[0])

    contract = json.loads((root/'docs/data/boya_native_contract.json').read_text())
    imported = json.loads((root/'docs/data/isaac_boya_import_phases.json').read_text())
    parameters = motor_parameters(contract)
    record['motor_parameters'] = parameters
    record['contract'] = dict(path='docs/data/boya_native_contract.json',
        sha256=hashlib.sha256((root/'docs/data/boya_native_contract.json').read_bytes()).hexdigest())
    record['usd'] = imported['usd']

    def setup(stage, hand, obj):
        global camera
        record['restored_collision_excludes'] = configure_constraints(stage, root, contract)
        if not args.no_video:
            light_cfg = sim_utils.DomeLightCfg(intensity=1800.)
            light_cfg.func('/World/DomeLight', light_cfg)
            camera = Camera(CameraCfg(prim_path='/World/Camera', width=960, height=540,
                update_period=0., data_types=['rgb'],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=24.,
                                               clipping_range=(.01, 10.))))

    save('constructing scene')
    sim, hand, obj, sensor, filters, cfg = make_scene(root, contract, root/imported['usd']['path'],
                            actuator_cfg=actuator_configuration(parameters), before_reset=setup)
    adapter = SharpaPositionAdapter(hand, contract, parameters)
    record['joint_names'] = list(hand.joint_names)
    record['body_names'] = list(hand.body_names)
    record['configuration'] = cfg.to_dict()
    record['contact_filters'] = filters
    record['armature_readback'] = to_np(hand.root_view.get_dof_armatures()).tolist()
    record['friction_readback'] = to_np(hand.root_view.get_dof_friction_properties()).tolist()
    state0 = snapshot()
    state0['commands'] = to_np(adapter.commands)[0]
    np.savez_compressed(args.out/'initial_state.npz', **state0)
    center = np.asarray(contract['object']['pos'])
    axis0 = axis_z_xyzw(wxyz_to_xyzw(contract['object']['quat']))
    names = record['joint_names']
    master = [names.index(c['master']) for c in contract['couplings']]
    slave = [names.index(c['slave']) for c in contract['couplings']]
    video_times = []
    last_image = None
    if camera is not None:
        from PIL import Image, ImageDraw, ImageFont
        font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
        camera.set_world_poses_from_view(eyes=[center + np.array([-.19,.23,.13])],
                                        targets=[center + np.array([.02,0.,-.025])])
        encoder = subprocess.Popen(['/usr/bin/ffmpeg','-n','-loglevel','error','-f','rawvideo',
            '-pix_fmt','rgb24','-s','960x620','-r','20','-i','-','-an','-c:v','libx264',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(args.out/'isaac_native.mp4')], stdin=subprocess.PIPE)

    def capture(elapsed, drift, final=False):
        global video_count, last_image
        if camera is None:
            return
        sim.render()
        camera.update(cfg.dt, force_recompute=True)
        rgb = to_np(camera.data.output['rgb'])[0,:,:,:3].astype(np.uint8)
        frame = Image.new('RGB',(960,620),(18,23,31))
        frame.paste(Image.fromarray(rgb),(0,40))
        draw = ImageDraw.Draw(frame)
        draw.text((12,5),'伯牙手 / Isaac Sim 原生画面 / Sharpa 式 PD 夹持适配',font=font,fill='white')
        draw.text((12,582),f'仿真 {elapsed:.3f}s   圆柱漂移 {drift:.3f}mm   '+
                  ('终止：'+record['stop_reason'] if final else '保持原抓取目标；尚未训练旋转'),font=font,fill='white')
        encoder.stdin.write(np.asarray(frame).tobytes())
        video_count += 1
        video_times.append(float(elapsed))
        last_image = frame

    capture(0.,0.)
    save('executing original-state fixed-target hold')
    planned = round(args.seconds/cfg.dt)
    record['stop_reason'] = 'time limit'
    max_force = max_speed = max_coupling = max_drift = 0.
    for step in range(planned):
        if time.monotonic()-started > 260:
            record['stop_reason'] = 'wall budget'; break
        if step % adapter.steps_per_control == 0:
            adapter.set_action(torch.zeros((1,13),device=hand.device))
        before_q = to_np(hand.data.joint_pos)[0]
        before_v = to_np(hand.data.joint_vel)[0]
        effort = to_np(adapter.apply())[0]
        obj.write_data_to_sim()
        sim.step(render=False)
        hand.update(cfg.dt); obj.update(cfg.dt); sensor.update(cfg.dt,force_recompute=True)
        state = snapshot()
        pose = state['object_state']
        force = to_np(sensor.data.normal_force_matrix_w)[0,0]
        friction_force = to_np(sensor.data.friction_force_matrix_w)[0,0]
        drift = float(np.linalg.norm(pose[:3]-center)*1000)
        tilt = float(np.degrees(np.arccos(np.clip(axis_z_xyzw(pose[3:7])@axis0,-1.,1.))))
        coupling = state['joint_pos'][slave]-state['joint_pos'][master]
        speed = float(np.abs(state['joint_vel']).max())
        normal = float(np.linalg.norm(force,axis=-1).max())
        state.update(joint_pos_before=before_q,joint_vel_before=before_v,
            action=np.zeros(13), commands=to_np(adapter.commands)[0],motor_effort_requested=effort,
            actuator_effort_forwarded=to_np(hand.actuators.applied_effort)[0],
            normal_force_matrix_w=force,friction_force_matrix_w=friction_force,
            drift_mm=drift,tilt_deg=tilt,coupling_error_rad=coupling,elapsed_s=(step+1)*cfg.dt)
        finite = all(np.isfinite(v).all() for v in state.values())
        state['finite'] = finite
        frames.append(state)
        record['physics_steps'] = step+1
        record['actual_s'] = (step+1)*cfg.dt
        if finite:
            max_force=max(max_force,normal);max_speed=max(max_speed,speed)
            max_coupling=max(max_coupling,float(np.abs(coupling).max()));max_drift=max(max_drift,drift)
        if not finite: record['stop_reason']='nonfinite'; break
        if drift>5.: record['stop_reason']='drift >5mm'; break
        if tilt>15.: record['stop_reason']='tilt >15deg'; break
        if speed>100.: record['stop_reason']='joint speed >100rad/s'; break
        if normal>12.: record['stop_reason']='link normal force >12N'; break
        if (step+1)%100==0:
            capture(record['actual_s'],drift)
        if (step+1)%2000==0:
            save('holding')
    record.update(max_drift_mm=max_drift,max_joint_speed_rad_s=max_speed,
                  max_link_normal_N=max_force,max_coupling_error_rad=max_coupling)
    record['hold_completed'] = record['physics_steps']==planned and record['stop_reason']=='time limit' and max_coupling<.05
    if record['physics_steps']==planned and max_coupling>=.05:
        record['stop_reason']='time limit; coupling requirement unmet'
    capture(record['actual_s'],frames[-1]['drift_mm'] if frames else 0.,final=True)
    if last_image is not None:
        last_image.save(args.out/'isaac_native_final.png')
        for _ in range(20):
            encoder.stdin.write(np.asarray(last_image).tobytes());video_count+=1
        record['video'] = dict(path=str((args.out/'isaac_native.mp4').relative_to(root)),fps=20,
                              frames=video_count,physics_sample_times_s=video_times,terminal_still_s=1.)
except BaseException as error:
    failed = True
    record['stop_reason'] = 'process error'
    record['error'] = dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
finally:
    if encoder is not None:
        encoder.stdin.close()
        record['video_encoder_return_code'] = encoder.wait()
    if frames:
        np.savez_compressed(args.out/'execution.npz',**{k:np.array([f[k] for f in frames]) for k in frames[0]},physics_dt=cfg.dt)
        for name in ('initial_state.npz','execution.npz'):
            record[name.removesuffix('.npz')] = dict(path=str((args.out/name).relative_to(root)),
                                sha256=hashlib.sha256((args.out/name).read_bytes()).hexdigest())
    save('completed' if not failed else 'error')
    app.close(exit_code=1 if failed else 0)
