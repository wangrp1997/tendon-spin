# Rendering/encoding interface reused from tendonspin/rl/video.py, itself adapted
# from botyard-inhand/research/learning/video.py at 204d197a9606fb7266e884f3b2e6110195be01cb.
# Changes: draw stored Isaac world-body XYZW poses directly with packaged CAD;
# never integrate or infer failed PhysX body poses from native joint kinematics.
"""Make a labeled, discrete pose replay of a recorded Isaac transfer diagnostic."""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from tendonspin.physics.native import verify_runtime
from tendonspin.rl.config import ROOT, digest, write

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def checked_source(entry):
    path = ROOT / entry['path']
    assert digest(path) == entry['sha256'], str(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    phase = json.loads(args.phase.read_text())
    assert phase['controller'] == 'boya_source_pd_hold_v2'
    assert phase['episode_resets'] == phase['controller_switches'] == 0
    assert phase['physics_steps'] == 6 and not phase['benchmark_validated']
    names = phase['body_names']
    assert len(names) == len(set(names)) == 24
    with np.load(checked_source(phase['initial_state']), allow_pickle=False) as raw:
        initial = {k: raw[k].copy() for k in raw.files}
    with np.load(checked_source(phase['execution']), allow_pickle=False) as raw:
        trajectory = {k: raw[k].copy() for k in raw.files}
    assert trajectory['body_pose'].shape == (6, 24, 7)
    assert not np.any(trajectory['action'])
    assert np.array_equal(trajectory['commands'], np.broadcast_to(initial['commands'], (6, 18)))
    poses = np.concatenate((initial['body_pose'][None], trajectory['body_pose'])).astype(float)
    objects = np.vstack((initial['object_state'], trajectory['object_state'])).astype(float)
    times = np.r_[0., trajectory['elapsed_s']]
    dt = float(trajectory['physics_dt'])
    np.testing.assert_allclose(times, np.arange(7) * dt, rtol=0, atol=1e-12)
    assert np.isfinite(poses).all() and np.isfinite(objects).all()
    drift = np.r_[0., trajectory['drift_mm']]
    # The archived drift uses the declared source center (before float32 transfer).
    contract = json.loads(checked_source(phase['contract']).read_text())
    np.testing.assert_allclose(drift[1:], np.linalg.norm(objects[1:, :3] - contract['object']['pos'], axis=1) * 1000,
                               rtol=0, atol=1e-6)
    speed = np.r_[np.abs(initial['joint_vel']).max(), np.abs(trajectory['joint_vel']).max(axis=1)]
    forces = np.linalg.norm(trajectory['normal_force_matrix_w'], axis=-1).max(axis=1)

    verify_runtime()
    manifest = json.loads((ROOT / 'assets/grasp/manifest.json').read_text())
    model_path = ROOT / 'assets/grasp/scene.xml'
    assert digest(model_path) == manifest['scene_sha256']
    for name, expected in manifest['meshes'].items():
        assert digest(ROOT / 'assets/grasp' / name) == expected
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    # This initializes only the static visualization background. No mj_forward,
    # mj_step or acceleration/contact solver is called anywhere in this renderer.
    mujoco.mj_kinematics(model, data)
    body_ids = [model.body(name).id for name in names] + [model.body('cylinder').id]
    geometry = [g for g in range(model.ngeom) if model.geom_bodyid[g] in body_ids]
    assert len(geometry) == 25
    assert {int(model.geom_bodyid[g]) for g in geometry} == set(body_ids)
    local_rotations = Rotation.from_quat(model.geom_quat[:, [1, 2, 3, 0]]).as_matrix()
    local_positions = model.geom_pos.copy()
    native_positions = data.geom_xpos.copy()
    native_rotations = data.geom_xmat.reshape(-1, 3, 3).copy()
    composition_error = 0.
    declared_local_position_difference_m = 0.
    for g in geometry:
        b = model.geom_bodyid[g]
        r = data.xmat[b].reshape(3, 3)
        # Preserve compiled sameframe optimizations (one CAD center is within
        # 7 nm of an inertial frame and the native renderer uses that frame).
        local_positions[g] = r.T @ (native_positions[g] - data.xpos[b])
        local_rotations[g] = r.T @ native_rotations[g]
        declared_local_position_difference_m = max(declared_local_position_difference_m,
            float(np.max(np.abs(local_positions[g] - model.geom_pos[g]))))
        composition_error = max(composition_error,
            float(np.max(np.abs(native_positions[g] - (data.xpos[b] + r @ local_positions[g])))),
            float(np.max(np.abs(native_rotations[g] - r @ local_rotations[g]))))
    assert composition_error < 1e-12

    # Color changes aid readability only; original CAD geometry and local
    # transforms are retained. Orange cylinder, blue thumb, neutral other fingers.
    model.geom_rgba[geometry] = [.76, .79, .84, 1.]
    for g in geometry:
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]).startswith('th'):
            model.geom_rgba[g] = [.32, .58, .83, 1.]
    model.geom_rgba[model.geom('cylinder_geom').id] = [1., .57, .16, 1.]
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    cameras = []
    for azimuth in (140, 40):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = objects[0, :3] + [.015, 0., -.025]
        camera.distance = .265
        camera.azimuth = azimuth
        camera.elevation = -20
        cameras.append(camera)
    fonts = {size: ImageFont.truetype(FONT, size) for size in (20, 22, 26, 28)}
    fps, width, height = 30, 1280, 768
    # Seven samples are held without interpolating unrecorded physical states.
    holds = [2., 1., 1., 1., 1., 1., 3.]
    counts = [int(fps * duration) for duration in holds]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
               '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-', '-an',
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.out)]
    reconstruction_error = 0.
    images = []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                for k in range(7):
                    world_poses = np.vstack((poses[k], objects[k, :7]))
                    rotations = Rotation.from_quat(world_poses[:, 3:7]).as_matrix()
                    for j, b in enumerate(body_ids):
                        data.xpos[b] = world_poses[j, :3]
                        data.xmat[b] = rotations[j].ravel()
                        data.xquat[b] = world_poses[j, [6, 3, 4, 5]]
                    for g in geometry:
                        b = model.geom_bodyid[g]
                        r = data.xmat[b].reshape(3, 3)
                        data.geom_xpos[g] = data.xpos[b] + r @ local_positions[g]
                        data.geom_xmat[g] = (r @ local_rotations[g]).ravel()
                        recovered_r = data.geom_xmat[g].reshape(3, 3) @ local_rotations[g].T
                        recovered_p = data.geom_xpos[g] - recovered_r @ local_positions[g]
                        reconstruction_error = max(reconstruction_error,
                            float(np.max(np.abs(recovered_r - r))),
                            float(np.max(np.abs(recovered_p - data.xpos[b]))))
                    frame = Image.new('RGB', (width, height), (18, 23, 31))
                    draw = ImageDraw.Draw(frame)
                    draw.text((22, 12), '伯牙手 / Isaac 最新失败过程 / 已保存姿态回放', font=fonts[28], fill='white')
                    draw.text((22, 56), '只要求保持原来的夹持位置，尚未开始转动控制。橙色是圆柱，蓝色是拇指。',
                              font=fonts[22], fill=(209, 218, 231))
                    status = '初始夹持' if k == 0 else ('异常终止：圆柱漂移超限' if k == 6 else '夹持过程开始出现异常运动')
                    color = (255, 144, 130) if k == 6 else (255, 226, 155)
                    draw.text((22, 99), f'实际仿真时间 {times[k] * 1000:.1f} ms  /  第 {k} 步  /  {status}',
                              font=fonts[26], fill=color)
                    for j, camera in enumerate(cameras):
                        renderer.update_scene(data, camera, scene_option=options)
                        frame.paste(Image.fromarray(renderer.render()), (j * 640, 136))
                        draw.rectangle((j * 640 + 12, 146, j * 640 + 120, 180), fill=(18, 23, 31))
                        draw.text((j * 640 + 20, 148), f'视角 {j + 1}', font=fonts[22], fill='white')
                    draw.text((22, 628), f'圆柱漂移 {drift[k]:.3f} mm     关节最大速度 {speed[k]:.2f} rad/s     夹持目标：始终不变',
                              font=fonts[26], fill=color)
                    draw.text((22, 671), '实际执行仅 3 毫秒；画面逐帧停留以便观察。没有新增物理仿真或训练。',
                              font=fonts[22], fill=(209, 218, 231))
                    for j in range(7):
                        x = 220 + j * 145
                        fill = color if j == k else (77, 87, 102)
                        draw.ellipse((x, 720, x + 12, 732), fill=fill)
                        draw.text((x - 19, 739), f'{times[j] * 1000:.1f}ms', font=fonts[20], fill=fill)
                    images.append(frame)
                    for _ in range(counts[k]):
                        encoder.stdin.write(np.asarray(frame).tobytes())
        finally:
            encoder.stdin.close()
        assert encoder.wait() == 0
    assert reconstruction_error < 1e-12
    images[0].save(args.out.with_name(args.out.stem + '_initial.png'))
    images[-1].save(args.out.with_name(args.out.stem + '_final.png'))
    comparison = Image.new('RGB', (width, height * 2))
    comparison.paste(images[0], (0, 0))
    comparison.paste(images[-1], (0, height))
    comparison_path = args.out.with_name(args.out.stem + '_comparison.png')
    comparison.save(comparison_path)
    write(args.record, dict(
        kind='Recorded Isaac physical-v2 world-body pose visualization',
        controller=phase['controller'], phase_record=dict(path=str(args.phase.resolve().relative_to(ROOT)), sha256=digest(args.phase)),
        execution=phase['execution'], initial_state=phase['initial_state'], contract=phase['contract'],
        renderer=dict(path=str(Path(__file__).resolve().relative_to(ROOT)), sha256=digest(Path(__file__))),
        geometry_source=dict(path=str(model_path.relative_to(ROOT)), sha256=digest(model_path)),
        video=dict(path=str(args.out.resolve().relative_to(ROOT)), sha256=digest(args.out)),
        comparison=dict(path=str(comparison_path.resolve().relative_to(ROOT)), sha256=digest(comparison_path)),
        new_physics_steps=0, physical_reexecution=False, rendering_only=True,
        rendering_engine='MuJoCo 3.13 CAD drawing only; dynamics are recorded PhysX',
        pose_source='stored world-body positions and XYZW quaternions, not native FK of joint positions',
        interpolation=False, unique_recorded_poses=7, includes_initial_state=True, includes_terminal_failure=True,
        physics_dt_s=dt, actual_execution_s=float(times[-1]), recorded_times_ms=(times * 1000).tolist(),
        held_frames_per_pose=counts, fps=fps, video_frames=sum(counts), video_duration_s=sum(holds),
        body_names=names, rendered_dynamic_geoms=25, model_geometry_composition_max_error=composition_error,
        compiled_sameframe_vs_declared_local_position_difference_m=declared_local_position_difference_m,
        saved_body_pose_geometry_roundtrip_max_error=reconstruction_error,
        drift_mm=drift.tolist(), max_joint_speed_rad_s=speed.tolist(), max_link_normal_N=forces.tolist(),
        terminal_reason=phase['stop_reason'], benchmark_validated=False,
        captions='Chinese; actual ms, unchanged targets, failure and discrete slowed replay explicitly labeled',
    ))
    print(json.dumps(dict(video=str(args.out), record=str(args.record), physical_steps=0,
                          frames=sum(counts), duration_s=sum(holds), roundtrip_error=reconstruction_error)))


if __name__ == '__main__':
    main()
