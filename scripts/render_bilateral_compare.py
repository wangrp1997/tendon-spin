"""Render left student and right teacher as two independent floating units."""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def matrix_from_pose(pose, Gf):
    x, y, z, w = map(float, pose[3:7])
    matrix = Gf.Matrix4d().SetRotate(
        Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    matrix.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
    return matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--left-track', type=Path, required=True)
    parser.add_argument('--right-track', type=Path, required=True)
    parser.add_argument('--scene-config', type=Path, required=True)
    parser.add_argument('--object-separation', type=float, default=.72)
    parser.add_argument('--case-index', type=int, default=9)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    left_path = args.left_track if args.left_track.is_absolute() \
        else ROOT/args.left_track
    right_path = args.right_track if args.right_track.is_absolute() \
        else ROOT/args.right_track
    scene_path = args.scene_config if args.scene_config.is_absolute() \
        else ROOT/args.scene_config
    with np.load(left_path, allow_pickle=False) as saved:
        left_track = {name: saved[name].copy() for name in saved.files}
    with np.load(right_path, allow_pickle=False) as saved:
        right_track = {name: saved[name].copy() for name in saved.files}
    if left_track['body_pose'].shape != right_track['body_pose'].shape:
        raise ValueError('Left/right pose track shapes differ')
    if not {'body_names', 'diameter_m', 'height_m'} <= set(right_track):
        raise ValueError('Right track lacks geometry or body names')

    from isaaclab.app import AppLauncher
    app = AppLauncher(
        headless=True, enable_cameras=True, device='cuda:0').app
    status = dict(status='running', phase='starting',
                  case_index=args.case_index)
    try:
        import torch
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab_physx.physics import PhysxCfg
        from tendonspin.physics.hand_profiles import resolve_hand_profile
        from tendonspin.physics.learning_scene import load_scene_configuration
        from tendonspin.physics.object_geometry import author_primitive
        from tendonspin.rl.native_video import NativePolicyVideo

        profile_left = resolve_hand_profile(ROOT, 'left')
        profile_right = resolve_hand_profile(ROOT, 'right')
        scene = load_scene_configuration(scene_path)
        geometry = deepcopy(scene['geometry'])
        geometry['diameter_m'] = float(right_track['diameter_m'])
        geometry['height_m'] = float(right_track['height_m'])
        geometry['dimensions_m'] = [
            geometry['diameter_m'], geometry['diameter_m'],
            geometry['height_m']]

        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=.0005, device='cuda:0', visualizer_cfgs=[],
                physics=PhysxCfg()))
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        left_ref = UsdGeom.Xform.Define(stage, '/World/LeftHand')
        left_ref.GetPrim().GetReferences().AddReference(
            str(profile_left.asset_path))
        right_ref = UsdGeom.Xform.Define(stage, '/World/RightHand')
        right_ref.GetPrim().GetReferences().AddReference(
            str(profile_right.asset_path))

        def visual_only(prefix):
            prims = [prim for prim in stage.Traverse()
                     if str(prim.GetPath()).startswith(prefix)]
            for prim in prims:
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            for prim in prims:
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(
                        prim).CreateRigidBodyEnabledAttr(False)
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(
                        prim).CreateCollisionEnabledAttr(False)
                if prim.IsA(UsdPhysics.Joint):
                    UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)

        stage.Load()
        visual_only('/World/LeftHand')
        visual_only('/World/RightHand')

        def material(path, color, roughness=.55):
            mat = UsdShade.Material.Define(stage, path)
            surface = UsdShade.Shader.Define(stage, path+'/Surface')
            surface.CreateIdAttr('UsdPreviewSurface')
            surface.CreateInput(
                'diffuseColor', Sdf.ValueTypeNames.Color3f).Set(color)
            surface.CreateInput(
                'roughness', Sdf.ValueTypeNames.Float).Set(roughness)
            mat.CreateSurfaceOutput().ConnectToSource(
                surface.ConnectableAPI(), 'surface')
            return mat

        object_roots = {}
        for side in ('left', 'right'):
            root = UsdGeom.Xform.Define(stage, f'/World/{side.title()}Object')
            object_geometry = author_primitive(
                stage, f'/World/{side.title()}Object/Geometry', geometry)
            UsdShade.MaterialBindingAPI.Apply(object_geometry).Bind(
                material(
                    f'/World/Looks/{side.title()}Object',
                    (.92, .93, .95), .48))
            marker = UsdGeom.Mesh.Define(
                stage, f'/World/{side.title()}Object/RotationMarker')
            radius = geometry['diameter_m']/2*.93
            half_height = geometry['height_m']/2
            points, counts, indices = [], [], []
            for z in (-half_height-4e-5, half_height+4e-5):
                center = len(points)
                points.append((0., 0., z))
                arc = np.linspace(0., .9, 10)
                points.extend(
                    (radius*np.cos(angle), radius*np.sin(angle), z)
                    for angle in arc)
                for index in range(len(arc)-1):
                    counts.append(3)
                    indices.extend((center, center+index+1, center+index+2))
            marker.CreatePointsAttr(points)
            marker.CreateFaceVertexCountsAttr(counts)
            marker.CreateFaceVertexIndicesAttr(indices)
            marker.CreateDoubleSidedAttr(True)
            UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(
                material(
                    f'/World/Looks/{side.title()}Marker',
                    (.015, .018, .022), .3))
            object_roots[side] = root.GetPrim()

        class BilateralVideo(NativePolicyVideo):
            def setup(self, physics):
                light = sim_utils.DomeLightCfg(
                    intensity=950., color=(.75, .80, .90))
                light.func('/World/DomeLight', light)
                key = sim_utils.DistantLightCfg(
                    intensity=900., color=(1., .94, .85))
                key.func('/World/KeyLight', key)
                self.camera = Camera(CameraCfg(
                    prim_path='/World/Camera',
                    width=960, height=540, update_period=0.,
                    background_color=(.035, .045, .065),
                    data_types=['rgb'],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24., horizontal_aperture=24.,
                        clipping_range=(.01, 10.))))

        center_z = float(
            (left_track['object_state'][0, 2] +
             right_track['object_state'][0, 2])/2)
        center = torch.tensor([[0., 0., center_z]])
        physics = SimpleNamespace(
            sim=sim, dt=.0005, center=center)
        video = BilateralVideo(
            out, 0,
            title='右手教师 / 左手纯本体学生 / 两个独立物体悬空回放',
            mode_label='左学生，右教师；各自操作一个物体')
        video.setup(physics)
        sim.reset()
        video.attach(physics)
        video.camera.set_world_poses_from_view(
            eyes=[np.array([0., -1.18, center_z+.36])],
            targets=[np.array([0., 0., center_z])])

        def body_prims(prefix):
            result = {}
            for prim in stage.Traverse():
                if (str(prim.GetPath()).startswith(prefix) and
                        prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                    result[prim.GetName()] = prim
            return result

        left_bodies = body_prims('/World/LeftHand')
        right_bodies = body_prims('/World/RightHand')
        right_names = [str(name) for name in right_track['body_names']]
        left_names = profile_left.runtime_names(right_names)
        if len(left_bodies) != 24 or len(right_bodies) != 24:
            raise ValueError(
                f'Expected 24+24 rigid bodies, got '
                f'{len(left_bodies)}+{len(right_bodies)}')
        if not set(left_names) <= set(left_bodies) or \
                not set(right_names) <= set(right_bodies):
            raise ValueError('Pose-track body names differ from CAD')
        body_ops = {
            **{name: UsdGeom.Xformable(prim).MakeMatrixXform()
               for name, prim in left_bodies.items()},
            **{name: UsdGeom.Xformable(prim).MakeMatrixXform()
               for name, prim in right_bodies.items()}}
        object_ops = {
            side: UsdGeom.Xformable(root).MakeMatrixXform()
            for side, root in object_roots.items()}
        left_rows = {name: index for index, name in enumerate(left_names)}
        right_rows = {name: index for index, name in enumerate(right_names)}
        cache = UsdGeom.XformCache()
        left_pivot = left_track['object_state'][0, :3].astype(float)
        right_pivot = right_track['object_state'][0, :3].astype(float)
        left_yaw = Rotation.from_euler('z', 180., degrees=True)
        right_yaw = Rotation.identity()
        left_target = np.array(
            [-args.object_separation/2, 0., center_z])
        right_target = np.array(
            [args.object_separation/2, 0., center_z])
        expected_world = {}
        expected_objects = {}
        max_pose_error = 0.
        max_object_error = 0.

        def transform_pose(pose, pivot, yaw, target):
            result = np.asarray(pose, dtype=float).copy()
            result[:3] = target + yaw.apply(result[:3]-pivot)
            result[3:7] = (
                yaw*Rotation.from_quat(result[3:7])).as_quat()
            return result

        def place(left_record, right_record):
            expected_world.clear()
            expected_objects.clear()
            for name in left_names:
                pose = transform_pose(
                    left_record['body_pose'][left_rows[name]],
                    left_pivot, left_yaw, left_target)
                world = matrix_from_pose(pose, Gf)
                expected_world[name] = world
                cache.Clear()
                parent = cache.GetLocalToWorldTransform(
                    left_bodies[name].GetParent())
                body_ops[name].Set(world*parent.GetInverse())
            for name in right_names:
                pose = transform_pose(
                    right_record['body_pose'][right_rows[name]],
                    right_pivot, right_yaw, right_target)
                world = matrix_from_pose(pose, Gf)
                expected_world[name] = world
                cache.Clear()
                parent = cache.GetLocalToWorldTransform(
                    right_bodies[name].GetParent())
                body_ops[name].Set(world*parent.GetInverse())
            for side, record, pivot, yaw, target in (
                    ('left', left_record, left_pivot,
                     left_yaw, left_target),
                    ('right', right_record, right_pivot,
                     right_yaw, right_target)):
                matrix = matrix_from_pose(
                    transform_pose(
                        record['object_state'][:7],
                        pivot, yaw, target), Gf)
                expected_objects[side] = matrix
                object_ops[side].Set(matrix)

        def check_poses():
            nonlocal max_pose_error, max_object_error
            for name, expected in expected_world.items():
                prim = left_bodies[name] if name in left_bodies \
                    else right_bodies[name]
                cache.Clear()
                actual = cache.GetLocalToWorldTransform(prim)
                max_pose_error = max(
                    max_pose_error,
                    float(np.max(np.abs(
                        np.asarray(actual)-np.asarray(expected)))))
            for side, expected in expected_objects.items():
                cache.Clear()
                actual = cache.GetLocalToWorldTransform(
                    object_roots[side])
                max_object_error = max(
                    max_object_error,
                    float(np.max(np.abs(
                        np.asarray(actual)-np.asarray(expected)))))
            if max_pose_error > 1e-5 or max_object_error > 1e-5:
                raise ValueError(
                    'Visual pose was overwritten: '
                    f'body={max_pose_error:.6g}, '
                    f'object={max_object_error:.6g}')

        count = len(left_track['body_pose'])
        first_frame = None
        for index in range(count):
            place(
                {name: left_track[name][index]
                 for name in ('body_pose', 'object_state')},
                {name: right_track[name][index]
                 for name in ('body_pose', 'object_state')})
            left_net = float(left_track['net_deg'][index])
            right_net = float(right_track['net_deg'][index])
            detail = (
                f'仿真 {index/20:.2f}s | 左学生 {left_net:+.1f}° | '
                f'右教师 {right_net:+.1f}° | 两个独立物体')
            video.capture(
                index/20., left_net, 0.,
                initial=index == 0,
                reason='time limit' if index == count-1 else None,
                detail=detail)
            check_poses()
            if index == 0:
                first_frame = np.asarray(video.last)[76:616].copy()
                video.last.save(out/'initial.png')
        last_frame = np.asarray(video.last)[76:616]
        delta = np.abs(
            first_frame.astype(np.int16)-last_frame.astype(np.int16))
        changed = int(np.count_nonzero(np.max(delta, axis=2) > 20))
        if changed < 2000:
            raise ValueError(
                f'Rendered action is visually static: {changed} pixels')
        cache.Clear()
        left_object_center = np.asarray(
            cache.GetLocalToWorldTransform(object_roots['left']))[3, :3]
        cache.Clear()
        right_object_center = np.asarray(
            cache.GetLocalToWorldTransform(object_roots['right']))[3, :3]
        cache.Clear()
        left_base = np.asarray(
            cache.GetLocalToWorldTransform(
                left_bodies['left_fabase']))[3, :3]
        cache.Clear()
        right_base = np.asarray(
            cache.GetLocalToWorldTransform(
                right_bodies['fabase']))[3, :3]
        object_distance = float(np.linalg.norm(
            left_object_center-right_object_center))
        hand_distance = float(np.linalg.norm(left_base-right_base))
        if object_distance < args.object_separation*.9:
            raise ValueError('The two operated objects overlap')
        video_info = video.close()
        video_path = out/'isaac_native.mp4'
        status.update(
            status='completed', phase='completed',
            video=str(video_path), video_sha256=digest(video_path),
            video_info=video_info,
            left_track=str(left_path),
            left_track_sha256=digest(left_path),
            right_track=str(right_path),
            right_track_sha256=digest(right_path),
            left_net_deg=float(left_track['net_deg'][-1]),
            right_net_deg=float(right_track['net_deg'][-1]),
            rendered_body_pose_max_error_m=max_pose_error,
            rendered_object_pose_max_error_m=max_object_error,
            rendered_hand_body_count=dict(
                left=len(left_bodies), right=len(right_bodies)),
            rendered_hand_root_separation_m=hand_distance,
            rendered_object_separation_m=object_distance,
            rendered_camera_changed_pixels_gt20=changed,
            floor=False,
            layout='left student and right teacher face inward; '
                   'each hand holds a separate floating object')
    except BaseException as error:
        import traceback
        status.update(
            status='error', error=str(error),
            traceback=traceback.format_exc())
        raise
    finally:
        (out/'status.json').write_text(
            json.dumps(status, indent=2)+'\n')
        app.close(exit_code=0 if status['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
