# References: Sharpa RL Lab 5accf024d376685eaa17da7aa4614498217eab4d
# sharpa_wave_env.py::_setup_scene (ground + dome light); TendonSpin native_video.py
# (camera/encoding), render_isaac_pose_replay.py (recorded world-body pose rendering).
# This is native Isaac RTX rendering of archived PhysX poses, without physics stepping.
# Delivery update: optional normal-speed replay and opt-in per-frame image export;
# the archived episode, physics, controller and metrics are never reexecuted.
# Object geometry: TendonSpin object_geometry.py; replay uses the episode's
# declared dimensions/shape, with the original Boya profile for older records.
"""Render a readable floor/grid studio from an already executed Boya episode."""
import argparse,hashlib,json,os,time
from pathlib import Path
from types import SimpleNamespace
os.environ['ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION']='0'
parser=argparse.ArgumentParser()
parser.add_argument('--record',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--playback-speed',type=float,choices=(.2,1.),default=.2)
parser.add_argument('--save-frames',action='store_true',help='Also export every sampled frame as PNG')
args=parser.parse_args();root=Path(__file__).resolve().parents[1]
source=args.record.resolve();episode=json.loads(source.read_text())
from tendonspin.physics.object_geometry import resolve_object_geometry,author_primitive
geometry=episode.get('object_geometry') or resolve_object_geometry(root)
out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
manifest=dict(motion_source=str(source),motion_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    renderer='Isaac Sim RTX; archived measured world-body poses',new_training_actions=0,
    new_physics_steps=0,controller_reexecuted=False,interpolation=False,
    source_actual_s=episode['actual_s'],source_net_deg=episode['net_deg'],source_stop_reason=episode['stop_reason'],
    floor_collision=False,style='dark slate checker floor, dome/key light, orange object and painted marker',
    object_geometry=geometry,
    geometry_source_sha256=hashlib.sha256((root/'tendonspin/physics/object_geometry.py').read_bytes()).hexdigest())
(out/'render_boya_isaac_studio.py').write_bytes(Path(__file__).read_bytes())
(out/'object_geometry.py').write_bytes((root/'tendonspin/physics/object_geometry.py').read_bytes())
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,enable_cameras=True,device='cuda:0').app
video=None;failed=False;started=time.monotonic()
try:
    import numpy as np
    import omni.usd
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf,Vt
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab.sensors import Camera,CameraCfg
    from tendonspin.rl.native_video import NativePolicyVideo
    from tendonspin.physics.isaac_parallel import tensor
    from PIL import Image,ImageDraw
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.0005,device='cuda:0',
        visualizer_cfgs=[],physics=PhysxCfg()))
    stage=omni.usd.get_context().get_stage()
    imported=json.loads((root/'docs/data/isaac_boya_import_phases.json').read_text())
    asset=root/imported['usd']['path']
    manifest['hand_asset']=dict(path=str(asset),sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    hand=stage.DefinePrim('/World/Hand','Xform');hand.GetReferences().AddReference(str(asset))
    bodies={p.GetName():p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    if set(bodies)!=set(episode['body_names']):raise ValueError('Recorded body names do not match CAD')
    # The replay stage is visual only. All recorded rigid bodies are driven directly
    # by measured world transforms; no articulated dynamics or contacts are executed.
    for p in list(stage.Traverse()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
        if p.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
        if p.IsA(UsdPhysics.Joint):UsdPhysics.Joint(p).CreateJointEnabledAttr(False)
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):p.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    def material(path,color,roughness=.6):
        m=UsdShade.Material.Define(stage,path);s=UsdShade.Shader.Define(stage,path+'/Surface')
        s.CreateIdAttr('UsdPreviewSurface');s.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(color)
        s.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface')
        return m
    object_root=UsdGeom.Xform.Define(stage,'/World/Cylinder')
    object_prim=author_primitive(stage,'/World/Cylinder/Geometry',geometry)
    orange=material('/World/Looks/Orange',(0.95,.20,.025),.38)
    UsdShade.MaterialBindingAPI.Apply(object_prim).Bind(orange)
    # A thin painted radial wedge makes axial rotation visible; purely visual.
    marker=UsdGeom.Mesh.Define(stage,'/World/Cylinder/PaintedMarker')
    rad=min(geometry['dimensions_m'][:2])/2*.94;z=geometry['dimensions_m'][2]/2+.00003
    if geometry['shape']=='sphere':
        # Small surface patch near the pole, projected onto the sphere.
        radius=geometry['diameter_m']/2
        points=[(0,0,radius+.00003)]
        for angle in (0.,.4):
            x=.35*radius*np.cos(angle);y=.35*radius*np.sin(angle)
            points.append((x,y,float(np.sqrt(radius*radius-x*x-y*y))+.00003))
    else:points=[(0,0,z),(rad,0,z),(rad*np.cos(.22),rad*np.sin(.22),z)]
    marker.CreatePointsAttr(points)
    marker.CreateFaceVertexCountsAttr([3]);marker.CreateFaceVertexIndicesAttr([0,1,2]);marker.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(material('/World/Looks/Marker',(.97,.98,1.)))
    floor=UsdGeom.Mesh.Define(stage,'/World/ReferenceFloor');points=[];indices=[];colors=[]
    for i in range(60):
        for j in range(60):
            x=-1.5+i*.05;y=-1.5+j*.05;n=len(points)
            points.extend([(x,y,-.10),(x+.05,y,-.10),(x+.05,y+.05,-.10),(x,y+.05,-.10)])
            indices.extend([n,n+1,n+2,n+3]);colors.append((.075,.09,.115) if (i+j)%2 else (.105,.125,.155))
    floor.CreatePointsAttr(points);floor.CreateFaceVertexCountsAttr([4]*len(colors));floor.CreateFaceVertexIndicesAttr(indices)
    floor.CreateDoubleSidedAttr(True);floor.CreateSubdivisionSchemeAttr('none')
    floor.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(colors)
    # No CollisionAPI or rigid body is authored for this reference floor.
    floor_mat=material('/World/Looks/Floor',(.10,.12,.15),.95)
    reader=UsdShade.Shader.Define(stage,'/World/Looks/Floor/Color');reader.CreateIdAttr('UsdPrimvarReader_float3')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('displayColor');reader.CreateOutput('result',Sdf.ValueTypeNames.Float3)
    UsdShade.Shader(stage.GetPrimAtPath('/World/Looks/Floor/Surface')).GetInput('diffuseColor').ConnectToSource(reader.ConnectableAPI(),'result')
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(floor_mat)

    class StudioVideo(NativePolicyVideo):
        def setup(self,physics):
            light=sim_utils.DomeLightCfg(intensity=950.,color=(.75,.80,.90));light.func('/World/DomeLight',light)
            key=sim_utils.DistantLightCfg(intensity=900.,color=(1.,.94,.85));key.func('/World/KeyLight',key)
            self.camera=Camera(CameraCfg(prim_path='/World/Camera',width=960,height=540,update_period=0.,
                background_color=(.035,.045,.065),data_types=['rgb'],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=24.,clipping_range=(.01,10.))))
        def capture(self,elapsed,net,drift,*,initial=False,reason=None):
            self.physics.sim.render();self.camera.update(self.physics.dt,force_recompute=True)
            rgb=tensor(self.camera.data.output['rgb'])[0,:,:,:3].cpu().numpy().astype(np.uint8)
            frame=Image.new('RGB',(960,660),(18,23,31));frame.paste(Image.fromarray(rgb),(0,76))
            draw=ImageDraw.Draw(frame)
            draw.text((12,4),'伯牙手 / Isaac RTX 原生渲染 / 已执行物理轨迹回放',font=self.font,fill='white')
            label='初始状态定格' if initial else ('终止状态定格' if reason else ('原速播放' if args.playback_speed==1. else '0.2倍速慢放'))
            shape_name={'cylinder':'圆柱','sphere':'球','box':'长方体'}[geometry['shape']]
            dimensions='×'.join(f'{v*1000:g}' for v in geometry['dimensions_m'])
            draw.text((12,39),f'{shape_name} {dimensions}mm | 训练 {self.training_actions:,} | {label}',font=self.font,fill='white')
            text=f'实际仿真 {elapsed:.4f}s | 有效净转角 {net:+.2f}° | 漂移 {drift:.2f}mm'
            stops={'drift >5mm':'终止：漂移超过5毫米','time limit':'到达评估时限',
                   'object below Boya manipulation region':'物体低于操作区',
                   'object outside Boya lateral region':'物体超出操作区横向边界'}
            if reason:text+=' | '+stops.get(reason,reason)
            draw.text((12,622),text,font=self.font,fill='white')
            repeats=20 if initial else (40 if reason else round(1/args.playback_speed))
            for _ in range(repeats):self.encoder.stdin.write(np.asarray(frame).tobytes())
            self.frames+=repeats;self.last=frame
            self.samples.append(dict(physics_s=elapsed,valid_net_deg=net,drift_mm=drift,repeated_frames=repeats,phase=label))
            if args.save_frames or initial or reason:
                frame.save(out/('initial.png' if initial else ('final.png' if reason else f'frame_{len(self.samples):03d}.png')))
            if len(self.samples)%100==0:
                print(f'RENDER_PROGRESS physics_s={elapsed:.2f}/{episode["actual_s"]:.2f}',flush=True)

    with np.load(source.parent/'initial_state.npz') as d:initial={k:d[k].copy() for k in d.files}
    manifest['initial_state_sha256']=hashlib.sha256((source.parent/'initial_state.npz').read_bytes()).hexdigest()
    chunk_files=sorted(source.parent.glob('physics_*.npz'))
    chunks=[];manifest['trajectory_sources']=[]
    for file in chunk_files:
        with np.load(file) as d:chunks.append({k:d[k].copy() for k in ('body_pose','object_state','elapsed_s','net_angle_deg','drift_mm')})
        manifest['trajectory_sources'].append(dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    arrays={k:np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
    import torch
    physics=SimpleNamespace(sim=sim,dt=.0005,center=torch.as_tensor(initial['object_state'][:,:3]))
    video=StudioVideo(out,episode['training_actions_in_checkpoint']);video.setup(physics)
    body_rows={name:i for i,name in enumerate(episode['body_names'])}
    prim_order=sorted(bodies.values(),key=lambda p:p.GetPath().pathElementCount)
    ops={str(p.GetPath()):UsdGeom.Xformable(p).MakeMatrixXform() for p in prim_order}
    object_op=UsdGeom.Xformable(object_root).MakeMatrixXform()
    cache=UsdGeom.XformCache();max_pose_error=0.
    def matrix(pose):
        x,y,z,w=map(float,pose[3:7]);m=Gf.Matrix4d().SetRotate(Gf.Quatd(w,Gf.Vec3d(x,y,z)))
        m.SetTranslateOnly(Gf.Vec3d(*map(float,pose[:3])));return m
    def place(body_poses,obj_pose):
        global max_pose_error
        for prim in prim_order:
            cache.Clear();parent=cache.GetLocalToWorldTransform(prim.GetParent())
            world=matrix(body_poses[body_rows[prim.GetName()]])
            ops[str(prim.GetPath())].Set(world*parent.GetInverse())
            cache.Clear();err=np.abs(np.asarray(cache.GetLocalToWorldTransform(prim))-np.asarray(world)).max()
            max_pose_error=max(max_pose_error,float(err))
        object_op.Set(matrix(obj_pose))
    place(initial['body_pose'][0],initial['object_state'][0]);sim.reset();video.attach(physics)
    # Warm rendering only; no physics integration, controller or interpolation.
    for _ in range(4):sim.render()
    video.capture(0.,0.,float(initial['drift_mm'][0]),initial=True)
    indices=list(range(99,len(arrays['elapsed_s'])-1,100))+[len(arrays['elapsed_s'])-1]
    for index in indices:
        place(arrays['body_pose'][index],arrays['object_state'][index])
        for _ in range(2):sim.render()
        terminal=index==len(arrays['elapsed_s'])-1
        video.capture(float(arrays['elapsed_s'][index]),float(arrays['net_angle_deg'][index]),float(arrays['drift_mm'][index]),
                      reason=episode['stop_reason'] if terminal else None)
    manifest['max_authored_body_world_matrix_error']=max_pose_error
    manifest['floor_has_collision_api']=floor.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
except BaseException as error:
    failed=True
    import traceback
    manifest['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
finally:
    if video is not None:
        manifest['video']=video.close()
        if manifest['video'] is not None:manifest['video']['motion_playback_speed']=args.playback_speed
    if manifest.get('video',{}).get('encoder_exit_code',0):failed=True
    manifest.update(status='error' if failed else 'completed',wall_s=time.monotonic()-started)
    (out/'render.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True);app.close(exit_code=1 if failed else 0)
