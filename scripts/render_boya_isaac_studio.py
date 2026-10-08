# References: Sharpa RL Lab 5accf024d376685eaa17da7aa4614498217eab4d
# sharpa_wave_env.py::_setup_scene (ground + dome light); TendonSpin native_video.py
# (camera/encoding), render_isaac_pose_replay.py (recorded world-body pose rendering).
# This is native Isaac RTX rendering of archived PhysX poses, without physics stepping.
"""Render a readable floor/grid studio from an already executed Boya episode."""
import argparse,hashlib,json,os,time
from pathlib import Path
from types import SimpleNamespace
os.environ['ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION']='0'
parser=argparse.ArgumentParser()
parser.add_argument('--record',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();root=Path(__file__).resolve().parents[1]
source=args.record.resolve();episode=json.loads(source.read_text())
out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
manifest=dict(motion_source=str(source),motion_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    renderer='Isaac Sim RTX; archived measured world-body poses',new_training_actions=0,
    new_physics_steps=0,controller_reexecuted=False,interpolation=False,
    source_actual_s=episode['actual_s'],source_net_deg=episode['net_deg'],source_stop_reason=episode['stop_reason'],
    floor_collision=False,style='dark slate checker floor, dome/key light, orange cylinder and painted marker')
(out/'render_boya_isaac_studio.py').write_bytes(Path(__file__).read_bytes())
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
    cylinder=UsdGeom.Cylinder.Define(stage,'/World/Cylinder')
    contract=json.loads((root/'docs/data/boya_native_contract.json').read_text())['object']
    cylinder.CreateRadiusAttr(contract['diameter_m']/2);cylinder.CreateHeightAttr(contract['length_m']);cylinder.CreateAxisAttr('Z')
    orange=material('/World/Looks/Orange',(0.95,.20,.025),.38)
    UsdShade.MaterialBindingAPI.Apply(cylinder.GetPrim()).Bind(orange)
    # A thin painted radial wedge makes axial rotation visible; purely visual.
    marker=UsdGeom.Mesh.Define(stage,'/World/Cylinder/PaintedMarker')
    rad=contract['diameter_m']/2*.94;z=contract['length_m']/2+.00003
    marker.CreatePointsAttr([(0,0,z),(rad,0,z),(rad*np.cos(.22),rad*np.sin(.22),z)])
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
            label='初始状态定格' if initial else ('终止状态定格' if reason else '0.2倍速慢放')
            draw.text((12,39),f'训练 {self.training_actions:,} 次动作 | {label} | 地面仅作视觉参考',font=self.font,fill='white')
            text=f'实际仿真 {elapsed:.4f}s | 有效净转角 {net:+.2f}° | 漂移 {drift:.2f}mm'
            if reason:text+=' | 终止：漂移超过5毫米' if reason=='drift >5mm' else ' | '+reason
            draw.text((12,622),text,font=self.font,fill='white')
            repeats=20 if initial else (40 if reason else 5)
            for _ in range(repeats):self.encoder.stdin.write(np.asarray(frame).tobytes())
            self.frames+=repeats;self.last=frame
            self.samples.append(dict(physics_s=elapsed,valid_net_deg=net,drift_mm=drift,repeated_frames=repeats,phase=label))
            frame.save(out/('initial.png' if initial else ('final.png' if reason else f'frame_{len(self.samples):03d}.png')))

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
    object_op=UsdGeom.Xformable(cylinder).MakeMatrixXform()
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
    if video is not None:manifest['video']=video.close()
    if manifest.get('video',{}).get('encoder_exit_code',0):failed=True
    manifest.update(status='error' if failed else 'completed',wall_s=time.monotonic()-started)
    (out/'render.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True);app.close(exit_code=1 if failed else 0)
