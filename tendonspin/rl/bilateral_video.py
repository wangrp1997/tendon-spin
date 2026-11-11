"""One camera directly records the live bilateral physics scene at 1x speed."""
from pathlib import Path
import subprocess
import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from pxr import Sdf, UsdGeom, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from tendonspin.physics.isaac_parallel import tensor


class BilateralCamera:
    def __init__(self, physics, out, fps=40, view='oblique', preview=False):
        self.physics, self.out, self.fps = physics, Path(out), fps
        self.preview, self.encoder, self.frames = preview, None, 0
        self.width, self.height, self.header, self.footer = 1920,1080,92,56
        self.font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',28)
        self.small = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',24)
        stage = physics.sim.stage
        def material(path, color, roughness=.6):
            mat = UsdShade.Material.Define(stage,path)
            shader = UsdShade.Shader.Define(stage,path+'/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            return mat
        white = material('/World/Looks/White',(.92,.93,.95),.45)
        black = material('/World/Looks/Black',(.015,.018,.025),.4)
        radius, h = physics.geometry['diameter_m']/2, physics.geometry['height_m']/2
        for side in ('Left','Right'):
            path = f'/World/envs/env_0/{side}Object'
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(white,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            points, counts, indices = [], [], []
            for z in (-h-.00005,h+.00005):
                center = len(points)
                points.append((0.,0.,z))
                arc = np.linspace(0.,.48,12)
                points.extend((radius*.96*np.cos(a),radius*.96*np.sin(a),z) for a in arc)
                for i in range(len(arc)-1):
                    counts.append(3);indices.extend((center,center+i+1,center+i+2))
            for a in np.linspace(0.,2*np.pi,4,endpoint=False):
                start=len(points)
                for z,da in ((-h,-.035),(-h,.035),(h,.035),(h,-.035)):
                    points.append(((radius+.00005)*np.cos(a+da),(radius+.00005)*np.sin(a+da),z))
                counts.append(4);indices.extend(range(start,start+4))
            marker = UsdGeom.Mesh.Define(stage,path+'/Paint')
            marker.CreatePointsAttr(points);marker.CreateFaceVertexCountsAttr(counts)
            marker.CreateFaceVertexIndicesAttr(indices);marker.CreateDoubleSidedAttr(True)
            marker.CreateSubdivisionSchemeAttr('none')
            # Bind the white material on actual object geometry instead of the
            # rigid parent, so the black child paint keeps its own material.
            UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path)).UnbindDirectBinding()
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith(path+'/geometry') and UsdGeom.Gprim(prim):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(white,
                        bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(black)
        floor=UsdGeom.Mesh.Define(stage,'/World/ReferenceFloor')
        points,indices,colors=[],[],[]
        for i in range(40):
            for j in range(40):
                x,y=(i-20)*.05,(j-20)*.05;n=len(points)
                points.extend(((x,y,-.12),(x+.05,y,-.12),(x+.05,y+.05,-.12),(x,y+.05,-.12)))
                indices.extend(range(n,n+4));colors.append((.065,.078,.10) if (i+j)%2 else (.085,.10,.125))
        floor.CreatePointsAttr(points);floor.CreateFaceVertexCountsAttr([4]*len(colors))
        floor.CreateFaceVertexIndicesAttr(indices);floor.CreateSubdivisionSchemeAttr('none')
        floor.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(colors)
        floor_mat=material('/World/Looks/Floor',(.08,.10,.13),.95)
        reader=UsdShade.Shader.Define(stage,'/World/Looks/Floor/Color')
        reader.CreateIdAttr('UsdPrimvarReader_float3')
        reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('displayColor')
        reader.CreateOutput('result',Sdf.ValueTypeNames.Float3)
        UsdShade.Shader(stage.GetPrimAtPath('/World/Looks/Floor/Surface')).GetInput('diffuseColor').ConnectToSource(reader.ConnectableAPI(),'result')
        UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(floor_mat)
        light=sim_utils.DomeLightCfg(intensity=1100.,color=(.80,.85,.95));light.func('/World/DomeLight',light)
        key=sim_utils.DistantLightCfg(intensity=1200.,color=(1.,.94,.86));key.func('/World/KeyLight',key)
        self.camera=Camera(CameraCfg(prim_path='/World/Camera',width=self.width,
            height=self.height-self.header-self.footer,update_period=0.,data_types=['rgb'],
            background_color=(.035,.045,.065),spawn=sim_utils.PinholeCameraCfg(
                focal_length=36.,horizontal_aperture=36.,clipping_range=(.01,10.))))
        self.view=view

    def attach(self):
        z=float(self.physics.sides['right'].initial_object[2])
        focus=np.array([0.,0.,z-.055])
        eye=focus+(np.array([.035,-.64,.57]) if self.view=='oblique' else np.array([0.,-.001,.82]))
        self.camera.set_world_poses_from_view(eyes=[eye],targets=[focus])
        self.camera_configuration=dict(eye=eye.tolist(),target=focus.tolist(),view=self.view)
        if not self.preview:
            self.encoder=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-n','-loglevel','error',
                '-f','rawvideo','-pix_fmt','rgb24','-s',f'{self.width}x{self.height}','-r',str(self.fps),
                '-i','-','-an','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(self.out/'bilateral_same_scene_1x.mp4')],stdin=subprocess.PIPE)
        for _ in range(4):self.physics.sim.render()

    def capture(self, elapsed, angles, failures, *, encode=True, save=None):
        # Explicitly synchronize PhysX transforms to Fabric before reading
        # the camera. This updates rendering/kinematics without integrating time.
        self.physics.sim.forward()
        self.physics.sim.render()
        self.camera.update(1/self.fps,force_recompute=True)
        rgb=tensor(self.camera.data.output['rgb'])[0,:,:,:3].cpu().numpy().astype(np.uint8)
        frame=Image.new('RGB',(self.width,self.height),(18,23,31));frame.paste(Image.fromarray(rgb),(0,self.header))
        draw=ImageDraw.Draw(frame)
        draw.text((30,12),'右手策略直接迁移到左手 · 同一份纯本体学生权重',font=self.font,fill='white')
        draw.text((30,52),'左手：严格镜像模型',font=self.small,fill=(160,204,255))
        draw.text((self.width-420,52),'右手：原始模型',font=self.small,fill=(255,204,156))
        text=f'{elapsed:05.2f} s    |    1× 原速    |    左手 {angles[0]:+.1f}°    ·    右手 {angles[1]:+.1f}°'
        if any(failures):text+='    |    任务终止'
        draw.text((30,self.height-43),text,font=self.small,fill='white')
        draw.text((self.width-500,self.height-43),'同一物理环境 · 双手同步闭环',font=self.small,fill=(180,190,205))
        if encode and self.encoder:
            self.encoder.stdin.write(np.asarray(frame).tobytes());self.frames+=1
        if save:frame.save(self.out/save)
        self.last=frame
        return rgb

    def close(self):
        if self.encoder is None:return
        self.encoder.stdin.close();code=self.encoder.wait();self.encoder=None
        if code:raise RuntimeError(f'ffmpeg exited with {code}')
