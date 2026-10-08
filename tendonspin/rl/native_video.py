# Reference: TendonSpin run_boya_sharpa_hold.py native Isaac recording; NVIDIA
# Isaac Lab Camera APIs (BSD-3-Clause). PIL labels and ffmpeg encode actual frames.
"""Native RTX camera footage; repeated frames are explicitly labelled slow motion."""
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera,CameraCfg
from tendonspin.physics.isaac_parallel import tensor


class NativePolicyVideo:
    def __init__(self,out,training_actions):
        self.out=Path(out);self.training_actions=training_actions
        self.camera=None;self.encoder=None;self.frames=0;self.samples=[];self.last=None
        self.font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)

    def setup(self,physics):
        light=sim_utils.DomeLightCfg(intensity=1800.)
        light.func('/World/DomeLight',light)
        self.camera=Camera(CameraCfg(prim_path='/World/Camera',width=960,height=540,update_period=0.,
            data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=24.,clipping_range=(.01,10.))))

    def attach(self,physics):
        self.physics=physics
        center=physics.center[0].cpu().numpy()
        self.camera.set_world_poses_from_view(eyes=[center+np.array([-.19,.23,.13])],targets=[center+np.array([.02,0.,-.025])])
        self.encoder=subprocess.Popen(['/usr/bin/ffmpeg','-n','-loglevel','error','-f','rawvideo',
            '-pix_fmt','rgb24','-s','960x660','-r','20','-i','-','-an','-c:v','libx264',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(self.out/'isaac_native.mp4')],stdin=subprocess.PIPE)

    def capture(self,elapsed,net,drift,*,initial=False,reason=None):
        self.physics.sim.render();self.camera.update(self.physics.dt,force_recompute=True)
        rgb=tensor(self.camera.data.output['rgb'])[0,:,:,:3].cpu().numpy().astype(np.uint8)
        frame=Image.new('RGB',(960,660),(18,23,31));frame.paste(Image.fromarray(rgb),(0,76))
        draw=ImageDraw.Draw(frame)
        draw.text((12,4),'伯牙手 / Isaac Sim 原生画面 / Hora PPO 早期策略预览',font=self.font,fill='white')
        label='初始状态定格' if initial else ('终止状态定格' if reason else '0.2倍速慢放')
        draw.text((12,39),f'累计训练 {self.training_actions:,} 次动作 | 固定策略独立执行 | {label}',font=self.font,fill='white')
        stops={'drift >5mm':'物体漂移超过5毫米','tilt >15deg':'物体倾斜超过15度',
               'link normal >12N':'接触力超过12牛','time limit':'到达预览时限','wall budget':'到达执行时间预算'}
        text=f'仿真 {elapsed:.4f}s | 有效净转角 {net:+.2f}° | 漂移 {drift:.2f}mm'
        if reason:text+=' | '+stops.get(reason,reason)
        draw.text((12,622),text,font=self.font,fill='white')
        repeats=20 if initial else (40 if reason else 5)
        data=np.asarray(frame).tobytes()
        for _ in range(repeats):self.encoder.stdin.write(data)
        self.frames+=repeats;self.last=frame
        self.samples.append(dict(physics_s=elapsed,valid_net_deg=net,drift_mm=drift,repeated_frames=repeats,phase=label))

    def close(self):
        if self.encoder is None:return None
        self.encoder.stdin.close();code=self.encoder.wait()
        if self.last is not None:self.last.save(self.out/'isaac_native_final.png')
        return dict(path=str(self.out/'isaac_native.mp4'),engine='Isaac Sim RTX native camera',
            encoder_exit_code=code,fps=20,frames=self.frames,duration_s=self.frames/20,
            motion_playback_speed=.2,initial_still_s=1,terminal_still_s=2,
            physics_sample_times=self.samples,interpolated=False)
