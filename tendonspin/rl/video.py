# New standalone renderer referencing the interface of botyard-inhand/research/learning/video.py
# Source revision: 204d197a9606fb7266e884f3b2e6110195be01cb.
# TendonSpin changes and evaluation identity are declared in docs/experiment_state.md.
# This local PPO is not a reproduced Hora/AnyRotate/Sharpa policy.
"""Render verified saved native trajectories without policy/physics reexecution."""
import argparse
import json
from pathlib import Path
import subprocess
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .config import ROOT, digest, write
from tendonspin.physics.native import scene, SIGNATURE

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=json.loads((args.episode/'results.json').read_text())
    assert result['original_state'] and not result['episode_resets'] and not result['controller_switches']
    assert not any(result['replay_errors'].values()) and result['inference_error']==0
    source=ROOT/result['execution']['path'];assert digest(source)==result['execution']['sha256']
    s,d,_=scene();m=s.model
    with np.load(source,allow_pickle=False) as raw:
        mujoco.mj_setState(m,d,raw['initial_state'],SIGNATURE)
        q=np.vstack((d.qpos.copy(),raw['qpos'][:result['valid_steps']]))
        a=np.r_[0.,raw['angle_deg'][:result['valid_steps']]]
        metrics=np.vstack((np.zeros(14),raw['metrics'][:result['valid_steps']]))
        dt=float(raw['physics_dt'])
    m.vis.global_.offwidth=640;m.vis.global_.offheight=480
    camera=mujoco.MjvCamera();camera.lookat[:]=q[0,s.qa:s.qa+3]
    camera.distance=.3;camera.azimuth=140;camera.elevation=-20
    fps=30;width=640;height=560
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',17)
    indices=np.r_[np.minimum((np.arange(0,result['valid_seconds'],1/fps)/dt).astype(int),len(q)-1),len(q)-1]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    command=['ffmpeg','-n','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-s',f'{width}x{height}',
             '-pix_fmt','rgb24','-r',str(fps),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(args.out)]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as encoder:
        try:
            with mujoco.Renderer(m,height=480,width=640) as renderer:
                for k in indices:
                    d.qpos[:]=q[k];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
                    renderer.update_scene(d,camera)
                    frame=Image.new('RGB',(width,height),(18,23,31))
                    frame.paste(Image.fromarray(renderer.render()),(0,40))
                    draw=ImageDraw.Draw(frame)
                    draw.text((8,5),f'TendonSpin / u{result["training_updates"]} / {k*dt:.3f}s / {a[k]:.2f}°',font=font,fill='white')
                    draw.text((8,524),f'漂移 {metrics[k,0]:.3f}mm / 轴倾 {metrics[k,1]:.2f}° / 特权学习策略',font=font,fill='white')
                    encoder.stdin.write(np.asarray(frame).tobytes())
                draw.rectangle((0,0,width,40),fill=(75,25,25))
                draw.text((8,5),f'终止: {result["reason"]} / {result["net_deg"]:.2f}°',font=font,fill='white')
                frame.save(args.out.with_suffix('.png'))
                for _ in range(fps):encoder.stdin.write(np.asarray(frame).tobytes())
        finally:encoder.stdin.close()
        assert encoder.wait()==0
    write(args.out.with_suffix('.json'),dict(execution=result['execution'],video=str(args.out.resolve().relative_to(ROOT)),
        video_sha256=digest(args.out),rendering_only=True,physical_reexecution=False,
        valid_seconds=result['valid_seconds'],net_deg=result['net_deg'],terminal_still_seconds=1.,
        attribution=result['attribution']))
    print(str(args.out),flush=True)

if __name__=='__main__':main()
