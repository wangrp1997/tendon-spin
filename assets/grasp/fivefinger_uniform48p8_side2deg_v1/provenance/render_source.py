# TendonSpin2aa981c native renderer; recorded final hold state, zero new physics.
import json,shutil
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path('/home/rw/Projects/tendon-spin')
OUT=ROOT/'outputs/boya_uniform32_side2deg_fit_v1'
HOLD=OUT/'hold30_native'
state=json.loads((HOLD/'state.json').read_text())
result=json.loads((HOLD/'hold30_result.json').read_text())
if not state['validated'] or not result['accepted'] or result['valid_formal_s']!=30.:raise RuntimeError('30s holding must have passed before replacing the image')
m=mujoco.MjModel.from_xml_path(str(OUT/'scene_visual.xml'));d=mujoco.MjData(m)
mujoco.mj_setState(m,d,np.array(state['integration_state']),mujoco.mjtState(state['signature']))
mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
colors={'th':(.32,.61,.95,1),'ff':(.27,.78,.86,1),'mf':(.69,.51,.92,1),'rf':(.38,.77,.43,1),'lf':(.94,.43,.62,1)}
for gid in range(m.ngeom):
 if m.geom_bodyid[gid] not in (0,m.body('cylinder').id):
  name=m.body(m.geom_bodyid[gid]).name;m.geom_rgba[gid]=next((c for prefix,c in colors.items() if name.startswith(prefix)),(.72,.76,.81,1))
m.vis.global_.offwidth=1100;m.vis.global_.offheight=950
option=mujoco.MjvOption();option.sitegroup[:]=0
camera=mujoco.MjvCamera();camera.lookat[:]=np.array(state['object_position'])+[.018,0,0];camera.distance=.235
canvas=Image.new('RGB',(2200,1070),(245,245,245));draw=ImageDraw.Draw(canvas)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',28)
draw.text((25,15),f'Cylinder {2*state["object"]["radius"]*1000:.1f} x 32 mm | Planned long side +/-1.8 deg | 30 s MuJoCo hold passed',font=font,fill=(20,20,20))
with mujoco.Renderer(m,height=950,width=1100) as renderer:
 for i,(label,azimuth,elevation) in enumerate([('TOP VIEW',90,-89.9),('SIDE VIEW',110,-28)]):
  camera.azimuth=azimuth;camera.elevation=elevation;renderer.update_scene(d,camera,scene_option=option)
  canvas.paste(Image.fromarray(renderer.render()),(i*1100,80));draw.text((i*1100+25,50),label,font=font,fill=(35,35,35))
draw.text((25,1035),f'Recorded final hold state | Five fingers throughout; loaded FF -2.05 / LF +2.03 deg | Max drift {result["formal_maxima"][0]:.3f} mm; tilt {result["formal_maxima"][1]:.3f} deg',font=font,fill=(45,45,45))
image=OUT/'grasp_top.png'
if image.exists():shutil.copyfile(image,OUT/'geometry_fit_top_original.png')
canvas.save(image)
record=dict(image=str(image),side_limits='planned1.8deg; actual loaded FF-2.051deg/LF+2.032deg, slightly beyond2deg',source_state=str(HOLD/'state.json'),source_motion='recorded final state of ONE uninterrupted native0.5s initialization+30s formal hold',renderer='native patched MuJoCo3.13.0 EGL',new_physics_steps=0,views=['top','side'],diameter_mm=2*state['object']['radius']*1000,height_mm=32.,holding_result=str(HOLD/'hold30_result.json'),formal_valid_s=30.,accepted=True,allfive_contact_fraction=result['formal_allfive_contact_fraction'])
(OUT/'combined_image_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
