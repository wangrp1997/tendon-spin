# Reuse TendonSpin2aa981c CAD DecoupledPlanner and native scene builder.
# One bounded geometry-only diameter fit, exact FF/LF side constraints.
import json,time,shutil,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont
from plan_boya_fivefinger_decoupled import DecoupledPlanner
from verify_boya_mujoco_grasp import build_scene,write,digest,ENGINE_SHA
from tendonspin.interfaces import ACTION_NAMES
ROOT=Path('/home/rw/Projects/tendon-spin')
SRC=ROOT/'outputs/boya_uniform32_zero_side_fit_v1/hold30_native'
OUT=ROOT/'outputs/boya_uniform32_side2deg_fit_v1'
OUT.mkdir(parents=True,exist_ok=False)
start=time.monotonic();g=DecoupledPlanner()
source=SRC/'state.json';saved=json.loads(source.read_text());seed=json.loads((SRC/'plan_with_initializer.json').read_text())
q0=np.array([saved['joint_pos'][n] for n in ACTION_NAMES]);pos0=np.array(seed['pos']);quat=np.array(seed['quat'])
if digest(Path(mujoco.__file__).parent/f'libmujoco.so.{mujoco.__version__}')!=ENGINE_SHA:raise RuntimeError('Engine identity differs')
fixed=np.array([],dtype=int);free=np.arange(20)
flex3=np.array([5,8,11,14]);flex2=np.array([6,9,12,15]);x0=np.r_[q0,pos0*1000,seed['object']['radius']*2000]
x0[[4,7,10,13]]=0.
g.quat=quat.copy();axis=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()[:,2]
def unpack(z):
 x=x0.copy();x[free]=z;x[fixed]=0.;return x
def measure(x):
 v=g.measure(x[:16],x[16:19]/1000,[x[19]/1000,.032]);points=v['lines'][:,:3];pos=x[16:19]/1000
 axial=(points-pos)@axis;radial=points-pos-axial[:,None]*axis
 normals=-radial/np.maximum(np.linalg.norm(radial,axis=1)[:,None],1e-10)
 v.update(points=points,pos=pos,axial=axial,normals=normals,pairs=np.linalg.norm(points[:,None]-points[None,:],axis=-1)[np.triu_indices(5,1)])
 return v
best=[None,np.inf];last=[start]
def residual(z):
 if time.monotonic()-start>75:raise TimeoutError('75s geometry budget')
 x=unpack(z);v=measure(x);local=v['local'];sign=np.array([-1,1,1,1,1])
 penalties=np.r_[np.maximum(.0003-v['nonpad'],0),np.maximum(.0003-v['clearance'],0),np.maximum(-.0045-sign*local[:,1],0),np.maximum(.006-local[:,2],0),np.maximum(local[:,2]-.032,0),np.maximum(np.abs(v['axial'])-.015,0),np.maximum(.010-v['pairs'],0),max(0.,.004-local[1,0]),max(0.,.004+local[4,0])]
 r=np.r_[(v['gaps']-.00002)*1000,penalties*1000,.2*(x[:16]-q0),2.*(x[flex3]+2*x[flex2]-np.pi/2),.5*(x[flex3]-np.mean(x[flex3])),.3*x[[4,7,10,13]],.04*(x[16:19]-pos0*1000),.03*x[19]]
 score=np.linalg.norm(r)
 if score<best[1]:best[:]=[z.copy(),score]
 if time.monotonic()-last[0]>10:
  print(json.dumps(dict(status='fitting',wall_s=time.monotonic()-start,diameter_mm=float(unpack(best[0])[19]),residual=best[1])),flush=True);last[0]=time.monotonic()
 return r
def jac(z):
 r=residual(z);j=np.empty((len(r),len(z)))
 for k,index in enumerate(free):
  zz=z.copy();step=1e-5 if index<16 else .001;zz[k]+=step;j[:,k]=(residual(zz)-r)/step
 return j
lo=np.r_[np.maximum(g.bounds[:,0],q0-.6),pos0*1000-20,10.]
hi=np.r_[np.minimum(g.bounds[:,1],q0+.6),pos0*1000+20,100.]
lo[[4,7,10,13]]=np.radians(-1.8);hi[[4,7,10,13]]=np.radians(1.8)
z0=np.clip(x0[free],lo[free]+1e-6,hi[free]-1e-6)
try:
 opt=least_squares(residual,z0,jac=jac,bounds=(lo[free],hi[free]),max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-7);z=opt.x;stop=str(opt.message)
except TimeoutError as e:z=best[0];stop=str(e)
x=unpack(z);v=measure(x)
plan=dict(joints=list(ACTION_NAMES),q=x[:16].tolist(),pos=v['pos'].tolist(),quat=quat.tolist(),points=v['points'].tolist(),normals=v['normals'].tolist(),contact_body_local=v['local'].tolist(),object=dict(shape='cylinder',radius=x[19]/2000,height=.032,mass=.05),validated=False)
write(OUT/'plan.json',plan)
rec=dict(kind='single bounded local CAD geometry fit; no stability claim',diameter_mm=float(x[19]),height_mm=32.,mass_kg=.05,source_state_sha256=digest(source),engine_sha256=ENGINE_SHA,physics_steps=0,training_actions=0,hard_zero_joints=[],allowed_long_side_deg=2.,planned_long_side_limit_deg=1.8,load_margin_deg=.2,side_angles_deg=dict(zip([ACTION_NAMES[i] for i in [4,7,10,13]],np.degrees(x[[4,7,10,13]]).tolist())),q=x[:16].tolist(),gaps_mm=(v['gaps']*1000).tolist(),min_nonpad_gap_mm=float(v['nonpad'].min()*1000),min_interfinger_gap_mm=float(v['clearance'].min()*1000),wall_s=time.monotonic()-start,solve_stop=stop,stability_tested=False,geometry_objective='distal proximity and collision clearance, natural uniform flexion, all long side within1.8deg,0.2deg load margin, small diameter, seed proximity; no minimum finger load')
write(OUT/'result.json',rec)
build_scene(OUT,dict(diameter_m=x[19]/1000,height_m=.032,mass_kg=.05))
BRAND=ROOT/'outputs/boya_uniform32_botyard_visual_v1'
for name in ['botyard_top.png','plain_white.png','side_right.png','side_left.png','side_front.png','side_back.png']:shutil.copyfile(BRAND/name,OUT/name)
tree=ET.parse(OUT/'scene.xml');root=tree.getroot();asset=root.find('asset')
ET.SubElement(asset,'texture',dict(name='botyard_surface',type='cube',fileup='side_front.png',filedown='side_back.png',fileright='side_right.png',fileleft='side_left.png',filefront='plain_white.png',fileback='botyard_top.png'))
ET.SubElement(asset,'material',dict(name='botyard_white',texture='botyard_surface',texuniform='false',rgba='1 1 1 1',specular='.1',shininess='.1'))
geom=root.find("./worldbody/body[@name='cylinder']/geom");geom.set('rgba','1 1 1 1');geom.set('material','botyard_white')
ET.indent(tree,space='  ');tree.write(OUT/'scene_visual.xml',encoding='unicode')
m=mujoco.MjModel.from_xml_path(str(OUT/'scene_visual.xml'));d=mujoco.MjData(m)
for j in json.loads((ROOT/'docs/data/boya_native_contract.json').read_text())['joints']:d.qpos[m.joint(j['name']).qposadr[0]]=j['qpos']
for name,value in zip(ACTION_NAMES,x[:16]):d.qpos[m.joint(name).qposadr[0]]=value
for f in ['FF','MF','RF','LF']:d.qpos[m.joint(f+'J1').qposadr[0]]=d.qpos[m.joint(f+'J2').qposadr[0]]
qa=m.joint('cylinder_free').qposadr[0];d.qpos[qa:qa+3]=v['pos'];d.qpos[qa+3:qa+7]=quat
mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
colors={'th':(.32,.61,.95,1),'ff':(.27,.78,.86,1),'mf':(.69,.51,.92,1),'rf':(.38,.77,.43,1),'lf':(.94,.43,.62,1)}
for gid in range(m.ngeom):
 if m.geom_bodyid[gid] not in (0,m.body('cylinder').id):m.geom_rgba[gid]=next((c for prefix,c in colors.items() if m.body(m.geom_bodyid[gid]).name.startswith(prefix)),(.72,.76,.81,1))
m.vis.global_.offwidth=1100;m.vis.global_.offheight=950
option=mujoco.MjvOption();option.sitegroup[:]=0
camera=mujoco.MjvCamera();camera.lookat[:]=v['pos']+[.018,0,0];camera.distance=.235;camera.azimuth=90;camera.elevation=-89.9
with mujoco.Renderer(m,height=950,width=1100) as renderer:
 renderer.update_scene(d,camera,scene_option=option);im=Image.fromarray(renderer.render())
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25);draw=ImageDraw.Draw(im)
draw.rectangle((0,0,1100,48),fill=(245,245,245));draw.text((20,12),f'Diameter {x[19]:.1f} mm   |   Height 32 mm   |   Long-finger side within +/-2 deg',font=font,fill=(20,20,20))
draw.rectangle((0,910,1100,950),fill=(245,245,245));draw.text((20,918),'CAD geometry fit - holding stability not tested',font=font,fill=(60,60,60))
image=OUT/'grasp_top.png';im.save(image);rec.update(image=str(image),renderer='native patched MuJoCo3.13.0 EGL',source_pose='optimized CAD qpos, zero dynamics');write(OUT/'result.json',rec)
(OUT/'run_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(rec),flush=True)
