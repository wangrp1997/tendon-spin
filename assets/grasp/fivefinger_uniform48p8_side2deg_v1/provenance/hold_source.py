# Reference: TendonSpin2aa981c native fixed-target grasp verification and
# DecoupledPlanner static-force initializer; ONE unchanged fitted-size hold.
import json,time
from pathlib import Path
import mujoco,numpy as np
from plan_boya_fivefinger_decoupled import force_solve
from verify_boya_mujoco_grasp import build_scene,initialize,screen,portable,write,digest,ENGINE_SHA

ROOT=Path('/home/rw/Projects/tendon-spin')
SRC=ROOT/'outputs/boya_uniform32_side2deg_fit_v1'
OUT=ROOT/'outputs/boya_uniform32_side2deg_fit_v1/hold30_native'
OUT.mkdir(parents=True,exist_ok=False)
started=time.monotonic();plan=json.loads((SRC/'plan.json').read_text())
engine=Path(mujoco.__file__).parent/f'libmujoco.so.{mujoco.__version__}'
if digest(engine)!=ENGINE_SHA:raise RuntimeError('Native engine identity differs')
assert np.max(np.abs(np.degrees(np.array(plan['q'])[[4,7,10,13]])))<=1.800001
normals=np.array(plan['normals']);points=np.array(plan['points']);pos=np.array(plan['pos'])
forces,diagnosis=force_solve(points,normals,pos,normals+np.array([0.,0.,.4905/5]))
plan['forces']=forces.reshape(5,3).tolist()
write(OUT/'plan_with_initializer.json',plan)
rec=dict(status='starting',source_plan=str(SRC/'plan.json'),source_plan_sha256=digest(SRC/'plan.json'),
    object=plan['object'],orientation='same fitted upright palm-normal WXYZ',object_quat_wxyz=plan['quat'],
    engine='native patched MuJoCo3.13.0',engine_sha256=ENGINE_SHA,
    actuator_interface='16 active finger motors, two held wrists, four passive1:1 mimics; same original position actuators and fixed preload formula',
    observations='privileged CAD/state/contact used for initialization and diagnostics only; no learned policy',
    collision='same packaged CAD mesh collision/excludes/friction/solver as original native scene; inactive object weld; no support',
    timing='dt0.0005s; frozen targets at original20Hz; uninterrupted0.5s initialization plus30s formal hold',
    budgets=dict(episodes=1,initialization_s=.5,formal_s=30.,training_actions=0,resets=0,controller_switches=0),
    acceptance='full30s formal; finite, drift<=5mm, axis tilt<=15deg, joint speed<=100rad/s, link normal<=12N, mimic error<=.05rad; >=2contact groups in >=90% final0.1s; five-finger contact diagnostic only',
    force_initializer_diagnostic=diagnosis,force_initializer_is_admission_gate=False,
    retries=0,sources=[])
for path in [Path(__file__),ROOT/'scripts/verify_boya_mujoco_grasp.py',ROOT/'scripts/plan_boya_fivefinger_decoupled.py',ROOT/'assets/grasp/scene.xml',ROOT/'docs/data/boya_native_contract.json',SRC/'plan.json']:
    rec['sources'].append(dict(path=str(path),sha256=digest(path)))
write(OUT/'result.json',rec)
m=build_scene(OUT,dict(shape='cylinder',diameter_m=2*plan['object']['radius'],height_m=plan['object']['height'],mass_kg=plan['object']['mass']))
d,preload=initialize(m,plan);rec['preload']=preload
write(OUT/'initial_state.json',portable(m,d,plan,OUT/'scene.xml'))
print(json.dumps(dict(status='holding',diameter_mm=2000*plan['object']['radius'],height_mm=1000*plan['object']['height'],engine=rec['engine'],formal_s=30.,initializer_feasible=diagnosis['success'])),flush=True)
rec['hold']=screen(m,d,plan,OUT,'hold30',seconds=30.5,initialization_s=.5)
with np.load(OUT/'hold30_execution.npz',allow_pickle=False) as trace:
 side_qids=[m.joint(n).qposadr[0] for n in ('FFJ4','MFJ4','RFJ4','LFJ4')]
 rec['max_abs_side_angles_deg_all_phases']=dict(zip(('FFJ4','MFJ4','RFJ4','LFJ4'),np.max(np.abs(np.degrees(trace['qpos'][:,side_qids])),axis=0).tolist()))
 rec['observed_long_side_within2deg']=all(x<=2. for x in rec['max_abs_side_angles_deg_all_phases'].values())
state=portable(m,d,plan,OUT/'scene.xml');state['validated']=rec['hold']['accepted'];write(OUT/'final_state.json',state)
if rec['hold']['accepted']:write(OUT/'state.json',state)
rec.update(status='completed',wall_s=time.monotonic()-started,final_side_angles_deg={name:float(np.degrees(state['joint_pos'][name])) for name in ('FFJ4','MFJ4','RFJ4','LFJ4')})
write(OUT/'result.json',rec);(OUT/'run_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(status=rec['status'],accepted=rec['hold']['accepted'],actual_s=rec['hold']['actual_s'],formal_s=rec['hold']['formal_s'],valid_formal_s=rec['hold']['valid_formal_s'],stop=rec['hold']['stop_reason'],maxima=rec['hold']['formal_maxima'],touch=rec['hold']['formal_finger_contact_fractions'],final_side_deg=rec['final_side_angles_deg'])),flush=True)
