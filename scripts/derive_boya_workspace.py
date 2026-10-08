# Source: packaged Boya CAD and native grasp44 body frames from botyard-inhand
# 204d197a9606fb7266e884f3b2e6110195be01cb. Static geometry only, no simulation.
from pathlib import Path
import json,hashlib
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parents[1]
cpath=root/'docs/data/boya_native_contract.json';c=json.loads(cpath.read_text())
vertices={};sources=[cpath]
for b in c['body_frames']:
 if b['name']!='palm' and not b['name'].endswith(('middle','distal')):continue
 p=root/'assets/hand/meshes'/(b['name']+'.stl')
 q=np.asarray(b['quat']);vertices[b['name']]=Rotation.from_quat(q[[1,2,3,0]]).apply(trimesh.load(p,process=False).vertices)+b['pos'];sources.append(p)
radius=float(np.hypot(c['object']['diameter_m']/2,c['object']['length_m']/2))
fingers=np.concatenate([v for name,v in vertices.items() if name!='palm'])
workspace={'schema':1,'basis':'Boya nominal palm and middle/distal CAD meshes transformed by original grasp44 body frames; task-region approximation, not a proven controllability boundary.',
 'nominal_object_position_m':c['object']['pos'],'palm_top_z_m':float(vertices['palm'][:,2].max()),'object_bounding_radius_m':radius,
 'reset_height_m':float(vertices['palm'][:,2].max()+radius),
 'xy_lower_m':(fingers[:,:2].min(0)-radius).tolist(),'xy_upper_m':(fingers[:,:2].max(0)+radius).tolist(),
 'confirmation_control_steps':2,'control_hz':20,
 'height_basis':'Palm top plus cylinder circumsphere radius: even a palm-supported cylinder is below this center-height plane. No claim that every position above is recoverable.',
 'lateral_basis':'Fixed nominal middle/distal mesh XY envelope expanded by cylinder circumsphere radius; all five fingers included.',
 'contact_policy':'Per-finger simulation contact diagnostics only; fewer than two fingers does not terminate. No tactile array or new actor input.',
 'sources':[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}
(root/'assets/grasp/rotation_workspace.json').write_text(json.dumps(workspace,indent=2)+'\n')
