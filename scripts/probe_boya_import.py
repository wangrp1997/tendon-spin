# Source/project: NVIDIA Isaac Lab 3.0.0rc1 UrdfConverter/AppLauncher APIs
# (BSD-3-Clause) and original Boya URDF/CAD from botyard-inhand204d197a... .
# New TendonSpin structural audit; not a rotation controller or physics success.
"""Import unchanged Boya meshes/URDF and inspect joint/collision USD schemas."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
source=root/'assets/hand/boya.urdf'
started=time.monotonic()
record=dict(kind='Boya structural import; zero dynamic steps',physics_steps=0,
            training_actions=0,object_loaded=False,grasp_validated=False,
            controller=None,complete=False,
            eula_acceptance_supplied=os.environ.get('OMNI_KIT_ACCEPT_EULA')=='YES',
            urdf=dict(path=str(source.relative_to(root)),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))

def save(phase):
    record['phase']=phase
    record['wall_s_at_phase']=time.monotonic()-started
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    print('TENDONSPIN_IMPORT '+json.dumps(record),flush=True)

save('before official launcher')
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,enable_cameras=False,device='cuda:0')
app=launcher.app
failed=False
try:
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from pxr import Usd, UsdGeom, UsdPhysics
    import inspect
    for key,cls in (('launcher',AppLauncher),('converter',UrdfConverter)):
        path=Path(inspect.getfile(cls))
        record[key+'_source']=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    cfg=UrdfConverterCfg(asset_path=str(source),usd_dir=str(root/'outputs/isaac_boya_import'),
        fix_base=True,merge_fixed_joints=False,merge_mesh=False,
        collision_from_visuals=False,collision_type='Convex Hull',self_collision=True,
        convert_mimic_joints_to_normal_joints=False,run_asset_transformer=False,
        force_usd_conversion=True,make_instanceable=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(target_type='none',drive_type='force',
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.)))
    record['conversion_configuration']=cfg.to_dict()
    save('before converter')
    converter=UrdfConverter(cfg)
    usd=Path(converter.usd_path)
    record['usd']=dict(path=str(usd.relative_to(root)),sha256=hashlib.sha256(usd.read_bytes()).hexdigest())
    save('conversion returned')
    stage=Usd.Stage.Open(str(usd))
    if stage is None: raise RuntimeError('Generated USD stage cannot be opened')
    joints=[]; mimics=[]; collisions=[]; articulations=[]
    for prim in stage.Traverse():
        schemas=list(prim.GetAppliedSchemas())
        if prim.IsA(UsdPhysics.Joint):
            joints.append(dict(name=prim.GetName(),path=str(prim.GetPath()),type=prim.GetTypeName(),schemas=schemas))
        if any('Mimic' in schema for schema in schemas):
            attrs={a.GetName():str(a.Get()) for a in prim.GetAttributes() if 'mimic' in a.GetName().lower()}
            rels={r.GetName():[str(p) for p in r.GetTargets()] for r in prim.GetRelationships() if 'mimic' in r.GetName().lower()}
            mimics.append(dict(name=prim.GetName(),path=str(prim.GetPath()),schemas=schemas,attributes=attrs,relationships=rels))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            approximation=str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()) if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None
            collisions.append(dict(path=str(prim.GetPath()),type=prim.GetTypeName(),approximation=approximation))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulations.append(dict(path=str(prim.GetPath()),attributes={a.GetName():str(a.Get()) for a in prim.GetAttributes() if 'selfcollision' in a.GetName().lower()}))
    record.update(joints=joints,mimics=mimics,collisions=collisions,articulations=articulations,
                  meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),up_axis=str(UsdGeom.GetStageUpAxis(stage)))
    contract=json.loads((root/'docs/data/boya_native_contract.json').read_text())
    expected={j['name'] for j in contract['joints']}
    actual={j['name'] for j in joints if j['type']=='PhysicsRevoluteJoint'}
    slaves={c['slave'] for c in contract['couplings']}
    record['checks']=dict(all_expected_revolute_joints=expected<=actual,
                         missing_joint_names=sorted(expected-actual),
                         all_mimic_slave_schemas=slaves<={m['name'] for m in mimics},
                         missing_mimic_slaves=sorted(slaves-{m['name'] for m in mimics}),
                         collision_geometry_present=bool(collisions))
    record['structural_checks_passed']=all(record['checks'][key] for key in ('all_expected_revolute_joints','all_mimic_slave_schemas','collision_geometry_present'))
    record['complete']=True
    save('structural inspection complete')
except BaseException as error:
    failed=True
    record['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    save('process error')
finally:
    save('before close')
    app.close(exit_code=1 if failed else 0)
