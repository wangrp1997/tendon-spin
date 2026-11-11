"""Import the strict mirrored V3.2 left URDF into an Isaac Sim USD asset."""
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'assets/hand/mirror_right_v3_2/left_dexhand.urdf'
USD_DIR = ROOT/'assets/hand/mirror_right_v3_2/usd'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=False, device='cuda:0').app
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
        from pxr import Usd, UsdPhysics

        if USD_DIR.exists():
            raise FileExistsError(USD_DIR)
        USD_DIR.mkdir(parents=True)
        config = UrdfConverterCfg(
            asset_path=str(SOURCE), usd_dir=str(USD_DIR), fix_base=True,
            merge_fixed_joints=False, merge_mesh=False,
            collision_from_visuals=False, collision_type='Convex Hull',
            self_collision=True, convert_mimic_joints_to_normal_joints=False,
            run_asset_transformer=False, force_usd_conversion=True,
            make_instanceable=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type='none', drive_type='force',
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0., damping=0.)))
        converter = UrdfConverter(config)
        usd = Path(converter.usd_path)
        stage = Usd.Stage.Open(str(usd))
        if stage is None:
            raise RuntimeError('Converted USD cannot be opened')
        joints = [str(prim.GetPath()) for prim in stage.Traverse()
                  if prim.IsA(UsdPhysics.RevoluteJoint)]
        bodies = [str(prim.GetPath()) for prim in stage.Traverse()
                  if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        record = dict(schema='tendonspin-mirror-left-import-v1',
                      source=dict(path=str(SOURCE.relative_to(ROOT)),
                                  sha256=digest(SOURCE)),
                      usd=dict(path=str(usd.relative_to(ROOT)),
                               sha256=digest(usd)),
                      default_prim=str(stage.GetDefaultPrim().GetPath()),
                      rigid_bodies=len(bodies), revolute_joints=len(joints),
                      collision_count=sum(
                          prim.HasAPI(UsdPhysics.CollisionAPI)
                          for prim in stage.Traverse()))
        (USD_DIR/'import_manifest.json').write_text(
            json.dumps(record, indent=2)+'\n')
        print(json.dumps(record))
    finally:
        app.close()


if __name__ == '__main__':
    main()
