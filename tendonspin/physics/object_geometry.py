# References: TendonSpin original Boya contract at dfad6a4; Sharpa RL Lab
# 95ccda3d948801bb5da4cb7ffea766e03067a63b README (physical cylinder recommendation);
# Isaac Lab 3.0.0rc1 sim/spawners/shapes/shapes.py (BSD-3-Clause), USD primitive
# attributes and child-geometry layout. New evaluation-only configuration adapter:
# author geometry before physics starts; preserve body pose, material and collision APIs.
"""Selectable object geometry for frozen-policy evaluations and pose replay."""
from copy import deepcopy
import json
import math
from pathlib import Path


SHAPES = ('cylinder', 'sphere', 'box')


def add_object_arguments(parser):
    profiles=json.loads((Path(__file__).resolve().parents[2]/'assets/objects/profiles.json').read_text())['profiles']
    parser.add_argument('--object-profile', choices=tuple(profiles), default='boya_original')
    parser.add_argument('--object-shape', choices=SHAPES)
    parser.add_argument('--object-diameter-mm', type=float)
    parser.add_argument('--object-height-mm', type=float)
    parser.add_argument('--object-size-mm', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--object-mass-g', type=float)


def resolve_object_geometry(root, args=None):
    root = Path(root)
    catalog = json.loads((root / 'assets/objects/profiles.json').read_text())
    profile = getattr(args, 'object_profile', 'boya_original')
    result = deepcopy(catalog['profiles'][profile])
    result.update(profile=profile, shape=getattr(args, 'object_shape', None) or result['shape'])
    diameter = getattr(args, 'object_diameter_mm', None)
    height = getattr(args, 'object_height_mm', None)
    size = getattr(args, 'object_size_mm', None)
    mass = getattr(args, 'object_mass_g', None)
    if diameter is not None:
        result['diameter_m'] = diameter / 1000
    if height is not None:
        result['height_m'] = height / 1000
    if mass is not None:
        result['mass_kg'] = mass / 1000
    if result['shape'] == 'box':
        if size is None or diameter is not None or height is not None:
            raise ValueError('Box requires --object-size-mm X Y Z; do not mix diameter/height overrides')
        result['size_m'] = [value / 1000 for value in size]
        result.pop('diameter_m'); result.pop('height_m')
        dimensions = result['size_m']
    elif result['shape'] == 'sphere':
        if size is not None or height is not None:
            raise ValueError('Sphere uses --object-diameter-mm only')
        result.pop('height_m')
        dimensions = [result['diameter_m']] * 3
    else:
        if size is not None:
            raise ValueError('Cylinder uses diameter/height; --object-size-mm is for boxes')
        dimensions = [result['diameter_m'], result['diameter_m'], result['height_m']]
    if not all(math.isfinite(value) and value > 0 for value in [*dimensions, result['mass_kg']]):
        raise ValueError('Object dimensions and mass must be finite and positive')
    result['dimensions_m'] = list(dimensions)
    result['overrides'] = {key: value for key, value in dict(shape=getattr(args, 'object_shape', None),
        diameter_mm=diameter, height_mm=height, size_mm=size, mass_g=mass).items() if value is not None}
    return result


def author_primitive(stage, path, geometry):
    """Use one definition for the simulated collider and the video geometry."""
    from pxr import Gf, UsdGeom, Vt
    shape = geometry['shape']
    if shape not in SHAPES:
        raise ValueError(shape)
    prim = stage.DefinePrim(path, dict(cylinder='Cylinder', sphere='Sphere', box='Cube')[shape])
    if shape == 'cylinder':
        item = UsdGeom.Cylinder(prim)
        item.CreateRadiusAttr(geometry['diameter_m'] / 2)
        item.CreateHeightAttr(geometry['height_m'])
        item.CreateAxisAttr('Z')
    elif shape == 'sphere':
        item = UsdGeom.Sphere(prim)
        item.CreateRadiusAttr(geometry['diameter_m'] / 2)
    else:
        item = UsdGeom.Cube(prim)
        item.CreateSizeAttr(1.)
    scale = geometry['size_m'] if shape == 'box' else (1., 1., 1.)
    xform = UsdGeom.Xformable(prim)
    scale_op = next((op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale), None)
    if scale_op is None:
        scale_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    scale_op.Set(Gf.Vec3d(*scale)) if scale_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else scale_op.Set(Gf.Vec3f(*scale))
    half = [0.5] * 3 if shape == 'box' else [value / 2 for value in geometry['dimensions_m']]
    UsdGeom.Boundable(prim).CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*[-value for value in half]), Gf.Vec3f(*half)]))
    return prim


def configure_object_scene(physics, geometry):
    """Existing BoyaParallel scene_setup hook runs before SimulationContext.reset."""
    from pxr import UsdPhysics
    if physics.sim.is_playing():
        raise RuntimeError('Object geometry must be configured before physics starts')
    stage = physics.sim.stage
    authored = []
    for index in range(physics.num_envs):
        body_path = f'/World/envs/env_{index}/Cylinder'
        body = stage.GetPrimAtPath(body_path)
        mesh_path = body_path + '/geometry/mesh'
        if not body or not stage.GetPrimAtPath(mesh_path):
            raise ValueError('Missing original object body/geometry: ' + body_path)
        prim = author_primitive(stage, mesh_path, geometry)
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Object collision API was lost')
        UsdPhysics.MassAPI(body).CreateMassAttr(geometry['mass_kg'])
        authored.append(dict(body_path=body_path, geometry_path=mesh_path, geometry_type=prim.GetTypeName(),
            mass_kg=float(UsdPhysics.MassAPI(body).GetMassAttr().Get()), collision_enabled=bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())))
    physics.object_geometry = deepcopy(geometry)
    return dict(before_physics_start=True, authored=authored, geometry=geometry)
