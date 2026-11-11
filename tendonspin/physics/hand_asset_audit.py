"""Static audit of a hand USD against the canonical policy contract."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from tendonspin.interfaces import ACTION_NAMES


def _attribute(prim, name):
    attribute = prim.GetAttribute(name)
    return attribute.Get() if attribute and attribute.HasValue() else None


def audit_hand_asset(profile, contract, tolerance_rad=5e-5):
    """Check names, action limits, coupling bodies and collision exclusions."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(profile.asset_path))
    if not stage:
        raise ValueError('Unable to open hand asset: '+str(profile.asset_path))
    source = stage.GetPrimAtPath(profile.source_prim)
    if not source:
        raise ValueError('Hand asset source_prim does not exist: '+profile.source_prim)

    joints = {}
    joint_order = []
    bodies = {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(profile.source_prim+'/'):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[profile.canonical_body_name(prim.GetName())] = prim
        if prim.IsA(UsdPhysics.RevoluteJoint):
            canonical = profile.canonical_body_name(prim.GetName())
            joints[canonical] = prim
            joint_order.append(canonical)

    actions = [a for a in contract['actuators'] if a['joint'] in ACTION_NAMES]
    expected_actions = list(ACTION_NAMES)
    errors = []
    limits = {}
    for name in expected_actions:
        prim = joints.get(name)
        if prim is None:
            errors.append(f'missing action joint {name}')
            continue
        lower = _attribute(prim, 'physics:lowerLimit')
        upper = _attribute(prim, 'physics:upperLimit')
        limits[name] = dict(usd_lower_deg=lower, usd_upper_deg=upper,
            usd_lower_rad=None if lower is None else float(np.deg2rad(lower)),
            usd_upper_rad=None if upper is None else float(np.deg2rad(upper)))

    by_name = {a['joint']: a for a in actions}
    for name in expected_actions:
        if name not in limits or name not in by_name:
            continue
        expected = by_name[name]['control_range']
        actual = limits[name]
        if actual['usd_lower_rad'] is None or actual['usd_upper_rad'] is None:
            errors.append(f'missing USD limits for {name}')
        elif (abs(actual['usd_lower_rad']-expected[0]) > tolerance_rad
              or abs(actual['usd_upper_rad']-expected[1]) > tolerance_rad):
            errors.append(f'limit mismatch for {name}: USD '
                f"{[actual['usd_lower_rad'], actual['usd_upper_rad']]} vs contract {expected}")

    passive = [c['slave'] for c in contract['couplings']]
    masters = [c['master'] for c in contract['couplings']]
    for name in passive+masters:
        if name not in joints:
            errors.append(f'missing coupling joint {name}')
    distal = [f'{finger}distal' for finger in ('th', 'ff', 'mf', 'rf', 'lf')]
    for name in distal:
        if name not in bodies:
            errors.append(f'missing distal body {name}')

    contact_scene = ET.parse(profile.contact_scene).getroot()
    exclusions = []
    missing_exclusions = []
    for entry in contact_scene.findall('./contact/exclude'):
        pair = (entry.attrib['body1'], entry.attrib['body2'])
        exclusions.append(pair)
        if pair[0] not in bodies or pair[1] not in bodies:
            missing_exclusions.append(pair)
    if missing_exclusions:
        errors.append(f'contact exclusions reference missing bodies: {missing_exclusions}')

    source_mimic = {}
    drive_parameters = {}
    for name in passive:
        prim = joints.get(name)
        if prim is None:
            continue
        source_mimic[name] = {
            'enabled': _attribute(prim, 'newton:mimicEnabled'),
            'coef0': _attribute(prim, 'newton:mimicCoef0'),
            'coef1': _attribute(prim, 'newton:mimicCoef1'),
        }
    for name in expected_actions:
        prim = joints.get(name)
        if prim is not None:
            drive_parameters[name] = {
                'stiffness': _attribute(prim, 'drive:angular:physics:stiffness'),
                'damping': _attribute(prim, 'drive:angular:physics:damping'),
                'axis': _attribute(prim, 'physics:axis'),
            }

    action_order = [name for name in joint_order if name in ACTION_NAMES]
    warnings = []
    if action_order != expected_actions:
        warnings.append('USD traversal action order differs; runtime uses canonical names')
    if not any(value['enabled'] for value in source_mimic.values()):
        warnings.append('source USD has no authored Newton mimic flags; runtime profile authors them')
    if profile.mirror_world_y:
        warnings.append('Left runtime uses the right TendonSpin control contract; '
                        'native USD drive values are provenance only')
    else:
        warnings.append('Right side preserves its historical contract instead of current USD drive gains')
    return dict(schema='tendonspin-hand-asset-audit-v1', hand_side=profile.side,
        asset_path=str(profile.asset_path), source_prim=profile.source_prim,
        all_pass=not errors, errors=errors, warnings=warnings,
        action_names=expected_actions, usd_action_traversal=action_order,
        joint_names=sorted(joints), passive_joint_names=passive,
        rigid_bodies=sorted(bodies), distal_bodies=distal,
        action_limits=limits, source_mimic=source_mimic,
        drive_parameters=drive_parameters, contact_exclusions=exclusions,
        tolerance_rad=tolerance_rad)
