"""Explicit hand-side asset and coordinate profiles.

The task, reward, randomization and student contracts are bilateral. Only the
hand asset, runtime joint/body names and world-mirroring rules belong here.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path


SIDES = ('right', 'left')
MODEL_SOURCE = 'botyard_v3_2_right'
LEFT_MODEL_SOURCE = 'botyard_v3_2_strict_right_mirror'
VENDOR_LEFT_MODEL_SOURCE = 'botyard_v3_2_vendor_left'
MODEL_REVISION = '28f4ba5649714c74c13f881891d579bfceb766e2'
CONTROL_SOURCE = 'right_tendonspin_contract'
LEFT_HAND_ENV = 'TENDONSPIN_LEFT_HAND_USD'
LEFT_HAND_RELATIVE = Path(
    'assets/hand/mirror_right_v3_2/usd/left_dexhand/left_dexhand.usda')
EXTERNAL_LEFT_HAND_RELATIVE = Path(
    'ros2_botyard/dexhand_description/usd/isaac_dexhand/left_hand.usd')
LEFT_VALIDATION_SCENE = Path(
    'assets/hand/mirror_right_v3_2/mjcf/v3_2/scene_left_validation.xml')
LEFT_CONTACT_SCENE = Path(
    'assets/hand/mirror_right_v3_2/mjcf/v3_2/left_dexhand.xml')


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def mirror_object_state_y(state):
    """Mirror an xyzw world pose/twist tensor or array across the x-z plane."""
    result = state.clone() if hasattr(state, 'clone') else state.copy()
    # position y, quaternion x/z, linear y and angular x/z (pseudovector).
    for index in (1, 3, 5, 8, 10, 12):
        result[..., index] *= -1
    return result


def _mirror_wxyz_pose(pose):
    result = deepcopy(pose)
    result['pos'][1] *= -1
    w, x, y, z = result['quat']
    result['quat'] = [w, -x, y, -z]
    if 'lin_vel_world' in result:
        result['lin_vel_world'][1] *= -1
    if 'ang_vel_world' in result:
        x, y, z = result['ang_vel_world']
        result['ang_vel_world'] = [-x, y, -z]
    return result


@dataclass(frozen=True)
class HandProfile:
    side: str
    asset_path: Path
    source_prim: str
    contract_path: Path
    contact_scene: Path
    joint_prefix: str = ''
    body_prefix: str = ''
    mirror_world_y: bool = False
    mirror_hand_root: bool = False
    parallel_asset_name: str = 'boya_parallel.usda'
    validation_scene: Path | None = None
    model_source: str = MODEL_SOURCE
    model_revision: str = MODEL_REVISION
    control_source: str = CONTROL_SOURCE

    def __post_init__(self):
        if self.side not in SIDES:
            raise ValueError('Hand side must be right or left')
        if not self.source_prim.startswith('/'):
            raise ValueError('Hand source prim must be an absolute USD path')
        for path in (self.asset_path, self.contract_path, self.contact_scene):
            if not Path(path).is_file():
                raise FileNotFoundError(str(path))
        if not self.parallel_asset_name.endswith(('.usda', '.usd')):
            raise ValueError('Parallel asset name must be USD')
        if self.validation_scene is not None and not Path(self.validation_scene).is_file():
            raise FileNotFoundError(str(self.validation_scene))

    def runtime_joint_name(self, canonical):
        return f'{self.joint_prefix}{canonical}'

    def canonical_body_name(self, runtime):
        if self.body_prefix and runtime.startswith(self.body_prefix):
            return runtime[len(self.body_prefix):]
        return runtime

    def runtime_names(self, names):
        return [self.runtime_joint_name(name) for name in names]

    def adapt_contract(self, contract):
        result = deepcopy(contract)
        if self.mirror_hand_root:
            result['hand_root'] = _mirror_wxyz_pose(result['hand_root'])
        if self.mirror_world_y:
            result['object'] = _mirror_wxyz_pose(result['object'])
            result['scene'] = f'{self.side}_mirror_y'
        return result

    def adapt_scene_configuration(self, spec):
        """Return the same right-hand contract or a deterministic left mirror."""
        if not self.mirror_world_y:
            return spec
        if spec.get('hand_side') == self.side:
            return spec
        result = deepcopy(spec)
        result['name'] = f"{result['name']}_{self.side}_mirror_y_v1"
        result['hand_side'] = self.side
        result['source_configuration_sha256'] = result.pop(
            'configuration_sha256', None) or _json_sha256(result)
        result['canonical_reference']['pos'][1] *= -1
        x, y, z, w = result['canonical_reference']['quat_xyzw']
        result['canonical_reference']['quat_xyzw'] = [-x, y, -z, w]
        state = result['nominal_state']
        state['joint_names'] = self.runtime_names(state['joint_names'])
        state['motor_names'] = self.runtime_names(state['motor_names'])
        import numpy as np
        state['object_state_xyzw_world_twist'] = mirror_object_state_y(
            np.asarray(state['object_state_xyzw_world_twist'], dtype=float)).tolist()
        result['evaluation_initial_state'] = (
            f"{result.get('evaluation_initial_state', 'configured nominal state')} "
            f"[{self.side} mirror-y profile]")
        result['configuration_sha256'] = _json_sha256(result)
        return result

    def adapt_object_state(self, state):
        return mirror_object_state_y(state) if self.mirror_world_y else state

    def adapt_policy_position(self, position):
        """Map physical world position back to the right-hand policy frame."""
        if not self.mirror_world_y:
            return position
        result = position.clone() if hasattr(position, 'clone') else position.copy()
        result[..., 1] *= -1
        return result

    def adapt_policy_privileged(self, privileged):
        """Canonicalize Hora's [position(3), scale, mass, friction, COM(3)]."""
        if privileged.shape[-1] != 9:
            raise ValueError('Expected nine Hora privileged channels')
        if not self.mirror_world_y:
            return privileged
        result = privileged.clone() if hasattr(privileged, 'clone') else privileged.copy()
        result[..., 1] *= -1
        result[..., 7] *= -1
        return result

    def cache_names_match(self, saved_joints, runtime_joints):
        return self.runtime_names([str(name) for name in saved_joints]) == [
            str(name) for name in runtime_joints]

    def motor_names_match(self, saved_motors, contract):
        expected = [a['joint'] for a in contract['actuators']]
        return [str(name) for name in saved_motors] == expected

    def asset_dependencies(self):
        dependencies = [self.asset_path]
        companion = self.asset_path.with_name(
            f'{self.asset_path.stem}_geometry{self.asset_path.suffix}')
        if companion.is_file():
            dependencies.append(companion)
        return dependencies

    def identity(self, include_asset_hash=True):
        value = dict(schema='tendonspin-hand-profile-v1', side=self.side,
            asset_path=str(self.asset_path), source_prim=self.source_prim,
            contract_path=str(self.contract_path), contact_scene=str(self.contact_scene),
            joint_prefix=self.joint_prefix, body_prefix=self.body_prefix,
            mirror_world_y=self.mirror_world_y,
            mirror_hand_root=self.mirror_hand_root, model_source=self.model_source,
            model_revision=self.model_revision, control_source=self.control_source)
        if include_asset_hash:
            value['contact_scene_sha256'] = _sha256(self.contact_scene)
        if self.validation_scene is not None:
            value['validation_scene'] = str(self.validation_scene)
            if include_asset_hash:
                value['validation_scene_sha256'] = _sha256(self.validation_scene)
        if include_asset_hash:
            value['asset_sha256'] = _sha256(self.asset_path)
            value['asset_dependencies'] = {
                str(path): _sha256(path) for path in self.asset_dependencies()}
        return value


def resolve_hand_profile(root, side='right', left_asset=None):
    root = Path(root).resolve()
    side = str(side).lower()
    if side not in SIDES:
        raise ValueError('hand_side must be right or left')
    contract = root/'docs/data/boya_native_contract.json'
    contact_scene = root/'assets/grasp/scene.xml'
    if side == 'right':
        imported = json.loads((root/'docs/data/isaac_boya_import_phases.json').read_text())
        return HandProfile(side='right', asset_path=root/imported['usd']['path'],
            source_prim='/dexhand', contract_path=contract, contact_scene=contact_scene)
    # An explicit missing asset must fail, never silently select another model.
    # Likewise, a missing strict mirror must not silently fall back to vendor CAD.
    selected = left_asset if left_asset is not None else os.environ.get(LEFT_HAND_ENV)
    resolved = Path(selected).expanduser() if selected is not None else root/LEFT_HAND_RELATIVE
    if not resolved.is_absolute():
        resolved = root/resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f'Left-hand USD not found: {resolved}. Generate/import the strict '
            'mirror, or explicitly select a left_asset.')
    strict_mirror = resolved == (root/LEFT_HAND_RELATIVE).resolve()
    source_prim = '/World/Hand'
    try:
        from pxr import Usd
        stage = Usd.Stage.Open(str(resolved))
        default = stage.GetDefaultPrim() if stage else None
        if default:
            source_prim = str(default.GetPath())
            strict_mirror = strict_mirror or default.GetName() == 'dexhand_strict_mirror_left'
    except Exception:
        pass
    vendor_validation = root/'assets/hand/ros2_botyard_39f63b1/mjcf/gs_s1/scene_left.xml'
    validation_candidates = (
        (root/LEFT_VALIDATION_SCENE,) if strict_mirror
        else (vendor_validation,
              resolved.parents[2]/'mjcf/gs_s1/scene_left.xml'))
    validation_scene = next(
        (path for path in validation_candidates if path.is_file()), None)
    left_contact_scene = (
        root/LEFT_CONTACT_SCENE if strict_mirror
        else root/'assets/hand/botyard_v3_2/mjcf/v3_2/left_dexhand.xml')
    if not left_contact_scene.is_file():
        left_contact_scene = contact_scene
    return HandProfile(side='left', asset_path=resolved, source_prim=source_prim,
        contract_path=contract, contact_scene=left_contact_scene,
        joint_prefix='left_', body_prefix='left_', mirror_world_y=True,
        mirror_hand_root=strict_mirror,
        parallel_asset_name='boya_left_parallel.usda',
        validation_scene=validation_scene,
        model_source=LEFT_MODEL_SOURCE if strict_mirror else VENDOR_LEFT_MODEL_SOURCE)
