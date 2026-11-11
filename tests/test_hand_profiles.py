"""Hand-side profiles select assets without duplicating the task environment."""
from copy import deepcopy
import hashlib
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tendonspin.physics.hand_profiles import HandProfile, mirror_object_state_y, resolve_hand_profile
from tendonspin.physics.hand_asset_audit import audit_hand_asset
from tendonspin.physics.left_grasp import LeftGraspCalibrator, left_local_object_state


ROOT = Path(__file__).resolve().parents[1]


class HandProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)/'tendon-spin'
        (self.root/'docs/data').mkdir(parents=True)
        (self.root/'assets/grasp').mkdir(parents=True)
        self.asset = self.root/'right.usda'
        self.asset.write_text('#usda 1.0\n')
        self.contract = self.root/'docs/data/boya_native_contract.json'
        self.contract.write_text('{}\n')
        self.scene = self.root/'assets/grasp/scene.xml'
        self.scene.write_text('<mujoco/>\n')

    def tearDown(self):
        self.temporary.cleanup()

    def profile(self, side='left'):
        return HandProfile(side=side, asset_path=self.asset, source_prim='/World/Hand',
            contract_path=self.contract, contact_scene=self.scene,
            joint_prefix='left_' if side == 'left' else '',
            body_prefix='left_' if side == 'left' else '',
            mirror_world_y=side == 'left',
            mirror_hand_root=side == 'left',
            model_source=('botyard_v3_2_strict_right_mirror' if side == 'left'
                          else 'botyard_v3_2_right'))

    def test_right_contract_is_returned_unchanged(self):
        spec = dict(schema='tendonspin-learning-scene-v1', configuration_sha256='source')
        profile = self.profile('right')
        self.assertIs(profile.adapt_scene_configuration(spec), spec)
        self.assertEqual(profile.runtime_names(['FFJ2']), ['FFJ2'])

    def test_left_contract_prefixes_runtime_names_and_preserves_source(self):
        spec = dict(schema='tendonspin-learning-scene-v1', name='scene',
            configuration_sha256='source',
            geometry=dict(shape='cylinder'), canonical_reference=dict(
                pos=[-0.3, 0.02, 0.1], quat_xyzw=[0.1, 0.2, 0.3, 0.9]),
            nominal_state=dict(joint_names=['FAJ1', 'FFJ2'],
                motor_names=['FAJ1', 'FFJ2'],
                object_state_xyzw_world_twist=[
                    -0.3, 0.02, 0.1, 0.1, 0.2, 0.3, 0.9,
                    1., 2., 3., 4., 5., 6.]),
            evaluation_initial_state='fixed', initial_state_atol={})
        original = deepcopy(spec)
        adapted = self.profile('left').adapt_scene_configuration(spec)
        self.assertIs(self.profile('left').adapt_scene_configuration(adapted), adapted)
        self.assertEqual(spec, original)
        self.assertEqual(adapted['hand_side'], 'left')
        self.assertEqual(adapted['source_configuration_sha256'], 'source')
        self.assertEqual(adapted['canonical_reference']['pos'][1], -0.02)
        self.assertEqual(adapted['nominal_state']['joint_names'], ['left_FAJ1', 'left_FFJ2'])
        self.assertEqual(adapted['nominal_state']['motor_names'], ['left_FAJ1', 'left_FFJ2'])
        state = adapted['nominal_state']['object_state_xyzw_world_twist']
        self.assertEqual(state[1], -0.02)
        self.assertEqual(state[3], -0.1)
        self.assertEqual(state[5], -0.3)

    def test_xyzw_state_mirror_covers_pose_and_twist(self):
        state = np.zeros((2, 13), dtype=np.float64)
        state[:, 1] = 2.
        state[:, 3] = 0.1
        state[:, 5] = 0.3
        state[:, 8] = 5.
        state[:, 10] = 7.
        state[:, 12] = 9.
        mirrored = mirror_object_state_y(state)
        np.testing.assert_array_equal(mirrored[:, 1], -2.)
        np.testing.assert_array_equal(mirrored[:, 3], -0.1)
        np.testing.assert_array_equal(mirrored[:, 5], -0.3)
        np.testing.assert_array_equal(mirrored[:, 8], -5.)
        np.testing.assert_array_equal(mirrored[:, 10], -7.)
        np.testing.assert_array_equal(mirrored[:, 12], -9.)
        np.testing.assert_array_equal(state[:, 1], 2.)

    def test_policy_position_is_returned_to_right_frame(self):
        position = np.asarray([[0.1, 0.2, 0.3]])
        mirrored = self.profile('left').adapt_policy_position(position)
        np.testing.assert_array_equal(mirrored, [[0.1, -0.2, 0.3]])
        np.testing.assert_array_equal(position, [[0.1, 0.2, 0.3]])
        self.assertIs(self.profile('right').adapt_policy_position(position), position)

    def test_cache_names_are_checked_in_runtime_namespace(self):
        profile = self.profile('left')
        self.assertTrue(profile.cache_names_match(
            ['FAJ1', 'FFJ2'], ['left_FAJ1', 'left_FFJ2']))
        self.assertFalse(profile.cache_names_match(
            ['FAJ1', 'FFJ2'], ['FFJ2', 'FAJ1']))
        contract = dict(actuators=[dict(joint='FAJ1'), dict(joint='FFJ2')])
        self.assertTrue(profile.motor_names_match(
            ['FAJ1', 'FFJ2'], contract))

    def test_left_contract_preserves_right_control_parameters(self):
        contract = json.loads((ROOT/'docs/data/boya_native_contract.json').read_text())
        adapted = self.profile('left').adapt_contract(contract)
        self.assertNotIn('control_source', adapted)
        self.assertNotIn('asset_controls', adapted)
        self.assertEqual(adapted['joints'], contract['joints'])
        self.assertEqual(adapted['actuators'], contract['actuators'])
        self.assertIsNot(adapted['joints'], contract['joints'])
        identity = self.profile('left').identity(include_asset_hash=False)
        self.assertEqual(identity['control_source'], 'right_tendonspin_contract')
        self.assertEqual(identity['model_source'],
                         'botyard_v3_2_strict_right_mirror')

    def test_left_profile_mirrors_hand_and_object_roots(self):
        root = dict(pos=[0.1, 0.2, 0.3],
                    quat=[0.5, 0.5, -0.5, -0.5])
        contract = dict(hand_root=deepcopy(root), object=deepcopy(root))
        adapted = self.profile('left').adapt_contract(contract)
        self.assertEqual(adapted['hand_root']['pos'], [0.1, -0.2, 0.3])
        self.assertEqual(adapted['hand_root']['quat'],
                         [0.5, -0.5, -0.5, 0.5])
        self.assertEqual(adapted['object']['pos'], [0.1, -0.2, 0.3])
        self.assertEqual(adapted['object']['quat'],
                         [0.5, -0.5, -0.5, 0.5])
        self.assertEqual(contract['hand_root'], root)
        self.assertEqual(contract['object'], root)

    def test_grasp_search_delta_is_bounded_by_contract_limits(self):
        lower, upper = LeftGraspCalibrator._delta_bounds(
            ['THJ4', 'LFJ4'], np.asarray([0.2, 0.2]),
            ('THJ4', 'LFJ4'),
            {'THJ4': [-1.0, 1.0], 'LFJ4': [-0.2618, 0.2618]})
        self.assertAlmostEqual(lower, -0.4618)
        self.assertAlmostEqual(upper, 0.0618)

    def test_left_local_state_applies_world_y_reflection_under_mirrored_root(self):
        root_wxyz = np.asarray([0.5, 0.5, -0.5, -0.5])
        mirrored_root_wxyz = np.asarray([0.5, -0.5, -0.5, 0.5])
        local_position = np.asarray([0.1, 0.2, 0.3])
        root = Rotation.from_quat(root_wxyz[[1, 2, 3, 0]])
        world = np.r_[root.apply(local_position), [0., 0., 0., 1.], np.zeros(6)]
        position, _ = left_local_object_state(mirrored_root_wxyz, world)
        np.testing.assert_allclose(position, [0.1, -0.2, 0.3], atol=1e-12)

    def test_explicit_left_asset_resolution(self):
        imported = self.root/'docs/data/isaac_boya_import_phases.json'
        imported.write_text(json.dumps(dict(usd=dict(path='right.usda'))))
        left = self.root/'left.usd'
        left.write_text('#usda 1.0\n')
        (self.root/'left_geometry.usd').write_text('#usda 1.0\n')
        profile = resolve_hand_profile(self.root, 'left', left)
        self.assertEqual(profile.asset_path, left.resolve())
        self.assertEqual(profile.source_prim, '/World/Hand')
        self.assertEqual(len(profile.identity()['asset_dependencies']), 2)
        with self.assertRaises(ValueError):
            resolve_hand_profile(self.root, 'middle', left)

    @unittest.skipUnless((ROOT/'assets/hand/mirror_right_v3_2/usd/left_dexhand/'
        'left_dexhand.usda').is_file(), 'mirrored left asset unavailable')
    def test_latest_left_asset_matches_action_contract(self):
        profile = resolve_hand_profile(ROOT, 'left')
        contract = json.loads(profile.contract_path.read_text())
        audit = audit_hand_asset(profile, contract)
        self.assertTrue(audit['all_pass'], audit['errors'])
        self.assertEqual(audit['action_names'], list(('THJ4', 'THJ3', 'THJ2', 'THJ1',
            'FFJ4', 'FFJ3', 'FFJ2', 'MFJ4', 'MFJ3', 'MFJ2', 'RFJ4', 'RFJ3',
            'RFJ2', 'LFJ4', 'LFJ3', 'LFJ2')))
        self.assertEqual(len(audit['contact_exclusions']), 28)
        self.assertEqual(profile.source_prim, '/dexhand_strict_mirror_left')
        self.assertEqual(profile.model_source, 'botyard_v3_2_strict_right_mirror')
        self.assertTrue(audit['warnings'][-1].startswith(
            'Left runtime uses the right TendonSpin control contract'))

    @unittest.skipUnless((ROOT/'assets/hand/source/v3_2_manifest.json').is_file(),
                         'V3.2 source manifest unavailable')
    def test_v3_2_source_files_are_pinned(self):
        manifest = json.loads((
            ROOT/'assets/hand/source/v3_2_manifest.json').read_text())
        self.assertEqual(manifest['source_revision'],
                         '28f4ba5649714c74c13f881891d579bfceb766e2')
        for name, expected in manifest['files'].items():
            path = ROOT/'assets/hand/source'/name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             expected, name)

    @unittest.skipUnless((ROOT/'assets/hand/mirror_right_v3_2/manifest.json').is_file(),
                         'strict mirror manifest unavailable')
    def test_strict_mirror_assets_are_pinned(self):
        manifest = json.loads((
            ROOT/'assets/hand/mirror_right_v3_2/manifest.json').read_text())
        for key in ('urdf', 'mjcf', 'validation_scene'):
            path = ROOT/manifest[key]['path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             manifest[key]['sha256'], key)
        self.assertEqual(len(manifest['meshes']), 24)


if __name__ == '__main__':
    unittest.main()
