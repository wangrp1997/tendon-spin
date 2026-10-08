# Sources: declared native/Isaac quaternion boundary; analytic rotation fixtures.
"""Catch wrong conventions using known geometry and the saved original grasp."""
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from tendonspin.physics.coordinates import (wxyz_to_xyzw, xyzw_to_wxyz,
    axis_z_xyzw, orientation_error_deg, audit_initial_state)


class CoordinateTests(unittest.TestCase):
    def test_known_rotation_moves_cylinder_axis(self):
        wxyz = np.array([np.sqrt(.5), 0., np.sqrt(.5), 0.])
        xyzw = wxyz_to_xyzw(wxyz)
        np.testing.assert_allclose(axis_z_xyzw(xyzw), [1., 0., 0.], atol=1e-14)
        np.testing.assert_allclose(Rotation.from_quat(xyzw).apply([0,0,1]), [1,0,0], atol=1e-14)
        np.testing.assert_array_equal(xyzw_to_wxyz(xyzw), wxyz)

    def test_original_hand_regression_detects_180_degree_wrong_order(self):
        root = Path(__file__).resolve().parents[1]
        contract = json.loads((root/'docs/data/boya_native_contract.json').read_text())
        quat = contract['hand_root']['quat']
        self.assertLess(float(orientation_error_deg(wxyz_to_xyzw(quat), quat)), 1e-5)
        self.assertAlmostEqual(float(orientation_error_deg(quat, quat)), 180., places=3)
        np.testing.assert_allclose(axis_z_xyzw(wxyz_to_xyzw(quat)), [-1.,0.,0.], atol=1e-5)

    def test_initialization_rejects_wrong_order_and_accepts_quaternion_sign(self):
        root = Path(__file__).resolve().parents[1]
        c = json.loads((root/'docs/data/boya_native_contract.json').read_text())
        bodies = [b['name'] for b in c['body_frames'] if b['mass']>0.]
        lookup = {b['name']: b for b in c['body_frames']}
        state = dict(body_pose=np.array([lookup[n]['pos']+(-wxyz_to_xyzw(lookup[n]['quat'])).tolist() for n in bodies]),
            joint_pos=np.array([j['qpos'] for j in c['joints']]),
            joint_vel=np.array([j['qvel'] for j in c['joints']]),
            object_state=np.array(c['object']['pos']+wxyz_to_xyzw(c['object']['quat']).tolist()+c['object']['lin_vel_world']+c['object']['ang_vel_world']))
        names = [j['name'] for j in c['joints']]
        self.assertTrue(audit_initial_state(c,names,bodies,state)['passed'])
        state['body_pose'][0,3:7] = c['hand_root']['quat']
        out = audit_initial_state(c,names,bodies,state)
        self.assertFalse(out['passed'])
        self.assertFalse(out['checks']['body_orientations'])
