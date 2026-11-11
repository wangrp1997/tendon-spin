"""Independent FK/inertia/mesh checks against the actual trained right-hand URDF."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
S = np.diag([1., -1., 1.])
H = np.diag([1., -1., 1., 1.])


def origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        result[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    return result


def fk(robot, values, prefix=''):
    frames = {'world': np.eye(4)}
    pending = list(robot.findall('joint'))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent not in frames:
                continue
            local = origin(joint.find('origin'))
            if joint.get('type') != 'fixed':
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(axis*values[joint.get('name').removeprefix(prefix)]).as_matrix()
                local = local@motion
            frames[joint.find('child').get('link')] = frames[parent]@local
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError('Disconnected joint graph')
    return frames


class StrictMirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.right = ET.parse(ROOT/'assets/hand/boya.urdf').getroot()
        cls.left = ET.parse(ROOT/'assets/hand/mirror_right_v3_2/left_dexhand.urdf').getroot()

    def test_all_link_frames_mirror_for_random_joint_states(self):
        rng = np.random.default_rng(7711)
        joints = [j for j in self.right.findall('joint') if j.get('type') != 'fixed']
        for _ in range(32):
            values = {j.get('name'): rng.uniform(float(j.find('limit').get('lower')), float(j.find('limit').get('upper'))) for j in joints}
            right, left = fk(self.right, values), fk(self.left, values, 'left_')
            for name, frame in right.items():
                np.testing.assert_allclose(left['world' if name == 'world' else 'left_'+name], H@frame@H, atol=3e-11)

    def test_mass_inertia_limits_and_mimic_preserved(self):
        left_links = {link.get('name'): link for link in self.left.findall('link')}
        for right in self.right.findall('link'):
            if right.find('inertial') is None:
                continue
            left = left_links['left_'+right.get('name')]
            self.assertEqual(right.find('inertial/mass').attrib, left.find('inertial/mass').attrib)
            np.testing.assert_allclose(origin(left.find('inertial/origin')), H@origin(right.find('inertial/origin'))@H, atol=1e-11)
            def inertia(link):
                a = {key: float(value) for key, value in link.find('inertial/inertia').attrib.items()}
                return np.array([[a['ixx'],a['ixy'],a['ixz']], [a['ixy'],a['iyy'],a['iyz']], [a['ixz'],a['iyz'],a['izz']]])
            np.testing.assert_allclose(inertia(left), S@inertia(right)@S, atol=1e-15)
        left_joints = {joint.get('name'): joint for joint in self.left.findall('joint')}
        for right in self.right.findall('joint'):
            left = left_joints['left_'+right.get('name')]
            for name in ('limit', 'dynamics'):
                if right.find(name) is not None:
                    self.assertEqual(right.find(name).attrib, left.find(name).attrib)
            if right.find('mimic') is not None:
                expected = dict(right.find('mimic').attrib)
                expected['joint'] = 'left_'+expected['joint']
                self.assertEqual(left.find('mimic').attrib, expected)

    @unittest.skipUnless((ROOT/'assets/hand/mirror_right_v3_2/meshes').is_dir(),
                         'generated mirror meshes unavailable; run the mirror generator')
    def test_mesh_vertices_and_winding_are_actual_reflections(self):
        dtype = np.dtype([('normal', '<3f4'), ('vertices', '<9f4'), ('attribute', '<u2')])
        meshes = sorted((ROOT/'assets/hand/meshes').glob('*.stl'))
        self.assertEqual(len(meshes), 24)
        for path in meshes:
            right = np.frombuffer(path.read_bytes(), dtype=dtype, offset=84)
            left = np.frombuffer((ROOT/'assets/hand/mirror_right_v3_2/meshes'/path.name).read_bytes(), dtype=dtype, offset=84)
            vertices = right['vertices'].reshape(-1,3,3)[:,[0,2,1],:]*np.array([1,-1,1])
            np.testing.assert_array_equal(left['vertices'].reshape(-1,3,3), vertices)
            np.testing.assert_array_equal(left['normal'], right['normal']*np.array([1,-1,1]))
