"""Generate a strict Y-mirrored Boya V3.2 left hand from the right source.

The vendor also publishes a left CAD model, but its link masses, inertias and
kinematic offsets are not an exact mirror of the right model used to train the
C1260M teacher. This generator preserves every right-hand physical parameter
and reflects geometry, frames and joint axes across the world Y axis.
"""
from hashlib import sha256
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URDF = ROOT/'assets/hand/source/right_dexhand.urdf'
SOURCE_MJCF = ROOT/'assets/hand/source/right_dexhand.xml'
SOURCE_MESHES = ROOT/'assets/hand/meshes'
OUT = ROOT/'assets/hand/mirror_right_v3_2'
MIRROR = np.diag([1., -1., 1.])


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def mirror_position(values):
    result = np.asarray([float(value) for value in values], dtype=float)
    result[1] *= -1
    return result


def mirror_rotation_rpy(values):
    rotation = Rotation.from_euler('xyz', [float(value) for value in values])
    reflected = MIRROR @ rotation.as_matrix() @ MIRROR
    return Rotation.from_matrix(reflected).as_euler('xyz')


def mirror_quaternion_wxyz(values):
    w, x, y, z = (float(value) for value in values)
    return np.asarray([w, -x, y, -z])


def mirror_axis(values):
    # Rotation axes are pseudovectors under reflection.
    return -MIRROR @ np.asarray([float(value) for value in values])


def set_vector(element, attribute, values):
    if element is not None:
        element.set(attribute, ' '.join(f'{value:.12g}' for value in values))


def mirror_origin(origin):
    if origin is None:
        return
    set_vector(origin, 'xyz', mirror_position(origin.get('xyz', '0 0 0').split()))
    set_vector(origin, 'rpy', mirror_rotation_rpy(origin.get('rpy', '0 0 0').split()))


def mirror_stl(source, destination):
    raw = Path(source).read_bytes()
    if len(raw) < 84:
        raise ValueError(f'Invalid STL: {source}')
    triangle_count = struct.unpack_from('<I', raw, 80)[0]
    record = np.dtype([('normal', '<3f4'), ('vertices', '<9f4'), ('attribute', '<u2')])
    expected = 84 + triangle_count*record.itemsize
    if len(raw) != expected:
        raise ValueError(f'Expected binary STL: {source}')
    triangles = np.frombuffer(raw, dtype=record, count=triangle_count, offset=84)
    transformed = np.empty(triangle_count, dtype=record)
    transformed['normal'] = triangles['normal'] * np.asarray([1., -1., 1.])
    vertices = triangles['vertices'].reshape(-1, 3, 3).copy()
    vertices[:, :, 1] *= -1
    # Reflection reverses winding; restore the original outward orientation.
    vertices = vertices[:, [0, 2, 1], :]
    transformed['vertices'] = vertices.reshape(-1, 9)
    transformed['attribute'] = triangles['attribute']
    header = f'Strict Y-mirror generated from {Path(source).name}'.encode()[:80].ljust(80, b'\0')
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_bytes(header+struct.pack('<I', triangle_count)+transformed.tobytes())


def mirror_urdf():
    tree = ET.parse(SOURCE_URDF)
    robot = tree.getroot()
    robot.set('name', 'dexhand_strict_mirror_left')
    links = {link.get('name'): (
        'world' if link.get('name') == 'world' else 'left_'+link.get('name'))
        for link in robot.findall('link')}
    joints = {joint.get('name'): 'left_'+joint.get('name')
              for joint in robot.findall('joint')}
    for link in robot.findall('link'):
        link.set('name', links[link.get('name')])
        inertial = link.find('inertial')
        if inertial is not None:
            mirror_origin(inertial.find('origin'))
            inertia = inertial.find('inertia')
            if inertia is not None:
                inertia.set('ixy', str(-float(inertia.get('ixy'))))
                inertia.set('iyz', str(-float(inertia.get('iyz'))))
        for geometry in link.findall('visual/geometry')+link.findall('collision/geometry'):
            mesh = geometry.find('mesh')
            if mesh is not None:
                filename = mesh.get('filename')
                mesh.set('filename', 'meshes/'+Path(filename).name.lower())
        for origin in link.findall('visual/origin')+link.findall('collision/origin'):
            mirror_origin(origin)
    for joint in robot.findall('joint'):
        joint.set('name', joints[joint.get('name')])
        mirror_origin(joint.find('origin'))
        joint.find('parent').set('link', links[joint.find('parent').get('link')])
        joint.find('child').set('link', links[joint.find('child').get('link')])
        mimic = joint.find('mimic')
        if mimic is not None:
            mimic.set('joint', joints[mimic.get('joint')])
        axis = joint.find('axis')
        if axis is not None and joint.get('type') in ('revolute', 'continuous'):
            set_vector(axis, 'xyz', mirror_axis(axis.get('xyz', '1 0 0').split()))
    ET.indent(tree, space='  ')
    destination = OUT/'left_dexhand.urdf'
    tree.write(destination, encoding='utf-8', xml_declaration=True)


def mirror_mjcf():
    tree = ET.parse(SOURCE_MJCF)
    root = tree.getroot()
    root.set('model', 'botyard_v3_2_strict_left')
    compiler = root.find('compiler')
    compiler.set('meshdir', '../../meshes')
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', Path(mesh.get('file')).name.lower())
    for body in root.findall('.//body'):
        if 'euler' in body.attrib or body.get('name') == 'fabase':
            body.attrib.pop('euler', None)
            body.set('quat', '1 0 0 0')
        if 'pos' in body.attrib:
            set_vector(body, 'pos', mirror_position(body.get('pos').split()))
        if 'quat' in body.attrib and body.get('name') != 'fabase':
            set_vector(body, 'quat', mirror_quaternion_wxyz(body.get('quat').split()))
        inertial = body.find('inertial')
        if inertial is not None:
            if 'pos' in inertial.attrib:
                set_vector(inertial, 'pos', mirror_position(inertial.get('pos').split()))
            if 'quat' in inertial.attrib:
                set_vector(inertial, 'quat', mirror_quaternion_wxyz(inertial.get('quat').split()))
        for joint in body.findall('joint'):
            if 'axis' in joint.attrib:
                set_vector(joint, 'axis', mirror_axis(joint.get('axis').split()))
    ET.indent(tree, space='  ')
    destination = OUT/'mjcf/v3_2/left_dexhand.xml'
    tree.write(destination, encoding='utf-8', xml_declaration=True)


def validation_scene():
    return '''<?xml version="1.0"?>
<mujoco model="botyard_v3_2_strict_left_validation">
  <include file="left_dexhand.xml"/>
  <option timestep="0.001" gravity="0 0 -9.81"><flag contact="enable"/></option>
  <worldbody>
    <light pos="0 0 1"/>
    <body name="object" pos="-0.3 0.01 0.1">
      <freejoint/>
      <geom type="ellipsoid" size="0.03 0.04 0.02" condim="6" priority="1"
        friction="0.5 0.01 0.003"/>
    </body>
  </worldbody>
</mujoco>
'''


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'mjcf/v3_2').mkdir(parents=True)
    (OUT/'meshes').mkdir()
    mirrored_meshes = {}
    for source in sorted(SOURCE_MESHES.glob('*.stl')):
        destination = OUT/'meshes'/source.name.lower()
        mirror_stl(source, destination)
        mirrored_meshes[str(destination.relative_to(ROOT))] = digest(destination)
    mirror_urdf()
    mirror_mjcf()
    scene = OUT/'mjcf/v3_2/scene_left_validation.xml'
    scene.write_text(validation_scene())
    manifest = dict(
        schema='tendonspin-strict-right-mirror-v1',
        transform='world Y reflection; mesh winding restored; rotation axes mirrored as pseudovectors',
        source_urdf=dict(path=str(SOURCE_URDF.relative_to(ROOT)), sha256=digest(SOURCE_URDF)),
        source_mjcf=dict(path=str(SOURCE_MJCF.relative_to(ROOT)), sha256=digest(SOURCE_MJCF)),
        urdf=dict(path=str((OUT/'left_dexhand.urdf').relative_to(ROOT)),
                  sha256=digest(OUT/'left_dexhand.urdf')),
        mjcf=dict(path=str((OUT/'mjcf/v3_2/left_dexhand.xml').relative_to(ROOT)),
                  sha256=digest(OUT/'mjcf/v3_2/left_dexhand.xml')),
        validation_scene=dict(path=str(scene.relative_to(ROOT)), sha256=digest(scene)),
        meshes=mirrored_meshes,
        intent='Preserve every right-hand physical parameter; reflect only handedness.')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT), meshes=len(mirrored_meshes),
                          urdf=manifest['urdf']['sha256'], mjcf=manifest['mjcf']['sha256'])))


if __name__ == '__main__':
    main()
