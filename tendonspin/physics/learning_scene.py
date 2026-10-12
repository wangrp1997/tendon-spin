# Reuse: TendonSpin a96c9a6 verify_boya_native_cache_isaac.py, BoyaParallel
# state restoration and object_geometry.configure_object_scene (MIT).
# Port change: optional frozen geometry/nominal/reference shared by train/eval;
# original defaults, actuator parameters and task bounds are preserved.
"""Explicit scene identity for learning from a separately migrated grasp cache."""
import hashlib
import json
from pathlib import Path


def load_scene_configuration(path):
    path = Path(path)
    spec = json.loads(path.read_text())
    if spec['schema'] != 'tendonspin-learning-scene-v1':
        raise ValueError('Unknown learning scene schema')
    spec['configuration_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return spec


def configure_nominal(physics, spec):
    import torch
    from .isaac_parallel import axis_z
    p = physics
    state = spec['nominal_state']
    if state['joint_names'] != p.names or state['motor_names'] != [a['joint'] for a in p.contract['actuators']]:
        raise ValueError('Learning scene joint/motor order differs from Isaac')
    def batch(values):
        return torch.tensor([values], device=p.device, dtype=torch.float32).repeat(p.num_envs, 1)
    p.q0 = batch(state['joint_pos'])
    p.v0 = batch(state['joint_vel'])
    p.object0 = batch(state['object_state_xyzw_world_twist'])
    p.object0[:, :3] += p.origins
    p.commands0 = batch(state['commands'])
    p.center = batch(spec['canonical_reference']['pos']) + p.origins
    p.axis0 = axis_z(batch(spec['canonical_reference']['quat_xyzw']))
    p.motor_bias = p.commands0 - p.q0[:, p.adapter.ids]
    p.reset_nominal()
    p.scene_configuration = spec
    return nominal_errors(p)


def nominal_errors(p):
    """Read back the fixed scene state; tolerances are numerical, not task gates."""
    m = p.measure()
    expected = dict(joint_pos=p.q0, joint_vel=p.v0, object_state=p.object0,
                    commands=p.commands0)
    actual = dict(m, commands=p.adapter.commands)
    errors = {key: float((actual[key] - value).abs().max()) for key, value in expected.items()}
    limits = p.scene_configuration['initial_state_atol']
    if any(errors[key] > limits[key] for key in errors):
        raise ValueError('Configured nominal state readback differs: ' + str(errors))
    return errors
