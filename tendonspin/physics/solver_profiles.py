# Sources: TendonSpin effective-zero diagnostic, commit86cf44f; installed Isaac
# Lab3.0.0rc1 PhysxCfg max-actor/clamp semantics and PhysxSchema110.3.2.
# Port: share the declared16/0 configuration across learning and evaluation.
# No solver, collision, actuator, reward or task implementation is replaced.
"""Explicit solver identities and composed-USD verification for velocity0."""
import hashlib
import json
import math

ORIGINAL = 'original_tgs16_4'
VELOCITY_ZERO = 'tgs16_velocity0'
PROFILES = (ORIGINAL, VELOCITY_ZERO)


def configuration(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown solver profile: '+str(profile))
    zero = profile == VELOCITY_ZERO
    return dict(profile=profile, solver_type='TGS', position_min=16, position_max=255,
        velocity_min=0 if zero else 4, velocity_max=0 if zero else 255,
        articulation_position=16, articulation_velocity=0 if zero else 4,
        rigid_body_position=16, rigid_body_velocity=0 if zero else 1,
        effective_velocity_iterations=0 if zero else 4,
        evidence='composed USD plus documented max-actor request clamped to scene bounds',
        direct_kernel_iteration_counter=False)


def configuration_json(value):
    """Encode nonfinite authored configuration limits, never physical state."""
    if isinstance(value, dict):
        return {str(k): configuration_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [configuration_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def author_zero_velocity(stage):
    from pxr import PhysxSchema, UsdPhysics
    counts = dict(rigid_bodies=0, articulations=0)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSolverVelocityIterationCountAttr().Set(0)
            counts['rigid_bodies'] += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateSolverVelocityIterationCountAttr().Set(0)
            counts['articulations'] += 1
    return counts


def audit_zero_velocity(stage, scene_path, num_envs):
    from pxr import UsdPhysics
    scene = stage.GetPrimAtPath(scene_path)
    expected = {'physxScene:solverType': 'TGS', 'physxScene:minPositionIterationCount': 16,
        'physxScene:maxPositionIterationCount': 255, 'physxScene:minVelocityIterationCount': 0,
        'physxScene:maxVelocityIterationCount': 0, 'physxScene:enableCCD': False}
    for name, value in expected.items():
        actual = scene.GetAttribute(name).Get()
        if actual != value:
            raise RuntimeError(f'Solver scene request mismatch: {name}={actual}, expected{value}')
    requests = []
    per_env = {f'env_{i}': dict(rigid_bodies=0, articulations=0) for i in range(num_envs)}
    for prim in stage.Traverse():
        kinds = []
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            kinds.append(('rigid_bodies', 'physxRigidBody'))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            kinds.append(('articulations', 'physxArticulation'))
        for kind, namespace in kinds:
            path = str(prim.GetPath())
            components = path.split('/')
            if len(components) < 5 or components[3] not in per_env:
                raise RuntimeError('Unexpected physics actor outside declared environments: '+path)
            position = prim.GetAttribute(namespace+':solverPositionIterationCount').Get()
            velocity = prim.GetAttribute(namespace+':solverVelocityIterationCount').Get()
            if (position, velocity) != (16, 0):
                raise RuntimeError(f'Actor solver request mismatch: {path} {namespace} {(position, velocity)}')
            per_env[components[3]][kind] += 1
            requests.append((path, namespace, position, velocity))
    if any(counts != dict(rigid_bodies=25, articulations=1) for counts in per_env.values()):
        raise RuntimeError('Expected25rigid bodies/1articulation per environment: '+str(per_env))
    return dict(num_envs=num_envs, rigid_bodies=25*num_envs, articulations=num_envs,
        every_environment_verified=True, rigid_bodies_per_env=25, articulations_per_env=1,
        actor_position_requests=[16], actor_velocity_requests=[0],
        actor_requests_sha256=hashlib.sha256(json.dumps(sorted(requests)).encode()).hexdigest(),
        scene_physics_attributes=configuration_json({a.GetName(): a.Get() for a in scene.GetAttributes()
            if a.GetName().startswith('physxScene:') and isinstance(a.Get(), (str, int, float, bool))}))
