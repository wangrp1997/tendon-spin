"""Scoped host-readback synchronization for the audited Isaac Lab PhysX wheel.

Warp copies to pinned CPU arrays are asynchronous. The audited asset setters
immediately pass these arrays to CPU PhysX. Wait on that copy's stream before
returning to those three callers; leave other Warp operations untouched.
No installed file, solver, property value or training source is rewritten.
"""
import hashlib
import importlib.util
from pathlib import Path
import sys

EXPECTED_SOURCE_SHA256 = '0fd17152d133805412776d8cc90eebe85159a383a2e052dafc2eafdd14e54390'
SETTERS = frozenset(('set_masses_index', 'set_coms_index', '_get_cpu_env_ids'))


def synchronized_copy(original, synchronize, source_filename, on_wait):
    """Keep GPU work asynchronous except at the audited CPU-consumer boundary."""
    def copy(dest, src, *args, **kwargs):
        result = original(dest, src, *args, **kwargs)
        caller = sys._getframe(1).f_code
        if (caller.co_filename == source_filename and caller.co_name in SETTERS
                and dest.device.is_cpu and src.device.is_cuda):
            synchronize(kwargs.get('stream') or src.device.stream)
            on_wait(caller.co_name)
        return result
    return copy


def audited_source():
    spec = importlib.util.find_spec('isaaclab_physx')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('Audited Isaac Lab PhysX package is unavailable')
    path = Path(next(iter(spec.submodule_search_locations)))/'assets/rigid_object/rigid_object.py'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError('PhysX setter implementation differs from the diagnosed wheel; stop without substitution')
    return path.resolve(), sha


def install(on_wait):
    import warp as wp
    path, sha = audited_source()
    if getattr(wp.copy, '_tendonspin_property_sync', False):
        raise RuntimeError('Property synchronization adapter is already installed')
    original = wp.copy
    copy = synchronized_copy(original, wp.synchronize_stream, str(path), on_wait)
    copy._tendonspin_property_sync = True
    wp.copy = copy
    return dict(schema='tendonspin-physx-property-host-sync-v1', backend_source=str(path),
        backend_source_sha256=sha, installed_file_modified=False,
        scope='Wait after existing pinned CPU D2H mass/COM/environment-ID copies before CPU PhysX consumes them',
        unchanged='PhysX backend/solver, requested properties, geometry, cache, reward, PPO, original source contracts')
