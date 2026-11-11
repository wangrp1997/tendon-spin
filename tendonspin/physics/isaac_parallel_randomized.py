# Derived from TendonSpin isaac_parallel.py at8cca30e (MIT); same Boya
# CAD/solver/actuator/timing. Only scene replication is disabled, as required by
# Isaac Lab3 randomize_rigid_body_scale for per-environment geometry.
# Existing driver and historical checkpoint source identity remain unchanged.
# References: Sharpa RL Lab 5accf024d376685eaa17da7aa4614498217eab4d
# sharpa_wave_env.py::_setup_scene/reset and Isaac Lab3 InteractiveScene (BSD-3-Clause).
# Reuse: TendonSpin Boya v3 dynamics and v4 16-input adapter. New batched scene,
# explicit source excludes in a derived USD layer, no original asset mutations.
# Optional solver profile reuses the declared effective-zero diagnostic86cf44f;
# default original16/4 dynamics are retained, and0 is separately recorded.
"""Boya hands with independently parsed sized objects and the same16 inputs."""
from .isaac_parallel import BoyaParallel, tensor
from .solver_profiles import ORIGINAL


class BoyaRandomizedParallel(BoyaParallel):
    def __init__(self, root, out, num_envs=64, scene_setup=None, engine_profile=ORIGINAL,
                 hand_profile=None):
        super().__init__(root, out, num_envs, scene_setup=scene_setup,
            engine_profile=engine_profile, hand_profile=hand_profile,
            replicate_physics=False)

    def advance(self):
        if hasattr(self,'domain_randomization'):
            dr=self.domain_randomization
            dr.before_physics_step()
            if dr.enabled and dr.first_force_readback is None:
                # One startup read before integration, no additional step.
                dr.verify_forwarded_force()
        return super().advance()

    def measure(self):
        values=super().measure()
        if hasattr(self,'domain_randomization'):
            dr=self.domain_randomization
            values.update(external_force_requested_w=dr.force,object_mass_kg=dr.mass,
                object_com_b_m=dr.com,object_friction=dr.friction,object_logical_scale=dr.scale,
                p_gain_ratio=dr.p_ratio,d_gain_ratio=dr.d_ratio)
        return values

