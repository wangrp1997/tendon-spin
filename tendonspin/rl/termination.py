# Reference: HaozhiQi/hora v0.0.1 (MIT), AllegroHandHora.check_termination,
# _init_object_pose and configs/task/AllegroHandHora.yaml. Boya coordinate port;
# legacy strict limits are retained only as a separately named profile/diagnostic.
"""Shared training/evaluation termination, independent of Isaac imports."""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from omegaconf import OmegaConf

from tendonspin.baselines.hora_training import load_termination

REASONS = ('active', 'nonfinite', 'drift >5mm', 'tilt >15deg',
           'speed >100rad/s', 'link normal >12N', 'mimic error >.05rad',
           'episode time limit', 'object below Hora height')
PROFILES = ('legacy_strict', 'hora_height')


def legacy_failure_codes(measurement):
    """Original per-physics-step rules, also used for labeled shadow scoring."""
    m = measurement
    failures = (~m['finite'], m['drift_mm'] > 5., m['tilt_deg'] > 15.,
                m['max_speed'] > 100., m['max_normal'] > 12., m['coupling_error'] > .05)
    codes = torch.zeros_like(m['finite'], dtype=torch.long)
    for code, bad in enumerate(failures, 1):
        codes[(codes == 0) & bad] = code
    return codes


@dataclass(frozen=True)
class TerminationSpec:
    profile: str
    nominal_height_m: float
    reset_height_m: float
    episode_control_steps: int
    reference_nominal_height_m: float = .65
    reference_reset_height_m: float = .645
    task_check_timing: str = 'control boundary (20 Hz)'
    numerical_check_timing: str = 'each physics step'

    def to_dict(self):
        return asdict(self)


def make_termination_spec(root, profile, nominal_height_m):
    if profile not in PROFILES:
        raise ValueError(f'Unknown termination profile: {profile}')
    reference = OmegaConf.load(Path(root) / 'third_party/hora/configs/task/AllegroHandHora.yaml')
    reference_height = float(reference.env.reset_height_threshold)
    # Upstream rotation scene's nominal z=.65 (_init_object_pose), then resets
    # load cached states. Translate ONE fixed plane, never recenter per reset.
    # This is an explicit cross-hand coordinate assumption, not measured drop.
    height = float(nominal_height_m) + (reference_height - .65)
    return TerminationSpec(profile, float(nominal_height_m), height,
                           int(reference.env.episodeLength),
                           reference_reset_height_m=reference_height,
                           task_check_timing=('each physics step' if profile == 'legacy_strict'
                                              else 'control boundary (20 Hz)'))


class TaskTermination:
    """Use the upstream height/time method; keep numerical validity separate."""
    def __init__(self, spec):
        self.spec = spec
        self.reference_check = load_termination()
        self.reset_z_threshold = spec.reset_height_m
        self.max_episode_length = spec.episode_control_steps

    def failure_codes(self, measurement, origins, *, control_boundary):
        if self.spec.profile == 'legacy_strict':
            return legacy_failure_codes(measurement)
        m = measurement
        codes = torch.zeros_like(m['finite'], dtype=torch.long)
        # Retained Boya numerical diagnostics, not paper task-success gates.
        for code, bad in ((1, ~m['finite']), (4, m['max_speed'] > 100.),
                          (6, m['coupling_error'] > .05)):
            codes[(codes == 0) & bad] = code
        if control_boundary:
            self.progress_buf = torch.zeros_like(codes)
            below_height = self.reference_check(self, m['object_state'][:, :3] - origins)
            codes[(codes == 0) & below_height] = 8
        return codes

    def timeouts(self, progress, failure_codes):
        return (progress >= self.max_episode_length) & (failure_codes == 0)
