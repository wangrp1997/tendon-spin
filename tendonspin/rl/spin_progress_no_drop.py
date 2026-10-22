"""D ablation: native C self-spin and penalties, zero explicit drop cost.

All rotation arithmetic and nonrotation reward computation are inherited from C.
Workspace failure still terminates the episode; no holding/alive bonus is added.
"""
import json
from pathlib import Path

from tendonspin.rl.rotation_reward import POSE_DELTA, RotationRewardSignal
from tendonspin.rl.spin_progress_reward import (
    SpinProgressRewardSignal, configuration as c_configuration,
)

CONFIG = Path(__file__).resolve().parents[2] / 'assets/config/boya_spin_progress_no_drop_reward.json'
PROFILE = 'boya_object_spin_positive_cap_drop0'


def configuration(path=CONFIG):
    expected = c_configuration()
    expected.update(profile=PROFILE, terminal_failure_cost=0.)
    expected['port_changes'][-1] = 'Remove explicit Boya failure cost; retain workspace termination and original Hora other penalties'
    expected['integration'] = 'Native D ablation; inherited C rotation and nonrotation terms, zero terminal cost'
    cfg = json.loads(Path(path).read_text())
    if cfg != expected:
        raise ValueError('D must differ from C only in profile and terminal_failure_cost=0')
    return cfg


class SpinProgressNoDropRewardSignal(SpinProgressRewardSignal):
    def __init__(self, config_path=CONFIG):
        super().__init__()
        self.configuration = configuration(config_path)
        # POSE_DELTA supplies zero terminal cost; inherited C computes everything else.
        self.drop = RotationRewardSignal(POSE_DELTA,
            self.configuration['physics_dt_s'], self.steps_per_control)
