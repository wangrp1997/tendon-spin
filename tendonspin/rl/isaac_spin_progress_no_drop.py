"""D environment: unchanged C physics/reset/control, no explicit drop penalty."""
from tendonspin.rl.isaac_spin_progress import SpinProgressBoyaEnv
from tendonspin.rl.spin_progress_reward import PROFILE as C_PROFILE
from tendonspin.rl.spin_progress_no_drop import PROFILE, SpinProgressNoDropRewardSignal


class SpinProgressNoDropBoyaEnv(SpinProgressBoyaEnv):
    def __init__(self, *args, reward_profile=PROFILE, **kwargs):
        if reward_profile != PROFILE:
            raise ValueError('D requires its explicit zero-drop self-spin profile')
        super().__init__(*args, reward_profile=C_PROFILE, **kwargs)
        self.rotation_reward_signal = SpinProgressNoDropRewardSignal()
        self.reward_configuration = dict(self.rotation_reward_signal.configuration,
            integration='Native D training and randomized evaluation through inherited C control_reward')
