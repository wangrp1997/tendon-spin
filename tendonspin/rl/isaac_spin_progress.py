"""C teacher: existing randomized Boya physics/PPO with native self-spin reward."""
from tendonspin.rl.isaac_hora_randomized import RandomizedHoraBoyaEnv
from tendonspin.rl.rotation_reward import POSE_DROP
from tendonspin.rl.spin_progress_reward import PROFILE, SpinProgressRewardSignal


class SpinProgressBoyaEnv(RandomizedHoraBoyaEnv):
    def __init__(self, *args, reward_profile=PROFILE, termination_profile='boya_workspace', **kwargs):
        if reward_profile != PROFILE or termination_profile != 'boya_workspace':
            raise ValueError('C requires its self-spin profile and existing Boya workspace rules')
        super().__init__(*args, reward_profile=POSE_DROP,
            termination_profile=termination_profile, **kwargs)
        signal = SpinProgressRewardSignal()
        if (self.physics.dt, self.physics.adapter.steps_per_control) != (
            signal.configuration['physics_dt_s'], signal.steps_per_control):
            raise ValueError('C reward timing differs from the unchanged physics contract')
        self.rotation_reward_signal = signal
        self.reward_configuration = dict(signal.configuration,
            integration='Native C training and frozen evaluation through SpinProgressBoyaEnv.control_reward')

    def control_reward(self, object_linvel, reported_velocity, pose, torque, work, failure_codes):
        components = self.rotation_reward_signal.compute_reward(
            object_linvel, pose, torque, work, failure_codes)
        components['base_reward'] = components['nonrotation_reward'] + components['rotation']
        components['rotation_velocity'] = self.rotation_reward_signal.angular_speed()
        return components
