# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import time
import os

import numpy as np
import torch
from collections.abc import Sequence

import carb
import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_conjugate, quat_mul, saturate

from .sharpa_wave_grasp_env_cfg import SharpaWaveEnvCfg
from .sharpa_wave_env import SharpaWaveInhandRotateEnv


class SharpaWaveInhandRotateGraspEnv(SharpaWaveInhandRotateEnv):
    def __init__(self, cfg: SharpaWaveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.saved_grasping_states = [torch.zeros((0, 29), dtype=torch.float32, device=self.device) for _ in range(self.cfg.scale_range[2])]
        self.replay_pose = None
        if cfg.replay_cache:
            replay = np.load(cfg.replay_cache)
            self.replay_pose = torch.tensor(replay[0], dtype=torch.float32, device=self.device)
            self.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))
        # Pose recorded while gravity points down, after the upward phase.
        self.down_pose = torch.zeros((self.num_envs, 29), dtype=torch.float32, device=self.device)
        self.has_down_pose = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.frozen = False
        self.frozen_env_id = None
        self.live_success = None
        self.gravity_id = 0
        self.gravity_all_directions = [
            carb.Float3(0.0, 0.0, 9.81),
            carb.Float3(0.0, 0.0, -9.81),
            carb.Float3(0.0, 9.81, 0.0),
            carb.Float3(0.0, -9.81, 0.0),
            carb.Float3(9.81, 0.0, 0.0),
            carb.Float3(-9.81, 0.0, 0.0),
        ]

    def _get_rewards(self) -> torch.Tensor:
        cond1 = (torch.norm(self.fingertip_pos - self.object_pos.unsqueeze(1), dim=-1, p=2) < 0.1).all(-1)
        filtered_force_matrix = torch.cat([self._contact_sensor[id].data.force_matrix_w[:, 0, 0, :].unsqueeze(1) for id in range(10)], dim=1)
        cond2 = (torch.norm(filtered_force_matrix, dim=-1, p=2) > 0.5).sum(-1) >= 3
        default_quat = self.object.data.default_root_state.torch[:, 3:7]
        cond3 = torch.less(quat_to_rot(quat_mul(self.object_rot, quat_conjugate(default_quat))), self.cfg.reset_angle_diff)
        cond = cond1.float() * cond2.float() * cond3.float()
        if not self.cfg.hold_pose and not self.frozen:
            self.reset_buf[cond < 1] = 1
            # gravity_id == 2 is the downward phase after the upward test.
            if self.gravity_id == 2:
                good = (cond > 0) & (self.episode_length_buf > 80)
                if torch.any(good):
                    state = torch.cat([self.hand_dof_pos, self.object_pos, self.object_rot], dim=1)
                    self.down_pose[good] = state[good]
                    self.has_down_pose[good] = True
        # Grasp search flips gravity through six axes, including straight up.
        # The viewer holds one pose, so leave gravity pointing down.
        if self.common_step_counter % 40 == 0 and not self.cfg.hold_pose and not self.frozen:
            self.physics_sim_view.set_gravity(self.gravity_all_directions[self.gravity_id])
            self.gravity_id += 1
            self.gravity_id %= len(self.gravity_all_directions)
        return 0

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        if self.frozen:
            self.episode_length_buf[env_ids] = 0
            return

        self._refresh_lab()
        success = (self.episode_length_buf == self.max_episode_length - 1) & self.has_down_pose
        if self.cfg.freeze_on_success and torch.any(success):
            winners = torch.nonzero(success.reshape(-1)).flatten()
            winner = int(winners[int(torch.randint(winners.numel(), (1,), device=winners.device))].item())
            self.frozen = True
            self.frozen_env_id = winner
            self.live_success = success.clone()
            self.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))
            self._focus_env(winner)
            self.episode_length_buf[:] = 0
            print(
                f"FROZEN env {winner}. live={int(success.sum().item())}. "
                "Camera moved there. Other envs stopped resampling.",
                flush=True,
            )
            return
        all_states = self.down_pose[success]
        saved_scale_ids = self.scale_ids[success].reshape(-1)
        target = max(int(self.cfg.max_grasps), 1)
        per_scale = max(target // self.cfg.scale_range[2], 1)
        sum_total = 0
        finish_scale = 0
        for state_id, saved_scale_id in enumerate(saved_scale_ids.tolist()):
            if self.saved_grasping_states[saved_scale_id].shape[0] < per_scale:
                state = all_states[state_id].reshape(1, 29)
                self.saved_grasping_states[saved_scale_id] = torch.cat([self.saved_grasping_states[saved_scale_id], state], dim=0)
        for saved_grasping_states in self.saved_grasping_states:
            if saved_grasping_states.shape[0] >= per_scale:
                finish_scale += 1
            sum_total += saved_grasping_states.shape[0]
        print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] current cache size: {sum_total}, finished: {finish_scale}', flush=True)
        if sum_total >= target:
            print('done!', flush=True)
            save_data = torch.zeros((0, 29), dtype=torch.float32, device=self.device)
            for saved_grasping_states in self.saved_grasping_states:
                save_data = torch.cat([save_data, saved_grasping_states], dim=0)
            os.makedirs('cache', exist_ok=True)
            name = f'cache/sharpa_grasp_linspace_{self.cfg.scale_range[0]}-{self.cfg.scale_range[1]}-{self.cfg.scale_range[2]}.npy'
            np.save(name, save_data.cpu().numpy())
            print(f'saved {name}', flush=True)
            exit()

        self.scene.reset(env_ids)

        # apply events such as randomization for environments that need a reset
        if self.cfg.events:
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # reset noise models
        if self.cfg.action_noise_model:
            self._action_noise_model.reset(env_ids)
        if self.cfg.observation_noise_model:
            self._observation_noise_model.reset(env_ids)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

        rand_floats = 2.0 * torch.rand((len(env_ids), self.num_hand_dofs), device=self.device) - 1.0
        joint_noise = 0.0 if self.cfg.hold_pose else 0.15
        
        # reset object
        if self.replay_pose is not None:
            root_pose = torch.zeros((len(env_ids), 7), device=self.device)
            root_pose[:, 0:3] = self.replay_pose[22:25] + self.scene.env_origins[env_ids]
            root_pose[:, 3:7] = self.replay_pose[25:29]
            self.object.write_root_pose_to_sim(root_pose, env_ids)
            self.object.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids)
        else:
            object_default_state = self.object.data.default_root_state.clone()[env_ids]
            object_default_state[:, :3] += self.scene.env_origins[env_ids]
            object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
            self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
            self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        self.rb_forces[env_ids, :] = 0.0

        self.reset_height_lower[env_ids] = self.cfg.reset_height_lower
        self.reset_height_upper[env_ids] = self.cfg.reset_height_upper

        # reset hand
        if self.replay_pose is not None:
            dof_pos = self.replay_pose[:22].unsqueeze(0).repeat(len(env_ids), 1)
            limits = self.hand.data.joint_pos_limits.torch.to(self.device)
            dof_pos = saturate(dof_pos, limits[env_ids, :, 0], limits[env_ids, :, 1])
        else:
            dof_pos = self.hand.data.default_joint_pos[env_ids] + joint_noise * rand_floats
            dof_pos = saturate(dof_pos, self.hand_dof_lower_limits[env_ids], self.hand_dof_upper_limits[env_ids],)
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self._refresh_lab()

        self.object_pos_prev[env_ids] = self.object_pos[env_ids]
        self.object_rot_prev[env_ids] = self.object_rot[env_ids]

        # reset data buffers
        self.last_contacts[env_ids] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.at_reset_buf[env_ids] = 1
        self.has_down_pose[env_ids] = False

    def _focus_env(self, env_id: int) -> None:
        origin = self.scene.env_origins[env_id]
        look = (origin + self.object_pos[env_id]).detach().cpu()
        eye = look + torch.tensor([0.126, 0.125, 0.361])
        sim_utils.SimulationContext.instance().set_camera_view(
            tuple(float(v) for v in eye.tolist()),
            tuple(float(v) for v in look.tolist()),
        )


@torch.jit.script
def quat_to_rot(quaternion: torch.Tensor):
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)
    # Lab 3 quaternions are (x, y, z, w).
    w = quaternion[:, 3].clamp(-1.0, 1.0)
    angle = 2 * torch.acos(w.abs())
    return angle
