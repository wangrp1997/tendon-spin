# Vendored reference: git@github.com:wangrp1997/sharpa-rl-lab.git (commit 5accf024d376685eaa17da7aa4614498217eab4d).
# Original relative file: rl_isaaclab/tasks/inhand_rotate/__init__.py.
# Added provenance header only; original copyright/license follows unchanged.
import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Inhand-Rotate-Sharpa-Wave-v0",
    entry_point=f"rl_isaaclab.tasks.inhand_rotate.sharpa_wave_env:SharpaWaveInhandRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_env_cfg:SharpaWaveEnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0",
    entry_point=f"rl_isaaclab.tasks.inhand_rotate.sharpa_wave_grasp_env:SharpaWaveInhandRotateGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_grasp_env_cfg:SharpaWaveEnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0",
    entry_point=f"rl_isaaclab.tasks.inhand_rotate.sharpa_wave_deploy_env:SharpaWaveInhandRotateDeployEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sharpa_wave_deploy_env_cfg:SharpaWaveEnvCfg",
        "agent_cfg_entry_point": f"{agents.__name__}:ppo_cfg.yaml",
    },
)
