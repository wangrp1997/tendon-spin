# Vendored reference: git@github.com:wangrp1997/sharpa-rl-lab.git (commit 5accf024d376685eaa17da7aa4614498217eab4d).
# Original relative file: rl_isaaclab/wrapper/config_wrapper.py.
# Added provenance header only; original copyright/license follows unchanged.
class ConfigWrapper:
    def __init__(self, agent_cfg, env_cfg, test=False):
        self.train = agent_cfg
        self.task = env_cfg
        self.test = test
