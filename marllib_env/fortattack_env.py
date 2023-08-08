# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
An example of integrating new tasks into MARLLib
About ma-gym: https://github.com/koulanurag/ma-gym
doc: https://github.com/koulanurag/ma-gym/wiki

Learn how to transform the environment to be compatible with MARLlib:
please refer to the paper: https://arxiv.org/abs/2210.13708

Install ma-gym before use
"""

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box
# from ma_gym.envs.checkers import Checkers
# from ma_gym.envs.switch import Switch
from gym_fortattack.fortattack_env_wrapper import FortAttackEnvWrapper
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
import time, os

# register all scenario with env class
REGISTRY = {}
# REGISTRY["Checkers"] = Checkers
# REGISTRY["Switch2"] = Switch
REGISTRY["FortAttack"] = FortAttackEnvWrapper
REGISTRY["FortAttackOffense"] = FortAttackEnvWrapper
REGISTRY["FortAttackDefense"] = FortAttackEnvWrapper

# provide detailed information of each scenario
# mostly for policy sharing
policy_mapping_dict = {
    # "Checkers": {
    #     "description": "two team cooperate",
    #     "team_prefix": ("red_", "blue_"),
    #     "all_agents_one_policy": True,
    #     "one_agent_one_policy": True,
    # },
    # "Switch2": {
    #     "description": "two team cooperate",
    #     "team_prefix": ("red_", "blue_"),
    #     "all_agents_one_policy": True,
    #     "one_agent_one_policy": True,
    # },
    "FortAttack": {
        "description": "FortAttack",
        "team_prefix": ("blue_", "red_", ),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "FortAttackOffense": {
        "description": "FortAttackOffense",
        "team_prefix": ("red_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
    "FortAttackDefense": {
        "description": "FortAttackDefense",
        "team_prefix": ("blue_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}

# must inherited from MultiAgentEnv class
class RLlibMAGym(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        # env_config.pop("map_name", None)

        self.env = REGISTRY[map](**env_config)
        # assume all agent same action/obs space
        self.action_space = self.env.action_space[0]
        # print('action_space', self.action_space)
        self.observation_space = GymDict({"obs": Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env.observation_space[0].shape[0],),
            dtype=np.dtype("float64"))})
        if map == "FortAttack":
            self.agents = ['blue_%d' % id for id in range(env_config["numGuards"])] + ['red_%d' % id for id in range(env_config["numAttackers"])]
        elif map == "FortAttackOffense":
            self.agents = ['red_%d' % id for id in range(env_config["numAttackers"])]
        elif map == "FortAttackDefense":
            self.agents = ['blue_%d' % id for id in range(env_config["numGuards"])]

        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i, name in enumerate(self.agents):
            obs[name] = {"obs": np.array(original_obs[i])}
        return obs

    def step(self, action_dict):
        action_ls = [action_dict[key] for key in action_dict.keys()]
        # print('----------------', action_ls, action_dict)
        o, r, d, info = self.env.step(action_ls)
        rewards = {}
        obs = {}
        for i, key in enumerate(action_dict.keys()):
            rewards[key] = r[i]
            obs[key] = {
                "obs": np.array(o[i])
            }
        dones = {"__all__": True if sum(d) == self.num_agents else False}
        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env.max_steps,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
