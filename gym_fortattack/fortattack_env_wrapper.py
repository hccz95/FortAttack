from gym_fortattack.fortattack import make_fortattack_env
import gym
# from pygame import mixer  # Load the popular external library
import numpy as np
import action_policy


class FortAttackEnvWrapper(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, max_steps=100, numGuards=2, numAttackers=2, map_name="FortAttackOffense"):
        self.env = make_fortattack_env(num_steps=max_steps, numGuards=numGuards, numAttackers=numAttackers)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_steps = max_steps
        self.numGuards = numGuards
        self.numAttackers = numAttackers
        self.map_name = map_name
        self.act = action_policy.Policy(numGuards, numAttackers)

    def reset(self):
        self._o = self.env.reset()
        return self._o

    def step(self, action_n):
        action_all = self.act.get_actions(self._o)
        if self.map_name == "FortAttackOffense":
            assert len(action_n) == self.numAttackers
            action_all[self.numGuards:] = action_n
        elif self.map_name == "FortAttackDefense":
            assert len(action_n) == self.numGuards
            action_all[:self.numGuards] = action_n
        else:
            action_all = action_n
        o, r, d, alive_g, info = self.env.step(action_all)

        self._o = o
        d = [d] * len(action_n)
        # info["alive_g"] = alive_g

        # TODO: 这里返回的info必须是{}，否则报错

        if self.map_name == "FortAttackOffense":
            return o[self.numGuards:], r[self.numGuards:], d, {}
        elif self.map_name == "FortAttackDefense":
            return o[:self.numGuards], r[:self.numGuards], d, {}
        else:
            return o, r, d, {}

    def render(self, mode='human'):
        return self.env.render(mode)

    def seed(self, seed):
        self.seed = seed

if __name__ == '__main__':
    env = FortAttackEnvWrapper()
    while True:
        s = env.reset()
        r_red = 0
        r_blue = 0
        while True:
            a_n = np.zeros(6)
            s_, r_n, d, info = env.step(a_n)
            # env.render()
            # print(s_, r_n, d, )
            r_red += sum(r_n[:3])
            r_blue += sum(r_n[3:])
            if all(d):
                print(r_red, r_blue)
                break
            s = s_