from gym_fortattack.fortattack import make_fortattack_env
import gym
# from pygame import mixer  # Load the popular external library
import numpy as np
import action_policy


class FortAttackEnvWrapper(gym.Env):
    metadata = {'render.modes': ['human']}
    # TODO: 后续要加一个env_arg，用来设置每个agent的action_selector，yaml文件里支持list和dict
    def __init__(self, max_steps=100, numGuards=2, numAttackers=2, map_name="FortAttackOffense", **kwargs):
        self.env = make_fortattack_env(num_steps=max_steps, numGuards=numGuards, numAttackers=numAttackers)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_steps = max_steps
        self.numGuards = numGuards
        self.numAttackers = numAttackers
        self.map_name = map_name

        if map_name == "FortAttackOffense":
            self.mask = [0] * numGuards + [1] * numAttackers
        elif map_name == "FortAttackDefense":
            self.mask = [1] * numGuards + [0] * numAttackers
        elif map_name == "FortAttack":
            self.mask = [1] * numGuards + [1] * numAttackers
        elif map_name == "FortAttackAdhoc":
            self.mask = [0] * numGuards + [1] + [0] * (numAttackers - 1)
        elif map_name == "FortAttackAuto":
            self.mask = [0] * numGuards + [0] * numAttackers
        else:
            raise NotImplementedError

        self.trainable_ids = [i for i, v in enumerate(self.mask) if v == 1]
        self.action_selectors = [action_policy.RulePolicy(numGuards, numAttackers, agent_id) # 默认使用1号策略, policy_id=1
                                    for agent_id in range(numGuards+numAttackers)]

    def reset(self):
        self._o = self.env.reset()
        return self._o[self.trainable_ids]

    def step(self, action_n):
        action_all = np.zeros((len(self.mask)), dtype=int)
        for i, act in enumerate(self.action_selectors):
            if self.mask[i] == 0:
                action_all[i] = act.get_action(self._o[i], self._o)
        action_all[self.trainable_ids] = action_n

        o, r, d, alive_g, info = self.env.step(action_all)
        self._o = o

        o_n = o[self.trainable_ids]
        r_n = np.array(r)[self.trainable_ids]
        d_n = [d] * max(1, len(action_n))
        # info["alive_g"] = alive_g  # TODO: 这里返回的info必须是{}，否则报错

        return o_n, r_n, d_n, {}

    def render(self, mode='human'):
        return self.env.render(mode)

    def seed(self, seed):
        self.seed = seed

if __name__ == '__main__':
    env = FortAttackEnvWrapper(numGuards=3, numAttackers=3, map_name="FortAttack")
    while True:
        s = env.reset()
        r_red = 0
        r_blue = 0
        while True:
            a_n = np.zeros(6)
            s_, r_n, d, info = env.step(a_n)
            # env.render()
            # print(s_, r_n, d, )
            r_blue += sum(r_n[:3])  # guard reward
            r_red += sum(r_n[3:])   # attacker reward
            if all(d):
                print(r_red, r_blue)
                break
            s = s_