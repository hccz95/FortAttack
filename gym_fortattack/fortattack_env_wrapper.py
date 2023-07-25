from gym_fortattack.fortattack import FortAttackGlobalEnv
import gym
# from pygame import mixer  # Load the popular external library
import numpy as np


class FortAttackEnvWrapper(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, max_steps=100):
        scenario = gym.make('fortattack-v1')
        world = scenario.world
        world.max_time_steps = max_steps
        print(world)
        self.env = FortAttackGlobalEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_steps = max_steps
        # super(FortAttackEnvWrapper, self).__init__()

    def reset(self):
        return self.env.reset()

    def step(self, action_n):
        o, r, d, alive_g, info = self.env.step(action_n)

        d = [d] * len(action_n)
        # info["alive_g"] = alive_g
        
        # TODO: 这里返回的info必须是{}，否则报错
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