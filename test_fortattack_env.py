import numpy as np
from gym_fortattack.fortattack import make_fortattack_env
import random
import time
import action_policy

if __name__ == '__main__':
    num_steps = 200
    num_guards = 3
    num_attackers = 3
    num_agents = num_guards + num_attackers

    env = make_fortattack_env(num_steps=num_steps, numGuards=num_guards, numAttackers=num_attackers, )
    obs = env.reset()
    act = action_policy.Policy(numAttackers=num_attackers, numGuards=num_guards)

    alive_ag = []
    start = time.time()
    for j in range(10):
        done = False
        step = 0
        previous_action = np.zeros((num_agents,2))
        agent_actions = np.zeros(num_agents) # set to zeros for previous action count
        ep_r_guard, ep_r_attacker = 0., 0.

        while not done:
            # actions_list = np.zeros((num_agents,2))
            actions_list = act.get_actions(obs)      # obs和actions都是先guard再attacker
            agent_actions = np.array(actions_list).reshape(-1)
            # first agent = ad hoc agent

            obs, reward, done, alive_ag, info = env.step(agent_actions)
            env.render()

            for agent in range(num_agents):
                previous_action[agent][0] = previous_action[agent][1]
                previous_action[agent][1] = agent_actions[agent]

            ep_r_guard += sum(reward[:num_guards])
            ep_r_attacker += sum(reward[num_guards:])

            step += 1
            if done:
                print('\t r_guard: %.2f' % ep_r_guard, '\t r_attacker: %.2f' % ep_r_attacker)
                obs = env.reset()
                # masks = torch.FloatTensor(obs[:,0]) #check agents alive or dead
                time.sleep(1)
    end = time.time()
