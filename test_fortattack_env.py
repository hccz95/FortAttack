import numpy as np
from gym_fortattack.fortattack import make_fortattack_env
import random
import time
import action_policy

if __name__ == '__main__':
    num_env_steps = 200
    
    env = make_fortattack_env(num_env_steps)
    obs = env.reset()
    act = action_policy.Policy()

    alive_ag = []
    start = time.time()
    for j in range(10):
        done = False
        step = 0
        previous_action = np.zeros((6,2))
        agent_actions = np.zeros(6) # set to zeros for previous action coutn

        while not done:
            # actions_list = np.zeros((6,2))
            actions_list = act.get_actions(obs)
            agent_actions = np.array(actions_list).reshape(-1)
            # first agent = ad hoc agent
            
            obs, reward, done, alive_ag, info = env.step(agent_actions)
            env.render()
            
            for agent in range(6):
                previous_action[agent][0] = previous_action[agent][1]
                previous_action[agent][1] = agent_actions[agent]

            step += 1 
            if done:
                obs = env.reset()
                # masks = torch.FloatTensor(obs[:,0]) #check agents alive or dead
                time.sleep(1)
    end = time.time()
