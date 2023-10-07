import numpy as np
import policies

ACTION_DIM = 8

class RulePolicy:
    def __init__(self, numGuards=2, numAttackers=2, agent_id=0, policy_id=1):
        self.wall_pos = [-1,1,-0.8,0.8]
        self.fortDim = 0.15   # radius
        self.doorLoc = np.array([0,0.8])
        # simulation timestep
        self.dt = 0.1
        self.max_speed = 3
        self.max_rot = 0.17
        self.size = 0.05
        self.shootRad = 0.8
        self.shootWin = np.pi/4
        self.numAgents = numGuards + numAttackers
        self.numGuards = numGuards
        self.numAttackers = numAttackers
        self.agent_id = agent_id
        self.policy_id = policy_id

    def get_action(self, obs, obs_n):
        if self.policy_id == 'random':
            return np.random.randint(0, ACTION_DIM)
        else:
            return self.get_single_action(obs, obs_n)

    def get_tri_pts_arr(self, x_pos, y_pos, ori):
        ang = ori
        pt1 = [x_pos, y_pos]+self.size*np.array([np.cos(ang), np.sin(ang)])
        pt2 = pt1 + self.shootRad*np.array([np.cos(ang+self.shootWin/2), np.sin(ang+self.shootWin/2)])
        pt3 = pt1 + self.shootRad*np.array([np.cos(ang-self.shootWin/2), np.sin(ang-self.shootWin/2)])

        A = np.array([[pt1[0], pt2[0], pt3[0]],
                      [pt1[1], pt2[1], pt3[1]],
                      [     1,      1,      1]])
        return(A)

    def get_single_action(self, single_obs, current_obs):
        if single_obs[0] != 1.0:
            action = 0 # agent dead - do nothing
        else:
            x_pos, y_pos, ori = policies.get_xya_from_obs(single_obs)
            A = self.get_tri_pts_arr(x_pos, y_pos, ori) # shoot cone
            if self.agent_id < self.numGuards:
                action = policies.guard_policy(policy_id=self.policy_id, current_obs=current_obs, agent_name=self.agent_id, A=A, num_guards=self.numGuards)
            else: # attacker
                action = policies.attacker_policy(policy_id=self.policy_id, current_obs=current_obs, agent_name=self.agent_id, A=A, num_guards=self.numGuards)
        return action

    def set_policy_id(self, policy_id):
        self.policy_id = policy_id


def generate_policy(policy_type='rule', **kwargs):
    if policy_type == 'rule':
        return RulePolicy(**kwargs)     # numGuards=2, numAttackers=2, agent_id=0, policy_id=1
    else:
        raise NotImplementedError
