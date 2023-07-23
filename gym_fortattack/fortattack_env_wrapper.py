from gym_fortattack.envs.fortattack_env_v1 import FortAttackEnvV1
import gym


class FortAttackEnvWrapper(FortAttackEnvV1):  
    metadata = {'render.modes': ['human']}   
    def __init__(self):
        super(FortAttackEnvWrapper, self).__init__()
        
    def reset_world(self):
        super(FortAttackEnvWrapper, self).reset_world()
        
    def reset_world(self):
        super(FortAttackEnvWrapper, self).reset_world()
        
    def reset_world(self):
        super(FortAttackEnvWrapper, self).reset_world()
        
    def reset_world(self):
        super(FortAttackEnvWrapper, self).reset_world()
