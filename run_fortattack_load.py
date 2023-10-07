# %%
import os
from pretrained_models import generate_model_from_checkpoint

params_path = './pretrained_models/fortattack/FortAttackOffense_fbd47/params.json'
model_path =  './pretrained_models/fortattack/FortAttackOffense_fbd47/checkpoint'
yaml_path = os.path.abspath('./marllib_env/fortattack.yaml')

pol = generate_model_from_checkpoint(params_path, model_path, yaml_path)

# %%
from gym_fortattack.fortattack_env_wrapper import FortAttackEnvWrapper
myenv = FortAttackEnvWrapper(numGuards=2, numAttackers=2, map_name="FortAttackAuto")    # auto模式自动运行，不需要输入动作，所有agent默认策略为policy_1

# # 将2个attacker的策略设为随机策略
# myenv.action_selectors[2].set_policy_id('random')
# myenv.action_selectors[3].set_policy_id('random')

# 将2个attacker的策略设为预训练策略
myenv.action_selectors[2] = pol
myenv.action_selectors[3] = pol

for i in range(20):
    obs_n = myenv.reset()
    r_ep = 0.
    while True:
        act_n = []
        obs_n, r_n, done, info = myenv.step(act_n)
        r_ep += sum(r_n)
        # myenv.render()
        if all(done):
            print('episode', i, 'reward:', r_ep)
            break
