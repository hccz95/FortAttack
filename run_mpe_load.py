# %%

# 包装环境，主要是为了obs和act的格式转换；
# MyEnv本身使用的obs_n和act_n都是简单的np.array，但是marllib里的环境使用的一般是dict
class MyEnv:

    def __init__(self, ):
        self.env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)[0]
        self.env.env.seed(1)   # 设置随机数种子，固定环境，方便调试

    def reset(self, ):
        dic = self.env.reset()
        obs = [agent['obs'] for key, agent in dic.items()]
        return obs

    def step(self, act_n):
        act_dict = {}
        for id, act in enumerate(act_n):
            act_dict['agent_'+str(id)] = act

        obs, rewards, dones, infos = self.env.step(act_dict)

        return [val['obs'] for val in obs.values()], \
               [val for val in rewards.values()], \
               [val for val in dones.values()], \
               [val for val in infos.values()]

    def render(self, ):
        self.env.render()

# %%

from marllib import marl

params_path = './pretrained_models/mpe/mpe_simple_spread_ff665/params.json'
model_path = './pretrained_models/mpe/mpe_simple_spread_ff665/checkpoint'

from pretrained_models import generate_model_from_checkpoint
pol = generate_model_from_checkpoint(params_path, model_path, "", "mpe")

# %%

myenv = MyEnv()
for it in range(20):
    obs_n = myenv.reset()
    r_ep = 0.
    while True:
        act_n = pol.get_actions(obs_n)
        obs_n, rewards, dones, info = myenv.step(act_n)
        r_ep += sum(rewards)
        if all(dones):
            print("episode", it, "ended!", "reward:", r_ep)
            break
        # myenv.render()
