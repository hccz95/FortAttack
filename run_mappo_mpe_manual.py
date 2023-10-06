# %%

# 递归合并两个字典，如果有冲突以master为主
def deep_merge_dicts(master, slave):
    for key in slave:
        if not (key in master):
            master[key] = slave[key]
        elif isinstance(master[key], dict) and isinstance(slave[key], dict):
                deep_merge_dicts(master[key], slave[key])
    return master

# 从marllib产生的checkpoint文件中读取模型参数
def load_checkpoint(checkpoint_path):
    import pickle, collections, torch
    checkpoint = pickle.load(open(checkpoint_path, 'rb'))
    worker = pickle.loads(checkpoint['worker'])
    state = worker['state']
    weights = state['shared_policy']['weights']

    state_dict = collections.OrderedDict()
    for name, val in weights.items():
        state_dict[name] = torch.as_tensor(val, dtype=torch.float32)
    return state, state_dict

def load_json(json_path):
    import json
    with open(json_path, 'r') as JSON:
        obj = json.load(JSON)
    return obj

# 包装环境，主要是为了obs和act的格式转换；
# MyEnv本身使用的obs_n和act_n都是简单的np.array，但是marllib里的环境使用的一般是dict
class MyEnv:

    def __init__(self, env):
        self.env = env
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

# 包装策略，包装输入obs和输出act都是np.array
class CustomPolicy:
    def __init__(self,
                 policy_cls,
                 model_cls,
                 obs_space,
                 act_space,
                 params,
                 state,
                 ):

        from ray.rllib.agents.trainer import COMMON_CONFIG
        merged_dict = deep_merge_dicts(params, COMMON_CONFIG)
        self.policy = policy_cls(obs_space, act_space, merged_dict)
        self.policy.set_state(state["shared_policy"])

    def get_actions(self, obs_n):
        input_dict = {'obs': np.array(obs_n)}
        act_n = self.policy.compute_actions_from_input_dict(input_dict)[0]
        return act_n


# %%

from marllib import marl
from marllib.marl.algos.core.CC.mappo import MAPPOTorchPolicy
from ray.rllib.models import ModelCatalog
import numpy as np

env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
mappo = marl.algos.mappo(hyperparam_source='mpe')
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

model_cls = model[0]
ModelCatalog.register_custom_model("Centralized_Critic_Model", model_cls)   # 注册神经网络模型类

# %%
# set the result path
params_path = 'exp_results/mappo_mlp_simple_spread/MAPPOTrainer_mpe_simple_spread_ff665_00000_0_2023-10-05_12-45-24/params.json'
model_path = 'exp_results/mappo_mlp_simple_spread/MAPPOTrainer_mpe_simple_spread_ff665_00000_0_2023-10-05_12-45-24/checkpoint_003000/checkpoint-3000'

params = load_json(params_path)
params["model"]["custom_model_config"]["space_obs"] = env[0].observation_space # json文件中的dict保存为字符串无法读取，所以这里恢复一下
params["model"]["custom_model_config"]["space_act"] = env[0].action_space # json文件中的dict保存为字符串无法读取，所以这里恢复一下
# params["model"]["max_seq_len"] = 20   # MARLlib/marllib/patch/rllib/policy/torch_policy.py的94行设置了默认值20, 或者直接使用COMMON_CONFIG就行了

state, state_dict = load_checkpoint(model_path)

# %%

obs_space = env[0].observation_space['obs'] # 注意，严格来说这里的obs_space应该要归一化到[-1,+1]，不过因为不训练，所以没有影响？
obs_space.original_space = env[0].observation_space
act_space = env[0].action_space

# !!! 核心就是这句话，为了加载marllib保存的模型，不得不大费周章，遵循其内部逻辑来定义Policy
# 根据所使用的marllib不同算法以及环境，来确定以下几个参数
pol = CustomPolicy(MAPPOTorchPolicy, model_cls, obs_space, act_space, params, state)

# %%

myenv = MyEnv(env[0])
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
