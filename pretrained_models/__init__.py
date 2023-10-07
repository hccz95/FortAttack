# %%

from ray.rllib.agents.trainer import COMMON_CONFIG
import pickle, collections, torch
import json

# 递归合并两个字典，如果有冲突以master为主
def deep_merge_dicts(master, slave):
    for key in slave:
        if not (key in master):
            master[key] = slave[key]
        elif isinstance(master[key], dict) and isinstance(slave[key], dict):
                deep_merge_dicts(master[key], slave[key])
    return master

# 从marllib产生的checkpoint文件中读取模型参数
def load_checkpoint(checkpoint_path, policy_name='default_policy'):
    checkpoint = pickle.load(open(checkpoint_path, 'rb'))
    worker = pickle.loads(checkpoint['worker'])

    state_policy = worker['state'][policy_name]
    weights = state_policy['weights']

    state_dict = collections.OrderedDict()
    for name, val in weights.items():
        state_dict[name] = torch.as_tensor(val, dtype=torch.float32)
    return state_policy, state_dict

def load_json(json_path):
    with open(json_path, 'r') as JSON:
        obj = json.load(JSON)
    return obj


# 包装策略，包装输入obs和输出act都是np.array
class PretrainedPolicy:
    def __init__(self,
                 policy_cls,
                 model_cls,
                 obs_space,
                 act_space,
                 params,
                 state_policy=None,
                 ):

        ModelCatalog.register_custom_model(params["model"]["custom_model"], model_cls)   # 注册神经网络模型类

        if params["model"]["custom_model_config"]["algorithm"] == "mappo":
            from marllib.marl.algos.core.CC.mappo import MAPPOTorchPolicy
            policy_cls = MAPPOTorchPolicy
        else:
            raise NotImplementedError

        merged_dict = deep_merge_dicts(params, COMMON_CONFIG)
        self.policy = policy_cls(obs_space, act_space, merged_dict)
        if state_policy:
            self.policy.set_state(state_policy)

    # obs为单个agent的观测向量shape(obs_len)，obs_n为所有agent的观测向量shape(n_agent, obs_len)
    def get_action(self, obs, obs_n=None):
        input_dict = {'obs': np.array([obs])}
        act = self.policy.compute_actions_from_input_dict(input_dict)[0][0]
        return act

    def get_actions(self, obs_n):
        if len(obs_n) == 0:
            return []
        input_dict = {'obs': np.array(obs_n)}
        act_n = self.policy.compute_actions_from_input_dict(input_dict)[0]
        return act_n


# %%

from marllib import marl
from marllib.marl.algos.core.CC.mappo import MAPPOTorchPolicy
from ray.rllib.models import ModelCatalog
import numpy as np
import marllib_env  # 必须import，才能注册自定义的环境

# yaml_path必须是绝对路径，如果是官方环境，则设置为""即可
def generate_model_from_checkpoint(params_path, model_path, yaml_path, hyperparam_source="test"):
    params = load_json(params_path)

    alg_name = params["model"]["custom_model_config"]["algorithm"]
    env_name = params["model"]["custom_model_config"]["env"]
    map_name = params["model"]["custom_model_config"]["env_args"]["map_name"]
    env = marl.make_env(environment_name=env_name, map_name=map_name, abs_path=yaml_path)

    if alg_name == "mappo":
        alg = marl.algos.mappo(hyperparam_source=hyperparam_source)
        policy_cls = MAPPOTorchPolicy
    else:
        raise NotImplementedError

    # customize model
    model = marl.build_model(env, alg, params["model"]["custom_model_config"]["model_arch_args"])
    model_cls = model[0]

    obs_space = env[0].observation_space['obs'] # 注意，严格来说这里的obs_space应该要归一化到[-1,+1]，不过因为不训练，所以没有影响？
    obs_space.original_space = env[0].observation_space
    act_space = env[0].action_space

    params["model"]["custom_model_config"]["space_obs"] = env[0].observation_space # json文件中的dict保存为字符串无法读取，所以这里恢复一下
    params["model"]["custom_model_config"]["space_act"] = env[0].action_space # json文件中的dict保存为字符串无法读取，所以这里恢复一下

    policy_name = list(eval(params["multiagent"]["policies"]))[0]
    state_policy, _ = load_checkpoint(model_path, policy_name)

    return PretrainedPolicy(policy_cls, model_cls, obs_space, act_space, params, state_policy)
