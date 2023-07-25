from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib_env.fortattack_env import RLlibMAGym
import os


if __name__ == '__main__':
    # register new env
    ENV_REGISTRY["magym"] = RLlibMAGym
    # initialize env
    path = os.getcwd() + '/' + './marllib_env/fortattack.yaml'
    print(path)
    env = marl.make_env(environment_name="magym", map_name="FortAttack", abs_path=path)
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="test")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})
    
    # start learning
    mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000}, local_mode=True, num_gpus=1,
              num_workers=4, share_policy='all', checkpoint_freq=50)
