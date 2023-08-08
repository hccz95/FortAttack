from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib_env.fortattack_env import RLlibMAGym
import os


def mappo(debug=False):
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="test")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128-128"})

    if debug:
        # start learning
        mappo.fit(env, model, stop={'episode_reward_mean': 20, 'timesteps_total': 500000000}, local_mode=True, num_gpus=0,
                num_workers=0, share_policy='all', checkpoint_freq=1000)
    else:
        # start learning
        mappo.fit(env, model, stop={'episode_reward_mean': 20, 'timesteps_total': 500000000}, local_mode=False, num_gpus=1,
                num_workers=32, share_policy='all', checkpoint_freq=1000)


if __name__ == '__main__':
    # register new env
    ENV_REGISTRY["magym"] = RLlibMAGym
    # initialize env
    path = os.getcwd() + '/' + './marllib_env/fortattack.yaml'
    print(path)
    env = marl.make_env(environment_name="magym", map_name="FortAttackOffense", abs_path=path)

    mappo(debug=False)   # debug mode uses a single thread, easy to debug
