from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib_env.fortattack_env import RLlibMAGym
import os


def train_mappo(debug=False):
    env = marl.make_env(environment_name="magym", map_name="FortAttackOffense", abs_path=path)
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="test")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})

    if debug:
        # start learning
        result = mappo.fit(
            env, model, stop={'episode_reward_mean': 25, 'timesteps_total': 5000},
            local_mode=True,
            num_gpus=0,
            num_workers=0,
            share_policy='all',
            checkpoint_freq=1000,
            seed=seed,
        )
    else:
        # start learning
        result = mappo.fit(
            env, model, stop={'episode_reward_mean': 30, 'timesteps_total': 200000000},
            local_mode=False,
            num_gpus=1,
            num_workers=32,
            share_policy='all',
            checkpoint_freq=1000,
            seed=seed,
        )

    # mark the best checkpoint
    best_model_dir = os.path.dirname(result._checkpoints[0]['checkpoint_manager']._best_checkpoints[0].value.value)
    os.system("cp -r " + best_model_dir + " " + os.path.dirname(best_model_dir) + "/_checkpoint_best")

    return result

# rendering: change the path of params and model in restore_path, and the videos will be saved in `checkpoint_videos`
def evaluate_mappo(params_path, model_path):
    env = marl.make_env(environment_name="magym", map_name="FortAttackOffense", abs_path=path)
    env[0].metadata['render.modes'] = ['rgb_array']
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="test")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})

    mappo.render(
        env, model,
        local_mode=True,
        restore_path={
            'params_path': params_path,
            'model_path': model_path,
            'render': True,         # tell MARLlib to render and recording
        },
        num_gpus=0,
        num_workers=0,
        share_policy='all',
    )


if __name__ == '__main__':
    # register new env
    ENV_REGISTRY["magym"] = RLlibMAGym
    # initialize env
    path = os.getcwd() + '/' + './marllib_env/fortattack_3v3.yaml'
    print(path)

    import sys
    seed = int(sys.argv[1])

    result = train_mappo(debug=False)

    params_path = result.trials[0].logdir + "/params.json"
    model_path =  result._checkpoints[0]['checkpoint_manager']._best_checkpoints[0].value.value # best checkpoint
    # model_path =  result._checkpoints[0]['checkpoint_manager'].newest_checkpoint.value          # newest checkpoint

    # params_path = "exp_results_2023-08-10-15-23-56_128-128mlp_low-reward_high-success/mappo_mlp_FortAttackOffense/MAPPOTrainer_magym_FortAttackOffense_ddaf2_00000_0_2023-08-10_15-23-56/params.json"
    # model_path = "exp_results_2023-08-10-15-23-56_128-128mlp_low-reward_high-success/mappo_mlp_FortAttackOffense/MAPPOTrainer_magym_FortAttackOffense_ddaf2_00000_0_2023-08-10_15-23-56/checkpoint_065105/checkpoint-65105"

    evaluate_mappo(params_path, model_path)

    # from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
    # COOP_ENV_REGISTRY["magym"] = RLlibMAGym
    # alg = marl.algos.iql(hyperparam_source="test")
    # model = marl.build_model(env, alg, {"core_arch": "mlp", "encode_layer": "128-128"})
    # alg.fit(env, model, stop={'episode_reward_mean': 25, 'timesteps_total': 100000000}, local_mode=False, num_gpus=1,
    #           num_workers=4, share_policy='all', checkpoint_freq=200)

    # # RLLIB的MADDPG只支持连续动作，怎么办？
    # maddpg = marl.algos.maddpg(hyperparam_source="test")
     # maddpg.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000}, local_mode=False, num_gpus=1,
    #           num_workers=4, share_policy='all', checkpoint_freq=100)


    # from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
    # COOP_ENV_REGISTRY["magym"] = RLlibMAGym

    # qmix = marl.algos.qmix(hyperparam_source="test")
    # model = marl.build_model(env, qmix, {"core_arch": "mlp", "encode_layer": "128-128"})
    # qmix.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000}, local_mode=False, num_gpus=1,
    #           num_workers=4, share_policy='all', checkpoint_freq=1000)