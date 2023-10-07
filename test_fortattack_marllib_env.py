from marllib import marl
import marllib_env     # 必须import，才能注册marllib_env目录下的自定义环境
import os, sys


def train_mappo(debug=False):
    env = marl.make_env(environment_name="magym", map_name="FortAttackOffense", abs_path=yaml_path)
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
    env = marl.make_env(environment_name="magym", map_name="FortAttackOffense", abs_path=yaml_path)
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
    # initialize env
    yaml_path = os.path.abspath('./marllib_env/fortattack.yaml')     # 默认2v2

    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    else:
        seed = 111

    result = train_mappo(debug=False)

    params_path = result.trials[0].logdir + "/params.json"
    model_path =  result._checkpoints[0]['checkpoint_manager']._best_checkpoints[0].value.value # best checkpoint
    # model_path =  result._checkpoints[0]['checkpoint_manager'].newest_checkpoint.value          # newest checkpoint

    evaluate_mappo(params_path, model_path)
