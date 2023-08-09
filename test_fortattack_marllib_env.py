from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib_env.fortattack_env import RLlibMAGym
import os


def train_mappo(debug=False):
    env = marl.make_env(environment_name="magym", map_name="FortAttackOffense", abs_path=path)
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="test")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128-128"})

    if debug:
        # start learning
        result = mappo.fit(
            env, model, stop={'episode_reward_mean': 25, 'timesteps_total': 50000},
            local_mode=True,
            num_gpus=0,
            num_workers=0,
            share_policy='all',
            checkpoint_freq=1000
        )
    else:
        # start learning
        result = mappo.fit(
            env, model, stop={'episode_reward_mean': 25, 'timesteps_total': 500000000},
            local_mode=False,
            num_gpus=1,
            num_workers=32,
            share_policy='all',
            checkpoint_freq=1000
        )

    return result

# rendering: change the path of params and model in restore_path, and the videos will be saved in `checkpoint_videos`
def evaluate_mappo(params_path, model_path):
    env = marl.make_env(environment_name="magym", map_name="FortAttackOffense", abs_path=path)
    env[0].metadata['render.modes'] = ['rgb_array']
    # pick mappo algorithms
    mappo = marl.algos.mappo(hyperparam_source="test")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128-128"})

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
    path = os.getcwd() + '/' + './marllib_env/fortattack.yaml'
    print(path)

    result = train_mappo(debug=False)

    params_path = result.trials[0].logdir + "/params.json"
    model_path =  result._checkpoints[0]['checkpoint_manager']._best_checkpoints[0].value.value # best checkpoint
    # model_path =  result._checkpoints[0]['checkpoint_manager'].newest_checkpoint.value          # newest checkpoint

    evaluate_mappo(params_path, model_path)
