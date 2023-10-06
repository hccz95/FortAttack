# copied from https://github.com/hccz95/marllib_tutorial/blob/master/run_mappo.py

# %%
from marllib import marl

# prepare env
env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source='mpe')

# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# %%
# start training
results = mappo.fit(
    env, model,
    stop={'timesteps_total': 10000},
    share_policy='group',
    local_mode=False,
    num_gpus=1,
    num_workers=32,
    seed=1,
)


# %%

# params = results._checkpoints[0]['config']
params_path = results.trials[0].logdir + '/params.json'
model_path =  results._checkpoints[0]['checkpoint_manager'].newest_checkpoint.value

env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)
env[0].metadata['render.modes'] = ['rgb_array']
mappo = marl.algos.mappo(hyperparam_source='mpe')
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# rendering
results = mappo.render(
    env, model,
    stop={'timesteps_total': 100},
    share_policy='group',
    local_mode=True,
    restore_path={
        'params_path': params_path,
        'model_path': model_path,
        # 'render': True,         # tell MARLlib to render and recording
    },
    num_gpus=0,
    num_workers=0,
    seed=1,
)
