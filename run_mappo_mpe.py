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
mappo.fit(
    env, model,
    stop={'timesteps_total': 10000},
    share_policy='group',
    local_mode=False,
    num_gpus=0,
    num_workers=1,
)

# %%
# # set the result path
# params_path = '.../params.json'
# model_path = '.../checkpoint'
#
# # rendering
# mappo.render(
#     env, model, 
#     local_mode=True, 
#     restore_path={
#         'params_path': params_path,
#         'model_path': model_path,
#     }
# )