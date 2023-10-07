from marllib.envs.base_env import ENV_REGISTRY
from marllib_env.fortattack_env import RLlibMAGym

# register new env
ENV_REGISTRY["magym"] = RLlibMAGym
