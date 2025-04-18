import jax.numpy as jnp
from ajax.types import BraxEnv, EnvType, GymnaxEnv


def check_if_environment_has_continuous_actions(env):
    env = get_raw_env(env)
    if check_env_is_brax(env):
        return True
    return env.action_space().dtype == jnp.float32


def get_raw_env(env: EnvType) -> EnvType:
    """
    Get the raw environment from the given environment.
    """
    if hasattr(env, "_env"):
        return env._env
    return env


def check_env_is_brax(env) -> bool:
    return isinstance(env, BraxEnv) or "brax" in str(type(env)).lower()


def check_env_is_gymnax(env) -> bool:
    return isinstance(env, GymnaxEnv) or "gymnax" in str(type(env)).lower()


def get_env_type(env: EnvType) -> str:
    """
    Get the type of the environment.
    """
    env = get_raw_env(env)
    if check_env_is_brax(env):
        return "brax"
    elif check_env_is_gymnax(env):
        return "gymnax"
    else:
        raise ValueError(f"Unsupported env type: {type(env)}")
