from typing import Optional, Tuple, Union

import brax
import brax.envs
import gymnax
from ajax.environments.utils import (
    EnvType,
    check_if_environment_has_continuous_actions,
    get_env_type,
)
from ajax.wrappers import get_wrappers
from gymnax import EnvParams


def build_env_from_id(
    env_id: str, num_envs: int = 1, **kwargs
) -> tuple[EnvType, Optional[EnvParams]]:
    if env_id in gymnax.registered_envs:
        env, env_params = gymnax.make(env_id)
        return env, env_params
    if env_id in list(brax.envs._envs.keys()):
        return brax.envs.create(env_id, batch_size=num_envs, **kwargs), None
    else:
        raise ValueError(f"Environment {env_id} not found in gymnax or brax")


def prepare_env(
    env_id: Union[str, EnvType],
    episode_length: Optional[int] = None,
    env_params: Optional[EnvParams] = None,
    num_envs: int = 1,
    normalize_obs: bool = True,
    normalize_reward: bool = False,
    gamma: Optional[float] = None,  # Discount factor for reward normalization
) -> Tuple[EnvType, Optional[EnvParams], Union[str, EnvType], bool]:
    if isinstance(env_id, str):
        env, env_params = build_env_from_id(
            env_id,
            episode_length=episode_length or 1000,
            num_envs=num_envs,
        )
    else:
        env = env_id  # Assume prebuilt env

    continuous = check_if_environment_has_continuous_actions(env)

    mode = get_env_type(env)

    ClipAction, NormalizeVecObservation, NormalizeVecReward = get_wrappers(mode)

    # Apply wrappers based on flags
    if normalize_obs:
        env = NormalizeVecObservation(env)
    if normalize_reward:
        assert gamma is not None, "Gamma must be provided for reward normalization."
        env = NormalizeVecReward(env, gamma=gamma)

    return env, env_params, env_id, continuous
