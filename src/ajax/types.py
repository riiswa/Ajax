from typing import TypeAlias, Union

import jax
import jax.numpy as jnp
import jaxlib
from brax.envs import Env as BraxEnv
from brax.envs.base import State as BraxEnvState
from flax.core.frozen_dict import FrozenDict
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.environment import EnvState as GymnaxEnvState

HiddenState = Union[jnp.ndarray, FrozenDict]
EnvType = Union[GymnaxEnv, BraxEnv]
EnvStateType = Union[BraxEnvState, GymnaxEnvState]
ActivationFunction: TypeAlias = Union[
    jax._src.custom_derivatives.custom_jvp, jaxlib.xla_extension.PjitFunction
]
