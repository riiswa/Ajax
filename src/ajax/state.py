from typing import Optional

import flashbax as fbx
import jax
import jax.numpy as jnp
from ajax.types import EnvStateType, EnvType, HiddenState
from flax import struct
from flax.training.train_state import TrainState
from gymnax import EnvParams


@struct.dataclass
class EnvironmentConfig:
    env: EnvType
    env_params: EnvParams
    num_envs: int
    continuous: bool


@struct.dataclass
class CollectorState:
    """The variables necessary to interact with the environment and collect the transitions"""

    rng: jax.Array
    env_state: EnvStateType
    last_obs: jnp.ndarray
    num_update: int = 0
    timestep: int = 0
    average_reward: float = 0.0
    last_done: Optional[jnp.ndarray] = None
    buffer_state: Optional[fbx.flat_buffer.TrajectoryBufferState] = None


@struct.dataclass
class MaybeRecurrentTrainState(TrainState):
    hidden_state: Optional[HiddenState] = None

    @classmethod
    def create(cls, *, hidden_state=None, **kwargs):
        # Forward all other parameters to super().create
        instance = super().create(**kwargs)
        # Store the hidden_state in the instance
        return instance.replace(hidden_state=hidden_state)
