from typing import Any, Callable, Optional

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
    hidden_state: Optional[Any] = (
        None  # HiddenState can be replaced with Any for flexibility
    )
    recurrent: bool = False

    @classmethod
    def create(cls, *, hidden_state=None, apply_fn: Callable = None, **kwargs):
        # Ensure apply_fn is passed to the parent TrainState
        instance = super().create(apply_fn=apply_fn, **kwargs)
        # Determine if the state is recurrent
        recurrent = hidden_state is not None
        # Return a new instance with hidden_state and recurrent attributes
        return instance.replace(hidden_state=hidden_state, recurrent=recurrent)

    def apply(self, params, *args, **kwargs):
        """Call the apply_fn with the given parameters and arguments."""
        return self.apply_fn(params, *args, **kwargs)


@struct.dataclass
class BaseAgentState:
    rng: jax.Array
    collector: CollectorState
    actor_state: MaybeRecurrentTrainState
    critic_state: MaybeRecurrentTrainState
    collector_state: CollectorState


@struct.dataclass
class AgentConfig:
    seed: int
    gamma: float
