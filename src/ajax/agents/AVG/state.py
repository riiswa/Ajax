import jax.numpy as jnp
from flax import struct

from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState


@struct.dataclass
class NormalizationInfo:
    value: jnp.array
    count: jnp.array
    mean: jnp.array
    mean_2: jnp.array


one = jnp.ones(1)


@struct.dataclass
class AVGState(BaseAgentState):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    alpha: LoadedTrainState  # Temperature parameter
    reward: NormalizationInfo
    gamma: NormalizationInfo
    G_return: NormalizationInfo
    scaling_coef: jnp.ndarray


@struct.dataclass
class AVGConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    target_entropy: float
    learning_starts: int = 100
    reward_scale: float = 5.0
