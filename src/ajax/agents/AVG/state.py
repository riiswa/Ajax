from flax import struct

from ajax.state import BaseAgentConfig, BaseAgentState, LoadedTrainState


@struct.dataclass
class AVGState(BaseAgentState):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    alpha: LoadedTrainState  # Temperature parameter


@struct.dataclass
class AVGConfig(BaseAgentConfig):
    """The agent properties to be carried over iterations of environment interaction and updates"""

    target_entropy: float
    tau: float = 0.005
    learning_starts: int = 100
    reward_scale: float = 5.0
