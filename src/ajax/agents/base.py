from typing import Optional

import jax
import jax.numpy as jnp
from ajax.state import AgentConfig, BaseAgentState, CollectorState, EnvironmentConfig


class BaseAgent:
    """
    Base class for all agents. This class provides the basic structure and
    functionality for agents in the AJAX framework.
    """

    def __init__(
        self, seed: int, env_config: EnvironmentConfig, agent_config: AgentConfig
    ):
        pass

    def train(self, seed: int, num_timesteps: int):
        """
        Train the agent using the provided state.
        """
        raise NotImplementedError("train method must be implemented by subclasses")
