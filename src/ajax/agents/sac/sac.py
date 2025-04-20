from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from ajax.agents.sac.state import SACConfig
from ajax.agents.sac.train_sac import make_train
from ajax.buffers.utils import get_buffer
from ajax.environments.create import prepare_env
from ajax.environments.utils import (
    check_if_environment_has_continuous_actions,
    get_action_dim,
)
from ajax.state import AlphaConfig, EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import EnvType
from gymnax import EnvParams


class SAC:
    """SAC Agent that allows simple training and testing"""

    def __init__(  # pylint: disable=W0102, R0913
        self,
        # total_timesteps: int,
        env_id: str | EnvType,  # TODO : see how to handle wrappers?
        num_envs: int = 1,
        learning_rate: float = 3e-4,
        actor_architecture=["256", "relu", "256", "relu"],
        critic_architecture=["256", "relu", "256", "relu"],
        gamma: float = 0.99,
        env_params: Optional[EnvParams] = None,
        max_grad_norm: float = 0.5,
        # save: bool = False,
        # save_folder: str = "./models",
        # log_video: bool = False,
        # log_video_frequency: Optional[int] = None,
        # save_frequency: Optional[int] = None,
        lstm_hidden_size: Optional[int] = None,
        # average_reward: bool = False,
        # window_size: int = 32,
        # episode_length: Optional[int] = None,
        # render_env_id: Optional[str] = None,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        learning_starts: int = 100,
        tau: float = 0.005,
        # num_episode_test: int = 2,  # FIXME : really useful?
        alpha_init: float = 1.0,  # FIXME: check value
        target_entropy_per_dim: float = -1.0,
    ) -> None:
        env, env_params, env_id, continuous = prepare_env(env_id, env_params=env_params)

        if not check_if_environment_has_continuous_actions(env):
            raise ValueError("SAC only supports continuous action spaces.")

        self.env_args = EnvironmentConfig(
            env=env,
            env_params=env_params,
            num_envs=num_envs,
            continuous=continuous,
        )

        self.alpha_args = AlphaConfig(
            learning_rate=learning_rate,
            alpha_init=alpha_init,
        )

        self.network_args = NetworkConfig(
            actor_architecture=actor_architecture,
            critic_architecture=critic_architecture,
            lstm_hidden_size=lstm_hidden_size,
            squash=True,
        )

        self.optimizer_args = OptimizerConfig(
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
        )
        action_dim = get_action_dim(env, env_params)
        target_entropy = target_entropy_per_dim * action_dim
        self.agent_args = SACConfig(
            gamma=gamma,
            tau=tau,
            learning_starts=learning_starts,
            target_entropy=target_entropy,
        )

        self.buffer = get_buffer(
            buffer_size=buffer_size, batch_size=batch_size, num_envs=num_envs
        )

    def train(
        self, seed: int | Sequence[int] = 42, num_timesteps: int = int(1e6)
    ) -> None:
        """Train the agent"""

        if isinstance(seed, int):
            seed = [seed]

        def set_key_and_train(seed):
            key = jax.random.PRNGKey(seed)

            train_jit = make_train(
                env_args=self.env_args,
                optimizer_args=self.optimizer_args,
                network_args=self.network_args,
                buffer=self.buffer,
                agent_args=self.agent_args,
                total_timesteps=num_timesteps,
                alpha_args=self.alpha_args,
            )

            agent_state = train_jit(key)
            return agent_state

        seed = jnp.array(seed)
        agent_state = jax.vmap(set_key_and_train, in_axes=0)(seed)


if __name__ == "__main__":
    env_id = "halfcheetah"
    sac_agent = SAC(env_id=env_id, learning_starts=100)
    sac_agent.train(seed=42, num_timesteps=int(1e5))
