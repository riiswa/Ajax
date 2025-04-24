from collections.abc import Sequence
from typing import Tuple, Union

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax.linen.initializers import (
    constant,
    xavier_normal,
    xavier_uniform,
)
from flax.serialization import to_state_dict

from ajax.agents.sac.utils import SquashedNormal
from ajax.environments.utils import get_action_dim, get_state_action_shapes
from ajax.networks.scanned_rnn import ScannedRNN
from ajax.networks.utils import get_adam_tx, parse_architecture, uniform_init
from ajax.state import (
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import ActivationFunction, HiddenState

"""
Heavy inspiration from https://github.com/Howuhh/sac-n-jax/blob/main/sac_n_jax_flax.py
"""


class Encoder(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]

    def setup(self):
        layers = parse_architecture(self.input_architecture)
        self.encoder = nn.Sequential(layers)

    @nn.compact
    def __call__(self, input):
        return self.encoder(input)


class Actor(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]
    action_dim: int
    continuous: bool = False
    squash: bool = False

    def setup(self):
        # Initialize the Encoder as a submodule
        self.encoder = Encoder(input_architecture=self.input_architecture)

        if self.continuous:
            self.mean = nn.Dense(
                self.action_dim,
                kernel_init=xavier_normal(),
                bias_init=constant(0.0),
            )
            self.log_std = nn.Dense(
                self.action_dim,
                kernel_init=xavier_uniform(),
                bias_init=constant(0.0),
            )
        else:
            self.model = nn.Sequential(
                [
                    nn.Dense(
                        self.action_dim,
                        kernel_init=uniform_init(1e-3),
                        bias_init=uniform_init(1e-3),
                    ),
                    distrax.Categorical,
                ],
            )

    @nn.compact
    def __call__(self, obs) -> distrax.Distribution:
        # Use the Encoder submodule
        embedding = self.encoder(obs)
        if self.continuous:
            mean = self.mean(embedding)
            log_std = jnp.clip(self.log_std(embedding), -20, 2)
            # log_std = self.log_std(embedding)
            std = jnp.exp(log_std)
            return (
                distrax.Normal(mean, std)
                if not self.squash
                else SquashedNormal(mean, std)
            )
        return self.model(embedding)


class Critic(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]

    def setup(self):
        self.encoder = Encoder(input_architecture=self.input_architecture)
        self.model = nn.Dense(
            1,
            kernel_init=uniform_init(3e-3),
            bias_init=uniform_init(3e-3),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        embedding = self.encoder(x)
        return self.model(embedding)


class MultiCritic(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]
    num: int = 1

    @nn.compact
    def __call__(self, *args, **kwargs):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num,
        )

        return ensemble(self.input_architecture)(*args, **kwargs)


def get_initialized_actor_critic(
    key: jax.Array,
    env_config: EnvironmentConfig,
    optimizer_config: OptimizerConfig,
    network_config: NetworkConfig,
    continuous: bool = False,
    action_value: bool = False,
    squash: bool = False,
    num_critics: int = 1,
) -> Tuple[LoadedTrainState, LoadedTrainState]:
    """Create actor and critic adapted to the environment and following the\
          given architectures
    """
    action_dim = get_action_dim(env_config.env, env_config.env_params)
    actor = Actor(
        input_architecture=network_config.actor_architecture,
        action_dim=action_dim,
        continuous=continuous,
        squash=squash,
    )
    critic = MultiCritic(
        input_architecture=network_config.critic_architecture,
        num=num_critics,
    )
    tx = get_adam_tx(**to_state_dict(optimizer_config))
    actor_key, critic_key = jax.random.split(key)
    observation_shape, action_shape = get_state_action_shapes(
        env_config.env,
        env_config.env_params,
    )
    init_obs = jnp.zeros((env_config.num_envs, *observation_shape))
    init_action = jnp.zeros((env_config.num_envs, *action_shape))
    actor_state = init_network_state(
        init_x=init_obs,
        network=actor,
        key=actor_key,
        tx=tx,
        recurrent=network_config.lstm_hidden_size is not None,
        lstm_hidden_size=network_config.lstm_hidden_size,
        num_envs=env_config.num_envs,
    )

    critic_state = init_network_state(
        init_x=jnp.hstack([init_obs, init_action]) if action_value else init_obs,
        network=critic,
        key=critic_key,
        tx=tx,
        recurrent=network_config.lstm_hidden_size is not None,
        lstm_hidden_size=network_config.lstm_hidden_size,
        num_envs=env_config.num_envs,
    )

    return actor_state, critic_state


def init_hidden_state(
    lstm_hidden_size: int,
    num_envs: int,
    rng: jax.random.PRNGKey,
) -> HiddenState:
    """Initialize the hidden state for the recurrent layer of the network."""
    # rng, _rng = jax.random.split(rng)
    return ScannedRNN(lstm_hidden_size).initialize_carry(rng, num_envs)


def init_network_state(init_x, network, key, tx, recurrent, lstm_hidden_size, num_envs):
    params = FrozenDict(network.init(key, init_x))
    if recurrent:
        _, hidden_state_key = jax.random.split(key)
        hidden_state = init_hidden_state(lstm_hidden_size, num_envs, hidden_state_key)
    else:
        hidden_state = None
    return LoadedTrainState.create(
        params=params,
        tx=tx,
        apply_fn=network.apply,
        hidden_state=hidden_state,
        recurrent=recurrent,
        target_params=params,
    )


def predict_value(
    critic_state: LoadedTrainState,
    critic_params: FrozenDict,
    x: jax.Array,
) -> jax.Array:
    return critic_state.apply_fn(critic_params, x)
