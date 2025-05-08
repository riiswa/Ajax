from typing import Sequence, Union, Callable

from flax import struct
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState

from ajax.environments.utils import get_state_action_shapes
from ajax.networks.networks import Encoder, init_network_state
from ajax.networks.utils import get_adam_tx
from ajax.state import OptimizerConfig, EnvironmentConfig
from ajax.types import ActivationFunction
import flax.linen as nn
import jax
import jax.numpy as jnp


class RNDNetwork(nn.Module):
    input_architecture: Sequence[Union[str, ActivationFunction]]

    def setup(self):
        self.predictor = Encoder(input_architecture=self.input_architecture)
        self.target = Encoder(input_architecture=self.input_architecture)

    @nn.compact
    def __call__(self, obs):
        prediction, target = self.predictor(obs), self.target(obs)
        return jnp.square(jax.lax.stop_gradient(target) - prediction).mean(axis=-1, keepdims=True)

def loss_fn(params: FrozenDict, train_state: TrainState, observations: jax.Array):
    return train_state.apply_fn(params, observations).mean()

@struct.dataclass
class RNDState:
    train_state: TrainState
    reward_fn: Callable[["RNDState", jax.Array], jax.Array]
    on_step_fn: Callable[["RNDState", jax.Array], "RNDState"]
    on_update_fn: Callable[["RNDState", jax.Array], "RNDState"]
    normalize_reward: bool = True

class RND:
    def __init__(self, env_config: EnvironmentConfig, rnd_architecture=(256, "relu", 256, "relu"), rnd_learning_rate=1e-4, reward_scale=1.0):
        self.rnd_architecture = rnd_architecture
        self.rnd_learning_rate = rnd_learning_rate
        self.env_config = env_config
        self.reward_scale = reward_scale

    def compute_intrinsic_reward(self, rnd_state: RNDState, observations):
        return rnd_state.train_state.apply_fn(rnd_state.train_state.params, observations)

    def on_step(self, rnd_state: RNDState, observations):
       return rnd_state

    def on_update(self, rnd_state: RNDState, observations):
        value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=False)

        loss, grads = value_and_grad_fn(rnd_state.train_state.params, rnd_state.train_state, observations)

        train_state = rnd_state.train_state.apply_gradients(grads=grads)

        return rnd_state.replace(train_state=train_state)

    def init(self, key):
        rnd_network = RNDNetwork(self.rnd_architecture)
        tx = get_adam_tx(**to_state_dict(OptimizerConfig(
            learning_rate=self.rnd_learning_rate,
            max_grad_norm=0.5,
            clipped=False
        )))

        observation_shape, action_shape = get_state_action_shapes(
            self.env_config.env,
            self.env_config.env_params,
        )
        init_obs = jnp.zeros((self.env_config.num_envs, *observation_shape))

        train_state = init_network_state(
            init_x=init_obs,
            network=rnd_network,
            key=key,
            tx=tx,
            recurrent=False,
            lstm_hidden_size=0,
            num_envs=self.env_config.num_envs,
        )

        return RNDState(
            train_state=train_state,
            reward_fn=jax.tree_util.Partial(self.compute_intrinsic_reward),
            on_step_fn=jax.tree_util.Partial(self.on_step),
            on_update_fn=jax.tree_util.Partial(self.on_update),
        )



