from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import flashbax as fbx
import jax
import jax.numpy as jnp
from ajax.environments.utils import get_state_action_shapes
from ajax.state import EnvironmentConfig
from jax.tree_util import Partial as partial


def get_buffer(
    buffer_size: int,
    batch_size: int,
    min_length: int = 1,
    num_envs: int = 1,
):
    return fbx.make_flat_buffer(
        max_length=buffer_size,
        sample_batch_size=batch_size,
        min_length=min_length,
        add_batch_size=num_envs,
    )


def init_buffer(
    buffer: fbx.flat_buffer.TrajectoryBuffer,
    env_args: EnvironmentConfig,
) -> fbx.flat_buffer.TrajectoryBufferState:
    # Get the state and action shapes for the environment
    observation_shape, action_shape = get_state_action_shapes(
        env_args.env, env_args.env_params
    )

    # Initialize the action as a single action for a single timestep (not batched)
    action = jnp.zeros(
        (action_shape[0],),  # Shape for a single action (e.g., [action_size])
        dtype=jnp.float32 if env_args.continuous else jnp.int32,
    )

    # Initialize the observation for a single timestep (shape: [observation_size])
    obsv = jnp.zeros((observation_shape[0],))

    # Initialize the reward and done flag for a single timestep
    reward = jnp.zeros((1,), dtype=jnp.float32)  # Shape for a single reward
    done = jnp.zeros((1,), dtype=jnp.float32)  # Shape for a single done flag

    # Initialize the buffer state with a single transition
    buffer_state = buffer.init(
        {
            "obs": obsv,  # Single observation (shape: [observation_size])
            "action": action,  # Single action (shape: [action_size])
            "reward": reward,  # Single reward (shape: [1])
            "done": done,  # Single done flag (shape: [1])
            "next_obs": obsv,  # Next observation (same shape as 'obs')
        }
    )

    return buffer_state


@partial(
    jax.jit,
    static_argnames=["buffer"],
)
def get_batch_from_buffer(buffer, buffer_state, key):
    batch = buffer.sample(buffer_state, key).experience
    return (
        batch.first["obs"],
        batch.first["done"],
        batch.first["next_obs"],
        batch.first["reward"],
        batch.first["action"],
    )
