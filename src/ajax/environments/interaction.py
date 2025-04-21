from typing import Any, Optional, Tuple

import chex
import distrax
import flashbax as fbx
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax.tree_util import Partial as partial

from ajax.buffers.utils import init_buffer
from ajax.state import (
    BaseAgentState,
    CollectorState,
    EnvironmentConfig,
    LoadedTrainState,
)


@partial(jax.jit, static_argnames=["mode", "env", "env_params"])
def reset_env(
    rng: jax.Array,
    env: Environment,
    mode: str,
    env_params: Optional[EnvParams] = None,
) -> tuple[jax.Array, EnvState]:
    """
    Reset the environment and return the initial observation and state.

    Args:
        rng (jax.Array): Random number generator key.
        env (Environment): Environment to reset.
        mode (str): Environment mode ("gymnax" or "brax").
        env_params (Optional[EnvParams]): Parameters for Gymnax environments.

    Returns:
        tuple[jax.Array, EnvState]: Initial observation and environment state.
    """
    if mode == "gymnax":
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(rng, env_params)
    else:
        env_state = env.reset(rng)  # ✅ no vmap
        obsv = env_state.obs
    return obsv, env_state


@partial(
    jax.jit,
    static_argnames=["mode", "env", "env_params"],
)
def step_env(
    rng: jax.Array,
    state: jax.Array,
    action: jax.Array,
    env: Environment,
    mode: str,
    env_params: Optional[EnvParams] = None,
) -> Tuple[jax.Array, EnvState, jax.Array, jax.Array, Any]:
    """
    Perform a step in the environment.

    Args:
        rng (jax.Array): Random number generator key.
        state (jax.Array): Current environment state.
        action (jax.Array): Action to take.
        env (Environment): Environment to step in.
        mode (str): Environment mode ("gymnax" or "brax").
        env_params (Optional[EnvParams]): Parameters for Gymnax environments.

    Returns:
        Tuple[jax.Array, EnvState, jax.Array, jax.Array, Any]: Observation, new state, reward, done flag, and info.
    """
    if mode == "gymnax":
        obsv, env_state, reward, done, info = jax.vmap(
            env.step,
            in_axes=(0, 0, 0, None),
        )(rng, state, action, env_params)
        done = jnp.float_(done)
    else:  # ✅ no vmap for brax
        env_state = env.step(state, action)
        obsv, reward, done, info = (
            env_state.obs,
            env_state.reward,
            env_state.done,
            env_state.info,
        )
    return obsv, env_state, reward, done, info


@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
def get_pi(
    actor_state: LoadedTrainState,
    actor_params: FrozenDict,
    obs: jax.Array,
    done: Optional[jax.Array] = None,
    recurrent: bool = False,
) -> Tuple[distrax.Distribution, LoadedTrainState]:
    """
    Get the policy distribution for the given observation and actor state.

    Args:
        actor_state (LoadedTrainState): Actor's train state.
        actor_params (FrozenDict): Parameters of the actor.
        obs (jax.Array): Current observation.
        done (Optional[jax.Array]): Done flags for recurrent mode.
        recurrent (bool): Whether the actor is recurrent.

    Returns:
        Tuple[distrax.Distribution, LoadedTrainState]: Policy distribution and updated actor state.
    """
    obs = maybe_add_axis(obs, recurrent)
    done = maybe_add_axis(done, recurrent)
    if recurrent:
        pi, new_actor_hidden_state = actor_state.apply(
            actor_params,
            obs,
            hidden_state=actor_state.hidden_state,
            done=done,
        )
    else:
        pi = actor_state.apply(actor_params, obs)
        new_actor_hidden_state = None

    return pi, actor_state.replace(hidden_state=new_actor_hidden_state)


@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
def maybe_add_axis(arr: jax.Array, recurrent: bool) -> jax.Array:
    """
    Add an axis to the array if in recurrent mode.

    Args:
        arr (jax.Array): Input array.
        recurrent (bool): Whether to add an axis.

    Returns:
        jax.Array: Array with an additional axis if recurrent.
    """
    return arr[jnp.newaxis, :] if recurrent else arr


@partial(jax.jit, static_argnames=["recurrent"])
def get_action_and_new_agent_state(
    agent_state: BaseAgentState,
    obs: jnp.ndarray,
    done: Optional[jax.Array] = None,
    recurrent: bool = False,
):
    """Get the action and updated agent state based on the current observation.

    Args:
        rng (jax.Array): Random number generator key.
        agent_state (BaseAgentState): Current agent state.
        obs (jnp.ndarray): Current observation.
        done (Optional[jax.Array]): Done flags for recurrent mode.
        recurrent (bool): Whether the agent is recurrent.

    Returns:
        Tuple[jax.Array, BaseAgentState]: Action and updated agent state.

    """
    if recurrent:
        chex.assert_tree_no_nones(done)

    pi, new_actor_state = get_pi(
        actor_state=agent_state.actor_state,
        actor_params=agent_state.actor_state.params,
        obs=obs,
        done=done,
        recurrent=recurrent,
    )

    rng, action_key = jax.random.split(agent_state.rng)

    action = pi.sample(seed=action_key)

    return (
        action,
        agent_state.replace(rng=rng, actor_state=new_actor_state),
    )


@partial(
    jax.jit,
    static_argnames=["recurrent", "mode", "env_args", "buffer"],
    donate_argnums=0,
)
def collect_experience(
    agent_state: BaseAgentState,
    _: Any,
    recurrent: bool,
    mode: str,
    env_args: EnvironmentConfig,
    buffer: fbx.flat_buffer.TrajectoryBuffer,
    uniform: bool = False,
):
    """Collect experience by interacting with the environment.

    Args:
        agent_state (BaseAgentState): Current agent state.
        _ (Any): Placeholder argument for compatibility.
        recurrent (bool): Whether the agent is recurrent.
        mode (str): The mode of the environment ("gymnax" or "brax").
        env_args (EnvironmentConfig): Configuration for the environment.
        buffer (fbx.flat_buffer.TrajectoryBuffer): Buffer to store trajectory data.

    Returns:
        Tuple[BaseAgentState, None]: Updated agent state and None.

    """
    rng, uniform_key = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)
    action, agent_state = get_action_and_new_agent_state(
        agent_state,
        agent_state.collector_state.last_obs,
        agent_state.collector_state.last_done,
        recurrent=recurrent,
    )
    uniform_action = jax.random.uniform(
        uniform_key,
        shape=action.shape,
        minval=-1.0,
        maxval=1.0,
    )

    # Use jax.lax.cond to choose between uniform sampling and policy sampling
    action = jax.lax.cond(
        uniform,
        lambda: uniform_action,
        lambda: action,
    )

    rng, step_key = jax.random.split(agent_state.rng)

    rng_step = (
        jax.random.split(step_key, env_args.num_envs) if mode == "gymnax" else step_key
    )
    obsv, env_state, reward, done, info = step_env(
        rng_step,
        agent_state.collector_state.env_state,
        action,
        env_args.env,
        mode,
        env_args.env_params,
    )

    buffer_state = jax.jit(buffer.add, donate_argnums=(1,))(
        agent_state.collector_state.buffer_state,
        {
            "obs": agent_state.collector_state.last_obs,
            "action": action,  # if action.ndim == 2 else action[:, None]
            "reward": reward[:, None],
            "done": agent_state.collector_state.last_done[:, None],
            "next_obs": obsv,
        },
    )

    new_collector_state = agent_state.collector_state.replace(
        rng=rng,
        env_state=env_state,
        last_obs=obsv,
        buffer_state=buffer_state,
        timestep=agent_state.collector_state.timestep + 1,
        last_done=done,
    )
    agent_state = agent_state.replace(collector_state=new_collector_state)

    return agent_state, None


@partial(jax.jit, static_argnames=["mode", "env_args", "buffer"])
def init_collector_state(
    rng: jax.Array,
    env_args: EnvironmentConfig,
    mode: str,
    buffer: Optional[fbx.flat_buffer.TrajectoryBuffer] = None,
):
    last_done = jnp.zeros(env_args.num_envs)

    reset_key, rng = jax.random.split(rng)
    reset_keys = (
        jax.random.split(reset_key, env_args.num_envs)
        if mode == "gymnax"
        else reset_key
    )
    last_obs, env_state = reset_env(reset_keys, env_args.env, mode, env_args.env_params)
    return CollectorState(
        rng=rng,
        env_state=env_state,
        last_obs=last_obs,
        buffer_state=init_buffer(buffer, env_args) if buffer is not None else None,
        timestep=0,
        last_done=last_done,
    )
