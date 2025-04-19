from typing import Any, Optional, Tuple

import chex
import flashbax as fbx
import jax
import jax.numpy as jnp
from ajax.state import BaseAgentState, EnvironmentConfig, LoadedTrainState
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax.tree_util import Partial as partial


@partial(jax.jit, static_argnames=["mode", "env", "env_params"])
def reset_env(
    rng: jax.Array,
    env: Environment,
    mode: str,
    env_params: Optional[EnvParams] = None,
) -> tuple[jax.Array, EnvState]:
    """
    Reset the environment and return the initial observation and environment state.

    Args:
        rng (jax.Array): Random number generator key.
        env (Environment): The environment to reset.
        mode (str): The mode of the environment ("gymnax" or "brax").
        env_params (Optional[EnvParams]): Parameters for the environment (only for Gymnax).

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
    jax.jit, static_argnames=["mode", "env", "env_params"], donate_argnames=["state"]
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
        env (Environment): The environment to step in.
        mode (str): The mode of the environment ("gymnax" or "brax").
        env_params (Optional[EnvParams]): Parameters for the environment (only for Gymnax).

    Returns:
        Tuple[jax.Array, EnvState, jax.Array, jax.Array, Any]:
        Observation, new environment state, reward, done flag, and additional info.
    """
    if mode == "gymnax":
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
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
    obs: jax.Array,
    done: Optional[jax.Array] = None,
    recurrent: bool = False,
) -> Tuple[jax.Array, LoadedTrainState]:
    """
    Get the policy distribution for the given observation and actor state.

    Args:
        actor_state (LoadedTrainState): The actor's train state.
        obs (jax.Array): The current observation.
        done (Optional[jax.Array]): Done flags for recurrent mode.
        recurrent (bool): Whether the actor is recurrent.

    Returns:
        Tuple[jax.Array, jax.Array]: Policy distribution and new hidden state (if recurrent).
    """

    if recurrent:
        pi, new_actor_hidden_state = actor_state.apply(
            actor_state.params, obs, hidden_state=actor_state.hidden_state, done=done
        )
    else:
        pi = actor_state.apply(actor_state.params, obs)
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


# @partial(
#     chex.chexify,
#     async_check=False,
# )
@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
def get_action_and_new_agent_state(
    agent_state: BaseAgentState,
    obs: jnp.ndarray,
    done: Optional[jax.Array] = None,
    recurrent: bool = False,
):
    """
    Get the action and updated agent state based on the current observation.

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
        obs=maybe_add_axis(obs, recurrent),
        done=maybe_add_axis(done, recurrent),
        recurrent=recurrent,
    )

    rng, action_key = jax.random.split(agent_state.rng)

    action = pi.sample(seed=action_key)

    return (
        action,
        agent_state.replace(rng=rng, actor_state=new_actor_state),
    )


@partial(jax.jit, static_argnames=["recurrent", "mode", "env_args", "buffer"])
def collect_experience(
    agent_state: BaseAgentState,
    _: Any,
    recurrent: bool,
    mode: str,
    env_args: EnvironmentConfig,
    buffer: fbx.flat_buffer.TrajectoryBuffer,
):
    """
    Collect experience by interacting with the environment.

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
    action, agent_state = get_action_and_new_agent_state(
        agent_state,
        agent_state.collector.last_obs,
        agent_state.collector.last_done,
        recurrent=recurrent,
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
            "action": action[:, None],
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
