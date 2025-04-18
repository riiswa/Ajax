from typing import Any, Optional, Tuple

import flashbax as fbx
import jax
import jax.numpy as jnp
from ajax.state import BaseAgentState, EnvironmentConfig, MaybeRecurrentTrainState
from gymnax.environments.environment import Environment, EnvParams, EnvState
from jax.tree_util import Partial as partial


@partial(jax.jit, static_argnames=["mode", "env", "env_params"])
def reset_env(
    rng: jax.Array,
    env: Environment,
    mode: str,
    env_params: Optional[EnvParams] = None,
) -> tuple[jax.Array, EnvState]:
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
    actor_state: MaybeRecurrentTrainState,
    obs: jax.Array,
    done: Optional[jax.Array] = None,
    recurrent: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    """
    Get the policy distribution for the given parameters and state.
    """
    if recurrent:
        pi, new_actor_hidden_state = actor_state.apply(
            actor_state.params, obs, hidden_state=actor_state.hidden_state, done=done
        )
    else:
        pi = actor_state.apply(actor_state.params, obs)
        new_actor_hidden_state = None
    return pi, new_actor_hidden_state


@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
def maybe_add_axis(arr: jax.Array, recurrent: bool) -> jax.Array:
    """
    Add an axis to the array if it doesn't already have it.
    """
    return arr[jnp.newaxis, :] if recurrent else arr


@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
def get_action_and_new_agent_state(
    rng: jax.Array,
    agent_state: BaseAgentState,
    obs: jnp.ndarray,
    done: Optional[jax.Array] = None,
    recurrent: bool = False,
):
    """
    Get the policy distribution for the given parameters and state.
    """
    pi, new_actor_hidden_state = get_pi(
        actor_state=agent_state.actor_state,
        obs=maybe_add_axis(obs, recurrent),
        done=maybe_add_axis(done, recurrent),
        recurrent=recurrent,
    )

    rng, action_key = jax.random.split(rng)

    action = pi.sample(seed=action_key)

    return (
        action,
        agent_state.replace(
            rng=rng,
            actor_state=agent_state.actor_state.replace(
                hidden_state=new_actor_hidden_state
            ),
        ),
    )


@partial(
    jax.jit, static_argnames=["recurrent", "mode", "env_args", "buffer", "squashing"]
)
def collect_experience(
    agent_state: BaseAgentState,
    _: Any,
    recurrent: bool,
    mode: str,
    env_args: EnvironmentConfig,
    buffer: fbx.flat_buffer.TrajectoryBuffer,
):
    action, agent_state = get_action_and_new_agent_state(
        agent_state, agent_state.collector.last_obs, agent_state.collector.last_done
    )
    action = maybe_add_axis(action, recurrent)

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
            "action": maybe_add_axis(action, mode == "gymnax"),
            "reward": reward[:, None],
            "done": agent_state.collector_state.last_done[:, None],
            "next_obs": obsv,
        },
    )
    agent_state.collector_state = agent_state.collector_state.replace(
        rng=rng,
        env_state=env_state,
        last_obs=obsv,
        buffer_state=buffer_state,
        timestep=agent_state.collector_state.timestep + 1,
        last_done=done,
    )

    return agent_state, None
