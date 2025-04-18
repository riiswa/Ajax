from typing import Any, Optional, Tuple

import flashbax as fbx
import jax
import jax.numpy as jnp
from ajax.state import CollectorState, EnvironmentConfig
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
    jax.jit, static_argnames=["recurrent", "mode", "env_args", "buffer", "squashing"]
)
def collect_experience(
    agent_state: BaseAgentState,
    _: Any,
    recurrent: bool,
    mode: str,
    env_args: EnvironmentConfig,
    buffer: fbx.flat_buffer.TrajectoryBuffer,
    squashing: bool = False,
):
    collector_state = agent_state.collector_state
    pi, new_actor_hidden_state = get_pi(
        agent_state.actor_state.state,
        agent_state.actor_state.state.params,
        (
            collector_state.last_obs[jnp.newaxis, :]
            if recurrent
            else collector_state.last_obs
        ),
        agent_state.actor_state.hidden_state if recurrent else None,
        collector_state.last_done[jnp.newaxis, :] if recurrent else None,  # type: ignore[index]
        recurrent,
    )
    rng, action_key = jax.random.split(collector_state.rng)
    action = (
        jnp.tanh(pi.sample(seed=action_key))
        if squashing
        else pi.sample(seed=action_key)
    )
    # action = pi.sample(seed=action_key)

    if env_args.continuous:
        action = jnp.float_(action)
    if recurrent:
        action = action.squeeze(0)

    rng, step_key = jax.random.split(rng)

    rng_step = (
        jax.random.split(step_key, env_args.num_envs) if mode == "gymnax" else step_key
    )

    obsv, env_state, reward, done, info = step_env(
        rng_step,
        collector_state.env_state,
        action,
        env_args.env,
        mode,
        env_args.env_params,
    )

    buffer_state = jax.jit(buffer.add, donate_argnums=(1,))(
        collector_state.buffer_state,
        {
            "obs": collector_state.last_obs,
            "action": action[:, None] if mode == "gymnax" else action,
            "reward": reward[:, None],
            "done": collector_state.last_done[:, None],
            "next_obs": obsv,
        },
    )

    actor_state = MaybeRecurrentTrainState(
        state=agent_state.actor_state.state, hidden_state=new_actor_hidden_state
    )
    new_collector_state = CollectorState(
        rng=rng,
        env_state=env_state,
        last_obs=obsv,
        buffer_state=buffer_state,
        timestep=collector_state.timestep + 1,
        last_done=done,
    )

    agent_state = agent_state.replace(
        actor_state=actor_state, collector_state=new_collector_state
    )

    return agent_state, None
