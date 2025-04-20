from typing import Optional

import jax
import jax.numpy as jnp
from ajax.environments.interaction import get_pi, reset_env, step_env
from ajax.environments.utils import check_env_is_gymnax
from brax.envs import create
from gymnax.environments.environment import EnvParams


def evaluate(
    env,
    actor_state,
    num_episodes: int,
    rng: jax.Array,
    env_params: Optional[EnvParams],
    recurrent: bool = False,
    lstm_hidden_size: Optional[int] = None,
    squash: bool = False,
) -> jax.Array:
    key = rng

    mode = "gymnax" if check_env_is_gymnax(env) else "brax"
    if mode == "brax":
        env_name = type(env.unwrapped).__name__.lower()
        env = create(env_name=env_name, batch_size=num_episodes)

    key, reset_key = jax.random.split(key, 2)
    reset_keys = (
        jax.random.split(reset_key, num_episodes) if mode == "gymnax" else reset_key
    )

    def get_action_and_entropy(
        obs: jax.Array,
        key: jax.Array,
        done: Optional[bool] = None,
    ) -> tuple[jax.Array, jax.Array]:
        if actor_state is None:
            raise ValueError("Actor not initialized.")
        pi, _ = get_pi(
            actor_state,
            actor_state.params,
            obs,
            done,
            recurrent,
        )
        action = pi.sample(seed=key)

        return action, pi.entropy()

    obs, state = reset_env(reset_keys, env, mode, env_params)
    done = jnp.zeros(num_episodes, dtype=jnp.int8)
    rewards = jnp.zeros(num_episodes)
    entropy_sum = jnp.zeros(1)
    step_count = jnp.zeros(1)

    carry = (rewards, key, obs, done, state, entropy_sum, step_count)

    def sample_action_and_step_env(carry):
        rewards, rng, obs, done, state, entropy_sum, step_count = carry
        rng, action_key, step_key = jax.random.split(rng, num=3)
        step_keys = (
            jax.random.split(step_key, num_episodes) if mode == "gymnax" else step_key
        )
        actions, entropy = get_action_and_entropy(
            obs, action_key, done if recurrent else None
        )
        obs, state, new_rewards, new_done, _ = step_env(
            step_keys,
            state,
            actions.squeeze(0) if recurrent else actions,
            env,
            mode,
            env_params,
        )

        still_running = 1 - done  # only count unfinished envs
        step_count += still_running.mean()
        entropy_sum += (entropy.mean() * still_running).mean()
        done = done | jnp.int8(new_done)
        rewards += new_rewards * (1 - done)
        return rewards, rng, obs, done, state, entropy_sum, step_count

    def env_not_done(carry):
        done = carry[3]
        return jnp.logical_not(done.all())

    rewards, _, _, _, _, entropy_sum, step_count = jax.lax.while_loop(
        env_not_done, sample_action_and_step_env, carry
    )

    avg_entropy = entropy_sum / jnp.maximum(step_count, 1.0)  # avoid divide by zero

    return rewards, avg_entropy.mean(axis=-1)
