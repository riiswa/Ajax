from typing import Any

import jax
import jax.numpy as jnp
import pytest
from ajax.wrappers import (
    ClipAction,
    ClipActionBrax,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeVecObservation,
    NormalizeVecObservationBrax,
    NormalizeVecReward,
    NormalizeVecRewardBrax,
)
from flax import struct


class MockGymnaxEnv:
    """Mock Gymnax environment for testing."""

    def __init__(self):
        self.observation_space = jnp.array([0.0, 0.0])
        self.action_space = jnp.array([-1.0, 1.0])

    def reset(self, key, params=None):
        obs = jnp.array([1.0, -1.0])
        state = {"step_count": 0}
        return obs, state

    def step(self, key, state, action, params=None):
        obs = jnp.array([1.0, -1.0])
        reward = 2.0
        done = state["step_count"] >= 5
        info = {}
        return obs, state, reward, done, info


class MockBraxEnv:
    """Mock Brax environment for testing."""

    def reset(self, key):
        return MockBraxState(
            obs=jnp.array([1.0, -1.0]),
            reward=0.0,
            done=False,
            metrics={},
            pipeline_state=None,
            info={},
        )

    def step(self, state, action):
        obs = jnp.array([1.0, -1.0])
        reward = 1.0
        done = reward >= 5.0
        return MockBraxState(
            obs=obs,
            reward=reward,
            done=done,
            metrics={},
            pipeline_state=None,
            info={},
        )


@struct.dataclass
class MockBraxState:
    """Mock Brax state for testing."""

    obs: jnp.ndarray
    reward: float
    done: bool
    metrics: dict
    pipeline_state: Any
    info: dict


@pytest.fixture
def gymnax_env():
    return MockGymnaxEnv(), None


@pytest.fixture
def brax_env():
    return MockBraxEnv()


def extract_obs(state, mode):
    """Extract observation based on environment type."""
    return state.obs if mode == "brax" else state[0]


@pytest.mark.parametrize(
    "wrapper,env_fixture,mode",
    [
        (FlattenObservationWrapper, "gymnax_env", "gymnax"),
    ],
)
def test_flatten_observation_wrapper(wrapper, env_fixture, mode, request):
    env_data = request.getfixturevalue(env_fixture)
    env, env_params = env_data

    wrapped_env = wrapper(env)
    key = jax.random.PRNGKey(0)

    obs, state = wrapped_env.reset(key, env_params)
    assert obs.ndim == 1  # Observation should be flattened

    obs, state, reward, done, info = wrapped_env.step(key, state, 0, env_params)
    assert obs.ndim == 1  # Observation should remain flattened


@pytest.mark.parametrize(
    "wrapper,env_fixture,mode",
    [
        (LogWrapper, "gymnax_env", "gymnax"),
    ],
)
def test_log_wrapper(wrapper, env_fixture, mode, request):
    env_data = request.getfixturevalue(env_fixture)
    env, env_params = env_data

    wrapped_env = wrapper(env)
    key = jax.random.PRNGKey(0)

    obs, state = wrapped_env.reset(key, env_params)
    assert hasattr(state, "episode_returns")
    assert hasattr(state, "episode_lengths")

    obs, state, reward, done, info = wrapped_env.step(key, state, 0, env_params)
    assert "returned_episode_returns" in info
    assert "returned_episode_lengths" in info


@pytest.mark.parametrize(
    "wrapper,env_fixture,low,high,mode",
    [
        (ClipAction, "gymnax_env", -0.5, 0.5, "gymnax"),
        (ClipActionBrax, "brax_env", -0.5, 0.5, "brax"),
    ],
)
def test_clip_action(wrapper, env_fixture, low, high, mode, request):
    env_data = request.getfixturevalue(env_fixture)
    if mode == "gymnax":
        env, env_params = env_data
    else:  # Brax
        env = env_data
        env_params = None

    wrapped_env = wrapper(env, low=low, high=high)
    key = jax.random.PRNGKey(0)

    if mode == "gymnax":
        obs, state = wrapped_env.reset(key, env_params)
    else:  # Brax
        state = wrapped_env.reset(key)
        obs = state.obs

    out_of_bound_action = jnp.array(1.0)
    if mode == "gymnax":
        obs_out, state_out, reward_out, done_out, info_out = wrapped_env.step(
            key, state, out_of_bound_action, env_params
        )
    else:  # Brax
        state_out = wrapped_env.step(state, out_of_bound_action)
        obs_out = state_out.obs

    at_bound_action = jnp.array(high)
    if mode == "gymnax":
        obs_bound, state_bound, reward_bound, done_bound, info_bound = wrapped_env.step(
            key, state, at_bound_action, env_params
        )
    else:  # Brax
        state_bound = wrapped_env.step(state, at_bound_action)
        obs_bound = state_bound.obs

    assert jnp.allclose(obs_out, obs_bound)
    if mode == "gymnax":
        assert reward_out == reward_bound
    else:
        assert state_out.reward == state_bound.reward


@pytest.mark.parametrize(
    "wrapper,env_fixture,mode",
    [
        (NormalizeVecObservation, "gymnax_env", "gymnax"),
        (NormalizeVecObservationBrax, "brax_env", "brax"),
    ],
)
def test_normalize_vec_observation(wrapper, env_fixture, mode, request):
    env_data = request.getfixturevalue(env_fixture)
    if mode == "gymnax":
        env, env_params = env_data
    else:  # Brax
        env = env_data
        env_params = None

    wrapped_env = wrapper(env)
    key = jax.random.PRNGKey(0)

    if mode == "gymnax":
        obs, state = wrapped_env.reset(key, env_params)
    else:  # Brax
        state = wrapped_env.reset(key)
        obs = state.obs

    observations = [obs]
    for _ in range(10):
        key, subkey = jax.random.split(key)
        action = jnp.array(0)
        if mode == "gymnax":
            obs, state, reward, done, info = wrapped_env.step(
                key, state, action, env_params
            )
        else:  # Brax
            state = wrapped_env.step(state, action)
            obs = state.obs
        observations.append(obs)

    observations = jnp.stack(observations)
    # assert jnp.allclose(jnp.mean(observations), 0, atol=1e-1)
    # assert jnp.allclose(jnp.std(observations), 1, atol=1e-1)


@pytest.mark.parametrize(
    "wrapper,env_fixture,mode",
    [
        (NormalizeVecReward, "gymnax_env", "gymnax"),
        (NormalizeVecRewardBrax, "brax_env", "brax"),
    ],
)
def test_normalize_vec_reward(wrapper, env_fixture, mode, request):
    env_data = request.getfixturevalue(env_fixture)
    if mode == "gymnax":
        env, env_params = env_data
    else:  # Brax
        env = env_data
        env_params = None

    wrapped_env = wrapper(env, gamma=0.99)
    key = jax.random.PRNGKey(0)

    if mode == "gymnax":
        obs, state = wrapped_env.reset(key, env_params)
    else:  # Brax
        state = wrapped_env.reset(key)
        obs = state.obs

    rewards = []
    for _ in range(10):
        key, subkey = jax.random.split(key)
        action = jnp.array(0)
        if mode == "gymnax":
            obs, state, reward, done, info = wrapped_env.step(
                key, state, action, env_params
            )
        else:  # Brax
            state = wrapped_env.step(state, action)
            reward = state.reward
        rewards.append(reward)

        if mode == "gymnax" and done:
            obs, state = wrapped_env.reset(key, env_params)
        elif mode == "brax" and state.done:
            state = wrapped_env.reset(key)
            obs = state.obs

    rewards = jnp.stack(rewards)

    # assert jnp.allclose(jnp.std(rewards), 1, atol=1e-1)
