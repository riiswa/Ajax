from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from brax.envs import create as create_brax_env
from flax import struct

from ajax.wrappers import (
    AutoResetWrapper,
    BraxToGymnasium,
    ClipAction,
    ClipActionBrax,
    FlattenObservationWrapper,
    LogWrapper,
    NormalizeVecObservation,
    NormalizeVecObservationBrax,
    NormalizeVecReward,
    NormalizeVecRewardBrax,
)


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
        _, state = wrapped_env.reset(key, env_params)
    else:  # Brax
        state = wrapped_env.reset(key)

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
            print(state.info)
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
            _, state = wrapped_env.reset(key, env_params)
        elif mode == "brax" and state.done:
            state = wrapped_env.reset(key)

    rewards = jnp.stack(rewards)


@pytest.fixture
def fast_brax_env():
    """Fixture to create a Brax environment."""
    return create_brax_env("fast", batch_size=None, auto_reset=False)


@pytest.fixture
def fail_brax_env():
    """Fixture to create a Brax environment."""
    return create_brax_env("fast")


def test_brax_to_gymnasium_reset(fast_brax_env):
    """Test the reset functionality of the BraxToGymnasium wrapper."""
    wrapped_env = BraxToGymnasium(fast_brax_env)
    obs, info = wrapped_env.reset(seed=42)
    assert obs.shape == (fast_brax_env.observation_size,)
    assert isinstance(info, dict)


def test_brax_to_gymnasium_step(fast_brax_env):
    """Test the step functionality of the BraxToGymnasium wrapper."""
    wrapped_env = BraxToGymnasium(fast_brax_env)
    wrapped_env.reset(seed=42)
    action = jnp.zeros((fast_brax_env.action_size,))
    obs, reward, done, truncated, info = wrapped_env.step(action)
    assert obs.shape == (fast_brax_env.observation_size,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_brax_to_gymnasium_action_space(fast_brax_env):
    """Test the action space of the BraxToGymnasium wrapper."""
    wrapped_env = BraxToGymnasium(fast_brax_env)
    action_space = wrapped_env.action_space
    assert action_space.shape == (fast_brax_env.action_size,)
    assert jnp.all(action_space.low == -1)
    assert jnp.all(action_space.high == 1)


def test_brax_to_gymnasium_observation_space(fast_brax_env):
    """Test the observation space of the BraxToGymnasium wrapper."""
    wrapped_env = BraxToGymnasium(fast_brax_env)
    observation_space = wrapped_env.observation_space
    assert observation_space.shape == (fast_brax_env.observation_size,)
    assert jnp.isinf(observation_space.low).all()
    assert jnp.isinf(observation_space.high).all()


def test_fails_with_autoreset(fail_brax_env):
    with pytest.raises(AssertionError):
        BraxToGymnasium(fail_brax_env)


N_STEPS = 5


@pytest.fixture
def brax_env_with_autoreset():
    """Fixture to create a Brax environment wrapped with AutoResetWrapper."""
    env = create_brax_env(
        "hopper", batch_size=2, auto_reset=False, episode_length=N_STEPS
    )  # fast env has a fixed initial obs, had to switch to a real one
    return AutoResetWrapper(env)


@pytest.mark.slow
def test_autoreset_initialization_differs(brax_env_with_autoreset):
    """Test that the initialization is different at each episode start."""
    wrapped_env = brax_env_with_autoreset
    rng = jax.random.PRNGKey(0)
    state = wrapped_env.reset(rng)
    initial_rng = state.info["rng"]
    initial_obs = state.obs

    # Step the environment until at least one episode ends
    for _ in range(N_STEPS):
        action = jnp.zeros((wrapped_env.n_envs, wrapped_env.action_size))
        state = wrapped_env.step(state, action)

    assert state.done[0]  # env should reset after N_STEPS steps
    # Check that the RNG used for initialization differs after reset

    new_rng = state.info["rng"]
    new_init_obs = state.obs

    assert not jnp.array_equal(initial_rng, new_rng), "RNG should differ after reset"
    assert not jnp.array_equal(
        initial_obs, new_init_obs
    ), "Initial obs should differ after reset"


class DummyEnv:
    """Mock Brax env for testing"""

    def __init__(self):
        self._obs_seq = [
            jnp.array([1.0, 2.0, 3.0]),  # reset obs
            jnp.array([4.0, 5.0, 6.0]),  # step 1 obs
            jnp.array([7.0, 8.0, 9.0]),  # step 2 obs
        ]
        self._step_count = 0

    def reset(self, key):
        obs = self._obs_seq[0]
        return SimpleNamespace(obs=obs, info={})

    def step(self, state, action):
        self._step_count += 1
        obs = self._obs_seq[self._step_count]
        return SimpleNamespace(obs=obs, info=state.info)


def test_normalize_vec_observation_2(brax_env):
    dummy_env = brax_env
    wrapper = NormalizeVecObservationBrax(dummy_env)

    # Initial reset
    key = 0  # dummy key
    state = wrapper.reset(key)

    # Check initial normalization
    assert jnp.all(state.info["count"] == 1)

    assert jnp.allclose(state.info["mean"], jnp.array([1.0, -1.0]))
    assert jnp.allclose(state.obs, jnp.zeros(2))  # normalized should be 0

    # Step 1
    action = None  # dummy action
    state = wrapper.step(state, action)
    expected_mean = jnp.array([1.0, -1.0])
    expected_mean_2 = jnp.array([0.0, 0.0])
    expected_obs = jnp.array([0.0, 0.0])

    assert jnp.all(state.info["count"] == 2)
    assert jnp.allclose(state.info["mean"], expected_mean)
    assert jnp.allclose(state.info["mean_2"], expected_mean_2)
    assert jnp.allclose(state.obs, expected_obs)

    # Step 2
    state = wrapper.step(state, action)
    expected_mean = jnp.array([1.0, -1.0])
    expected_mean_2 = jnp.array([0.0, 0.0])
    expected_obs = jnp.array([0.0, 0.0])

    assert jnp.all(state.info["count"] == 3)
    assert jnp.allclose(state.info["mean"], expected_mean)
    assert jnp.allclose(state.info["mean_2"], expected_mean_2)
    assert jnp.allclose(state.obs, expected_obs)
