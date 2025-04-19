import flashbax as fbx
import jax
import jax.numpy as jnp
import pytest
from ajax.buffers.utils import get_batch_from_buffer, get_buffer, init_buffer
from ajax.environments.create import build_env_from_id
from ajax.environments.utils import get_state_action_shapes
from ajax.state import EnvironmentConfig


@pytest.fixture
def buffer_fixture():
    buffer_size = 100
    batch_size = 10
    min_length = 5
    num_envs = 2

    buffer = get_buffer(buffer_size, batch_size, min_length, num_envs)
    return buffer, buffer_size, batch_size, num_envs


@pytest.fixture
def buffer_state_fixture(buffer_fixture):
    buffer, buffer_size, batch_size, num_envs = buffer_fixture

    # Create a simple environment configuration
    env_name = "CartPole-v1"
    env, env_params = build_env_from_id(env_name)
    env_args = EnvironmentConfig(
        env=env, env_params=env_params, continuous=False, num_envs=num_envs
    )

    buffer_state = init_buffer(buffer, env_args)
    observation_shape, action_shape = get_state_action_shapes(
        env_args.env, env_args.env_params
    )

    return buffer_state, buffer, env_args, observation_shape, action_shape, batch_size


@pytest.mark.dependency()
def test_get_buffer(buffer_fixture):
    buffer, buffer_size, batch_size, num_envs = buffer_fixture

    assert buffer.init.keywords["add_batch_size"] == num_envs
    assert buffer.init.keywords["max_length_time_axis"] == buffer_size // num_envs
    assert buffer.can_sample.keywords["min_length_time_axis"] == 3


@pytest.mark.dependency(depends=["test_get_buffer"])
def test_init_buffer(buffer_fixture, buffer_state_fixture):
    buffer, buffer_size, batch_size, num_envs = buffer_fixture
    buffer_state, buffer, env_args, observation_shape, action_shape, _ = (
        buffer_state_fixture
    )

    expected_buffer_size = buffer_size // num_envs

    # Check the buffer state structure
    for key in ["obs", "action", "reward", "done", "next_obs"]:
        assert key in buffer_state.experience.keys()

    # Validate shapes
    assert buffer_state.experience["obs"].shape == (
        env_args.num_envs,
        expected_buffer_size,
        *observation_shape,
    )
    assert buffer_state.experience["action"].shape == (
        env_args.num_envs,
        expected_buffer_size,
        *action_shape,
    )
    assert buffer_state.experience["reward"].shape == (
        env_args.num_envs,
        expected_buffer_size,
        1,
    )
    assert buffer_state.experience["done"].shape == (
        env_args.num_envs,
        expected_buffer_size,
        1,
    )
    assert buffer_state.experience["next_obs"].shape == (
        env_args.num_envs,
        expected_buffer_size,
        *observation_shape,
    )


@pytest.mark.dependency(depends=["test_init_buffer"])
def test_get_batch_from_buffer(buffer_state_fixture):
    buffer_state, buffer, env_args, observation_shape, action_shape, batch_size = (
        buffer_state_fixture
    )

    # Create a random key for sampling
    key = jax.random.PRNGKey(0)

    # Sample a batch from the buffer
    obs, done, next_obs, reward, action = get_batch_from_buffer(
        buffer, buffer_state, key
    )

    # Validate shapes of the sampled batch
    assert obs.shape == (batch_size, *observation_shape)
    assert done.shape == (batch_size, 1)
    assert next_obs.shape == (batch_size, *observation_shape)
    assert reward.shape == (batch_size, 1)
    assert action.shape == (batch_size, *action_shape)
