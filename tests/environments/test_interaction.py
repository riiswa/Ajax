import distrax
import jax
import jax.numpy as jnp
import optax
import pytest
from ajax.environments.interaction import (
    get_action_and_new_agent_state,
    get_pi,
    reset_env,
    step_env,
)
from ajax.state import (
    BaseAgentState,
    CollectorState,
    EnvironmentConfig,
    EnvStateType,
    MaybeRecurrentTrainState,
)
from brax.envs import create as create_brax_env
from flax.training.train_state import TrainState
from gymnax import make as make_gymnax_env

NUM_ENVS = 4


@pytest.fixture
def gymnax_env():
    env, env_params = make_gymnax_env("CartPole-v1")
    return env, env_params


@pytest.fixture
def brax_env():
    # Create a Brax environment with 4 parallel environments
    env = create_brax_env("fast", batch_size=NUM_ENVS)
    return env


@pytest.fixture
def mock_actor_state():
    """Fixture to create a mock actor state."""
    tx = optax.adam(learning_rate=0.001)
    return MaybeRecurrentTrainState.create(
        apply_fn=lambda params, obs: distrax.Normal(
            jnp.zeros(obs.shape[0]), jnp.ones(obs.shape[0])
        ),
        params={"weights": jnp.array([1.0, 2.0])},
        tx=tx,
    )


def test_reset_env_gymnax(gymnax_env):
    """Test reset_env with a Gymnax environment."""
    env, env_params = gymnax_env
    rng = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)  # Simulate 4 environments
    obs, env_state = reset_env(rng, env, mode="gymnax", env_params=env_params)
    assert obs.shape == (NUM_ENVS, *env.obs_shape)  # Ensure batch size matches
    assert isinstance(env_state, EnvStateType)


def test_reset_env_brax(brax_env):
    """Test reset_env with a Brax environment."""
    env = brax_env
    rng = jax.random.PRNGKey(0)
    obs, env_state = reset_env(rng, env, mode="brax")
    assert obs.shape == (
        NUM_ENVS,
        env.observation_size,
    )  # Ensure observation shape is valid
    assert hasattr(env_state, "obs")  # Ensure env_state has the obs attribute


def test_step_env_gymnax(gymnax_env):
    """Test step_env with a Gymnax environment."""
    env, env_params = gymnax_env
    rng = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)  # Simulate 4 environments
    obs, env_state = reset_env(rng, env, mode="gymnax", env_params=env_params)
    action = jnp.zeros((NUM_ENVS,))  # Dummy actions
    obs, env_state, reward, done, info = step_env(
        rng, env_state, action, env, mode="gymnax", env_params=env_params
    )
    assert obs.shape == (NUM_ENVS, *env.obs_shape)  # Ensure batch size matches
    assert reward.shape == (NUM_ENVS,)
    assert done.shape == (NUM_ENVS,)


def test_step_env_brax(brax_env):
    """Test step_env with a Brax environment."""
    env = brax_env
    rng = jax.random.PRNGKey(0)
    obs, env_state = reset_env(rng, env, mode="brax")
    action = jnp.zeros((NUM_ENVS,))  # Dummy actions
    obs, env_state, reward, done, info = step_env(
        rng, env_state, action, env, mode="brax"
    )
    assert obs.shape == (NUM_ENVS, env.observation_size)
    assert reward.shape == (NUM_ENVS,)
    assert done.shape == (NUM_ENVS,)


def test_get_pi(mock_actor_state):
    """Test get_pi with a mock actor state."""
    obs = jnp.array([[1.0, 2.0]])
    pi, new_hidden_state = get_pi(mock_actor_state, obs, recurrent=False)
    assert isinstance(pi, distrax.Distribution)
    assert new_hidden_state is None


def test_get_action_and_new_agent_state(mock_actor_state, gymnax_env):
    """Test get_action_and_new_agent_state."""
    env, env_params = gymnax_env
    reset_key, rng = jax.random.split(jax.random.PRNGKey(0))
    reset_keys = jax.random.split(reset_key, NUM_ENVS)  # Simulate 4 environments
    obs, env_state = reset_env(reset_keys, env, mode="gymnax", env_params=env_params)

    collector_state = CollectorState(
        rng=rng,
        env_state=env_state,
        last_obs=obs,
    )

    agent_state = BaseAgentState(
        rng=rng,
        collector=collector_state,
        actor_state=mock_actor_state,
        critic_state=mock_actor_state,
        collector_state=collector_state,
    )

    action, new_agent_state = get_action_and_new_agent_state(
        rng, agent_state, obs, recurrent=False
    )
    assert action.shape[0] == obs.shape[0]
    assert new_agent_state.rng is not None


def test_step_env_scan_compatibility(gymnax_env):
    """Test step_env compatibility with jax.lax.scan."""
    env, env_params = gymnax_env
    rngs = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
    obs, env_state = reset_env(rngs, env, mode="gymnax", env_params=env_params)

    def scan_step(carry, _):
        rng, env_state = carry
        action = jnp.zeros((NUM_ENVS,))  # Dummy actions
        obs, env_state, reward, done, info = step_env(
            rng, env_state, action, env, "gymnax", env_params
        )
        return (rng, env_state), (obs, reward, done)

    (_, final_env_state), (obs_seq, reward_seq, done_seq) = jax.lax.scan(
        scan_step, (rngs, env_state), None, length=10
    )
    assert obs_seq.shape == (10, NUM_ENVS, *env.obs_shape)
    assert reward_seq.shape == (10, NUM_ENVS)
    assert done_seq.shape == (10, NUM_ENVS)


def test_step_env_scan_compatibility_brax(brax_env):
    """Test step_env compatibility with jax.lax.scan for Brax."""
    env = brax_env
    rng = jax.random.PRNGKey(0)
    obs, env_state = reset_env(rng, env, mode="brax")

    def scan_step(carry, _):
        rng, env_state = carry
        action = jnp.zeros((NUM_ENVS,))  # Dummy actions
        obs, env_state, reward, done, info = step_env(
            rng, env_state, action, env, "brax"
        )
        return (rng, env_state), (obs, reward, done)

    (_, final_env_state), (obs_seq, reward_seq, done_seq) = jax.lax.scan(
        scan_step, (rng, env_state), None, length=10
    )
    assert obs_seq.shape == (10, NUM_ENVS, env.observation_size)
    assert reward_seq.shape == (10, NUM_ENVS)
    assert done_seq.shape == (10, NUM_ENVS)
