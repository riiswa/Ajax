import distrax
import flashbax as fbx
import jax
import jax.numpy as jnp
import optax
import pytest
from ajax.agents.sac.utils import SquashedNormal
from ajax.buffers.utils import get_buffer, init_buffer
from ajax.environments.interaction import (
    collect_experience,
    get_action_and_new_agent_state,
    get_pi,
    init_collector_state,
    reset_env,
    step_env,
)
from ajax.environments.utils import check_if_environment_has_continuous_actions
from ajax.state import (
    BaseAgentState,
    CollectorState,
    EnvironmentConfig,
    EnvStateType,
    LoadedTrainState,
)
from brax.envs import create as create_brax_env
from flax.training.train_state import TrainState
from gymnax import make as make_gymnax_env

NUM_ENVS = 4


class ReshapedCategorical(distrax.Categorical):
    """A Normal distribution with tanh-squashed samples and corrected log probabilities."""

    def sample(self, seed: jax.Array) -> jax.Array:
        return super().sample(seed=seed)[:, None]


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
def mock_actor_state_continuous():
    """Fixture to create a mock actor state."""
    tx = optax.adam(learning_rate=0.001)
    return LoadedTrainState.create(
        apply_fn=lambda params, obs: distrax.Normal(
            jnp.zeros(NUM_ENVS), jnp.ones(NUM_ENVS)
        ),
        params={"weights": jnp.array([1.0, 2.0])},
        tx=tx,
    )


@pytest.fixture
def mock_actor_state_continuous_squashed():
    """Fixture to create a mock actor state."""
    tx = optax.adam(learning_rate=0.001)
    return LoadedTrainState.create(
        apply_fn=lambda params, obs: SquashedNormal(
            jnp.zeros((obs.shape[0], 1)), jnp.ones((obs.shape[0], 1))
        ),
        params={"weights": jnp.array([1.0, 2.0])},
        tx=tx,
    )


@pytest.fixture
def mock_actor_state_discrete():
    """Fixture to create a mock actor state."""
    tx = optax.adam(learning_rate=0.001)
    return LoadedTrainState.create(
        apply_fn=lambda params, obs: distrax.Categorical(
            probs=jnp.ones((NUM_ENVS, 4)) / 4
        ),
        params={"weights": jnp.array([1.0, 2.0])},
        tx=tx,
    )


@pytest.fixture
def mock_recurrent_actor_state():
    """Fixture to create a mock recurrent actor state."""
    tx = optax.adam(learning_rate=0.001)

    def apply_fn(params, obs, hidden_state=None, done=None):
        # Simulate a recurrent network output
        new_hidden_state = hidden_state + 1 if hidden_state is not None else None
        # pi = distrax.Normal(jnp.zeros(obs.shape[1]), jnp.ones(obs.shape[1]))
        pi = distrax.Categorical(probs=jnp.ones((NUM_ENVS, 4)) / 4)

        return pi, new_hidden_state

    return LoadedTrainState.create(
        apply_fn=apply_fn,
        params={"weights": jnp.array([1.0, 2.0])},
        tx=tx,
        hidden_state=jnp.zeros((NUM_ENVS, 2)),  # Initialize hidden state
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


def test_get_pi(mock_actor_state_discrete):
    """Test get_pi with a mock actor state."""
    obs = jnp.array([[1.0, 2.0]])
    pi, new_actor_state = get_pi(
        actor_state=mock_actor_state_discrete,
        actor_params=mock_actor_state_discrete.params,
        obs=obs,
        recurrent=False,
    )
    assert isinstance(pi, distrax.Categorical)
    assert new_actor_state.hidden_state is None


def test_get_pi(mock_actor_state_continuous):
    """Test get_pi with a mock actor state."""
    obs = jnp.array([[1.0, 2.0]])
    pi, new_actor_state = get_pi(
        actor_state=mock_actor_state_continuous,
        actor_params=mock_actor_state_continuous.params,
        obs=obs,
        recurrent=False,
    )
    assert isinstance(pi, distrax.Normal)
    assert new_actor_state.hidden_state is None


def test_get_pi(mock_actor_state_continuous_squashed):
    """Test get_pi with a mock actor state."""
    obs = jnp.array([[1.0, 2.0]])
    pi, new_actor_state = get_pi(
        mock_actor_state_continuous_squashed,
        mock_actor_state_continuous_squashed.params,
        obs,
        recurrent=False,
    )
    assert isinstance(pi, SquashedNormal)
    assert new_actor_state.hidden_state is None


def test_get_pi_recurrent(mock_recurrent_actor_state):
    """Test get_pi with a mock actor state in recurrent mode."""
    obs = jnp.array([[1.0, 2.0]])
    done = jnp.array([0.0])
    mock_recurrent_actor_state = mock_recurrent_actor_state.replace(
        hidden_state=jnp.zeros((1, 2))
    )
    pi, new_actor_state = get_pi(
        actor_state=mock_recurrent_actor_state,
        actor_params=mock_recurrent_actor_state.params,
        obs=obs,
        done=done,
        recurrent=True,
    )
    assert isinstance(pi, distrax.Distribution)
    assert new_actor_state.hidden_state is not None
    assert new_actor_state.hidden_state.shape == (1, 2)


def test_get_action_and_new_agent_state(mock_actor_state_discrete, gymnax_env):
    """Test get_action_and_new_agent_state for discrete action environments."""
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
        actor_state=mock_actor_state_discrete,
        critic_state=mock_actor_state_discrete,
        collector_state=collector_state,
    )

    action, new_agent_state = get_action_and_new_agent_state(
        agent_state, obs, recurrent=False
    )

    assert action.shape[0] == obs.shape[0]
    assert new_agent_state.rng is not None


def test_get_action_and_new_agent_state_continuous(
    mock_actor_state_continuous, gymnax_env
):
    """Test get_action_and_new_agent_state for continuous action environments."""
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
        actor_state=mock_actor_state_continuous,
        critic_state=mock_actor_state_continuous,
        collector_state=collector_state,
    )

    action, new_agent_state = get_action_and_new_agent_state(
        agent_state, obs, recurrent=False
    )
    assert action.shape[0] == obs.shape[0]
    assert new_agent_state.rng is not None


def test_get_action_and_new_agent_state_recurrent(
    mock_recurrent_actor_state, gymnax_env
):
    mock_actor_state = mock_recurrent_actor_state
    """Test get_action_and_new_agent_state in recurrent mode."""
    env, env_params = gymnax_env
    reset_key, rng = jax.random.split(jax.random.PRNGKey(0))
    reset_keys = jax.random.split(reset_key, NUM_ENVS)  # Simulate 4 environments
    obs, env_state = reset_env(reset_keys, env, mode="gymnax", env_params=env_params)

    collector_state = CollectorState(
        rng=rng,
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros((NUM_ENVS,)),
    )

    mock_actor_state = mock_actor_state.replace(hidden_state=jnp.zeros((NUM_ENVS, 2)))

    agent_state = BaseAgentState(
        rng=rng,
        actor_state=mock_actor_state,
        critic_state=mock_actor_state,
        collector_state=collector_state,
    )

    action, new_agent_state = get_action_and_new_agent_state(
        agent_state, obs, done=jnp.zeros((NUM_ENVS,)), recurrent=True
    )

    assert action.shape[0] == obs.shape[0]
    assert new_agent_state.rng is not None
    assert new_agent_state.actor_state.hidden_state.shape == (NUM_ENVS, 2)


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


def test_step_env_scan_compatibility_recurrent(gymnax_env):
    """Test step_env compatibility with jax.lax.scan in recurrent mode."""
    env, env_params = gymnax_env
    rngs = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
    obs, env_state = reset_env(rngs, env, mode="gymnax", env_params=env_params)

    def scan_step(carry, _):
        rng, env_state, hidden_state = carry
        action = jnp.zeros((NUM_ENVS,))  # Dummy actions
        obs, env_state, reward, done, info = step_env(
            rng, env_state, action, env, "gymnax", env_params
        )
        hidden_state = hidden_state + 1  # Simulate hidden state update
        return (rng, env_state, hidden_state), (obs, reward, done)

    hidden_state = jnp.zeros((NUM_ENVS, 2))  # Dummy hidden state
    (_, final_env_state, final_hidden_state), (obs_seq, reward_seq, done_seq) = (
        jax.lax.scan(scan_step, (rngs, env_state, hidden_state), None, length=10)
    )
    assert obs_seq.shape == (10, NUM_ENVS, *env.obs_shape)
    assert reward_seq.shape == (10, NUM_ENVS)
    assert done_seq.shape == (10, NUM_ENVS)
    assert final_hidden_state.shape == (NUM_ENVS, 2)


def test_get_action_and_new_agent_state_recurrent_without_done(
    mock_recurrent_actor_state, gymnax_env
):
    """Test get_action_and_new_agent_state raises AssertionError when `done` is missing in recurrent mode."""
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
        actor_state=mock_recurrent_actor_state,
        critic_state=mock_recurrent_actor_state,
        collector_state=collector_state,
    )

    with pytest.raises(AssertionError):
        get_action_and_new_agent_state(agent_state, obs, recurrent=True)


@pytest.mark.skip
def test_collect_experience_non_recurrent_discrete(
    mock_actor_state_discrete, gymnax_env
):
    """Test collect_experience in non-recurrent mode for discrete action environments."""
    env, env_params = gymnax_env
    reset_key, rng = jax.random.split(jax.random.PRNGKey(0))
    reset_keys = jax.random.split(reset_key, NUM_ENVS)
    obs, env_state = reset_env(reset_keys, env, mode="gymnax", env_params=env_params)
    env_args = EnvironmentConfig(
        env=env,
        env_params=env_params,
        num_envs=NUM_ENVS,
        continuous=False,
    )
    buffer = get_buffer(buffer_size=10, batch_size=2, num_envs=NUM_ENVS)
    buffer_state = init_buffer(buffer, env_args=env_args)
    collector_state = CollectorState(
        rng=rng,
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros((NUM_ENVS,)),
        buffer_state=buffer_state,
    )

    agent_state = BaseAgentState(
        rng=rng,
        actor_state=mock_actor_state_discrete,
        critic_state=mock_actor_state_discrete,
        collector_state=collector_state,
    )

    updated_agent_state, _ = collect_experience(
        agent_state,
        None,
        recurrent=False,
        mode="gymnax",
        env_args=env_args,
        buffer=buffer,
    )

    assert updated_agent_state.collector_state.last_obs.shape == obs.shape
    assert updated_agent_state.collector_state.last_done.shape == (NUM_ENVS,)


@pytest.mark.skip
def test_collect_experience_non_recurrent_continuous(
    mock_actor_state_continuous, gymnax_env
):
    """Test collect_experience in non-recurrent mode for continuous action environments."""
    env, env_params = gymnax_env
    reset_key, rng = jax.random.split(jax.random.PRNGKey(0))
    reset_keys = jax.random.split(reset_key, NUM_ENVS)
    obs, env_state = reset_env(reset_keys, env, mode="gymnax", env_params=env_params)
    env_args = EnvironmentConfig(
        env=env,
        env_params=env_params,
        num_envs=NUM_ENVS,
        continuous=True,
    )
    buffer = get_buffer(buffer_size=10, batch_size=2, num_envs=NUM_ENVS)
    buffer_state = init_buffer(buffer, env_args=env_args)
    collector_state = CollectorState(
        rng=rng,
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros((NUM_ENVS,)),
        buffer_state=buffer_state,
    )

    agent_state = BaseAgentState(
        rng=rng,
        actor_state=mock_actor_state_continuous,
        critic_state=mock_actor_state_continuous,
        collector_state=collector_state,
    )

    updated_agent_state, _ = collect_experience(
        agent_state,
        None,
        recurrent=False,
        mode="gymnax",
        env_args=env_args,
        buffer=buffer,
    )

    assert updated_agent_state.collector_state.last_obs.shape == obs.shape
    assert updated_agent_state.collector_state.last_done.shape == (NUM_ENVS,)


@pytest.mark.skip
def test_collect_experience_recurrent(mock_recurrent_actor_state, gymnax_env):
    """Test collect_experience in recurrent mode."""
    env, env_params = gymnax_env
    reset_key, rng = jax.random.split(jax.random.PRNGKey(0))
    reset_keys = jax.random.split(reset_key, NUM_ENVS)
    obs, env_state = reset_env(reset_keys, env, mode="gymnax", env_params=env_params)

    env_args = EnvironmentConfig(
        env=env, env_params=env_params, num_envs=NUM_ENVS, continuous=False
    )

    buffer = get_buffer(buffer_size=10, batch_size=2, num_envs=NUM_ENVS)
    buffer_state = init_buffer(buffer, env_args=env_args)

    collector_state = CollectorState(
        rng=rng,
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros((NUM_ENVS,)),
        buffer_state=buffer_state,
    )

    mock_recurrent_actor_state = mock_recurrent_actor_state.replace(
        hidden_state=jnp.zeros((NUM_ENVS, 2))
    )

    agent_state = BaseAgentState(
        rng=rng,
        actor_state=mock_recurrent_actor_state,
        critic_state=mock_recurrent_actor_state,
        collector_state=collector_state,
    )

    updated_agent_state, _ = collect_experience(
        agent_state,
        None,
        recurrent=True,
        mode="gymnax",
        env_args=env_args,
        buffer=buffer,
    )

    assert updated_agent_state.collector_state.last_obs.shape == obs.shape
    assert updated_agent_state.collector_state.last_done.shape == (NUM_ENVS,)
    assert updated_agent_state.actor_state.hidden_state.shape == (NUM_ENVS, 2)


def test_init_collector_state_gymnax(gymnax_env):
    """Test init_collector_state with a Gymnax environment."""
    env, env_params = gymnax_env
    rng = jax.random.PRNGKey(0)
    buffer = get_buffer(buffer_size=10, batch_size=2, num_envs=NUM_ENVS)
    env_args = EnvironmentConfig(
        env=env,
        env_params=env_params,
        num_envs=NUM_ENVS,
        continuous=False,
    )
    collector_state = init_collector_state(rng, env_args, mode="gymnax", buffer=buffer)

    assert collector_state.last_obs.shape == (NUM_ENVS, *env.obs_shape)
    assert collector_state.last_done.shape == (NUM_ENVS,)
    assert collector_state.buffer_state is not None
    assert collector_state.timestep == 0


def test_init_collector_state_brax(brax_env):
    """Test init_collector_state with a Brax environment."""
    env = brax_env
    rng = jax.random.PRNGKey(0)
    buffer = get_buffer(buffer_size=10, batch_size=2, num_envs=NUM_ENVS)
    env_args = EnvironmentConfig(
        env=env,
        env_params=None,
        num_envs=NUM_ENVS,
        continuous=True,
    )
    collector_state = init_collector_state(rng, env_args, mode="brax", buffer=buffer)

    assert collector_state.last_obs.shape == (NUM_ENVS, env.observation_size)
    assert collector_state.last_done.shape == (NUM_ENVS,)
    assert collector_state.buffer_state is not None
    assert collector_state.timestep == 0
