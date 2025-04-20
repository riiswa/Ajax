import jax.numpy as jnp
import optax
import pytest
from flax.training.train_state import TrainState

from ajax.state import LoadedTrainState


@pytest.fixture
def mock_train_state():
    """Fixture to create a mock TrainState."""
    tx = optax.adam(learning_rate=0.001)
    return TrainState.create(
        apply_fn=lambda x: x,
        params={"weights": jnp.array([1.0, 2.0])},
        tx=tx,
    )


def test_loaded_train_state_initialization(mock_train_state):
    """Test the initialization of LoadedTrainState."""
    # Initialize without hidden_state and target_params
    state = LoadedTrainState.create(
        apply_fn=mock_train_state.apply_fn,
        params=mock_train_state.params,
        tx=mock_train_state.tx,
    )
    assert state.hidden_state is None
    assert state.target_params is None
    assert state.params == mock_train_state.params

    # Initialize with hidden_state and target_params
    hidden_state = jnp.array([0.5, 0.5])
    target_params = {"weights": jnp.array([0.8, 1.8])}
    state_with_hidden = LoadedTrainState.create(
        apply_fn=mock_train_state.apply_fn,
        params=mock_train_state.params,
        tx=mock_train_state.tx,
        hidden_state=hidden_state,
        target_params=target_params,
    )
    assert state_with_hidden.hidden_state is not None
    assert jnp.array_equal(state_with_hidden.hidden_state, hidden_state)
    assert state_with_hidden.target_params == target_params


def test_loaded_train_state_soft_update(mock_train_state):
    """Test the soft update functionality of LoadedTrainState."""
    params = {"weights": jnp.array([1.0, 2.0])}
    target_params = {"weights": jnp.array([0.8, 1.8])}
    state = LoadedTrainState.create(
        apply_fn=mock_train_state.apply_fn,
        params=params,
        tx=mock_train_state.tx,
        target_params=target_params,
    )

    # Perform soft update with tau = 0.5
    updated_state = state.soft_update(tau=0.5)
    expected_target_params = {"weights": jnp.array([0.9, 1.9])}
    assert jnp.allclose(
        updated_state.target_params["weights"], expected_target_params["weights"]
    )
