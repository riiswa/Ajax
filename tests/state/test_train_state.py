import jax.numpy as jnp
import optax
import pytest
from ajax.state import MaybeRecurrentTrainState
from flax.training.train_state import TrainState


@pytest.fixture
def mock_train_state():
    """Fixture to create a mock TrainState."""
    tx = optax.adam(learning_rate=0.001)
    return TrainState.create(
        apply_fn=lambda x: x,
        params={"weights": jnp.array([1.0, 2.0])},
        tx=tx,
    )


def test_maybe_recurrent_train_state_initialization(mock_train_state):
    """Test the initialization of MaybeRecurrentTrainState."""
    # Initialize without hidden_state
    state = MaybeRecurrentTrainState.create(
        apply_fn=mock_train_state.apply_fn,
        params=mock_train_state.params,
        tx=mock_train_state.tx,
    )
    assert state.hidden_state is None
    assert state.params == mock_train_state.params

    # Initialize with hidden_state
    hidden_state = jnp.array([0.5, 0.5])
    state_with_hidden = MaybeRecurrentTrainState.create(
        apply_fn=mock_train_state.apply_fn,
        params=mock_train_state.params,
        tx=mock_train_state.tx,
        hidden_state=hidden_state,
    )
    assert state_with_hidden.hidden_state is not None
    assert jnp.array_equal(state_with_hidden.hidden_state, hidden_state)
