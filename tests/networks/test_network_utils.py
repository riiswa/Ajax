import optax
import pytest

from ajax.networks.utils import get_adam_tx


def test_get_adam_tx_without_clipping():
    """Test get_adam_tx without gradient clipping."""
    tx = get_adam_tx(learning_rate=0.001, clipped=False)
    assert isinstance(tx, optax.GradientTransformationExtraArgs)


def test_get_adam_tx_with_clipping():
    """Test get_adam_tx with gradient clipping."""
    tx = get_adam_tx(learning_rate=0.001, max_grad_norm=0.5, clipped=True)
    assert isinstance(tx, optax.GradientTransformationExtraArgs)


def test_get_adam_tx_clipping_without_norm():
    """Test get_adam_tx raises ValueError when clipping is requested without max_grad_norm."""
    with pytest.raises(
        ValueError, match="Gradient clipping requested but no norm provided."
    ):
        get_adam_tx(learning_rate=0.001, max_grad_norm=None, clipped=True)
