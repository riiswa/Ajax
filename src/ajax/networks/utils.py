from collections.abc import Sequence
from typing import Callable, Optional, Union, cast, get_args

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.linen.initializers import constant, orthogonal
from optax import GradientTransformationExtraArgs

from ajax.types import ActivationFunction


def get_adam_tx(
    learning_rate: Union[float, Callable[[int], float]] = 1e-3,
    max_grad_norm: Optional[float] = 0.5,
    eps: float = 1e-5,
    clipped=True,
) -> GradientTransformationExtraArgs:
    """Return an Adam optimizer with optional gradient clipping.

    Args:
        learning_rate (Union[float, Callable[[int], float]]): Learning rate for the optimizer.
        max_grad_norm (Optional[float]): Maximum gradient norm for clipping.
        eps (float): Epsilon value for numerical stability.
        clipped (bool): Whether to apply gradient clipping.

    Returns:
        GradientTransformationExtraArgs: The configured optimizer.

    """
    if clipped:
        if max_grad_norm is None:
            raise ValueError("Gradient clipping requested but no norm provided.")
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=eps),
        )
    return optax.adam(learning_rate=learning_rate, eps=eps)


def parse_activation(activation: Union[str, ActivationFunction]) -> ActivationFunction:  # type: ignore[return]
    """Parse string representing activation or jax activation function towards\
        jax activation function
    """
    activation_matching = {"relu": nn.relu, "tanh": nn.tanh}

    match activation:
        case str():
            if activation in activation_matching:
                return cast("ActivationFunction", activation_matching[activation])
            raise ValueError(
                (
                    f"Unrecognized activation name {activation}, acceptable activations"
                    f" names are : {activation_matching.keys()}"
                ),
            )
        case activation if isinstance(activation, get_args(ActivationFunction)):
            return activation
        case _:
            raise ValueError(f"Unrecognized activation {activation}")


def parse_layer(
    layer: Union[str, ActivationFunction],
) -> Union[nn.Dense, ActivationFunction]:
    """Parse a layer representation into either a Dense or an activation function"""
    if str(layer).isnumeric():
        return nn.Dense(
            int(cast("str", layer)),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
    return parse_activation(activation=layer)


def parse_architecture(
    architecture: Sequence[Union[str, ActivationFunction]],
) -> Sequence[Union[nn.Dense, ActivationFunction]]:
    """Parse a list of string/module architecture into a list of jax modules"""
    return [parse_layer(layer) for layer in architecture]


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(
            key,
            shape=shape,
            minval=-bound,
            maxval=bound,
            dtype=dtype,
        )

    return _init
