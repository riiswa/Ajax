import jax.numpy as jnp


def online_normalize(
    x: jnp.array, count: int, mean: float, mean_2: float, eps: float = 1e-8
) -> tuple[jnp.array, int, float, float, float]:
    count += 1
    delta = x - mean
    mean = mean + delta / count
    delta_2 = x - mean
    mean_2 = mean_2 + delta * delta_2
    variance = mean_2 / count
    std = jnp.sqrt(variance + eps)
    x_norm = delta_2 / std
    return x_norm, count, mean, mean_2, variance
