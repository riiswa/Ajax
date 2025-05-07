import jax.numpy as jnp

from ajax.utils import online_normalize


def test_online_normalize_initial():
    x = jnp.array([1.0, 2.0, 3.0])
    count = 0
    mean = 0.0
    mean_2 = 0.0

    x_norm, new_count, new_mean, new_mean_2, sigma = online_normalize(
        x, count, mean, mean_2
    )

    assert new_count == 1
    assert jnp.allclose(new_mean, x)
    assert jnp.allclose(new_mean_2, 0.0)
    assert jnp.allclose(sigma, 0.0)
    assert jnp.allclose(x_norm, 0.0)  # x == mean, std == 0 => norm = 0


def test_online_normalize_update():
    x = jnp.array([4.0, 5.0, 6.0])
    count = 1
    mean = jnp.array([1.0, 2.0, 3.0])
    mean_2 = jnp.array([0.0, 0.0, 0.0])

    x_norm, new_count, new_mean, new_mean_2, sigma = online_normalize(
        x, count, mean, mean_2
    )

    expected_mean = jnp.array([2.5, 3.5, 4.5])
    expected_mean_2 = jnp.array([4.5, 4.5, 4.5])
    expected_sigma = jnp.array([2.25, 2.25, 2.25])
    expected_std = jnp.sqrt(expected_sigma)
    expected_x_norm = (x - expected_mean) / expected_std  # delta2 / std

    assert new_count == 2
    assert jnp.allclose(new_mean, expected_mean)
    assert jnp.allclose(new_mean_2, expected_mean_2)
    assert jnp.allclose(sigma, expected_sigma)
    assert jnp.allclose(x_norm, expected_x_norm)


def test_online_normalize_multiple_updates():
    x = jnp.array([7.0, 8.0, 9.0])
    count = 2
    mean = jnp.array([2.5, 3.5, 4.5])
    mean_2 = jnp.array([4.5, 4.5, 4.5])

    x_norm, new_count, new_mean, new_mean_2, sigma = online_normalize(
        x, count, mean, mean_2
    )

    expected_mean = jnp.array([4.0, 5.0, 6.0])
    expected_mean_2 = jnp.array([18.0, 18.0, 18.0])
    expected_sigma = jnp.array([6.0, 6.0, 6.0])
    expected_std = jnp.sqrt(expected_sigma)
    expected_x_norm = (x - expected_mean) / expected_std

    assert new_count == 3
    assert jnp.allclose(new_mean, expected_mean)
    assert jnp.allclose(new_mean_2, expected_mean_2)
    assert jnp.allclose(sigma, expected_sigma)
    assert jnp.allclose(x_norm, expected_x_norm)
