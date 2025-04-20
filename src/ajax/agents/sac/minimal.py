import chex
import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial


@partial(jax.jit)
def some_function(x, y):
    return x + y


def create_function(key):
    return partial(some_function, y=1)


def minimal_breaking_example(seed: int):
    """Minimal breaking example for the agent."""
    key = jax.random.PRNGKey(seed)
    chex.assert_trees_all_equal(jnp.ones(1), jnp.zeros(1))
    f = create_function(key)
    chex.block_until_chexify_assertions_complete()
    return f(key)


seed = jnp.array([0, 1, 2, 3, 4])
chex.chexify(jax.vmap(minimal_breaking_example, in_axes=0), async_check=False)(seed)
