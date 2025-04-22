import functools
import os

import jax


def trace_profiler(base_path):
    os.makedirs(base_path, exist_ok=True)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with jax.profiler.trace(base_path):
                return jax.block_until_ready(func(*args, **kwargs))

        return wrapper

    return decorator
