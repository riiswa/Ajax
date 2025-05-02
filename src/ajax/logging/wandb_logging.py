"""Helpers for weights&biases logging"""

import functools
import os
import queue
import threading
from queue import Queue
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import wandb
from flax import struct
from flax.serialization import to_state_dict


@struct.dataclass
class LoggingConfig:
    """Pass along the wandb config cleanly"""

    project_name: str
    run_name: str
    config: dict
    log_frequency: int = 1000
    mode: str = "online"
    group_name: Optional[str] = None
    chunk_size: int = 1000


# Global queue for async logging
logging_queue = Queue()  #  type: ignore[var-annotated]
logging_thread = None
stop_logging = threading.Event()


def init_logging(
    logging_config: LoggingConfig,
    folder: Optional[str] = None,
):
    """Init the wandb run with the logging config"""

    wandb.init(
        project=logging_config.project_name,
        group=logging_config.group_name,
        name=logging_config.run_name,
        save_code=False,
        monitor_gym=False,
        config=logging_config.config,
        mode=logging_config.mode,
        dir=folder,
    )


def log_variables(variables_to_log: dict, commit: bool = True):
    """Log variables (in form of a dict of names and values) into wandb.
    Commit to finish a step."""
    wandb.log(variables_to_log, commit=commit)


def finish_logging():
    """Terminate the wandb run to start a new one"""
    wandb.finish()


def start_async_logging():
    """Start the async logging thread"""
    global logging_thread
    if logging_thread is None or not logging_thread.is_alive():
        stop_logging.clear()
        logging_thread = threading.Thread(target=_logging_worker)
        logging_thread.daemon = True
        logging_thread.start()


def stop_async_logging():
    """Stop the async logging thread"""
    global logging_thread
    if logging_thread is not None:
        stop_logging.set()
        logging_thread.join()
        logging_thread = None


def _logging_worker():
    """Worker thread that processes logging queue"""
    while not stop_logging.is_set():
        try:
            # Non-blocking get from queue
            item = logging_queue.get(timeout=0.1)
            if item is None:
                continue

            run_id, metrics, step, project, name = item

            # Initialize run if not already active
            run = wandb.init(
                project=project,  # project_name
                name=f"{name} {run_id}",  # run_name
                id=run_id,
                resume="must",
                reinit=True,
            )

            # Log metrics
            run.log(metrics, step=step)

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in logging worker: {e}")
            continue


def flatten_dict(dict: Dict) -> Dict:
    return_dict = {}
    for key, val in dict.items():
        if isinstance(val, Dict):
            for subkey, subval in val.items():
                return_dict[f"{key}/{subkey}"] = subval
        else:
            return_dict[key] = val
    return return_dict


def prepare_metrics(aux):
    log_metrics = flatten_dict(to_state_dict(aux))
    return {key: val for (key, val) in log_metrics.items() if not (jnp.isnan(val))}


def vmap_log(
    log_metrics: Dict[str, Any],
    index: int,
    run_ids: Tuple[int],
    logging_config: LoggingConfig,
):
    """
    Log metrics in a vmap fashion, allowing to log multiple runs in parallel.
    This version dumps metrics to a queue for async processing.
    """

    run_id = run_ids[index]

    # Convert JAX arrays to numpy for queue compatibility
    metrics_np = {
        k: jax.device_get(v) for k, v in log_metrics.items() if not jnp.isnan(v)
    }

    step = log_metrics["timestep"]
    # Put metrics in queue for async processing
    logging_queue.put(
        (run_id, metrics_np, step, logging_config.project_name, logging_config.run_name)
    )

    return None  # Return nothing as we're just dumping to queue


def safe_get_env_var(var_name: str, default: str = "") -> str:
    """
    Safely retrieve an environment variable.

    Args:
        var_name (str): The name of the environment variable.
        default (Optional[str]): Default value if the variable is not set.

    Returns:
        Optional[str]: The value of the environment variable or default.
    """
    value = os.environ.get(var_name)
    if value is None:
        return default
    return value


def with_wandb_silent(func: Callable) -> Callable:
    """
    Decorator to temporarily set WANDB_SILENT to true during a function's execution,
    restoring it afterward.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        initial_wandb_silent = safe_get_env_var("WANDB_SILENT")
        try:
            os.environ["WANDB_SILENT"] = "true"
            return func(*args, **kwargs)
        finally:
            os.environ["WANDB_SILENT"] = initial_wandb_silent

    return wrapper
