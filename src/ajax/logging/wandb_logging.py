"""Helpers for weights&biases logging"""

import functools
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import wandb


@dataclass
class LoggingConfig:
    """Pass along the wandb config cleanly"""

    project_name: str
    run_name: str
    config: dict
    mode: str = "online"
    group_name: Optional[str] = None


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


def vmap_log(
    log_metrics: Dict[str, Any],
    index: int,
    run_ids: Tuple[int],
    logging_config: LoggingConfig,
):
    """
    Log metrics in a vmap fashion, allowing to log multiple runs in parallel.
    """
    run_id = run_ids[index]
    run = wandb.init(
        project=logging_config.project_name,
        name=f"{logging_config.run_name}  {index}",
        id=run_id,
        resume="must",
        reinit=True,
    )
    run.log(log_metrics)


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
