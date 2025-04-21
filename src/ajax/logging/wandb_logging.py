"""Helpers for weights&biases logging"""

from dataclasses import dataclass
from typing import Optional

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
