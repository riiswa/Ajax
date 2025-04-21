from unittest.mock import patch

import pytest

from ajax.logging.wandb_logging import (
    LoggingConfig,
    finish_logging,
    init_logging,
    log_variables,
)


@pytest.fixture
def mock_wandb():
    """Fixture to mock the wandb module."""
    with patch("ajax.logging.wandb_logging.wandb") as mock_wandb:
        yield mock_wandb


def test_init_logging(mock_wandb):
    """Test the initialization of wandb logging."""
    logging_config = LoggingConfig(
        project_name="test_project",
        run_name="test_run",
        config={"learning_rate": 0.001},
        mode="offline",
        group_name="test_group",
    )

    init_logging(logging_config, folder="/tmp/logs")

    mock_wandb.init.assert_called_once_with(
        project="test_project",
        group="test_group",
        name="test_run",
        save_code=False,
        monitor_gym=False,
        config={"learning_rate": 0.001},
        mode="offline",
        dir="/tmp/logs",
    )


def test_log_variables(mock_wandb):
    """Test logging variables to wandb."""
    variables_to_log = {"accuracy": 0.95, "loss": 0.05}

    log_variables(variables_to_log, commit=True)

    mock_wandb.log.assert_called_once_with(variables_to_log, commit=True)


def test_finish_logging(mock_wandb):
    """Test finishing the wandb logging session."""
    finish_logging()

    mock_wandb.finish.assert_called_once()
