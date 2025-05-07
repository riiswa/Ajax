import pytest

from ajax.agents.AVG.AVG import AVG
from ajax.state import AlphaConfig, EnvironmentConfig, NetworkConfig, OptimizerConfig
from ajax.types import BufferType


def test_sac_initialization():
    """Test AVG agent initialization with default parameters."""
    env_id = "Pendulum-v1"
    sac_agent = AVG(env_id=env_id)

    for expected_attr, expected_type in zip(
        (
            "env_args",
            "optimizer_args",
            "network_args",
            "alpha_args",
        ),
        (EnvironmentConfig, OptimizerConfig, NetworkConfig, AlphaConfig, BufferType),
    ):
        assert hasattr(sac_agent, expected_attr)
        assert isinstance(getattr(sac_agent, expected_attr), expected_type)


def test_sac_initialization_with_discrete_env():
    """Test AVG agent initialization fails with a discrete environment."""
    env_id = "CartPole-v1"
    with pytest.raises(ValueError, match="AVG only supports continuous action spaces."):
        AVG(env_id=env_id)


def test_sac_train_single_seed():
    """Test AVG agent's train method with a single seed."""
    env_id = "Pendulum-v1"
    sac_agent = AVG(env_id=env_id, learning_starts=10)
    sac_agent.train(seed=42, num_timesteps=100)
    # try:
    #     sac_agent.train(seed=42, num_timesteps=1000)
    #     success = True  # Placeholder: Add assertions or checks as needed
    # except Exception as e:
    #     success = False
    #     print(f"Training failed with single seed: {e}")
    # assert success, "Training failed for the single seed"


def test_sac_train_multiple_seeds():
    """Test AVG agent's train method with multiple seeds using jax.vmap."""
    env_id = "Pendulum-v1"
    sac_agent = AVG(env_id=env_id, learning_starts=10)
    seeds = [42, 43, 44]
    num_timesteps = 100
    sac_agent.train(seed=seeds, num_timesteps=num_timesteps)
    # try:
    #     sac_agent.train(seed=seeds, num_timesteps=1000)
    #     success = True  # Placeholder: Add assertions or checks as needed
    # except Exception as e:
    #     success = False
    #     print(f"Training failed with multiple seeds: {e}")
    # assert success, "Training failed for one or more seeds"
