import pytest
from ajax.environments.create import build_env_from_id, prepare_env
from ajax.wrappers import (
    NormalizeVecObservation,
    NormalizeVecObservationBrax,
    NormalizeVecReward,
    NormalizeVecRewardBrax,
)
from brax.envs import create as create_brax_env
from gymnax import make as make_gymnax_env


@pytest.mark.parametrize(
    "env_id,expected_params",
    [
        ("CartPole-v1", True),  # Gymnax discrete environment
        ("Pendulum-v1", True),  # Gymnax continuous environment
        ("fast", False),  # Brax environment
    ],
)
def test_build_env_from_id(env_id, expected_params):
    env, env_params = build_env_from_id(env_id)
    assert env is not None
    assert (env_params is not None) == expected_params

    if env_id == "fast":
        with pytest.raises(ValueError):
            build_env_from_id("unknown_env")


@pytest.mark.parametrize(
    "env_input,normalize_obs,normalize_rew,expected_continuous",
    [
        ("CartPole-v1", True, False, False),  # Gymnax discrete environment
        ("Pendulum-v1", True, True, True),  # Gymnax continuous environment
        ("fast", False, True, True),  # Brax environment
    ],
)
def test_prepare_env(env_input, normalize_obs, normalize_rew, expected_continuous):
    gamma = 0.99  # Example gamma value for reward normalization
    env, env_params, env_id_out, continuous = prepare_env(
        env_input,
        normalize_obs=normalize_obs,
        normalize_reward=normalize_rew,
        gamma=gamma if normalize_rew else None,
    )
    assert env is not None
    if isinstance(env_input, str):
        assert env_id_out == env_input
        assert continuous == expected_continuous

    # Test Gymnax environments
    if env_input in ["CartPole-v1", "Pendulum-v1"]:
        real_env, real_env_params = make_gymnax_env(env_input)
        assert env_params == real_env_params
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

    # Test Brax environments
    if env_input == "fast":
        real_env = create_brax_env(env_input)
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

    # Check if normalization wrappers are applied
    if normalize_obs:
        assert isinstance(
            env, NormalizeVecObservation | NormalizeVecObservationBrax
        ) or isinstance(env._env, NormalizeVecObservation | NormalizeVecObservationBrax)
    if normalize_rew:
        assert isinstance(env, NormalizeVecReward | NormalizeVecRewardBrax)
