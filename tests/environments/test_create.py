import pytest
from ajax.environments.create import build_env_from_id, prepare_env


@pytest.mark.parametrize(
    "env_id,expected_params",
    [
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
    "env_id,normalize_obs,expected_continuous",
    [
        ("Pendulum-v1", True, True),  # Gymnax continuous environment
        ("fast", True, True),  # Brax environment
    ],
)
def test_prepare_env(env_id, normalize_obs, expected_continuous):
    env, env_params, env_id_out, continuous = prepare_env(
        env_id, normalize_obs=normalize_obs
    )
    assert env is not None
    assert env_id_out == env_id
    assert continuous == expected_continuous
