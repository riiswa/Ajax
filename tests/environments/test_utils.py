import gymnax
import pytest
from ajax.environments.utils import (
    check_env_is_brax,
    check_env_is_gymnax,
    check_if_environment_has_continuous_actions,
    get_env_type,
    get_raw_env,
)
from brax.envs import create as create_brax_env


@pytest.mark.parametrize(
    "env,expected_continuous",
    [
        (gymnax.make("Pendulum-v1")[0], True),
    ],
)
def test_check_if_environment_has_continuous_actions(env, expected_continuous):
    assert check_if_environment_has_continuous_actions(env) == expected_continuous


def test_get_raw_env():
    env, _ = gymnax.make("CartPole-v1")
    raw_env = get_raw_env(env)
    assert raw_env is env


@pytest.mark.parametrize(
    "env,expected_is_brax",
    [
        (create_brax_env("fast"), True),
    ],
)
def test_check_env_is_brax(env, expected_is_brax):
    assert check_env_is_brax(env) == expected_is_brax


@pytest.mark.parametrize(
    "env,expected_is_gymnax",
    [
        (gymnax.make("CartPole-v1")[0], True),
    ],
)
def test_check_env_is_gymnax(env, expected_is_gymnax):
    assert check_env_is_gymnax(env) == expected_is_gymnax


@pytest.mark.parametrize(
    "env,expected_type",
    [
        (create_brax_env("fast"), "brax"),
        (gymnax.make("Pendulum-v1")[0], "gymnax"),
    ],
)
def test_get_env_type(env, expected_type):
    assert get_env_type(env) == expected_type

    with pytest.raises(ValueError):
        get_env_type("unsupported_env")
