import distrax
import jax
import jax.numpy as jnp

from ajax.agents.AVG.state import NormalizationInfo
from ajax.utils import online_normalize


class SquashedNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        self.tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=self.tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())

    def entropy(self):
        return self.distribution.entropy()


def no_op(x, *args):
    return x


def no_op_tuple(x, *args):
    return x, jnp.ones_like(x.value)


def _normalize_and_update(
    info: NormalizationInfo, square_value: bool
) -> tuple[NormalizationInfo, jnp.array]:
    value = jax.lax.cond(
        square_value,
        jnp.square,
        no_op,
        operand=info.value,
    )
    _, count, mean, mean_2, sigma = online_normalize(
        value, info.count, info.mean, info.mean_2
    )
    updated_info = info.replace(count=count, mean=mean, mean_2=mean_2)
    return updated_info, sigma


def compute_td_error_scaling(
    reward: NormalizationInfo,
    gamma: NormalizationInfo,
    G_return: NormalizationInfo,
) -> tuple[jnp.array, NormalizationInfo, NormalizationInfo, NormalizationInfo]:
    reward, sigma_reward = _normalize_and_update(reward, square_value=False)
    gamma, sigma_gamma = _normalize_and_update(gamma, square_value=False)
    if_nan = jnp.all(jnp.isnan(G_return.value))

    def conditional_replace(new_info, old_info, mask):
        return NormalizationInfo(
            value=jnp.where(mask, old_info.value, new_info.value),
            count=jnp.where(mask, old_info.count, new_info.count),
            mean=jnp.where(mask, old_info.mean, new_info.mean),
            mean_2=jnp.where(mask, old_info.mean_2, new_info.mean_2),
        )

    # Always compute updated G_return
    G_return_norm, _ = _normalize_and_update(G_return, True)

    # Conditionally replace using where
    G_return = conditional_replace(G_return_norm, G_return, if_nan)

    scaling = jnp.sqrt(sigma_reward**2 + G_return.mean * sigma_gamma)

    # td_error_scaling = jax.lax.cond(
    #     G_return.count > 1, no_op, jnp.ones_like, operand=scaling
    # )
    td_error_scaling = jnp.where(G_return.count > 1, scaling, jnp.ones_like(scaling))

    return td_error_scaling, reward, gamma, G_return


def scale_td_error(td_error: jnp.array, scaling_coef: jnp.array) -> jnp.array:
    return td_error / scaling_coef
