import distrax
import jax
import jax.numpy as jnp

# def correct_log_probs(log_prob: jax.Array, raw_action: jax.Array) -> jax.Array:
#     correction = jnp.sum(
#         2.0 * (jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)), axis=-1
#     )
#     return log_prob - correction


# def sample_actions_and_log_prob(pi: distrax.Distribution, key: jax.Array) -> tuple:
#     # Sample pre-squash action
#     raw_action = pi.sample(seed=key)
#     squashed_action = jnp.tanh(raw_action)

#     # Log-prob of the unsquashed action
#     raw_log_prob = pi.log_prob(raw_action)  # shape (batch_size, action_dim)

#     log_prob = jnp.sum(raw_log_prob, axis=-1)  # shape (batch_size,)

#     corrected_log_prob = correct_log_probs(log_prob=log_prob, raw_action=raw_action)
#     import pdb

#     pdb.set_trace()
#     return squashed_action, corrected_log_prob


# class SquashedNormal(distrax.Transformed):
#     def __init__(self, loc, scale):
#         normal_dist = distrax.Normal(loc, scale)
#         tanh_bijector = distrax.Tanh()
#         super().__init__(distribution=normal_dist, bijector=tanh_bijector)

#     def mean(self):
#         return self.bijector.forward(self.distribution.mean())

#     def entropy(self):
#         return self.distribution.entropy()


class SquashedNormal(distrax.Normal):
    """A Normal distribution with tanh-squashed samples and corrected log probabilities."""

    def sample(self, seed: jax.Array) -> jax.Array:
        """
        Samples an action and applies tanh squashing.

        Args:
            seed (jax.Array): PRNG key for sampling.

        Returns:
            jax.Array: Squashed action.
        """
        # Sample raw action
        raw_action = super().sample(seed=seed)

        # Apply tanh squashing
        squashed_action = jnp.tanh(raw_action)

        return squashed_action

    def mean(self) -> jax.Array:
        """
        Samples an action and applies tanh squashing.

        Args:
            seed (jax.Array): PRNG key for sampling.

        Returns:
            jax.Array: Squashed action.
        """
        # Sample raw action
        raw_mean = super().mean()

        # Apply tanh squashing
        squashed_mean = jnp.tanh(raw_mean)

        return squashed_mean

    def log_prob(self, value: jax.Array) -> jax.Array:
        """
        Computes the corrected log probability for a given squashed action.

        Args:
            value (jax.Array): Squashed action.

        Returns:
            jax.Array: Corrected log probability.
        """
        # Inverse tanh to get the raw action
        raw_action = jnp.arctanh(jnp.clip(value, -0.999999, 0.999999))

        # Compute the raw log probability
        raw_log_prob = super().log_prob(raw_action)

        # Compute the correction for the log probability
        correction = 2.0 * (
            jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)
        )

        return raw_log_prob - correction

    def sample_and_log_prob(self, seed: jax.Array) -> tuple:
        """
        Samples an action, applies tanh squashing, and computes the corrected log probability.

        Args:
            seed (jax.Array): PRNG key for sampling.

        Returns:
            tuple: Squashed action and corrected log probability.
        """
        # Sample raw action and compute its log probability
        raw_action, raw_log_prob = super().sample_and_log_prob(seed=seed)

        # Apply tanh squashing
        squashed_action = jnp.tanh(raw_action)

        # Compute the correction for the log probability
        correction = 2.0 * (
            jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)
        )

        assert (
            raw_log_prob.shape == correction.shape
        ), "Shape mismatch between squashed action and correction."

        corrected_log_prob = raw_log_prob - correction

        assert (
            corrected_log_prob.shape == squashed_action.shape
        ), "Shape mismatch between squashed action and corrected log probability."

        return squashed_action, corrected_log_prob
