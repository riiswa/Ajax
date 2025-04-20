from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from ajax.agents.sac.state import SACConfig, SACState
from ajax.buffers.utils import get_batch_from_buffer
from ajax.environments.interaction import (
    collect_experience,
    get_pi,
    init_collector_state,
)
from ajax.environments.utils import check_env_is_gymnax, get_state_action_shapes
from ajax.evaluate import evaluate
from ajax.networks.networks import (
    get_adam_tx,
    get_initialized_actor_critic,
    predict_value,
)
from ajax.state import (
    AlphaConfig,
    EnvironmentConfig,
    LoadedTrainState,
    NetworkConfig,
    OptimizerConfig,
)
from ajax.types import BufferType
from flax.core import FrozenDict
from flax.serialization import to_state_dict
from flax.training.train_state import TrainState
from jax.tree_util import Partial as partial


def create_alpha_train_state(
    learning_rate: float = 3e-4,
    alpha_init: float = 1.0,
) -> TrainState:
    log_alpha = jnp.log(alpha_init)
    params = FrozenDict({"log_alpha": log_alpha})
    tx = get_adam_tx(learning_rate)
    return TrainState.create(
        apply_fn=lambda params: jnp.exp(params["log_alpha"]),  # Optional
        params=params,
        tx=tx,
    )


def init_sac(
    key: jax.Array,
    env_args: EnvironmentConfig,
    optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    alpha_args: AlphaConfig,
    buffer: BufferType,
) -> SACState:
    (
        rng,
        init_key,
        collector_key,
    ) = jax.random.split(key, num=3)

    actor_state, critic_state = get_initialized_actor_critic(
        key=init_key,
        env_config=env_args,
        optimizer_config=optimizer_args,
        network_config=network_args,
        continuous=True,
        action_value=True,
        num_critics=2,
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    collector_state = init_collector_state(
        collector_key, env_args=env_args, mode=mode, buffer=buffer
    )

    alpha = create_alpha_train_state(**to_state_dict(alpha_args))

    return SACState(
        rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=alpha,
        collector_state=collector_state,
    )


@partial(jax.jit, static_argnames=["recurrent", "gamma", "reward_scale"])
def value_loss_function(
    critic_params: FrozenDict,
    critic_states: LoadedTrainState,
    rng: jax.Array,
    actor_state: LoadedTrainState,
    actions: jax.Array,
    observations: jax.Array,
    next_observations: jax.Array,
    dones: Optional[jax.Array],
    rewards: jax.Array,
    gamma: float,
    alpha: jax.Array,
    recurrent: bool,
    reward_scale: float = 5.0,  # Add reward scaling factor here
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    # Apply the reward scaling here
    rewards = rewards * reward_scale

    # Sample next actions from policy π(a|s_{t+1})

    next_pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_state.params,
        obs=next_observations,
        done=dones,
        recurrent=recurrent,
    )
    sample_key, rng = jax.random.split(rng)
    next_actions, log_probs = next_pi.sample_and_log_prob(seed=sample_key)

    # Predict Q-values from critics
    q_preds = predict_value(
        critic_state=critic_states,
        critic_params=critic_params,
        x=jnp.hstack([observations, actions]),
    )
    # Target Q-values using target networks
    q_targets = predict_value(
        critic_state=critic_states,
        critic_params=critic_states.target_params,
        x=jnp.hstack([next_observations, next_actions]),
    )

    # Unpack and unsqueeze if needed
    q1_pred, q2_pred = jnp.split(q_preds, 2, axis=0)
    q1_target, q2_target = jnp.split(q_targets, 2, axis=0)

    # Bellman target and losses
    min_q_target = jnp.minimum(q1_target, q2_target)

    target_q = jax.lax.stop_gradient(
        rewards + gamma * (1.0 - dones) * (min_q_target - alpha * log_probs)
    )

    loss_q1 = jnp.mean((q1_pred - target_q) ** 2)
    loss_q2 = jnp.mean((q2_pred - target_q) ** 2)
    total_loss = loss_q1 + loss_q2

    aux = dict(
        critic_loss=total_loss,
        q1_loss=loss_q1,
        q2_loss=loss_q2,
        q1_pred=q1_pred,
        q2_pred=q2_pred,
        target_q=target_q,
        log_probs=log_probs,
    )

    return total_loss, aux


@partial(
    jax.jit,
    static_argnames=["recurrent"],
)
def policy_loss_function(
    actor_params: FrozenDict,
    actor_state: LoadedTrainState,
    critic_states: LoadedTrainState,
    observations: jax.Array,
    dones: Optional[jax.Array],
    recurrent: bool,
    alpha: jax.Array,
    rng: jax.random.PRNGKey,
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    pi, _ = get_pi(
        actor_state=actor_state,
        actor_params=actor_params,
        obs=observations,
        done=dones,
        recurrent=recurrent,
    )
    sample_key, rng = jax.random.split(rng)
    actions, log_probs = pi.sample_and_log_prob(seed=sample_key)

    # Predict Q-values from critics
    q_preds = predict_value(
        critic_state=critic_states,
        critic_params=critic_states.params,
        x=jnp.hstack([observations, actions]),
    )

    # Unpack and unsqueeze if needed
    q1_pred, q2_pred = jnp.split(q_preds, 2, axis=0)
    q_min = jnp.minimum(q1_pred, q2_pred)
    loss = (alpha * log_probs - q_min).mean()

    return loss, {
        "policy_loss": loss,
        "log_pi": log_probs.mean(),
        "q_min": q_min.mean(),
    }


@partial(
    jax.jit,
    static_argnames=["target_entropy"],
)
def alpha_loss_function(
    log_alpha_params: FrozenDict,
    corrected_log_probs: jax.Array,
    target_entropy: float,
) -> Tuple[jax.Array, Dict[str, Any]]:
    log_alpha = log_alpha_params["log_alpha"]
    alpha = jnp.exp(log_alpha)

    loss = (alpha * (-corrected_log_probs - target_entropy)).mean()

    return loss, {
        "alpha_loss": loss,
        "alpha": alpha,
        "log_alpha": log_alpha,
    }


@partial(jax.jit, static_argnames=["recurrent", "gamma", "reward_scale"])
def update_value_functions(
    observations: jax.Array,
    actions: jax.Array,
    next_observations: jax.Array,
    dones: Optional[jax.Array],
    agent_state: SACState,
    recurrent: bool,
    rewards: jax.Array,
    gamma: float,
    reward_scale: float = 5.0,  # Add reward scaling factor here
) -> Tuple[SACState, Dict[str, Any]]:
    value_loss_key, rng = jax.random.split(agent_state.rng)
    value_and_grad_fn = jax.value_and_grad(value_loss_function, has_aux=True)
    log_alpha = agent_state.alpha.params["log_alpha"]
    alpha = jnp.exp(log_alpha)

    # Call the value loss function with reward scaling applied
    (loss, aux), grads = value_and_grad_fn(
        agent_state.critic_state.params,
        agent_state.critic_state,
        value_loss_key,
        agent_state.actor_state,
        actions,
        observations,
        next_observations,
        dones,
        rewards,
        gamma,
        alpha,
        recurrent,
        reward_scale,  # Pass reward scaling factor here
    )

    updated_critic_state = agent_state.critic_state.apply_gradients(grads=grads)

    return (
        SACState(
            rng=rng,
            actor_state=agent_state.actor_state,
            critic_state=updated_critic_state,
            alpha=agent_state.alpha,
            collector_state=agent_state.collector_state,
        ),
        aux,
    )


@partial(jax.jit, static_argnames=["recurrent"])
def update_policy(
    observations: jax.Array,
    done: Optional[jax.Array],
    agent_state: SACState,
    recurrent: bool,
) -> Tuple[SACState, Dict[str, Any]]:
    rng, policy_key = jax.random.split(agent_state.rng)
    value_and_grad_fn = jax.value_and_grad(policy_loss_function, has_aux=True)
    log_alpha = agent_state.alpha.params["log_alpha"]
    alpha = jnp.exp(log_alpha)
    alpha_min = 0.1
    alpha = jnp.maximum(jnp.exp(log_alpha), alpha_min)
    (loss, aux), grads = value_and_grad_fn(
        agent_state.actor_state.params,
        agent_state.actor_state,
        agent_state.critic_state,
        observations,
        done,
        recurrent,
        alpha,
        policy_key,
    )

    updated_actor_state = agent_state.actor_state.apply_gradients(grads=grads)

    return (
        SACState(
            rng=rng,
            actor_state=updated_actor_state,
            critic_state=agent_state.critic_state,
            alpha=agent_state.alpha,
            collector_state=agent_state.collector_state,
        ),
        aux,
    )


@partial(
    jax.jit,
    static_argnames=["target_entropy", "recurrent"],
)
def update_temperature(
    agent_state: SACState,
    observations: jax.Array,
    dones: Optional[jax.Array],
    target_entropy: float,
    recurrent: bool,
) -> Tuple[SACState, Dict[str, Any]]:
    loss_fn = jax.value_and_grad(alpha_loss_function, has_aux=True)

    pi, _ = get_pi(
        actor_state=agent_state.actor_state,
        actor_params=agent_state.actor_state.params,
        obs=observations,
        done=dones,
        recurrent=recurrent,
    )
    rng, sample_key = jax.random.split(agent_state.rng)
    _, log_probs = pi.sample_and_log_prob(seed=sample_key)

    (loss, aux), grads = loss_fn(
        agent_state.alpha.params,
        log_probs,
        target_entropy,
    )

    new_alpha_state = agent_state.alpha.apply_gradients(grads=grads)

    return (
        SACState(
            rng=rng,
            actor_state=agent_state.actor_state,
            critic_state=agent_state.critic_state,
            alpha=new_alpha_state,
            collector_state=agent_state.collector_state,
        ),
        aux,
    )


@partial(
    jax.jit,
    static_argnames=["tau"],
)
def update_target_networks(
    agent_state: SACState,
    tau: float,
) -> SACState:
    new_critic_state = agent_state.critic_state.soft_update(tau=tau)

    return SACState(
        rng=agent_state.rng,
        actor_state=agent_state.actor_state,
        critic_state=new_critic_state,
        alpha=agent_state.alpha,
        collector_state=agent_state.collector_state,
    )


@partial(
    jax.jit,
    static_argnames=["recurrent", "buffer", "gamma", "tau", "action_dim"],
)
def update_agent(
    agent_state: SACState,
    _: Any,
    buffer: BufferType,
    recurrent: bool,
    gamma: float,
    action_dim: int,
    tau: float,
    num_critic_updates: int = 2,
    target_update_frequency: int = 2,
) -> Tuple[SACState, None]:
    # Sample buffer

    sample_key, rng = jax.random.split(agent_state.rng)
    observations, dones, next_observations, rewards, actions = get_batch_from_buffer(
        buffer, agent_state.collector_state.buffer_state, sample_key
    )
    agent_state = agent_state.replace(rng=rng)

    # Update Q functions
    def critic_update_step(carry, _):
        agent_state = carry
        agent_state, aux_value = update_value_functions(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            agent_state=agent_state,
            recurrent=recurrent,
            rewards=rewards,
            gamma=gamma,
        )

        return agent_state, aux_value

    agent_state, aux_value = jax.lax.scan(
        critic_update_step,
        agent_state,
        None,
        length=num_critic_updates,
    )

    # Update policy
    agent_state, aux_policy = update_policy(
        observations=observations,
        done=dones,
        agent_state=agent_state,
        recurrent=recurrent,
    )

    # Adjust temperature
    target_entropy = -action_dim
    agent_state, aux_temperature = update_temperature(
        agent_state,
        observations=observations,
        target_entropy=target_entropy,
        recurrent=recurrent,
        dones=dones,
    )

    # Update target networks
    # TODO : Only update every update_target_network steps
    agent_state = update_target_networks(agent_state, tau=tau)

    return agent_state, None


@partial(
    jax.jit,
    static_argnames=["env_args", "mode", "recurrent", "buffer"],
)
def training_iteration(
    agent_state: SACState,
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    buffer: BufferType,
    agent_args: SACConfig,
    action_dim: int,
    lstm_hidden_size: Optional[int] = None,
    log_frequency: int = 1000,
):
    """
    Run one iteration of the algorithm : Collect experience from the environment and use it to update the agent.
    """
    # collector_state = agent_state.collector_state
    collect_scan_fn = partial(
        collect_experience,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        buffer=buffer,
    )
    agent_state, _ = jax.lax.scan(collect_scan_fn, agent_state, xs=None, length=1)

    timestep = agent_state.collector_state.timestep

    def do_update(agent_state):
        update_scan_fn = partial(
            update_agent,
            buffer=buffer,
            recurrent=recurrent,
            gamma=agent_args.gamma,
            action_dim=action_dim,
            tau=agent_args.tau,
        )
        agent_state, _ = jax.lax.scan(update_scan_fn, agent_state, xs=None, length=1)
        return agent_state

    def skip_update(agent_state):
        return agent_state

    agent_state = jax.lax.cond(
        timestep >= agent_args.learning_starts,
        do_update,
        skip_update,
        operand=agent_state,
    )

    def run_and_log(agent_state):
        eval_key, rng = jax.random.split(agent_state.rng)
        rewards, entropy = evaluate(
            env_args.env,
            actor_state=agent_state.actor_state,
            num_episodes=4,
            rng=eval_key,
            env_params=env_args.env_params,
            recurrent=recurrent,
            lstm_hidden_size=lstm_hidden_size,
            squash=True,
        )

        # jax.debug.callback can be used to log during JIT
        def log_fn(val):
            timestep_val, rewards_val, entropy_val = val
            print(
                f"[Eval] Step={timestep_val}, Reward={rewards_val},"
                f" Entropy={entropy_val}"
            )

        jax.debug.callback(log_fn, (timestep, rewards, entropy))

        jax.debug.print(
            "[Alpha] {alpha}", alpha=jnp.exp(agent_state.alpha.params["log_alpha"])
        )

        return agent_state.replace(rng=rng)

    def no_op(agent_state):
        return agent_state

    agent_state = jax.lax.cond(
        (timestep % log_frequency) == 0,
        run_and_log,
        no_op,
        agent_state,
    )
    return agent_state, None


def make_train(
    env_args: EnvironmentConfig,
    optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_args: SACConfig,
    total_timesteps: int,
    alpha_args: AlphaConfig,
):
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"

    @partial(
        jax.jit,
    )
    def train(key):
        agent_state = init_sac(
            key=key,
            env_args=env_args,
            optimizer_args=optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            buffer=buffer,
        )

        num_updates = total_timesteps // env_args.num_envs
        _, action_shape = get_state_action_shapes(env_args.env, env_args.env_params)

        training_iteration_scan_fn = partial(
            training_iteration,
            buffer=buffer,
            recurrent=network_args.lstm_hidden_size is not None,
            action_dim=action_shape[0],
            agent_args=agent_args,
            mode=mode,
            env_args=env_args,
        )

        agent_state, _ = jax.lax.scan(
            f=training_iteration_scan_fn, init=agent_state, xs=None, length=num_updates
        )
        return agent_state

    return train
