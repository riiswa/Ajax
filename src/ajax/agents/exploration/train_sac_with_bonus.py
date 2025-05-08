from dataclasses import fields
from typing import Any, Callable, Optional, Tuple, Sequence

import jax
import jax.numpy as jnp
from flax.serialization import to_state_dict
from jax.tree_util import Partial as partial
from wandb.wandb_agent import agent

from ajax.agents.exploration.rnd import RNDState, RND
from ajax.agents.sac.sac import SAC
from ajax.agents.sac.state import SACConfig, SACState
from ajax.agents.sac.train_sac import ValueAuxiliaries, AuxiliaryLogs, flatten_dict, no_op, \
    create_alpha_train_state, update_value_functions, update_policy, update_temperature, update_target_networks
from ajax.environments.interaction import (
    should_use_uniform_sampling, get_action_and_new_agent_state, assert_shape, _select_uniform_action,
    _select_policy_action, step_env, reset_env,
)
from ajax.environments.utils import get_state_action_shapes, check_env_is_gymnax
from ajax.evaluate import compute_episodic_mean_reward, evaluate
from ajax.logging.wandb_logging import LoggingConfig, vmap_log, start_async_logging, with_wandb_silent, \
    stop_async_logging
from ajax.networks.networks import get_initialized_actor_critic
from ajax.state import (
    EnvironmentConfig, BaseAgentState, OptimizerConfig, NetworkConfig, AlphaConfig, CollectorState,
)
from ajax.types import BufferType

import flashbax as fbx

import wandb

@partial(jax.jit, static_argnames=["mode", "env_args", "buffer"])
def init_collector_state_with_bonus(
    rng: jax.Array,
    env_args: EnvironmentConfig,
    mode: str,
    buffer: Optional[fbx.flat_buffer.TrajectoryBuffer] = None,
):
    last_done = jnp.zeros(env_args.num_envs)

    reset_key, rng = jax.random.split(rng)
    reset_keys = (
        jax.random.split(reset_key, env_args.num_envs)
        if mode == "gymnax"
        else reset_key
    )
    last_obs, env_state = reset_env(reset_keys, env_args.env, mode, env_args.env_params)
    return CollectorState(
        rng=rng,
        env_state=env_state,
        last_obs=last_obs,
        buffer_state=init_buffer_with_bonus(buffer, env_args) if buffer is not None else None,
        timestep=0,
        last_done=last_done,
    )

def init_sac_with_bonus(
    key: jax.Array,
    env_args: EnvironmentConfig,
    optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    alpha_args: AlphaConfig,
    buffer: BufferType,
) -> SACState:
    """
    Initialize the SAC agent's state, including actor, critic, alpha, and collector states.

    Args:
        key (jax.Array): Random number generator key.
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        alpha_args (AlphaConfig): Alpha configuration.
        buffer (BufferType): Replay buffer.

    Returns:
        SACState: Initialized SAC agent state.
    """
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
        squash=True,
        num_critics=2,
    )
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    collector_state = init_collector_state_with_bonus(
        collector_key,
        env_args=env_args,
        mode=mode,
        buffer=buffer,
    )

    alpha = create_alpha_train_state(**to_state_dict(alpha_args))

    return SACState(
        rng=rng,
        eval_rng=rng,
        actor_state=actor_state,
        critic_state=critic_state,
        alpha=alpha,
        collector_state=collector_state,
    )

def init_buffer_with_bonus(
    buffer: fbx.flat_buffer.TrajectoryBuffer,
    env_args: EnvironmentConfig,
) -> fbx.flat_buffer.TrajectoryBufferState:
    # Get the state and action shapes for the environment
    observation_shape, action_shape = get_state_action_shapes(
        env_args.env,
        env_args.env_params,
    )

    # Initialize the action as a single action for a single timestep (not batched)
    action = jnp.zeros(
        (action_shape[0],),  # Shape for a single action (e.g., [action_size])
        dtype=jnp.float32 if env_args.continuous else jnp.int32,
    )

    # Initialize the observation for a single timestep (shape: [observation_size])
    obsv = jnp.zeros((observation_shape[0],))

    # Initialize the reward and done flag for a single timestep
    reward = jnp.zeros((1,), dtype=jnp.float32)  # Shape for a single reward
    done = jnp.zeros((1,), dtype=jnp.float32)  # Shape for a single done flag

    # Initialize the buffer state with a single transition
    buffer_state = buffer.init(
        {
            "obs": obsv,  # Single observation (shape: [observation_size])
            "action": action,  # Single action (shape: [action_size])
            "reward": reward,  # Single reward (shape: [1])
            "intrinsic_reward": reward,  # Single intrinsic reward (shape: [1])
            "done": done,  # Single done flag (shape: [1])
            "next_obs": obsv,  # Next observation (same shape as 'obs')
        },
    )

    return buffer_state


@partial(
    jax.jit,
    static_argnames=["recurrent", "mode", "env_args", "buffer"],
    # donate_argnums=0,
)
def collect_experience_with_bonus(
    args: Tuple[BaseAgentState, RNDState],
    _: Any,
    recurrent: bool,
    mode: str,
    env_args: EnvironmentConfig,
    buffer: BufferType,
    uniform: bool = False,
):
    """Collect experience by interacting with the environment.

    Args:
        agent_state (BaseAgentState): Current agent state.
        _ (Any): Placeholder argument for compatibility.
        recurrent (bool): Whether the agent is recurrent.
        mode (str): The mode of the environment ("gymnax" or "brax").
        env_args (EnvironmentConfig): Configuration for the environment.
        buffer (fbx.flat_buffer.TrajectoryBuffer): Buffer to store trajectory data.

    Returns:
        Tuple[BaseAgentState, None]: Updated agent state and None.

    """
    agent_state, rnd_state = args
    rng, uniform_key = jax.random.split(agent_state.rng)
    agent_state = agent_state.replace(rng=rng)
    agent_action, agent_state = get_action_and_new_agent_state(
        agent_state,
        agent_state.collector_state.last_obs,
        agent_state.collector_state.last_done,
        recurrent=recurrent,
    )
    uniform_action = jax.random.uniform(
        uniform_key,
        shape=agent_action.shape,
        minval=-1.0,
        maxval=1.0,
    )

    assert_shape(uniform_action, agent_action.shape)

    # Use jax.lax.cond to choose between uniform sampling and policy sampling
    action = jax.lax.cond(
        uniform,
        _select_uniform_action,
        _select_policy_action,
        uniform_action,
        agent_action,
    )

    rng, step_key = jax.random.split(agent_state.rng)

    rng_step = (
        jax.random.split(step_key, env_args.num_envs) if mode == "gymnax" else step_key
    )
    obsv, env_state, reward, done, info = step_env(
        rng_step,
        agent_state.collector_state.env_state,
        action,
        env_args.env,
        mode,
        env_args.env_params,
    )

    intrinsic_reward = rnd_state.reward_fn(rnd_state, obsv)
    rnd_state = rnd_state.on_step_fn(rnd_state, obsv)

    buffer_state = buffer.add(
        agent_state.collector_state.buffer_state,
        {
            "obs": agent_state.collector_state.last_obs,
            "action": action,  # if action.ndim == 2 else action[:, None]
            "reward": reward[:, None],
            "intrinsic_reward": intrinsic_reward,
            "done": agent_state.collector_state.last_done[:, None],
            "next_obs": obsv,
        },
    )

    new_collector_state = agent_state.collector_state.replace(
        rng=rng,
        env_state=env_state,
        last_obs=obsv,
        buffer_state=buffer_state,
        timestep=agent_state.collector_state.timestep + 1,
        last_done=done,
    )
    agent_state = agent_state.replace(collector_state=new_collector_state)

    return (agent_state, rnd_state), None

@partial(
    jax.jit,
    static_argnames=["buffer"],
)
def get_batch_from_buffer_with_bonus(buffer, buffer_state, key):
    batch = buffer.sample(buffer_state, key).experience

    obs = batch.first["obs"]
    act = batch.first["action"]
    rew = batch.first["reward"]
    next_obs = batch.first["next_obs"]
    i_rew = batch.first["intrinsic_reward"]
    done = batch.first["done"]

    return obs, done, next_obs, rew, i_rew, act

@partial(
    jax.jit,
    static_argnames=["recurrent", "buffer", "gamma", "tau", "action_dim"],
)
def update_agent_with_bonus(
    args: Tuple[SACState, RNDState],
    _: Any,
    buffer: BufferType,
    recurrent: bool,
    gamma: float,
    action_dim: int,
    tau: float,
    num_critic_updates: int = 1,
    target_update_frequency: int = 1,
    reward_scale: float = 5.0,
) -> Tuple[SACState, AuxiliaryLogs]:
    """
    Update the SAC agent, including critic, actor, and temperature updates.

    Args:
        agent_state (SACState): Current SAC agent state.
        _ (Any): Placeholder for scan compatibility.
        buffer (BufferType): Replay buffer.
        recurrent (bool): Whether the model is recurrent.
        gamma (float): Discount factor.
        action_dim (int): Action dimensionality.
        tau (float): Soft update coefficient.
        num_critic_updates (int): Number of critic updates per step.
        target_update_frequency (int): Frequency of target network updates.
        reward_scale (float): Reward scaling factor.

    Returns:
        Tuple[SACState, None]: Updated agent state.
    """
    agent_state, rnd_state = args

    # Sample buffer

    sample_key, rng = jax.random.split(agent_state.rng)
    observations, dones, next_observations, rewards, intrinsic_rewards, actions = get_batch_from_buffer_with_bonus(
        buffer,
        agent_state.collector_state.buffer_state,
        sample_key,
    )
    agent_state = agent_state.replace(rng=rng)

    rnd_state = rnd_state.on_update_fn(rnd_state, observations)

    rewards = rewards + jax.lax.cond(
        rnd_state.normalize_reward,
        lambda: (intrinsic_rewards - intrinsic_rewards.mean(axis=-1, keepdims=True)) / jnp.sqrt(intrinsic_rewards.var(axis=-1, keepdims=True) + 1e-8),
        lambda: intrinsic_rewards
    )

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
            reward_scale=reward_scale,
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
    aux = AuxiliaryLogs(
        temperature=aux_temperature,
        policy=aux_policy,
        value=ValueAuxiliaries(
            **{key: val.flatten() for key, val in to_state_dict(aux_value).items()}
        ),
    )
    return (agent_state, rnd_state), aux

@partial(
    jax.jit,
    static_argnames=[
        "env_args",
        "mode",
        "recurrent",
        "buffer",
        "log_frequency",
        "num_episode_test",
        "log_fn",
        "log",
        "verbose",
        "action_dim",
        "lstm_hidden_size",
        "agent_args",
        "chunk_size",
        "horizon",
    ],
)
def training_iteration_with_bonus(
    args: Tuple[SACState, RNDState],
    _: Any,
    env_args: EnvironmentConfig,
    mode: str,
    recurrent: bool,
    buffer: BufferType,
    agent_args: SACConfig,
    action_dim: int,
    lstm_hidden_size: Optional[int] = None,
    log_frequency: int = 1000,
    chunk_size: int = 1000,
    horizon: int = 10000,
    num_episode_test: int = 10,
    log_fn: Optional[Callable] = None,
    index: Optional[int] = None,
    log: bool = False,
    verbose: bool = False,
):
    """
    Perform one training iteration, including experience collection and agent updates.

    Args:
        agent_state (SACState): Current SAC agent state.
        _ (Any): Placeholder for scan compatibility.
        env_args (EnvironmentConfig): Environment configuration.
        mode (str): Environment mode ("gymnax" or "brax").
        recurrent (bool): Whether the model is recurrent.
        buffer (BufferType): Replay buffer.
        agent_args (SACConfig): SAC agent configuration.
        action_dim (int): Action dimensionality.
        lstm_hidden_size (Optional[int]): LSTM hidden size for recurrent models.
        log_frequency (int): Frequency of logging and evaluation.
        num_episode_test (int): Number of episodes for evaluation.

    Returns:
        Tuple[SACState, None]: Updated agent state.
    """
    # collector_state = agent_state.collector_state

    agent_state, rnd_state = args

    timestep = agent_state.collector_state.timestep
    uniform = should_use_uniform_sampling(timestep, agent_args.learning_starts)

    collect_scan_fn = partial(
        collect_experience_with_bonus,
        recurrent=recurrent,
        mode=mode,
        env_args=env_args,
        buffer=buffer,
        uniform=uniform,
    )
    (agent_state, rnd_state), _ = jax.lax.scan(collect_scan_fn, (agent_state, rnd_state), xs=None, length=1)
    timestep = agent_state.collector_state.timestep

    def do_update(args):
        agent_state, rnd_state =  args
        update_scan_fn = partial(
            update_agent_with_bonus,
            buffer=buffer,
            recurrent=recurrent,
            gamma=agent_args.gamma,
            action_dim=action_dim,
            tau=agent_args.tau,
            reward_scale=agent_args.reward_scale,
        )
        (agent_state, rnd_state), aux = jax.lax.scan(update_scan_fn, (agent_state, rnd_state), xs=None, length=1)
        aux = aux.replace(
            value=ValueAuxiliaries(
                **{key: val.flatten() for key, val in to_state_dict(aux.value).items()}
            )
        )
        return agent_state, rnd_state, aux

    def fill_with_nan(dataclass):
        """
        Recursively fills all fields of a dataclass with jnp.nan.
        """
        nan = jnp.ones(1) * jnp.nan
        dict = {}
        for field in fields(dataclass):
            sub_dataclass = field.type
            if hasattr(
                sub_dataclass, "__dataclass_fields__"
            ):  # Check if the field is another dataclass
                dict[field.name] = fill_with_nan(sub_dataclass)
            else:
                dict[field.name] = nan
        return dataclass(**dict)

    def skip_update(args):
        agent_state, rnd_state = args
        return agent_state, rnd_state, fill_with_nan(AuxiliaryLogs)

    agent_state, rnd_state, aux = jax.lax.cond(
        timestep >= agent_args.learning_starts,
        do_update,
        skip_update,
        operand=(agent_state, rnd_state),
    )

    def run_and_log(agent_state, aux, index):
        eval_key = agent_state.eval_rng
        eval_rewards, eval_entropy = evaluate(
            env_args.env,
            actor_state=agent_state.actor_state,
            num_episodes=num_episode_test,
            rng=eval_key,
            env_params=env_args.env_params,
            recurrent=recurrent,
            lstm_hidden_size=lstm_hidden_size,
        )

        if log:
            episodic_mean_reward = compute_episodic_mean_reward(
                agent_state, chunk_size=chunk_size, horizon=horizon
            )
            metrics_to_log = {
                "timestep": timestep,
                "Eval/episodic mean reward": eval_rewards,
                "Eval/episodic entropy": eval_entropy,
                "Train/episodic mean reward": episodic_mean_reward,
            }
            metrics_to_log.update(flatten_dict(to_state_dict(aux)))
            jax.debug.callback(log_fn, metrics_to_log, index)

        if verbose:
            jax.debug.print(
                (
                    "[Eval] Step={timestep_val}, Reward={rewards_val},"
                    " Entropy={entropy_val}"
                ),
                timestep_val=timestep,
                rewards_val=eval_rewards,
                entropy_val=eval_entropy,
            )

    if log:
        _, eval_rng = jax.random.split(agent_state.eval_rng)
        agent_state = agent_state.replace(eval_rng=eval_rng)
        flag = jnp.logical_and((timestep % log_frequency) == 1, timestep > 1)
        jax.lax.cond(flag, run_and_log, no_op, agent_state, aux, index)
        del aux

    jax.clear_caches()
    # gc.collect()
    return (agent_state, rnd_state), None

def make_train_with_bonus(
    env_args: EnvironmentConfig,
    optimizer_args: OptimizerConfig,
    network_args: NetworkConfig,
    buffer: BufferType,
    agent_args: SACConfig,
    alpha_args: AlphaConfig,
    total_timesteps: int,
    num_episode_test: int,
    run_ids: Optional[Sequence[str]] = None,
    logging_config: Optional[LoggingConfig] = None,
):
    """
    Create the training function for the SAC agent.

    Args:
        env_args (EnvironmentConfig): Environment configuration.
        optimizer_args (OptimizerConfig): Optimizer configuration.
        network_args (NetworkConfig): Network configuration.
        buffer (BufferType): Replay buffer.
        agent_args (SACConfig): SAC agent configuration.
        alpha_args (AlphaConfig): Alpha configuration.
        total_timesteps (int): Total timesteps for training.
        num_episode_test (int): Number of episodes for evaluation during training.

    Returns:
        Callable: JIT-compiled training function.
    """
    mode = "gymnax" if check_env_is_gymnax(env_args.env) else "brax"
    log = logging_config is not None
    log_fn = partial(vmap_log, run_ids=run_ids, logging_config=logging_config)

    # Start async logging if logging is enabled
    if logging_config is not None:
        start_async_logging()

    @partial(jax.jit)
    def train(key, index: Optional[int] = None):
        """Train the SAC agent."""
        rng_agent, rng_bonus = jax.random.split(key)
        agent_state = init_sac_with_bonus(
            key=rng_agent,
            env_args=env_args,
            optimizer_args=optimizer_args,
            network_args=network_args,
            alpha_args=alpha_args,
            buffer=buffer,
        )

        rnd_state = RND(env_args).init(rng_bonus)

        num_updates = total_timesteps // env_args.num_envs
        _, action_shape = get_state_action_shapes(env_args.env, env_args.env_params)

        training_iteration_scan_fn = partial(
            training_iteration_with_bonus,
            buffer=buffer,
            recurrent=network_args.lstm_hidden_size is not None,
            action_dim=action_shape[0],
            agent_args=agent_args,
            mode=mode,
            env_args=env_args,
            num_episode_test=num_episode_test,
            log_fn=log_fn,
            index=index,
            log=log,
            log_frequency=(
                logging_config.log_frequency if logging_config is not None else None
            ),
            chunk_size=(
                logging_config.chunk_size if logging_config is not None else None
            ),
            horizon=(logging_config.horizon if logging_config is not None else None),
        )

        (agent_state, rnd_state), _ = jax.lax.scan(
            f=training_iteration_scan_fn,
            init=(agent_state, rnd_state),
            xs=None,
            length=num_updates,
        )

        # Stop async logging if it was started
        # if logging_config is not None:
        #     stop_async_logging()

        return agent_state

    return train

class SACWithBonus(SAC):

    @with_wandb_silent
    def train(
        self,
        seed: int | Sequence[int] = 42,
        num_timesteps: int = int(1e6),
        num_episode_test: int = 10,
        logging_config: Optional[LoggingConfig] = None,
    ) -> None:
        """
        Train the SAC agent.

        Args:
            seed (int | Sequence[int]): Random seed(s) for training.
            num_timesteps (int): Total number of timesteps for training.
            num_episode_test (int): Number of episodes for evaluation during training.
        """
        if isinstance(seed, int):
            seed = [seed]

        if logging_config is not None:
            logging_config.config.update(self.config)
            run_ids = [wandb.util.generate_id() for _ in range(len(seed))]
            for index, run_id in enumerate(run_ids):
                wandb.init(
                    project=logging_config.project_name,
                    name=f"{logging_config.run_name}  {index}",
                    id=run_id,
                    resume="never",
                    reinit=True,
                    config=logging_config.config,
                )
        else:
            run_ids = None

        def set_key_and_train(seed, index):
            key = jax.random.PRNGKey(seed)

            train_jit = make_train_with_bonus(
                env_args=self.env_args,
                optimizer_args=self.optimizer_args,
                network_args=self.network_args,
                buffer=self.buffer,
                agent_args=self.agent_args,
                total_timesteps=num_timesteps,
                alpha_args=self.alpha_args,
                num_episode_test=num_episode_test,
                run_ids=run_ids,
                logging_config=logging_config,
            )

            agent_state = train_jit(key, index)
            stop_async_logging()
            return agent_state

        index = jnp.arange(len(seed))
        seed = jnp.array(seed)
        jax.vmap(set_key_and_train, in_axes=0)(seed, index)


if __name__ == "__main__":
    wandb.login(key="b4fddd9baeda8e4f846c9ff0fd1412fffb5a13a6")
    n_seeds = 1
    log_frequency = 20_000
    chunk_size = 1000
    logging_config = LoggingConfig(
        "match_SAC_reproducibility",
        "test",
        config={
            "debug": False,
            "log_frequency": log_frequency,
            "n_seeds": n_seeds,
            "chunk_size": chunk_size,
            "chunk_size": chunk_size,
        },
        log_frequency=log_frequency,
        chunk_size=chunk_size,
        horizon=10_000,
    )
    env_id = "halfcheetah"
    sac_agent = SACWithBonus(env_id=env_id, learning_starts=int(1e4), batch_size=256)
    sac_agent.train(
        seed=list(range(n_seeds)),
        num_timesteps=int(1e5),
        logging_config=logging_config,
    )