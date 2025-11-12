import functools
from typing import Any, Tuple, Dict
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import pytest
import matplotlib.pyplot as plt

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.emitters.dcrl_me_emitter import DCRLMEConfig, DCRLMEEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition
from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.custom_types import EnvState, Params, RNGKey
from qdax.environments import behavior_descriptor_extractor, reward_offset, create
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting_utils import plot_2d_map_elites_repertoire, plot_oi_map_elites_results


def run_dcrlme_sphere_oi_test(num_iterations: int = 10) -> None:
    """Run DCRLME test with SphereEnv and the LZ76 wrapper with visualization."""

    print(f"Num devices: {jax.device_count()}")
    seed = 42

    env_name = "sphere_oi"
    num_dimensions = 3
    episode_length = 30
    minval = -5.12
    maxval = 5.12
    
    min_bd = (0, -1)
    max_bd = (1, 1) 

    batch_size = 256

    # Archive
    num_init_cvt_samples = 50000
    num_centroids = 1024
    policy_hidden_layer_sizes = (124, 124)

    # DCRL-ME
    ga_batch_size = 128
    dcrl_batch_size = 64
    ai_batch_size = 64
    lengthscale = 0.1

    # GA emitter
    iso_sigma = 0.005
    line_sigma = 0.05

    # DCRL emitter
    critic_hidden_layer_size = (128, 128)
    num_critic_training_steps = 1000
    num_pg_training_steps = 150
    replay_buffer_size = 500_000
    discount = 0.99
    reward_scaling = 1.0
    critic_learning_rate = 3e-4
    actor_learning_rate = 3e-4
    policy_learning_rate = 5e-3
    noise_clip = 0.5
    policy_noise = 0.2
    soft_tau_update = 0.005
    policy_delay = 2

    # Init a random key
    random_key = jax.random.PRNGKey(seed)

    # Init environment with OI wrapper
    env = create(
        env_name,
        n_dimensions=num_dimensions,
        episode_length=episode_length,
        minval=minval,
        maxval=maxval,
        fixed_init_state=False,  # Sphere env doesn't support FixedInitialStateWrapper
        qdax_wrappers_kwargs=[{
            "episode_length": episode_length
        }]
    )
    
    # Note: We do NOT use OffsetRewardWrapper or ClipRewardWrapper for Sphere
    # because the sphere function naturally gives negative rewards that should not be clipped
    # The fitness values from sphere are in the range of [-1000, 0] approximately

    reset_fn = jax.jit(env.reset)

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    actor_dc_network = MLPDC(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch_obs = jnp.zeros(shape=(batch_size, env.observation_size))
    init_params = jax.vmap(policy_network.init)(keys, fake_batch_obs)

    # Define the fonction to play a step with the policy in the environment
    def play_step_fn(
        env_state: EnvState, policy_params: Params, random_key: RNGKey
    ) -> Tuple[EnvState, Params, RNGKey, DCRLTransition]:
        actions = policy_network.apply(policy_params, env_state.obs)
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = DCRLTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
            desc=jnp.zeros(
                env.behavior_descriptor_length,
            )
            * jnp.nan,
            desc_prime=jnp.zeros(
                env.behavior_descriptor_length,
            )
            * jnp.nan,
        )

        return next_state, policy_params, random_key, transition

    # Prepare the scoring function
    bd_extraction_fn = behavior_descriptor_extractor[env_name]
    base_scoring_fn = functools.partial(
        reset_based_scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    sphere_scoring = lambda x: -jnp.sum((x + minval * 0.4) ** 2)
    worst_fitness = episode_length * sphere_scoring(-jnp.ones(num_dimensions) * maxval)
    best_fitness = episode_length * sphere_scoring(jnp.ones(num_dimensions) * maxval * 0.4)

    def scoring_fn(genotypes, random_key):
        fitnesses, descriptors, extra_scores, random_key = base_scoring_fn(genotypes, random_key)
        normalized_fitnesses = (fitnesses - worst_fitness) * 100 / (best_fitness - worst_fitness)
        return normalized_fitnesses, descriptors, extra_scores, random_key

    def metrics_fn(repertoire: MapElitesRepertoire) -> Dict[str, jnp.ndarray]:
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses, where=~grid_empty, initial=-jnp.inf)
        return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}
    
    metrics_function = metrics_fn

    # Define the DCRL-emitter config
    dcrl_emitter_config = DCRLMEConfig(
        ga_batch_size=ga_batch_size,
        dcrl_batch_size=dcrl_batch_size,
        ai_batch_size=ai_batch_size,
        lengthscale=lengthscale,
        critic_hidden_layer_size=critic_hidden_layer_size,
        num_critic_training_steps=num_critic_training_steps,
        num_pg_training_steps=num_pg_training_steps,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        discount=discount,
        reward_scaling=reward_scaling,
        critic_learning_rate=critic_learning_rate,
        actor_learning_rate=actor_learning_rate,
        policy_learning_rate=policy_learning_rate,
        noise_clip=noise_clip,
        policy_noise=policy_noise,
        soft_tau_update=soft_tau_update,
        policy_delay=policy_delay,
    )

    # Get the emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=iso_sigma, line_sigma=line_sigma
    )

    dcrl_emitter = DCRLMEEmitter(
        config=dcrl_emitter_config,
        policy_network=policy_network,
        actor_network=actor_dc_network,
        env=env,
        variation_fn=variation_fn,
    )

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=dcrl_emitter,
        metrics_function=metrics_function,
    )

    # compute initial repertoire
    repertoire, emitter_state, random_key = map_elites.init(
        init_params, centroids, random_key
    )

    print("Running DCRL-ME with Sphere OI wrapper...")

    @jax.jit
    def update_scan_fn(carry: Any, unused: Any) -> Any:
        # iterate over grid
        repertoire, emitter_state, metrics, random_key = map_elites.update(*carry)
        return (repertoire, emitter_state, random_key), metrics

    # Run the algorithm
    (
        repertoire,
        emitter_state,
        random_key,
    ), metrics = jax.lax.scan(
        update_scan_fn,
        (repertoire, emitter_state, random_key),
        (),
        length=num_iterations,
    )
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = "./graficas"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Visualize results
    env_steps = jnp.arange(num_iterations) * episode_length * batch_size
    
    fig1, axes = plot_oi_map_elites_results(
        env_steps=env_steps,
        metrics=metrics,
        repertoire=repertoire,
        min_bd=min_bd,  
        max_bd=max_bd,  
    )
    
    fig1.savefig(os.path.join(plots_dir, f"dcrlm_sphere_metrics_{timestamp}.png"))
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(10, 10))
    plot_2d_map_elites_repertoire(
        repertoire=repertoire,
        ax=ax,
        min_bd=min_bd,  
        max_bd=max_bd,    
        title=f"Archive Final - {env_name} (DCRL-ME)"
    )
    fig2.savefig(os.path.join(plots_dir, f"dcrlm_sphere_archive_{timestamp}.png"))
    plt.close(fig2)
    
    print(f"Final Coverage: {metrics['coverage'][-1]}")
    print(f"Final Max Fitness: {metrics['max_fitness'][-1]}")
    print(f"Final QD Score: {metrics['qd_score'][-1]}")
    
    return repertoire


@pytest.mark.parametrize(
    "num_iterations",
    [5],
)
def test_dcrlme_sphere_oi(num_iterations: int) -> None:
    """Test function for pytest."""
    repertoire = run_dcrlme_sphere_oi_test(num_iterations=50)
    assert repertoire is not None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repertoire_path = f"./repertoires/dcrlm_sphere_oi/{timestamp}/"
    os.makedirs(repertoire_path, exist_ok=True)
    repertoire.save(path=repertoire_path)


if __name__ == "__main__":
    # Run with more iterations for actual experiments
    repertoire = run_dcrlme_sphere_oi_test(num_iterations=100)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repertoire_path = f"./repertoires/dcrlm_sphere_oi/{timestamp}/"
    os.makedirs(repertoire_path, exist_ok=True)
    repertoire.save(path=repertoire_path)
    print(f"Repertoire saved to {repertoire_path}")
