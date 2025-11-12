import os
import time
from typing import Any, Dict, Tuple, Type
import jax
import jax.numpy as jnp
import pytest
import matplotlib.pyplot as plt

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.emitters.cma_emitter import CMAEmitter
from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter
from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter
from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter
from qdax.core.emitters.cma_rnd_emitter import CMARndEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Descriptor, ExtraScores, Fitness, RNGKey
from qdax.environments import create
from qdax.utils.plotting_utils import plot_2d_map_elites_repertoire, plot_oi_map_elites_results

@pytest.mark.parametrize(
    "emitter_type",
    [CMARndEmitter],
)
def test_cma_me_rastrigin(emitter_type: Type[CMAEmitter]) -> None:
    """
    Test CMA-ME algorithm on the RastriginEnv.
    This test also saves a plot of the metrics and a heatmap of the final repertoire.

    """
    env_name = "rastrigin_oi"
    num_iterations = 600
    num_dimensions = 5
    episode_length = 30
    num_init_cvt_samples = 50000
    num_centroids = 1024
    batch_size = 64
    sigma_g = 0.5
    minval = -5.12
    maxval = 5.12
    pool_size = 3
    noise_level = 0.0
    policy_hidden_layer_sizes = (64, 64)

    # Create RastriginEnv with LZ76Wrapper
    env = create(
        env_name,
        n_dimensions=num_dimensions,
        episode_length=episode_length,
        minval=minval,
        maxval=maxval,
        qdax_wrappers_kwargs=[{"name": "lz", "episode_length": episode_length}],
    )

    random_key = jax.random.PRNGKey(0)

    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    random_key, init_key = jax.random.split(random_key)
    example_params = policy_network.init(
        init_key, jnp.zeros((env.observation_size,))
    )
    flat_example, unravel_fn = jax.flatten_util.ravel_pytree(example_params)
    genotype_dim = flat_example.shape[0]

    # Define rastrigin scoring for normalization
    # Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    # Global minimum is at x = 0 with f(0) = 0
    # We negate it for maximization: reward = -f(x)
    A = 10.0
    n = num_dimensions
    rastrigin_scoring = lambda x: -(A * n + jnp.sum(x**2 - A * jnp.cos(2 * jnp.pi * x)))
    
    # Worst fitness: when x is at boundaries (maxval)
    worst_fitness = episode_length * rastrigin_scoring(jnp.ones(num_dimensions) * maxval)
    # Best fitness: when x is at global optimum (0)
    best_fitness = episode_length * rastrigin_scoring(jnp.zeros(num_dimensions))

    # Define scoring function that runs a full episode with a noisy policy
    def scoring_function(
        x: jnp.ndarray, random_key: RNGKey
    ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        random_key, rollout_key = jax.random.split(random_key)
        params_batch = jax.vmap(unravel_fn)(x)
        keys = jax.random.split(rollout_key, x.shape[0])

        action_scale = maxval - minval

        def single_scoring(
            params: Any, sub_key: RNGKey
        ) -> Tuple[Fitness, Descriptor, ExtraScores]:
            state = env.reset(sub_key)

            def step_fn(
                carry: Tuple[Any, jnp.ndarray, RNGKey], _: Any
            ) -> Tuple[Tuple[Any, jnp.ndarray, RNGKey], Any]:
                state, total_reward, key = carry
                key, noise_key = jax.random.split(key)
                raw_action = policy_network.apply(params, state.obs)
                action = raw_action * action_scale
                action += (
                    jax.random.normal(noise_key, shape=action.shape) * noise_level
                )
                next_state = env.step(state, action)
                total_reward += next_state.reward
                return (next_state, total_reward, key), ()

            initial_carry = (state, jnp.float32(0.0), sub_key)
            (final_state, total_reward, _), _ = jax.lax.scan(
                step_fn, initial_carry, (), length=episode_length
            )

            fitness = total_reward
            descriptor = final_state.info["state_descriptor"]

            return fitness, descriptor, {}

        fitnesses, descriptors, extra_scores = jax.vmap(single_scoring)(
            params_batch, keys
        )

        # Normalize fitness to [0, 100] range (0 = worst, 100 = best)
        normalized_fitnesses = (fitnesses - worst_fitness) * 100 / (best_fitness - worst_fitness)

        return (
            normalized_fitnesses,
            descriptors,
            extra_scores,
            random_key,
        )

    # Get behavior descriptor limits from the environment
    min_bd, max_bd = env.behavior_descriptor_limits

    # Define metrics function (now fitness is already normalized)
    def metrics_fn(repertoire: MapElitesRepertoire) -> Dict[str, jnp.ndarray]:
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses, where=~grid_empty, initial=-jnp.inf)
        return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}

    # Initial population
    random_key, init_pop_key = jax.random.split(random_key)
    initial_population = flat_example + 0.1 * jax.random.normal(
        init_pop_key, shape=(batch_size, genotype_dim)
    )

    # Create centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    # Create emitter
    emitter_kwargs = {
        "batch_size": batch_size,
        "genotype_dim": genotype_dim,
        "centroids": centroids,
        "sigma_g": sigma_g,
    }
    emitter = emitter_type(**emitter_kwargs)
    emitter = CMAPoolEmitter(num_states=pool_size, emitter=emitter)

    # Create MAP-Elites algorithm
    map_elites = MAPElites(
        scoring_function=scoring_function, emitter=emitter, metrics_function=metrics_fn
    )

    # Initialize
    repertoire, emitter_state, random_key = map_elites.init(
        initial_population, centroids, random_key
    )

    # Run the algorithm
    (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
        map_elites.scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=num_iterations,
    )

    # --- Plotting ---
    plots_dir = "graficas"
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Metrics plot
    env_steps = jnp.arange(num_iterations) * batch_size * episode_length
    # Save metrics plot
    fig1, axes = plot_oi_map_elites_results(
        env_steps=env_steps,
        metrics=metrics,
        repertoire=repertoire,
        min_bd=min_bd,  
        max_bd=max_bd,  
    )
    fig1.savefig(os.path.join(plots_dir, f"cma_rastrigin_{emitter_type.__name__}_metrics_{timestamp}.png"))
    plt.close(fig1)

    # Save archive plot
    fig2, ax = plt.subplots(figsize=(10, 10))
    plot_2d_map_elites_repertoire(
        repertoire=repertoire,
        ax=ax,
        min_bd=min_bd,  
        max_bd=max_bd,    
        title=f"Archive Final - {env_name} with {emitter_type.__name__}"
    )
    fig2.savefig(os.path.join(plots_dir, f"cma_rastrigin_{emitter_type.__name__}_archive_{timestamp}.png"))
    plt.close(fig2)

    # Count distinct genotypes in the repertoire
    valid_mask = repertoire.fitnesses != -jnp.inf
    valid_genotypes = repertoire.genotypes[valid_mask]
    if valid_genotypes.size > 0:
        unique_genotypes = jnp.unique(valid_genotypes, axis=0)
        num_distinct_genotypes = unique_genotypes.shape[0]
    else:
        num_distinct_genotypes = 0
    print(f"Number of distinct genotypes in repertoire: {num_distinct_genotypes}")

    print(f"Final Coverage: {metrics['coverage'][-1]}")
    print(f"Final Max Fitness: {metrics['max_fitness'][-1]}")
    print(f"Final QD Score: {metrics['qd_score'][-1]}")

if __name__ == "__main__":
    test_cma_me_rastrigin(emitter_type=CMARndEmitter)
