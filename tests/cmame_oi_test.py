import argparse
import functools
import json
import os
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

EMITTER_NAMES = ("optimizing", "random", "improvement")


def _configure_jax_runtime() -> None:
    # Avoid large up-front VRAM reservations, which were making GPU startup flaky.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _assert_gpu_linear_algebra_available(jax: Any, jnp: Any) -> None:
    if jax.default_backend() != "gpu":
        return

    try:
        eigenvalues, _ = jnp.linalg.eigh(jnp.eye(2, dtype=jnp.float32))
        eigenvalues.block_until_ready()
    except Exception as exc:  # pragma: no cover - exercised in WSL GPU env
        raise RuntimeError(
            "GPU CMA-ME requires a working cuSolver backend. "
            "This environment fails on a minimal jnp.linalg.eigh call, so "
            "CMA-ES will crash when updating the covariance matrix. "
            "Repair the JAX CUDA installation or run CMA-ME on CPU."
        ) from exc


def run_cmame_oi_test(
    env_name: str = "halfcheetah_oil",
    num_iterations: int = 1000,
    emitter_name: str = "random",
    policy_hidden_layer_sizes: tuple[int, ...] = (128, 128),
    batch_size: int = 36,
    pool_size: int = 3,
    num_init_cvt_samples: int = 50000,
) -> Any:
    if emitter_name not in EMITTER_NAMES:
        raise ValueError(f"Unknown emitter_name '{emitter_name}'")

    _configure_jax_runtime()

    import jax
    import jax.numpy as jnp

    # Import inside the runner to avoid import-order crashes observed in WSL.
    from qdax.environments.base_wrappers import QDEnv as _QDEnvBootstrap  # noqa: F401
    import qdax.environments as environments
    from qdax.core.containers.mapelites_repertoire import (
        compute_cvt_centroids,
    )
    from qdax.core.emitters.cma_improvement_emitter import CMAImprovementEmitter
    from qdax.core.emitters.cma_opt_emitter import CMAOptimizingEmitter
    from qdax.core.emitters.cma_pool_emitter import CMAPoolEmitter
    from qdax.core.emitters.cma_rnd_emitter import CMARndEmitter
    from qdax.core.map_elites import MAPElites
    from qdax.core.neuroevolution.networks.networks import MLP
    from qdax.tasks.brax_envs import (
        make_policy_network_play_step_fn_brax,
        reset_based_scoring_function_brax_envs,
    )
    from qdax.utils.metrics import default_qd_metrics
    from qdax.utils.plotting_utils import (
        plot_2d_map_elites_repertoire,
        plot_oi_map_elites_results,
    )

    emitter_types: Dict[str, type] = {
        "optimizing": CMAOptimizingEmitter,
        "random": CMARndEmitter,
        "improvement": CMAImprovementEmitter,
    }

    print(f"Num devices: {jax.device_count()}")
    print(f"Device: {jax.devices()[0]}")
    _assert_gpu_linear_algebra_available(jax, jnp)

    seed = 42
    episode_length = 100
    sigma_g = 0.02
    grid_shape = (64, 64)
    num_centroids = grid_shape[0] * grid_shape[1]

    random_key = jax.random.PRNGKey(seed)

    env = environments.create(
        env_name,
        episode_length=episode_length,
        fixed_init_state=True,
        qdax_wrappers_kwargs=[{"episode_length": episode_length}],
    )

    min_bd, max_bd = env.behavior_descriptor_limits

    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    play_step_fn = make_policy_network_play_step_fn_brax(env, policy_network)
    reset_fn = jax.jit(env.reset)
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]
    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=environments.reward_offset[env_name] * episode_length,
    )

    random_key, init_key = jax.random.split(random_key)
    example_params = policy_network.init(init_key, jnp.zeros((env.observation_size,)))
    flat_example, unravel_fn = jax.flatten_util.ravel_pytree(example_params)
    genotype_dim = flat_example.shape[0]

    def scoring_fn(
        flat_policies_params: jnp.ndarray,
        random_key: jnp.ndarray,
    ) -> Any:
        policies_params = jax.vmap(unravel_fn)(flat_policies_params)
        return reset_based_scoring_function_brax_envs(
            policies_params=policies_params,
            random_key=random_key,
            episode_length=episode_length,
            play_reset_fn=reset_fn,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=bd_extraction_fn,
        )

    random_key, init_pop_key = jax.random.split(random_key)
    initial_population = flat_example + 0.1 * jax.random.normal(
        init_pop_key,
        shape=(batch_size, genotype_dim),
    )

    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=num_init_cvt_samples,
        num_centroids=num_centroids,
        minval=min_bd,
        maxval=max_bd,
        random_key=random_key,
    )

    emitter_type = emitter_types[emitter_name]
    print(f"CMA emitter: {emitter_name}")
    print(
        "Config: "
        f"hidden_layers={policy_hidden_layer_sizes}, "
        f"batch_size={batch_size}, "
        f"pool_size={pool_size}, "
        f"num_init_cvt_samples={num_init_cvt_samples}"
    )

    emitter_kwargs = {
        "batch_size": batch_size,
        "genotype_dim": genotype_dim,
        "centroids": centroids,
        "sigma_g": sigma_g,
    }
    emitter = emitter_type(**emitter_kwargs)
    emitter = CMAPoolEmitter(num_states=pool_size, emitter=emitter)

    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
    )

    repertoire, emitter_state, random_key = map_elites.init(
        initial_population, centroids, random_key
    )

    (repertoire, emitter_state, random_key), metrics = jax.lax.scan(
        map_elites.scan_update,
        (repertoire, emitter_state, random_key),
        (),
        length=num_iterations,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = "./oil_figures"
    os.makedirs(plots_dir, exist_ok=True)

    repertoire_dir = f"./repertoires/cmame_oil/{timestamp}/"
    os.makedirs(repertoire_dir, exist_ok=True)

    env_steps = jnp.arange(num_iterations) * batch_size * episode_length

    fig1, _ = plot_oi_map_elites_results(
        env_steps=env_steps,
        metrics=metrics,
        repertoire=repertoire,
        min_bd=min_bd,
        max_bd=max_bd,
    )
    metrics_plot_path = os.path.join(plots_dir, f"cmame_metrics_{timestamp}.png")
    fig1.savefig(metrics_plot_path)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(10, 10))
    plot_2d_map_elites_repertoire(
        repertoire=repertoire,
        ax=ax,
        min_bd=min_bd,
        max_bd=max_bd,
        title=f"Archive Final - {env_name} (CMA-ME)",
    )
    archive_plot_path = os.path.join(plots_dir, f"cmame_archive_{timestamp}.png")
    fig2.savefig(archive_plot_path)
    plt.close(fig2)

    repertoire.save(path=repertoire_dir)

    metrics_np = {
        "coverage": np.asarray(metrics["coverage"]),
        "max_fitness": np.asarray(metrics["max_fitness"]),
        "qd_score": np.asarray(metrics["qd_score"]),
        "env_steps": np.asarray(env_steps),
    }
    metrics_npz_path = os.path.join(repertoire_dir, "metrics.npz")
    np.savez(metrics_npz_path, **metrics_np)

    final_coverage = float(metrics["coverage"][-1])
    final_max_fitness = float(metrics["max_fitness"][-1])
    final_qd_score = float(metrics["qd_score"][-1])

    print(f"Final Coverage: {final_coverage}")
    print(f"Final Max Fitness: {final_max_fitness}")
    print(f"Final QD Score: {final_qd_score}")

    summary = {
        "timestamp": timestamp,
        "artifacts_dir": repertoire_dir,
        "config": {
            "env_name": env_name,
            "num_iterations": num_iterations,
            "emitter_name": emitter_name,
            "policy_hidden_layer_sizes": list(policy_hidden_layer_sizes),
            "batch_size": batch_size,
            "pool_size": pool_size,
            "num_init_cvt_samples": num_init_cvt_samples,
            "grid_shape": list(grid_shape),
            "num_centroids": num_centroids,
            "seed": seed,
            "episode_length": episode_length,
            "sigma_g": sigma_g,
        },
        "results": {
            "final_coverage": final_coverage,
            "final_max_fitness": final_max_fitness,
            "final_qd_score": final_qd_score,
        },
        "files": {
            "metrics_npz": metrics_npz_path,
            "summary_json": os.path.join(repertoire_dir, "summary.json"),
            "metrics_plot": metrics_plot_path,
            "archive_plot": archive_plot_path,
            "repertoire_dir": repertoire_dir,
        },
    }
    summary_json_path = os.path.join(repertoire_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Artifacts directory: {repertoire_dir}")
    print(f"Metrics plot: {metrics_plot_path}")
    print(f"Archive plot: {archive_plot_path}")
    print(f"Metrics npz: {metrics_npz_path}")
    print(f"Summary json: {summary_json_path}")
    print(
        "RESULT "
        f"emitter={emitter_name} "
        f"hidden_layers={','.join(map(str, policy_hidden_layer_sizes))} "
        f"batch_size={batch_size} "
        f"pool_size={pool_size} "
        f"repertoire_dir={repertoire_dir} "
        f"coverage={final_coverage} "
        f"max_fitness={final_max_fitness} "
        f"qd_score={final_qd_score}"
    )

    return repertoire


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="halfcheetah_oil")
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--pool-size", type=int, default=3)
    parser.add_argument("--num-init-cvt-samples", type=int, default=50000)
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[128, 128],
    )
    parser.add_argument(
        "--emitter",
        choices=sorted(EMITTER_NAMES),
        default="random",
    )
    args = parser.parse_args()

    run_cmame_oi_test(
        env_name=args.env_name,
        num_iterations=args.num_iterations,
        emitter_name=args.emitter,
        policy_hidden_layer_sizes=tuple(args.hidden_layers),
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        num_init_cvt_samples=args.num_init_cvt_samples,
    )
