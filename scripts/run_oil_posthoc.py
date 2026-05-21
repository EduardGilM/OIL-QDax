#!/usr/bin/env python3
import argparse
import functools
import json
import os
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.dcrl_me_emitter import DCRLMEConfig, DCRLMEEmitter
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition, QDTransition
from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.environments import behavior_descriptor_extractor
from qdax.environments.oil_posthoc import (
    OILArchiveEnvProxy,
    compute_oil_descriptor_batch,
)
from qdax.environments.wrappers import (
    ClipRewardWrapper,
    OffsetRewardWrapper,
)
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting_utils import (
    plot_2d_map_elites_repertoire,
    plot_oi_map_elites_results,
)


def _base_env_name(env_name: str) -> str:
    return env_name.replace("_uni", "").replace("_omni", "")


def _make_play_step(env, policy_network, use_dcrl_transition: bool):
    def play_step(env_state, policy_params, random_key):
        actions = policy_network.apply(policy_params, env_state.obs)
        next_state = env.step(env_state, actions)
        transition_kwargs = dict(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions,
            state_desc=env_state.info["state_descriptor"],
            next_state_desc=next_state.info["state_descriptor"],
        )
        if use_dcrl_transition:
            transition = DCRLTransition(
                **transition_kwargs,
                desc=jnp.zeros(2) * jnp.nan,
                desc_prime=jnp.zeros(2) * jnp.nan,
            )
        else:
            transition = QDTransition(**transition_kwargs)
        return next_state, policy_params, random_key, transition

    return play_step


def _make_emitter(name, env, oil_env, policy_network, actor_network, batch_size):
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=0.005, line_sigma=0.05
    )
    if name == "mapelites":
        return MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=batch_size,
        )
    if name == "pga":
        return PGAMEEmitter(
            config=PGAMEConfig(
                env_batch_size=batch_size,
                proportion_mutation_ga=0.5,
                num_critic_training_steps=3000,
                num_pg_training_steps=150,
                replay_buffer_size=1_000_000,
                critic_hidden_layer_size=(256, 256),
                critic_learning_rate=3e-4,
                greedy_learning_rate=3e-4,
                policy_learning_rate=5e-3,
                batch_size=batch_size,
            ),
            policy_network=policy_network,
            env=env,
            variation_fn=variation_fn,
        )
    if name == "dcrlme":
        return DCRLMEEmitter(
            config=DCRLMEConfig(
                ga_batch_size=128,
                dcrl_batch_size=64,
                ai_batch_size=64,
                lengthscale=0.1,
                critic_hidden_layer_size=(256, 256),
                num_critic_training_steps=3000,
                num_pg_training_steps=150,
                batch_size=batch_size,
                replay_buffer_size=1_000_000,
                critic_learning_rate=3e-4,
                actor_learning_rate=3e-4,
                policy_learning_rate=5e-3,
            ),
            policy_network=policy_network,
            actor_network=actor_network,
            env=oil_env,
            variation_fn=variation_fn,
        )
    raise ValueError(f"Unknown emitter: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", nargs="?", default="halfcheetah_uni")
    parser.add_argument("emitter", nargs="?", choices=("mapelites", "pga", "dcrlme"), default="dcrlme")
    parser.add_argument("num_iterations", nargs="?", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("OIL_SEED", "42")))
    args = parser.parse_args()

    episode_length = 100
    batch_size = 256
    num_centroids = 1024
    base_env_name = _base_env_name(args.env_name)

    print("=" * 70)
    print(f"OIL-Posthoc-{args.emitter}")
    print("=" * 70)
    print(f"Config: env={args.env_name}, bs={batch_size}, centroids={num_centroids}")
    print(f"  iters={args.num_iterations}, seed={args.seed}")
    print("  archive descriptor: oil posthoc")

    env = environments.create(args.env_name, episode_length=episode_length)
    env = OffsetRewardWrapper(env, offset=environments.reward_offset[args.env_name])
    env = ClipRewardWrapper(env, clip_min=0.0)
    oil_env = OILArchiveEnvProxy(env)
    reset_fn = jax.jit(env.reset)

    policy_network = MLP(
        layer_sizes=(128, 128, env.action_size),
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    actor_network = MLPDC(
        layer_sizes=(128, 128, env.action_size),
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    init_params = jax.vmap(policy_network.init)(
        jax.random.split(subkey, batch_size),
        jnp.zeros((batch_size, env.observation_size)),
    )

    base_scoring = functools.partial(
        reset_based_scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=_make_play_step(
            env, policy_network, use_dcrl_transition=args.emitter == "dcrlme"
        ),
        behavior_descriptor_extractor=behavior_descriptor_extractor[args.env_name],
    )

    def scoring_fn(genotypes, random_key):
        fitnesses, behavior_descriptors, extra_scores, random_key = base_scoring(
            genotypes, random_key
        )
        oil_descriptors = compute_oil_descriptor_batch(
            extra_scores["transitions"].next_obs, base_env_name
        )
        extra_scores["behavior_descriptors"] = behavior_descriptors
        extra_scores["oil_descriptors"] = oil_descriptors
        return fitnesses, oil_descriptors, extra_scores, random_key

    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=environments.reward_offset[args.env_name] * episode_length,
    )

    key, subkey = jax.random.split(key)
    min_bd, max_bd = oil_env.behavior_descriptor_limits
    centroids, key = compute_cvt_centroids(
        2, 50_000, num_centroids, min_bd, max_bd, subkey
    )

    emitter = _make_emitter(
        args.emitter, env, oil_env, policy_network, actor_network, batch_size
    )
    map_elites = MAPElites(scoring_fn, emitter, metrics_fn)

    print("\nInitializing...")
    repertoire, emitter_state, key = map_elites.init(init_params, centroids, key)
    print(f"  Init fitness={float(jnp.max(repertoire.fitnesses)):.2f}")

    @jax.jit
    def scan_fn(carry, _):
        rep, state, scan_key = carry
        rep, state, metrics, scan_key = map_elites.update(rep, state, scan_key)
        return (rep, state, scan_key), metrics

    print(f"\nTraining {args.num_iterations} iterations...")
    print("-" * 70)
    print(f"{'Iter':>6} | {'Fitness':>10} | {'QD':>12} | {'Cov':>6} | {'Time':>6}")
    print("-" * 70)
    all_metrics = {}
    start_time = time.time()
    completed = 0
    while completed < args.num_iterations:
        block = min(10, args.num_iterations - completed)
        step_start = time.time()
        (repertoire, emitter_state, key), metrics = jax.lax.scan(
            scan_fn, (repertoire, emitter_state, key), (), length=block
        )
        completed += block
        print(
            f"{completed:6d} | {float(jnp.max(repertoire.fitnesses)):10.2f} | "
            f"{float(metrics['qd_score'][-1]):12.0f} | "
            f"{float(metrics['coverage'][-1]):5.1f}% | {time.time() - step_start:5.1f}s"
        )
        for name, values in metrics.items():
            all_metrics[name] = (
                jnp.concatenate([all_metrics[name], values])
                if name in all_metrics
                else values
            )

    elapsed = time.time() - start_time
    fitness = float(jnp.max(repertoire.fitnesses))
    qd_score = float(all_metrics["qd_score"][-1])
    coverage = float(all_metrics["coverage"][-1])
    print("-" * 70)
    print(f"\nFitness: {fitness:.2f}")
    print(f"QD:      {qd_score:.0f}")
    print(f"Cov:     {coverage:.1f}%")
    print(f"Time:    {elapsed:.0f}s ({elapsed / 60:.1f} min)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"repertoires/oil_posthoc/{args.env_name}/{args.emitter}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    repertoire.save(output_dir + "/")

    metrics_np = {name: np.asarray(values) for name, values in all_metrics.items()}
    metric_len = len(next(iter(metrics_np.values())))
    metrics_np["env_steps"] = np.arange(1, metric_len + 1) * batch_size * episode_length
    np.savez(f"{output_dir}/metrics.npz", **metrics_np)
    _, behavior_descriptors, _, key = base_scoring(repertoire.genotypes, key)
    np.save(f"{output_dir}/behavior_descriptors.npy", np.asarray(behavior_descriptors))

    figures_dir = f"{output_dir}/figures"
    os.makedirs(figures_dir, exist_ok=True)
    fig1, _ = plot_oi_map_elites_results(
        env_steps=jnp.asarray(metrics_np["env_steps"]),
        metrics={name: jnp.asarray(values) for name, values in all_metrics.items()},
        repertoire=repertoire,
        min_bd=min_bd,
        max_bd=max_bd,
    )
    metrics_png = f"{figures_dir}/oil_posthoc_{args.emitter}_metrics.png"
    fig1.savefig(metrics_png, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(10, 10))
    plot_2d_map_elites_repertoire(
        repertoire=repertoire,
        ax=ax,
        min_bd=min_bd,
        max_bd=max_bd,
        title=f"Archive Final - {args.env_name} ({args.emitter} OIL posthoc)",
    )
    archive_png = f"{figures_dir}/oil_posthoc_{args.emitter}_archive.png"
    fig2.savefig(archive_png, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "timestamp": timestamp,
        "config": {
            "emitter": args.emitter,
            "env_name": args.env_name,
            "iters": args.num_iterations,
            "bs": batch_size,
            "centroids": num_centroids,
            "seed": args.seed,
            "episode_length": episode_length,
            "archive_descriptor_type": "oil_posthoc",
            "oil_env_name": base_env_name,
        },
        "results": {
            "fitness": fitness,
            "qd": qd_score,
            "coverage": coverage,
            "time": elapsed,
        },
        "files": {
            "repertoire_dir": output_dir,
            "metrics_npz": f"{output_dir}/metrics.npz",
            "behavior_descriptors_npy": f"{output_dir}/behavior_descriptors.npy",
            "metrics_png": metrics_png,
            "archive_png": archive_png,
            "summary_json": f"{output_dir}/summary.json",
        },
    }
    with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
