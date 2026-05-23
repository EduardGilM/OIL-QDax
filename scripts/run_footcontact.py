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
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition, QDTransition
from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.environments import behavior_descriptor_extractor
from qdax.environments.wrappers import ClipRewardWrapper, OffsetRewardWrapper
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting import (
    plot_2d_map_elites_repertoire,
    plot_map_elites_results,
    plot_multidimensional_map_elites_grid,
)

from scripts.run_oil_posthoc import _make_emitter


def _plot_1d_archive(repertoire, ax, min_bd, max_bd, title):
    fitnesses = np.asarray(repertoire.fitnesses)
    centroids = np.asarray(repertoire.centroids).reshape(-1)
    descriptors = np.asarray(repertoire.descriptors).reshape(-1)
    valid = np.isfinite(fitnesses)
    order = np.argsort(centroids)
    sorted_centroids = centroids[order]
    min_x = float(np.asarray(min_bd).reshape(-1)[0])
    max_x = float(np.asarray(max_bd).reshape(-1)[0])
    bounds = np.empty(sorted_centroids.size + 1)
    bounds[0], bounds[-1] = min_x, max_x
    bounds[1:-1] = 0.5 * (sorted_centroids[:-1] + sorted_centroids[1:])
    vmin = float(np.min(fitnesses[valid])) if np.any(valid) else 0.0
    vmax = float(np.max(fitnesses[valid])) if np.any(valid) else 1.0
    cmap = cm.viridis
    norm = Normalize(vmin=vmin, vmax=vmax)

    for rank, idx in enumerate(order):
        color = cmap(norm(fitnesses[idx])) if valid[idx] else "white"
        alpha = 0.8 if valid[idx] else 0.05
        ax.fill(
            [bounds[rank], bounds[rank + 1], bounds[rank + 1], bounds[rank]],
            [0.0, 0.0, 1.0, 1.0],
            alpha=alpha,
            edgecolor="black",
            facecolor=color,
            lw=1,
        )

    if np.any(valid):
        ax.scatter(
            descriptors[valid],
            np.full(np.sum(valid), 0.5),
            c=fitnesses[valid],
            cmap=cmap,
            s=10,
            zorder=5,
            norm=norm,
        )

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Behavior Descriptor 1")
    ax.set_ylabel("Behavior Descriptor 2")
    ax.set_title(title)
    ax.set_aspect("equal")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)


def _plot_archive(repertoire, ax, min_bd, max_bd, title):
    if repertoire.centroids.shape[1] == 1:
        _plot_1d_archive(repertoire, ax, min_bd, max_bd, title)
        return
    if repertoire.centroids.shape[1] > 2:
        grid_shape = (
            (8, 8, 4, 4)
            if repertoire.centroids.shape[1] == 4
            else tuple([4] * repertoire.centroids.shape[1])
        )
        plot_multidimensional_map_elites_grid(
            repertoire,
            min_bd,
            max_bd,
            grid_shape=grid_shape,
            ax=ax,
        )
        ax.set_title(title)
        return
    plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=min_bd,
        maxval=max_bd,
        repertoire_descriptors=repertoire.descriptors,
        ax=ax,
    )
    ax.set_title(title)


def _plot_results(env_steps, metrics, repertoire, min_bd, max_bd):
    if repertoire.centroids.shape[1] == 2:
        return plot_map_elites_results(env_steps, metrics, repertoire, min_bd, max_bd)

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))
    axes[0].plot(env_steps, metrics["coverage"])
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Coverage in %")
    axes[0].set_title("Coverage evolution during training")
    axes[1].plot(env_steps, metrics["max_fitness"])
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Maximum fitness")
    axes[1].set_title("Maximum fitness evolution during training")
    axes[2].plot(env_steps, metrics["qd_score"])
    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("QD Score")
    axes[2].set_title("QD Score evolution during training")
    _plot_archive(repertoire, axes[3], min_bd, max_bd, "MAP-Elites Grid")
    return fig, axes


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
                desc=jnp.zeros(env.behavior_descriptor_length) * jnp.nan,
                desc_prime=jnp.zeros(env.behavior_descriptor_length) * jnp.nan,
            )
        else:
            transition = QDTransition(**transition_kwargs)
        return next_state, policy_params, random_key, transition

    return play_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", nargs="?", default="halfcheetah_uni")
    parser.add_argument(
        "emitter",
        nargs="?",
        choices=("mapelites", "pga", "dcrlme", "dcrl"),
        default="dcrlme",
    )
    parser.add_argument("num_iterations", nargs="?", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=int(os.environ.get("OIL_SEED", "42")))
    args = parser.parse_args()
    emitter_name = "dcrlme" if args.emitter == "dcrl" else args.emitter
    if not args.env_name.endswith("_uni"):
        raise ValueError("Foot-contact baselines use *_uni environments.")

    episode_length = 100
    batch_size = 256
    num_centroids = 1024

    print("=" * 70)
    print(f"FootContact-{emitter_name}")
    print("=" * 70)
    print(f"Config: env={args.env_name}, bs={batch_size}, centroids={num_centroids}")
    print(f"  iters={args.num_iterations}, seed={args.seed}")
    print("  archive descriptor: environment behavior descriptor")

    env = environments.create(args.env_name, episode_length=episode_length)
    env = OffsetRewardWrapper(env, offset=environments.reward_offset[args.env_name])
    env = ClipRewardWrapper(env, clip_min=0.0)
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

    scoring_fn = functools.partial(
        reset_based_scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=_make_play_step(
            env, policy_network, use_dcrl_transition=emitter_name == "dcrlme"
        ),
        behavior_descriptor_extractor=behavior_descriptor_extractor[args.env_name],
    )

    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=environments.reward_offset[args.env_name] * episode_length,
    )

    key, subkey = jax.random.split(key)
    min_bd, max_bd = env.behavior_descriptor_limits
    centroids, key = compute_cvt_centroids(
        env.behavior_descriptor_length, 50_000, num_centroids, min_bd, max_bd, subkey
    )

    emitter = _make_emitter(
        emitter_name, env, env, policy_network, actor_network, batch_size
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
    output_dir = f"repertoires/foot_contact/{args.env_name}/{emitter_name}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    repertoire.save(output_dir + "/")

    metrics_np = {name: np.asarray(values) for name, values in all_metrics.items()}
    metric_len = len(next(iter(metrics_np.values())))
    metrics_np["env_steps"] = np.arange(1, metric_len + 1) * batch_size * episode_length
    np.savez(f"{output_dir}/metrics.npz", **metrics_np)
    np.save(f"{output_dir}/behavior_descriptors.npy", np.asarray(repertoire.descriptors))

    figures_dir = f"{output_dir}/figures"
    os.makedirs(figures_dir, exist_ok=True)
    metrics_jnp = {name: jnp.asarray(values) for name, values in all_metrics.items()}
    fig1, _ = _plot_results(
        jnp.asarray(metrics_np["env_steps"]), metrics_jnp, repertoire, min_bd, max_bd
    )
    metrics_png = f"{figures_dir}/foot_contact_{emitter_name}_metrics.png"
    fig1.savefig(metrics_png, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(10, 10))
    _plot_archive(
        repertoire=repertoire,
        ax=ax,
        min_bd=min_bd,
        max_bd=max_bd,
        title=f"Archive Final - {args.env_name} ({emitter_name} foot-contact)",
    )
    archive_png = f"{figures_dir}/foot_contact_{emitter_name}_archive.png"
    fig2.savefig(archive_png, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "timestamp": timestamp,
        "config": {
            "emitter": emitter_name,
            "env_name": args.env_name,
            "iters": args.num_iterations,
            "bs": batch_size,
            "centroids": num_centroids,
            "seed": args.seed,
            "episode_length": episode_length,
            "archive_descriptor_type": "foot_contact",
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
