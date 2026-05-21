#!/usr/bin/env python3
import functools
import json
import os
import sys
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

from qdax import environments
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.dcrl_emitter import DCRLConfig, DCRLTransition
from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.oil_credit_path_emitter import (
    OILAdvantageDCRLEmitter,
    OILTransportEmitter,
)
from qdax.core.map_elites import MAPElites
from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.environments import behavior_descriptor_extractor
from qdax.environments.oil_posthoc import compute_oil_descriptor_batch
from qdax.environments.wrappers import (
    ClipRewardWrapper,
    OffsetRewardWrapper,
)
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs
from qdax.utils.metrics import default_qd_metrics


def main() -> None:
    seed = int(os.environ.get("OIL_SEED") or "42")
    env_name = sys.argv[1] if len(sys.argv) > 1 else "halfcheetah_uni"
    oil_reward_alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    num_iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    mode = sys.argv[4] if len(sys.argv) > 4 else "oil_advantage"
    if mode != "oil_advantage":
        raise ValueError("This cleaned runner only supports oil_advantage.")

    episode_length = 100
    num_centroids = 1024
    ga_bs, dcrl_bs, ai_bs = 128, 64, 64
    total_bs = ga_bs + dcrl_bs + ai_bs
    iso_sigma = 0.005
    line_sigma = 0.05
    base_env_name = env_name.replace("_uni", "").replace("_omni", "")
    oil_observation_mode = (
        "angular_sincos" if base_env_name == "halfcheetah" else "default"
    )

    print("=" * 70)
    print("OIL-Advantage-DCRL-ME")
    print("=" * 70)
    print(f"Config: env={env_name}, bs={total_bs}, centroids={num_centroids}")
    print(f"  iters={num_iterations}, oil_alpha={oil_reward_alpha}")
    print(f"  OIL observation mode: {oil_observation_mode}")

    env = environments.create(env_name, episode_length=episode_length)
    env = OffsetRewardWrapper(env, offset=environments.reward_offset[env_name])
    env = ClipRewardWrapper(env, clip_min=0.0)
    reset_fn = jax.jit(env.reset)

    policy_net = MLP(
        layer_sizes=(128, 128, env.action_size),
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    actor_net = MLPDC(
        layer_sizes=(128, 128, env.action_size),
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, total_bs)
    init_params = jax.vmap(policy_net.init)(
        keys, jnp.zeros((total_bs, env.observation_size))
    )

    def play_step(env_state, params, random_key):
        actions = policy_net.apply(params, env_state.obs)
        next_state = env.step(env_state, actions)
        transition = DCRLTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions,
            state_desc=env_state.info["state_descriptor"],
            next_state_desc=next_state.info["state_descriptor"],
            desc=jnp.zeros(env.behavior_descriptor_length) * jnp.nan,
            desc_prime=jnp.zeros(env.behavior_descriptor_length) * jnp.nan,
        )
        return next_state, params, random_key, transition

    base_scoring = functools.partial(
        reset_based_scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step,
        behavior_descriptor_extractor=behavior_descriptor_extractor[env_name],
    )

    def scoring_fn(genotypes, random_key):
        fitnesses, descriptors, extra_scores, random_key = base_scoring(
            genotypes, random_key
        )
        extra_scores["oil_descriptors"] = compute_oil_descriptor_batch(
            extra_scores["transitions"].next_obs, base_env_name
        )
        return fitnesses, descriptors, extra_scores, random_key

    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=environments.reward_offset[env_name] * episode_length,
    )

    key, subkey = jax.random.split(key)
    min_bd, max_bd = env.behavior_descriptor_limits
    centroids, key = compute_cvt_centroids(
        env.behavior_descriptor_length,
        50_000,
        num_centroids,
        min_bd,
        max_bd,
        subkey,
    )

    dcrl_cfg = DCRLConfig(
        dcrl_batch_size=dcrl_bs,
        ai_batch_size=ai_bs,
        lengthscale=0.1,
        critic_hidden_layer_size=(256, 256),
        num_critic_training_steps=3000,
        num_pg_training_steps=150,
        batch_size=total_bs,
        replay_buffer_size=1_000_000,
        discount=0.99,
        reward_scaling=1.0,
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        policy_learning_rate=5e-3,
        noise_clip=0.5,
        policy_noise=0.2,
        soft_tau_update=0.005,
        policy_delay=2,
    )
    emitter = MultiEmitter(
        emitters=(
            OILAdvantageDCRLEmitter(
                dcrl_cfg, policy_net, actor_net, env, centroids, oil_reward_alpha
            ),
            OILTransportEmitter(
                ga_bs,
                num_centroids,
                centroids,
                iso_sigma=iso_sigma,
                line_sigma=line_sigma,
                grad_step=0.0,
            ),
        )
    )
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=emitter,
        metrics_function=metrics_fn,
    )

    print("\nInitializing...")
    repertoire, emitter_state, key = map_elites.init(init_params, centroids, key)
    print(f"  Init fitness={float(jnp.max(repertoire.fitnesses)):.2f}")

    @jax.jit
    def scan_fn(carry, _):
        rep, state, scan_key = carry
        rep, state, metrics, scan_key = map_elites.update(rep, state, scan_key)
        return (rep, state, scan_key), metrics

    print(f"\nTraining {num_iterations} iterations...")
    print("-" * 70)
    print(f"{'Iter':>6} | {'Fitness':>10} | {'QD':>12} | {'Cov':>6} | {'Time':>6}")
    print("-" * 70)

    all_metrics = {}
    start_time = time.time()
    completed = 0
    while completed < num_iterations:
        block = min(10, num_iterations - completed)
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
    output_dir = f"repertoires/oilcreditpath/{env_name}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    repertoire.save(output_dir + "/")

    metrics_np = {name: np.asarray(values) for name, values in all_metrics.items()}
    metric_len = len(next(iter(metrics_np.values())))
    metrics_np["env_steps"] = np.arange(1, metric_len + 1) * total_bs * episode_length
    np.savez(f"{output_dir}/metrics.npz", **metrics_np)

    oil_state = emitter_state.emitter_states[1]
    np.save(f"{output_dir}/oil_state_descriptors.npy", np.asarray(oil_state.oil_per_cell))
    np.save(f"{output_dir}/oil_state_filled.npy", np.asarray(oil_state.oil_filled))

    summary = {
        "timestamp": timestamp,
        "config": {
            "emitter": "OIL-Advantage-DCRL-ME",
            "env": env_name,
            "env_name": env_name,
            "iters": num_iterations,
            "bs": total_bs,
            "centroids": num_centroids,
            "seed": seed,
            "episode_length": episode_length,
            "oil_reward_alpha": oil_reward_alpha,
            "oil_observation_mode": oil_observation_mode,
            "oil_env_name": base_env_name,
            "archive_descriptor_type": "behavior",
            "oil_control": mode,
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
            "oil_state_descriptors_npy": f"{output_dir}/oil_state_descriptors.npy",
            "oil_state_filled_npy": f"{output_dir}/oil_state_filled.npy",
            "summary_json": f"{output_dir}/summary.json",
        },
    }
    with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
