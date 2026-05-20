from functools import partial

import flax.struct
import jax
import jax.numpy as jnp

from qdax.core.emitters.dcrl_emitter import DCRLEmitter, DCRLEmitterState
from qdax.core.emitters.emitter import Emitter
from qdax.custom_types import Params, RNGKey


class OILPathState(flax.struct.PyTreeNode):
    oil_per_cell: jnp.ndarray
    oil_filled: jnp.ndarray
    attempts: jnp.ndarray
    successes: jnp.ndarray
    oil_path: Params
    oil_grad: Params
    path_strength: jnp.ndarray
    key: RNGKey


class OILCreditDCRLEmitter(DCRLEmitter):
    """DCRL trained with online OIL novelty credit."""

    def __init__(self, config, policy_network, actor_network, env, centroids, oil_alpha):
        super().__init__(config, policy_network, actor_network, env)
        self._centroids = centroids
        self._oil_alpha = oil_alpha
        centroid_dists = jnp.linalg.norm(
            centroids[:, None, :] - centroids[None, :, :], axis=-1
        )
        neighbors = jnp.argsort(centroid_dists, axis=1)[:, 1:13]
        scale = jnp.mean(jnp.take_along_axis(centroid_dists, neighbors[:, -1:], axis=1))
        self._frontier_kernel = jnp.exp(-centroid_dists / (scale + 1e-8))

    def _sample_frontier_targets(self, repertoire, key, batch_size):
        filled = jnp.where(repertoire.fitnesses > -jnp.inf, 1.0, 0.0)
        valid = filled > 0.0
        fit_raw = jnp.where(valid, repertoire.fitnesses, -1e10)
        f_lo = jnp.min(fit_raw + 1e10 * (1.0 - filled))
        f_hi = jnp.max(fit_raw)
        fit_n = jnp.where(valid, (fit_raw - f_lo) / (f_hi - f_lo + 1e-8), 0.0)
        frontier_score = valid * (0.2 + 0.8 * fit_n)
        target_weights = (1.0 - filled) * (
            0.25 + jnp.max(self._frontier_kernel * frontier_score[None, :], axis=1)
        )
        fallback = filled / (jnp.sum(filled) + 1e-8)
        target_weights = jnp.where(
            jnp.sum(target_weights) > 0.0,
            target_weights / (jnp.sum(target_weights) + 1e-8),
            fallback,
        )
        target_idx = jax.random.choice(
            key, self._centroids.shape[0], (batch_size,), p=target_weights
        )
        target_desc = self._centroids[target_idx]
        desc_dists = jnp.linalg.norm(
            repertoire.descriptors[None, :, :] - target_desc[:, None, :], axis=-1
        )
        parent_idx = jnp.argmin(jnp.where(valid[None, :], desc_dists, jnp.inf), axis=1)
        parents = jax.tree_util.tree_map(lambda x: x[parent_idx], repertoire.genotypes)
        return parents, target_desc

    def state_update(
        self, emitter_state, repertoire, genotypes, fitnesses, descriptors, extra_scores
    ):
        oil = extra_scores["oil_descriptors"]
        diff = oil[:, None, :] - oil[None, :, :]
        dists = jnp.where(
            jnp.eye(oil.shape[0]) > 0, jnp.inf, jnp.linalg.norm(diff, axis=-1)
        )
        novelty = jnp.mean(
            jnp.sort(dists, axis=1)[:, : min(10, oil.shape[0] - 1)], axis=1
        )
        novelty = novelty / (jnp.max(novelty) + 1e-8)
        idx = jnp.argmin(
            jnp.linalg.norm(
                descriptors[:, None, :] - self._centroids[None, :, :], axis=-1
            ),
            axis=1,
        )
        inserted = fitnesses >= repertoire.fitnesses[idx] - 1e-6
        coverage = jnp.mean(jnp.isfinite(repertoire.fitnesses).astype(jnp.float32))
        global_gate = jnp.clip((coverage - 0.92) / 0.05, 0.0, 1.0)
        productive_alpha = self._oil_alpha * (1.0 - global_gate) * jnp.square(
            jnp.clip((0.92 - coverage) / 0.20, 0.0, 1.0)
        )
        global_alpha = self._oil_alpha * global_gate
        credit = 0.15 + 0.85 * inserted.astype(jnp.float32)
        transitions = extra_scores["transitions"]
        transitions = transitions.replace(
            rewards=transitions.rewards
            + (
                (global_alpha + productive_alpha * credit)
                * novelty
                / transitions.obs.shape[1]
            )[:, None]
        )
        return super().state_update(
            emitter_state,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            {**extra_scores, "transitions": transitions},
        )

    @partial(jax.jit, static_argnames=("self",))
    def emit(self, repertoire, emitter_state, key):
        key, k_pg, k_ai = jax.random.split(key, 3)
        parents_pg, descs_pg = self._sample_frontier_targets(
            repertoire, k_pg, self._config.dcrl_batch_size
        )
        _, descs_ai = self._sample_frontier_targets(
            repertoire, k_ai, self._config.ai_batch_size
        )
        genotypes_pg = self.emit_pg(emitter_state, parents_pg, descs_pg)
        genotypes_ai = self.emit_ai(emitter_state, descs_ai)
        genotypes = jax.tree_util.tree_map(
            lambda x1, x2: jnp.concatenate((x1, x2), axis=0), genotypes_pg, genotypes_ai
        )
        return genotypes, {"desc_prime": jnp.concatenate([descs_pg, descs_ai], axis=0)}, key


class OILAdvantageDCRLState(DCRLEmitterState):
    oil_per_cell: jnp.ndarray
    oil_filled: jnp.ndarray
    oil_advantage: jnp.ndarray


class OILAdvantageDCRLEmitter(OILCreditDCRLEmitter):
    """Descriptor DCRL guided by an online OIL advantage signal."""

    def init(self, key, repertoire, genotypes, fitnesses, descriptors, extra_scores):
        base_state, key = DCRLEmitter.init(
            self, key, repertoire, genotypes, fitnesses, descriptors, extra_scores
        )
        state = OILAdvantageDCRLState(
            critic_params=base_state.critic_params,
            critic_opt_state=base_state.critic_opt_state,
            actor_params=base_state.actor_params,
            actor_opt_state=base_state.actor_opt_state,
            target_critic_params=base_state.target_critic_params,
            target_actor_params=base_state.target_actor_params,
            replay_buffer=base_state.replay_buffer,
            key=base_state.key,
            steps=base_state.steps,
            oil_per_cell=jnp.zeros((self._centroids.shape[0], 2)),
            oil_filled=jnp.zeros(self._centroids.shape[0]),
            oil_advantage=jnp.zeros(self._centroids.shape[0]),
        )
        return self._update_oil_state(
            state, repertoire, fitnesses, descriptors, extra_scores["oil_descriptors"]
        ), key

    def _update_oil_state(self, state, repertoire, fitnesses, descriptors, oil, advantage=None):
        idx = jnp.argmin(
            jnp.linalg.norm(descriptors[:, None, :] - self._centroids[None, :, :], axis=-1),
            axis=1,
        )
        inserted = fitnesses >= repertoire.fitnesses[idx] - 1e-6
        safe_idx = jnp.where(inserted, idx, self._centroids.shape[0])
        filled = jax.ops.segment_max(
            inserted.astype(jnp.float32), idx, num_segments=self._centroids.shape[0]
        )
        if advantage is None:
            advantage = inserted.astype(jnp.float32)
        count = jax.ops.segment_sum(
            inserted.astype(jnp.float32), idx, num_segments=self._centroids.shape[0]
        )
        adv = jax.ops.segment_sum(
            advantage * inserted, idx, num_segments=self._centroids.shape[0]
        )
        adv = adv / (count + 1e-8)
        return state.replace(
            oil_per_cell=state.oil_per_cell.at[safe_idx].set(oil),
            oil_filled=jnp.maximum(state.oil_filled, filled),
            oil_advantage=jnp.where(
                count > 0,
                0.9 * state.oil_advantage + 0.1 * adv,
                0.995 * state.oil_advantage,
            ),
        )

    def _sample_advantage_targets(self, repertoire, state, key, batch_size):
        filled = jnp.where(repertoire.fitnesses > -jnp.inf, 1.0, 0.0)
        valid = filled > 0.0
        coverage = jnp.mean(filled)
        fit = jnp.where(valid, repertoire.fitnesses, -1e10)
        f_lo = jnp.min(fit + 1e10 * (1.0 - filled))
        f_hi = jnp.max(fit)
        fit_n = jnp.where(valid, (fit - f_lo) / (f_hi - f_lo + 1e-8), 0.0)
        adv_n = state.oil_advantage / (jnp.max(state.oil_advantage) + 1e-8)
        frontier = jnp.max(self._frontier_kernel * (1.0 - filled)[None, :], axis=1)

        source = valid * (0.2 + 0.8 * fit_n) * (0.3 + adv_n)
        empty_w = (1.0 - filled) * (
            0.2 + jnp.max(self._frontier_kernel * source[None, :], axis=1)
        )
        empty_w = empty_w / (jnp.sum(empty_w) + 1e-8)
        filled_w = valid * jnp.exp(2.0 * fit_n + 2.5 * adv_n + 0.5 * frontier)
        filled_w = filled_w / (jnp.sum(filled_w) + 1e-8)
        explore = jnp.clip((0.995 - coverage) / 0.20, 0.10, 0.70)
        target_w = explore * empty_w + (1.0 - explore) * filled_w
        target_w = target_w / (jnp.sum(target_w) + 1e-8)

        target_idx = jax.random.choice(
            key, self._centroids.shape[0], (batch_size,), p=target_w
        )
        target_desc = self._centroids[target_idx]
        desc_dists = jnp.linalg.norm(
            repertoire.descriptors[None, :, :] - target_desc[:, None, :], axis=-1
        )
        parent_idx = jnp.argmin(jnp.where(valid[None, :], desc_dists, jnp.inf), axis=1)
        parents = jax.tree_util.tree_map(lambda x: x[parent_idx], repertoire.genotypes)
        return parents, target_desc

    @partial(jax.jit, static_argnames=("self",))
    def emit(self, repertoire, emitter_state, key):
        key, k_pg, k_ai = jax.random.split(key, 3)
        parents_pg, descs_pg = self._sample_advantage_targets(
            repertoire, emitter_state, k_pg, self._config.dcrl_batch_size
        )
        _, descs_ai = self._sample_advantage_targets(
            repertoire, emitter_state, k_ai, self._config.ai_batch_size
        )
        genotypes_pg = self.emit_pg(emitter_state, parents_pg, descs_pg)
        genotypes_ai = self.emit_ai(emitter_state, descs_ai)
        genotypes = jax.tree_util.tree_map(
            lambda x1, x2: jnp.concatenate((x1, x2), axis=0), genotypes_pg, genotypes_ai
        )
        return genotypes, {
            "desc_prime": jnp.concatenate([descs_pg, descs_ai], axis=0),
            "adv_prev_fitnesses": repertoire.fitnesses,
        }, key

    def state_update(self, emitter_state, repertoire, genotypes, fitnesses, descriptors, extra_scores):
        oil = extra_scores["oil_descriptors"]
        idx = jnp.argmin(
            jnp.linalg.norm(descriptors[:, None, :] - self._centroids[None, :, :], axis=-1),
            axis=1,
        )
        prev = extra_scores["adv_prev_fitnesses"][idx]
        gain = jnp.where(jnp.isfinite(prev), jnp.maximum(fitnesses - prev, 0.0), fitnesses)
        gain = gain / (jnp.max(gain) + 1e-8)
        d = jnp.linalg.norm(oil[:, None, :] - emitter_state.oil_per_cell[None, :, :], axis=-1)
        d = jnp.where(emitter_state.oil_filled[None, :] > 0, d, jnp.inf)
        novelty = jnp.mean(jnp.sort(d, axis=1)[:, :10], axis=1)
        novelty = jnp.where(jnp.isfinite(novelty), novelty / (jnp.max(novelty) + 1e-8), 0.0)
        advantage = 0.7 * gain + 0.3 * novelty
        emitter_state = self._update_oil_state(
            emitter_state, repertoire, fitnesses, descriptors, oil, advantage
        )
        transitions = extra_scores["transitions"]
        transitions = transitions.replace(
            rewards=transitions.rewards
            + (0.5 * self._oil_alpha * advantage / transitions.obs.shape[1])[:, None]
        )
        return DCRLEmitter.state_update(
            self,
            emitter_state,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            {**extra_scores, "transitions": transitions},
        )


class OILTransportEmitter(Emitter):
    """GA emitter that learns productive genotype directions from OIL deltas."""

    def __init__(
        self,
        batch_size,
        num_centroids,
        centroids,
        iso_sigma=0.005,
        line_sigma=0.05,
        novelty_weight=0.8,
        coverage_weight=0.4,
        oil_partner_temp=2.0,
        path_step=0.4,
        grad_step=0.0,
        path_decay=0.98,
        path_lr=0.12,
        oil_dim=2,
    ):
        self._batch_size = batch_size
        self._num_centroids = num_centroids
        self._centroids = centroids
        self._iso_sigma = iso_sigma
        self._line_sigma = line_sigma
        self._novelty_weight = novelty_weight
        self._coverage_weight = coverage_weight
        self._oil_partner_temp = oil_partner_temp
        self._path_step = path_step
        self._grad_step = grad_step
        self._path_decay = path_decay
        self._path_lr = path_lr
        self._oil_dim = oil_dim
        centroid_dists = jnp.linalg.norm(
            centroids[:, None, :] - centroids[None, :, :], axis=-1
        )
        self._neighbors = jnp.argsort(centroid_dists, axis=1)[:, 1:13]
        scale = jnp.mean(jnp.take_along_axis(centroid_dists, self._neighbors[:, -1:], axis=1))
        self._target_kernel = jnp.exp(-centroid_dists / (scale + 1e-8))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def use_all_data(self):
        return True

    def init(self, random_key, repertoire, genotypes, fitnesses, descriptors, extra_scores):
        random_key, subkey = jax.random.split(random_key)
        state = OILPathState(
            oil_per_cell=jnp.zeros((self._num_centroids, self._oil_dim)),
            oil_filled=jnp.zeros(self._num_centroids),
            attempts=jnp.zeros(self._num_centroids),
            successes=jnp.zeros(self._num_centroids),
            oil_path=jax.tree_util.tree_map(jnp.zeros_like, repertoire.genotypes),
            oil_grad=jax.tree_util.tree_map(
                lambda x: jnp.zeros((x.shape[0], self._oil_dim) + x.shape[1:], dtype=x.dtype),
                repertoire.genotypes,
            ),
            path_strength=jnp.zeros(self._num_centroids),
            key=subkey,
        )
        return (
            self._update_oil(state, genotypes, fitnesses, descriptors, extra_scores, repertoire),
            random_key,
        )

    @partial(jax.jit, static_argnames=("self",))
    def emit(self, repertoire, emitter_state, random_key):
        random_key, k1, k2, k3, k4 = jax.random.split(random_key, 5)
        filled = jnp.where(repertoire.fitnesses > -jnp.inf, 1.0, 0.0)
        valid = filled > 0.0
        coverage = jnp.mean(filled)

        oil = emitter_state.oil_per_cell
        oil_mask = emitter_state.oil_filled * filled
        dists = jnp.linalg.norm(oil[:, None, :] - oil[None, :, :], axis=-1)
        dists = jnp.where(oil_mask[None, :] > 0, dists, jnp.inf)
        novelty = jnp.mean(jnp.sort(dists, axis=1)[:, 1:11], axis=1)
        novelty = jnp.where(jnp.isfinite(novelty), novelty, 0.0)

        fit_raw = jnp.where(filled > 0, repertoire.fitnesses, -1e10)
        f_lo = jnp.min(fit_raw + 1e10 * (1 - filled))
        f_hi = jnp.max(fit_raw)
        fit_n = jnp.where(filled > 0, (fit_raw - f_lo) / (f_hi - f_lo + 1e-8), 0.0)
        nov_n = novelty / (jnp.max(novelty) + 1e-8)
        success_rate = emitter_state.successes / (emitter_state.attempts + 1.0)
        success_n = success_rate / (jnp.max(success_rate) + 1e-8)
        cover_n = 1.0 / jnp.sqrt(repertoire.modification_counts.astype(jnp.float32) + 1.0)
        cover_n = cover_n / (jnp.max(cover_n) + 1e-8)
        path_n = emitter_state.path_strength / (jnp.max(emitter_state.path_strength) + 1e-8)
        empty_neigh = jnp.mean(1.0 - filled[self._neighbors], axis=1)
        frontier_n = empty_neigh / (jnp.max(empty_neigh) + 1e-8)
        frontier_score = (0.2 + 0.8 * fit_n) * (0.2 + frontier_n) * (
            0.3 + path_n + success_n
        )
        quality_score = (0.2 + 0.8 * fit_n) * (0.3 + success_n) * (
            1.2 - 0.6 * frontier_n
        )
        explore_ratio = jnp.clip(0.20 + 0.45 * (1.0 - coverage), 0.30, 0.65)

        combined = (
            explore_ratio * (1.4 * frontier_score + self._coverage_weight * cover_n)
            + (1.0 - explore_ratio) * quality_score
            + self._novelty_weight * nov_n
            + 0.5 * path_n
        )
        weights = jnp.where(valid, jnp.exp(combined * 2.0), 0.0)
        weights = weights / (jnp.sum(weights) + 1e-8)

        base_p1_idx = jax.random.choice(
            k1, self._num_centroids, (self._batch_size,), p=weights
        )
        target_weights = (1.0 - filled) * (
            0.50 + jnp.max(self._target_kernel * (valid * frontier_score)[None, :], axis=1)
        )
        target_weights = jnp.where(
            jnp.sum(target_weights) > 0.0,
            target_weights / (jnp.sum(target_weights) + 1e-8),
            weights,
        )
        target_idx = jax.random.choice(
            k3, self._num_centroids, (self._batch_size,), p=target_weights
        )
        target_desc = self._centroids[target_idx]
        desc_dists = jnp.linalg.norm(
            repertoire.descriptors[None, :, :] - target_desc[:, None, :], axis=-1
        )
        desc_dists = jnp.where(valid[None, :], desc_dists, jnp.inf)
        frontier_p1_idx = jnp.argmin(desc_dists, axis=1)
        use_frontier = jax.random.bernoulli(k4, explore_ratio, (self._batch_size,))
        p1_idx = jnp.where(use_frontier, frontier_p1_idx, base_p1_idx)
        p1 = jax.tree_util.tree_map(lambda x: x[p1_idx], repertoire.genotypes)
        p_path = jax.tree_util.tree_map(lambda x: x[p1_idx], emitter_state.oil_path)
        p_grad = jax.tree_util.tree_map(lambda x: x[p1_idx], emitter_state.oil_grad)

        oil_dists = jnp.linalg.norm(oil[p1_idx, None, :] - oil[None, :, :], axis=-1)
        oil_dists = oil_dists / (jnp.max(oil_dists, axis=1, keepdims=True) + 1e-8)
        partner_logits = jnp.log(weights + 1e-8)[None, :] + self._oil_partner_temp * oil_dists
        partner_probs = jax.nn.softmax(
            jnp.where(valid[None, :], partner_logits, -jnp.inf), axis=1
        )
        partner_keys = jax.random.split(k2, self._batch_size)
        p2_idx = jax.vmap(
            lambda key, probs: jax.random.choice(key, self._num_centroids, p=probs)
        )(partner_keys, partner_probs)
        p2 = jax.tree_util.tree_map(lambda x: x[p2_idx], repertoire.genotypes)
        desired_oil = jnp.clip(oil[p2_idx] - oil[p1_idx], -0.35, 0.35)

        novelty_parent = nov_n[p1_idx]
        success_parent = success_n[p1_idx]
        frontier_parent = frontier_n[p1_idx]
        iso = jnp.clip(
            self._iso_sigma
            * (0.7 + 0.8 * frontier_parent + novelty_parent - 0.3 * success_parent),
            0.002,
            0.018,
        )
        line = self._line_sigma * (0.7 + 0.8 * frontier_parent + 0.6 * novelty_parent)

        leaves_1, tree_def = jax.tree_util.tree_flatten(p1)
        leaves_2 = jax.tree_util.tree_leaves(p2)
        leaves_path = jax.tree_util.tree_leaves(p_path)
        random_key, subkey = jax.random.split(random_key)
        line_noise = line * jax.random.normal(subkey, (self._batch_size,))
        random_key, subkey = jax.random.split(random_key)
        noise_keys = jax.random.split(subkey, len(leaves_1))
        flow_step = self._path_step * jnp.clip(0.2 + frontier_parent, 0.0, 1.0) * path_n[p1_idx]
        grad_step = self._grad_step * (0.25 + frontier_parent)

        def path_leaf(l1, l2, lp, lg, nk):
            shape = (self._batch_size,) + (1,) * (l1.ndim - 1)
            grad_shape = (self._batch_size, self._oil_dim) + (1,) * (l1.ndim - 1)
            oil_delta = jnp.sum(desired_oil.reshape(grad_shape) * lg, axis=1)
            return (
                l1
                + iso.reshape(shape) * jax.random.normal(nk, l1.shape)
                + line_noise.reshape(shape) * (l2 - l1)
                + flow_step.reshape(shape) * lp
                + grad_step.reshape(shape) * oil_delta
            )

        leaves = [
            path_leaf(l1, l2, lp, lg, nk)
            for l1, l2, lp, lg, nk in zip(
                leaves_1,
                leaves_2,
                leaves_path,
                jax.tree_util.tree_leaves(p_grad),
                noise_keys,
            )
        ]
        return (
            jax.tree_util.tree_unflatten(tree_def, leaves),
            {
                "oil_parent_idx": p1_idx,
                "oil_parent_fitness": repertoire.fitnesses[p1_idx],
                "oil_parent_genotypes": p1,
                "oil_parent_oil": oil[p1_idx],
                "oil_prev_fitnesses": repertoire.fitnesses,
            },
            random_key,
        )

    @partial(jax.jit, static_argnames=("self",))
    def state_update(self, emitter_state, repertoire, genotypes, fitnesses, descriptors, extra_scores):
        return self._update_oil(
            emitter_state, genotypes, fitnesses, descriptors, extra_scores, repertoire
        )

    def _update_oil(self, state, genotypes, fitnesses, descriptors, extra_scores, repertoire):
        if "oil_descriptors" not in extra_scores:
            return state

        oil = extra_scores["oil_descriptors"]
        idx = jnp.argmin(
            jnp.linalg.norm(descriptors[:, None, :] - self._centroids[None, :, :], axis=-1),
            axis=1,
        )
        inserted = fitnesses >= repertoire.fitnesses[idx] - 1e-6
        attempts = state.attempts + jax.ops.segment_sum(
            jnp.ones_like(fitnesses), idx, num_segments=self._num_centroids
        )
        successes = state.successes + jax.ops.segment_sum(
            inserted.astype(jnp.float32), idx, num_segments=self._num_centroids
        )
        safe_idx = jnp.where(inserted, idx, self._num_centroids)
        filled_update = jax.ops.segment_max(
            inserted.astype(jnp.float32), idx, num_segments=self._num_centroids
        )
        new_oil = state.oil_per_cell.at[safe_idx].set(oil)
        new_filled = jnp.maximum(state.oil_filled, filled_update)
        if "oil_parent_idx" not in extra_scores:
            return state.replace(
                oil_per_cell=new_oil,
                oil_filled=new_filled,
                attempts=attempts,
                successes=successes,
            )

        parent_idx = extra_scores["oil_parent_idx"]
        parent_fit = extra_scores["oil_parent_fitness"]
        parent_genotypes = extra_scores["oil_parent_genotypes"]
        parent_oil = extra_scores["oil_parent_oil"]
        prev_fitnesses = extra_scores["oil_prev_fitnesses"]
        child_genotypes = jax.tree_util.tree_map(lambda x: x[-self._batch_size :], genotypes)
        child_fitnesses = fitnesses[-self._batch_size :]
        child_descriptors = descriptors[-self._batch_size :]
        child_oil = oil[-self._batch_size :]
        child_idx = jnp.argmin(
            jnp.linalg.norm(
                child_descriptors[:, None, :] - self._centroids[None, :, :], axis=-1
            ),
            axis=1,
        )

        prev_cell_fitness = prev_fitnesses[child_idx]
        qd_gain = jnp.where(
            jnp.isfinite(prev_cell_fitness),
            jnp.maximum(child_fitnesses - prev_cell_fitness, 0.0),
            jnp.maximum(child_fitnesses - parent_fit, 0.0) + 1.0,
        )
        qd_improved = qd_gain > 1e-6
        gain_n = qd_gain / (jnp.max(qd_gain) + 1e-8)
        delta_oil = jnp.clip(child_oil - parent_oil, -0.5, 0.5)
        oil_norm = jnp.sum(jnp.square(delta_oil), axis=1) + 1e-6
        path_weight = qd_improved.astype(jnp.float32) * (0.25 + 0.75 * gain_n) * (
            1.0 + jnp.sqrt(oil_norm)
        )
        path_count = jax.ops.segment_sum(
            path_weight, parent_idx, num_segments=self._num_centroids
        )
        path_mask = path_count > 0.0

        def update_path(old_leaf, child_leaf, parent_leaf):
            delta = child_leaf - parent_leaf
            weighted = delta * path_weight.reshape(
                (self._batch_size,) + (1,) * (delta.ndim - 1)
            )
            summed = jax.ops.segment_sum(
                weighted, parent_idx, num_segments=self._num_centroids
            )
            mean_delta = summed / (
                path_count.reshape((self._num_centroids,) + (1,) * (summed.ndim - 1))
                + 1e-8
            )
            mask = path_mask.reshape((self._num_centroids,) + (1,) * (old_leaf.ndim - 1))
            proposed = self._path_decay * old_leaf + self._path_lr * mean_delta
            return jnp.where(mask, proposed, self._path_decay * old_leaf)

        oil_path = jax.tree_util.tree_map(
            update_path, state.oil_path, child_genotypes, parent_genotypes
        )
        grad_weight = path_weight / oil_norm

        def update_grad(old_leaf, child_leaf, parent_leaf):
            delta = child_leaf - parent_leaf
            weighted = (
                delta[:, None]
                * delta_oil.reshape((self._batch_size, self._oil_dim) + (1,) * (delta.ndim - 1))
                * grad_weight.reshape((self._batch_size, 1) + (1,) * (delta.ndim - 1))
            )
            summed = jax.ops.segment_sum(
                weighted, parent_idx, num_segments=self._num_centroids
            )
            count = jax.ops.segment_sum(
                path_weight, parent_idx, num_segments=self._num_centroids
            )
            mean_grad = summed / (
                count.reshape((self._num_centroids, 1) + (1,) * (delta.ndim - 1)) + 1e-8
            )
            mask = path_mask.reshape((self._num_centroids, 1) + (1,) * (delta.ndim - 1))
            return jnp.where(
                mask,
                self._path_decay * old_leaf + self._path_lr * mean_grad,
                self._path_decay * old_leaf,
            )

        oil_grad = jax.tree_util.tree_map(
            update_grad, state.oil_grad, child_genotypes, parent_genotypes
        )
        return state.replace(
            oil_per_cell=new_oil,
            oil_filled=new_filled,
            attempts=attempts,
            successes=successes,
            oil_path=oil_path,
            oil_grad=oil_grad,
            path_strength=self._path_decay * state.path_strength + self._path_lr * path_count,
        )
