from functools import partial
from typing import Dict, Optional, Tuple
import jax.lax as lax
from jax.scipy.special import gamma

import flax.struct
import jax
import jax.numpy as jnp
from brax.v1 import jumpy as jp
from brax.v1.envs import Env, State, Wrapper
from qdax.environments.lz76 import (
    LZ76_jax,
    quantize_observation_bins,
)


class CompletedEvalMetrics(flax.struct.PyTreeNode):
    current_episode_metrics: Dict[str, jp.ndarray]
    completed_episodes_metrics: Dict[str, jp.ndarray]
    completed_episodes: jp.ndarray
    completed_episodes_steps: jp.ndarray


class CompletedEvalWrapper(Wrapper):
    """Brax env with eval metrics for completed episodes."""

    STATE_INFO_KEY = "completed_eval_metrics"

    def reset(self, rng: jp.ndarray) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=jax.tree_util.tree_map(
                jp.zeros_like, reset_state.metrics
            ),
            completed_episodes_metrics=jax.tree_util.tree_map(
                lambda x: jp.zeros_like(jp.sum(x)), reset_state.metrics
            ),
            completed_episodes=jp.zeros(()),
            completed_episodes_steps=jp.zeros(()),
        )
        reset_state.info[self.STATE_INFO_KEY] = eval_metrics
        return reset_state

    def step(self, state: State, action: jp.ndarray) -> State:
        state_metrics = state.info[self.STATE_INFO_KEY]
        if not isinstance(state_metrics, CompletedEvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info[self.STATE_INFO_KEY]
        nstate = self.env.step(state, action)
        nstate.metrics["reward"] = nstate.reward
        # steps stores the highest step reached when done = True, and then
        # the next steps becomes action_repeat
        completed_episodes_steps = state_metrics.completed_episodes_steps + jp.sum(
            nstate.info["steps"] * nstate.done
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b, state_metrics.current_episode_metrics, nstate.metrics
        )
        completed_episodes = state_metrics.completed_episodes + jp.sum(nstate.done)
        completed_episodes_metrics = jax.tree_util.tree_map(
            lambda a, b: a + jp.sum(b * nstate.done),
            state_metrics.completed_episodes_metrics,
            current_episode_metrics,
        )
        current_episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a * (1 - nstate.done) + b * nstate.done,
            current_episode_metrics,
            nstate.metrics,
        )

        eval_metrics = CompletedEvalMetrics(
            current_episode_metrics=current_episode_metrics,
            completed_episodes_metrics=completed_episodes_metrics,
            completed_episodes=completed_episodes,
            completed_episodes_steps=completed_episodes_steps,
        )
        nstate.info[self.STATE_INFO_KEY] = eval_metrics
        return nstate


class ClipRewardWrapper(Wrapper):
    """Wraps gym environments to clip the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(
        self,
        env: Env,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
    ) -> None:
        super().__init__(env)
        self._clip_min = clip_min
        self._clip_max = clip_max

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        return state.replace(
            reward=jp.clip(state.reward, a_min=self._clip_min, a_max=self._clip_max)
        )

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        return state.replace(
            reward=jp.clip(state.reward, a_min=self._clip_min, a_max=self._clip_max)
        )


class OffsetRewardWrapper(Wrapper):
    """Wraps gym environments to offset the reward to be greater than 0.

    Utilisation is simple: create an environment with Brax, pass
    it to the wrapper with the name of the environment, and it will
    work like before and will simply clip the reward to be greater than 0.
    """

    def __init__(self, env: Env, offset: float = 0.0) -> None:
        super().__init__(env)
        self._offset = offset

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        return state.replace(reward=state.reward + self._offset)

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        return state.replace(reward=state.reward + self._offset)

@partial(jax.jit, static_argnames=("k",))
def _kth_neighbor_index(data: jnp.ndarray, k: int) -> jnp.ndarray:
    """Match the previous annax search output ordering without host transfers."""
    similarities = jnp.matmul(data, data.T)
    candidate_indices = jnp.argpartition(similarities, -(k + 1), axis=-1)[..., -(k + 1) :]
    candidate_values = jnp.take_along_axis(similarities, candidate_indices, axis=-1)
    sorted_positions = jnp.argsort(-candidate_values, axis=-1)
    sorted_indices = jnp.take_along_axis(candidate_indices, sorted_positions, axis=-1)
    return sorted_indices[:, k].astype(data.dtype)


def k_l_entropy(data, k=1):
    """Calculate entropy estimate using k-nearest neighbors with pure JAX.
    
    Args:
        data: array of shape (n_samples, n_dimensions)
        k: number of neighbors (excluding self)
    
    Returns:
        entropy: float, entropy estimate
    """
    n_samples, n_dimensions = data.shape

    vol_hypersphere = jnp.pi**(n_dimensions/2) / gamma(n_dimensions/2 + 1)
    epsilon = _kth_neighbor_index(data, k)
    entropy = (n_dimensions * jnp.mean(jnp.log(epsilon + 1e-10)) + 
               jnp.log(vol_hypersphere + 1e-10) + 0.577216 + jnp.log(n_samples-1))
    
    return jnp.float32(entropy)


def k_l_entropy_batch(data, k=1):
    """Batched equivalent of the legacy annax-based k_l_entropy."""
    n_samples = data.shape[-2]
    n_dimensions = data.shape[-1]
    similarities = jnp.matmul(data, jnp.swapaxes(data, -1, -2))
    candidate_indices = jnp.argpartition(similarities, -(k + 1), axis=-1)[..., -(k + 1) :]
    candidate_values = jnp.take_along_axis(similarities, candidate_indices, axis=-1)
    sorted_positions = jnp.argsort(-candidate_values, axis=-1)
    sorted_indices = jnp.take_along_axis(candidate_indices, sorted_positions, axis=-1)
    epsilon = sorted_indices[..., k].astype(data.dtype)
    vol_hypersphere = jnp.pi ** (n_dimensions / 2) / gamma(n_dimensions / 2 + 1)
    entropy = (
        n_dimensions * jnp.mean(jnp.log(epsilon + 1e-10), axis=-1)
        + jnp.log(vol_hypersphere + 1e-10)
        + 0.577216
        + jnp.log(n_samples - 1)
    )
    return entropy.astype(jnp.float32)


def extract_single_column(matrix, col_idx):
    """Extract a single column from a matrix in a JAX-safe way.
    
    Args:
        matrix: Input matrix with shape [rows, cols]
        col_idx: Column index to extract
    
    Returns:
        A column vector with shape [rows, 1]
    """
    rows, cols = matrix.shape
    
    def get_element(row_idx):
        element = lax.dynamic_slice(matrix[row_idx], (col_idx,), (1,))
        return element[0]

    column_data = jax.vmap(get_element)(jnp.arange(rows))

    return column_data.reshape(-1, 1)

def exclude_column(matrix, col_idx):
    """Create a new matrix excluding the specified column in a JAX-safe way.
    
    Args:
        matrix: Input matrix with shape [rows, cols]
        col_idx: Column index to exclude
        
    Returns:
        A matrix with shape [rows, cols-1] with col_idx removed
    """
    rolled_matrix = jnp.roll(matrix, shift=-col_idx, axis=1)

    result_matrix = rolled_matrix[:, 1:] 

    return result_matrix

NORMALIZED_LZ76 = {
    "ant": (20, 42),
    "halfcheetah": (31, 52),
    "walker2d": (-538.19, 538.19), # Placeholder, need to compute
    "hopper": (-538.19, 538.19), # Placeholder, need to compute
    "humanoid": (-538.19, 538.19), # Placeholder, need to compute
    "grasp": (1112, 1088), # Placeholder, need to compute
    "fetch": (950, 970), # Placeholder, need to compute
    "sphereenv": (90, 130),
    "rastriginenv": (90, 130),
}

NORMALIZED_OI = {
    "ant": (-55, 175),
    "halfcheetah": (-30, 150),
    "walker2d": (-538.19, 538.19), # Placeholder, need to compute
    "hopper": (-122, 116), # Placeholder, need to compute
    "humanoid": (-538.19, 538.19), # Placeholder, need to compute
    "grasp": (-800, 1462), # Placeholder, need to compute
    "fetch": (-600, 1300), # Placeholder, need to compute
    "sphereenv": (70, 90),
    "rastriginenv": (70, 90),
}

LZ_NUM_BINS = 64
LZ_NUM_SAMPLES = 100
DEFAULT_LZ_OBS_LIMIT = 20.0
ANT_OIL_ANGULAR_FEATURES = (5, 13)
HALFCHEETAH_OIL_ANGULAR_FEATURES = (3, 9)
LZ_OBSERVATION_BOUNDS = {
    "ant": (
        jnp.array(
            [
                0.3,
                0.1,
                -0.2,
                -0.2,
                -0.7,
                -0.6,
                0.5,
                -0.6,
                -1.3,
                -0.6,
                -1.3,
                -0.6,
                0.5,
                -1.2,
                -1.2,
                -1.4,
                -2.7,
                -2.7,
                -6.0,
                -10.0,
                -9.6,
                -9.6,
                -10.3,
                -9.9,
                -10.5,
                -9.9,
                -10.0,
            ],
            dtype=jnp.float32,
        ),
        jnp.array(
            [
                0.8,
                1.0,
                0.2,
                0.2,
                1.0,
                0.6,
                1.3,
                0.6,
                -0.4,
                0.6,
                -0.5,
                0.6,
                1.3,
                1.3,
                1.4,
                1.4,
                2.8,
                2.7,
                6.3,
                9.9,
                10.3,
                9.9,
                10.1,
                10.4,
                9.3,
                10.3,
                10.3,
            ],
            dtype=jnp.float32,
        ),
    ),
    "halfcheetah": (
        jnp.array(
            [
                -2.0,
                -1.0,
                -1.0,
                -4.0,
                -4.0,
                -4.0,
                -4.0,
                -4.0,
                -4.0,
                -8.0,
                -8.0,
                -15.0,
                -40.0,
                -80.0,
                -120.0,
                -50.0,
                -120.0,
                -150.0,
            ],
            dtype=jnp.float32,
        ),
        jnp.array(
            [
                2.0,
                1.0,
                1.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                4.0,
                8.0,
                8.0,
                15.0,
                40.0,
                80.0,
                120.0,
                50.0,
                120.0,
                150.0,
            ],
            dtype=jnp.float32,
        ),
    ),
}


def _get_lz_observation_bounds(
    env_name: str, obs_dim: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    bounds = LZ_OBSERVATION_BOUNDS.get(env_name)
    if bounds is not None and bounds[0].shape[0] == obs_dim:
        return bounds

    obs_limit = jnp.full((obs_dim,), DEFAULT_LZ_OBS_LIMIT, dtype=jnp.float32)
    return -obs_limit, obs_limit


def _sample_lz_observations(obs_sequence: jnp.ndarray) -> jnp.ndarray:
    num_samples = min(obs_sequence.shape[0], LZ_NUM_SAMPLES)
    indices = jnp.linspace(0, obs_sequence.shape[0] - 1, num_samples).astype(jnp.int32)
    return obs_sequence[indices]


def _oil_observation(env_name: str, obs: jnp.ndarray) -> jnp.ndarray:
    if env_name == "ant":
        angles = obs[ANT_OIL_ANGULAR_FEATURES[0] : ANT_OIL_ANGULAR_FEATURES[1]]
        return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)))
    if env_name in ("halfcheetah", "halfcheetah_angular"):
        angles = obs[
            HALFCHEETAH_OIL_ANGULAR_FEATURES[0] : HALFCHEETAH_OIL_ANGULAR_FEATURES[1]
        ]
        return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)))
    return obs


def compute_o_information(obs_sequence: jnp.ndarray) -> jnp.ndarray:
    n_vars = obs_sequence.shape[1]
    k = 3
    h_joint = k_l_entropy(obs_sequence, k)
    columns = jnp.swapaxes(obs_sequence, 0, 1)[..., jnp.newaxis]
    h_xj = k_l_entropy_batch(columns, 1)
    base_indices = jnp.arange(n_vars - 1)
    excluded_indices = jax.vmap(lambda j: base_indices + (base_indices >= j))(
        jnp.arange(n_vars)
    )
    excluded_data = jnp.take(obs_sequence, excluded_indices, axis=1).transpose(
        1, 0, 2
    )
    h_excl_j = k_l_entropy_batch(excluded_data, max(k - 1, 1))
    return (n_vars - 2) * h_joint + jnp.sum(h_xj - h_excl_j)


def compute_oil_descriptor(obs_sequence: jnp.ndarray, env_name: str) -> jnp.ndarray:
    obs_sequence = jax.vmap(lambda obs: _oil_observation(env_name, obs))(obs_sequence)
    norm_env_name = "halfcheetah" if env_name == "halfcheetah_angular" else env_name
    lz_obs_min, lz_obs_max = _get_lz_observation_bounds(norm_env_name, obs_sequence.shape[-1])
    if env_name in ("ant", "halfcheetah", "halfcheetah_angular"):
        lz_obs_min = jnp.full((obs_sequence.shape[-1],), -1.0, dtype=jnp.float32)
        lz_obs_max = jnp.full((obs_sequence.shape[-1],), 1.0, dtype=jnp.float32)

    complexity_obs_sequence = _sample_lz_observations(obs_sequence)
    obs_bins = quantize_observation_bins(
        complexity_obs_sequence,
        lz_obs_min,
        lz_obs_max,
        LZ_NUM_BINS,
    )
    raw_lz = jnp.float32(jnp.mean(jax.vmap(LZ76_jax, in_axes=1)(obs_bins)))
    lz_min, lz_max = NORMALIZED_LZ76[norm_env_name]
    oi_min, oi_max = NORMALIZED_OI[norm_env_name]
    lz = jnp.clip((raw_lz - lz_min) / (lz_max - lz_min + 1e-8), 0.0, 1.0)
    oi = jnp.clip(
        2.0 * ((compute_o_information(obs_sequence) - oi_min) / (oi_max - oi_min + 1e-8)) - 1.0,
        -1.0,
        1.0,
    )
    return jnp.array([lz, oi])


compute_oil_descriptor_batch = jax.vmap(compute_oil_descriptor, in_axes=(0, None))


class OILWrapper(Wrapper):
    """Wraps gym environments to add both Lempel-Ziv complexity and O-Information of the observations."""

    def __init__(self, env: Env, episode_length: int = 1000, **kwargs):
        super().__init__(env)
        self.episode_length = episode_length
        self._debug = kwargs.pop("debug", False)

        unwrapped_env = env
        while hasattr(unwrapped_env, "env"):
            unwrapped_env = unwrapped_env.env
        self._base_env_name = unwrapped_env.__class__.__name__.lower()
        
    @property
    def behavior_descriptor_length(self):
        return 2

    @property
    def state_descriptor_length(self) -> int:
        return self.behavior_descriptor_length
    
    @property
    def behavior_descriptor_limits(self):
        return (jnp.array([0.0, -1.0]), jnp.array([1.0, 1.0]))

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        
        obs_dim = _oil_observation(self._base_env_name, state.obs).shape[0]

        lz_obs_min, lz_obs_max = _get_lz_observation_bounds(self._base_env_name, obs_dim)
        if self._base_env_name in ("ant", "halfcheetah"):
            lz_obs_min = jnp.full((obs_dim,), -1.0, dtype=jnp.float32)
            lz_obs_max = jnp.full((obs_dim,), 1.0, dtype=jnp.float32)
        state.info["obs_sequence"] = jnp.zeros((self.episode_length, obs_dim), dtype=jnp.float32)
        state.info["lz_obs_min"] = lz_obs_min
        state.info["lz_obs_max"] = lz_obs_max
        state.info["current_step"] = 0
        state.info["lz76_complexity"] = jnp.float32(0)
        state.info["o_info_value"] = jnp.float32(0)
        state.info["state_descriptor"] = jnp.zeros(2, dtype=jnp.float32)
        return state

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)    
        
        obs = _oil_observation(self._base_env_name, state.obs)
        obs_dim = state.info["obs_sequence"].shape[1]
        
        obs = obs[:obs_dim]
        
        current_step = state.info["current_step"]
        obs_sequence = state.info["obs_sequence"].at[current_step, :].set(obs)
        
        is_final_step = current_step == (self.episode_length - 2)
        complexities = jnp.float32(state.info["lz76_complexity"])
        o_info_values = jnp.float32(state.info["o_info_value"])
        state_descriptor = state.info["state_descriptor"]
        lz_obs_min = state.info["lz_obs_min"]
        lz_obs_max = state.info["lz_obs_max"]
        
        def compute_final_metrics(obs_seq):
            complexity_obs_seq = _sample_lz_observations(obs_seq)
            obs_bins = quantize_observation_bins(
                complexity_obs_seq,
                lz_obs_min,
                lz_obs_max,
                LZ_NUM_BINS,
            )
            raw_complexity = jnp.float32(
                jnp.mean(jax.vmap(LZ76_jax, in_axes=1, out_axes=0)(obs_bins))
            )
            min_samples_for_o_info = 12
            raw_o_info = lax.cond(
                obs_seq.shape[0] >= min_samples_for_o_info,
                self._compute_o_information,
                lambda x: jnp.float32(0.0),
                obs_seq,
            )

            lz76_min, lz76_max = NORMALIZED_LZ76[self._base_env_name]
            oi_min, oi_max = NORMALIZED_OI[self._base_env_name]
            
            normalized_complexity = jnp.clip((raw_complexity - lz76_min) / (lz76_max - lz76_min), 0.0, 1.0)
            normalized_o_info = jnp.clip(2.0 * ((raw_o_info - oi_min) / (oi_max - oi_min)) - 1.0, -1.0, 1.0)

            if self._debug:
                jax.debug.print("Raw LZ complexity: {x}", x=raw_complexity)
                jax.debug.print("Raw OI: {x}", x=raw_o_info)
                jax.debug.print("Normalized complexity: {x}", x=normalized_complexity)
                jax.debug.print("Normalized o-info: {x}", x=normalized_o_info)
            
            return raw_complexity, raw_o_info, jnp.array([normalized_complexity, normalized_o_info])
        
        def keep_previous(_):
            return complexities, o_info_values, state_descriptor
        
        
        complexities, o_info_values, state_descriptor = jax.lax.cond(
            is_final_step,
            compute_final_metrics,
            keep_previous,
            obs_sequence
        )

        state.info.update({
            "obs_sequence": obs_sequence,
            "current_step": current_step + 1,
            "lz76_complexity": complexities,
            "o_info_value": o_info_values,
            "state_descriptor": state_descriptor
        })

        #jax.debug.print("State info: {x}", x=state.info["state_descriptor"])

        return state
    
    def _compute_o_information(self, obs_sequence):
        """Compute O-Information with fully optimized JAX operations."""
        return compute_o_information(obs_sequence)
