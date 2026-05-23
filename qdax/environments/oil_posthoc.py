from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.special import gamma

from qdax.environments.lz76 import LZ76_jax, quantize_observation_bins


class OILArchiveEnvProxy:
    """Expose OIL descriptor metadata while delegating env behavior."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def behavior_descriptor_length(self):
        return 2

    @property
    def state_descriptor_length(self):
        return 2

    @property
    def behavior_descriptor_limits(self):
        return jnp.array([0.0, -1.0]), jnp.array([1.0, 1.0])


@partial(jax.jit, static_argnames=("k",))
def _kth_neighbor_index(data: jnp.ndarray, k: int) -> jnp.ndarray:
    """Match the previous annax search output ordering without host transfers."""
    similarities = jnp.matmul(data, data.T)
    candidate_indices = jnp.argpartition(similarities, -(k + 1), axis=-1)[
        ..., -(k + 1) :
    ]
    candidate_values = jnp.take_along_axis(similarities, candidate_indices, axis=-1)
    sorted_positions = jnp.argsort(-candidate_values, axis=-1)
    sorted_indices = jnp.take_along_axis(candidate_indices, sorted_positions, axis=-1)
    return sorted_indices[:, k].astype(data.dtype)


def k_l_entropy(data, k=1):
    """Calculate entropy estimate using k-nearest neighbors with pure JAX."""
    n_samples, n_dimensions = data.shape
    vol_hypersphere = jnp.pi ** (n_dimensions / 2) / gamma(n_dimensions / 2 + 1)
    epsilon = _kth_neighbor_index(data, k)
    entropy = (
        n_dimensions * jnp.mean(jnp.log(epsilon + 1e-10))
        + jnp.log(vol_hypersphere + 1e-10)
        + 0.577216
        + jnp.log(n_samples - 1)
    )
    return jnp.float32(entropy)


def k_l_entropy_batch(data, k=1):
    """Batched equivalent of the legacy annax-based k_l_entropy."""
    n_samples = data.shape[-2]
    n_dimensions = data.shape[-1]
    similarities = jnp.matmul(data, jnp.swapaxes(data, -1, -2))
    candidate_indices = jnp.argpartition(similarities, -(k + 1), axis=-1)[
        ..., -(k + 1) :
    ]
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


NORMALIZED_LZ76 = {
    "ant": (20, 42),
    "halfcheetah": (31, 52),
    "walker2d": (2, 30),
    "hopper": (2, 30),
    "humanoid": (8, 40),
    "grasp": (1112, 1088),
    "fetch": (950, 970),
    "sphereenv": (90, 130),
    "rastriginenv": (90, 130),
}

NORMALIZED_OI = {
    "ant": (-55, 175),
    "halfcheetah": (-30, 150),
    "walker2d": (-200, 450),
    "hopper": (-50, 80),
    "humanoid": (-300, 350),
    "grasp": (-800, 1462),
    "fetch": (-600, 1300),
    "sphereenv": (70, 90),
    "rastriginenv": (70, 90),
}

LZ_NUM_BINS = 64
LZ_NUM_SAMPLES = 100
DEFAULT_LZ_OBS_LIMIT = 20.0
ANT_OIL_ANGULAR_FEATURES = (5, 13)
HALFCHEETAH_OIL_ANGULAR_FEATURES = (3, 9)
HOPPER_OIL_ANGULAR_FEATURES = (2, 5)
WALKER2D_OIL_ANGULAR_FEATURES = (2, 8)
HUMANOID_OIL_ANGULAR_FEATURES = (5, 22)
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
    if env_name == "hopper":
        angles = obs[HOPPER_OIL_ANGULAR_FEATURES[0] : HOPPER_OIL_ANGULAR_FEATURES[1]]
        return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)))
    if env_name == "walker2d":
        angles = obs[
            WALKER2D_OIL_ANGULAR_FEATURES[0] : WALKER2D_OIL_ANGULAR_FEATURES[1]
        ]
        return jnp.concatenate((jnp.sin(angles), jnp.cos(angles)))
    if env_name == "humanoid":
        angles = obs[
            HUMANOID_OIL_ANGULAR_FEATURES[0] : HUMANOID_OIL_ANGULAR_FEATURES[1]
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
    lz_obs_min, lz_obs_max = _get_lz_observation_bounds(
        norm_env_name, obs_sequence.shape[-1]
    )
    if env_name in (
        "ant",
        "halfcheetah",
        "halfcheetah_angular",
        "hopper",
        "walker2d",
        "humanoid",
    ):
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
        2.0
        * ((compute_o_information(obs_sequence) - oi_min) / (oi_max - oi_min + 1e-8))
        - 1.0,
        -1.0,
        1.0,
    )
    return jnp.array([lz, oi])


compute_oil_descriptor_batch = jax.vmap(compute_oil_descriptor, in_axes=(0, None))
