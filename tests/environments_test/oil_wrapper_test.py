import annax
import jax
import jax.numpy as jnp
from jax.scipy.special import gamma

from qdax.environments.lz76 import quantize_observation_sequence
from qdax.environments.wrappers import k_l_entropy


def _legacy_annax_entropy(data: jnp.ndarray, k: int) -> jnp.ndarray:
    indices, _ = annax.Index(data).search(data, k=k + 1)
    epsilon = indices[:, k].astype(data.dtype)
    n_samples, n_dimensions = data.shape
    vol_hypersphere = jnp.pi ** (n_dimensions / 2) / gamma(n_dimensions / 2 + 1)
    entropy = (
        n_dimensions * jnp.mean(jnp.log(epsilon + 1e-10))
        + jnp.log(vol_hypersphere + 1e-10)
        + 0.577216
        + jnp.log(n_samples - 1)
    )
    return jnp.float32(entropy)


def test_k_l_entropy_matches_legacy_annax_behavior() -> None:
    data = jax.random.normal(jax.random.PRNGKey(0), (32, 11))

    assert jnp.allclose(k_l_entropy(data, 3), _legacy_annax_entropy(data, 3)).item()
    assert jnp.allclose(
        k_l_entropy(data[:, :1], 1),
        _legacy_annax_entropy(data[:, :1], 1),
    ).item()


def test_quantized_lz_symbols_preserve_dimension_identity() -> None:
    observations = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    mins = jnp.array([0.0, 0.0], dtype=jnp.float32)
    maxs = jnp.array([1.0, 1.0], dtype=jnp.float32)

    symbols = quantize_observation_sequence(observations, mins, maxs, num_bins=4)

    assert symbols.tolist() == [0, 4, 3, 7]
