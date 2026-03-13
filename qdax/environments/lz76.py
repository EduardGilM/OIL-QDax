import jax
import jax.numpy as jnp
from brax.v1 import jumpy as jp


def action_to_binary(action: jnp.ndarray) -> jnp.ndarray:
    """Convert actions to binary representation.
    
    For each floating-point action value, uses 16 bits to represent it
    instead of the full 32 bits for float32 to reduce complexity.
    """
    action_float16 = action.astype(jnp.float16)
    action_view = action_float16.view(jnp.uint16)
    binary_rep = jnp.unpackbits(
        action_view.view(jnp.uint8), bitorder='big', axis=-1
    )
    binary_rep_flat = binary_rep.reshape(-1)
    return binary_rep_flat

def action_to_binary_padded(action: jp.ndarray) -> jnp.ndarray:
    """Converts actions into a fixed-length binary representation.
    
    Args:
        action: Array of actions to convert
        max_length: Maximum length of the binary representation
        
    Returns:
        Tuple of (padded binary representation, real length)
    """
    action = action.flatten()
    
    return action_to_binary(action)


def quantize_observation_sequence(
    observations: jnp.ndarray,
    min_values: jnp.ndarray,
    max_values: jnp.ndarray,
    num_bins: int,
) -> jnp.ndarray:
    """Discretize continuous observations into a flat symbol sequence.

    Each observation dimension gets its own symbol range so that identical bin
    indices from different dimensions are not conflated by LZ76.
    """
    observations = observations.astype(jnp.float32)
    min_values = min_values.astype(jnp.float32)
    max_values = max_values.astype(jnp.float32)

    clipped = jnp.clip(observations, min_values, max_values)
    ranges = jnp.maximum(max_values - min_values, 1e-6)
    normalized = (clipped - min_values) / ranges
    bin_indices = jnp.floor(normalized * num_bins).astype(jnp.int32)
    bin_indices = jnp.clip(bin_indices, 0, num_bins - 1)

    dim_offsets = jnp.arange(observations.shape[-1], dtype=jnp.int32) * num_bins
    return (bin_indices + dim_offsets).reshape(-1)


def quantize_observation_bins(
    observations: jnp.ndarray,
    min_values: jnp.ndarray,
    max_values: jnp.ndarray,
    num_bins: int,
) -> jnp.ndarray:
    """Discretize each observation component into a bin index."""
    observations = observations.astype(jnp.float32)
    min_values = min_values.astype(jnp.float32)
    max_values = max_values.astype(jnp.float32)

    clipped = jnp.clip(observations, min_values, max_values)
    ranges = jnp.maximum(max_values - min_values, 1e-6)
    normalized = (clipped - min_values) / ranges
    bin_indices = jnp.floor(normalized * num_bins).astype(jnp.int32)
    return jnp.clip(bin_indices, 0, num_bins - 1)


def hash_quantized_observations(bin_indices: jnp.ndarray) -> jnp.ndarray:
    """Pack one discretized state vector into a single symbol per timestep."""
    modulus = jnp.int32(2_147_483_647)
    multiplier = jnp.int32(131)

    def pack_row(row: jnp.ndarray) -> jnp.ndarray:
        def scan_fn(acc: jnp.ndarray, value: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            updated = jnp.mod(acc * multiplier + value + 1, modulus)
            return updated, None

        packed, _ = jax.lax.scan(scan_fn, jnp.int32(0), row.astype(jnp.int32))
        return packed

    return jax.vmap(pack_row)(bin_indices)


def LZ76_jax(ss: jnp.ndarray) -> jnp.int32:
    """Implementation of the LZ76 algorithm."""
    n = ss.size
    if n == 0:
        return jnp.int32(0)

    def cond_fun(state):
        i, k, l, k_max, c = state
        return (l + k) <= n

    def body_fun(state):
        i, k, l, k_max, c = state
        same = ss[i + k - 1] == ss[l + k - 1]

        def true_branch(_):
            return i, k + 1, l, k_max, c

        def false_branch(_):
            k_max_updated = jax.lax.max(k_max, k)
            i_updated = i + 1

            def inner_true(_):
                c_updated = c + 1
                l_updated = l + k_max_updated
                return 0, 1, l_updated, 1, c_updated

            i_eq_l = i_updated == l
            i_new, k_new, l_new, k_max_new, c_new = jax.lax.cond(
                i_eq_l,
                inner_true,
                lambda _: (i_updated, 1, l, k_max_updated, c),
                operand=None
            )
            return i_new, k_new, l_new, k_max_new, c_new

        i_new, k_new, l_new, k_max_new, c_new = jax.lax.cond(
            same,
            true_branch,
            false_branch,
            operand=None
        )
        return i_new, k_new, l_new, k_max_new, c_new

    state = (0, 1, 1, 1, 1)
    state = jax.lax.while_loop(cond_fun, body_fun, state)
    _, _, _, _, c = state
    return c
