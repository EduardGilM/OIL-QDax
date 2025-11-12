import jax
import jax.numpy as jnp
from brax.v1.envs import Env
from flax import struct
from typing import Optional

@struct.dataclass
class DummyQP:
    """Dummy QP class to be compatible with Brax wrappers."""
    pos: jnp.ndarray

@struct.dataclass
class RastriginState:
    """State for the Rastrigin environment."""
    pipeline_state: Optional[None]
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    qp: DummyQP
    metrics: dict = struct.field(default_factory=dict)
    info: dict = struct.field(default_factory=dict)

class RastriginEnv(Env):
    """
    A simple, purely mathematical environment where the agent's goal is to
    find the highest reward point on an N-dimensional Rastrigin function.
    
    The Rastrigin function is a highly multimodal function commonly used
    as a performance test problem for optimization algorithms.
    """
    def __init__(
        self,
        n_dimensions: int = 20,
        episode_length: int = 100,
        minval: float = -5.12,
        maxval: float = 5.12,
        obs_radius: float = 1.0,
        num_adyacentes: int = 4,
    ):
        super().__init__(config=None)
        self._n_dimensions = n_dimensions
        self._episode_length = episode_length
        self.minval = minval
        self.maxval = maxval
        self._obs_radius = obs_radius
        self._num_adyacentes = num_adyacentes

    @property
    def observation_size(self) -> int:
        # 1 posición actual + 2 vecinos por dimensión (+ y -)
        return (2 * self._n_dimensions + 1) * self._n_dimensions

    @property
    def action_size(self) -> int:
        return self._n_dimensions

    def _generate_observation(self, position: jnp.ndarray) -> jnp.ndarray:
        """The observation is the position of the agent and its adjacent positions within the observation radius.
        
        Generates 2 neighbors per dimension (one in each direction +/- obs_radius).
        """
        # Generate adjacent positions for ALL dimensions
        num_neighbors = 2 * self._n_dimensions
        offsets = jnp.zeros((num_neighbors, self._n_dimensions))
        
        # Create offsets: for each dimension, add +obs_radius and -obs_radius
        for dim in range(self._n_dimensions):
            offsets = offsets.at[2*dim, dim].set(self._obs_radius)      # positive direction
            offsets = offsets.at[2*dim + 1, dim].set(-self._obs_radius) # negative direction
        
        adyacentes = position + offsets
        adyacentes = jnp.clip(adyacentes, self.minval, self.maxval)
        obs = jnp.concatenate([position.reshape(1, -1), adyacentes], axis=0)
        return obs.flatten()

    def reset(self, rng: jnp.ndarray) -> RastriginState:
        """Resets the environment to an initial state."""
        # Generate a random initial position
        init_pos = jax.random.uniform(
            rng,
            shape=(self._n_dimensions,),
            minval=self.minval,
            maxval=self.maxval,
        )

        # Generate initial observation
        init_obs = self._generate_observation(init_pos)

        # Create dummy qp
        dummy_qp = DummyQP(pos=init_pos)

        # Initial state
        state = RastriginState(
            pipeline_state=None,
            obs=init_obs,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            qp=dummy_qp,
            metrics={"reward": jnp.zeros(())},
            info={
                "steps": jnp.zeros((), dtype=jnp.int32),
                "pos": init_pos,
            },
        )
        return state

    def step(self, state: RastriginState, action: jnp.ndarray) -> RastriginState:
        """Run one timestep of the environment's dynamics."""
        # The action is a displacement vector added to the current position
        # This creates temporal correlation: position evolves incrementally
        new_pos = jnp.clip(state.qp.pos + action, self.minval, self.maxval)

        # Calculate reward using the Rastrigin function
        # Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
        # We negate it to maximize (find minimum)
        A = 10.0
        n = self._n_dimensions
        rastrigin_value = A * n + jnp.sum(new_pos**2 - A * jnp.cos(2 * jnp.pi * new_pos))
        reward = -rastrigin_value  # Negate to maximize

        # Update steps and check for termination
        steps = state.info["steps"] + 1
        done = jnp.zeros(())  # Let the EpisodeWrapper handle termination

        # Generate new observation
        new_obs = self._generate_observation(new_pos)

        # Create dummy qp
        dummy_qp = DummyQP(pos=new_pos)

        # Update state
        new_info = state.info | {"steps": steps, "pos": new_pos}
        state = state.replace(
            obs=new_obs,
            reward=reward,
            done=done,
            info=new_info,
            qp=dummy_qp,
        )
        state.metrics.update(reward=reward)

        return state
