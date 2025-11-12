import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import argparse

from qdax.environments import create
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Params, RNGKey


def load_repertoire(repertoire_path: str):
    """Load the repertoire from disk."""
    print(f"\n{'='*80}")
    print(f"Loading repertoire from: {repertoire_path}")
    print(f"{'='*80}")
    
    centroids = np.load(os.path.join(repertoire_path, "centroids.npy"))
    descriptors = np.load(os.path.join(repertoire_path, "descriptors.npy"))
    fitnesses = np.load(os.path.join(repertoire_path, "fitnesses.npy"))
    
    genotypes_array = np.load(os.path.join(repertoire_path, "genotypes.npy"), allow_pickle=True)
    
    if genotypes_array.ndim == 0:
        genotypes = genotypes_array.item()
        genotypes_are_pytree = True
    else:
        genotypes = genotypes_array
        genotypes_are_pytree = False
    
    valid_mask = fitnesses != -np.inf
    valid_fitnesses = fitnesses[valid_mask]
    
    print(f"  Centroides: {centroids.shape}")
    print(f"  Descriptores: {descriptors.shape}")
    print(f"  Fitness: {fitnesses.shape}")
    print(f"  Genotypes: {'pytree' if genotypes_are_pytree else genotypes.shape}")
    print(f"\nStatistics:")
    print(f"  Valid solutions: {np.sum(valid_mask)} / {len(fitnesses)}")
    print(f"  Coverage: {100 * np.mean(valid_mask):.2f}%")
    if len(valid_fitnesses) > 0:
        print(f"  Max fitness: {np.max(valid_fitnesses):.4f}")
        print(f"  Average fitness: {np.mean(valid_fitnesses):.4f}")
        print(f"  Min fitness: {np.min(valid_fitnesses):.4f}")
    print(f"{'='*80}\n")
    
    return centroids, descriptors, fitnesses, genotypes, genotypes_are_pytree


def get_quadrant_agents(descriptors: np.ndarray, fitnesses: np.ndarray):
    """
    Divide the descriptor space into 4 quadrants and obtains the best and worst agent from each one.
    
    Quadrants based on:
    - OI (Descriptor 1): Optimal Innovation
    - LZ (Descriptor 2): Local Zone
    
    The quadrants are divided based on the median of each descriptor
    
    Returns:
        dict: Dictionary with information of the 8 selected agents (best and worst of each quadrant)
    """
    valid_mask = fitnesses != -np.inf
    
    oi = descriptors[:, 0]  # Descriptor 1 (OI)
    lz = descriptors[:, 1]  # Descriptor 2 (LZ)
    
    # Calculate medians only for valid agents
    oi_median = np.median(oi[valid_mask])
    lz_median = np.median(lz[valid_mask])
    
    print(f"Median OI: {oi_median:.4f}")
    print(f"Median LZ: {lz_median:.4f}")
    
    quadrants = {
        'Q1': (oi >= oi_median) & (lz < lz_median) & valid_mask,  # High OI, Low LZ
        'Q2': (oi < oi_median) & (lz < lz_median) & valid_mask,   # Low OI, Low LZ
        'Q3': (oi >= oi_median) & (lz >= lz_median) & valid_mask, # High OI, High LZ
        'Q4': (oi < oi_median) & (lz >= lz_median) & valid_mask   # Low OI, High LZ
    }
    
    selected_agents = {}
    
    print(f"\n{'='*80}")
    print(f"QUADRANT ANALYSIS (OI vs LZ)")
    print(f"Quadrants divided by medians:")
    print(f"  Q1: OI ≥ {oi_median:.3f}, LZ < {lz_median:.3f}")
    print(f"  Q2: OI < {oi_median:.3f}, LZ < {lz_median:.3f}")
    print(f"  Q3: OI ≥ {oi_median:.3f}, LZ ≥ {lz_median:.3f}")
    print(f"  Q4: OI < {oi_median:.3f}, LZ ≥ {lz_median:.3f}")
    print(f"{'='*80}\n")
    
    for q_name, q_mask in quadrants.items():
        q_indices = np.where(q_mask)[0]
        
        if len(q_indices) == 0:
            print(f"{q_name}: No valid agents")
            selected_agents[q_name] = {'best': None, 'worst': None}
            continue
        
        q_fitnesses = fitnesses[q_indices]
        best_idx_in_q = np.argmax(q_fitnesses)
        worst_idx_in_q = np.argmin(q_fitnesses)
        
        best_agent_idx = q_indices[best_idx_in_q]
        worst_agent_idx = q_indices[worst_idx_in_q]
        
        selected_agents[q_name] = {
            'best': {
                'index': best_agent_idx,
                'fitness': fitnesses[best_agent_idx],
                'descriptor': descriptors[best_agent_idx]
            },
            'worst': {
                'index': worst_agent_idx,
                'fitness': fitnesses[worst_agent_idx],
                'descriptor': descriptors[worst_agent_idx]
            },
            'count': len(q_indices)
        }
        
        print(f"{q_name} ({len(q_indices)} agents):")
        print(f"  Best:  Index={best_agent_idx:4d}, Fitness={fitnesses[best_agent_idx]:.4f}, "
              f"OI={descriptors[best_agent_idx][0]:6.3f}, LZ={descriptors[best_agent_idx][1]:.3f}")
        print(f"  Worst: Index={worst_agent_idx:4d}, Fitness={fitnesses[worst_agent_idx]:.4f}, "
              f"OI={descriptors[worst_agent_idx][0]:6.3f}, LZ={descriptors[worst_agent_idx][1]:.3f}")
        print()
    
    print(f"{'='*80}\n")
    
    return selected_agents


def list_all_agents(descriptors: np.ndarray, fitnesses: np.ndarray, sort_by_fitness: bool = True):
    """Lists all agents with their descriptors and fitness."""
    print(f"\n{'='*80}")
    print(f"LIST OF ALL AGENTS")
    print(f"{'='*80}\n")
    
    valid_mask = fitnesses != -np.inf
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print("No valid agents in the repertoire.")
        return []
    
    if sort_by_fitness:
        sorted_order = np.argsort(fitnesses[valid_indices])[::-1]
        valid_indices = valid_indices[sorted_order]
    
    print(f"{'Rank':<6} {'Index':<8} {'Fitness':<12} {'Descriptor 1':<14} {'Descriptor 2':<14}")
    print(f"{'-'*6} {'-'*8} {'-'*12} {'-'*14} {'-'*14}")
    
    for rank, idx in enumerate(valid_indices, 1):
        fit = fitnesses[idx]
        desc = descriptors[idx]
        
        if rank <= 10:
            color = '\033[92m'
        elif rank <= 50:
            color = '\033[93m'
        else:
            color = '\033[0m'

        print(f"{color}{rank:<6} {idx:<8} {fit:<12.4f} {desc[0]:<14.4f} {desc[1]:<14.4f}\033[0m")
    
    print(f"\n{'='*80}")
    print(f"Total valid agents: {len(valid_indices)}")
    print(f"{'='*80}\n")
    
    return valid_indices


def extract_policy_params(genotypes, index: int, genotypes_are_pytree: bool, 
                         policy_network, env, seed: int):
    """Extracts the policy parameters for a given index."""
    if genotypes_are_pytree:
        return jax.tree_map(lambda x: x[index], genotypes)
    else:
        random_key = jax.random.PRNGKey(seed + index)
        fake_obs = jnp.zeros(shape=(env.observation_size,))
        template_params = policy_network.init(random_key, fake_obs)
        
        flat_template, tree_def = jax.tree_util.tree_flatten(template_params)
        flat_genotype = genotypes[index]
        
        split_genotype = []
        offset = 0
        for param in flat_template:
            size = param.size
            split_genotype.append(jnp.array(flat_genotype[offset:offset+size]).reshape(param.shape))
            offset += size
        
        return jax.tree_util.tree_unflatten(tree_def, split_genotype)

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

def rollout_policy(env, policy_network, policy_params: Params, 
                  seed: int, episode_length: int) -> Tuple[List[np.ndarray], List[float]]:
    """Executes a policy and collects the trajectory."""
    random_key = jax.random.PRNGKey(seed)
    reset_key, random_key = jax.random.split(random_key)
    env_state = env.reset(reset_key)
    
    positions = []
    rewards = []
    
    if hasattr(env_state, 'info') and 'pos' in env_state.info:
        positions.append(np.array(env_state.info['pos']))
    
    # Rollout
    for step in range(episode_length):
        action = policy_network.apply(policy_params, env_state.obs)
        env_state = env.step(env_state, action)
        #jax.debug.print("{x}", x=_generate_observation(env, env_state.info['pos']))
        
        if hasattr(env_state, 'info') and 'pos' in env_state.info:
            positions.append(np.array(env_state.info['pos']))
        rewards.append(float(env_state.reward))
    
    return positions, rewards


def create_function_surface(function_type: str, x_range, y_range, minval: float, maxval: float, 
                           n_dimensions: int, episode_length: int, z_fixed=0.0):
    """Creates the surface of the objective function.
    
    For sphere: We show the classic valley (visual minimization) but with colors
    that reflect that the stored fitness is high=better (maximization in rewards).
    """
    X, Y = np.meshgrid(x_range, y_range)
    
    if function_type == 'sphere':
        Z_classic_sphere = (X + minval * 0.4)**2 + (Y + minval * 0.4)**2 + z_fixed**2

        max_z = ((maxval + minval * 0.4)**2 + (maxval + minval * 0.4)**2 + z_fixed**2) * n_dimensions
        min_z = 0  

        Z = 100 - (Z_classic_sphere / max_z * 100)
        
    else:  # rastrigin
        A = 10
        n = n_dimensions
        Z_raw = -(A * n + (X**2 - A * np.cos(2 * np.pi * X)) + 
                  (Y**2 - A * np.cos(2 * np.pi * Y)) + z_fixed**2)
        
        rastrigin_scoring = lambda x: -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
        worst_fitness_step = rastrigin_scoring(np.ones(n_dimensions) * maxval)
        best_fitness_step = rastrigin_scoring(np.zeros(n_dimensions))

        Z = (Z_raw - worst_fitness_step) * 100 / (best_fitness_step - worst_fitness_step)
    
    return X, Y, Z


def visualize_8_trajectories(all_trajectories: dict, function_type: str, n_dimensions: int, 
                            minval: float, maxval: float):
    """
    Visualizes 8 trajectories simultaneously (best and worst from each quadrant).
    
    Args:
        all_trajectories: Dict with trajectories per quadrant
        function_type: 'sphere' or 'rastrigin'
        n_dimensions: Number of dimensions
        minval: Minimum value of the space
        maxval: Maximum value of the space
    """
    if n_dimensions < 2:
        print("At least 2 dimensions are required for 3D visualization")
        return

    fig = plt.figure(figsize=(20, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Colors and styles for each quadrant
    quadrant_styles = {
        'Q1': {'color_best': '#FF0000', 'color_worst': '#FF9999', 'marker': 'o', 'name': 'Q1: OI alto, LZ bajo'},
        'Q2': {'color_best': '#0000FF', 'color_worst': '#9999FF', 'marker': 's', 'name': 'Q2: OI bajo, LZ bajo'},
        'Q3': {'color_best': '#FF00FF', 'color_worst': '#FF99FF', 'marker': '^', 'name': 'Q3: OI alto, LZ alto'},
        'Q4': {'color_best': '#00FF00', 'color_worst': '#99FF99', 'marker': 'D', 'name': 'Q4: OI bajo, LZ alto'},
    }

    z_fixed = 0.0
    if n_dimensions >= 3:
        all_z = []
        for q_name, q_data in all_trajectories.items():
            if q_data['best'] is not None:
                all_z.extend([pos[2] for pos in q_data['best']['positions']])
            if q_data['worst'] is not None:
                all_z.extend([pos[2] for pos in q_data['worst']['positions']])
        if all_z:
            z_fixed = np.mean(all_z)

    x_range = np.linspace(minval, maxval, 80)
    y_range = np.linspace(minval, maxval, 80)
    X, Y, Z = create_function_surface(function_type, x_range, y_range, minval, maxval, n_dimensions, 1, z_fixed)
    
    surf = ax.plot_surface(
        X, Y, Z,
        cmap='coolwarm',
        alpha=0.3,
        edgecolor='none',
        zorder=1
    )

    def calculate_z_trajectory(x_traj, y_traj):
        z_traj = []
        
        if function_type == 'sphere':
            max_z = ((maxval + minval * 0.4)**2 + (maxval + minval * 0.4)**2 + z_fixed**2) * n_dimensions
            
            for x, y in zip(x_traj, y_traj):
                z_classic = (x + minval * 0.4)**2 + (y + minval * 0.4)**2 + z_fixed**2
                z_normalized = 100 - (z_classic / max_z * 100)
                z_traj.append(z_normalized)
        else:  # rastrigin
            A = 10
            n = n_dimensions
            rastrigin_scoring = lambda x: -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
            worst_fitness_step = rastrigin_scoring(np.ones(n_dimensions) * maxval)
            best_fitness_step = rastrigin_scoring(np.zeros(n_dimensions))
            
            for x, y in zip(x_traj, y_traj):
                z_raw = -(A * n + (x**2 - A * np.cos(2 * np.pi * x)) + 
                         (y**2 - A * np.cos(2 * np.pi * y)) + z_fixed**2)
                z_normalized = (z_raw - worst_fitness_step) * 100 / (best_fitness_step - worst_fitness_step)
                z_traj.append(z_normalized)
        
        return np.array(z_traj)

    legend_elements = []
    trajectory_count = 0
    
    for q_name in ['Q1', 'Q2', 'Q3', 'Q4']:
        q_data = all_trajectories[q_name]
        style = quadrant_styles[q_name]
        
        if q_data['best'] is not None:
            trajectory_count += 1
            print(f"Plotting {q_name} - Best (Index {q_data['best']['index']})")
            positions = np.array(q_data['best']['positions'])
            x_traj = positions[:, 0]
            y_traj = positions[:, 1]
            z_traj = calculate_z_trajectory(x_traj, y_traj)
            
            ax.plot(
                x_traj, y_traj, z_traj,
                color=style['color_best'],
                linewidth=2.5,
                alpha=0.8,
                zorder=100
            )
            
            ax.scatter(
                x_traj, y_traj, z_traj,
                c=style['color_best'],
                s=60,
                marker=style['marker'],
                alpha=0.7,
                edgecolors='black',
                linewidths=1,
                zorder=101
            )
            
            ax.scatter(
                [x_traj[0]], [y_traj[0]], [z_traj[0]],
                c=style['color_best'],
                s=400,
                marker='*',
                edgecolors='black',
                linewidths=2,
                zorder=102
            )
            
            ax.scatter(
                [x_traj[-1]], [y_traj[-1]], [z_traj[-1]],
                c=style['color_best'],
                s=300,
                marker='X',
                edgecolors='black',
                linewidths=2,
                zorder=102
            )
        
        if q_data['worst'] is not None:
            trajectory_count += 1
            print(f"Plotting {q_name} - Worst (Index {q_data['worst']['index']})")
            positions = np.array(q_data['worst']['positions'])
            x_traj = positions[:, 0]
            y_traj = positions[:, 1]
            z_traj = calculate_z_trajectory(x_traj, y_traj)
            
            ax.plot(
                x_traj, y_traj, z_traj,
                color=style['color_worst'],
                linewidth=2,
                alpha=0.6,
                linestyle='--',
                zorder=90
            )
            
            ax.scatter(
                x_traj, y_traj, z_traj,
                c=style['color_worst'],
                s=40,
                marker=style['marker'],
                alpha=0.6,
                edgecolors='white',
                linewidths=0.5,
                zorder=91
            )
            
            ax.scatter(
                [x_traj[0]], [y_traj[0]], [z_traj[0]],
                c=style['color_worst'],
                s=250,
                marker='*',
                edgecolors='white',
                linewidths=1.5,
                zorder=92
            )
            
            ax.scatter(
                [x_traj[-1]], [y_traj[-1]], [z_traj[-1]],
                c=style['color_worst'],
                s=200,
                marker='X',
                edgecolors='white',
                linewidths=1.5,
                zorder=92
            )
        
        from matplotlib.lines import Line2D
        if q_data['best'] is not None:
            best_line = Line2D([0], [0], color=style['color_best'], linewidth=3, 
                             marker=style['marker'], markersize=8, label=f"{style['name']} - Mejor")
            legend_elements.append(best_line)
        if q_data['worst'] is not None:
            worst_line = Line2D([0], [0], color=style['color_worst'], linewidth=2, 
                              linestyle='--', marker=style['marker'], markersize=6, 
                              label=f"{style['name']} - Peor")
            legend_elements.append(worst_line)

    ax.set_xlabel('X (Dimension 1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (Dimension 2)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Normalized Fitness (0-100)', fontsize=12, fontweight='bold')
    
    print(f"\nTotal trajectories plotted: {trajectory_count}")
    
    title = (f'{trajectory_count} Trajectories: Best (solid line) and Worst (dashed line) per Quadrant\n'
            f'Function: {function_type.capitalize()} | '
            f'Markers: ○=Q1, □=Q2, △=Q3, ◇=Q4 | '
            f'★=Start, ✕=End')
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.show()


def visualize_3d(positions: List[np.ndarray], fitness: float, agent_idx: int,
                function_type: str, n_dimensions: int, minval: float, maxval: float):
    """Visualizes the trajectory in 3D."""
    if n_dimensions < 2:
        print("At least 2 dimensions are required for 3D visualization")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    positions_array = np.array(positions)
    x_traj = positions_array[:, 0]
    y_traj = positions_array[:, 1]

    z_fixed = 0.0
    if n_dimensions >= 3:
        z_fixed = np.mean(positions_array[:, 2])

    x_range = np.linspace(minval, maxval, 100)
    y_range = np.linspace(minval, maxval, 100)
    X, Y, Z = create_function_surface(function_type, x_range, y_range, minval, maxval, n_dimensions, 1, z_fixed)
    
    surf = ax.plot_surface(
        X, Y, Z,
        cmap='coolwarm',
        alpha=0.6,
        edgecolor='none'
    )
    
    z_traj = []
    
    if function_type == 'sphere':
        max_z = ((maxval + minval * 0.4)**2 + (maxval + minval * 0.4)**2 + z_fixed**2) * n_dimensions
        
        for x, y in zip(x_traj, y_traj):
            z_classic = (x + minval * 0.4)**2 + (y + minval * 0.4)**2 + z_fixed**2
            z_normalized = 100 - (z_classic / max_z * 100)
            z_traj.append(z_normalized)
    else:  # rastrigin
        A = 10
        n = n_dimensions
        rastrigin_scoring = lambda x: -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
        worst_fitness_step = rastrigin_scoring(np.ones(n_dimensions) * maxval)
        best_fitness_step = rastrigin_scoring(np.zeros(n_dimensions))
        
        for x, y in zip(x_traj, y_traj):
            z_raw = -(A * n + (x**2 - A * np.cos(2 * np.pi * x)) + 
                     (y**2 - A * np.cos(2 * np.pi * y)) + z_fixed**2)
            z_normalized = (z_raw - worst_fitness_step) * 100 / (best_fitness_step - worst_fitness_step)
            z_traj.append(z_normalized)
    
    z_traj = np.array(z_traj)

    ax.plot(
        x_traj, y_traj, z_traj,
        'r-',
        linewidth=2,
        alpha=0.5,
        label=f'Trajectory (Agent {agent_idx})',
        zorder=100
    )
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(x_traj)))
    
    overlap_threshold = 0.15
    
    for i in range(len(x_traj)):
        is_overlapping = False
        for j in range(i):
            dist = np.sqrt((x_traj[i] - x_traj[j])**2 + 
                          (y_traj[i] - y_traj[j])**2 + 
                          (z_traj[i] - z_traj[j])**2)
            if dist < overlap_threshold:
                is_overlapping = True
                break
        
        size = 30 + (i / len(x_traj)) * 50
        
        marker = 'X' if is_overlapping else 'o'
        linewidth = 1.5 if is_overlapping else 0.5 
        
        ax.scatter(
            [x_traj[i]], [y_traj[i]], [z_traj[i]],
            c=[colors[i]],
            s=size,
            marker=marker,
            zorder=100 + i,
            alpha=0.7,
            edgecolors='black',
            linewidths=linewidth
        )

    ax.scatter(
        [x_traj[0]], [y_traj[0]], [z_traj[0]],
        c='lime',
        s=300,
        marker='o',
        label='Inicio',
        zorder=200,
        edgecolors='black',
        linewidths=3
    )
    
    ax.scatter(
        [x_traj[-1]], [y_traj[-1]], [z_traj[-1]],
        c='red',
        s=300,
        marker='X',
        label='Fin',
        zorder=200,
        edgecolors='black',
        linewidths=3
    )
    
    ax.set_xlabel('X (Dimension 1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (Dimension 2)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Normalized Fitness (0-100)', fontsize=12, fontweight='bold')
    
    title = (f'Trajectory of Agent {agent_idx} ({len(positions)} steps)\n'
            f'Function: {function_type.capitalize()}, '
            f'Normalized Fitness: {fitness:.2f}/100\n'
            f'Color: Blue (start) → Red (end) | X = Overlapping points')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.legend(loc='upper left')
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='List and visualize specific agents from repertoires'
    )
    parser.add_argument(
        '--repertoire-path',
        type=str,
        required=True,
        help='Path to the repertoire directory'
    )
    parser.add_argument(
        '--function',
        type=str,
        choices=['sphere', 'rastrigin'],
        required=True,
        help='Function type (sphere or rastrigin)'
    )
    parser.add_argument(
        '--agent-index',
        type=int,
        default=None,
        help='Index of the agent to visualize (optional; will be asked if not provided)'
    )
    parser.add_argument(
        '--n-dimensions',
        type=int,
        default=3,
        help='Number of environment dimensions (default: 3)'
    )
    parser.add_argument(
        '--episode-length',
        type=int,
        default=30,
        help='Episode length (default: 30)'
    )
    parser.add_argument(
        '--minval',
        type=float,
        default=-5.12,
        help='Minimum value of the space (default: -5.12)'
    )
    parser.add_argument(
        '--maxval',
        type=float,
        default=5.12,
        help='Maximum value of the space (default: 5.12)'
    )
    parser.add_argument(
        '--policy-hidden-sizes',
        type=int,
        nargs='+',
        default=[128, 128],
        help='Sizes of hidden layers (default: 128 128)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--sort-by-fitness',
        action='store_true',
        default=True,
        help='Sort agents by fitness (default: True)'
    )
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list agents without visualizing'
    )
    parser.add_argument(
        '--quadrant-view',
        action='store_true',
        help='Visualize 8 trajectories (best and worst of each OI/LZ quadrant)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.repertoire_path):
        print(f"Error: No se encontró el repertorio en {args.repertoire_path}")
        return
    
    centroids, descriptors, fitnesses, genotypes, genotypes_are_pytree = load_repertoire(args.repertoire_path)
    
    valid_indices = list_all_agents(descriptors, fitnesses, args.sort_by_fitness)
    
    if len(valid_indices) == 0:
        return
    
    if args.list_only:
        return
    
    if args.quadrant_view:
        print(f"\n{'='*80}")
        print(f"QUADRANT VIEW MODE")
        print(f"{'='*80}\n")
        
        selected_agents = get_quadrant_agents(descriptors, fitnesses)
        
        # Configure environment
        env_name = f"{args.function}_oi"
        env = create(
            env_name,
            n_dimensions=args.n_dimensions,
            episode_length=args.episode_length,
            minval=args.minval,
            maxval=args.maxval,
            fixed_init_state=False,
            qdax_wrappers_kwargs=[{
                "episode_length": args.episode_length
            }]
        )
        
        policy_layer_sizes = tuple(args.policy_hidden_sizes) + (env.action_size,)
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )
        
        all_trajectories = {}
        
        for q_name in ['Q1', 'Q2', 'Q3', 'Q4']:
            q_data = selected_agents[q_name]
            all_trajectories[q_name] = {'best': None, 'worst': None}
            
            # Best agent
            if q_data['best'] is not None:
                agent_idx = q_data['best']['index']
                print(f"Generating trajectory for {q_name} - Best (Index {agent_idx})...")
                
                policy_params = extract_policy_params(
                    genotypes, agent_idx, genotypes_are_pytree,
                    policy_network, env, args.seed
                )
                
                positions, rewards = rollout_policy(
                    env, policy_network, policy_params,
                    args.seed, args.episode_length
                )
                
                all_trajectories[q_name]['best'] = {
                    'positions': positions,
                    'rewards': rewards,
                    'index': agent_idx,
                    'fitness': q_data['best']['fitness'],
                    'descriptor': q_data['best']['descriptor']
                }

            if q_data['worst'] is not None:
                agent_idx = q_data['worst']['index']
                print(f"Generating trajectory for {q_name} - Worst (Index {agent_idx})...")
                
                policy_params = extract_policy_params(
                    genotypes, agent_idx, genotypes_are_pytree,
                    policy_network, env, args.seed
                )
                
                positions, rewards = rollout_policy(
                    env, policy_network, policy_params,
                    args.seed, args.episode_length
                )
                
                all_trajectories[q_name]['worst'] = {
                    'positions': positions,
                    'rewards': rewards,
                    'index': agent_idx,
                    'fitness': q_data['worst']['fitness'],
                    'descriptor': q_data['worst']['descriptor']
                }
        
        print("\n✓ All trajectories generated. Creating visualization...\n")
        
        # Visualize the 8 trajectories
        visualize_8_trajectories(
            all_trajectories, args.function, args.n_dimensions,
            args.minval, args.maxval
        )
        
        return
    
    if args.agent_index is not None:
        agent_idx = args.agent_index
        if fitnesses[agent_idx] == -np.inf:
            print(f"\nError: Agent {agent_idx} does not have a valid solution (fitness = -inf)")
            return
    else:
        while True:
            try:
                user_input = input(f"\nWhich agent do you want to visualize? (Index 0-{len(fitnesses)-1}, or 'q' to quit): ")
                if user_input.lower() == 'q':
                    print("Exiting...")
                    return
                
                agent_idx = int(user_input)
                if agent_idx < 0 or agent_idx >= len(fitnesses):
                    print(f"Error: Index out of range. Must be between 0 and {len(fitnesses)-1}")
                    continue
                
                if fitnesses[agent_idx] == -np.inf:
                    print(f"Error: Agent {agent_idx} does not have a valid solution (fitness = -inf)")
                    continue
                
                break
            except ValueError:
                print("Error: Please enter a valid number or 'q' to quit")
                continue
    
    print(f"\n{'='*80}")
    print(f"Setting up environment and generating trajectory...")
    print(f"{'='*80}\n")
    
    env_name = f"{args.function}_oi"
    env = create(
        env_name,
        n_dimensions=args.n_dimensions,
        episode_length=args.episode_length,
        minval=args.minval,
        maxval=args.maxval,
        fixed_init_state=False,
        qdax_wrappers_kwargs=[{
            "episode_length": args.episode_length
        }]
    )
    
    policy_layer_sizes = tuple(args.policy_hidden_sizes) + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    
    policy_params = extract_policy_params(
        genotypes, agent_idx, genotypes_are_pytree,
        policy_network, env, args.seed
    )
    
    positions, rewards = rollout_policy(
        env, policy_network, policy_params,
        args.seed, args.episode_length
    )
    
    print(f"\nAgente {agent_idx}:")
    print(f"  Fitness: {fitnesses[agent_idx]:.4f}")
    print(f"  Descriptor: [{descriptors[agent_idx][0]:.4f}, {descriptors[agent_idx][1]:.4f}]")
    print(f"  Steps: {len(positions)}")
    print(f"  Total reward: {sum(rewards):.4f}")
    print(f"  Average reward: {np.mean(rewards):.4f}\n")
    
    # Visualize
    visualize_3d(
        positions, fitnesses[agent_idx], agent_idx,
        args.function, args.n_dimensions, args.minval, args.maxval
    )


if __name__ == "__main__":
    main()
