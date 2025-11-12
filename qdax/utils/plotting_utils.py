import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Union
from matplotlib.colors import Normalize
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import Voronoi

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire


def compute_coverage(repertoire: MapElitesRepertoire) -> float:
    """Compute the coverage of a repertoire (percentage of filled cells).

    Args:
        repertoire: A MAP-Elites repertoire

    Returns:
        The coverage percentage
    """
    filled_cells = jnp.sum(repertoire.fitnesses != -jnp.inf)
    total_cells = repertoire.fitnesses.size
    return (filled_cells / total_cells) * 100.0


def compute_metrics_from_repertoire(
    repertoire: MapElitesRepertoire
) -> Dict[str, float]:
    """Compute the main QD metrics from a repertoire.

    Args:
        repertoire: A MAP-Elites repertoire

    Returns:
        A dictionary with the computed metrics
    """
    # Extract valid fitnesses (non -inf values)
    valid_fitnesses = repertoire.fitnesses[repertoire.fitnesses != -jnp.inf]
    
    metrics = {
        "coverage": compute_coverage(repertoire),
        "max_fitness": jnp.max(valid_fitnesses) if valid_fitnesses.size > 0 else 0.0,
        "mean_fitness": jnp.mean(valid_fitnesses) if valid_fitnesses.size > 0 else 0.0,
        "qd_score": jnp.sum(valid_fitnesses) if valid_fitnesses.size > 0 else 0.0,
    }
    
    return metrics


def calculate_oi_metrics(repertoire: MapElitesRepertoire) -> Dict[str, float]:
    """Calculate metrics specific for OIL experiments."""
    valid_mask = repertoire.fitnesses != -jnp.inf

    valid_descriptors = repertoire.descriptors[valid_mask]

    lz76_values = valid_descriptors[:, 0] if valid_descriptors.shape[0] > 0 else jnp.array([])
    o_info_values = valid_descriptors[:, 1] if valid_descriptors.shape[0] > 0 else jnp.array([])

    metrics = {
        "mean_lz76": jnp.mean(lz76_values) if lz76_values.size > 0 else 0.0,
        "max_lz76": jnp.max(lz76_values) if lz76_values.size > 0 else 0.0,
        "mean_o_info": jnp.mean(o_info_values) if o_info_values.size > 0 else 0.0,
        "max_o_info": jnp.max(o_info_values) if o_info_values.size > 0 else 0.0,
        "min_o_info": jnp.min(o_info_values) if o_info_values.size > 0 else 0.0,
    }
    
    return metrics


def prepare_metrics_for_plotting(
    metrics: Dict[str, Union[List, jnp.ndarray]], 
    repertoire: Optional[MapElitesRepertoire] = None,
    env_steps: jnp.ndarray = None
) -> Dict[str, jnp.ndarray]:
    """Prepare metrics for plotting by ensuring they have the right format."""
    processed_metrics = dict(metrics)

    if repertoire is not None:
        final_metrics = compute_metrics_from_repertoire(repertoire)
        oi_metrics = calculate_oi_metrics(repertoire)

        final_metrics.update(oi_metrics)
        
        for key, value in final_metrics.items():
            if key not in processed_metrics:
                processed_metrics[key] = jnp.array([value])

    if env_steps is not None:
        for key in processed_metrics:
            if not isinstance(processed_metrics[key], (list, np.ndarray, jnp.ndarray)):
                processed_metrics[key] = jnp.array([processed_metrics[key]])
                
            if len(processed_metrics[key]) == 1 and len(env_steps) > 1:
                processed_metrics[key] = jnp.full_like(env_steps, processed_metrics[key][0])
    
    return processed_metrics


def get_voronoi_finite_polygons_2d(
    centroids: np.ndarray, radius: Optional[float] = None
) -> Tuple[List, np.ndarray]:
    """Reconstruct infinite voronoi regions in a 2D diagram to finite regions."""
    voronoi_diagram = Voronoi(centroids)
    if voronoi_diagram.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = voronoi_diagram.vertices.tolist()

    center = voronoi_diagram.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(voronoi_diagram.points, axis=0).max()

    # Construct a map containing all ridges for a given point
    all_ridges: Dict[jnp.ndarray, jnp.ndarray] = {}
    for (p1, p2), (v1, v2) in zip(
        voronoi_diagram.ridge_points, voronoi_diagram.ridge_vertices
    ):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(voronoi_diagram.point_region):
        vertices = voronoi_diagram.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = voronoi_diagram.points[p2] - voronoi_diagram.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = voronoi_diagram.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = voronoi_diagram.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_2d_map_elites_repertoire(
    repertoire: MapElitesRepertoire,
    ax: Optional[plt.Axes] = None,
    min_bd: Union[float, Tuple[float, float], List[float], np.ndarray] = 0.0,
    max_bd: Union[float, Tuple[float, float], List[float], np.ndarray] = 1.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fitness_measure: str = "fitness",
    cmap: str = "viridis",
    title: str = "MAP-Elites Archive",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    use_voronoi: bool = True,
    show_stats: bool = False,
) -> Tuple[Optional[Figure], Axes]:
    """
    Plot a 2D visualization of a MAP-Elites repertoire with LZ76 and O-Information as descriptors.
    
    Args:
        repertoire: The repertoire to visualize
        ax: Optional matplotlib axes
        min_bd: Minimum behavior descriptor values [min_x, min_y] or single float for both
        max_bd: Maximum behavior descriptor values [max_x, max_y] or single float for both
        vmin: Minimum fitness for colormap
        vmax: Maximum fitness for colormap
        fitness_measure: Measure to use for color ("fitness" or "density")
        cmap: Colormap name
        title: Plot title
        xlim: Optional limits for x-axis (overrides min_bd/max_bd for x)
        ylim: Optional limits for y-axis (overrides min_bd/max_bd for y)
        use_voronoi: Whether to use Voronoi tessellation (like the original plot function)
        show_stats: Whether to show statistics on the plot
        
    Returns:
        The matplotlib figure and axes
    """
    # Create the figure if not provided
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="white", edgecolor="white")
    
    # Extract valid solutions from the repertoire
    grid_empty = repertoire.fitnesses == -jnp.inf
    valid_mask = ~grid_empty
    
    # Set default vmin/vmax based on valid fitnesses
    fitnesses = repertoire.fitnesses
    if vmin is None:
        vmin = float(jnp.min(fitnesses[valid_mask])) if jnp.any(valid_mask) else 0.0
    if vmax is None:
        vmax = float(jnp.max(fitnesses[valid_mask])) if jnp.any(valid_mask) else 1.0
    
    # Set color map and normalization
    my_cmap = cm.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Set plot parameters
    font_size = 12
    params = {
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
    }
    mpl.rcParams.update(params)
    
    # Process min_bd and max_bd to support separate X and Y axis limits
    if isinstance(min_bd, (float, int)):
        min_bd = [min_bd, min_bd]
    if isinstance(max_bd, (float, int)):
        max_bd = [max_bd, max_bd]
    
    # Set axis limits
    if xlim is None:
        xlim = (min_bd[0], max_bd[0])
    if ylim is None:
        ylim = (min_bd[1], max_bd[1])
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #ax.set(adjustable="box", aspect="equal")
    ax.set_box_aspect(1)
    # Check if any valid solutions exist
    if not jnp.any(valid_mask):
        ax.text(0.5, 0.5, "No valid solutions in repertoire", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_xlabel("LZ76 Complexity")
        ax.set_ylabel("O-Information")
        ax.set_title(title)
        return fig, ax
    
    # Extract centroids and valid descriptors
    centroids = repertoire.centroids
    valid_fitnesses = fitnesses[valid_mask]
    valid_descriptors = repertoire.descriptors[valid_mask]
    
    # Voronoi visualization
    if use_voronoi and len(centroids) > 3:  # Need at least 4 points for Voronoi
        try:
            # Create the regions and vertices from centroids
            regions, vertices = get_voronoi_finite_polygons_2d(centroids)
            
            # Fill the plot with contours (empty cells)
            for region in regions:
                polygon = vertices[region]
                ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)
            
            # Fill the plot with colors (filled cells)
            for idx, fitness in enumerate(fitnesses):
                if fitness > -jnp.inf:
                    region = regions[idx]
                    polygon = vertices[region]
                    ax.fill(*zip(*polygon), alpha=0.8, color=my_cmap(norm(fitness)))
        except Exception as e:
            print(f"Warning: Could not create Voronoi diagram: {e}")
            use_voronoi = False
    
    # Add scatter points for the valid solutions
    if valid_descriptors.shape[0] > 0:
        scatter = ax.scatter(
            valid_descriptors[:, 0],
            valid_descriptors[:, 1],
            c=valid_fitnesses,
            cmap=my_cmap,
            s=20 if use_voronoi else 50,
            zorder=5 if use_voronoi else 0,
            alpha=0.7,
            edgecolors='none' if use_voronoi else 'black',
            norm=norm
        )
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap),
        cax=cax
    )
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label('Fitness', fontsize=font_size)
    
    # Add labels and title
    ax.set_xlabel("LZ76 Complexity", fontsize=font_size)
    ax.set_ylabel("O-Information", fontsize=font_size)
    ax.set_title(title, fontsize=font_size+2)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add stats to the plot
    if show_stats and jnp.any(valid_mask):
        stats_text = (
            f"Coverage: {compute_coverage(repertoire):.1f}%\n"
            f"Max Fitness: {jnp.max(valid_fitnesses):.2f}\n"
            f"Mean Fitness: {jnp.mean(valid_fitnesses):.2f}\n"
            f"Solutions: {jnp.sum(valid_mask)}\n"
            f"Max LZ76: {jnp.max(valid_descriptors[:, 0]):.2f}\n"
            f"Max O-Info: {jnp.max(valid_descriptors[:, 1]):.2f}"
        )
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=font_size-1,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    return fig, ax


def plot_oi_map_elites_results(
    env_steps: jnp.ndarray,
    metrics: Dict[str, jnp.ndarray],
    repertoire: MapElitesRepertoire,
    min_bd: Union[float, Tuple[float, float], List[float], np.ndarray] = 0.0,
    max_bd: Union[float, Tuple[float, float], List[float], np.ndarray] = 1.0,
    figsize: Tuple[int, int] = (20, 10)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot the results of a MAP-Elites OI experiment with LZ76 and O-Information descriptors.
    
    Args:
        env_steps: Array of environment steps
        metrics: Dictionary of metrics to plot
        repertoire: The MAP-Elites repertoire
        min_bd: Minimum behavior descriptor value
        max_bd: Maximum behavior descriptor value
        figsize: Figure size
        
    Returns:
        Figure and axes with plots
    """
    all_metrics = prepare_metrics_for_plotting(metrics, repertoire, env_steps)

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    if "coverage" in all_metrics:
        axes[0].plot(env_steps, all_metrics["coverage"])
        axes[0].set_xlabel("Environment Steps")
        axes[0].set_ylabel("Coverage (%)")
        axes[0].set_title("Coverage")
    
    if "max_fitness" in all_metrics:
        axes[1].plot(env_steps, all_metrics["max_fitness"])
        axes[1].set_xlabel("Environment Steps")
        axes[1].set_ylabel("Max Fitness")  
        axes[1].set_title("Max Fitness")
    
    if "qd_score" in all_metrics:
        axes[2].plot(env_steps, all_metrics["qd_score"])
        axes[2].set_xlabel("Environment Steps")
        axes[2].set_ylabel("QD Score")
        axes[2].set_title("QD Score")
    
    plot_2d_map_elites_repertoire(
        repertoire=repertoire,
        ax=axes[5],
        min_bd=min_bd,
        max_bd=max_bd,
        fitness_measure="fitness",
    )
    
    plt.tight_layout()
    return fig, axes