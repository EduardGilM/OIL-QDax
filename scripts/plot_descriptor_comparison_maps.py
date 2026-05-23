#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial import Voronoi


EMITTERS = ("mapelites", "pga", "dcrlme")
LABELS = {"mapelites": "MAP-Elites", "pga": "PGA-ME", "dcrlme": "DCRL-ME"}


def get_voronoi_finite_polygons_2d(points: np.ndarray, radius: float | None = None):
    vor = Voronoi(points)
    if radius is None:
        radius = float(np.ptp(points, axis=0).max() * 2)
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = points.mean(axis=0)
    ridges = {}
    for (point_a, point_b), (vertex_a, vertex_b) in zip(
        vor.ridge_points, vor.ridge_vertices
    ):
        ridges.setdefault(point_a, []).append((point_b, vertex_a, vertex_b))
        ridges.setdefault(point_b, []).append((point_a, vertex_a, vertex_b))
    for point_idx, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(vertex >= 0 for vertex in vertices):
            new_regions.append(vertices)
            continue
        region = [vertex for vertex in vertices if vertex >= 0]
        for neighbor, vertex_a, vertex_b in ridges[point_idx]:
            if vertex_b < 0:
                vertex_a, vertex_b = vertex_b, vertex_a
            if vertex_a >= 0:
                continue
            tangent = points[neighbor] - points[point_idx]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = points[[point_idx, neighbor]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            new_vertices.append((vor.vertices[vertex_b] + direction * radius).tolist())
            region.append(len(new_vertices) - 1)
        polygon = np.asarray([new_vertices[vertex] for vertex in region])
        centroid = polygon.mean(axis=0)
        angles = np.arctan2(polygon[:, 1] - centroid[1], polygon[:, 0] - centroid[0])
        new_regions.append([vertex for _, vertex in sorted(zip(angles, region))])
    return new_regions, np.asarray(new_vertices)


def _latest_summary(root: Path, env: str, emitter: str):
    files = sorted((root / env / emitter).glob("*/summary.json"))
    return files[-1] if files else None


def _load_run(summary_path: Path):
    summary = json.loads(summary_path.read_text())
    run_dir = Path(summary["files"]["repertoire_dir"])
    return {
        "summary": summary,
        "centroids": np.load(run_dir / "centroids.npy"),
        "descriptors": np.load(run_dir / "descriptors.npy"),
        "fitnesses": np.load(run_dir / "fitnesses.npy"),
    }


def _plot_voronoi(ax, run, dims, xlim, ylim, xlabel, ylabel, title, norm, cmap):
    centroids = run["centroids"][:, dims]
    descriptors = run["descriptors"][:, dims]
    fitnesses = run["fitnesses"]
    empty = ~np.isfinite(fitnesses)
    regions, vertices = get_voronoi_finite_polygons_2d(centroids)

    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=0.4)
    for idx, fitness in enumerate(fitnesses):
        if not empty[idx]:
            polygon = vertices[regions[idx]]
            ax.fill(*zip(*polygon), alpha=0.85, color=cmap(norm(fitness)))
    ax.scatter(
        descriptors[~empty, 0],
        descriptors[~empty, 1],
        c=fitnesses[~empty],
        cmap=cmap,
        norm=norm,
        s=3,
        zorder=5,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_box_aspect(1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _grid_shape(num_descriptors: int) -> tuple[int, ...]:
    if num_descriptors == 4:
        return (8, 8, 4, 4)
    return tuple([4] * num_descriptors)


def _project_1d(integer_coordinates, bases) -> int:
    coordinate = 0
    for value, base in zip(integer_coordinates, bases):
        coordinate = coordinate * base + value
    return int(coordinate)


def _project_2d(integer_coordinates, bases) -> tuple[int, int]:
    return (
        _project_1d(integer_coordinates[::2], bases[::2]),
        _project_1d(integer_coordinates[1::2], bases[1::2]),
    )


def _plot_multidim_grid(ax, run, minval, maxval, title, norm, cmap):
    grid_shape = _grid_shape(run["descriptors"].shape[1])
    resolutions = np.asarray(grid_shape)
    size_x = int(np.prod(resolutions[0::2]))
    size_y = int(np.prod(resolutions[1::2]))
    grid = np.full((size_x, size_y), np.nan)
    valid = np.isfinite(run["fitnesses"])
    descriptors = np.floor(
        resolutions
        * (run["descriptors"][valid] - minval)
        / (maxval - minval + 1e-12)
    ).astype(np.int32)
    descriptors = np.clip(descriptors, 0, resolutions - 1)

    for descriptor, fitness in zip(descriptors, run["fitnesses"][valid]):
        x, y = _project_2d(descriptor, grid_shape)
        if np.isnan(grid[x, y]) or fitness > grid[x, y]:
            grid[x, y] = fitness

    ax.imshow(grid.T, origin="lower", aspect="equal", norm=norm, cmap=cmap)
    ax.set_xlabel("Behavior Dimension 1")
    ax.set_ylabel("Behavior Dimension 2")
    ax.set_title(title)
    ax.set_box_aspect(1)
    major_x_step = int(np.prod(grid_shape[2::2]))
    major_y_step = int(np.prod(grid_shape[3::2]))
    minor_x_step = int(np.prod(grid_shape[4::2]))
    minor_y_step = int(np.prod(grid_shape[5::2]))
    major_x = np.arange(0, size_x + 1, major_x_step) - 0.5
    major_y = np.arange(0, size_y + 1, major_y_step) - 0.5
    ax.set_xticks(major_x)
    ax.set_yticks(major_y)
    ax.set_xticks(np.arange(0, size_x + 1, minor_x_step) - 0.5, minor=True)
    ax.set_yticks(np.arange(0, size_y + 1, minor_y_step) - 0.5, minor=True)
    labels_x = [
        f"{value:.2f}".rstrip("0").rstrip(".")
        for value in np.linspace(minval[0], maxval[0], len(major_x))
    ]
    labels_y = [
        f"{value:.2f}".rstrip("0").rstrip(".")
        for value in np.linspace(minval[1], maxval[1], len(major_y))
    ]
    ax.set_xticklabels(labels_x)
    ax.set_yticklabels(labels_y)
    ax.grid(which="minor", alpha=1.0, color="#000000", linewidth=0.3)
    ax.grid(which="major", alpha=1.0, color="#000000", linewidth=1.2)


def _plot_foot_contact(ax, run, title, norm, cmap):
    descriptor_dim = run["descriptors"].shape[1]
    if descriptor_dim > 2:
        minval = np.zeros(descriptor_dim)
        maxval = np.ones(descriptor_dim)
        _plot_multidim_grid(ax, run, minval, maxval, title, norm, cmap)
        return
    _plot_voronoi(
        ax,
        run,
        slice(0, 2),
        (0.0, 1.0),
        (0.0, 1.0),
        "Foot Contact 1",
        "Foot Contact 2",
        title,
        norm,
        cmap,
    )


def _complete_envs(foot_root: Path, oil_root: Path):
    envs = sorted({path.name for path in foot_root.iterdir() if path.is_dir()})
    complete = []
    for env in envs:
        if all(_latest_summary(foot_root, env, e) for e in EMITTERS) and all(
            _latest_summary(oil_root, env, e) for e in EMITTERS
        ):
            complete.append(env)
    return complete


def main():
    foot_root = Path("repertoires/foot_contact")
    oil_root = Path("repertoires/oil_posthoc")
    out_dir = Path("repertoires/comparison_maps")
    out_dir.mkdir(parents=True, exist_ok=True)

    for env in _complete_envs(foot_root, oil_root):
        foot_runs = [_load_run(_latest_summary(foot_root, env, e)) for e in EMITTERS]
        oil_runs = [_load_run(_latest_summary(oil_root, env, e)) for e in EMITTERS]
        all_fitnesses = np.concatenate(
            [run["fitnesses"][np.isfinite(run["fitnesses"])] for run in foot_runs + oil_runs]
        )
        norm = Normalize(vmin=float(all_fitnesses.min()), vmax=float(all_fitnesses.max()))
        cmap = plt.cm.viridis
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), constrained_layout=True)

        for col, emitter in enumerate(EMITTERS):
            _plot_foot_contact(
                axes[0, col],
                foot_runs[col],
                f"{LABELS[emitter]} baseline",
                norm,
                cmap,
            )
            _plot_voronoi(
                axes[1, col],
                oil_runs[col],
                slice(0, 2),
                (0.0, 1.0),
                (-1.0, 1.0),
                "LZ",
                "O-Information",
                f"{LABELS[emitter]} OIL",
                norm,
                cmap,
            )

        fig.suptitle(
            f"{env}: shared fitness scale [{norm.vmin:.2f}, {norm.vmax:.2f}]",
            fontsize=14,
        )
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=axes.ravel().tolist(),
            shrink=0.78,
            label="Fitness",
        )
        fig.savefig(out_dir / f"{env}_footcontact_vs_oil.png", dpi=250, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
