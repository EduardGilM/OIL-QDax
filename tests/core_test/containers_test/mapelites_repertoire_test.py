import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
)
from qdax.custom_types import ExtraScores


def test_mapelites_repertoire() -> None:

    batch_size = 2
    genotype_size = 12
    num_centroids = 4
    grid_shape = (2, 2)

    # get num descriptors from grid shape
    num_descriptors = len(grid_shape)

    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=0.0,
        maxval=1.0,
    )

    expected_centroids = jnp.array(
        [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ]
    )

    assert jnp.allclose(centroids, expected_centroids, atol=1e-6).item()

    # create an instance
    repertoire = MapElitesRepertoire(
        genotypes=jnp.zeros(shape=(num_centroids, genotype_size)),
        fitnesses=jnp.ones(shape=(num_centroids,)) * (-jnp.inf),
        descriptors=jnp.zeros(shape=(num_centroids, num_descriptors)),
        centroids=centroids,
    )

    # create fake genotypes and scores to add
    fake_genotypes = jnp.ones(shape=(batch_size, genotype_size))
    fake_fitnesses = jnp.zeros(shape=(batch_size,))
    fake_descriptors = jnp.array([[0.1, 0.1], [0.9, 0.9]])
    fake_extra_scores: ExtraScores = {}

    # do an addition
    repertoire = repertoire.add(
        fake_genotypes, fake_descriptors, fake_fitnesses, fake_extra_scores
    )

    # check that the repertoire looks like expected
    expected_genotypes = jnp.array(
        [
            [1.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [1.0 for _ in range(genotype_size)],
        ]
    )
    expected_fitnesses = jnp.array([0.0, -jnp.inf, -jnp.inf, 0.0])
    expected_descriptors = jnp.array(
        [
            [0.1, 0.1],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.9, 0.9],
        ]
    )
    expected_modification_counts = jnp.array([1, 0, 0, 1], dtype=jnp.int32)

    # check values
    assert jnp.allclose(repertoire.genotypes, expected_genotypes, atol=1e-6).item()
    assert jnp.allclose(repertoire.fitnesses, expected_fitnesses, atol=1e-6).item()
    assert jnp.allclose(repertoire.descriptors, expected_descriptors, atol=1e-6).item()
    assert jnp.array_equal(
        repertoire.modification_counts, expected_modification_counts
    ).item()

    # update one occupied cell with a better genotype and keep the other unchanged
    updated_genotypes = jnp.array(
        [
            [2.0 for _ in range(genotype_size)],
            [3.0 for _ in range(genotype_size)],
        ]
    )
    updated_fitnesses = jnp.array([-1.0, 1.0])
    updated_descriptors = fake_descriptors

    repertoire = repertoire.add(
        updated_genotypes, updated_descriptors, updated_fitnesses, fake_extra_scores
    )

    expected_genotypes = jnp.array(
        [
            [1.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [0.0 for _ in range(genotype_size)],
            [3.0 for _ in range(genotype_size)],
        ]
    )
    expected_fitnesses = jnp.array([0.0, -jnp.inf, -jnp.inf, 1.0])
    expected_modification_counts = jnp.array([1, 0, 0, 2], dtype=jnp.int32)

    assert jnp.allclose(repertoire.genotypes, expected_genotypes, atol=1e-6).item()
    assert jnp.allclose(repertoire.fitnesses, expected_fitnesses, atol=1e-6).item()
    assert jnp.array_equal(
        repertoire.modification_counts, expected_modification_counts
    ).item()


def test_mapelites_repertoire_save_load_modification_counts(tmp_path) -> None:
    centroids = compute_euclidean_centroids(
        grid_shape=(2,),
        minval=0.0,
        maxval=1.0,
    )
    repertoire = MapElitesRepertoire(
        genotypes=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        fitnesses=jnp.array([0.5, 1.5]),
        descriptors=jnp.array([[0.25], [0.75]]),
        centroids=centroids,
        modification_counts=jnp.array([2, 5], dtype=jnp.int32),
    )

    repertoire.save(path=f"{tmp_path}/")
    loaded_repertoire = MapElitesRepertoire.load(
        reconstruction_fn=lambda x: x,
        path=f"{tmp_path}/",
    )

    assert jnp.allclose(loaded_repertoire.genotypes, repertoire.genotypes).item()
    assert jnp.allclose(loaded_repertoire.fitnesses, repertoire.fitnesses).item()
    assert jnp.allclose(loaded_repertoire.descriptors, repertoire.descriptors).item()
    assert jnp.allclose(loaded_repertoire.centroids, repertoire.centroids).item()
    assert jnp.array_equal(
        loaded_repertoire.modification_counts, repertoire.modification_counts
    ).item()


def test_mapelites_repertoire_load_legacy_archive_without_modification_counts(
    tmp_path,
) -> None:
    genotypes = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    fitnesses = jnp.array([0.5, 1.5])
    descriptors = jnp.array([[0.25], [0.75]])
    centroids = compute_euclidean_centroids(
        grid_shape=(2,),
        minval=0.0,
        maxval=1.0,
    )

    jnp.save(f"{tmp_path}/genotypes.npy", genotypes)
    jnp.save(f"{tmp_path}/fitnesses.npy", fitnesses)
    jnp.save(f"{tmp_path}/descriptors.npy", descriptors)
    jnp.save(f"{tmp_path}/centroids.npy", centroids)

    loaded_repertoire = MapElitesRepertoire.load(
        reconstruction_fn=lambda x: x,
        path=f"{tmp_path}/",
    )

    assert jnp.array_equal(
        loaded_repertoire.modification_counts,
        jnp.zeros_like(fitnesses, dtype=jnp.int32),
    ).item()
