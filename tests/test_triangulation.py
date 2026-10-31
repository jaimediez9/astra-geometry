"""Contract and property tests for triangulation (triangulation.py).

Ground truth is built with the independent Rodrigues helper in _helpers.py and
the library's own projection (camera.project is the forward model, which is the
legitimate independent direction for an inverse problem). Expected 3D points are
never produced by the function under test.
"""

import numpy as np
import pytest

from _helpers import rotation_matrix_about, transform_from_blocks

from astra_geometry.camera import intrinsic_matrix, project
from astra_geometry.transforms import identity_transform, transformation_matrix
from astra_geometry.triangulation import (
    projection_matrix,
    triangulate,
    triangulate_points,
)

K1 = intrinsic_matrix(800.0, 800.0, 640.0, 360.0)
K2 = intrinsic_matrix(780.0, 780.0, 610.0, 380.0)


def _scene(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack((
        rng.uniform(-1.0, 1.0, n),
        rng.uniform(-1.0, 1.0, n),
        rng.uniform(4.0, 9.0, n),
    ))


def _relative_pose(axis=(0.0, 0.2, 0.98), angle=0.2, t=(0.9, -0.3, 0.15)):
    return transform_from_blocks(axis, angle, t)


def _observations(X, T_2_from_1):
    """Pixels in camera 1 (frame 1 is the reference) and camera 2."""
    pixels1 = project(X, K1, identity_transform())
    pixels2 = project(X, K2, T_2_from_1)
    return pixels1, pixels2


# --- projection_matrix ---

def test_projection_matrix_shape_and_value():
    T = _relative_pose()
    P = projection_matrix(K1, T)
    assert P.shape == (3, 4)
    np.testing.assert_allclose(P, K1 @ T[:3, :], atol=0.0)


def test_projection_matrix_of_the_identity_is_K_padded_with_a_zero_column():
    P = projection_matrix(K1, identity_transform())
    np.testing.assert_array_equal(P[:, 3], np.zeros(3))
    np.testing.assert_allclose(P[:, :3], K1, atol=0.0)


@pytest.mark.parametrize("bad_T", [np.eye(3), np.zeros((4, 4)), np.eye(4) * 2.0])
def test_projection_matrix_rejects_invalid_pose(bad_T):
    with pytest.raises(ValueError):
        projection_matrix(K1, bad_T)


def test_projection_matrix_rejects_invalid_intrinsics():
    with pytest.raises(ValueError):
        projection_matrix(np.zeros((3, 3)), identity_transform())


# --- exact reconstruction ---

def test_recovers_the_scene_in_the_first_frame():
    X = _scene()
    pixels1, pixels2 = _observations(X, _relative_pose())
    result = triangulate(pixels1, pixels2, K1, K2, _relative_pose())
    np.testing.assert_allclose(result, X, atol=1e-9)


@pytest.mark.parametrize("n", [1, 2, 5, 50])
def test_recovers_scenes_of_several_sizes(n):
    X = _scene(n=n, seed=n)
    T = _relative_pose()
    pixels1, pixels2 = _observations(X, T)
    np.testing.assert_allclose(
        triangulate(pixels1, pixels2, K1, K2, T), X, atol=1e-9
    )


def test_reconstruction_is_in_the_first_camera_frame():
    # A translation-only pose: the reconstructed coordinates must equal the
    # original ones expressed in frame 1.
    X = _scene(seed=3)
    T = transform_from_blocks((0.0, 0.0, 1.0), 0.0, (1.0, 0.0, 0.0))
    pixels1, pixels2 = _observations(X, T)
    result = triangulate(pixels1, pixels2, K1, K2, T)
    np.testing.assert_allclose(result, X, atol=1e-9)
    np.testing.assert_allclose(result[:, 2], X[:, 2], atol=1e-12)


def test_triangulated_points_reproject_onto_the_observations():
    X = _scene(seed=7)
    T = _relative_pose(angle=0.35, t=(1.2, 0.4, -0.2))
    pixels1, pixels2 = _observations(X, T)
    result = triangulate(pixels1, pixels2, K1, K2, T)

    np.testing.assert_allclose(
        project(result, K1, identity_transform()), pixels1, atol=1e-8
    )
    np.testing.assert_allclose(
        project(result, K2, T), pixels2, atol=1e-8
    )


def test_triangulate_matches_the_explicit_projection_matrices():
    X = _scene(seed=11)
    T = _relative_pose()
    pixels1, pixels2 = _observations(X, T)

    P1 = projection_matrix(K1, identity_transform())
    P2 = projection_matrix(K2, T)

    np.testing.assert_allclose(
        triangulate(pixels1, pixels2, K1, K2, T),
        triangulate_points(pixels1, pixels2, P1, P2),
        atol=1e-12,
    )


def test_depth_sign_follows_the_geometry():
    X = _scene(seed=13)
    T = _relative_pose()
    pixels1, pixels2 = _observations(X, T)
    result = triangulate(pixels1, pixels2, K1, K2, T)
    assert np.all(result[:, 2] > 0.0)
    assert np.all((result @ T[:3, :3].T + T[:3, 3])[:, 2] > 0.0)


def test_a_wrong_pose_gives_a_wrong_reconstruction():
    X = _scene(seed=17)
    T = _relative_pose()
    pixels1, pixels2 = _observations(X, T)
    T_wrong = transform_from_blocks((0.3, 0.1, 0.95), 0.5, (1.0, 0.0, 0.0))
    result = triangulate(pixels1, pixels2, K1, K2, T_wrong)
    assert not np.allclose(result, X, atol=1e-3)


def test_noise_degrades_the_reconstruction_but_stays_bounded():
    X = _scene(seed=19)
    T = _relative_pose()
    pixels1, pixels2 = _observations(X, T)
    rng = np.random.default_rng(5)
    sigma = 0.5  # pixels
    noisy1 = pixels1 + rng.normal(0.0, sigma, pixels1.shape)
    noisy2 = pixels2 + rng.normal(0.0, sigma, pixels2.shape)

    result = triangulate(noisy1, noisy2, K1, K2, T)
    error = np.linalg.norm(result - X, axis=1)
    assert error.max() < 0.5  # ~2 sigma * depth / focal, generous bound


# --- degeneracy ---

def test_rejects_a_point_at_infinity():
    # Rays from the two cameras are parallel for a point at infinity, so the
    # homogeneous coordinate vanishes and the depth is unbounded.
    direction = np.array([0.2, -0.3, 1.0])
    direction = direction / np.linalg.norm(direction)

    T = _relative_pose()
    d2 = T[:3, :3] @ direction

    p1 = K1 @ direction
    p2 = K2 @ d2
    pixel1 = (p1[:2] / p1[2])[None, :]
    pixel2 = (p2[:2] / p2[2])[None, :]

    with pytest.raises(ValueError, match="infinito|degenerada"):
        triangulate(pixel1, pixel2, K1, K2, T)


def test_atol_controls_the_infinity_threshold():
    X = _scene(seed=23)
    T = _relative_pose()
    pixels1, pixels2 = _observations(X, T)
    # A finite point is accepted with a normal atol and rejected with an absurd one.
    assert triangulate(pixels1, pixels2, K1, K2, T).shape == (len(X), 3)
    with pytest.raises(ValueError):
        triangulate(pixels1, pixels2, K1, K2, T, atol=1.0)


# --- input validation ---

def test_rejects_mismatched_correspondence_counts():
    with pytest.raises(ValueError):
        triangulate(np.zeros((3, 2)), np.zeros((4, 2)), K1, K2, _relative_pose())


@pytest.mark.parametrize("bad", [
    np.zeros((3, 3)),                      # (N, 3)
    np.zeros((3,)),                        # (N,)
    np.full((3, 2), np.nan),               # non-finite
    np.ones((3, 2), dtype=complex),        # complex
    "no es una nube",
    None,
])
def test_rejects_invalid_pixels(bad):
    with pytest.raises(ValueError):
        triangulate(bad, np.zeros((3, 2)), K1, K2, _relative_pose())


@pytest.mark.parametrize("bad", [np.full((3, 2), np.nan), np.full((3, 2), np.inf)])
def test_non_finite_pixels_name_the_reason(bad):
    # numpy's LinAlgError is a subclass of ValueError, so without the explicit
    # finiteness check the call would still "fail with ValueError" — but only
    # deep inside the SVD. The explicit check is what makes the rejection say why.
    with pytest.raises(ValueError, match="finitos"):
        triangulate(bad, np.zeros((3, 2)), K1, K2, _relative_pose())
    with pytest.raises(ValueError, match="finitos"):
        triangulate(np.zeros((3, 2)), bad, K1, K2, _relative_pose())


@pytest.mark.parametrize("bad", [np.zeros((3, 3)), np.zeros((4, 3)), np.full((3, 4), np.inf)])
def test_triangulate_points_rejects_invalid_projection_matrices(bad):
    with pytest.raises(ValueError):
        triangulate_points(
            np.zeros((3, 2)), np.zeros((3, 2)), bad, projection_matrix(K2, np.eye(4))
        )


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf, None, 1j, "no", [1e-8]])
def test_rejects_invalid_atol(atol):
    with pytest.raises(ValueError):
        triangulate(
            np.zeros((3, 2)), np.zeros((3, 2)), K1, K2, _relative_pose(), atol=atol
        )


def test_tolerance_is_keyword_only():
    pixels = np.array([[640.0, 360.0], [700.0, 300.0]])
    T = _relative_pose()
    with pytest.raises(TypeError):
        triangulate(pixels, pixels, K1, K2, T, 1e-8)  # type: ignore[reportCallIssue]

    assert triangulate(pixels, pixels, K1, K2, T, atol=1e-8).shape == (2, 3)


def test_handles_an_empty_batch():
    assert triangulate(np.empty((0, 2)), np.empty((0, 2)), K1, K2, _relative_pose()).shape == (0, 3)
