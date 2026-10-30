"""Contract and property tests for the camera chapter (pinhole model).

They check mathematical properties (projective invariance, projection /
back-projection round trips, agreement with an explicit long-form formula) and
the error contract, not only examples. Expected values are computed with an
independent expression, never by the function under test.
"""

import numpy as np
import pytest

from _helpers import rotation_matrix_about, transform_from_blocks

from astra_geometry.camera import (
    backproject_pixels,
    intrinsic_matrix,
    is_intrinsic_matrix,
    project,
    project_points,
    reprojection_residual,
    world_to_camera,
)

FX, FY, CX, CY = 800.0, 795.0, 640.0, 360.0


def _K(skew=0.0):
    return intrinsic_matrix(FX, FY, CX, CY, skew=skew)


def _reference_project(points, fx=FX, fy=FY, cx=CX, cy=CY, skew=0.0):
    """Explicit long-form projection formula, independent of the module."""
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    u = (fx * X + skew * Y) / Z + cx
    v = fy * Y / Z + cy
    return np.column_stack((u, v))


def _camera_pose(axis=(0.0, 0.0, 1.0), angle=0.6, t=(0.5, -0.3, 2.0)):
    return transform_from_blocks(axis, angle, t)


# --- intrinsic_matrix ---

def test_intrinsic_matrix_builds_the_stated_layout():
    K = _K()
    expected = np.array([
        [FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_array_equal(K, expected)
    assert K.dtype == np.float64


def test_intrinsic_matrix_accepts_non_zero_skew():
    K = _K(skew=12.5)
    assert K[0, 1] == 12.5
    assert is_intrinsic_matrix(K)


@pytest.mark.parametrize("fx,fy", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0)])
def test_intrinsic_matrix_rejects_non_positive_focal_lengths(fx, fy):
    with pytest.raises(ValueError):
        intrinsic_matrix(fx, fy, CX, CY)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [1.0], 1.0 + 0j, "no"])
def test_intrinsic_matrix_rejects_invalid_scalars(value):
    with pytest.raises(ValueError):
        intrinsic_matrix(value, FY, CX, CY)


# --- is_intrinsic_matrix ---

def test_is_intrinsic_matrix_accepts_a_valid_matrix():
    assert is_intrinsic_matrix(_K())
    assert is_intrinsic_matrix(_K(skew=7.0))
    assert is_intrinsic_matrix([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("K", [
    np.eye(3)[:2, :],                                   # wrong shape
    np.eye(3, dtype=complex) + 1j,                      # complex
    np.full((3, 3), np.nan),                            # non-finite
    [[800.0, 0.0, 640.0], [1.0, 800.0, 360.0], [0.0, 0.0, 1.0]],  # K[1,0] != 0
    [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 0.0]],  # last row wrong
    [[0.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],    # fx = 0
    [[800.0, 0.0, 640.0], [0.0, -1.0, 360.0], [0.0, 0.0, 1.0]],   # fy < 0
    "no es una matriz",
    None,
])
def test_is_intrinsic_matrix_rejects_invalid_inputs(K):
    assert not is_intrinsic_matrix(K)


def test_is_intrinsic_matrix_is_exact_for_the_structure():
    """K[2,2] within any atol is still invalid: it is structure, not geometry."""
    K = _K()
    K[2, 2] = 1.0 + 1e-12
    assert not is_intrinsic_matrix(K)

    K = _K()
    K[1, 0] = 1e-12
    assert not is_intrinsic_matrix(K)


# --- project_points ---

def test_project_points_maps_the_optical_axis_to_the_principal_point():
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 10.0]])
    result = project_points(points, _K())
    np.testing.assert_allclose(result, [[CX, CY], [CX, CY]], atol=1e-12)


def test_project_points_matches_the_explicit_formula():
    points = np.array([
        [1.0, 2.0, 5.0],
        [-3.0, 0.5, 2.0],
        [0.25, -1.5, 8.0],
    ])
    np.testing.assert_allclose(
        project_points(points, _K()), _reference_project(points), atol=1e-12
    )


def test_project_points_is_invariant_to_a_positive_scale():
    points = np.array([[1.0, -2.0, 4.0], [0.5, 0.5, 1.0]])
    base = project_points(points, _K())
    for k in (0.1, 3.0, 1000.0):
        np.testing.assert_allclose(project_points(points * k, _K()), base, atol=1e-12)


def test_project_points_skew_couples_u_to_y():
    points = np.array([[0.0, 3.0, 5.0]])
    without = project_points(points, _K(skew=0.0))
    with_skew = project_points(points, _K(skew=20.0))
    assert without[0, 1] == pytest.approx(with_skew[0, 1])          # v unaffected
    assert with_skew[0, 0] - without[0, 0] == pytest.approx(20.0 * 3.0 / 5.0)


def test_project_points_agrees_with_the_matrix_product():
    """lambda [u,v,1]^T = K p, with lambda = Z_c (conventions.md section 6)."""
    points = np.array([[2.0, -1.0, 4.0]])
    K = _K(skew=5.0)
    homogeneous = points @ K.T
    assert homogeneous[0, 2] == pytest.approx(points[0, 2])
    uv = project_points(points, K)
    np.testing.assert_allclose(uv, homogeneous[:, :2] / homogeneous[:, 2:3])


@pytest.mark.parametrize("Z", [0.0, -1.0, -5.0])
def test_project_points_rejects_points_behind_the_image_plane(Z):
    with pytest.raises(ValueError):
        project_points(np.array([[1.0, 1.0, Z]]), _K())


def test_project_points_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        project_points(np.array([[0.0, 0.0, 1.0, 1.0]]), _K())     # (N,4)
    with pytest.raises(ValueError):
        project_points(np.full((2, 3), np.nan), _K())
    with pytest.raises(ValueError):
        project_points(np.array([[1.0, 1.0, 1.0]]),
                       np.array([[800.0, 0.0, 640.0],
                                 [0.0, 800.0, 360.0],
                                 [0.0, 0.0, 0.0]]))                 # invalid K


def test_project_points_handles_an_empty_batch():
    assert project_points(np.empty((0, 3)), _K()).shape == (0, 2)


# --- world_to_camera / project ---

def test_world_to_camera_identity_is_a_no_op():
    points = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(world_to_camera(points, np.eye(4)), points)


def test_world_to_camera_matches_the_homogeneous_formula():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.5]])
    T = _camera_pose()
    R = T[:3, :3]
    t = T[:3, 3]
    np.testing.assert_allclose(world_to_camera(points, T), points @ R.T + t, atol=1e-12)


def test_project_equals_projecting_the_camera_frame_points():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.5], [0.0, 1.0, 4.0]])
    K = _K()
    T = _camera_pose()
    np.testing.assert_allclose(
        project(points, K, T), project_points(world_to_camera(points, T), K), atol=1e-12
    )


def test_project_rejects_an_invalid_pose():
    with pytest.raises(ValueError):
        project(np.array([[1.0, 1.0, 3.0]]), _K(), np.eye(4) * 2.0)


# --- backproject_pixels ---

def test_backproject_pixels_maps_the_principal_point_to_the_optical_axis():
    result = backproject_pixels(np.array([[CX, CY]]), _K(), np.array([5.0]))
    np.testing.assert_allclose(result, [[0.0, 0.0, 5.0]], atol=1e-12)


def test_backproject_pixels_round_trips_through_the_independent_projection():
    rng = np.random.default_rng(0)
    points = np.column_stack((
        rng.uniform(-2.0, 2.0, 20),
        rng.uniform(-2.0, 2.0, 20),
        rng.uniform(0.5, 20.0, 20),
    ))
    for skew in (0.0, 15.0):
        K = _K(skew=skew)
        pixels = _reference_project(points, skew=skew)
        recovered = backproject_pixels(pixels, K, points[:, 2])
        np.testing.assert_allclose(recovered, points, atol=1e-9)


def test_backproject_pixels_round_trips_through_the_module():
    rng = np.random.default_rng(1)
    points = np.column_stack((
        rng.uniform(-1.0, 1.0, 15),
        rng.uniform(-1.0, 1.0, 15),
        rng.uniform(1.0, 10.0, 15),
    ))
    K = _K(skew=9.0)
    np.testing.assert_allclose(
        backproject_pixels(project_points(points, K), K, points[:, 2]), points, atol=1e-9
    )


@pytest.mark.parametrize("depths", [[0.0], [-1.0]])
def test_backproject_pixels_rejects_non_positive_depths(depths):
    with pytest.raises(ValueError):
        backproject_pixels(np.array([[CX, CY]]), _K(), np.array(depths))


def test_backproject_pixels_rejects_length_mismatch():
    with pytest.raises(ValueError):
        backproject_pixels(np.array([[CX, CY], [1.0, 1.0]]), _K(), np.array([1.0]))


def test_backproject_pixels_rejects_invalid_pixels():
    with pytest.raises(ValueError):
        backproject_pixels(np.array([[CX, CY, 1.0]]), _K(), np.array([1.0]))


# --- reprojection_residual ---

def test_reprojection_residual_is_zero_for_consistent_correspondences():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.5], [0.0, 1.0, 4.0]])
    K = _K()
    T = _camera_pose()
    pixels = project(points, K, T)
    residual = reprojection_residual(points, pixels, K, T)
    np.testing.assert_allclose(residual, np.zeros(3), atol=1e-9)


def test_reprojection_residual_equals_the_pixel_distance():
    points = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 5.0]])
    K = _K()
    T = _camera_pose()
    pixels = project(points, K, T)
    shifted = pixels + np.array([[3.0, 0.0], [0.0, 4.0]])
    residual = reprojection_residual(points, shifted, K, T)
    np.testing.assert_allclose(residual, [3.0, 4.0], atol=1e-9)


def test_reprojection_residual_grows_with_a_pose_perturbation():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.5], [0.5, -0.5, 4.0]])
    K = _K()
    T = _camera_pose()
    pixels = project(points, K, T)

    T_bad = T.copy()
    T_bad[:3, :3] = rotation_matrix_about((0.0, 1.0, 0.0), 0.05) @ T[:3, :3]

    residual = reprojection_residual(points, pixels, K, T_bad)
    assert np.all(residual > 0.0)


def test_reprojection_residual_rejects_length_mismatch():
    points = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 5.0]])
    K = _K()
    T = _camera_pose()
    with pytest.raises(ValueError):
        reprojection_residual(points, np.array([[CX, CY]]), K, T)
