"""Contract and property tests for rigid estimation (Kabsch, estimation.py).

Ground truth is built with the independent Rodrigues helper in _helpers.py and
the documented row convention (target = source @ R.T + t), never with the
function under test nor with the library's own transform application, so a
row/column convention mistake cannot cancel out silently.
"""

import numpy as np
import pytest

from _helpers import rotation_matrix_about, transform_from_blocks

from astra_geometry.camera import project, reprojection_residual
from astra_geometry.estimation import estimate_rigid_transform
from astra_geometry.rotation import is_rotation_matrix
from astra_geometry.transforms import (
    compose_transforms,
    identity_transform,
    inverse_transform,
    is_transformation_matrix,
    transform_points,
)


def _cloud(n, seed=0, low=-1.0, high=1.0):
    return np.random.default_rng(seed).uniform(low, high, (n, 3))


def _pose(axis=(0.0, 0.0, 1.0), angle=np.pi / 4, t=(0.3, -0.2, 1.5)):
    return rotation_matrix_about(axis, angle), np.asarray(t, dtype=np.float64)


def _correspondences(n, seed=0, axis=(0.0, 0.0, 1.0), angle=np.pi / 4, t=(0.3, -0.2, 1.5)):
    """source and target with a KNOWN rigid transform, via the row convention."""
    source = _cloud(n, seed=seed)
    R, t = _pose(axis, angle, t)
    target = source @ R.T + t          # q_row = p_row R^T + t_row
    return source, target, R, t


# --- exact recovery ---

def test_recovers_a_pure_rotation():
    source, target, R, _ = _correspondences(12, angle=0.9, t=(0.0, 0.0, 0.0))
    T = estimate_rigid_transform(source, target)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], np.zeros(3), atol=1e-12)


def test_recovers_a_pure_translation():
    source = _cloud(10, seed=3)
    t = np.array([1.0, -2.0, 0.5])
    target = source + t
    T = estimate_rigid_transform(source, target)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-12)


@pytest.mark.parametrize("n", [3, 4, 10, 100, 1000])
def test_recovers_a_combined_transform_for_various_correspondence_counts(n):
    source, target, R, t = _correspondences(n, seed=n, axis=(0.3, -0.5, 0.81), angle=1.1)
    T = estimate_rigid_transform(source, target)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-12)


def test_estimating_a_cloud_against_itself_gives_the_identity():
    source = _cloud(8, seed=7)
    np.testing.assert_allclose(
        estimate_rigid_transform(source, source), np.eye(4), atol=1e-12
    )


def test_rotation_recovery_is_independent_of_the_translation_part():
    source, target, R, t = _correspondences(15, angle=0.6, t=(1.0, 1.0, 1.0))
    T = estimate_rigid_transform(source, target)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-12)


# --- noise ---

def test_moderate_noise_keeps_the_rotation_close():
    source, target, R, t = _correspondences(200, seed=11, angle=0.8)
    rng = np.random.default_rng(99)
    noisy = target + rng.normal(0.0, 1e-3, target.shape)

    T = estimate_rigid_transform(source, noisy)
    assert np.linalg.norm(T[:3, :3] - R) < 1e-2
    assert np.linalg.norm(T[:3, 3] - t) < 1e-2


def test_residual_after_alignment_matches_the_noise_level():
    source, target, _, _ = _correspondences(300, seed=13, angle=0.4)
    rng = np.random.default_rng(5)
    sigma = 5e-3
    noisy = target + rng.normal(0.0, sigma, target.shape)

    T = estimate_rigid_transform(source, noisy)
    residual = np.linalg.norm(transform_points(source, T) - noisy, axis=1)
    # For isotropic 3-D noise the expected magnitude is ~1.6 sigma (below the
    # ~1.73 sigma of a single axis); the fit reduces it slightly further.
    assert 1.0 * sigma < residual.mean() < 2.5 * sigma


# --- degenerate configurations ---

def test_rejects_two_correspondences():
    source = _cloud(2, seed=1)
    with pytest.raises(ValueError):
        estimate_rigid_transform(source, source)


def test_rejects_fewer_than_three_correspondences():
    with pytest.raises(ValueError):
        estimate_rigid_transform(_cloud(1, seed=1), _cloud(1, seed=1))
    with pytest.raises(ValueError):
        estimate_rigid_transform(np.empty((0, 3)), np.empty((0, 3)))


def test_rejects_collinear_points():
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        estimate_rigid_transform(line, line)


def test_rejects_collinear_points_not_axis_aligned():
    direction = np.array([1.0, 2.0, -0.5])
    parameter = np.array([-2.0, -1.0, 0.0, 1.0, 5.0])
    line = parameter[:, None] * direction
    with pytest.raises(ValueError):
        estimate_rigid_transform(line, line)


def test_rejects_coincident_points():
    same = np.tile(np.array([1.0, 2.0, 3.0]), (5, 1))
    with pytest.raises(ValueError):
        estimate_rigid_transform(same, same)


def test_atol_controls_the_degeneracy_threshold():
    # A very flat (but technically non-collinear) cloud is rejected with a large
    # atol and accepted with the default, documenting the absolute criterion.
    flat = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]) * 1e-10
    with pytest.raises(ValueError):
        estimate_rigid_transform(flat, flat, atol=1e-8)
    assert estimate_rigid_transform(flat, flat, atol=1e-30).shape == (4, 4)


# --- coplanar (accepted) ---

def test_accepts_coplanar_points():
    source = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.25, 0.0],
    ])
    R, t = _pose(axis=(0.2, 0.9, -0.3), angle=0.7, t=(0.4, 0.1, -0.6))
    target = source @ R.T + t

    T = estimate_rigid_transform(source, target)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], t, atol=1e-12)


def test_accepts_a_plane_not_through_the_origin():
    source = np.array([
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
        [1.0, 1.0, 2.0],
    ])
    R, t = _pose(angle=0.3, t=(0.0, 0.0, 0.0))
    T = estimate_rigid_transform(source, source @ R.T + t)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-12)


# --- reflection correction ---

def test_returns_a_proper_rotation_for_a_mirrored_correspondence():
    source = _cloud(20, seed=21)
    mirrored = source * np.array([-1.0, 1.0, 1.0])   # reflection: det = -1

    T = estimate_rigid_transform(source, mirrored)

    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-12)
    assert is_rotation_matrix(T[:3, :3])

    # The uncorrected SVD of H genuinely yields a reflection for this data, so
    # the correction is exactly what keeps the result in SO(3).
    centered_source = source - source.mean(axis=0)
    centered_mirrored = mirrored - mirrored.mean(axis=0)
    U, _, Vt = np.linalg.svd(centered_source.T @ centered_mirrored)
    assert np.linalg.det(Vt.T @ U.T) < 0.0


def test_never_returns_a_reflection():
    rng = np.random.default_rng(4)
    for seed in range(5):
        source = np.random.default_rng(seed).uniform(-1, 1, (10, 3))
        target = source + rng.normal(0.0, 0.1, source.shape)
        T = estimate_rigid_transform(source, target)
        assert np.linalg.det(T[:3, :3]) > 0.0


# --- convention and compatibility with transforms.py ---

def test_result_maps_source_onto_target():
    source, target, _, _ = _correspondences(30, seed=17, angle=1.3)
    T = estimate_rigid_transform(source, target)
    np.testing.assert_allclose(transform_points(source, T), target, atol=1e-12)


def test_result_satisfies_the_documented_row_convention():
    source, target, R, t = _correspondences(9, seed=19, angle=0.5)
    T = estimate_rigid_transform(source, target)
    # Column form q = R p + t must hold for every correspondence.
    for i in range(source.shape[0]):
        np.testing.assert_allclose(T[:3, :3] @ source[i] + T[:3, 3], target[i], atol=1e-12)


def test_result_is_a_valid_se3_transform():
    source, target, _, _ = _correspondences(11, seed=23)
    T = estimate_rigid_transform(source, target)
    assert is_transformation_matrix(T)
    np.testing.assert_array_equal(T[3], np.array([0.0, 0.0, 0.0, 1.0]))


def test_composes_with_inverse_to_the_identity():
    source, target, _, _ = _correspondences(14, seed=29)
    T = estimate_rigid_transform(source, target)
    np.testing.assert_allclose(
        compose_transforms(T, inverse_transform(T)), identity_transform(), atol=1e-12
    )


def test_the_inverse_transform_maps_back():
    source, target, _, _ = _correspondences(16, seed=31)
    T = estimate_rigid_transform(source, target)
    back = inverse_transform(T)
    np.testing.assert_allclose(transform_points(target, back), source, atol=1e-12)


def test_agrees_with_a_pose_built_by_transform_from_blocks():
    source = _cloud(25, seed=37)
    T_truth = transform_from_blocks((0.1, 0.2, 0.97), 0.85, (0.5, -0.4, 2.0))
    target = transform_points(source, T_truth)
    T = estimate_rigid_transform(source, target)
    np.testing.assert_allclose(T, T_truth, atol=1e-12)


# --- bridge with the camera chapter (end to end) ---

def test_reprojection_residual_collapses_after_estimation():
    from astra_geometry.camera import intrinsic_matrix

    K = intrinsic_matrix(800.0, 800.0, 640.0, 360.0)
    world = np.array([[x, y, 5.0] for y in (-1.0, 0.0, 1.0, 0.5)
                      for x in (-1.0, 0.0, 1.0, -0.5)])
    T_truth = transform_from_blocks((0.0, 1.0, 0.0), 0.15, (0.2, 0.0, -0.3))
    pixels = project(world, K, T_truth)

    # Recover the pose from 3D-3D correspondences (world -> camera frame points).
    camera_points = transform_points(world, T_truth)
    T_est = estimate_rigid_transform(world, camera_points)

    np.testing.assert_allclose(
        reprojection_residual(world, pixels, K, T_est), np.zeros(world.shape[0]), atol=1e-9
    )


# --- input validation ---

@pytest.mark.parametrize("bad", [
    np.zeros((3, 2)),                       # (N, 2)
    np.zeros((3,)),                         # (N,)
    np.zeros((2, 3, 1)),                    # 3-D
    np.full((4, 3), np.nan),                # non-finite
    np.full((4, 3), np.inf),                # non-finite
    np.ones((4, 3), dtype=complex),         # complex
    "no es una nube",
    None,
])
def test_rejects_invalid_source(bad):
    with pytest.raises(ValueError):
        estimate_rigid_transform(bad, _cloud(4, seed=2))


def test_rejects_mismatched_correspondence_counts():
    with pytest.raises(ValueError):
        estimate_rigid_transform(_cloud(5, seed=2), _cloud(6, seed=2))


@pytest.mark.parametrize("bad", [np.full((4, 3), np.nan), np.full((4, 3), np.inf)])
def test_non_finite_input_names_the_reason(bad):
    # The explicit finiteness check is what makes the rejection say WHY, instead
    # of failing later inside the degeneracy test or the rotation validation.
    with pytest.raises(ValueError, match="finitos"):
        estimate_rigid_transform(bad, _cloud(4, seed=2))
    with pytest.raises(ValueError, match="finitos"):
        estimate_rigid_transform(_cloud(4, seed=2), bad)


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf, -np.inf, None, 1j, "no", [1e-8]])
def test_rejects_invalid_atol(atol):
    with pytest.raises(ValueError):
        estimate_rigid_transform(_cloud(5, seed=2), _cloud(5, seed=2), atol=atol)


def test_tolerance_is_keyword_only():
    source = _cloud(5, seed=2)
    with pytest.raises(TypeError):
        estimate_rigid_transform(source, source, 1e-8)  # type: ignore[reportCallIssue]

    assert estimate_rigid_transform(source, source, atol=1e-8).shape == (4, 4)


def test_accepts_list_inputs():
    source = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    target = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
    T = estimate_rigid_transform(source, target)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], np.array([1.0, 0.0, 0.0]), atol=1e-12)
