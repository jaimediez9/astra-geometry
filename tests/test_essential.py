"""Contract and property tests for epipolar geometry (essential.py).

Ground truth is built independently: poses with the Rodrigues helper in
_helpers.py, the essential matrix with an explicit skew product (never with the
module's constructor), and the correspondences with camera.project (the forward
model, the legitimate independent direction for an inverse problem).
"""

import numpy as np
import pytest

from _helpers import rotation_matrix_about, transform_from_blocks

from astra_geometry.camera import intrinsic_matrix, project
from astra_geometry.essential import (
    decompose_essential_matrix,
    essential_matrix,
    estimate_essential_matrix,
    estimate_relative_pose,
    is_essential_matrix,
    select_candidate,
)
from astra_geometry.rotation import is_rotation_matrix
from astra_geometry.transforms import identity_transform, transformation_matrix
from astra_geometry.triangulation import triangulate

K1 = intrinsic_matrix(800.0, 800.0, 640.0, 360.0)
K2 = intrinsic_matrix(780.0, 780.0, 610.0, 380.0)

# Varias poses, incluida una con baseline grande que hace det(U)det(V) = -1
# (el caso que exige la corrección de signos).
POSE_CASES = [
    ((0.0, 0.0, 1.0), 0.20, (0.9, -0.3, 0.15)),
    ((0.1, 0.3, 0.95), 0.25, (0.8, -0.2, 0.1)),
    ((0.2, -0.7, 0.68), 0.40, (2.5, -1.0, 0.5)),
    ((-0.5, 0.1, 0.86), 0.15, (-1.8, 0.6, -0.3)),
    ((0.7, 0.7, 0.14), 0.60, (1.2, 1.4, 0.2)),
]


def _skew(t):
    """Producto vectorial explícito, independiente del módulo."""
    return np.array([
        [0.0, -t[2], t[1]],
        [t[2], 0.0, -t[0]],
        [-t[1], t[0], 0.0],
    ])


def _scene(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack((
        rng.uniform(-2.0, 2.0, n),
        rng.uniform(-2.0, 2.0, n),
        rng.uniform(5.0, 12.0, n),
    ))


def _correspondences(axis, angle, t, n=60, seed=0):
    X = _scene(n=n, seed=seed)
    T = transform_from_blocks(axis, angle, t)
    pixels1 = project(X, K1, identity_transform())
    pixels2 = project(X, K2, T)
    return pixels1, pixels2, T[:3, :3], np.asarray(t, dtype=np.float64)


def _normalized(pixels, K):
    homogeneous = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    normalized = homogeneous @ np.linalg.inv(K).T
    return normalized[:, :2] / normalized[:, 2:3]


def _homogeneous_normalized(pixels, K):
    return np.hstack((pixels, np.ones((pixels.shape[0], 1)))) @ np.linalg.inv(K).T


# --- is_essential_matrix ---

def test_accepts_a_valid_essential_matrix():
    R = rotation_matrix_about((0.1, 0.3, 0.95), 0.25)
    assert is_essential_matrix(_skew(np.array([0.9, -0.3, 0.15])) @ R)


def test_accepts_any_non_zero_scale():
    R = rotation_matrix_about((0.1, 0.3, 0.95), 0.25)
    E = _skew(np.array([0.9, -0.3, 0.15])) @ R
    for c in (1e-6, 0.5, 3.0, 1e6, -2.0):
        assert is_essential_matrix(c * E)


@pytest.mark.parametrize("E", [
    np.eye(3),                                   # no es rango 2
    np.diag([2.0, 1.0, 0.0]),                    # valores singulares desiguales
    np.zeros((3, 3)),                            # matriz nula
    np.ones((3, 3)),
    np.zeros((3, 2)),                            # forma incorrecta
    np.full((3, 3), np.nan),
    np.eye(3, dtype=complex) + 1j,
    "no es una matriz",
    None,
])
def test_rejects_invalid_essential_matrices(E):
    assert not is_essential_matrix(E)


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf, None, 1j, "no"])
def test_is_essential_matrix_rejects_invalid_atol(atol):
    R = rotation_matrix_about((0.1, 0.3, 0.95), 0.25)
    E = _skew(np.array([0.9, -0.3, 0.15])) @ R
    assert not is_essential_matrix(E, atol=atol)


# --- essential_matrix ---

def test_essential_matrix_matches_the_explicit_skew_product():
    R = rotation_matrix_about((0.1, 0.3, 0.95), 0.25)
    t = np.array([0.9, -0.3, 0.15])
    np.testing.assert_allclose(essential_matrix(R, t), _skew(t) @ R, atol=1e-15)


def test_essential_matrix_keeps_the_scale_of_t():
    R = rotation_matrix_about((0.0, 0.0, 1.0), 0.3)
    t = np.array([2.0, 0.0, 0.0])
    E = essential_matrix(R, t)
    assert np.linalg.norm(E) == pytest.approx(np.sqrt(2.0) * np.linalg.norm(t))


def test_essential_matrix_rejects_a_null_translation():
    R = rotation_matrix_about((0.0, 0.0, 1.0), 0.3)
    with pytest.raises(ValueError, match="nulo"):
        essential_matrix(R, np.zeros(3))


def test_essential_matrix_rejects_an_invalid_rotation():
    with pytest.raises(ValueError):
        essential_matrix(np.eye(3) * 2.0, np.array([1.0, 0.0, 0.0]))


@pytest.mark.parametrize("bad_t", [np.zeros((2,)), np.zeros((4,)), np.full((3,), np.inf)])
def test_essential_matrix_rejects_invalid_translation(bad_t):
    R = rotation_matrix_about((0.0, 0.0, 1.0), 0.3)
    with pytest.raises(ValueError):
        essential_matrix(R, bad_t)


# --- the epipolar constraint and the convention ---

@pytest.mark.parametrize("axis,angle,t", POSE_CASES)
def test_epipolar_constraint_holds_for_the_stated_convention(axis, angle, t):
    pixels1, pixels2, R, t = _correspondences(axis, angle, t)
    E = _skew(t) @ R
    x1 = _homogeneous_normalized(pixels1, K1)
    x2 = _homogeneous_normalized(pixels2, K2)
    residual = np.einsum("ni,ij,nj->n", x2, E, x1)
    assert np.abs(residual).max() < 1e-12


@pytest.mark.parametrize("axis,angle,t", POSE_CASES)
def test_the_reversed_product_does_not_satisfy_the_constraint(axis, angle, t):
    """E = [t]_x R is normative: R [t]_x belongs to the reversed pairing."""
    pixels1, pixels2, R, t = _correspondences(axis, angle, t)
    E_reversed = R @ _skew(t)
    x1 = _homogeneous_normalized(pixels1, K1)
    x2 = _homogeneous_normalized(pixels2, K2)
    residual = np.abs(np.einsum("ni,ij,nj->n", x2, E_reversed, x1)).max()
    assert residual > 1e-6


# --- estimate_essential_matrix ---

@pytest.mark.parametrize("axis,angle,t", POSE_CASES)
def test_estimates_an_essential_matrix_satisfying_the_constraint(axis, angle, t):
    pixels1, pixels2, _, _ = _correspondences(axis, angle, t)
    x1, x2 = _normalized(pixels1, K1), _normalized(pixels2, K2)
    x1h, x2h = _homogeneous_normalized(pixels1, K1), _homogeneous_normalized(pixels2, K2)

    E = estimate_essential_matrix(x1, x2)

    assert is_essential_matrix(E)
    assert np.linalg.norm(E) == pytest.approx(np.sqrt(2.0))
    residual = np.einsum("ni,ij,nj->n", x2h, E, x1h)
    assert np.abs(residual).max() < 1e-10


def test_estimated_matrix_is_parallel_to_the_truth():
    pixels1, pixels2, R, t = _correspondences(*POSE_CASES[0])
    x1, x2 = _normalized(pixels1, K1), _normalized(pixels2, K2)
    E_true = _skew(t) @ R
    E = estimate_essential_matrix(x1, x2)

    # Comparison up to the (arbitrary) global scale and sign of E.
    direction = E / np.linalg.norm(E)
    truth = E_true / np.linalg.norm(E_true)
    assert min(
        np.abs(direction - truth).max(), np.abs(direction + truth).max()
    ) < 1e-9


def test_the_projection_onto_the_essential_manifold_is_effective():
    pixels1, pixels2, _, _ = _correspondences(*(POSE_CASES[2]))
    x1, x2 = _normalized(pixels1, K1), _normalized(pixels2, K2)
    singular = np.linalg.svd(estimate_essential_matrix(x1, x2), compute_uv=False)
    assert singular[0] == pytest.approx(singular[1], abs=1e-12)
    assert singular[2] == pytest.approx(0.0, abs=1e-12)


def test_estimation_with_exactly_eight_correspondences():
    pixels1, pixels2, _, _ = _correspondences(*(POSE_CASES[0]), n=8)
    x1, x2 = _normalized(pixels1, K1), _normalized(pixels2, K2)
    assert is_essential_matrix(estimate_essential_matrix(x1, x2))


def test_estimation_needs_at_least_eight_correspondences():
    pixels1, pixels2, _, _ = _correspondences(*(POSE_CASES[0]), n=7)
    x1, x2 = _normalized(pixels1, K1), _normalized(pixels2, K2)
    with pytest.raises(ValueError, match="8"):
        estimate_essential_matrix(x1, x2)


def test_estimation_rejects_mismatched_counts_and_bad_shapes():
    x = np.zeros((9, 2))
    with pytest.raises(ValueError):
        estimate_essential_matrix(x, np.zeros((10, 2)))
    with pytest.raises(ValueError):
        estimate_essential_matrix(np.zeros((9, 3)), x)


# --- decompose_essential_matrix ---

def test_decomposition_returns_four_candidates():
    pixels1, pixels2, R, t = _correspondences(*POSE_CASES[0])
    candidates = decompose_essential_matrix(_skew(t) @ R)
    assert len(candidates) == 4
    for Rc, tc in candidates:
        assert Rc.shape == (3, 3)
        assert tc.shape == (3,)
        assert np.linalg.norm(tc) == pytest.approx(1.0)


@pytest.mark.parametrize("axis,angle,t", POSE_CASES)
def test_every_candidate_is_a_proper_rotation(axis, angle, t):
    """Guarantee of the sign correction: no candidate may be a reflection."""
    _, _, R, t = _correspondences(axis, angle, t)
    for Rc, _ in decompose_essential_matrix(_skew(t) @ R):
        assert np.linalg.det(Rc) == pytest.approx(1.0, abs=1e-12)
        assert is_rotation_matrix(Rc)


@pytest.mark.parametrize("axis,angle,t", POSE_CASES)
def test_recovers_the_true_rotation_and_direction_of_translation(axis, angle, t):
    _, _, R, t = _correspondences(axis, angle, t)
    candidates = decompose_essential_matrix(_skew(t) @ R)

    matches = [
        (Rc, tc) for Rc, tc in candidates if np.abs(Rc - R).max() < 1e-9
    ]
    assert len(matches) == 2  # la rotación correcta, con +t y -t

    unit_true = t / np.linalg.norm(t)
    dots = [float(np.dot(tc, unit_true)) for _, tc in matches]
    assert max(dots) == pytest.approx(1.0)
    assert min(dots) == pytest.approx(-1.0)


def test_the_sign_correction_is_exercised_and_works():
    """At least one pose must hit det(U)det(V) = -1; all must be recovered."""
    same_branch = 0
    negative_branch = 0

    for axis, angle, t in POSE_CASES:
        _, _, R, t = _correspondences(axis, angle, t)
        E = _skew(t) @ R

        U, _, Vt = np.linalg.svd(E)
        V = Vt.T
        if np.linalg.det(U) * np.linalg.det(V) < 0.0:
            negative_branch += 1
        else:
            same_branch += 1

        candidates = decompose_essential_matrix(E)
        best = min(np.abs(Rc - R).max() for Rc, _ in candidates)
        assert best < 1e-12, "la rotación no se recupera en esta pose"

    assert negative_branch >= 1, "los casos de prueba no ejercitan la corrección"
    assert same_branch >= 1, "los casos de prueba no cubren la rama directa"


def _same_candidate_sets(left, right, atol=1e-9):
    """¿Contienen las dos secuencias los mismos candidatos, sin importar el orden?"""
    if len(left) != len(right):
        return False
    remaining = list(right)
    for Ra, ta in left:
        for index, (Rb, tb) in enumerate(remaining):
            if np.abs(Ra - Rb).max() < atol and np.abs(ta - tb).max() < atol:
                remaining.pop(index)
                break
        else:
            return False
    return True


def test_decomposition_set_is_invariant_to_a_positive_scale():
    """El CONJUNTO de candidatos es invariante; el orden no (sigma_1 = sigma_2)."""
    _, _, R, t = _correspondences(*POSE_CASES[2])
    E = _skew(t) @ R
    base = decompose_essential_matrix(E)

    for c in (0.5, 3.0, 1e5):
        scaled = decompose_essential_matrix(c * E)
        assert _same_candidate_sets(base, scaled)

        # Y en todos los casos la rotación verdadera sigue estando en el conjunto.
        assert min(np.abs(Rc - R).max() for Rc, _ in scaled) < 1e-12


def test_candidate_order_is_not_stable_across_scales_but_content_is():
    """Documenta la garantía real: el orden cambia, el contenido no."""
    _, _, R, t = _correspondences(*POSE_CASES[2])
    E = _skew(t) @ R
    base = decompose_essential_matrix(E)
    scaled = decompose_essential_matrix(3.0 * E)

    assert not all(
        np.abs(Ra - Rb).max() < 1e-9
        for (Ra, _), (Rb, _) in zip(base, scaled)
    ), "si el orden fuese estable este test ya no describe la realidad"
    assert _same_candidate_sets(base, scaled)


def test_decomposition_rejects_a_non_essential_matrix():
    with pytest.raises(ValueError):
        decompose_essential_matrix(np.eye(3))


# --- the twisted pair ---

def test_the_twisted_pair_is_not_the_true_rotation():
    _, _, R, t = _correspondences(*POSE_CASES[0])
    candidates = decompose_essential_matrix(_skew(t) @ R)
    proper = candidates[0][0]
    twisted = candidates[2][0]
    assert np.abs(twisted - proper).max() > 0.1
    assert np.abs(twisted - R).max() > 0.1
    # La pareja torcida también es una rotación (no una reflexión).
    assert np.linalg.det(twisted) == pytest.approx(1.0, abs=1e-12)


# --- select_candidate and cheirality ---

@pytest.mark.parametrize("axis,angle,t", POSE_CASES)
def test_cheirality_selects_the_physically_valid_candidate(axis, angle, t):
    pixels1, pixels2, R, t = _correspondences(axis, angle, t)
    candidates = decompose_essential_matrix(_skew(t) @ R)

    R_sel, t_sel = select_candidate(candidates, pixels1, pixels2, K1, K2)

    assert np.abs(R_sel - R).max() < 1e-9
    unit_true = t / np.linalg.norm(t)
    assert float(np.dot(t_sel, unit_true)) == pytest.approx(1.0)


def test_select_candidate_rejects_an_ambiguous_set():
    pixels1, pixels2, R, t = _correspondences(*POSE_CASES[0])
    identity_candidate = (np.eye(3), np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="ambigua|ningún"):
        select_candidate([identity_candidate, identity_candidate],
                         pixels1, pixels2, K1, K2)


def test_cheirality_requires_positive_depth_in_both_cameras():
    """Contar solo la cámara 1 NO basta: elegiría otro candidato.

    Escena descentrada con baseline pequeño. Con el criterio incompleto (solo
    profundidad en la cámara 1) ganaría un candidato distinto del correcto;
    exigiendo profundidad positiva en AMBAS gana el correcto. Es exactamente la
    diferencia que hace que la quiralidad sea quiralidad (conventions.md 6.2/6.3).
    """
    rng = np.random.default_rng(0)
    X = np.column_stack((
        rng.uniform(-1.5, 1.5, 40) - 4.0,
        rng.uniform(-1.5, 1.5, 40) + 1.0,
        rng.uniform(3.0, 10.0, 40),
    ))
    axis, angle = (0.2, -0.7, 0.68), 0.4
    t = np.array([0.1, -0.03, 0.02])
    T = transform_from_blocks(axis, angle, t)
    R = rotation_matrix_about(axis, angle)

    pixels1 = project(X, K1, identity_transform())
    pixels2 = project(X, K2, T)
    candidates = decompose_essential_matrix(_skew(t) @ R)

    # Criterio incompleto: solo profundidad en la cámara 1.
    only_camera1 = []
    for Rc, tc in candidates:
        points = triangulate(pixels1, pixels2, K1, K2, transformation_matrix(Rc, tc))
        only_camera1.append(int(np.count_nonzero(points[:, 2] > 0.0)))
    winner_incomplete = int(np.argmax(only_camera1))

    # Criterio correcto.
    R_sel, t_sel = select_candidate(candidates, pixels1, pixels2, K1, K2)
    winner_correct = next(
        index for index, (Rc, tc) in enumerate(candidates)
        if np.abs(Rc - R_sel).max() < 1e-12 and np.abs(tc - t_sel).max() < 1e-12
    )

    assert winner_incomplete != winner_correct
    assert np.abs(R_sel - R).max() < 1e-9
    assert float(np.dot(t_sel, t / np.linalg.norm(t))) == pytest.approx(1.0)


def test_select_candidate_rejects_invalid_candidates():
    pixels1, pixels2, _, _ = _correspondences(*POSE_CASES[0])
    with pytest.raises(ValueError):
        select_candidate([(np.eye(3) * 2.0, np.array([1.0, 0.0, 0.0]))],
                         pixels1, pixels2, K1, K2)
    with pytest.raises(ValueError):
        select_candidate([(np.eye(3), np.zeros(3))], pixels1, pixels2, K1, K2)
    with pytest.raises(ValueError):
        select_candidate([], pixels1, pixels2, K1, K2)


# --- estimate_relative_pose (end to end) ---

@pytest.mark.parametrize("axis,angle,t", POSE_CASES)
def test_end_to_end_recovers_rotation_and_translation_direction(axis, angle, t):
    pixels1, pixels2, R, t = _correspondences(axis, angle, t)

    R_est, t_est = estimate_relative_pose(pixels1, pixels2, K1, K2)

    np.testing.assert_allclose(R_est, R, atol=1e-9)
    unit_true = t / np.linalg.norm(t)
    assert float(np.dot(t_est, unit_true)) == pytest.approx(1.0)
    assert np.linalg.norm(t_est) == pytest.approx(1.0)


def test_end_to_end_returns_a_unit_translation_not_the_metric_one():
    """Only the direction of t is observable from E (conventions.md 6.2)."""
    t_big = np.array([5.0, -1.0, 0.5])       # traslación de norma != 1
    pixels1, pixels2, R, t = _correspondences((0.2, -0.7, 0.68), 0.4, t_big)

    _, t_est = estimate_relative_pose(pixels1, pixels2, K1, K2)

    assert np.linalg.norm(t_est) == pytest.approx(1.0)
    assert not np.allclose(t_est, t_big, atol=1e-3)


def test_end_to_end_with_noise_stays_close():
    pixels1, pixels2, R, t = _correspondences(*(POSE_CASES[0]), n=200, seed=3)
    rng = np.random.default_rng(7)
    sigma = 0.3  # px
    noisy1 = pixels1 + rng.normal(0.0, sigma, pixels1.shape)
    noisy2 = pixels2 + rng.normal(0.0, sigma, pixels2.shape)

    R_est, t_est = estimate_relative_pose(noisy1, noisy2, K1, K2)

    assert np.abs(R_est - R).max() < 1e-2
    unit_true = t / np.linalg.norm(t)
    assert float(np.dot(t_est, unit_true)) > 0.99


def test_end_to_end_rejects_too_few_correspondences():
    pixels1, pixels2, _, _ = _correspondences(*(POSE_CASES[0]), n=5)
    with pytest.raises(ValueError):
        estimate_relative_pose(pixels1, pixels2, K1, K2)


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf, None, 1j, "no", [1e-8]])
def test_rejects_invalid_atol(atol):
    pixels1, pixels2, _, _ = _correspondences(*(POSE_CASES[0]))
    with pytest.raises(ValueError):
        estimate_relative_pose(pixels1, pixels2, K1, K2, atol=atol)


def test_tolerance_is_keyword_only():
    pixels1, pixels2, _, _ = _correspondences(*(POSE_CASES[0]))
    with pytest.raises(TypeError):
        estimate_relative_pose(pixels1, pixels2, K1, K2, 1e-8)  # type: ignore[reportCallIssue]

    assert len(estimate_relative_pose(pixels1, pixels2, K1, K2, atol=1e-8)) == 2


# --- the linear solve: full vs reduced right singular vectors -----------------
#
# estimate_essential_matrix reads the right singular vector of the SMALLEST
# singular value of an (N, 9) design matrix. A reduced SVD returns a (9, 9) Vh
# only when min(N, 9) = 9, that is from N = 9 on. With N = 8 it returns (8, 9),
# which contains the first eight right singular vectors and NOT the ninth — the
# one in the null space, which is the solution. Using it would return a wrong
# essential matrix silently, so the short case keeps the full decomposition.
# These tests pin the boundary, the equivalence with the full path, and that
# nothing else changed.


def _hartley_reference(xy):
    """Hartley normalization written independently, as this test's own oracle."""
    centroid = xy.mean(axis=0)
    scale = np.sqrt(2.0) / np.linalg.norm(xy - centroid, axis=1).mean()
    normal = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])
    centred = (xy - centroid) * scale
    return normal, np.hstack((centred, np.ones((len(centred), 1))))


def _design_matrix(x1, x2):
    """The design matrix, rebuilt here from independent code."""
    _, h1 = _hartley_reference(x1)
    _, h2 = _hartley_reference(x2)
    return np.einsum("ni,nj->nij", h2, h1).reshape(-1, 9)


def _reference_essential(x1, x2):
    """E from the FULL SVD: the pre-correction path, used as the oracle.

    Same algorithm, same normalization, same essential-manifold projection —
    only the amount of V that gets materialized differs.
    """
    rows = _design_matrix(x1, x2)
    normal1, _ = _hartley_reference(x1)
    normal2, _ = _hartley_reference(x2)
    _, _, Vt = np.linalg.svd(rows, full_matrices=True)
    raw = normal2.T @ Vt[-1].reshape(3, 3) @ normal1
    U, singular, Vt_E = np.linalg.svd(raw)
    sigma = 0.5 * (singular[0] + singular[1])
    E = U @ np.diag([sigma, sigma, 0.0]) @ Vt_E
    return E / (np.linalg.norm(E) / np.sqrt(2.0))


def _same_essential(left, right, atol=1e-9):
    """Compare two essential matrices up to the global scale AND sign.

    Both are already normalized to ||E||_F = sqrt(2), so only the sign is
    free: E and -E describe the same two-view geometry.
    """
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return min(np.abs(left - right).max(), np.abs(left + right).max()) <= atol


@pytest.mark.parametrize("n", [9, 10, 16, 24, 64, 256])
def test_the_reduced_svd_gives_the_same_null_vector_from_nine_on(n):
    """min(N, 9) = 9 from N = 9 on, so nothing is lost by not forming U(N,N)."""
    pixels1, pixels2, _, _ = _correspondences(*POSE_CASES[0], n=n)
    rows = _design_matrix(_normalized(pixels1, K1), _normalized(pixels2, K2))

    full = np.linalg.svd(rows, full_matrices=True)[2]
    reduced = np.linalg.svd(rows, full_matrices=False)[2]

    assert full.shape == (9, 9) and reduced.shape == (9, 9)
    tolerance = 1e-12  # component-wise, on unit-norm singular vectors
    np.testing.assert_allclose(reduced[-1], full[-1], atol=tolerance)
    # Both are null vectors of the same system.
    scale = np.linalg.norm(rows, 2)
    assert np.abs(rows @ reduced[-1]).max() / scale < 1e-14


def test_eight_correspondences_need_the_full_right_singular_vectors():
    """The 8x9 case: the reduced Vh cannot contain the ninth vector."""
    pixels1, pixels2, _, _ = _correspondences(*POSE_CASES[0], n=8)
    rows = _design_matrix(_normalized(pixels1, K1), _normalized(pixels2, K2))
    assert rows.shape == (8, 9)

    full = np.linalg.svd(rows, full_matrices=True)[2]
    reduced = np.linalg.svd(rows, full_matrices=False)[2]

    assert full.shape == (9, 9)
    assert reduced.shape == (8, 9)  # the null vector is simply absent

    scale = np.linalg.norm(rows, 2)
    assert np.abs(rows @ full[-1]).max() / scale < 1e-14          # the solution
    # The naive alternative is not merely imprecise: it is a different vector,
    # far from the null space of the system.
    assert np.abs(rows @ reduced[-1]).max() / scale > 1e-3


@pytest.mark.parametrize("n", [8, 9, 10, 16, 64, 256])
@pytest.mark.parametrize("axis,angle,t", POSE_CASES)
def test_the_estimate_matches_the_full_svd_reference(axis, angle, t, n):
    """Across the branch boundary, the estimate equals the full-SVD one."""
    pixels1, pixels2, _, _ = _correspondences(axis, angle, t, n=n)
    x1, x2 = _normalized(pixels1, K1), _normalized(pixels2, K2)

    estimate = estimate_essential_matrix(x1, x2)
    reference = _reference_essential(x1, x2)

    assert is_essential_matrix(estimate)
    assert _same_essential(estimate, reference), "difieren más allá del signo y la escala"
    residual = np.einsum("ni,ij,nj->n",
                         _homogeneous_normalized(pixels2, K2), estimate,
                         _homogeneous_normalized(pixels1, K1))
    assert np.abs(residual).max() < 1e-10


@pytest.mark.parametrize("n", [8, 9, 16, 64])
def test_the_pose_is_equivalent_across_the_branch_boundary(n):
    """A sign flip of E does not change the pose cheirality selects."""
    pixels1, pixels2, _, _ = _correspondences(*POSE_CASES[2], n=n)
    R_est, t_hat = estimate_relative_pose(pixels1, pixels2, K1, K2)
    _, _, R_true, t_true = _correspondences(*POSE_CASES[2], n=n)

    np.testing.assert_allclose(R_est, R_true, atol=1e-9)
    assert float(np.dot(t_hat, t_true / np.linalg.norm(t_true))) == pytest.approx(1.0)


@pytest.mark.parametrize("n", [8, 24])
def test_degenerate_configurations_match_the_full_svd_reference(n):
    """Collinear, planar and pure-translation data behave as they did before."""
    rng = np.random.default_rng(41)
    with_translation = transform_from_blocks((0.0, 0.2, 0.98), 0.2, (1.4, -0.6, 0.35))
    configurations = {
        "colineales": np.column_stack((np.arange(n, dtype=float),
                                       2.0 * np.arange(n, dtype=float),
                                       6.0 + 3.0 * np.arange(n, dtype=float))),
        "coplanares": np.column_stack((rng.uniform(-1.5, 1.5, n),
                                       rng.uniform(-1.5, 1.5, n), np.full(n, 7.0))),
    }
    for label, points in configurations.items():
        for pose in (with_translation,):
            x1 = _normalized(project(points, K1, identity_transform()), K1)
            x2 = _normalized(project(points, K2, pose), K2)
            estimate = estimate_essential_matrix(x1, x2)
            assert np.all(np.isfinite(estimate)), label
            assert _same_essential(estimate, _reference_essential(x1, x2)), label

    # Pure translation: the rotation is unobservable but the estimate is still
    # the same matrix either way, and it must not depend on the branch.
    points = _scene(n=n, seed=44)
    translation_only = transform_from_blocks((0.0, 0.0, 1.0), 0.0, (1.4, -0.6, 0.35))
    x1 = _normalized(project(points, K1, identity_transform()), K1)
    x2 = _normalized(project(points, K2, translation_only), K2)
    assert _same_essential(estimate_essential_matrix(x1, x2),
                           _reference_essential(x1, x2))


@pytest.mark.parametrize("n", [8, 9, 64])
@pytest.mark.parametrize("sigma", [0.5, 2.0])
def test_pixel_noise_keeps_the_two_paths_in_agreement(n, sigma):
    """With noise the estimate degrades, but both paths degrade identically."""
    pixels1, pixels2, _, _ = _correspondences(*POSE_CASES[0], n=n, seed=5)
    rng = np.random.default_rng(9)
    noisy1 = pixels1 + rng.normal(0.0, sigma, pixels1.shape)
    noisy2 = pixels2 + rng.normal(0.0, sigma, pixels2.shape)
    x1, x2 = _normalized(noisy1, K1), _normalized(noisy2, K2)

    estimate = estimate_essential_matrix(x1, x2)
    assert is_essential_matrix(estimate)
    assert _same_essential(estimate, _reference_essential(x1, x2))


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_intrinsics_scaling_does_not_affect_the_estimate(scale):
    """A pixel-unit change is absorbed by Hartley normalization, not by the SVD."""
    K_scaled = intrinsic_matrix(800.0 * scale, 800.0 * scale,
                                640.0 * scale, 360.0 * scale)
    K_scaled2 = intrinsic_matrix(780.0 * scale, 780.0 * scale,
                                 610.0 * scale, 380.0 * scale)
    # The observations are made THROUGH the scaled intrinsics: this is a change
    # of pixel units, not a reinterpretation of the same pixels.
    points = _scene(n=64, seed=6)
    pose = transform_from_blocks(*POSE_CASES[0])
    pixels1 = project(points, K_scaled, identity_transform())
    pixels2 = project(points, K_scaled2, pose)
    x1, x2 = _normalized(pixels1, K_scaled), _normalized(pixels2, K_scaled2)

    estimate = estimate_essential_matrix(x1, x2)
    assert _same_essential(estimate, _reference_essential(x1, x2))

    residual = np.einsum("ni,ij,nj->n",
                         _homogeneous_normalized(pixels2, K_scaled2), estimate,
                         _homogeneous_normalized(pixels1, K_scaled))
    assert np.abs(residual).max() < 1e-10

    # And the pose is still recovered, so the scale never reaches the solver.
    axis, angle, _ = POSE_CASES[0]
    R_est, t_hat = estimate_relative_pose(pixels1, pixels2, K_scaled, K_scaled2)
    np.testing.assert_allclose(R_est, rotation_matrix_about(axis, angle), atol=1e-9)
    assert np.linalg.norm(t_hat) == pytest.approx(1.0)


def test_the_eight_correspondence_minimum_is_unchanged():
    """The branch boundary is 8, not 9: exactly eight still solves."""
    pixels1, pixels2, _, _ = _correspondences(*POSE_CASES[0], n=8)
    x1, x2 = _normalized(pixels1, K1), _normalized(pixels2, K2)
    assert is_essential_matrix(estimate_essential_matrix(x1, x2))

    pixels1, pixels2, _, _ = _correspondences(*POSE_CASES[0], n=7)
    x1, x2 = _normalized(pixels1, K1), _normalized(pixels2, K2)
    with pytest.raises(ValueError, match="8"):
        estimate_essential_matrix(x1, x2)
