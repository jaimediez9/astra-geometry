"""Shared helpers for the test suite.

These build inputs independently of the library code, so that a function under
test is never fed an input produced by the library itself.
"""

import numpy as np


def rotation_matrix_about(axis, theta):
    """Rodrigues rotation matrix for a rotation of theta about axis."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis
    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])

    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def transform_from_blocks(axis, theta, t):
    """Rigid transform assembled by hand from a rotation and a translation."""
    T = np.eye(4)
    T[:3, :3] = rotation_matrix_about(axis, theta)
    T[:3, 3] = np.asarray(t, dtype=np.float64)

    return T
