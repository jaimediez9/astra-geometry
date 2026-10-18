
"""Astra Geometry: herramientas geométricas para robótica y visión artificial.

El paquete proporciona operaciones de rotación, validación geométrica
y transformaciones espaciales.
"""

from astra_geometry.rotation import (
    compose_rotations,
    inverse_rotation,
    is_rotation_matrix,
    rotate_points,
    rotation_matrix2D,
    rotation_matrix3D,
)

__all__ = [
    "compose_rotations",
    "inverse_rotation",
    "is_rotation_matrix",
    "rotate_points",
    "rotation_matrix2D",
    "rotation_matrix3D",
]

