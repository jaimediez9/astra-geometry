// SO(3): validation, construction, composition, inversion and application to
// points -- a port of src/astra_geometry/rotation.py.
//
// Names mirror the Python API exactly (snake_case), so the conformance dispatch
// is auditable line against line and no translation layer can hide a mismatch.
//
// Header-only, fixed-size types only: no dynamic allocation in any of these
// operations except `rotate_points`, whose contract is to return a new (N, 3)
// array. That single allocation is measured and reported, not assumed away.

#pragma once

#include <Eigen/Core>
#include <Eigen/LU>  // determinant() for fixed-size matrices lives here, not in Core

#include <cmath>
#include <stdexcept>

#include "astra/tolerance.hpp"
#include "astra/types.hpp"

namespace astra::so3 {

/// §7.1: the default floating-point representation is double.
using astra::Matrix2;
using astra::Matrix3;
using astra::Points;
using astra::Tolerance;
using astra::Vector3;
using astra::kReferenceAtol;

namespace detail {

[[nodiscard]] inline bool is_rotation_block(const Matrix3& R, Tolerance atol) noexcept {
  const Matrix3 gram = R.transpose() * R;
  const double orthogonality = (gram - Matrix3::Identity()).cwiseAbs().maxCoeff();
  if (!(orthogonality <= atol.value)) {  // written so that NaN fails the test
    return false;
  }

  const double determinant = R.determinant();
  return std::isfinite(determinant) && std::abs(determinant - 1.0) <= atol.value;
}

[[nodiscard]] inline bool axis_and_angle_are_valid(const Vector3& axis, double theta) noexcept {
  return std::isfinite(theta) && axis.allFinite();
}

}  // namespace detail

/// Port of rotation.is_rotation_matrix: orthogonality (max|R^T R - I| <= atol,
/// rtol = 0) and a determinant of exactly +1 within atol. A reflection is
/// orthogonal but not a rotation, which is why both are checked.
/// Returns false for a violated tolerance contract rather than throwing.
[[nodiscard]] inline bool is_rotation_matrix(const Matrix3& R,
                                             Tolerance atol = kReferenceAtol) noexcept {
  if (!is_valid_tolerance(atol)) {
    return false;
  }
  if (!R.allFinite()) {
    return false;
  }
  return detail::is_rotation_block(R, atol);
}

/// Port of rotation.rotation_matrix3D: Rodrigues from an axis and an angle.
///
/// The axis is normalised in two steps (divide by the largest absolute
/// component, then by the norm) exactly as the reference does: it is a scaling
/// choice that affects rounding, and the port copies it rather than improving
/// it.
[[nodiscard]] inline Matrix3 rotation_matrix3D(const Vector3& axis_in, double theta) {
  if (!std::isfinite(theta)) {
    throw std::invalid_argument("theta debe ser un escalar real finito");
  }
  if (!axis_in.allFinite()) {
    throw std::invalid_argument("axis debe contener valores finitos");
  }

  const double scale = axis_in.cwiseAbs().maxCoeff();
  if (scale == 0.0) {
    throw std::invalid_argument("El eje de rotación no puede ser nulo");
  }

  Vector3 axis = axis_in / scale;
  axis /= axis.norm();

  const double x = axis[0];
  const double y = axis[1];
  const double z = axis[2];

  Matrix3 K;
  K << 0.0, -z, y,
       z, 0.0, -x,
       -y, x, 0.0;

  return Matrix3::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * (K * K);
}

/// Port of rotation.rotation_matrix2D.
[[nodiscard]] inline Matrix2 rotation_matrix2D(double theta) {
  if (!std::isfinite(theta)) {
    throw std::invalid_argument("theta debe ser un escalar real finito");
  }

  const double c = std::cos(theta);
  const double s = std::sin(theta);

  Matrix2 m;
  m << c, -s,
       s, c;
  return m;
}

/// Port of rotation.rotate_points: P' = P R^T for P stored by rows (N, 3).
///
/// The result is a new array, as in the reference; that is the one dynamic
/// allocation this operation is allowed to make, and the memory report says so.
[[nodiscard]] inline Points rotate_points(const Points& points, const Vector3& axis, double theta) {
  if (!points.allFinite()) {
    throw std::invalid_argument("points debe contener valores finitos");
  }

  const Matrix3 R = rotation_matrix3D(axis, theta);  // validates axis and theta
  return points * R.transpose();
}

/// Port of rotation.compose_rotations: the result is R2 @ R1, i.e. R1 is applied
/// first and R2 afterwards. The order is part of the API, not a detail.
/// Validation uses the reference default tolerance, as the reference does.
[[nodiscard]] inline Matrix3 compose_rotations(const Matrix3& R1, const Matrix3& R2) {
  if (!is_rotation_matrix(R1)) {
    throw std::invalid_argument("R1 debe ser una matriz de rotación válida");
  }
  if (!is_rotation_matrix(R2)) {
    throw std::invalid_argument("R2 debe ser una matriz de rotación válida");
  }

  return R2 * R1;
}

/// Port of rotation.inverse_rotation: the inverse of a rotation is its
/// transpose. Note that the reference takes no tolerance here, so neither does
/// this function -- accepting one would be a wider API than Python's.
[[nodiscard]] inline Matrix3 inverse_rotation(const Matrix3& R) {
  if (!is_rotation_matrix(R)) {
    throw std::invalid_argument("R debe ser una matriz de rotación válida");
  }

  return R.transpose();  // materialised into a Matrix3 value, not left as an expression
}

}  // namespace astra::so3
