// Quaternion algebra -- a port of src/astra_geometry/quaternions.py.
//
// Convention (docs/conventions.md §3): scalar first, (w, x, y, z), Hamilton
// product. These are algebraic operations and do not require unit norm, with
// the two exceptions that validate explicitly (is_unit_quaternion and
// quaternion_to_rotation_matrix).

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "astra/so3.hpp"
#include "astra/tolerance.hpp"
#include "astra/types.hpp"

namespace astra::quaternion {

using astra::Matrix3;
using astra::Tolerance;
using astra::Vector4;
using astra::kReferenceAtol;

namespace detail {

/// Port of quaternions._as_quaternion: shape (4,) and finiteness, nothing about
/// the norm. The shape is enforced by the type here; the finiteness is not.
inline void require_quaternion(const Vector4& q, const char* nombre) {
  if (!q.allFinite()) {
    throw std::invalid_argument(std::string(nombre) + " debe contener valores finitos");
  }
}

/// Port of quaternions._hamilton_product: left * right.
[[nodiscard]] inline Vector4 hamilton_product(const Vector4& left, const Vector4& right) noexcept {
  const double wl = left[0];
  const double xl = left[1];
  const double yl = left[2];
  const double zl = left[3];

  const double wr = right[0];
  const double xr = right[1];
  const double yr = right[2];
  const double zr = right[3];

  return Vector4(wl * wr - xl * xr - yl * yr - zl * zr,
                 wl * xr + xl * wr + yl * zr - zl * yr,
                 wl * yr - xl * zr + yl * wr + zl * xr,
                 wl * zr + xl * yr - yl * xr + zl * wr);
}

}  // namespace detail

/// Port of quaternions.is_unit_quaternion: the norm is 1 within atol, rtol = 0.
/// Returns false for a violated tolerance contract rather than throwing.
[[nodiscard]] inline bool is_unit_quaternion(const Vector4& q,
                                             Tolerance atol = kReferenceAtol) noexcept {
  if (!is_valid_tolerance(atol)) {
    return false;
  }
  if (!q.allFinite()) {
    return false;
  }

  const double norm = q.norm();
  return std::isfinite(norm) && std::abs(norm - 1.0) <= atol.value;
}

/// Port of quaternions.normalize_quaternion.
///
/// Scaled by the largest absolute component first, then by the norm: the
/// reference does it in that order to avoid overflow while computing the norm,
/// and the port keeps it.
[[nodiscard]] inline Vector4 normalize_quaternion(const Vector4& q) {
  detail::require_quaternion(q, "q");

  const double scale = q.cwiseAbs().maxCoeff();
  if (scale == 0.0) {
    throw std::invalid_argument("A zero quaternion cannot be normalized");
  }

  const Vector4 scaled = q / scale;
  return scaled / scaled.norm();
}

/// Port of quaternions.conjugate_quaternion: (w, -x, -y, -z). For a unit
/// quaternion this is the inverse; a general inverse is not computed.
[[nodiscard]] inline Vector4 conjugate_quaternion(const Vector4& q) {
  detail::require_quaternion(q, "q");
  return Vector4(q[0], -q[1], -q[2], -q[3]);
}

/// Port of quaternions.quaternion_multiply.
///
/// The argument order follows compose_rotations: q1 is applied first and q2
/// afterwards, so the result is q2 * q1 and its rotation is R2 @ R1. The
/// product is not commutative and this ordering is part of the API.
[[nodiscard]] inline Vector4 quaternion_multiply(const Vector4& q1, const Vector4& q2) {
  detail::require_quaternion(q1, "q1");
  detail::require_quaternion(q2, "q2");

  return detail::hamilton_product(q2, q1);
}

/// Port of quaternions.rotation_matrix_to_quaternion: branch on the trace, then
/// normalise, then pick the representative with w >= 0 when the rule can decide.
///
/// For a half turn (w == 0) both q and -q are valid and the sign is decided by
/// which extraction branch ran: the reference documents that the sign must not
/// be assumed stable near that case, only the rotation it reconstructs. The port
/// reproduces the same branches, so it has the same (documented) limitation.
[[nodiscard]] inline Vector4 rotation_matrix_to_quaternion(const Matrix3& R,
                                                           Tolerance atol = kReferenceAtol) {
  if (!is_valid_tolerance(atol)) {
    throw std::invalid_argument("atol debe ser un escalar real finito no negativo");
  }
  if (!so3::is_rotation_matrix(R, atol)) {
    throw std::invalid_argument("R debe representar una matriz de rotación válida de SO(3)");
  }

  const double m00 = R(0, 0);
  const double m01 = R(0, 1);
  const double m02 = R(0, 2);
  const double m10 = R(1, 0);
  const double m11 = R(1, 1);
  const double m12 = R(1, 2);
  const double m20 = R(2, 0);
  const double m21 = R(2, 1);
  const double m22 = R(2, 2);

  const double trace = m00 + m11 + m22;

  Vector4 q;
  if (trace > 0.0) {
    const double s = 2.0 * std::sqrt(std::max(0.0, 1.0 + trace));
    q = Vector4(0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s);
  } else if (m00 >= m11 && m00 >= m22) {
    const double s = 2.0 * std::sqrt(std::max(0.0, 1.0 + m00 - m11 - m22));
    q = Vector4((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s);
  } else if (m11 >= m22) {
    const double s = 2.0 * std::sqrt(std::max(0.0, 1.0 + m11 - m00 - m22));
    q = Vector4((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s);
  } else {
    const double s = 2.0 * std::sqrt(std::max(0.0, 1.0 + m22 - m00 - m11));
    q = Vector4((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s);
  }

  q /= q.norm();

  if (q[0] < 0.0) {
    q = -q;
  }

  return q;
}

/// Port of quaternions.quaternion_to_rotation_matrix. q and -q give the same
/// matrix, as they represent the same rotation.
[[nodiscard]] inline Matrix3 quaternion_to_rotation_matrix(const Vector4& q,
                                                           Tolerance atol = kReferenceAtol) {
  if (!is_valid_tolerance(atol)) {
    throw std::invalid_argument("atol debe ser finito y no negativo");
  }
  if (!is_unit_quaternion(q, atol)) {
    throw std::invalid_argument("q debe ser un cuaternión unitario válido");
  }

  const double w = q[0];
  const double x = q[1];
  const double y = q[2];
  const double z = q[3];

  Matrix3 R;
  R << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
      2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
      2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y);
  return R;
}

}  // namespace astra::quaternion
