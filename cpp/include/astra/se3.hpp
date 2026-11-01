// SE(3) -- a port of src/astra_geometry/transforms.py.
//
// Representation: the homogeneous 4x4 matrix T = [[R, t], [0, 1]].
//
// The last row [0, 0, 0, 1] is structure of the representation, not an
// approximated geometric property (docs/conventions.md §7.3): it is required
// exactly, with no tolerance, and atol governs only the rotation block. That is
// what makes the product of two valid transformations preserve it exactly.
//
// Frame convention: T_{A<-B} maps coordinates of frame B into frame A. Frame
// identifiers are documented per function and are not numerically verifiable.

#pragma once

#include <Eigen/Core>

#include <cmath>
#include <stdexcept>
#include <string>

#include "astra/so3.hpp"
#include "astra/tolerance.hpp"
#include "astra/types.hpp"

namespace astra::se3 {

using astra::Matrix3;
using astra::Matrix4;
using astra::Points;
using astra::Tolerance;
using astra::Vector3;
using astra::kReferenceAtol;

/// Declared before the helpers below need it; defined after them.
[[nodiscard]] bool is_transformation_matrix(const Matrix4& T, Tolerance atol) noexcept;

namespace detail {

/// The last row is compared element-wise for equality, not bitwise: the
/// reference uses np.array_equal, for which -0.0 == 0.0 holds. A bitwise test
/// would reject a matrix the reference accepts.
[[nodiscard]] inline bool last_row_is_exact(const Matrix4& T) noexcept {
  return T(3, 0) == 0.0 && T(3, 1) == 0.0 && T(3, 2) == 0.0 && T(3, 3) == 1.0;
}

/// Port of transforms._as_transform: structural failures first (shape and
/// finiteness), then the SE(3) contract, then the tolerance contract. Raises
/// ValueError in Python; std::invalid_argument here.
inline void require_transform(const Matrix4& T, const char* nombre, Tolerance atol) {
  if (!T.allFinite()) {
    throw std::invalid_argument(std::string(nombre) + " debe contener valores finitos");
  }
  if (!is_transformation_matrix(T, atol)) {
    throw std::invalid_argument(
        std::string(nombre) + " debe representar una transformación rígida válida de SE(3)");
  }
}

inline void require_translation(const Vector3& t, const char* nombre) {
  if (!t.allFinite()) {
    throw std::invalid_argument(std::string(nombre) + " debe contener valores finitos");
  }
}

}  // namespace detail

/// Port of transforms.is_transformation_matrix.
///
/// Order matters and is copied: the tolerance contract, then finiteness, then
/// the exact last row, and only then the rotation block through the SO(3)
/// validation. A matrix with a last row deviating by 1e-6 is REJECTED even
/// though 1e-6 is far inside atol: structure is validated exactly, geometry
/// approximately.
[[nodiscard]] inline bool is_transformation_matrix(const Matrix4& T,
                                                   Tolerance atol = kReferenceAtol) noexcept {
  if (!is_valid_tolerance(atol)) {
    return false;
  }
  if (!T.allFinite()) {
    return false;
  }
  if (!detail::last_row_is_exact(T)) {
    return false;
  }

  return so3::is_rotation_matrix(T.topLeftCorner<3, 3>(), atol);
}

/// Port of transforms.transformation_matrix: builds T = [[R, t], [0, 1]] with
/// the last row written exactly. The result is independent of its inputs.
[[nodiscard]] inline Matrix4 transformation_matrix(const Matrix3& R, const Vector3& t,
                                                   Tolerance atol = kReferenceAtol) {
  if (!so3::is_rotation_matrix(R, atol)) {
    throw std::invalid_argument("R debe representar una matriz de rotación válida de SO(3)");
  }
  detail::require_translation(t, "t");

  Matrix4 T = Matrix4::Identity();
  T.topLeftCorner<3, 3>() = R;
  T.topRightCorner<3, 1>() = t;
  return T;
}

/// Port of transforms.identity_transform.
[[nodiscard]] inline Matrix4 identity_transform() noexcept {
  return Matrix4::Identity();
}

/// Port of transforms.rotation_part: an independent (3, 3) block. The input is
/// validated first, last row included.
[[nodiscard]] inline Matrix3 rotation_part(const Matrix4& T, Tolerance atol = kReferenceAtol) {
  detail::require_transform(T, "T", atol);
  return T.topLeftCorner<3, 3>();
}

/// Port of transforms.translation_part.
[[nodiscard]] inline Vector3 translation_part(const Matrix4& T, Tolerance atol = kReferenceAtol) {
  detail::require_transform(T, "T", atol);
  return T.topRightCorner<3, 1>();
}

/// Port of transforms.compose_transforms: the result is T2 @ T1, i.e. T1 is
/// applied first and T2 afterwards, which is T_{A<-C} = T_{A<-B} T_{B<-C}.
///
/// Both inputs are validated and the result is NOT revalidated, as in the
/// reference: in floating point the orthogonality of a chained product can drift
/// and must not be assumed to keep passing the same tolerance indefinitely. The
/// last row stays exact, because the inputs have it exact.
[[nodiscard]] inline Matrix4 compose_transforms(const Matrix4& T1, const Matrix4& T2,
                                                Tolerance atol = kReferenceAtol) {
  detail::require_transform(T1, "T1", atol);
  detail::require_transform(T2, "T2", atol);

  return T2 * T1;
}

/// Port of transforms.inverse_transform: T^-1 = [[R^T, -R^T t], [0, 1]].
/// The shortcut is valid for R in SO(3) and the input contract is checked
/// before applying it. The result is not revalidated, as in the reference.
[[nodiscard]] inline Matrix4 inverse_transform(const Matrix4& T, Tolerance atol = kReferenceAtol) {
  detail::require_transform(T, "T", atol);

  const Matrix3 R = T.topLeftCorner<3, 3>();
  const Vector3 t = T.topRightCorner<3, 1>();

  Matrix4 T_inverse = Matrix4::Identity();
  T_inverse.topLeftCorner<3, 3>() = R.transpose();
  T_inverse.topRightCorner<3, 1>() = -(R.transpose() * t);
  return T_inverse;
}

/// Port of transforms.transform_points: P' = P R^T + t for P stored by rows.
/// The result is a new (N, 3) array, as in the reference; that is the one
/// dynamic allocation this operation is allowed to make.
[[nodiscard]] inline Points transform_points(const Points& points, const Matrix4& T,
                                             Tolerance atol = kReferenceAtol) {
  if (!points.allFinite()) {
    throw std::invalid_argument("points debe contener valores finitos");
  }
  detail::require_transform(T, "T", atol);

  const Matrix3 R = T.topLeftCorner<3, 3>();
  const Vector3 t = T.topRightCorner<3, 1>();

  Points result(points.rows(), 3);
  for (Eigen::Index i = 0; i < points.rows(); ++i) {
    result.row(i) = (points.row(i) * R.transpose() + t.transpose());
  }
  return result;
}

}  // namespace astra::se3
