// Fixed-size and batch types shared by the core headers.
//
// Row-major storage for the point batches, because the reference stores points
// one per row ((N, 3), docs/conventions.md §2.1) and the port copies semantics.

#pragma once

#include <Eigen/Core>

namespace astra {

using Matrix2 = Eigen::Matrix2d;
using Matrix3 = Eigen::Matrix3d;
using Matrix4 = Eigen::Matrix4d;

using Vector3 = Eigen::Vector3d;
/// Quaternions are (w, x, y, z), scalar first. Eigen::Quaterniond is NOT used
/// as the public type: it stores its coefficients in a different internal order
/// and brings its own multiplication semantics, so mixing the two conventions
/// in one binary would be a silent source of the very swap this project
/// identified as its most dangerous failure mode.
using Vector4 = Eigen::Vector4d;

/// Points stored one per row, (N, 3).
using Points = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

}  // namespace astra
