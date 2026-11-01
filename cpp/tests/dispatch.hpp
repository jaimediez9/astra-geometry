// Dispatch of the fixture `function` names onto the core, harness-side only.
//
// The table is 1:1 with the Python names on purpose: the port copies the API, so
// the dispatch is auditable line against line and no translation layer can hide
// a mismatch. Twenty functions are reachable here, the ones exercised by the
// twenty-three vectors of the SO(3)/SE(3)/quaternion subset.

#pragma once

#include <Eigen/Core>

#include <string>
#include <string_view>
#include <variant>

#include "astra/quaternion.hpp"
#include "astra/se3.hpp"
#include "astra/so3.hpp"
#include "fixture_io.hpp"

namespace astra::harness {

/// Modules of the fixture file that this spike does not cover. They must be
/// reported, never silently skipped.
[[nodiscard]] inline bool is_module_in_scope(const std::string& module) {
  return module == "rotation" || module == "transforms" || module == "quaternions";
}

// Every value crosses into the comparison as an Eigen::MatrixXd: a vector keeps
// its column shape and a point batch keeps (N, 3), so the shape check in
// compare_close is meaningful rather than always true.
[[nodiscard]] inline Eigen::MatrixXd to_matrix(const Matrix2& m) { return m; }
[[nodiscard]] inline Eigen::MatrixXd to_matrix(const Matrix3& m) { return m; }
[[nodiscard]] inline Eigen::MatrixXd to_matrix(const Matrix4& m) { return m; }
[[nodiscard]] inline Eigen::MatrixXd to_matrix(const Vector3& v) { return v; }
[[nodiscard]] inline Eigen::MatrixXd to_matrix(const Vector4& v) { return v; }
[[nodiscard]] inline Eigen::MatrixXd to_matrix(const Points& p) { return p; }

/// Call the operation named by the fixture. Throws fixture_error for an unknown
/// name -- a silent skip would be exactly the kind of gap this harness exists to
/// expose.
[[nodiscard]] inline Value dispatch(const std::string& function, const json& inputs) {
  // SO(3)
  if (function == "is_rotation_matrix") {
    return so3::is_rotation_matrix(matrix3_from(inputs, "R"));
  }
  if (function == "rotation_matrix3D") {
    return to_matrix(so3::rotation_matrix3D(vector3_from(inputs, "axis"), as_scalar(inputs, "theta")));
  }
  if (function == "rotation_matrix2D") {
    return to_matrix(so3::rotation_matrix2D(as_scalar(inputs, "theta")));
  }
  if (function == "compose_rotations") {
    return to_matrix(so3::compose_rotations(matrix3_from(inputs, "R1"), matrix3_from(inputs, "R2")));
  }
  if (function == "inverse_rotation") {
    return to_matrix(so3::inverse_rotation(matrix3_from(inputs, "R")));
  }
  if (function == "rotate_points") {
    return to_matrix(so3::rotate_points(points_from(inputs, "points"), vector3_from(inputs, "axis"),
                                        as_scalar(inputs, "theta")));
  }

  // quaternions
  if (function == "is_unit_quaternion") {
    return quaternion::is_unit_quaternion(vector4_from(inputs, "q"));
  }
  if (function == "normalize_quaternion") {
    return to_matrix(quaternion::normalize_quaternion(vector4_from(inputs, "q")));
  }
  if (function == "conjugate_quaternion") {
    return to_matrix(quaternion::conjugate_quaternion(vector4_from(inputs, "q")));
  }
  if (function == "quaternion_multiply") {
    return to_matrix(quaternion::quaternion_multiply(vector4_from(inputs, "q1"),
                                                     vector4_from(inputs, "q2")));
  }
  if (function == "rotation_matrix_to_quaternion") {
    return to_matrix(quaternion::rotation_matrix_to_quaternion(matrix3_from(inputs, "R")));
  }
  if (function == "quaternion_to_rotation_matrix") {
    return to_matrix(quaternion::quaternion_to_rotation_matrix(vector4_from(inputs, "q")));
  }

  // SE(3)
  if (function == "is_transformation_matrix") {
    return se3::is_transformation_matrix(matrix4_from(inputs, "T"));
  }
  if (function == "transformation_matrix") {
    return to_matrix(se3::transformation_matrix(matrix3_from(inputs, "R"), vector3_from(inputs, "t")));
  }
  if (function == "identity_transform") {
    return to_matrix(se3::identity_transform());
  }
  if (function == "rotation_part") {
    return to_matrix(se3::rotation_part(matrix4_from(inputs, "T")));
  }
  if (function == "translation_part") {
    return to_matrix(se3::translation_part(matrix4_from(inputs, "T")));
  }
  if (function == "compose_transforms") {
    return to_matrix(
        se3::compose_transforms(matrix4_from(inputs, "T1"), matrix4_from(inputs, "T2")));
  }
  if (function == "inverse_transform") {
    return to_matrix(se3::inverse_transform(matrix4_from(inputs, "T")));
  }
  if (function == "transform_points") {
    return to_matrix(
        se3::transform_points(points_from(inputs, "points"), matrix4_from(inputs, "T")));
  }

  throw fixture_error("function '" + function + "' is not dispatched by this spike");
}

}  // namespace astra::harness
