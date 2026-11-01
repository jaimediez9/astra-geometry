// Property tests for SE(3), including the structural rule that has no tolerance:
// the last row of a homogeneous matrix is structure, not geometry (§7.3).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "astra/se3.hpp"
#include "astra/so3.hpp"
#include "astra/tolerance.hpp"
#include "test_helpers.hpp"

namespace {

using astra::Matrix3;
using astra::Matrix4;
using astra::Points;
using astra::Tolerance;
using astra::Vector3;
using astra::se3::compose_transforms;
using astra::se3::identity_transform;
using astra::se3::inverse_transform;
using astra::se3::is_transformation_matrix;
using astra::se3::rotation_part;
using astra::se3::transform_points;
using astra::se3::transformation_matrix;
using astra::se3::translation_part;

const double kNotANumber = std::numeric_limits<double>::quiet_NaN();

[[nodiscard]] bool last_row_is_exactly_structural(const Matrix4& T) {
  return T(3, 0) == 0.0 && T(3, 1) == 0.0 && T(3, 2) == 0.0 && T(3, 3) == 1.0;
}

[[nodiscard]] Matrix4 sample_transform() {
  return transformation_matrix(astra::so3::rotation_matrix3D(Vector3(0.3, -0.5, 0.81), 0.7),
                               Vector3(0.1, -0.2, 0.3));
}

}  // namespace

TEST_CASE("is_transformation_matrix requires an exact last row", "[se3]") {
  const Matrix4 valid = sample_transform();
  REQUIRE(is_transformation_matrix(valid));
  REQUIRE(is_transformation_matrix(identity_transform()));

  // The case the fixtures pin as `bad_last_row`: the rotation block is a valid
  // rotation and the deviation is 1e-6, far INSIDE atol, yet the matrix must be
  // rejected because the last row is structure and carries no tolerance.
  Matrix4 noisy_row = valid;
  noisy_row(3, 3) = 1.0 + 1e-6;
  REQUIRE(astra::so3::is_rotation_matrix(noisy_row.topLeftCorner<3, 3>()));
  REQUIRE_FALSE(is_transformation_matrix(noisy_row));

  // The same argument for a translation bleeding into the last row.
  Matrix4 noisy_translation = valid;
  noisy_translation(3, 0) = 1e-9;
  REQUIRE_FALSE(is_transformation_matrix(noisy_translation));

  // And a matrix whose rotation block is not a rotation at all.
  Matrix4 bad_block = valid;
  bad_block(0, 1) = 0.5;
  REQUIRE_FALSE(is_transformation_matrix(bad_block));

  Matrix4 not_finite = valid;
  not_finite(1, 2) = kNotANumber;
  REQUIRE_FALSE(is_transformation_matrix(not_finite));
}

TEST_CASE("the structural row survives composition and inversion exactly", "[se3]") {
  const Matrix4 a = sample_transform();
  const Matrix4 b =
      transformation_matrix(astra::so3::rotation_matrix3D(Vector3(0.7, 0.7, -0.7), -1.1),
                            Vector3(-0.4, 0.9, 0.2));

  REQUIRE(last_row_is_exactly_structural(compose_transforms(a, b)));
  REQUIRE(last_row_is_exactly_structural(inverse_transform(a)));
  REQUIRE(last_row_is_exactly_structural(compose_transforms(inverse_transform(a), a)));

  // Exactness is what makes the product of two valid transformations valid
  // again; a tolerance there would let noise accumulate into the structure.
  REQUIRE(is_transformation_matrix(compose_transforms(a, b)));
}

TEST_CASE("transformation_matrix lays out R and t correctly", "[se3]") {
  const Matrix3 R = astra::so3::rotation_matrix3D(Vector3(0.3, -0.5, 0.81), 0.7);
  const Vector3 t(0.1, -0.2, 0.3);
  const Matrix4 T = transformation_matrix(R, t);

  REQUIRE(astra::test::all_bit_identical<Matrix3>(T.topLeftCorner<3, 3>(), R));
  REQUIRE(astra::test::all_bit_identical<Vector3>(T.topRightCorner<3, 1>(), t));
  REQUIRE(last_row_is_exactly_structural(T));

  Matrix3 not_a_rotation = Matrix3::Identity();
  not_a_rotation(0, 0) = 2.0;
  REQUIRE_THROWS_AS(transformation_matrix(not_a_rotation, t), std::invalid_argument);
  REQUIRE_THROWS_AS(transformation_matrix(R, Vector3(1.0, kNotANumber, 0.0)),
                    std::invalid_argument);
}

TEST_CASE("the accessors return the blocks and validate the input", "[se3]") {
  const Matrix4 T = sample_transform();
  REQUIRE(astra::test::all_bit_identical<Matrix3>(rotation_part(T), T.topLeftCorner<3, 3>()));
  REQUIRE(astra::test::all_bit_identical<Vector3>(translation_part(T), T.topRightCorner<3, 1>()));

  Matrix4 bad = T;
  bad(3, 3) = 2.0;
  REQUIRE_THROWS_AS(rotation_part(bad), std::invalid_argument);
  REQUIRE_THROWS_AS(translation_part(bad), std::invalid_argument);
}

TEST_CASE("compose_transforms applies T1 first and T2 afterwards", "[se3]") {
  const Matrix4 a =
      transformation_matrix(Matrix3::Identity(), Vector3(1.0, 0.0, 0.0));
  const Matrix4 b =
      transformation_matrix(astra::so3::rotation_matrix3D(Vector3(0.0, 0.0, 1.0), 1.5707963267948966),
                            Vector3(0.0, 0.0, 0.0));

  // T1 first, then T2: the result is T2 @ T1 and a point at the origin lands on
  // (0, 1, 0), not on (1, 0, 0) as the other order would give.
  const Matrix4 composed = compose_transforms(a, b);
  REQUIRE(astra::test::all_bit_identical<Matrix4>(composed, b * a));

  const Points origin = (Eigen::MatrixXd(1, 3) << 0.0, 0.0, 0.0).finished();
  const Points moved = transform_points(origin, composed);
  REQUIRE(std::abs(moved(0, 0) - 0.0) <= astra::test::kTolerance);
  REQUIRE(std::abs(moved(0, 1) - 1.0) <= astra::test::kTolerance);
}

TEST_CASE("inverse_transform is the inverse and uses the transpose shortcut", "[se3]") {
  const Matrix4 T = sample_transform();
  const Matrix4 inverse = inverse_transform(T);

  REQUIRE(astra::test::matrices_close(T * inverse, Matrix4::Identity()));
  REQUIRE(astra::test::matrices_close(inverse * T, Matrix4::Identity()));
  REQUIRE(astra::test::all_bit_identical<Matrix3>(rotation_part(inverse),
                                                  rotation_part(T).transpose()));

  Matrix4 bad = T;
  bad(3, 1) = 1e-3;
  REQUIRE_THROWS_AS(inverse_transform(bad), std::invalid_argument);
}

TEST_CASE("transform_points applies T row by row", "[se3]") {
  const Matrix4 T = sample_transform();
  const Matrix3 R = rotation_part(T);
  const Vector3 t = translation_part(T);

  const Eigen::MatrixXd raw = (Eigen::MatrixXd(4, 3) << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, -1.0, -2.0,
                               -3.0, 0.5, 0.25, -0.75)
                                  .finished();
  const Points points = raw;

  const Eigen::RowVector3d translation_row(t[0], t[1], t[2]);
  // NOTE on the expectation: Eigen does NOT broadcast a row vector in
  // operator+; the idiom is rowwise(). Writing `product + row_vector` compiles,
  // and because NDEBUG removes Eigen's shape assertion in a Release build it
  // silently reads out of bounds instead of aborting -- which is exactly how
  // this expectation was first written and why it produced wrong numbers.
  REQUIRE(astra::test::matrices_close(
      transform_points(points, T), Points((points * R.transpose()).rowwise() + translation_row)));

  const Points single = raw.topRows(1);
  REQUIRE(astra::test::matrices_close(
      transform_points(single, T), Points((single * R.transpose()).rowwise() + translation_row)));

  Points with_nan = points;
  with_nan(0, 2) = kNotANumber;
  REQUIRE_THROWS_AS(transform_points(with_nan, T), std::invalid_argument);
}

TEST_CASE("identity_transform is the identity", "[se3]") {
  const Matrix4 I = identity_transform();
  REQUIRE(last_row_is_exactly_structural(I));
  REQUIRE(astra::test::all_bit_identical<Matrix4>(I, Matrix4::Identity()));

  const Matrix4 T = sample_transform();
  REQUIRE(astra::test::all_bit_identical<Matrix4>(compose_transforms(I, T), T));
  REQUIRE(astra::test::all_bit_identical<Matrix4>(compose_transforms(T, I), T));
}
