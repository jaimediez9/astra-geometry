// Property tests for SO(3) -- the checks that do not depend on the fixtures:
// orthogonality, determinant, the composition order and the batch application.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "astra/so3.hpp"
#include "astra/tolerance.hpp"
#include "test_helpers.hpp"

namespace {

using astra::Matrix3;
using astra::Points;
using astra::Tolerance;
using astra::Vector3;
using astra::so3::compose_rotations;
using astra::so3::inverse_rotation;
using astra::so3::is_rotation_matrix;
using astra::so3::rotate_points;
using astra::so3::rotation_matrix2D;
using astra::so3::rotation_matrix3D;

const double kNotANumber = std::numeric_limits<double>::quiet_NaN();
const double kTol = astra::test::kTolerance;

[[nodiscard]] bool is_orthogonal(const Matrix3& R, double tolerance = kTol) {
  return astra::test::matrices_close(R.transpose() * R, Matrix3::Identity(), tolerance);
}

}  // namespace

TEST_CASE("is_rotation_matrix accepts rotations and rejects everything else", "[so3]") {
  REQUIRE(is_rotation_matrix(Matrix3::Identity()));
  REQUIRE(is_rotation_matrix(rotation_matrix3D(Vector3(0.3, -0.5, 0.81), 0.7)));
  REQUIRE(is_rotation_matrix(rotation_matrix3D(Vector3(1.0, 0.0, 0.0), 3.1)));

  // A reflection is orthogonal but has determinant -1: orthogonality alone is
  // not enough, which is why both properties are checked.
  Matrix3 reflection = Matrix3::Identity();
  reflection(2, 2) = -1.0;
  REQUIRE_FALSE(is_rotation_matrix(reflection));

  // Scaled identity: orthogonality fails.
  REQUIRE_FALSE(is_rotation_matrix(2.0 * Matrix3::Identity()));

  // A single perturbed entry, far above any tolerance.
  Matrix3 perturbed = Matrix3::Identity();
  perturbed(0, 1) = 0.5;
  REQUIRE_FALSE(is_rotation_matrix(perturbed));

  // A deviation inside the default tolerance is still accepted: 1e-9 < 1e-8.
  Matrix3 almost = Matrix3::Identity();
  almost(0, 1) = 1e-9;
  REQUIRE(is_rotation_matrix(almost));
  // ...and the same matrix with a tighter tolerance is not.
  REQUIRE_FALSE(is_rotation_matrix(almost, Tolerance{1e-12}));

  Matrix3 not_finite = Matrix3::Identity();
  not_finite(1, 1) = kNotANumber;
  REQUIRE_FALSE(is_rotation_matrix(not_finite));
}

TEST_CASE("rotation_matrix3D produces an element of SO(3)", "[so3]") {
  for (const double angle : {0.0, 0.3, 1.7, -2.2, 3.14159265358979, 6.0}) {
    const Matrix3 R = rotation_matrix3D(Vector3(0.3, -0.5, 0.81), angle);
    REQUIRE(is_orthogonal(R));
    REQUIRE(std::abs(R.determinant() - 1.0) <= kTol);
  }
}

TEST_CASE("rotation_matrix3D normalises the axis", "[so3]") {
  // The reference scales the axis by its largest component and then by the
  // norm, so any non-zero multiple of an axis gives the same rotation.
  const Vector3 axis(0.3, -0.5, 0.81);
  const Matrix3 reference = rotation_matrix3D(axis, 0.7);
  for (const double factor : {1e-3, 0.5, 2.0, 1e3}) {
    REQUIRE(astra::test::matrices_close(rotation_matrix3D(factor * axis, 0.7), reference));
  }
  REQUIRE(astra::test::matrices_close(rotation_matrix3D(-axis, -0.7), reference, 1e-11));
}

TEST_CASE("rotation_matrix3D rejects a null axis and non-finite input", "[so3]") {
  REQUIRE_THROWS_AS(rotation_matrix3D(Vector3(0.0, 0.0, 0.0), 0.7), std::invalid_argument);
  REQUIRE_THROWS_AS(rotation_matrix3D(Vector3(1.0, kNotANumber, 0.0), 0.7), std::invalid_argument);
  REQUIRE_THROWS_AS(rotation_matrix3D(Vector3(1.0, 0.0, 0.0), kNotANumber), std::invalid_argument);
}

TEST_CASE("rotation_matrix2D is a plane rotation", "[so3]") {
  const double theta = 0.7;
  const astra::Matrix2 R = rotation_matrix2D(theta);
  REQUIRE(std::abs(R(0, 0) - std::cos(theta)) <= kTol);
  REQUIRE(std::abs(R(0, 1) + std::sin(theta)) <= kTol);
  REQUIRE(std::abs(R(1, 0) - std::sin(theta)) <= kTol);
  REQUIRE(std::abs(R(1, 1) - std::cos(theta)) <= kTol);
  REQUIRE_THROWS_AS(rotation_matrix2D(kNotANumber), std::invalid_argument);
}

TEST_CASE("compose_rotations applies R1 first and R2 afterwards", "[so3]") {
  // This is the order that the reference documents and the one the port must
  // reproduce: the result is R2 @ R1. A swap is invisible to round-trip tests
  // and is the failure mode this project calls its most dangerous.
  const Matrix3 about_x = rotation_matrix3D(Vector3(1.0, 0.0, 0.0), 0.9);
  const Matrix3 about_y = rotation_matrix3D(Vector3(0.0, 1.0, 0.0), 0.4);

  const Matrix3 composed = compose_rotations(about_x, about_y);
  REQUIRE(astra::test::all_bit_identical<Matrix3>(composed, about_y * about_x));
  // ...and the two orders differ, so the test above is not vacuous.
  REQUIRE_FALSE(astra::test::matrices_close(composed, about_x * about_y, 1e-6));
}

TEST_CASE("compose_rotations validates both operands", "[so3]") {
  Matrix3 not_a_rotation = Matrix3::Identity();
  not_a_rotation(0, 1) = 0.5;
  REQUIRE_THROWS_AS(compose_rotations(not_a_rotation, Matrix3::Identity()), std::invalid_argument);
  REQUIRE_THROWS_AS(compose_rotations(Matrix3::Identity(), not_a_rotation), std::invalid_argument);
}

TEST_CASE("composition is associative up to rounding", "[so3]") {
  const Matrix3 a = rotation_matrix3D(Vector3(0.3, -0.5, 0.81), 0.7);
  const Matrix3 b = rotation_matrix3D(Vector3(-0.2, 0.9, 0.1), -1.3);
  const Matrix3 c = rotation_matrix3D(Vector3(0.7, 0.7, -0.7), 2.0);

  const Matrix3 left = compose_rotations(compose_rotations(a, b), c);
  const Matrix3 right = compose_rotations(a, compose_rotations(b, c));
  REQUIRE(astra::test::matrices_close(left, right));
}

TEST_CASE("inverse_rotation is the transpose and inverts the rotation", "[so3]") {
  const Matrix3 R = rotation_matrix3D(Vector3(0.3, -0.5, 0.81), 0.7);
  const Matrix3 inverse = inverse_rotation(R);

  REQUIRE(astra::test::all_bit_identical<Matrix3>(inverse, R.transpose()));
  REQUIRE(astra::test::matrices_close(R * inverse, Matrix3::Identity()));

  Matrix3 not_a_rotation = Matrix3::Identity();
  not_a_rotation(1, 1) = 2.0;
  REQUIRE_THROWS_AS(inverse_rotation(not_a_rotation), std::invalid_argument);
}

TEST_CASE("rotate_points applies R to every row", "[so3]") {
  const Vector3 axis(0.3, -0.5, 0.81);
  const double theta = 0.7;
  const Matrix3 R = rotation_matrix3D(axis, theta);

  const Eigen::MatrixXd raw = (Eigen::MatrixXd(3, 3) << 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, -1.0, -2.0,
                               -3.0)
                                  .finished();
  const Points points = raw;

  REQUIRE(astra::test::matrices_close(rotate_points(points, axis, theta), points * R.transpose()));

  // One point is still (1, 3), and an empty batch is still valid.
  const Points single = raw.topRows(1);
  REQUIRE(astra::test::matrices_close(rotate_points(single, axis, theta),
                                      single * R.transpose()));

  Points with_nan = points;
  with_nan(1, 1) = kNotANumber;
  REQUIRE_THROWS_AS(rotate_points(with_nan, axis, theta), std::invalid_argument);
}
