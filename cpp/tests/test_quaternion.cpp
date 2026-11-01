// Property tests for the quaternion algebra: the scalar-first convention, the
// Hamilton product and, above all, the multiplication ORDER, which is the one
// convention a compiler cannot check for you.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "astra/quaternion.hpp"
#include "astra/so3.hpp"
#include "astra/tolerance.hpp"
#include "test_helpers.hpp"

namespace {

using astra::Matrix3;
using astra::Tolerance;
using astra::Vector3;
using astra::Vector4;

const double kNotANumber = std::numeric_limits<double>::quiet_NaN();
const double kInfinity = std::numeric_limits<double>::infinity();

[[nodiscard]] Vector4 quaternion_from_axis_angle(const Vector3& axis, double theta) {
  const Matrix3 R = astra::so3::rotation_matrix3D(axis, theta);
  return astra::quaternion::rotation_matrix_to_quaternion(R);
}

}  // namespace

TEST_CASE("is_unit_quaternion checks the norm against the tolerance", "[quaternion]") {
  REQUIRE(astra::quaternion::is_unit_quaternion(Vector4(1.0, 0.0, 0.0, 0.0)));
  REQUIRE(astra::quaternion::is_unit_quaternion(quaternion_from_axis_angle({0.3, -0.5, 0.81}, 0.7)));
  REQUIRE_FALSE(astra::quaternion::is_unit_quaternion(Vector4(2.0, 0.0, 0.0, 0.0)));
  REQUIRE_FALSE(astra::quaternion::is_unit_quaternion(Vector4(1.0, 1.0, 0.0, 0.0)));
  REQUIRE_FALSE(astra::quaternion::is_unit_quaternion(Vector4(1.0, 0.0, kNotANumber, 0.0)));
  // A norm inside the default tolerance is accepted, and rejected with a
  // tighter one: 1e-9 > 1e-12.
  REQUIRE(astra::quaternion::is_unit_quaternion(Vector4(1.0 + 1e-9, 0.0, 0.0, 0.0)));
  REQUIRE_FALSE(
      astra::quaternion::is_unit_quaternion(Vector4(1.0 + 1e-9, 0.0, 0.0, 0.0), Tolerance{1e-12}));
}

TEST_CASE("normalize_quaternion returns a unit quaternion of the same direction",
          "[quaternion]") {
  const Vector4 q = astra::quaternion::normalize_quaternion(Vector4(2.0, -3.0, 4.0, -5.0));
  REQUIRE(std::abs(q.norm() - 1.0) <= astra::test::kTolerance);

  // The direction is preserved: the normalised result is the input scaled by a
  // positive factor.
  const Vector4 original(2.0, -3.0, 4.0, -5.0);
  const double factor = q[0] / original[0];
  REQUIRE(factor > 0.0);
  for (int i = 0; i < 4; ++i) {
    REQUIRE(std::abs(q[i] - factor * original[i]) <= astra::test::kTolerance);
  }

  REQUIRE_THROWS_AS(astra::quaternion::normalize_quaternion(Vector4(0.0, 0.0, 0.0, 0.0)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(astra::quaternion::normalize_quaternion(Vector4(kInfinity, 0.0, 0.0, 0.0)),
                    std::invalid_argument);
}

TEST_CASE("normalize_quaternion survives a norm that would overflow", "[quaternion]") {
  // Scaling by the largest component first is what makes this work; computing
  // the norm directly would overflow to infinity.
  const double huge = 1e200;
  const Vector4 q = astra::quaternion::normalize_quaternion(Vector4(huge, huge, huge, huge));
  REQUIRE(q.allFinite());
  REQUIRE(std::abs(q.norm() - 1.0) <= astra::test::kTolerance);
}

TEST_CASE("conjugate_quaternion inverts a unit quaternion", "[quaternion]") {
  const Vector4 q = quaternion_from_axis_angle({0.3, -0.5, 0.81}, 0.7);
  const Vector4 conjugate = astra::quaternion::conjugate_quaternion(q);

  REQUIRE(conjugate[0] == q[0]);
  REQUIRE(conjugate[1] == -q[1]);
  REQUIRE(conjugate[2] == -q[2]);
  REQUIRE(conjugate[3] == -q[3]);

  // The product is the identity to round-off, NOT bit for bit: the scalar part
  // lands on exactly 1 (it is the squared norm, computed as a sum of squares),
  // while the vector part keeps a residual around 1e-17 from the cancellation of
  // the cross terms. Measured here: (1, 0, -1.39e-17, 0).
  const Vector4 identity = Vector4(1.0, 0.0, 0.0, 0.0);
  const Vector4 product = astra::quaternion::quaternion_multiply(q, conjugate);
  REQUIRE(astra::test::matrices_close(product, identity, 1e-16));
  REQUIRE(std::abs(product.norm() - 1.0) <= astra::test::kTolerance);

  REQUIRE_THROWS_AS(astra::quaternion::conjugate_quaternion(Vector4(1.0, 0.0, 0.0, kNotANumber)),
                    std::invalid_argument);
}

TEST_CASE("quaternion_multiply follows the composition order of SE(3)", "[quaternion]") {
  // quaternion_multiply(q1, q2) must be q2 * q1, whose rotation is R2 @ R1,
  // matching compose_rotations. This single test is what pins the convention.
  const Matrix3 R1 = astra::so3::rotation_matrix3D(Vector3(1.0, 0.0, 0.0), 0.9);
  const Matrix3 R2 = astra::so3::rotation_matrix3D(Vector3(0.0, 1.0, 0.0), 0.4);
  const Vector4 q1 = astra::quaternion::rotation_matrix_to_quaternion(R1);
  const Vector4 q2 = astra::quaternion::rotation_matrix_to_quaternion(R2);

  const Vector4 product = astra::quaternion::quaternion_multiply(q1, q2);
  REQUIRE(std::abs(product.norm() - 1.0) <= astra::test::kTolerance);

  const Matrix3 from_quaternion = astra::quaternion::quaternion_to_rotation_matrix(product);
  REQUIRE(astra::test::matrices_close(from_quaternion, astra::so3::compose_rotations(R1, R2)));
  // ...and the other order does not agree, so the assertion is not vacuous.
  REQUIRE_FALSE(astra::test::matrices_close(from_quaternion, astra::so3::compose_rotations(R2, R1),
                                            1e-6));
}

TEST_CASE("quaternion_multiply composes the norms", "[quaternion]") {
  const Vector4 a(1.0, 2.0, 3.0, 4.0);
  const Vector4 b(-2.0, 0.5, 1.5, -0.25);
  const Vector4 product = astra::quaternion::quaternion_multiply(a, b);
  REQUIRE(std::abs(product.norm() - a.norm() * b.norm()) <= 1e-12);

  REQUIRE_THROWS_AS(astra::quaternion::quaternion_multiply(Vector4(1.0, 0.0, 0.0, kNotANumber), b),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(astra::quaternion::quaternion_multiply(a, Vector4(1.0, 0.0, 0.0, kNotANumber)),
                    std::invalid_argument);
}

TEST_CASE("matrix and quaternion conversions round-trip", "[quaternion]") {
  for (const double angle : {0.1, 0.7, 1.7, 2.9, -2.2}) {
    for (const Vector3& axis : {Vector3(1.0, 0.0, 0.0), Vector3(0.0, 1.0, 0.0),
                                Vector3(0.3, -0.5, 0.81), Vector3(-0.4, 0.2, 0.9)}) {
      const Matrix3 R = astra::so3::rotation_matrix3D(axis, angle);
      const Vector4 q = astra::quaternion::rotation_matrix_to_quaternion(R);
      REQUIRE(std::abs(q.norm() - 1.0) <= astra::test::kTolerance);
      REQUIRE(astra::test::matrices_close(astra::quaternion::quaternion_to_rotation_matrix(q), R));
    }
  }
}

TEST_CASE("q and -q describe the same rotation", "[quaternion]") {
  const Vector4 q = quaternion_from_axis_angle({0.3, -0.5, 0.81}, 1.25);
  REQUIRE(astra::test::matrices_close(astra::quaternion::quaternion_to_rotation_matrix(q),
                                      astra::quaternion::quaternion_to_rotation_matrix(-q)));
}

TEST_CASE("rotation_matrix_to_quaternion rejects a non-rotation and a bad tolerance",
          "[quaternion]") {
  Matrix3 not_a_rotation = Matrix3::Identity();
  not_a_rotation(2, 2) = 0.5;
  REQUIRE_THROWS_AS(astra::quaternion::rotation_matrix_to_quaternion(not_a_rotation),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      astra::quaternion::rotation_matrix_to_quaternion(Matrix3::Identity(), Tolerance{-1.0}),
      std::invalid_argument);
}

TEST_CASE("a half turn is reconstructed exactly even though its sign is not decidable",
          "[quaternion]") {
  // documented limitation: for w == 0 both q and -q are valid and the sign is
  // decided by the extraction branch. Only the reconstructed rotation is
  // promised to be stable, and that is what is asserted here.
  const Matrix3 half_turn = astra::so3::rotation_matrix3D(Vector3(0.0, 1.0, 0.0), 3.14159265358979);
  const Vector4 q = astra::quaternion::rotation_matrix_to_quaternion(half_turn);
  REQUIRE(std::abs(q[0]) <= 1e-8);
  REQUIRE(astra::test::matrices_close(astra::quaternion::quaternion_to_rotation_matrix(q),
                                      half_turn, 1e-11));
}

TEST_CASE("quaternion_to_rotation_matrix rejects a non-unit quaternion", "[quaternion]") {
  REQUIRE_THROWS_AS(
      astra::quaternion::quaternion_to_rotation_matrix(Vector4(2.0, 0.0, 0.0, 0.0)),
      std::invalid_argument);
  REQUIRE_THROWS_AS(
      astra::quaternion::quaternion_to_rotation_matrix(Vector4(1.0, 0.0, 0.0, 0.0), Tolerance{-1.0}),
      std::invalid_argument);
}
