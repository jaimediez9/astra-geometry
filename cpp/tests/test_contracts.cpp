// Contract tests for the tolerance type and the shape of the ported API.
//
// These are the checks that make "the C++ API does not offer more, or less, than
// the frozen Python API" a compile-time fact rather than a claim.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include "astra/quaternion.hpp"
#include "astra/se3.hpp"
#include "astra/so3.hpp"
#include "astra/tolerance.hpp"
#include "test_helpers.hpp"

namespace {

using astra::Matrix3;
using astra::Matrix4;
using astra::Tolerance;
using astra::Vector3;
using astra::Vector4;
using astra::kReferenceAtol;
using astra::kReferenceAtolValue;

const double kNotANumber = std::numeric_limits<double>::quiet_NaN();
const double kInfinity = std::numeric_limits<double>::infinity();

const Matrix3 kIdentity3 = Matrix3::Identity();
const Matrix4 kIdentity4 = Matrix4::Identity();
const Vector4 kUnitQuaternion(1.0, 0.0, 0.0, 0.0);
const Vector3 kAxis(0.3, -0.5, 0.81);

// --- the type itself ---------------------------------------------------------

static_assert(!std::is_default_constructible_v<Tolerance>,
              "Tolerance{} must not compile: nobody should silently get 0 where the "
              "reference default 1e-8 was meant");
static_assert(!std::is_convertible_v<double, Tolerance>,
              "a bare double must never bind to a tolerance (the keyword-only rule of §7.2)");
static_assert(!std::is_convertible_v<Tolerance, double>,
              "a tolerance must not decay into a number");
static_assert(std::is_same_v<decltype(kReferenceAtol.value), double>);

/// The eleven Python operations that take `atol` must take a Tolerance here.
static_assert(std::is_invocable_v<decltype(&astra::so3::is_rotation_matrix), Matrix3, Tolerance>);
static_assert(
    std::is_invocable_v<decltype(&astra::se3::is_transformation_matrix), Matrix4, Tolerance>);
static_assert(
    std::is_invocable_v<decltype(&astra::quaternion::is_unit_quaternion), Vector4, Tolerance>);
static_assert(std::is_invocable_v<decltype(&astra::se3::transformation_matrix), Matrix3, Vector3,
                                  Tolerance>);
static_assert(std::is_invocable_v<decltype(&astra::se3::rotation_part), Matrix4, Tolerance>);
static_assert(std::is_invocable_v<decltype(&astra::se3::translation_part), Matrix4, Tolerance>);
static_assert(std::is_invocable_v<decltype(&astra::se3::compose_transforms), Matrix4, Matrix4,
                                  Tolerance>);
static_assert(std::is_invocable_v<decltype(&astra::se3::inverse_transform), Matrix4, Tolerance>);
static_assert(std::is_invocable_v<decltype(&astra::se3::transform_points), astra::Points, Matrix4,
                                  Tolerance>);
static_assert(std::is_invocable_v<decltype(&astra::quaternion::rotation_matrix_to_quaternion),
                                  Matrix3, Tolerance>);
static_assert(std::is_invocable_v<decltype(&astra::quaternion::quaternion_to_rotation_matrix),
                                  Vector4, Tolerance>);

/// The nine that do NOT take a tolerance in Python must not accept one here
/// either: accepting one would be a wider API than the frozen reference.
static_assert(!std::is_invocable_v<decltype(&astra::so3::rotation_matrix3D), Vector3, Tolerance>);
static_assert(!std::is_invocable_v<decltype(&astra::so3::rotation_matrix2D), Tolerance>);
static_assert(!std::is_invocable_v<decltype(&astra::so3::rotate_points), astra::Points, Vector3,
                                   Tolerance>);
static_assert(!std::is_invocable_v<decltype(&astra::so3::compose_rotations), Matrix3, Matrix3,
                                   Tolerance>);
static_assert(!std::is_invocable_v<decltype(&astra::so3::inverse_rotation), Matrix3, Tolerance>);
static_assert(!std::is_invocable_v<decltype(&astra::quaternion::normalize_quaternion), Vector4,
                                   Tolerance>);
static_assert(!std::is_invocable_v<decltype(&astra::quaternion::conjugate_quaternion), Vector4,
                                   Tolerance>);
static_assert(!std::is_invocable_v<decltype(&astra::quaternion::quaternion_multiply), Vector4,
                                   Vector4, Tolerance>);
static_assert(!std::is_invocable_v<decltype(&astra::se3::identity_transform), Tolerance>);

// --- the value ---------------------------------------------------------------

TEST_CASE("the reference tolerance is the one the Python modules share", "[tolerance]") {
  REQUIRE(kReferenceAtol.value == 1e-8);
  REQUIRE(kReferenceAtolValue == 1e-8);
}

TEST_CASE("a non-finite or negative tolerance makes predicates return false", "[tolerance]") {
  // §7.2: predicates return False, they do not raise.
  for (const double bad : {-1.0, kNotANumber, kInfinity, -kInfinity}) {
    REQUIRE_FALSE(astra::so3::is_rotation_matrix(kIdentity3, Tolerance{bad}));
    REQUIRE_FALSE(astra::quaternion::is_unit_quaternion(kUnitQuaternion, Tolerance{bad}));
    REQUIRE_FALSE(astra::se3::is_transformation_matrix(kIdentity4, Tolerance{bad}));
  }
}

TEST_CASE("a non-finite or negative tolerance makes constructors throw", "[tolerance]") {
  for (const double bad : {-1.0, kNotANumber, kInfinity, -kInfinity}) {
    REQUIRE_THROWS_AS(astra::se3::transformation_matrix(kIdentity3, Vector3(1.0, 2.0, 3.0),
                                                        Tolerance{bad}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(astra::se3::rotation_part(kIdentity4, Tolerance{bad}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(astra::se3::translation_part(kIdentity4, Tolerance{bad}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(astra::se3::compose_transforms(kIdentity4, kIdentity4, Tolerance{bad}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(astra::se3::inverse_transform(kIdentity4, Tolerance{bad}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(astra::quaternion::rotation_matrix_to_quaternion(kIdentity3, Tolerance{bad}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        astra::quaternion::quaternion_to_rotation_matrix(kUnitQuaternion, Tolerance{bad}),
        std::invalid_argument);
  }
}

TEST_CASE("zero is a valid tolerance, tighter than the default", "[tolerance]") {
  // Python accepts atol = 0.0; it is a stricter contract, not a violated one.
  REQUIRE(astra::is_valid_tolerance(Tolerance{0.0}));
  REQUIRE(astra::so3::is_rotation_matrix(kIdentity3, Tolerance{0.0}));
  REQUIRE(astra::quaternion::is_unit_quaternion(kUnitQuaternion, Tolerance{0.0}));
  REQUIRE(astra::se3::is_transformation_matrix(kIdentity4, Tolerance{0.0}));
}

TEST_CASE("the default and an explicit default are bit-identical", "[tolerance]") {
  // A default argument that behaved even one ULP differently from the explicit
  // value would be a silent divergence for every fixture that omits `atol`.
  const Tolerance explicit_default{kReferenceAtolValue};
  const astra::Points points = (Eigen::MatrixXd(2, 3) << 0.5, -1.5, 2.0, 3.0, 0.25, -0.75)
                                   .finished();
  const Matrix3 rotation = astra::so3::rotation_matrix3D(kAxis, 0.7);
  const Matrix4 transform =
      astra::se3::transformation_matrix(rotation, Vector3(0.1, -0.2, 0.3));

  SECTION("SO(3)") {
    REQUIRE(astra::so3::is_rotation_matrix(rotation) ==
            astra::so3::is_rotation_matrix(rotation, explicit_default));
  }
  SECTION("SE(3) accessors") {
    REQUIRE(astra::test::all_bit_identical<Matrix3>(astra::se3::rotation_part(transform),
                                                    astra::se3::rotation_part(transform,
                                                                             explicit_default)));
    REQUIRE(astra::test::all_bit_identical<Vector3>(
        astra::se3::translation_part(transform),
        astra::se3::translation_part(transform, explicit_default)));
  }
  SECTION("SE(3) composition and inversion") {
    REQUIRE(astra::test::all_bit_identical<Matrix4>(
        astra::se3::compose_transforms(transform, transform),
        astra::se3::compose_transforms(transform, transform, explicit_default)));
    REQUIRE(astra::test::all_bit_identical<Matrix4>(
        astra::se3::inverse_transform(transform),
        astra::se3::inverse_transform(transform, explicit_default)));
  }
  SECTION("point batches") {
    REQUIRE(astra::test::all_bit_identical<astra::Points>(
        astra::se3::transform_points(points, transform),
        astra::se3::transform_points(points, transform, explicit_default)));
  }
  SECTION("quaternion conversions") {
    REQUIRE(astra::test::all_bit_identical<Vector4>(
        astra::quaternion::rotation_matrix_to_quaternion(rotation),
        astra::quaternion::rotation_matrix_to_quaternion(rotation, explicit_default)));
    const Vector4 q = astra::quaternion::rotation_matrix_to_quaternion(rotation);
    REQUIRE(astra::test::all_bit_identical<Matrix3>(
        astra::quaternion::quaternion_to_rotation_matrix(q),
        astra::quaternion::quaternion_to_rotation_matrix(q, explicit_default)));
  }
}

}  // namespace
