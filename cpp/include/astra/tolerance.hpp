// Shared types and the tolerance contract of the planned C++20 core.
//
// This header is part of the core: it must not include any JSON, Catch2 or I/O
// header. The conformance runner is the only place allowed to know about JSON.

#pragma once

#include <cmath>

namespace astra {

/// Absolute tolerance, as a distinct type on purpose.
///
/// The frozen Python API declares `atol` keyword-only (docs/conventions.md
/// §7.2): a bare float after the primary argument is a silent misbinding, read
/// as a second data argument and invisible to round-trip tests. C++ has no
/// keyword arguments, so the same protection comes from the type system: a
/// plain `double` cannot be passed where a `Tolerance` is expected, and a
/// `Tolerance` never decays into a number.
///
/// `Tolerance` deliberately does NOT validate in its constructor, because that
/// would change the semantics rather than transplant them. In Python:
///   - predicates  : a non-finite or negative atol makes them return False;
///   - constructors: it makes them raise ValueError.
/// Validating on construction would make `is_rotation_matrix(R, Tolerance{-1})`
/// throw where Python returns false. The contract is therefore checked at the
/// point of use, exactly where Python checks it, and this type only transports
/// the value. There is no default constructor: `Tolerance{}` does not compile,
/// so nobody can silently ask for "the default" and obtain zero.
struct Tolerance {
  constexpr explicit Tolerance(double value_in) noexcept : value{value_in} {}
  double value;
};

/// The shared reference tolerance, 1e-8: the default of every Python operation
/// that takes one (rotation.py, quaternions.py, transforms.py all agree).
inline constexpr double kReferenceAtolValue = 1e-8;
inline constexpr Tolerance kReferenceAtol{kReferenceAtolValue};

/// §7.2: "a non-finite or negative tolerance is a violated contract".
[[nodiscard]] inline bool is_valid_tolerance(Tolerance atol) noexcept {
  return std::isfinite(atol.value) && atol.value >= 0.0;
}

}  // namespace astra
