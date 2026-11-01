// Small shared helpers for the native tests.

#pragma once

#include <Eigen/Core>

#include <bit>
#include <cstdint>
#include <type_traits>

namespace astra::test {

/// The spike's own comparison tolerance for native property tests, chosen the
/// same way the fixtures choose theirs: far above double round-off (~1e-16 for
/// the magnitudes involved) and far below any algorithmic difference.
inline constexpr double kTolerance = 1e-12;

[[nodiscard]] inline bool matrices_close(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b,
                                         double tolerance = kTolerance) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return false;
  }
  if (a.size() == 0) {
    return true;
  }
  return (a - b).cwiseAbs().maxCoeff() <= tolerance;
}

/// Bit-level equality of two values, used where the claim is "the default and an
/// explicit default are indistinguishable", which a tolerance-based comparison
/// would not establish.
[[nodiscard]] inline bool bit_identical(double a, double b) {
  return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

/// Bit-for-bit comparison of two Eigen objects OR expressions (a product, a
/// transpose, a block, an identity expression: all of them reach here without
/// being materialised first, so the test compares exactly what was produced).
///
/// The first template parameter is an optional label naming the type the caller
/// believes it is comparing. It takes no part in the comparison; it only asserts
/// that the compared expressions really do produce that type, so a mislabelled
/// call fails to compile instead of silently comparing the wrong thing.
template <typename Label = void, typename Left, typename Right>
[[nodiscard]] bool all_bit_identical(const Left& left, const Right& right) {
  static_assert(std::is_void_v<Label> || (std::is_constructible_v<Label, Left> &&
                                          std::is_constructible_v<Label, Right>),
                "the label passed to all_bit_identical does not describe the compared expressions");

  if (left.rows() != right.rows() || left.cols() != right.cols()) {
    return false;
  }
  for (Eigen::Index row = 0; row < left.rows(); ++row) {
    for (Eigen::Index column = 0; column < left.cols(); ++column) {
      if (!bit_identical(left(row, column), right(row, column))) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace astra::test
