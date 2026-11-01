// Fixture reading and comparison, harness-side only.
//
// This file is the ONLY place in the spike that knows about JSON. The core
// headers must stay free of it: the port must not grow a serialization
// dependency it does not need.
//
// The comparison semantics below are a literal reproduction of
// benchmarks/reference.py::_compare_value:
//   - "exact" : booleans compared as booleans, arrays element-wise for equality;
//   - "close" : equal shapes and max|produced - expected| <= atol, rtol = 0.
// The vector's `atol` is a COMPARISON tolerance. It is never forwarded as an
// argument to the operation -- in the reference, `function(**inputs)` is called
// with `inputs` alone. Forwarding it would silently change the operation's
// behaviour and could let a wrong result pass.

#pragma once

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <variant>

#include "astra/types.hpp"

namespace astra::harness {

using json = nlohmann::json;

/// Anything that goes wrong while interpreting a fixture is a harness fault, not
/// a library failure, and is reported as such.
class fixture_error : public std::runtime_error {
 public:
  explicit fixture_error(const std::string& what) : std::runtime_error(what) {}
};

/// A produced value: either a boolean predicate result or a matrix (a matrix is
/// also how vectors and point batches are carried, since every comparison here
/// is element-wise).
using Value = std::variant<bool, Eigen::MatrixXd>;

[[nodiscard]] inline double as_scalar(const json& inputs, const char* key) {
  const auto& node = inputs.at(key);
  if (!node.is_number()) {
    throw fixture_error(std::string("input '") + key + "' is not a number");
  }
  return node.get<double>();
}

[[nodiscard]] inline Eigen::MatrixXd as_matrix(const json& node, const std::string& what) {
  if (!node.is_array() || node.empty()) {
    throw fixture_error(what + ": expected a non-empty array");
  }

  if (node.front().is_number()) {  // a flat vector
    Eigen::MatrixXd out(static_cast<Eigen::Index>(node.size()), 1);
    for (size_t i = 0; i < node.size(); ++i) {
      out(static_cast<Eigen::Index>(i), 0) = node[i].get<double>();
    }
    return out;
  }

  if (!node.front().is_array()) {
    throw fixture_error(what + ": expected an array of numbers or an array of arrays");
  }

  const size_t rows = node.size();
  const size_t cols = node.front().size();
  Eigen::MatrixXd out(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
  for (size_t r = 0; r < rows; ++r) {
    if (node[r].size() != cols) {
      throw fixture_error(what + ": ragged nested array");
    }
    for (size_t c = 0; c < cols; ++c) {
      if (!node[r][c].is_number()) {
        throw fixture_error(what + ": nested element is not a number");
      }
      out(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = node[r][c].get<double>();
    }
  }
  return out;
}

/// Read one input and check the shape the operation needs, failing loudly
/// instead of reshaping silently.
[[nodiscard]] inline Eigen::MatrixXd as_input(const json& inputs, const char* key,
                                              int expected_rows, int expected_cols) {
  Eigen::MatrixXd value = as_matrix(inputs.at(key), std::string("input '") + key + "'");
  if (value.rows() != expected_rows || (expected_cols > 0 && value.cols() != expected_cols)) {
    throw fixture_error(std::string("input '") + key + "': expected shape (" +
                        std::to_string(expected_rows) + ", " + std::to_string(expected_cols) +
                        ") but the fixture has (" + std::to_string(value.rows()) + ", " +
                        std::to_string(value.cols()) + ")");
  }
  return value;
}

[[nodiscard]] inline astra::Matrix3 matrix3_from(const json& inputs, const char* key) {
  return as_input(inputs, key, 3, 3);
}

[[nodiscard]] inline astra::Matrix4 matrix4_from(const json& inputs, const char* key) {
  return as_input(inputs, key, 4, 4);
}

[[nodiscard]] inline astra::Vector3 vector3_from(const json& inputs, const char* key) {
  const Eigen::MatrixXd m = as_input(inputs, key, 3, -1);
  if (m.cols() != 1) {
    throw fixture_error(std::string("input '") + key + "': expected a vector of 3 values");
  }
  astra::Vector3 v;
  v << m(0, 0), m(1, 0), m(2, 0);
  return v;
}

[[nodiscard]] inline astra::Vector4 vector4_from(const json& inputs, const char* key) {
  const Eigen::MatrixXd m = as_input(inputs, key, 4, -1);
  if (m.cols() != 1) {
    throw fixture_error(std::string("input '") + key + "': expected a vector of 4 values");
  }
  astra::Vector4 v;
  v << m(0, 0), m(1, 0), m(2, 0), m(3, 0);
  return v;
}

/// Points arrive as (N, 3); N is whatever the fixture carries.
[[nodiscard]] inline astra::Points points_from(const json& inputs, const char* key) {
  const json& node = inputs.at(key);
  if (!node.is_array()) {
    throw fixture_error(std::string("input '") + key + "': expected an array of points");
  }
  const Eigen::MatrixXd m = as_matrix(node, std::string("input '") + key + "'");
  if (m.cols() != 3) {
    throw fixture_error(std::string("input '") + key + "': expected points of shape (N, 3)");
  }
  return m;
}

/// The verdict of comparing a produced value against a fixture vector.
struct Comparison {
  bool passed{false};
  std::string mode;      ///< "exact" or "close"
  double atol{0.0};      ///< the fixture's comparison tolerance
  double max_abs_delta{0.0};
  std::string detail;    ///< failure reason, or "ok"
};

[[nodiscard]] inline Comparison compare_boolean(bool produced, const json& expected,
                                                const std::string& name) {
  Comparison out;
  out.mode = "exact";
  if (!expected.is_boolean()) {
    out.detail = name + ": exact contract with a non-boolean expected value";
    return out;
  }
  out.passed = (produced == expected.get<bool>());
  out.detail = out.passed ? "ok"
                          : name + ": expected " + expected.dump() + ", produced " +
                                (produced ? std::string("true") : std::string("false"));
  return out;
}

[[nodiscard]] inline Comparison compare_close(const Eigen::MatrixXd& produced,
                                              const Eigen::MatrixXd& expected, double atol,
                                              const std::string& name) {
  Comparison out;
  out.mode = "close";
  out.atol = atol;
  if (produced.rows() != expected.rows() || produced.cols() != expected.cols()) {
    out.detail = name + ": shape (" + std::to_string(produced.rows()) + ", " +
                 std::to_string(produced.cols()) + ") != expected (" +
                 std::to_string(expected.rows()) + ", " + std::to_string(expected.cols()) + ")";
    return out;
  }
  const double delta =
      produced.size() == 0 ? 0.0 : (produced - expected).cwiseAbs().maxCoeff();
  out.max_abs_delta = delta;
  out.passed = (delta <= atol);  // written so that NaN fails, as in the reference
  out.detail = out.passed ? "ok"
                          : name + ": max|delta| = " + std::to_string(delta) + " exceeds atol " +
                                std::to_string(atol);
  return out;
}

}  // namespace astra::harness
