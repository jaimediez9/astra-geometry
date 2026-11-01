// Conformance runner for the C++20 spike.
//
// It consumes the SAME fixture file the Python reference writes
// (benchmarks/reference/vectors.json, schema 2) and the frozen schema-1 document
// (benchmarks/reference/vectors-schema1.json), without modifying either. Both
// schemas go through one reader and one code path, and compatibility is
// established by RUNNING this program against both files and comparing the
// per-vector outcomes -- never by inspecting the documents.
//
// Coverage is reported honestly: vectors outside SO(3)/SE(3)/quaternions, and
// vectors carrying an error contract, are listed as not covered with the reason,
// so "everything passes" can never mean "I only ran what I wrote".
//
// Usage:
//   conformance --fixtures <path> [--label <name>] [--out <report.json>]
//               [--expect-fixture-sha <hex>]
//               [--compare-outcomes-with <other-report.json>]
//
// Exit status: 0 when every covered vector passed and the digest matched,
// 1 otherwise, 2 for a usage or harness error.

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "astra_build_info.hpp"

#include "dispatch.hpp"
#include "fixture_io.hpp"
#include "sha256.hpp"

namespace {

using astra::harness::json;
using astra::harness::Comparison;
using astra::harness::fixture_error;

struct Options {
  std::string fixtures;
  std::string label;
  std::string out;
  std::string expected_sha;
  std::string compare_outcomes_with;
};

struct VectorOutcome {
  std::string name;
  std::string function;
  std::string module;
  std::string status;  // "pass", "fail", "not_covered"
  std::string reason;
  Comparison comparison;
};

struct Report {
  std::string label;
  std::string fixture_file;
  std::string fixture_sha256;
  int declared_schema{0};
  int total_vectors{0};
  int covered{0};
  int passed{0};
  int failed{0};
  int not_covered{0};
  std::vector<VectorOutcome> outcomes;
  json predicate_margins = json::array();
  std::map<std::string, double> worst_delta_by_function;
};

[[nodiscard]] std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw fixture_error("cannot open '" + path + "'");
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        throw fixture_error(std::string("missing value for ") + what);
      }
      return argv[++i];
    };
    if (argument == "--fixtures") {
      options.fixtures = next("--fixtures");
    } else if (argument == "--label") {
      options.label = next("--label");
    } else if (argument == "--out") {
      options.out = next("--out");
    } else if (argument == "--expect-fixture-sha") {
      options.expected_sha = next("--expect-fixture-sha");
    } else if (argument == "--compare-outcomes-with") {
      options.compare_outcomes_with = next("--compare-outcomes-with");
    } else {
      throw fixture_error("unknown argument '" + argument + "'");
    }
  }
  if (options.fixtures.empty()) {
    throw fixture_error("--fixtures <path> is required");
  }
  return options;
}

/// The discriminating quantity behind each predicate, reported next to its
/// threshold so a passing test is visibly not a coin flip. This is the margin
/// report promised in the design: a predicate is only meaningful if it is far
/// from its boundary.
void add_predicate_margins(const std::string& function, const json& inputs, Report& report) {
  const double default_atol = astra::kReferenceAtolValue;
  auto push = [&](const std::string& quantity, double value, double threshold) {
    report.predicate_margins.push_back(
        {{"function", function}, {"quantity", quantity}, {"value", value},
         {"threshold", threshold}, {"margin", threshold - value},
         {"quantity_is_exact", threshold == 0.0}});
  };

  if (function == "is_rotation_matrix" || function == "is_transformation_matrix") {
    const astra::Matrix3 R = function == "is_rotation_matrix"
                                 ? astra::harness::matrix3_from(inputs, "R")
                                 : astra::harness::matrix4_from(inputs, "T").topLeftCorner<3, 3>();
    const astra::Matrix3 gram = R.transpose() * R;
    push("max|R^T R - I|", (gram - astra::Matrix3::Identity()).cwiseAbs().maxCoeff(), default_atol);
    push("|det(R) - 1|", std::abs(R.determinant() - 1.0), default_atol);
    if (function == "is_transformation_matrix") {
      const astra::Matrix4 T = astra::harness::matrix4_from(inputs, "T");
      const astra::Vector4 target(0.0, 0.0, 0.0, 1.0);
      push("max|last row - [0,0,0,1]| (exact)", (T.row(3).transpose() - target).cwiseAbs().maxCoeff(),
           0.0);
    }
  } else if (function == "is_unit_quaternion") {
    const astra::Vector4 q = astra::harness::vector4_from(inputs, "q");
    push("||q| - 1|", std::abs(q.norm() - 1.0), default_atol);
  }
}

[[nodiscard]] Report run_vectors(const Options& options, const json& document,
                                 const std::string& digest) {
  Report report;
  report.label = options.label;
  report.fixture_file = options.fixtures;
  report.fixture_sha256 = digest;
  report.declared_schema = document.value("schema", 0);
  report.total_vectors = static_cast<int>(document.at("vectors").size());

  for (const auto& vector : document.at("vectors")) {
    const std::string name = vector.at("name").get<std::string>();
    const std::string module = vector.at("module").get<std::string>();
    const std::string function = vector.at("function").get<std::string>();

    VectorOutcome outcome;
    outcome.name = name;
    outcome.module = module;
    outcome.function = function;

    if (!astra::harness::is_module_in_scope(module)) {
      outcome.status = "not_covered";
      outcome.reason = "module '" + module + "' is outside this spike (SO(3)/SE(3)/quaternions)";
      ++report.not_covered;
      report.outcomes.push_back(outcome);
      continue;
    }
    if (vector.contains("expected_error")) {
      outcome.status = "not_covered";
      outcome.reason = "error contract: this spike covers value contracts only";
      ++report.not_covered;
      report.outcomes.push_back(outcome);
      continue;
    }

    ++report.covered;
    const json& inputs = vector.at("inputs");

    if (function == "is_rotation_matrix" || function == "is_unit_quaternion" ||
        function == "is_transformation_matrix") {
      add_predicate_margins(function, inputs, report);
    }

    try {
      const astra::harness::Value produced = astra::harness::dispatch(function, inputs);
      const json& expected = vector.at("expected");
      const std::string compare = vector.at("compare").get<std::string>();

      if (compare == "exact") {
        if (!std::holds_alternative<bool>(produced)) {
          outcome.status = "fail";
          outcome.comparison.detail = "operation produced a matrix where the contract is boolean";
        } else {
          outcome.comparison = astra::harness::compare_boolean(std::get<bool>(produced), expected, name);
          outcome.status = outcome.comparison.passed ? "pass" : "fail";
        }
      } else if (compare == "close") {
        const double atol = vector.value("atol", 1e-12);
        const Eigen::MatrixXd reference = astra::harness::as_matrix(expected, name);
        if (std::holds_alternative<bool>(produced)) {
          outcome.status = "fail";
          outcome.comparison.detail = "operation produced a boolean where the contract is numeric";
        } else {
          outcome.comparison =
              astra::harness::compare_close(std::get<Eigen::MatrixXd>(produced), reference, atol, name);
          outcome.status = outcome.comparison.passed ? "pass" : "fail";
          report.worst_delta_by_function[function] =
              std::max(report.worst_delta_by_function.count(function)
                           ? report.worst_delta_by_function[function]
                           : 0.0,
                       outcome.comparison.max_abs_delta);
        }
      } else {
        outcome.status = "fail";
        outcome.comparison.detail = "unknown compare mode '" + compare + "'";
      }
    } catch (const std::invalid_argument& error) {
      outcome.status = "fail";
      outcome.comparison.detail = std::string("unexpected std::invalid_argument: ") + error.what();
    }

    if (outcome.status == "pass") {
      ++report.passed;
    } else {
      ++report.failed;
    }
    report.outcomes.push_back(outcome);
  }

  return report;
}

[[nodiscard]] json to_json(const Report& report) {
  json document;
  document["label"] = report.label;
  document["fixture_file"] = report.fixture_file;
  document["fixture_sha256"] = report.fixture_sha256;
  document["declared_schema"] = report.declared_schema;
  document["total_vectors"] = report.total_vectors;
  document["covered"] = report.covered;
  document["passed"] = report.passed;
  document["failed"] = report.failed;
  document["not_covered"] = report.not_covered;
  document["toolchain"] = {
      {"compiler_id", ASTRA_COMPILER_ID},
      {"compiler_version", ASTRA_COMPILER_VERSION},
      {"build_type", ASTRA_BUILD_TYPE},
      {"cxx_standard", ASTRA_CXX_STANDARD},
      {"cxx_flags", ASTRA_CXX_FLAGS},
      {"cmake", ASTRA_CMAKE_VERSION},
      {"eigen", ASTRA_EIGEN_VERSION},
      {"eigen_source", ASTRA_EIGEN_SOURCE},
      {"eigen_tag", ASTRA_EIGEN_TAG},
      {"catch2", ASTRA_CATCH2_VERSION},
      {"catch2_tag", ASTRA_CATCH2_TAG},
      {"json", ASTRA_JSON_VERSION},
      {"json_tag", ASTRA_JSON_TAG},
  };

  json vectors = json::array();
  for (const auto& outcome : report.outcomes) {
    json entry = {{"name", outcome.name},
                  {"function", outcome.function},
                  {"module", outcome.module},
                  {"status", outcome.status}};
    if (!outcome.reason.empty()) {
      entry["reason"] = outcome.reason;
    }
    if (!outcome.comparison.detail.empty()) {
      entry["mode"] = outcome.comparison.mode;
      entry["atol"] = outcome.comparison.atol;
      entry["max_abs_delta"] = outcome.comparison.max_abs_delta;
      entry["detail"] = outcome.comparison.detail;
    }
    vectors.push_back(entry);
  }
  document["vectors"] = vectors;
  document["predicate_margins"] = report.predicate_margins;

  json worst = json::object();
  for (const auto& [function, delta] : report.worst_delta_by_function) {
    worst[function] = delta;
  }
  document["worst_abs_delta_by_function"] = worst;
  return document;
}

/// Pairwise outcome comparison between two runs: this is how schema 1 and
/// schema 2 compatibility is asserted, by execution of the same operations and
/// agreement of the results, not by reading the two documents.
[[nodiscard]] int compare_outcomes(const Report& current, const std::string& other_path) {
  const json other = json::parse(read_file(other_path));
  std::map<std::string, std::string> other_status;
  for (const auto& entry : other.at("vectors")) {
    other_status[entry.at("name").get<std::string>()] = entry.at("status").get<std::string>();
  }

  int mismatches = 0;
  int compared = 0;
  for (const auto& outcome : current.outcomes) {
    const auto found = other_status.find(outcome.name);
    if (found == other_status.end()) {
      continue;
    }
    ++compared;
    if (found->second != outcome.status) {
      ++mismatches;
      std::cout << "  MISMATCH " << outcome.name << ": this run says " << outcome.status
                << ", the other run says " << found->second << "\n";
    }
  }
  std::cout << "outcome comparison against " << other_path << ": " << compared
            << " shared vectors, " << mismatches << " mismatches\n";
  return mismatches == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  try {
    options = parse_options(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "usage error: " << error.what() << "\n";
    return 2;
  }

  try {
    const std::string contents = read_file(options.fixtures);
    const std::string digest = astra::harness::sha256_hex(contents);

    if (!options.expected_sha.empty() && options.expected_sha != digest) {
      std::cerr << "fixture digest mismatch: expected " << options.expected_sha << ", got " << digest
                << "\n  the run is invalid: the fixture is not the one the report is pinned to\n";
      return 1;
    }

    const json document = json::parse(contents);
    const Report report = run_vectors(options, document, digest);

    std::cout << "astra spike conformance";
    if (!options.label.empty()) {
      std::cout << " [" << options.label << "]";
    }
    std::cout << "\n  fixture      : " << options.fixtures << "\n"
              << "  sha256       : " << digest << "\n"
              << "  schema       : " << report.declared_schema << "\n"
              << "  vectors      : " << report.total_vectors << " total, " << report.covered
              << " covered by this spike, " << report.not_covered << " not covered\n"
              << "  result       : " << report.passed << " passed, " << report.failed << " failed\n";

    if (report.failed > 0) {
      std::cout << "  failures:\n";
      for (const auto& outcome : report.outcomes) {
        if (outcome.status == "fail") {
          std::cout << "    - " << outcome.name << " (" << outcome.function << "): "
                    << outcome.comparison.detail << "\n";
        }
      }
    }

    if (!report.worst_delta_by_function.empty()) {
      std::cout << "  largest |C++ - Python| per function:\n";
      for (const auto& [function, delta] : report.worst_delta_by_function) {
        std::cout << "    " << function << ": " << delta << "\n";
      }
    }

    if (!report.predicate_margins.empty()) {
      std::cout << "  predicate margins (distance from the boundary):\n";
      for (const auto& margin : report.predicate_margins) {
        std::cout << "    " << margin.at("function").get<std::string>() << " / "
                  << margin.at("quantity").get<std::string>() << ": value "
                  << margin.at("value").get<double>() << ", threshold "
                  << margin.at("threshold").get<double>();
        if (!margin.at("quantity_is_exact").get<bool>()) {
          std::cout << ", margin " << margin.at("margin").get<double>();
        } else {
          std::cout << " (exact structural check)";
        }
        std::cout << "\n";
      }
    }

    const json document_out = to_json(report);
    if (!options.out.empty()) {
      std::ofstream file(options.out);
      file << document_out.dump(2) << "\n";
      std::cout << "  report written to " << options.out << "\n";
    }

    int status = report.failed == 0 ? 0 : 1;
    if (!options.compare_outcomes_with.empty()) {
      status = std::max(status, compare_outcomes(report, options.compare_outcomes_with));
    }
    return status;
  } catch (const fixture_error& error) {
    std::cerr << "harness error: " << error.what() << "\n";
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "unexpected error: " << error.what() << "\n";
    return 2;
  }
}
