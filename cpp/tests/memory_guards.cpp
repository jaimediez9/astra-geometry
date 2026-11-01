// Allocation guards and the measurements they support.
//
// This translation unit replaces operator new/new[]/delete and overrides
// eigen_assert, so memory_guard.hpp comes FIRST: it must be processed before any
// Eigen header.

#include "memory_guard.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <new>
#include <string>
#include <vector>

#include "astra/quaternion.hpp"
#include "astra/se3.hpp"
#include "astra/so3.hpp"
#include "test_helpers.hpp"

// --- layer 1: the C++ allocator ---------------------------------------------
// Replaced here rather than interposed: it is portable, and it is what sees the
// standard library and our own code. Eigen does NOT use operator new (it calls
// aligned_malloc, which reaches std::malloc), so this layer alone would miss
// every Eigen allocation -- which is the complementarity the report claims.

namespace {

inline void note_cxx_allocation() noexcept {
  auto& counters = astra::guard::counters();
  if (counters.armed.load(std::memory_order_relaxed)) {
    counters.cxx_allocations.fetch_add(1, std::memory_order_relaxed);
  }
}

/// Results are accumulated here, with volatile semantics and external linkage, so
/// that the optimiser cannot eliminate the call being measured. Without a sink
/// this whole measurement would be vacuous at -O2: a pure function whose result
/// nobody reads is exactly what a compiler is entitled to delete.
volatile double g_result_sink = 0.0;

template <typename Derived>
[[nodiscard]] double consume(const Eigen::MatrixBase<Derived>& value) noexcept {
  return value.derived().sum();
}

}  // namespace

void* operator new(std::size_t size) {
  note_cxx_allocation();
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  return pointer;
}

void* operator new[](std::size_t size) {
  // Implemented independently rather than delegating to operator new: a
  // replacement that hands a scalar-new pointer to array-delete is a mismatched
  // pair, which the compiler flags (and rightly so).
  note_cxx_allocation();
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  return pointer;
}

void* operator new(std::size_t size, std::align_val_t alignment) {
  note_cxx_allocation();
  const std::size_t align = static_cast<std::size_t>(alignment);
  const std::size_t rounded = ((size == 0 ? 1 : size) + align - 1) / align * align;
  void* pointer = std::aligned_alloc(align, rounded);
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  return pointer;
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
  note_cxx_allocation();
  const std::size_t align = static_cast<std::size_t>(alignment);
  const std::size_t rounded = ((size == 0 ? 1 : size) + align - 1) / align * align;
  void* pointer = std::aligned_alloc(align, rounded);
  if (pointer == nullptr) {
    throw std::bad_alloc();
  }
  return pointer;
}

#if defined(__GNUC__) && !defined(__clang__)
// GCC's -Wmismatched-new-delete cannot see that the operator new above is
// implemented on top of std::malloc, so it flags the matching std::free. The
// pairing here IS malloc/free by construction, which is exactly what the warning
// exists to catch elsewhere, so it is silenced for these definitions only rather
// than "fixed" with a mismatched allocator.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::align_val_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { std::free(pointer); }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// --- layers 2 and 3 ----------------------------------------------------------

namespace astra::guard {

namespace {

void report_and_abort(const char* condition, const char* file, int line) noexcept {
  // Reachable only when no guard is armed, i.e. a genuine Eigen assertion
  // outside a measurement. Allocation-free on purpose.
  std::fprintf(stderr, "astra guard: assertion failed (%s) at %s:%d\n", condition, file, line);
  std::abort();
}

}  // namespace

void on_eigen_assert_failure(const char* condition, const char* file, int line) {
  if (counters().armed.load(std::memory_order_relaxed)) {
    counters().eigen_violations.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  report_and_abort(condition, file, line);
}

const MallocLayer& malloc_layer() noexcept {
  static const MallocLayer layer = [] {
    MallocLayer resolved;
    resolved.reset =
        reinterpret_cast<MallocLayer::VoidFn>(dlsym(RTLD_DEFAULT, "astra_malloc_guard_reset"));
    resolved.arm =
        reinterpret_cast<MallocLayer::VoidFn>(dlsym(RTLD_DEFAULT, "astra_malloc_guard_arm"));
    resolved.disarm =
        reinterpret_cast<MallocLayer::VoidFn>(dlsym(RTLD_DEFAULT, "astra_malloc_guard_disarm"));
    resolved.count =
        reinterpret_cast<MallocLayer::CountFn>(dlsym(RTLD_DEFAULT, "astra_malloc_guard_count"));
    return resolved;
  }();
  return layer;
}

void arm() noexcept {
  // Resolve the layer FIRST: dlsym may allocate, and that allocation belongs to
  // the harness, not to the operation under measurement.
  const bool malloc_available = malloc_layer().available();
  if (malloc_available && malloc_layer().reset != nullptr) {
    malloc_layer().reset();
  }

  counters().cxx_allocations.store(0, std::memory_order_relaxed);
  counters().eigen_violations.store(0, std::memory_order_relaxed);
  counters().armed.store(true, std::memory_order_relaxed);

  if (malloc_available) {
    malloc_layer().arm();
  }
  // Eigen 3.4.0 checks this as the first statement of aligned_malloc(), but it
  // is an eigen_assert: it does nothing unless this target is compiled without
  // NDEBUG. See the CMake target and the header comment.
  Eigen::internal::set_is_malloc_allowed(false);
}

void disarm() noexcept {
  Eigen::internal::set_is_malloc_allowed(true);
  if (malloc_layer().available()) {
    malloc_layer().disarm();
  }
  counters().armed.store(false, std::memory_order_relaxed);
}

long cxx_allocations() noexcept {
  return counters().cxx_allocations.load(std::memory_order_relaxed);
}

long eigen_violations() noexcept {
  return counters().eigen_violations.load(std::memory_order_relaxed);
}

long malloc_allocations() noexcept {
  return malloc_layer().available() ? malloc_layer().count() : -1;
}

bool malloc_layer_available() noexcept { return malloc_layer().available(); }

const char* malloc_layer_platform() noexcept {
  using PlatformFn = const char* (*)();
  static const PlatformFn fn =
      reinterpret_cast<PlatformFn>(dlsym(RTLD_DEFAULT, "astra_malloc_guard_platform"));
  return fn != nullptr ? fn() : "unavailable (interposer not preloaded)";
}

}  // namespace astra::guard

// --- inputs used by the measurements ----------------------------------------

namespace {

using astra::Matrix3;
using astra::Matrix4;
using astra::Points;
using astra::Vector3;
using astra::Vector4;

const Vector3 kAxis(0.3, -0.5, 0.81);
const double kTheta = 0.7;
const Matrix4 kTransform = astra::se3::transformation_matrix(
    astra::so3::rotation_matrix3D(kAxis, kTheta), Vector3(0.1, -0.2, 0.3));
const Vector4 kUnitQuaternion(1.0, 0.0, 0.0, 0.0);
const Vector4 kQuaternionB(0.5, 0.5, 0.5, 0.5);

[[nodiscard]] Points make_points(Eigen::Index count) {
  Points points(count, 3);
  for (Eigen::Index i = 0; i < count; ++i) {
    const double x = static_cast<double>(i) * 0.01;
    points.row(i) << x, -x, 1.0 + x;
  }
  return points;
}

struct Measurement {
  long eigen{0};
  long malloc_layer{-1};  ///< -1 means NOT MEASURED, never "none"
  long cxx_new{0};
};

/// Run the callable with every available layer armed and collect the counts.
/// No Catch2 machinery is used inside the guarded region: the assertion
/// framework is part of the harness and allocates as such.
template <typename Callable>
[[nodiscard]] Measurement measure(Callable&& callable) {
  astra::guard::arm();
  callable();
  astra::guard::disarm();

  Measurement measurement;
  measurement.eigen = astra::guard::eigen_violations();
  measurement.malloc_layer = astra::guard::malloc_allocations();
  measurement.cxx_new = astra::guard::cxx_allocations();
  return measurement;
}

void report(const char* label, Eigen::Index n, const Measurement& measurement) {
  std::printf("MEMORY op=%-30s n=%-6ld eigen_runtime=%-3ld malloc_layer=%-3ld cxx_new=%-3ld\n",
              label, static_cast<long>(n), measurement.eigen, measurement.malloc_layer,
              measurement.cxx_new);
}

[[nodiscard]] bool no_allocation_anywhere(const Measurement& measurement) {
  return measurement.eigen == 0 && measurement.cxx_new == 0 && measurement.malloc_layer <= 0;
}

}  // namespace

// --- controls ----------------------------------------------------------------

TEST_CASE("positive control: the guards detect allocations they must detect", "[guard][control]") {
  // Without this, "no allocations detected" is unfalsifiable: a guard that never
  // fires proves nothing about the code it is supposed to watch.
  const bool malloc_available = astra::guard::malloc_layer_available();
  std::printf("GUARD malloc_layer_available=%s platform=%s\n", malloc_available ? "yes" : "no",
              astra::guard::malloc_layer_platform());

  SECTION("a dynamic Eigen matrix is seen by the Eigen layer and the malloc layer") {
    const Measurement measurement = measure([] {
      Eigen::MatrixXd dynamic = Eigen::MatrixXd::Zero(8, 8);
      dynamic(0, 0) = 1.0;
      g_result_sink = consume(dynamic);
    });
    report("CONTROL Eigen::MatrixXd(8,8)", 8, measurement);

    REQUIRE(measurement.eigen > 0);
    if (malloc_available) {
      REQUIRE(measurement.malloc_layer > 0);
    }
    // ...and layer 1 does NOT see it, because Eigen does not use operator new.
    // That is the evidence that the layers complement each other.
    REQUIRE(measurement.cxx_new == 0);
  }

  SECTION("a std::vector is seen by the operator new layer") {
    const Measurement measurement = measure([] {
      const std::size_t length = static_cast<std::size_t>(g_result_sink) + 100;
      std::vector<double> values(length, 1.0);
      g_result_sink = values[0] + values.back();
    });
    report("CONTROL std::vector<double>(100)", 100, measurement);
    REQUIRE(measurement.cxx_new > 0);
    if (malloc_available) {
      REQUIRE(measurement.malloc_layer > 0);
    }
  }

  SECTION("a std::string is seen by the operator new layer") {
    const Measurement measurement = measure([] {
      // The length comes from a volatile read so that the optimiser cannot fold
      // the whole construction away: with a literal size, the first version of
      // this control measured zero allocations because the string never existed.
      const std::size_t length = static_cast<std::size_t>(g_result_sink) + 100;
      std::string text(length, 'x');
      g_result_sink = static_cast<double>(text.size()) + static_cast<double>(text.front());
    });
    report("CONTROL std::string(100)", 100, measurement);
    REQUIRE(measurement.cxx_new > 0);
  }
}

TEST_CASE("the Eigen layer is armed rather than vacuous", "[guard][control]") {
  // The most important layer depends on eigen_assert being live, which depends
  // on this target NOT being compiled with NDEBUG: Eigen defines EIGEN_NO_DEBUG
  // from NDEBUG and turns eigen_assert into a no-op. If someone changed that,
  // this test fails instead of the report quietly showing zero.
#if defined(NDEBUG)
  FAIL("memory_guards was compiled with NDEBUG: the Eigen layer is vacuous");
#else
  const Measurement measurement = measure([] {
    Eigen::MatrixXd dynamic = Eigen::MatrixXd::Ones(4, 4);
    g_result_sink = consume(dynamic);
  });
  REQUIRE(measurement.eigen > 0);
#endif
}

// --- the operations ----------------------------------------------------------

TEST_CASE("the fixed-size operations make no dynamic allocation", "[guard][memory]") {
  // The eighteen operations that return fixed-size values. rotate_points and
  // transform_points are measured separately, because their contract is to
  // return a new (N, 3) array.
  struct Case {
    const char* label;
    Measurement measurement;
  };
  std::vector<Case> cases;

  cases.push_back({"is_rotation_matrix", measure([] {
                     const Matrix3 R = Matrix3::Identity();
                     g_result_sink = astra::so3::is_rotation_matrix(R) ? 1.0 : 0.0;
                   })});
  cases.push_back({"rotation_matrix3D", measure([] {
                     const Matrix3 R = astra::so3::rotation_matrix3D(kAxis, kTheta);
                     g_result_sink = consume(R);
                   })});
  cases.push_back({"rotation_matrix2D", measure([] {
                     const astra::Matrix2 R = astra::so3::rotation_matrix2D(kTheta);
                     g_result_sink = R.sum();
                   })});
  cases.push_back({"compose_rotations", measure([] {
                     const Matrix3 R = astra::so3::compose_rotations(Matrix3::Identity(),
                                                                    Matrix3::Identity());
                     g_result_sink = consume(R);
                   })});
  cases.push_back({"inverse_rotation", measure([] {
                     const Matrix3 R = astra::so3::inverse_rotation(Matrix3::Identity());
                     g_result_sink = consume(R);
                   })});
  cases.push_back({"is_unit_quaternion", measure([] {
                     g_result_sink = astra::quaternion::is_unit_quaternion(kUnitQuaternion) ? 1.0 : 0.0;
                   })});
  cases.push_back({"normalize_quaternion", measure([] {
                     const Vector4 q = astra::quaternion::normalize_quaternion(Vector4(2.0, 0.0, 0.0, 1.0));
                     g_result_sink = consume(q);
                   })});
  cases.push_back({"conjugate_quaternion", measure([] {
                     const Vector4 q = astra::quaternion::conjugate_quaternion(kUnitQuaternion);
                     g_result_sink = consume(q);
                   })});
  cases.push_back({"quaternion_multiply", measure([] {
                     const Vector4 q = astra::quaternion::quaternion_multiply(kUnitQuaternion, kQuaternionB);
                     g_result_sink = consume(q);
                   })});
  cases.push_back({"rotation_matrix_to_quaternion", measure([] {
                     const Vector4 q = astra::quaternion::rotation_matrix_to_quaternion(Matrix3::Identity());
                     g_result_sink = consume(q);
                   })});
  cases.push_back({"quaternion_to_rotation_matrix", measure([] {
                     const Matrix3 R = astra::quaternion::quaternion_to_rotation_matrix(kQuaternionB);
                     g_result_sink = consume(R);
                   })});
  cases.push_back({"is_transformation_matrix", measure([] {
                     g_result_sink = astra::se3::is_transformation_matrix(kTransform) ? 1.0 : 0.0;
                   })});
  cases.push_back({"transformation_matrix", measure([] {
                     const Matrix4 T = astra::se3::transformation_matrix(Matrix3::Identity(), Vector3::Zero());
                     g_result_sink = consume(T);
                   })});
  cases.push_back({"identity_transform", measure([] {
                     const Matrix4 T = astra::se3::identity_transform();
                     g_result_sink = consume(T);
                   })});
  cases.push_back({"rotation_part", measure([] {
                     const Matrix3 R = astra::se3::rotation_part(kTransform);
                     g_result_sink = consume(R);
                   })});
  cases.push_back({"translation_part", measure([] {
                     const Vector3 t = astra::se3::translation_part(kTransform);
                     g_result_sink = consume(t);
                   })});
  cases.push_back({"compose_transforms", measure([] {
                     const Matrix4 T = astra::se3::compose_transforms(kTransform, kTransform);
                     g_result_sink = consume(T);
                   })});
  cases.push_back({"inverse_transform", measure([] {
                     const Matrix4 T = astra::se3::inverse_transform(kTransform);
                     g_result_sink = consume(T);
                   })});

  int failures = 0;
  for (const auto& entry : cases) {
    report(entry.label, 1, entry.measurement);
    if (!no_allocation_anywhere(entry.measurement)) {
      ++failures;
    }
  }
  REQUIRE(failures == 0);
}

TEST_CASE("the point-batch operations allocate a constant number of times", "[guard][memory]") {
  // Their contract is to return a new (N, 3) array, so the useful claim is not
  // "zero allocations" but "the count does not grow with N": one result buffer
  // and no per-point scratch. The exact value is reported rather than assumed.
  constexpr Eigen::Index kSmall = 3;
  constexpr Eigen::Index kLarge = 2000;

  const Points small = make_points(kSmall);
  const Points large = make_points(kLarge);

  const Measurement small_rotate = measure([&] {
    const Points moved = astra::so3::rotate_points(small, kAxis, kTheta);
    g_result_sink = consume(moved);
  });
  const Measurement large_rotate = measure([&] {
    const Points moved = astra::so3::rotate_points(large, kAxis, kTheta);
    g_result_sink = consume(moved);
  });
  const Measurement small_transform = measure([&] {
    const Points moved = astra::se3::transform_points(small, kTransform);
    g_result_sink = consume(moved);
  });
  const Measurement large_transform = measure([&] {
    const Points moved = astra::se3::transform_points(large, kTransform);
    g_result_sink = consume(moved);
  });

  report("rotate_points", kSmall, small_rotate);
  report("rotate_points", kLarge, large_rotate);
  report("transform_points", kSmall, small_transform);
  report("transform_points", kLarge, large_transform);

  // The count is the claim: identical for N = 3 and N = 2000.
  REQUIRE(small_rotate.eigen == large_rotate.eigen);
  REQUIRE(small_transform.eigen == large_transform.eigen);
  REQUIRE(small_rotate.eigen <= 4);
  REQUIRE(small_transform.eigen <= 4);
  REQUIRE(small_rotate.cxx_new == 0);
  REQUIRE(small_transform.cxx_new == 0);
  if (astra::guard::malloc_layer_available()) {
    REQUIRE(small_rotate.malloc_layer == large_rotate.malloc_layer);
    REQUIRE(small_transform.malloc_layer == large_transform.malloc_layer);
  }
}
