// Allocation guards for the spike.
//
// THIS HEADER MUST BE INCLUDED BEFORE ANY EIGEN HEADER IN A TRANSLATION UNIT.
// It overrides `eigen_assert`, and Eigen's Macros.h only installs its own
// version if the macro is not already defined (the header says so literally:
// "eigen_assert can be overridden").
//
// Three layers, because no single one is sufficient:
//
//   1. C++ `operator new` / `new[]`, replaced in the guard translation unit.
//      Portable, and it sees the standard library and our own code -- but not
//      Eigen, which does not use operator new.
//   2. Eigen's own runtime check: EIGEN_RUNTIME_NO_MALLOC plus
//      Eigen::internal::set_is_malloc_allowed(false). Verified in the source of
//      Eigen 3.4.0: check_that_malloc_is_allowed() is the first statement of
//      aligned_malloc(), so every Eigen heap allocation goes through it.
//      CRITICAL CAVEAT: that check is an eigen_assert, and eigen_assert becomes
//      a NO-OP when NDEBUG is defined (Macros.h defines EIGEN_NO_DEBUG from
//      NDEBUG). A plain Release build would therefore arm this layer and detect
//      nothing. The memory_guards target is consequently compiled WITHOUT
//      NDEBUG while keeping the Release optimisation level.
//   3. malloc-family interposition through LD_PRELOAD (tools/interpose_malloc.c).
//      PLATFORM-SPECIFIC: Linux/glibc only. It exists because layer 2 depends on
//      our macro override being in effect, and because it also catches
//      allocations made deep inside a dependency. When the library is not
//      preloaded this layer reports itself UNAVAILABLE; it is never reported as
//      "zero allocations".
//
// The scope of the claim: only the operations and the paths actually executed
// between arm() and disarm() are covered. Error paths are excluded on purpose
// (throwing allocates), and so is everything the harness does.

#pragma once

#include <atomic>
#include <cstddef>

namespace astra::guard {

/// Called by the overridden eigen_assert. Records the violation when a guard is
/// armed; aborts otherwise, which is what Eigen's own assertion would do.
void on_eigen_assert_failure(const char* condition, const char* file, int line);

struct Counters {
  std::atomic<bool> armed{false};
  std::atomic<long> cxx_allocations{0};
  std::atomic<long> eigen_violations{0};
};

[[nodiscard]] inline Counters& counters() noexcept {
  static Counters value;
  return value;
}

/// The malloc layer, resolved from the preloaded interposer at runtime.
struct MallocLayer {
  using VoidFn = void (*)();
  using CountFn = long (*)();
  VoidFn reset{nullptr};
  VoidFn arm{nullptr};
  VoidFn disarm{nullptr};
  CountFn count{nullptr};

  [[nodiscard]] bool available() const noexcept { return arm != nullptr && count != nullptr; }
};

[[nodiscard]] const MallocLayer& malloc_layer() noexcept;

/// Arm every available layer and reset the counters.
void arm() noexcept;

/// Disarm. Counters keep their values until the next arm().
void disarm() noexcept;

[[nodiscard]] long cxx_allocations() noexcept;
[[nodiscard]] long eigen_violations() noexcept;
/// -1 when the malloc layer is not available: the caller must report "not
/// measured" rather than "none".
[[nodiscard]] long malloc_allocations() noexcept;
[[nodiscard]] bool malloc_layer_available() noexcept;
/// Human-readable identification of the platform the malloc layer depends on.
[[nodiscard]] const char* malloc_layer_platform() noexcept;

}  // namespace astra::guard

/// Eigen's assertion hook, installed before Eigen is included.
#define eigen_assert(condition)                                                          \
  do {                                                                                   \
    if (!(condition)) {                                                                  \
      ::astra::guard::on_eigen_assert_failure(#condition, __FILE__, __LINE__);            \
    }                                                                                    \
  } while (false)

#include <Eigen/Core>

#if !defined(EIGEN_RUNTIME_NO_MALLOC)
#error \
    "memory_guard.hpp needs EIGEN_RUNTIME_NO_MALLOC: without it Eigen has no runtime allocation check to arm"
#endif
