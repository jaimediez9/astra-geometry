# Dependencies for the C++20 spike.
#
# ---------------------------------------------------------------------------
# Pin precedence: this is the contract, and the mechanism below implements it.
#
#   1. An explicit `-D<VAR>=<value>` on the configure line wins.
#   2. Otherwise the default declared in THIS FILE is used.
#   3. A value left in a build directory by an EARLIER REVISION of this file
#      never wins: it is purged, loudly.
#
# Why (3) exists. The pins used to be declared with `set(... CACHE STRING ...)`,
# and an existing cache entry takes precedence over such a declaration. Measured
# consequence: replacing the nlohmann/json pin in this file had NO effect on the
# four build directories that already existed -- they kept the old value, and the
# generated build header reported that old value while the source declared the
# new one. A pin that cannot be changed is not a pin.
#
# The mechanism tells the two kinds of cache entry apart by TYPE, which is what
# CMake itself uses: `-DVAR=value` produces an entry of type `UNINITIALIZED`,
# while one written by an earlier revision of this file has type `STRING`. A
# `STRING` entry is stale by construction, because this file no longer writes any
# cache entry for the pins at all. Verified with a minimal CMake project before
# being adopted here: clean -> file default; `-D` -> UNINITIALIZED, honoured;
# injected `STRING` -> purged, file default applied.
#
# Worth knowing: a `-D` override is sticky, as cache entries are in CMake. It
# applies to every later configure of that build directory until the entry is
# removed (delete the line from CMakeCache.txt, or configure a fresh directory).
# ---------------------------------------------------------------------------

# Declare a dependency pin; see the precedence contract above.
function(astra_pin VARIABLE DEFAULT)
  if(DEFINED "${VARIABLE}")
    get_property(_astra_cached_type CACHE "${VARIABLE}" PROPERTY TYPE)
    if(NOT _astra_cached_type STREQUAL "UNINITIALIZED")
      message(WARNING
        "astra: purging stale cache entry ${VARIABLE} (type ${_astra_cached_type}, "
        "value '${${VARIABLE}}'): it was written by an earlier revision of "
        "cmake/Dependencies.cmake and must not override the current default.")
      unset("${VARIABLE}" CACHE)
    endif()
  endif()
  if(NOT DEFINED "${VARIABLE}")
    set("${VARIABLE}" "${DEFAULT}" PARENT_SCOPE)
  endif()
endfunction()

# Every dependency is pinned by the COMMIT its tag resolves to. Tags are mutable;
# a commit SHA is content-addressed, so it identifies the same tree no matter
# which host serves it, which is what makes the pin survive a flaky mirror.
#
# Host note (measured 2026-11-01): the official Eigen host, gitlab.com, failed
# with a TLS error while github.com worked. This file probes the official URL at
# configure time and falls back to the GitHub mirror automatically, recording
# which one was used in the generated build info. The SHA is what pins the
# content either way; see cpp/README-spike.md.

astra_pin(ASTRA_EIGEN_VERSION "3.4.0")
astra_pin(ASTRA_EIGEN_GIT_TAG "3147391d946bb4b6c68edd901f2add6ac1f31f8c")
astra_pin(ASTRA_EIGEN_OFFICIAL_URL "https://gitlab.com/libeigen/eigen.git")
astra_pin(ASTRA_EIGEN_MIRROR_URL "https://github.com/eigen-mirror/eigen.git")

astra_pin(ASTRA_CATCH2_VERSION "3.16.1")
astra_pin(ASTRA_CATCH2_GIT_TAG "08092139210881f01a34e60c5f2fd6afa0ea4024")
astra_pin(ASTRA_CATCH2_URL "https://github.com/catchorg/Catch2.git")

astra_pin(ASTRA_JSON_VERSION "3.12.0")
# The COMMIT, not the tag object. `refs/tags/v3.12.0` is an annotated tag whose
# object is 65ee68451d8eb2b5f3a30b410476ab83deb3289b; the commit it points to is
# the one below, and that commit is what git checks out. Pinning the tag object
# would record an identifier that is not the installed tree, which is exactly the
# traceability defect an audit should catch. `git ls-remote --tags` prints the
# commit on the `^{}` line.
astra_pin(ASTRA_JSON_GIT_TAG "55f93686c01528224f448c19128836e7df245f72")
astra_pin(ASTRA_JSON_URL "https://github.com/nlohmann/json.git")

# --- Eigen -------------------------------------------------------------------
# Eigen is consumed as HEADERS ONLY, and deliberately without running its
# CMakeLists.
#
# Measured while building this spike, in this order:
#   1. FetchContent_MakeAvailable(eigen) runs Eigen's own CMakeLists, which calls
#      export(PACKAGE Eigen3). That writes ~/.cmake/packages/Eigen3 pointing at
#      the build tree, and the next configure then reports "using system Eigen3
#      3.4.0" while no Eigen exists on the machine: find_package had found the
#      build tree through the registry, silently replacing the pinned commit.
#   2. It also registers Eigen's ~950 test cases with CTest, which buries the
#      spike's own tests.
#
# Both are side effects of a subproject we never wanted: only the headers are
# needed. SOURCE_SUBDIR set to a directory with no CMakeLists.txt tells
# FetchContent to populate the sources and NOT add the subdirectory, which
# removes the registry write and the test registration at the source rather than
# patching their symptoms. The include directory is added explicitly below.
#
# The unset and the registry flags are kept as a second line of defence, for a
# machine that already carries a stale entry from an earlier build.
unset(Eigen3_DIR CACHE)
find_package(Eigen3 3.4 QUIET NO_MODULE)

if(Eigen3_FOUND AND NOT ASTRA_FORCE_FETCH_EIGEN)
  message(STATUS "astra: using system Eigen3 ${Eigen3_VERSION}")
  set(ASTRA_EIGEN_SOURCE "system (find_package): ${Eigen3_DIR}")
  add_library(astra_eigen INTERFACE)
  target_link_libraries(astra_eigen INTERFACE Eigen3::Eigen)
else()
  # Probe the official host. If it does not answer, use the mirror and say so.
  execute_process(
    COMMAND git ls-remote --exit-code "${ASTRA_EIGEN_OFFICIAL_URL}" HEAD
    RESULT_VARIABLE _astra_gitlab_probe
    OUTPUT_QUIET ERROR_QUIET TIMEOUT 25)

  if(_astra_gitlab_probe EQUAL 0)
    set(_astra_eigen_url "${ASTRA_EIGEN_OFFICIAL_URL}")
    set(ASTRA_EIGEN_SOURCE "FetchContent: official host")
  else()
    set(_astra_eigen_url "${ASTRA_EIGEN_MIRROR_URL}")
    set(ASTRA_EIGEN_SOURCE "FetchContent: mirror (official host unreachable)")
    message(WARNING
      "astra: ${ASTRA_EIGEN_OFFICIAL_URL} is unreachable; falling back to "
      "${ASTRA_EIGEN_MIRROR_URL}. The pin is the commit SHA, so the content is "
      "identical either way.")
  endif()

  include(FetchContent)
  FetchContent_Declare(eigen
    GIT_REPOSITORY "${_astra_eigen_url}"
    GIT_TAG "${ASTRA_EIGEN_GIT_TAG}"
    # Deliberately a directory that holds no CMakeLists.txt: populate, do not
    # add_subdirectory. See the block comment above.
    SOURCE_SUBDIR "astra-header-only-no-subproject"
    GIT_PROGRESS TRUE)
  FetchContent_MakeAvailable(eigen)

  add_library(astra_eigen INTERFACE)
  target_include_directories(astra_eigen INTERFACE "${eigen_SOURCE_DIR}")
endif()

# --- Catch2 (native tests only) ----------------------------------------------
include(FetchContent)

FetchContent_Declare(catch2
  GIT_REPOSITORY "${ASTRA_CATCH2_URL}"
  GIT_TAG "${ASTRA_CATCH2_GIT_TAG}"
  GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(catch2)

# --- nlohmann/json (conformance harness only; never the core) ----------------
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY "${ASTRA_JSON_URL}"
  GIT_TAG "${ASTRA_JSON_GIT_TAG}"
  GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(nlohmann_json)
