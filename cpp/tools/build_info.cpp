// Trivial entry point for step 1 of the spike: prove the toolchain and the
// dependency pins resolve before any mathematics is written.
//
// It also prints the compile-time facts the conformance report records, so a
// build can be fingerprinted from the binary alone.

#include <Eigen/Core>

#include <iostream>

int main() {
  const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
  std::cout << "astra spike toolchain probe\n";
  std::cout << "__cplusplus=" << __cplusplus << "\n";
  std::cout << "eigen_world=" << EIGEN_WORLD_VERSION << "."
            << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << "\n";
  std::cout << "compiler=" <<
#if defined(__clang__)
      "clang"
#elif defined(__GNUC__)
      "gcc"
#else
      "unknown"
#endif
            << "\n";
#if defined(NDEBUG)
  std::cout << "ndebug=defined\n";
#else
  std::cout << "ndebug=undefined\n";
#endif
  std::cout << "identity_trace=" << identity.trace() << "\n";
  return 0;
}
