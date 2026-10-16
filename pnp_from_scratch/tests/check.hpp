#pragma once

// Minimal self-checking test helpers for the scratch pipeline tests.

#include <cstdio>
#include <filesystem>
#include <string>

#include "common.hpp"

namespace scratch_test {

inline int &Failures() {
  static int n = 0;
  return n;
}

#define CHECK(cond, ...)                           \
  do {                                             \
    const bool ok_ = (cond);                       \
    std::printf("  [%s] ", ok_ ? "PASS" : "FAIL"); \
    std::printf(__VA_ARGS__);                      \
    std::printf("\n");                             \
    if (!ok_) ++scratch_test::Failures();          \
  } while (0)

inline int Finish(const char *name) {
  const int f = Failures();
  std::printf("\n%s: %s (%d failure%s)\n", name, f ? "FAIL" : "PASS", f, f == 1 ? "" : "s");
  return f ? 1 : 0;
}

// Repository root, found upward from the current directory.
inline std::filesystem::path Repo() {
  const auto r = scratch::FindRepo(std::filesystem::current_path());
  if (r.empty()) {
    std::fprintf(stderr, "repository root not found (run from inside the repository)\n");
    std::exit(2);
  }
  return r;
}

}  // namespace scratch_test
