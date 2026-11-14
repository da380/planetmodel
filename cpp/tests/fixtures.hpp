#ifndef PLANETMODEL_TESTS_FIXTURES_HPP
#define PLANETMODEL_TESTS_FIXTURES_HPP

// The fixtures make_fixtures.py writes.  A test whose fixture is missing
// is skipped, naming the script, unless PLANETMODEL_REQUIRE_FIXTURES is
// set in the environment (as CI sets it), when it fails.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/stat.h>
#endif

namespace planetmodel::testing {

inline std::filesystem::path FixturesDir() {
  return std::filesystem::path{PLANETMODEL_FIXTURES_DIR};
}

/// Where the writer's tests leave the files they write, made where it is
/// missing.
inline std::filesystem::path WrittenDir() {
  const auto out = std::filesystem::path{PLANETMODEL_WRITTEN_DIR};
  std::filesystem::create_directories(out);
  return out;
}

inline bool FixturesRequired() {
  return std::getenv("PLANETMODEL_REQUIRE_FIXTURES") != nullptr;
}

#if defined(__unix__) || defined(__APPLE__)
/// The permission bits of a file.
inline unsigned Permissions(const std::filesystem::path& path) {
  return static_cast<unsigned>(std::filesystem::status(path).permissions()) & 0777u;
}

/// A writer's files have the permissions a plain open for writing gives:
/// a new file the umask's (0644 under 022), an existing one its own
/// (0640 kept under 077); `write(path)` writes the file.
template <class Write>
void ExpectPlainPermissions(const std::filesystem::path& path, Write write) {
  std::filesystem::remove(path);
  const auto old = ::umask(022);
  write(path);
  EXPECT_EQ(Permissions(path), 0644u) << path;
  std::filesystem::permissions(path, std::filesystem::perms(0640),
                               std::filesystem::perm_options::replace);
  ::umask(077);
  write(path);
  ::umask(old);
  EXPECT_EQ(Permissions(path), 0640u) << path;
  auto left = 0;
  for (const auto& entry : std::filesystem::directory_iterator(path.parent_path())) {
    left += entry.path().filename().string().starts_with("." + path.filename().string());
  }
  EXPECT_EQ(left, 0) << "a partial file is left beside " << path;
}
#endif

}  // namespace planetmodel::testing

/// The path of fixture `name`, or the test skipped (failed, when fixtures
/// are required) where the script has not written it.
#define PLANETMODEL_FIXTURE(path_variable, name)                                    \
  const auto path_variable = ::planetmodel::testing::FixturesDir() / (name);        \
  if (!std::filesystem::exists(path_variable)) {                                   \
    if (::planetmodel::testing::FixturesRequired()) {                               \
      FAIL() << "fixture " << path_variable << " is missing";                       \
    }                                                                               \
    GTEST_SKIP() << "fixture " << path_variable                                     \
                 << " is missing: run cpp/tests/make_fixtures.py";                 \
  }

#endif  // PLANETMODEL_TESTS_FIXTURES_HPP
