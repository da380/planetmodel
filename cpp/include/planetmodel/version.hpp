#ifndef PLANETMODEL_VERSION_HPP
#define PLANETMODEL_VERSION_HPP

/**
 * @file
 * @brief The library's version, which is the Python package's: the CMake
 * target reads it from `pyproject.toml` and defines it for every code it
 * links.
 *
 * The target defines `PLANETMODEL_VERSION`, the version as the Python
 * package states it (`"1.3.0"`, say), and `PLANETMODEL_VERSION_MAJOR`,
 * `PLANETMODEL_VERSION_MINOR` and `PLANETMODEL_VERSION_PATCH`, the
 * integers of its first three numbers (a number it lacks is 0, and what
 * follows them, a pre-release's `rc1` say, is not counted), which the
 * preprocessor can test where it cannot test the string.
 * `PLANETMODEL_VERSION_AT_LEAST(major, minor, patch)` tests for a release
 * or a later one:
 *
 * @code
 * #include <planetmodel/version.hpp>
 * #if PLANETMODEL_VERSION_AT_LEAST(1, 6, 0)
 * // what 1.6.0 added
 * #endif
 * @endcode
 *
 * Between releases the version is the last release's, so a code that
 * follows the development of a feature tests the feature's own macro,
 * which the header providing it defines (`PLANETMODEL_HAS_...`, listed in
 * the README).  A code that does not link the target lacks the four
 * definitions, and is refused here; the three integers are held to the
 * string at compile time.
 */

#include <array>
#include <cstddef>
#include <string_view>

#if !defined(PLANETMODEL_VERSION) || !defined(PLANETMODEL_VERSION_MAJOR) || \
    !defined(PLANETMODEL_VERSION_MINOR) || !defined(PLANETMODEL_VERSION_PATCH)
#error "planetmodel's version is defined by its CMake target: link planetmodel::planetmodel"
#endif

/**
 * @brief Whether the library's version is major.minor.patch or later, by
 * its three integers; a preprocessor's test, for `#if`.
 * @param major The major version.
 * @param minor The minor version.
 * @param patch The patch version.
 */
#define PLANETMODEL_VERSION_AT_LEAST(major, minor, patch)                 \
  (PLANETMODEL_VERSION_MAJOR > (major) ||                                 \
   (PLANETMODEL_VERSION_MAJOR == (major) &&                               \
    (PLANETMODEL_VERSION_MINOR > (minor) ||                               \
     (PLANETMODEL_VERSION_MINOR == (minor) && PLANETMODEL_VERSION_PATCH >= (patch)))))

namespace planetmodel {

/**
 * @brief The version of the release the library was fetched at, as the
 * Python package states it (`1.3.0`, say).
 */
[[nodiscard]] constexpr std::string_view Version() noexcept {
  return PLANETMODEL_VERSION;
}

namespace detail {

/// The integers of a version's first three numbers, as the CMake target
/// reads them for `PLANETMODEL_VERSION_MAJOR`, `_MINOR` and `_PATCH`: a
/// number the version lacks is 0, and the numbers end at the first
/// character that is neither a digit nor a dot before one.
[[nodiscard]] constexpr std::array<int, 3> VersionNumbers(std::string_view version) noexcept {
  const auto digit = [&version](std::size_t k) {
    return k < version.size() && version[k] >= '0' && version[k] <= '9';
  };
  auto out = std::array<int, 3>{0, 0, 0};
  auto k = std::size_t{0};
  for (auto& number : out) {
    if (!digit(k)) break;
    while (digit(k)) number = 10 * number + (version[k++] - '0');
    if (!(k < version.size() && version[k] == '.' && digit(k + 1))) break;
    ++k;
  }
  return out;
}

}  // namespace detail

static_assert(detail::VersionNumbers(PLANETMODEL_VERSION) ==
                  std::array<int, 3>{PLANETMODEL_VERSION_MAJOR, PLANETMODEL_VERSION_MINOR,
                                     PLANETMODEL_VERSION_PATCH},
              "PLANETMODEL_VERSION_MAJOR, _MINOR and _PATCH are the numbers of "
              "PLANETMODEL_VERSION");

}  // namespace planetmodel

#endif  // PLANETMODEL_VERSION_HPP
