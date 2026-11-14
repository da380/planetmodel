// Dimensions and scales, against the Python's numbers in constants.txt;
// and the version, its three integers the numbers of its string.
#include "planetmodel/units.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/version.hpp"

namespace {

using planetmodel::Scales;
using planetmodel::units::Dimensions;

double ToDouble(const std::string& text) {
  auto value = 0.0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  EXPECT_EQ(error, std::errc{}) << text;
  EXPECT_EQ(end, text.data() + text.size()) << text;
  return value;
}

/// One line of constants.txt read as its words.
std::vector<std::string> Words(const std::string& line) {
  auto in = std::istringstream{line};
  auto out = std::vector<std::string>{};
  for (auto word = std::string{}; in >> word;) out.push_back(word);
  return out;
}

Dimensions Exponents(const std::vector<std::string>& w, std::size_t at) {
  return {.mass = std::stoi(w[at]),
          .length = std::stoi(w[at + 1]),
          .time = std::stoi(w[at + 2]),
          .temperature = std::stoi(w[at + 3])};
}

}  // namespace

TEST(Dimensions, MultiplyByAddingExponents) {
  using namespace planetmodel::units;
  EXPECT_EQ(MODULUS, DENSITY * VELOCITY.Pow(2));
  EXPECT_EQ(DENSITY / DENSITY, DIMENSIONLESS);
  EXPECT_EQ(GRAVITY, LENGTH / TIME.Pow(2));
  EXPECT_TRUE(DIMENSIONLESS.IsDimensionless());
  EXPECT_FALSE(TEMPERATURE.IsDimensionless());
  static_assert(VISCOSITY == MODULUS * TIME);
}

TEST(Dimensions, UnitStrings) {
  using namespace planetmodel::units;
  EXPECT_EQ(DENSITY.UnitString(), "kg m-3");
  EXPECT_EQ(DIMENSIONLESS.UnitString(), "1");
  EXPECT_EQ(DENSITY.UnitString(false), "1");
  EXPECT_EQ(UnitString(std::nullopt), "unknown");
  EXPECT_EQ(ScaledUnit(std::nullopt, Scales::SI()), "unknown");
  EXPECT_EQ(ScaledUnit(VELOCITY, Scales::SI()), "m s-1");
}

TEST(Scales, AreSIByDefault) {
  EXPECT_TRUE(Scales{}.IsSI());
  EXPECT_EQ(Scales{}, Scales::SI());
  EXPECT_FALSE(Scales({.length = 2.0}).IsSI());
  EXPECT_EQ(Scales({.length = 2.0}).Mass(), 1.0);
}

TEST(Scales, RefuseWhatIsNotAPositiveFiniteNumber) {
  for (const auto bad : {0.0, -1.0, std::nan(""), HUGE_VAL}) {
    EXPECT_THROW(Scales({.time = bad}), planetmodel::Error) << bad;
  }
  try {
    Scales({.temperature = -2.0});
    FAIL();
  } catch (const planetmodel::Error& e) {
    EXPECT_EQ(e.Where(), "temperature");
  }
  EXPECT_THROW(static_cast<void>(Scales::Geophysical(0.0)), planetmodel::Error);
  EXPECT_THROW(static_cast<void>(Scales::Geophysical(1.0, -1.0)), planetmodel::Error);
}

TEST(Scales, GeophysicalMakeGOne) {
  const auto s = Scales::Geophysical(6371e3);
  const auto g = planetmodel::G_SI / s.Factor(planetmodel::units::GRAVITATIONAL_CONSTANT);
  EXPECT_NEAR(g, 1.0, 1e-15);
  EXPECT_EQ(s.Temperature(), 1.0);
}

TEST(Scales, AgreeWithThePython) {
  PLANETMODEL_FIXTURE(path, "constants.txt");
  auto in = std::ifstream{path};
  auto scales = std::map<std::string, Scales>{};
  auto checked = 0;
  for (auto line = std::string{}; std::getline(in, line);) {
    const auto w = Words(line);
    if (w.empty()) continue;
    const auto& key = w[0];
    if (key == "version") {
      EXPECT_EQ(w[1], planetmodel::Version());
    } else if (key == "G_SI") {
      EXPECT_EQ(ToDouble(w[1]), planetmodel::G_SI);
    } else if (key == "EARTH_MEAN_DENSITY") {
      EXPECT_EQ(ToDouble(w[1]), planetmodel::EARTH_MEAN_DENSITY);
    } else if (key == "scales") {
      const auto s = Scales({.length = ToDouble(w[2]),
                             .mass = ToDouble(w[3]),
                             .time = ToDouble(w[4]),
                             .temperature = ToDouble(w[5])});
      scales.insert_or_assign(w[1], s);
      if (w[1] == "earth") {
        EXPECT_EQ(s, Scales::Geophysical(6371e3));
      }
    } else if (key == "factor") {
      const auto& s = scales.at(w[1]);
      const auto want = ToDouble(w[6]);
      EXPECT_DOUBLE_EQ(s.Factor(Exponents(w, 2)), want) << line;
      ++checked;
    } else if (key == "unit") {
      auto want = std::string{};
      for (auto i = std::size_t{6}; i < w.size(); ++i) {
        if (!want.empty()) want += ' ';
        want += w[i];
      }
      EXPECT_EQ(planetmodel::units::ScaledUnit(Exponents(w, 2), scales.at(w[1])), want)
          << line;
      ++checked;
    }
  }
  EXPECT_GT(checked, 60);
}

TEST(Version, TheThreeIntegersAreTheNumbersOfTheString) {
  constexpr auto first = PLANETMODEL_VERSION_MAJOR;
  constexpr auto second = PLANETMODEL_VERSION_MINOR;
  constexpr auto third = PLANETMODEL_VERSION_PATCH;
  const auto version = planetmodel::Version();
  // the integers CMake defines are the string's numbers, a number it lacks
  // 0; a version of three numbers begins with them
  EXPECT_EQ(planetmodel::detail::VersionNumbers(version), (std::array{first, second, third}))
      << version;
  if (std::ranges::count(version, '.') >= 2) {
    EXPECT_TRUE(version.starts_with(std::format("{}.{}.{}", first, second, third))) << version;
  }
  // the reading of a version, as CMake reads it
  using planetmodel::detail::VersionNumbers;
  static_assert(VersionNumbers("1.6.0") == std::array{1, 6, 0});
  static_assert(VersionNumbers("1.6.0rc1") == std::array{1, 6, 0});
  static_assert(VersionNumbers("1.6.0.dev2") == std::array{1, 6, 0});
  static_assert(VersionNumbers("1.6") == std::array{1, 6, 0});
  static_assert(VersionNumbers("2") == std::array{2, 0, 0});
  static_assert(VersionNumbers("10.20.30.40") == std::array{10, 20, 30});
  static_assert(VersionNumbers("1.x") == std::array{1, 0, 0});
  // at least the version itself, not anything after it
  static_assert(PLANETMODEL_VERSION_AT_LEAST(first, second, third));
  static_assert(PLANETMODEL_VERSION_AT_LEAST(first, second, 0));
  static_assert(PLANETMODEL_VERSION_AT_LEAST(first - 1, 99, 99));
  static_assert(!PLANETMODEL_VERSION_AT_LEAST(first, second, third + 1));
  static_assert(!PLANETMODEL_VERSION_AT_LEAST(first, second + 1, 0));
  static_assert(!PLANETMODEL_VERSION_AT_LEAST(first + 1, 0, 0));
#if !PLANETMODEL_VERSION_AT_LEAST(1, 0, 0) || PLANETMODEL_VERSION_AT_LEAST(1000, 0, 0)
  ADD_FAILURE() << "the preprocessor does not read the version";
#endif
}
