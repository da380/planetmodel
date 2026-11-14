// The interpolation rules against the Python's: every lateral case on every
// kind of grid, every radial case, values and derivatives; the frames and
// the reduction of longitudes; and the refusals.
#include "planetmodel/rules.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/frames.hpp"
#include "planetmodel/netcdf.hpp"

#ifndef PLANETMODEL_HAS_LOG_RULES
#error "rules.hpp defines the macro of its rules in the logarithm"
#endif

namespace {

namespace nc = planetmodel::netcdf;
using planetmodel::Error;

std::vector<std::filesystem::path> Cases(std::string_view prefix) {
  auto out = std::vector<std::filesystem::path>{};
  for (const auto& entry : std::filesystem::directory_iterator(planetmodel::testing::FixturesDir())) {
    const auto name = entry.path().filename().string();
    if (name.starts_with(prefix) && entry.path().extension() == ".nc") out.push_back(entry.path());
  }
  std::ranges::sort(out);
  return out;
}

double Largest(const std::vector<double>& v) {
  auto out = 0.0;
  for (const auto x : v) out = std::max(out, std::abs(x));
  return out;
}

double ToDouble(const std::string& text) {
  auto value = 0.0;
  std::from_chars(text.data(), text.data() + text.size(), value);
  return value;
}

}  // namespace

TEST(Rules, ReadEveryLateralCaseAsThePythonDoes) {
  PLANETMODEL_FIXTURE(probe, "case_lateral_gauss_legendre_cubic.nc");
  const auto cases = Cases("case_lateral_");
  ASSERT_EQ(cases.size(), 10u);
  for (const auto& path : cases) {
    SCOPED_TRACE(path.filename().string());
    const auto file = nc::File(path);
    const auto source = path.string();
    auto read = [&](std::string_view name) { return nc::ReadDoubles(file.Id(), name, source).values; };
    const auto kind = *nc::TextAttribute(file.Id(), nc::GLOBAL, "kind", source);
    const auto method = *nc::TextAttribute(file.Id(), nc::GLOBAL, "method", source);
    const auto grid = planetmodel::AngularGrid(read("colatitude"), read("longitude"), kind);
    const auto values = read("values");
    const auto poles = read("poles");
    const auto f = planetmodel::LateralInterpolant(grid, values, poles,
                                                   planetmodel::LateralRuleNamed(method));
    const auto theta = read("query_colatitude");
    const auto phi = read("query_longitude");
    const auto expected = read("expected");
    const auto dtheta = read("expected_dtheta");
    const auto dphi = read("expected_dphi");
    const auto scale = Largest(values);
    const auto dscale = std::max(Largest(dtheta), Largest(dphi));
    for (auto i = std::size_t{0}; i < theta.size(); ++i) {
      SCOPED_TRACE(std::format("({}, {})", theta[i], phi[i]));
      EXPECT_NEAR(f.Value(theta[i], phi[i]), expected[i], 1e-12 * scale);
      const auto [gt, gp] = f.Gradient(theta[i], phi[i]);
      EXPECT_NEAR(gt, dtheta[i], 1e-10 * dscale);
      EXPECT_NEAR(gp, dphi[i], 1e-10 * dscale);
    }
  }
}

TEST(Rules, ReadEveryRadialCaseAsThePythonDoes) {
  for (const auto* method : {"polynomial", "linear", "log-polynomial", "log-linear"}) {
    SCOPED_TRACE(method);
    PLANETMODEL_FIXTURE(path, std::format("case_radial_{}.nc", method));
    const auto file = nc::File(path);
    const auto source = path.string();
    auto read = [&](std::string_view name) { return nc::ReadDoubles(file.Id(), name, source).values; };
    const auto ngll = static_cast<std::size_t>(
        nc::NumberAttribute(file.Id(), nc::GLOBAL, "ngll", source)->at(0));
    const auto radius = read("radius");
    const auto span = radius.back() - radius.front();
    const auto f = planetmodel::RadialInterpolant(
        planetmodel::RadialLayout(radius, ngll, 1e-9 * span), read("values"),
        planetmodel::RadialRuleNamed(*nc::TextAttribute(file.Id(), nc::GLOBAL, "method",
                                                        source)));
    const auto r = read("query_radius");
    const auto expected = read("expected");
    const auto dr = read("expected_dr");
    for (auto i = std::size_t{0}; i < r.size(); ++i) {
      SCOPED_TRACE(std::format("r = {}", r[i]));
      EXPECT_NEAR(f.Value(r[i]), expected[i], 1e-13 * (Largest(expected) + 1.0));
      EXPECT_NEAR(f.Derivative(r[i]), dr[i], 1e-11 * (Largest(dr) + 1.0));
    }
  }
}

TEST(Rules, TheFrameAndTheReductionAreThePythons) {
  PLANETMODEL_FIXTURE(path, "frames.txt");
  auto in = std::ifstream(path);
  auto frames = 0;
  for (auto line = std::string{}; std::getline(in, line);) {
    auto words = std::istringstream{line};
    auto key = std::string{};
    words >> key;
    auto numbers = std::vector<double>{};
    for (auto w = std::string{}; words >> w;) numbers.push_back(ToDouble(w));
    if (key == "frame") {
      const auto R = planetmodel::SphericalFrame(numbers[0], numbers[1]);
      for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
          EXPECT_NEAR(R[i][j], numbers[2 + 3 * i + j], 1e-15) << line;
        }
      }
      const auto v = planetmodel::Vector3{0.3, -1.2, 2.5};
      const auto back = planetmodel::ToSpherical(
          planetmodel::ToCartesian(v, numbers[0], numbers[1]), numbers[0], numbers[1]);
      for (auto i = 0; i < 3; ++i) EXPECT_NEAR(back[i], v[i], 1e-15) << line;
      ++frames;
    } else if (key == "wrapped") {
      EXPECT_EQ(planetmodel::Wrapped(numbers[0], numbers[1]), numbers[2]) << line;
    }
  }
  EXPECT_GE(frames, 9);
}

TEST(Rules, RefuseTheHarmonicRuleNamingTheWayOut) {
  try {
    static_cast<void>(planetmodel::LateralRuleNamed("harmonic", "m.nc", "part_0/rho"));
    FAIL();
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "part_0/rho");
    EXPECT_NE(std::string{e.what()}.find("interpolation={name:"), std::string::npos);
  }
  EXPECT_THROW(static_cast<void>(planetmodel::LateralRuleNamed("nearest")), Error);
  EXPECT_THROW(static_cast<void>(planetmodel::RadialRuleNamed("cubic")), Error);
}

TEST(Rules, HoldThePolesAsThePythonDoes) {
  constexpr auto pi = std::numbers::pi;
  // a grid with both poles as nodes: the rows are the poles' values
  const auto both = planetmodel::AngularGrid({0.0, 1.0, 2.0, pi}, {0.0, 2.0, 4.0});
  auto values = std::vector<double>{5, 5, 5, 1, 2, 3, 4, 5, 6, 7, 7, 7};
  EXPECT_EQ(planetmodel::PolarValues(both, values, {}), (std::array<double, 2>{5.0, 7.0}));
  const auto f = planetmodel::LateralInterpolant(both, values, {},
                                                 planetmodel::LateralRule::Cubic);
  EXPECT_NEAR(f.Value(0.0, 1.3), 5.0, 1e-14);
  EXPECT_NEAR(f.Value(pi, -2.0), 7.0, 1e-14);
  EXPECT_NEAR(f.Value(1.0, 2.0), 2.0, 1e-14);
  values[1] = 5.1;
  EXPECT_THROW(static_cast<void>(planetmodel::PolarValues(both, values, {})), Error);
  // a grid lacking both: their values are given, and must be
  const auto none = planetmodel::AngularGrid({0.5, 1.5, 2.5}, {0.0, 2.0, 4.0});
  const auto inner = std::vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9};
  EXPECT_THROW(static_cast<void>(planetmodel::PolarValues(none, inner, {})), Error);
  const auto nan = std::vector<double>{std::nan(""), 1.0};
  EXPECT_THROW(static_cast<void>(planetmodel::PolarValues(none, inner, nan)), Error);
  const auto given = std::vector<double>{0.0, 10.0};
  const auto g = planetmodel::LateralInterpolant(none, inner, given,
                                                 planetmodel::LateralRule::Linear);
  EXPECT_NEAR(g.Value(0.0, 3.0), 0.0, 1e-15);
  EXPECT_NEAR(g.Value(pi, 3.0), 10.0, 1e-15);
  // a longitude and its turns read alike
  EXPECT_NEAR(g.Value(1.0, 1.0), g.Value(1.0, 1.0 + 6.0 * pi), 1e-13);
  EXPECT_NEAR(g.Value(1.0, 1.0), g.Value(1.0, 1.0 - 4.0 * pi), 1e-13);
  // a point off the sphere is refused
  EXPECT_THROW(static_cast<void>(g.Value(-0.1, 0.0)), Error);
  EXPECT_THROW(static_cast<void>(g.Value(1.0, std::nan(""))), Error);
  // the cubic rule needs rows and columns enough
  const auto thin = planetmodel::AngularGrid({0.0, pi}, {0.0, 3.0});
  EXPECT_THROW(planetmodel::LateralInterpolant(thin, std::vector<double>{1, 1, 2, 2}, {},
                                               planetmodel::LateralRule::Cubic),
               Error);
}

TEST(Rules, ARuleInTheLogarithmIsExponentialBetweenNodesAndKeepsAConstant) {
  using planetmodel::RadialRule;
  EXPECT_TRUE(planetmodel::IsLogRule(RadialRule::LogLinear));
  EXPECT_FALSE(planetmodel::IsLogRule(RadialRule::Linear));
  EXPECT_EQ(planetmodel::BaseRule(RadialRule::LogPolynomial), RadialRule::Polynomial);
  for (const auto* name : {"polynomial", "linear", "log-polynomial", "log-linear"}) {
    EXPECT_EQ(planetmodel::RuleName(planetmodel::RadialRuleNamed(name)), name);
  }
  // geometric between the two nodes of an element, the slope y log(b / a)
  const auto layout = planetmodel::RadialLayout({0.0, 1.0, 1.0, 2.0}, 2, 1e-9);
  const auto f = planetmodel::RadialInterpolant(layout, {1e19, 1e22, 3e21, 3e21},
                                                RadialRule::LogLinear);
  EXPECT_NEAR(f.Value(0.5), std::sqrt(1e19 * 1e22), 1e-15 * 1e21);
  EXPECT_NEAR(f.Value(1.0 / 3.0), 1e20, 1e-15 * 1e20);
  EXPECT_NEAR(f.Derivative(0.25), f.Value(0.25) * std::log(1e3), 1e-14 * f.Value(0.25) * 7.0);
  // an element of one value reads it to the bit, with no slope
  for (const auto r : {1.0, 1.1, 1.5, 1.9, 2.0}) {
    EXPECT_EQ(f.Value(r), 3e21) << r;
    EXPECT_EQ(f.Derivative(r), 0.0) << r;
  }
  const auto g = planetmodel::RadialInterpolant(
      planetmodel::RadialLayout({0.0, 0.5, 1.0}, 3, 1e-9), {7e20, 7e20, 7e20},
      RadialRule::LogPolynomial);
  EXPECT_EQ(g.Value(0.3), 7e20);
  // a value that has no logarithm is refused
  EXPECT_THROW(planetmodel::RadialInterpolant(layout, {1.0, 0.0, 1.0, 1.0}, RadialRule::LogLinear),
               Error);
  EXPECT_THROW(planetmodel::RadialInterpolant(layout, {1.0, -2.0, 1.0, 1.0},
                                              RadialRule::LogPolynomial),
               Error);
  EXPECT_NO_THROW(planetmodel::RadialInterpolant(layout, {1.0, -2.0, 1.0, 1.0},
                                                 RadialRule::Linear));
}

TEST(Rules, ARadialLayoutIsElementsThatTile) {
  EXPECT_THROW(planetmodel::RadialLayout({0.0, 0.5, 1.0}, 2, 1e-9), Error);
  EXPECT_THROW(planetmodel::RadialLayout({0.0, 0.5, 0.6, 1.0}, 2, 1e-9), Error);
  EXPECT_THROW(planetmodel::RadialLayout({0.0, 0.0, 0.5, 1.0}, 2, 1e-9), Error);
  const auto layout = planetmodel::RadialLayout({0.0, 0.5, 0.5, 1.0}, 2, 1e-9);
  EXPECT_EQ(layout.ElementOf(0.5), 1u);  // the element above at a shared radius
  EXPECT_EQ(layout.ElementOf(1.0), 1u);
  EXPECT_EQ(layout.ElementOf(1.0 + 1e-12), 1u);
  EXPECT_THROW(static_cast<void>(layout.ElementOf(1.1)), Error);
  // a value discontinuous at the shared radius reads from its own side
  const auto f = planetmodel::RadialInterpolant(layout, {1.0, 2.0, 5.0, 6.0},
                                                planetmodel::RadialRule::Linear);
  EXPECT_DOUBLE_EQ(f.Value(0.5), 5.0);
  EXPECT_DOUBLE_EQ(f.Value(0.5 - 1e-12), 2.0 - 2e-12);
  EXPECT_DOUBLE_EQ(f.Derivative(0.25), 2.0);
}
