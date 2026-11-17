// The flat lateral rules against the Python's: every case on a box's grid
// and a section's, values and derivatives, over several periods and at the
// seam; the domain and the grid as the Python lays them; the local frame;
// and the refusals.
#include "planetmodel/cartesian/rules.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <string>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/cartesian/coordinates.hpp"
#include "planetmodel/cartesian/domain.hpp"
#include "planetmodel/cartesian/grid.hpp"
#include "planetmodel/netcdf.hpp"

namespace {

namespace nc = planetmodel::netcdf;
namespace flat = planetmodel::cartesian;
using planetmodel::Error;
using planetmodel::LateralRule;

std::vector<std::filesystem::path> Cases() {
  auto out = std::vector<std::filesystem::path>{};
  for (const auto& entry : std::filesystem::directory_iterator(planetmodel::testing::FixturesDir())) {
    const auto name = entry.path().filename().string();
    if (name.starts_with("flat_case_lateral_") && entry.path().extension() == ".nc") {
      out.push_back(entry.path());
    }
  }
  std::ranges::sort(out);
  return out;
}

double Largest(const std::vector<double>& v) {
  auto out = 0.0;
  for (const auto x : v) out = std::max(out, std::abs(x));
  return out;
}

}  // namespace

TEST(CartesianRules, ReadEveryLateralCaseAsThePythonDoes) {
  PLANETMODEL_FIXTURE(probe, "flat_case_lateral_box_cubic.nc");
  const auto cases = Cases();
  ASSERT_EQ(cases.size(), 8u);
  auto worst_value = 0.0;
  auto worst_gradient = 0.0;
  for (const auto& path : cases) {
    SCOPED_TRACE(path.filename().string());
    const auto file = nc::File(path);
    const auto source = path.string();
    auto read = [&](std::string_view name) { return nc::ReadDoubles(file.Id(), name, source).values; };
    const auto kind = *nc::TextAttribute(file.Id(), nc::GLOBAL, "domain", source);
    const auto method = *nc::TextAttribute(file.Id(), nc::GLOBAL, "method", source);
    const auto bounds = *nc::NumberAttribute(file.Id(), nc::GLOBAL, "bounds", source);
    const auto padding = *nc::NumberAttribute(file.Id(), nc::GLOBAL, "padding", source);
    const auto x = read("x");
    const auto box = kind == "box";
    const auto domain =
        box ? flat::Domain::Box({bounds[0], bounds[1]}, {bounds[2], bounds[3]},
                                {padding[0], padding[1]})
            : flat::Domain::Rectangle({bounds[0], bounds[1]}, padding[0]);
    EXPECT_EQ(domain.Kind(), kind);
    const auto grid = box ? flat::Grid(domain, x.size(), read("y").size())
                          : flat::Grid(domain, x.size());
    // the nodes are the Python's, to the bit
    EXPECT_EQ(grid.X(), x);
    if (box) {
      EXPECT_EQ(grid.Y(), read("y"));
    }
    const auto values = read("values");
    const auto f = flat::LateralInterpolant(grid, values, flat::LateralRuleNamed(method));
    const auto qx = read("query_x");
    const auto qy = read("query_y");
    const auto expected = read("expected");
    const auto dx = read("expected_dx");
    const auto dy = read("expected_dy");
    const auto scale = Largest(values);
    const auto dscale = std::max(Largest(dx), Largest(dy));
    for (auto i = std::size_t{0}; i < qx.size(); ++i) {
      SCOPED_TRACE(std::format("({}, {})", qx[i], qy[i]));
      const auto value = f.Value(qx[i], qy[i]);
      EXPECT_NEAR(value, expected[i], 1e-12 * scale);
      const auto [gx, gy] = f.Gradient(qx[i], qy[i]);
      EXPECT_NEAR(gx, dx[i], 1e-10 * dscale);
      EXPECT_NEAR(gy, dy[i], 1e-10 * dscale);
      worst_value = std::max(worst_value, std::abs(value - expected[i]) / scale);
      worst_gradient = std::max({worst_gradient, std::abs(gx - dx[i]) / dscale,
                                 std::abs(gy - dy[i]) / dscale});
    }
    const auto mean = nc::NumberAttribute(file.Id(), nc::GLOBAL, "mean", source)->at(0);
    EXPECT_NEAR(f.Mean(), mean, 1e-14 * scale);
    EXPECT_NEAR(grid.Mean(values), mean, 1e-14 * scale);
  }
  RecordProperty("worst_relative_value", std::format("{:.3e}", worst_value));
  RecordProperty("worst_relative_gradient", std::format("{:.3e}", worst_gradient));
}

TEST(CartesianRules, TheRulesArePeriodicAndReadTheNodes) {
  const auto domain = flat::Domain::Box({0.0, 4.0}, {-1.0, 2.0}, {1.0, 0.0});
  const auto grid = flat::Grid(domain, 6, 5);
  auto values = std::vector<double>{};
  for (auto index = std::size_t{0}; index < grid.Size(); ++index) {
    const auto [x, y] = grid.Node(index);
    values.push_back(std::sin(x) + 0.3 * y * y);
  }
  for (const auto rule : {LateralRule::Linear, LateralRule::Cubic}) {
    const auto f = flat::LateralInterpolant(grid, values, rule);
    for (auto index = std::size_t{0}; index < grid.Size(); ++index) {
      const auto [x, y] = grid.Node(index);
      EXPECT_NEAR(f.Value(x, y), values[index], 1e-14);
      // one period on along either axis, and two back
      EXPECT_NEAR(f.Value(x + 6.0, y), values[index], 1e-13);
      EXPECT_NEAR(f.Value(x, y - 6.0), values[index], 1e-13);
    }
    EXPECT_NEAR(f.Value(0.7, 0.3), f.Value(0.7 - 12.0, 0.3 + 9.0), 1e-13);
  }
  // a section's y is never read
  const auto section = flat::Grid(flat::Domain::Rectangle({0.0, 1.0}), 4);
  const auto g = flat::LateralInterpolant(section, std::vector{1.0, 2.0, 4.0, 3.0},
                                          LateralRule::Cubic);
  EXPECT_EQ(g.Value(0.3, 0.0), g.Value(0.3, 1e9));
  EXPECT_EQ(g.Gradient(0.3, -5.0)[1], 0.0);
  EXPECT_NEAR(g.Value(0.5, 0.0), 4.0, 1e-14);
  EXPECT_NEAR(g.Value(-0.25, 0.0), 3.0, 1e-14);
}

TEST(CartesianRules, TheDomainAndTheGridAreAsThePythonLaysThem) {
  const auto box = flat::Domain::Box({0.0, 400e3}, {0.0, 300e3}, {50e3, 0.0});
  EXPECT_TRUE(box.IsBox());
  EXPECT_EQ(box.Dimension(), 2u);
  EXPECT_EQ(box.Extent(0), (std::array{-50e3, 450e3}));
  EXPECT_EQ(box.Length(0), 500e3);
  EXPECT_EQ(box.Length(1), 300e3);
  EXPECT_EQ(box.BoundsList(), (std::vector{0.0, 400e3, 0.0, 300e3}));
  EXPECT_EQ(box.PaddingList(), (std::vector{50e3, 0.0}));
  EXPECT_TRUE(box.Contains(400e3, 150e3));
  EXPECT_FALSE(box.Contains(-10e3, 150e3));  // the padding is outside
  const auto section = flat::Domain::Rectangle({0.0, 2.0}, 0.3);
  EXPECT_FALSE(section.IsBox());
  EXPECT_EQ(section.Kind(), "section");
  EXPECT_EQ(section.BoundsList(), (std::vector{0.0, 2.0}));
  EXPECT_EQ(section.PaddingList(), (std::vector{0.3}));
  EXPECT_TRUE(section.Contains(1.0, 1e12));  // every y is inside
  EXPECT_THROW(static_cast<void>(section.Bounds(1)), Error);
  EXPECT_NE(box, flat::Domain::Box({0.0, 400e3}, {0.0, 300e3}));
  EXPECT_THROW(static_cast<void>(flat::Domain::Box({1.0, 1.0}, {0.0, 1.0})), Error);
  EXPECT_THROW(static_cast<void>(flat::Domain::Rectangle({0.0, 1.0}, -0.1)), Error);

  const auto grid = flat::Grid(box, 10, 6);
  EXPECT_EQ(grid.Shape(), (std::vector<std::size_t>{10, 6}));
  EXPECT_EQ(grid.X().front(), -50e3);
  EXPECT_EQ(grid.X()[1], -50e3 + 500e3 / 10.0);
  EXPECT_EQ(grid.Spacing(1), 50e3);
  EXPECT_EQ(grid.Area(), 500e3 * 300e3);
  EXPECT_EQ(grid.Node(7), (std::array{grid.X()[1], grid.Y()[1]}));
  const auto line = flat::Grid(section, 5);
  EXPECT_EQ(line.NY(), 1u);
  EXPECT_TRUE(line.Y().empty());
  EXPECT_EQ(line.Node(3), (std::array{line.X()[3], 0.0}));
  EXPECT_THROW(static_cast<void>(flat::Grid(box, 4)), Error);
  EXPECT_THROW(static_cast<void>(flat::Grid(section, 4, 4)), Error);
  EXPECT_THROW(static_cast<void>(flat::Grid(section, 0)), Error);
  EXPECT_THROW(static_cast<void>(grid.Mean(std::vector{1.0, 2.0})), Error);
}

TEST(CartesianRules, TheLocalFrameIsUpXY) {
  // a vector's first local component is its upward one: (x, y, z) = (l1, l2, l0)
  const auto local = std::vector{3.0, 1.0, 2.0};
  EXPECT_EQ(flat::ToCartesian(local, 1, false), (std::vector{1.0, 2.0, 3.0}));
  EXPECT_EQ(flat::ToLocal(flat::ToCartesian(local, 1, false), 1, false), local);
  // a symmetric tensor in Voigt form (11, 22, 33, 23, 13, 12): local 1 is up
  const auto stress = std::vector{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  // Cartesian xx = local 22, yy = local 33, zz = local 11, yz = local 13 (l31),
  // xz = local 12 (l21), xy = local 23
  EXPECT_EQ(flat::ToCartesian(stress, 2, true), (std::vector{2.0, 3.0, 1.0, 5.0, 6.0, 4.0}));
  EXPECT_EQ(flat::ToLocal(flat::ToCartesian(stress, 2, true), 2, true), stress);
  const auto point = flat::FlatPoint({1.0, 2.0, -30.0});
  EXPECT_EQ(point.depth, 30.0);
  EXPECT_EQ(flat::CartesianPoint(point), (planetmodel::Vector3{1.0, 2.0, -30.0}));
}

TEST(CartesianRules, RefuseWhatTheGridCannotCarryAndTheFourierRule) {
  const auto domain = flat::Domain::Box({0.0, 1.0}, {0.0, 1.0});
  const auto thin = flat::Grid(domain, 4, 2);
  const auto values = std::vector<double>(8, 1.0);
  EXPECT_NO_THROW(static_cast<void>(flat::LateralInterpolant(thin, values, LateralRule::Linear)));
  EXPECT_THROW(static_cast<void>(flat::LateralInterpolant(thin, values, LateralRule::Cubic)),
               Error);
  EXPECT_THROW(flat::CheckRule(thin, LateralRule::Cubic), Error);
  EXPECT_THROW(static_cast<void>(flat::LateralInterpolant(
                   thin, std::vector<double>(7, 1.0), LateralRule::Linear)),
               Error);
  auto holed = values;
  holed[3] = std::nan("");
  EXPECT_THROW(static_cast<void>(flat::LateralInterpolant(thin, holed, LateralRule::Linear)),
               Error);
  const auto f = flat::LateralInterpolant(thin, values, LateralRule::Linear);
  EXPECT_THROW(static_cast<void>(f.Value(std::nan(""), 0.0)), Error);
  EXPECT_EQ(flat::LateralRuleNamed("linear"), LateralRule::Linear);
  EXPECT_EQ(flat::LateralRuleNamed("cubic"), LateralRule::Cubic);
  try {
    static_cast<void>(flat::LateralRuleNamed("fourier", "a file", "part_0/vs"));
    FAIL();
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "part_0/vs");
    EXPECT_NE(std::string{e.what()}.find("the Python's alone"), std::string::npos);
  }
  EXPECT_THROW(static_cast<void>(flat::LateralRuleNamed("harmonic")), Error);
}
