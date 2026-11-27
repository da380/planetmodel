// The surface file: the fixtures read and held to their expectations, the
// history's values and its fields as functions among them, and written
// back; a file built from known fields written, read back and left for the
// Python to read; the malformed copies refused at their places; and the
// refusals of the writer, before anything is written.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/grid.hpp"
#include "planetmodel/surfacefile.hpp"
#include "planetmodel/time.hpp"
#include "planetmodel/units.hpp"
#include "planetmodel/version.hpp"

namespace {

namespace pm = planetmodel;

constexpr auto kNaN = std::numeric_limits<double>::quiet_NaN();

/// The flat index of an expectation's sample in a variable's values.
std::size_t FlatIndex(const std::string& tag, std::size_t i, std::size_t j, std::size_t ntheta,
                      std::size_t nphi) {
  const auto k = tag == "-" ? std::size_t{0} : static_cast<std::size_t>(std::stoul(tag));
  return (k * ntheta + i) * nphi + j;
}

/// What `HeldToItsExpectations` met of a file's expectations.
struct Met {
  int facts = 0;     ///< Every line.
  int functions = 0; ///< The lines of a field as a function.
};

/// A surface file held to the expectations the Python wrote beside it, one
/// fact a line: its numbers and metadata, its history's values at a date
/// and, by its lateral rule, each field there as a function.
void HeldToItsExpectations(const pm::SurfaceFile& file, const std::filesystem::path& expect,
                           Met& met) {
  ASSERT_TRUE(file.grid);
  const auto ntheta = file.grid->NTheta();
  const auto nphi = file.grid->NPhi();
  const auto history = pm::SurfaceHistory(file);
  EXPECT_EQ(history.GetGrid(), *file.grid);
  auto options = [](const std::string& extrapolation) {
    return pm::DateOptions{.extrapolation = extrapolation == "hold"
                                                ? pm::Extrapolation::Hold
                                                : pm::Extrapolation::Refuse};
  };
  auto in = std::ifstream(expect);
  ASSERT_TRUE(in) << "cannot open " << expect;
  auto line = std::string{};
  auto& checked = met.facts;
  while (std::getline(in, line)) {
    auto words = std::istringstream(line);
    auto what = std::string{};
    words >> what;
    if (what == "scales") {
      double length = 0, mass = 0, time = 0, temperature = 0;
      words >> length >> mass >> time >> temperature;
      EXPECT_EQ(file.scales.Length(), length) << line;
      EXPECT_EQ(file.scales.Mass(), mass) << line;
      EXPECT_EQ(file.scales.Time(), time) << line;
      EXPECT_EQ(file.scales.Temperature(), temperature) << line;
    } else if (what == "radius") {
      double radius = 0;
      words >> radius;
      ASSERT_TRUE(file.radius) << line;
      EXPECT_EQ(*file.radius, radius) << line;
    } else if (what == "grid") {
      auto kind = std::string{};
      auto lmax = 0;
      auto nt = std::size_t{0}, np = std::size_t{0};
      words >> kind >> lmax >> nt >> np;
      EXPECT_EQ(file.grid->Kind(), kind) << line;
      ASSERT_TRUE(file.grid->Lmax()) << line;
      EXPECT_EQ(*file.grid->Lmax(), lmax) << line;
      EXPECT_EQ(ntheta, nt) << line;
      EXPECT_EQ(nphi, np) << line;
    } else if (what == "time" || what == "instant") {
      auto k = std::size_t{0};
      auto value = 0.0;
      words >> k >> value;
      const auto& held = what == "time" ? file.time : file.instant;
      ASSERT_TRUE(held) << line;
      const auto& axis = *held;
      ASSERT_LT(k, axis.size()) << line;
      EXPECT_EQ(axis[k], value) << line;
    } else if (what == "variable") {
      auto name = std::string{}, support = std::string{}, in_time = std::string{},
           lateral = std::string{}, first = std::string{};
      words >> name >> support >> in_time >> lateral >> first;
      ASSERT_TRUE(file.variables.contains(name)) << line;
      const auto& attrs = file.variables.at(name).attributes;
      EXPECT_EQ(pm::detail::TextOf(attrs, "time_support").value_or(""), support) << line;
      EXPECT_EQ(pm::detail::TextOf(attrs, "time_interpolation").value_or(""), in_time) << line;
      EXPECT_EQ(pm::detail::TextOf(attrs, "lateral_interpolation").value_or(""), lateral) << line;
      EXPECT_EQ(pm::TimeSupportName(history.Support(name)), support) << line;
      if (first == "-") {
        EXPECT_FALSE(attrs.contains("dimensions")) << line;
      } else {
        auto exps = std::vector<double>{std::stod(first)};
        auto e = 0.0;
        while (words >> e) exps.push_back(e);
        EXPECT_EQ(pm::detail::NumbersOf(attrs, "dimensions").value_or(std::vector<double>{}),
                  exps)
            << line;
      }
    } else if (what == "value" || what == "absent") {
      auto name = std::string{}, tag = std::string{};
      auto i = std::size_t{0}, j = std::size_t{0};
      words >> name >> tag >> i >> j;
      ASSERT_TRUE(file.variables.contains(name)) << line;
      const auto& values = file.variables.at(name).values;
      const auto flat = FlatIndex(tag, i, j, ntheta, nphi);
      ASSERT_LT(flat, values.size()) << line;
      if (what == "absent") {
        EXPECT_TRUE(std::isnan(values[flat])) << line;
      } else {
        auto value = 0.0;
        words >> value;
        EXPECT_EQ(values[flat], value) << line;
      }
    } else if (what == "poles") {
      auto name = std::string{}, tag = std::string{};
      auto north = 0.0, south = 0.0;
      words >> name >> tag >> north >> south;
      const auto key = name + "_poles";
      ASSERT_TRUE(file.variables.contains(key)) << line;
      const auto& values = file.variables.at(key).values;
      const auto k = tag == "-" ? std::size_t{0} : static_cast<std::size_t>(std::stoul(tag));
      ASSERT_LT(2 * k + 1, values.size()) << line;
      EXPECT_EQ(values[2 * k], north) << line;
      EXPECT_EQ(values[2 * k + 1], south) << line;
    } else if (what == "sample") {
      auto name = std::string{}, tag = std::string{}, extrap = std::string{};
      auto i = std::size_t{0}, j = std::size_t{0};
      auto value = 0.0;
      words >> name >> tag >> extrap >> i >> j >> value;
      const auto date = tag == "-" ? std::optional<double>{} : std::optional{std::stod(tag)};
      const auto at = history.Values(name, date, options(extrap));
      ASSERT_LT(i * nphi + j, at.size()) << line;
      EXPECT_DOUBLE_EQ(at[i * nphi + j], value) << line;
    } else if (what == "function") {
      auto name = std::string{}, tag = std::string{}, extrap = std::string{};
      auto theta = 0.0, phi = 0.0, value = 0.0, dtheta = 0.0, dphi = 0.0;
      words >> name >> tag >> extrap >> theta >> phi >> value >> dtheta >> dphi;
      const auto date = tag == "-" ? std::optional<double>{} : std::optional{std::stod(tag)};
      const auto f = history.Function(name, date, options(extrap));
      auto scale = 0.0;
      for (const auto v : history.Values(name, date, options(extrap))) {
        scale = std::max(scale, std::abs(v));
      }
      EXPECT_NEAR(f.Value(theta, phi), value, 1e-12 * scale) << line;
      const auto [gt, gp] = f.Gradient(theta, phi);
      // a derivative is of the size of the values over the grid's spacing
      const auto slope = scale * static_cast<double>(std::max(ntheta, nphi));
      EXPECT_NEAR(gt, dtheta, 1e-10 * slope) << line;
      EXPECT_NEAR(gp, dphi, 1e-10 * slope) << line;
      ++met.functions;
    } else {
      FAIL() << "unknown expectation: " << line;
    }
    ++checked;
  }
}

TEST(SurfaceFile, ReadsTheFixtureToItsExpectations) {
  PLANETMODEL_FIXTURE(path, "surface_basic.nc");
  PLANETMODEL_FIXTURE(expect, "surface_basic.expect");
  const auto file = pm::ReadSurfaceFile(path);
  EXPECT_EQ(file.schema, pm::SURFACE_SCHEMA);
  EXPECT_FALSE(file.source.empty());
  EXPECT_FALSE(file.created.empty());
  ASSERT_TRUE(file.grid && file.time && file.instant);
  EXPECT_TRUE(file.grid->LacksPole());
  EXPECT_EQ(pm::detail::TextOf(file.meta, "epoch").value_or(""), "present");
  EXPECT_EQ(file.citation, std::vector<std::string>{"a work"});
  auto met = Met{};
  HeldToItsExpectations(file, expect, met);
  EXPECT_GT(met.facts, 60);
  // the fields with stored polar values, read by the bilinear rule and by
  // the bicubic, between nodes in time and held beyond them
  EXPECT_GE(met.functions, 30);
}

TEST(SurfaceFile, ReadsAFileOnAGridHoldingBothPolesToItsExpectations) {
  // the fields carry no polar values, and need none: the poles are nodes
  PLANETMODEL_FIXTURE(path, "surface_poles.nc");
  PLANETMODEL_FIXTURE(expect, "surface_poles.expect");
  const auto file = pm::ReadSurfaceFile(path);
  ASSERT_TRUE(file.grid && file.time);
  EXPECT_FALSE(file.grid->LacksPole());
  EXPECT_FALSE(file.instant);
  EXPECT_FALSE(file.radius);
  EXPECT_TRUE(file.source.starts_with("planetmodel "));
  for (const auto& [name, var] : file.variables) EXPECT_FALSE(name.ends_with("_poles")) << name;
  auto met = Met{};
  HeldToItsExpectations(file, expect, met);
  EXPECT_GT(met.facts, 30);
  EXPECT_GE(met.functions, 20);
}

/// The file the writer's test leaves for the Python, from formulaic
/// fields `test_cpp_written.py` rebuilds.
pm::SurfaceFile WrittenSurface() {
  auto file = pm::SurfaceFile{};
  file.scales = pm::Scales({.length = 2.0, .mass = 3.0, .time = 5.0});
  file.grid = pm::GaussLegendre(4);
  file.time = std::vector<double>{0.0, 1.0, 2.5};
  file.instant = std::vector<double>{1.5, 2.5};
  file.radius = 1.25;
  file.source = "planetmodel cpp tests";
  file.title = "written surface";
  file.meta.emplace("epoch", std::string{"present"});
  file.meta.emplace("lmax", std::vector<long long>{4});
  file.citation = {"one", "two"};
  const auto ntheta = file.grid->NTheta();
  const auto nphi = file.grid->NPhi();
  auto on_grid = [&](std::size_t leading, auto value) {
    auto out = std::vector<double>{};
    for (auto k = std::size_t{0}; k < leading; ++k) {
      for (auto i = std::size_t{0}; i < ntheta; ++i) {
        for (auto j = std::size_t{0}; j < nphi; ++j) out.push_back(value(k, i, j));
      }
    }
    return out;
  };
  auto topography = on_grid(1, [](auto, auto i, auto j) { return 1.0 * i + 0.01 * j; });
  topography[1] = kNaN;
  file.variables.emplace(
      "topography",
      pm::SurfaceField("topography", std::move(topography),
                       {.dimensions = pm::units::LENGTH, .meaning = "present-day topography"}));
  file.variables.emplace("topography_poles",
                         pm::FileVariable{.name = "topography_poles", .values = {1.5, -0.5}});
  file.variables.emplace(
      "ice_thickness",
      pm::SurfaceField("ice_thickness",
                       on_grid(3, [](auto k, auto i, auto j) {
                         return 100.0 * k + 1.0 * i + 0.01 * j;
                       }),
                       {.support = pm::TimeSupport::Node,
                        .time_interpolation = pm::TimeRule::Linear,
                        .dimensions = pm::units::LENGTH}));
  file.variables.emplace("rate", pm::SurfaceField("rate", on_grid(2, [](auto k, auto i, auto j) {
                                                    return 10.0 * k + 1.0 * i - 0.1 * j;
                                                  }),
                                                  {.support = pm::TimeSupport::Interval}));
  file.variables.emplace(
      "instantaneous",
      pm::SurfaceField("instantaneous",
                       on_grid(2, [](auto k, auto i, auto j) { return 1.0 * k + 2.0 * i + 0.2 * j; }),
                       {.support = pm::TimeSupport::Instant,
                        .lateral_interpolation = pm::LateralRule::Cubic}));
  return file;
}

TEST(SurfaceFile, WritesReadsBackAndLeavesTheFileForThePython) {
  const auto file = WrittenSurface();
  const auto path = pm::testing::WrittenDir() / "surface_written.nc";
  pm::WriteSurfaceFile(path, file);
  const auto read = pm::ReadSurfaceFile(path);
  EXPECT_EQ(read.source, file.source);
  EXPECT_FALSE(read.created.empty());
  EXPECT_NE(read.created.find('T'), std::string::npos);
  EXPECT_EQ(read.title, file.title);
  EXPECT_EQ(read.scales.Length(), 2.0);
  EXPECT_EQ(read.time, file.time);
  EXPECT_EQ(read.instant, file.instant);
  EXPECT_EQ(read.radius, file.radius);
  EXPECT_EQ(read.citation, file.citation);
  EXPECT_EQ(pm::detail::TextOf(read.meta, "epoch").value_or(""), "present");
  ASSERT_EQ(read.variables.size(), file.variables.size());
  for (const auto& [name, var] : file.variables) {
    ASSERT_TRUE(read.variables.contains(name)) << name;
    const auto& got = read.variables.at(name).values;
    ASSERT_EQ(got.size(), var.values.size()) << name;
    for (auto n = std::size_t{0}; n < got.size(); ++n) {
      if (std::isnan(var.values[n])) {
        EXPECT_TRUE(std::isnan(got[n])) << name;
      } else {
        EXPECT_EQ(got[n], var.values[n]) << name;
      }
    }
  }
  const auto& thickness = read.variables.at("ice_thickness").attributes;
  EXPECT_EQ(pm::detail::TextOf(thickness, "time_interpolation").value_or(""), "linear");
  EXPECT_TRUE(thickness.contains("units"));
  // the reader holds the file to the structure, so reading is the check
  EXPECT_EQ(read.schema, pm::SURFACE_SCHEMA);
}

TEST(SurfaceHistory, ServesEverySupportAndRefusesTheRest) {
  const auto file = WrittenSurface();
  const auto history = pm::SurfaceHistory(file);
  const auto size = file.grid->NTheta() * file.grid->NPhi();
  // linear: the midpoint of the first two nodes, exactly half their sum
  const auto& nodes = file.variables.at("ice_thickness").values;
  const auto mid = history.Values("ice_thickness", 0.5);
  for (auto n = std::size_t{0}; n < size; ++n) {
    EXPECT_DOUBLE_EQ(mid[n], 0.5 * nodes[n] + 0.5 * nodes[size + n]);
  }
  // held outside the span: the last node's values
  const auto hold = pm::DateOptions{.extrapolation = pm::Extrapolation::Hold};
  const auto held = history.Values("ice_thickness", 9.0, hold);
  for (auto n = std::size_t{0}; n < size; ++n) EXPECT_EQ(held[n], nodes[2 * size + n]);
  // the last interval is closed at its end
  const auto& rate = file.variables.at("rate").values;
  EXPECT_EQ(history.Values("rate", 2.5).front(), rate[size]);
  // an instant is served exactly, and never held
  EXPECT_EQ(history.Values("instantaneous", 1.5).front(),
            file.variables.at("instantaneous").values.front());
  EXPECT_THROW(static_cast<void>(history.Values("instantaneous", 1.6, hold)), pm::Error);
  // a date a rounding error from an instant is that instant, unless the
  // tolerance is zero
  EXPECT_EQ(history.Values("instantaneous", 1.5 + 1e-14).front(),
            file.variables.at("instantaneous").values.front());
  EXPECT_THROW(static_cast<void>(history.Values("instantaneous", 1.5 + 1e-14, {.time_rtol = 0.0})),
               pm::Error);
  // a static field takes any date; the NaN point stays NaN
  EXPECT_TRUE(std::isnan(history.Values("topography", 7.0)[1]));
  EXPECT_THROW(static_cast<void>(history.Values("ice_thickness")), pm::Error);
  EXPECT_THROW(static_cast<void>(history.Values("ice_thickness", 9.0)), pm::Error);
  EXPECT_THROW(static_cast<void>(history.Values("missing", 0.0)), pm::Error);
  EXPECT_THROW(static_cast<void>(history.Values("topography_poles")), pm::Error);
  EXPECT_EQ(history.Names().size(), 4u);
  EXPECT_EQ(history.Support("rate"), pm::TimeSupport::Interval);
  EXPECT_EQ(history.Support("topography"), pm::TimeSupport::Static);
  EXPECT_THROW(static_cast<void>(history.Support("missing")), pm::Error);
  EXPECT_EQ(history.GetGrid(), *file.grid);
}

/// A file of one smooth field at two time nodes on `grid`, read between
/// them by `rule`, with its polar values where `poles` are given: at node
/// k the field is (1 + k) (2 + cos(theta) + sin(theta) cos(phi)).
pm::SurfaceFile SmoothSurface(const pm::AngularGrid& grid, pm::LateralRule rule, bool poles) {
  auto file = pm::SurfaceFile{};
  file.grid = grid;
  file.source = "planetmodel cpp tests";
  file.time = std::vector<double>{0.0, 2.0};
  auto values = std::vector<double>{};
  for (const auto k : {1.0, 2.0}) {
    for (const auto theta : grid.Colatitudes()) {
      for (const auto phi : grid.Longitudes()) {
        values.push_back(k * (2.0 + std::cos(theta) + std::sin(theta) * std::cos(phi)));
      }
    }
  }
  file.variables.emplace(
      "h", pm::SurfaceField("h", std::move(values),
                            {.support = pm::TimeSupport::Node,
                             .time_interpolation = pm::TimeRule::Linear,
                             .lateral_interpolation = rule}));
  if (poles) {
    file.variables.emplace("h_poles", pm::FileVariable{.name = "h_poles",
                                                       .values = {3.0, 1.0, 6.0, 2.0}});
  }
  return file;
}

TEST(SurfaceHistory, GivesAFieldAtADateAsAFunctionWithItsStoredPoles) {
  // the stored polar values are read at the date by the rows the values are
  const auto grid = pm::GaussLegendre(8);
  for (const auto rule : {pm::LateralRule::Linear, pm::LateralRule::Cubic}) {
    const auto file = SmoothSurface(grid, rule, true);
    pm::ValidateSurfaceFile(file, "a built file");
    const auto history = pm::SurfaceHistory(file);
    const auto f = history.Function("h", 0.5);
    // a quarter of the way from the first node to the second: 1.25 times it
    EXPECT_DOUBLE_EQ(f.Value(0.0, 1.0), 1.25 * 3.0);
    EXPECT_DOUBLE_EQ(f.Value(std::numbers::pi, 4.0), 1.25 * 1.0);
    const auto at = history.Values("h", 0.5);
    for (auto i = std::size_t{0}; i < grid.NTheta(); ++i) {
      for (auto j = std::size_t{0}; j < grid.NPhi(); j += 5) {
        EXPECT_NEAR(f.Value(grid.Colatitudes()[i], grid.Longitudes()[j]), at[i * grid.NPhi() + j],
                    1e-13);
      }
    }
    // held beyond the span: the last node's values and poles
    const auto held = history.Function("h", 9.0, {.extrapolation = pm::Extrapolation::Hold});
    EXPECT_DOUBLE_EQ(held.Value(0.0, 0.0), 6.0);
    EXPECT_THROW(static_cast<void>(history.Function("h", 9.0)), pm::Error);
    EXPECT_THROW(static_cast<void>(history.Function("h")), pm::Error);
    EXPECT_THROW(static_cast<void>(history.Function("h_poles", 0.5)), pm::Error);
    EXPECT_THROW(static_cast<void>(history.Function("missing", 0.5)), pm::Error);
  }
  // the bicubic rule reads a smooth field between the nodes closely
  const auto smooth = pm::SurfaceHistory(SmoothSurface(grid, pm::LateralRule::Cubic, true));
  const auto f = smooth.Function("h", 0.0);
  EXPECT_NEAR(f.Value(1.1, 0.7), 2.0 + std::cos(1.1) + std::sin(1.1) * std::cos(0.7), 5e-4);
}

TEST(SurfaceHistory, RefusesAFunctionItCannotEvaluateAndKeepsItsValues) {
  const auto grid = pm::GaussLegendre(8);
  // on a grid lacking a pole, a field without stored polar values: the
  // Python finds them from the grid's expansion, which the C++ does not hold
  const auto bare = pm::SurfaceHistory(SmoothSurface(grid, pm::LateralRule::Linear, false));
  EXPECT_NO_THROW(static_cast<void>(bare.Values("h", 0.5)));
  try {
    static_cast<void>(bare.Function("h", 0.5));
    ADD_FAILURE() << "a function was made without its polar values";
  } catch (const pm::Error& e) {
    EXPECT_EQ(e.Where(), "h");
    EXPECT_NE(std::string{e.what()}.find("the grid lacks the north pole"), std::string::npos)
        << e.what();
  }
  // on a grid holding both poles none are needed
  const auto both = pm::Equiangular(9, 16, true);
  const auto whole = pm::SurfaceHistory(SmoothSurface(both, pm::LateralRule::Cubic, false));
  EXPECT_NEAR(whole.Function("h", 2.0).Value(0.0, 0.3), 6.0, 1e-12);
  // the harmonic rule, as a variable of a model file stating it: the field
  // is refused by name as a function, and its numbers are kept
  auto harmonic = SmoothSurface(grid, pm::LateralRule::Linear, true);
  harmonic.variables.at("h").attributes["lateral_interpolation"] = std::string{"harmonic"};
  const auto history = pm::SurfaceHistory(harmonic);
  EXPECT_EQ(history.Values("h", 0.0).size(), grid.NTheta() * grid.NPhi());
  try {
    static_cast<void>(history.Function("h", 0.0));
    ADD_FAILURE() << "a harmonic field was made a function";
  } catch (const pm::Error& e) {
    EXPECT_EQ(e.Where(), "h");
    EXPECT_NE(std::string{e.what()}.find("states the harmonic rule"), std::string::npos)
        << e.what();
  }
  // a point the field does not cover: no rule reads a NaN
  auto holed = SmoothSurface(grid, pm::LateralRule::Linear, true);
  holed.variables.at("h").values[3] = kNaN;
  const auto partial = pm::SurfaceHistory(holed);
  EXPECT_TRUE(std::isnan(partial.Values("h", 0.0)[3]));
  try {
    static_cast<void>(partial.Function("h", 0.0));
    ADD_FAILURE() << "a field with a NaN was made a function";
  } catch (const pm::Error& e) {
    EXPECT_EQ(e.Where(), "h");
    EXPECT_NE(std::string{e.what()}.find("take finite values"), std::string::npos) << e.what();
  }
  // the second node is whole, and a date on it reads it alone
  EXPECT_NO_THROW(static_cast<void>(partial.Function("h", 2.0)));
}

TEST(SurfaceFile, RefusesWhatTheFormatCannotHold) {
  const auto dir = pm::testing::WrittenDir();
  auto good = WrittenSurface();
  const auto refused = dir / "refused_surface.nc";

  auto expect_refusal = [&](auto edit, std::string_view what) {
    auto file = good;
    edit(file);
    EXPECT_THROW(pm::WriteSurfaceFile(refused, file), pm::Error) << what;
    EXPECT_FALSE(std::filesystem::exists(refused)) << what;
  };
  expect_refusal([](pm::SurfaceFile& f) { f.source = "  "; }, "a blank source");
  expect_refusal([](pm::SurfaceFile& f) { f.radius = -1.0; }, "a negative radius");
  expect_refusal([](pm::SurfaceFile& f) { f.time = std::vector<double>{1.0, 1.0, 2.0}; },
                 "a time axis that does not increase");
  expect_refusal(
      [](pm::SurfaceFile& f) {
        f.variables.emplace("time", pm::SurfaceField("time", f.variables.at("rate").values,
                                                     {.support = pm::TimeSupport::Interval}));
      },
      "a reserved name");
  expect_refusal(
      [](pm::SurfaceFile& f) {
        f.variables.at("rate").attributes["time_support"] = std::string{"sometimes"};
      },
      "an unknown support");
  expect_refusal(
      [](pm::SurfaceFile& f) {
        f.variables.at("rate").attributes["time_interpolation"] = std::string{"linear"};
      },
      "a time rule on an interval field");
  expect_refusal(
      [](pm::SurfaceFile& f) {
        f.variables.at("topography").attributes["lateral_interpolation"] =
            std::string{"nearest"};
      },
      "an unknown lateral rule");
  expect_refusal([](pm::SurfaceFile& f) { f.variables.at("rate").values.pop_back(); },
                 "too few values");
  expect_refusal([](pm::SurfaceFile& f) { f.time.reset(); },
                 "time-supported fields without a time axis");
  expect_refusal(
      [](pm::SurfaceFile& f) {
        f.grid = pm::Equiangular(static_cast<int>(f.grid->NTheta()),
                                 static_cast<int>(f.grid->NPhi()), true);
      },
      "polar values on a grid holding both poles");
}

TEST(SurfaceFile, RefusesALateralRuleItsGridCannotCarry) {
  // the harmonic rule needs a grid of a shipped kind with its band, and the
  // cubic rule rows and longitudes enough, on writing as on reading
  const auto shipped = pm::GaussLegendre(4);
  const auto custom = pm::AngularGrid(shipped.Colatitudes(), shipped.Longitudes());
  const auto bandless = pm::AngularGrid(shipped.Colatitudes(), shipped.Longitudes(),
                                        shipped.Kind(), std::nullopt, shipped.Weights());
  const auto thin = pm::Equiangular(4, 2);
  // the rule by its name, as a file states it: `harmonic` is no `LateralRule`
  auto on = [](const pm::AngularGrid& grid, std::string rule) {
    auto file = pm::SurfaceFile{};
    file.grid = grid;
    file.source = "planetmodel cpp tests";
    auto field = pm::SurfaceField("h", std::vector<double>(grid.NTheta() * grid.NPhi(), 1.0));
    field.attributes.insert_or_assign("lateral_interpolation", std::move(rule));
    file.variables.emplace("h", std::move(field));
    return file;
  };
  const auto refused = pm::testing::WrittenDir() / "refused_surface_rule.nc";
  struct Case {
    pm::AngularGrid grid;
    std::string rule;
    std::string phrase;
  };
  for (const auto& [grid, rule, phrase] :
       {Case{custom, "harmonic", "this grid is custom"},
        Case{bandless, "harmonic", "states no grid_lmax"},
        Case{thin, "cubic", "has 2 longitudes"}}) {
    const auto file = on(grid, rule);
    try {
      pm::ValidateSurfaceFile(file, "a built file");
      ADD_FAILURE() << rule << " was accepted on a grid of kind " << grid.Kind();
    } catch (const pm::Error& e) {
      EXPECT_EQ(e.Source(), "a built file");
      EXPECT_EQ(e.Where(), "h");
      EXPECT_NE(std::string{e.what()}.find("states the lateral rule " + rule), std::string::npos)
          << e.what();
      EXPECT_NE(std::string{e.what()}.find(phrase), std::string::npos) << e.what();
    }
    EXPECT_THROW(pm::WriteSurfaceFile(refused, file), pm::Error) << rule;
    EXPECT_FALSE(std::filesystem::exists(refused)) << rule;
  }
  // each grid carries the rules it can
  EXPECT_NO_THROW(pm::ValidateSurfaceFile(on(shipped, "harmonic"), "a built file"));
  EXPECT_NO_THROW(pm::ValidateSurfaceFile(on(custom, "cubic"), "a built file"));
  EXPECT_NO_THROW(pm::ValidateSurfaceFile(on(thin, "linear"), "a built file"));
}

TEST(SurfaceFile, StatesAFieldsSupportAndRulesByTheirNames) {
  // the enumerations are the interface's, the names the file's
  const auto field = pm::SurfaceField(
      "h", {1.0}, {.support = pm::TimeSupport::Node, .time_interpolation = pm::TimeRule::Previous,
                   .lateral_interpolation = pm::LateralRule::Cubic});
  EXPECT_EQ(pm::detail::TextOf(field.attributes, "time_support").value_or(""), "node");
  EXPECT_EQ(pm::detail::TextOf(field.attributes, "time_interpolation").value_or(""), "previous");
  EXPECT_EQ(pm::detail::TextOf(field.attributes, "lateral_interpolation").value_or(""), "cubic");
  const auto plain = pm::SurfaceField("h", {1.0});
  EXPECT_EQ(pm::detail::TextOf(plain.attributes, "time_support").value_or(""), "static");
  EXPECT_EQ(pm::detail::TextOf(plain.attributes, "time_interpolation").value_or(""), "none");
  EXPECT_EQ(pm::detail::TextOf(plain.attributes, "lateral_interpolation").value_or(""), "linear");
}

TEST(SurfaceFile, StampsThisLibraryAsItsSourceUnlessGivenOne) {
  // as the model file's writer does
  auto file = WrittenSurface();
  file.source = "";
  const auto path = pm::testing::WrittenDir() / "surface_stamped.nc";
  pm::WriteSurfaceFile(path, file);
  EXPECT_EQ(pm::ReadSurfaceFile(path).source, "planetmodel C++ " + std::string{pm::Version()});
  // a file as read states its source: validation alone refuses one without
  EXPECT_THROW(pm::ValidateSurfaceFile(file, "a built file"), pm::Error);
}

TEST(SurfaceFile, WritesACitationByTheRuleOfAModels) {
  // each work trimmed of white space at its ends, an empty one or a repeat
  // dropped, and a work of more than one line refused
  auto file = WrittenSurface();
  file.citation = {" A. ", "", "B.", "A.", "\tC.\n"};
  const auto path = pm::testing::WrittenDir() / "surface_cited.nc";
  pm::WriteSurfaceFile(path, file);
  EXPECT_EQ(pm::ReadSurfaceFile(path).citation, (std::vector<std::string>{"A.", "B.", "C."}));
  file.citation = {"   ", ""};
  pm::WriteSurfaceFile(path, file);
  EXPECT_TRUE(pm::ReadSurfaceFile(path).citation.empty());
  file.citation = {"A.\nB."};
  const auto refused = pm::testing::WrittenDir() / "refused_surface_citation.nc";
  try {
    pm::WriteSurfaceFile(refused, file);
    ADD_FAILURE() << "a work of two lines was written";
  } catch (const pm::Error& e) {
    EXPECT_EQ(e.Where(), "citation");
    EXPECT_NE(std::string{e.what()}.find("more than one line"), std::string::npos) << e.what();
  }
  EXPECT_FALSE(std::filesystem::exists(refused));
}

TEST(SurfaceFile, RefusesWhatThePythonRefusesAtItsPlace) {
  // a field that does not run over its support's dimension and the grid's,
  // polar values that do not run over their field's and the poles, a scale
  // that is not finite, the harmonic rule on a grid that cannot carry it,
  // polar values on a grid holding both poles, and a `pole` dimension that
  // is not of two
  PLANETMODEL_FIXTURE(index, "surface_cases/bad.txt");
  auto in = std::ifstream(index);
  auto cases = 0;
  for (auto line = std::string{}; std::getline(in, line);) {
    const auto tab1 = line.find('\t');
    const auto tab2 = line.find('\t', tab1 + 1);
    const auto label = line.substr(0, tab1);
    const auto where = line.substr(tab1 + 1, tab2 - tab1 - 1);
    const auto phrase = line.substr(tab2 + 1);
    const auto path = index.parent_path() / ("bad_" + label + ".nc");
    try {
      static_cast<void>(pm::ReadSurfaceFile(path));
      ADD_FAILURE() << label << " was accepted";
    } catch (const pm::Error& e) {
      EXPECT_EQ(e.Where(), where) << label << ": " << e.what();
      EXPECT_NE(std::string{e.what()}.find(phrase), std::string::npos)
          << label << ": " << e.what();
    }
    ++cases;
  }
  EXPECT_GT(cases, 7);
}

TEST(SurfaceFile, WritesBackTheFixtureItReads) {
  // a variable read carries netCDF's own attributes (`_FillValue`), which
  // are the writer's to state
  PLANETMODEL_FIXTURE(fixture, "surface_basic.nc");
  const auto before = pm::ReadSurfaceFile(fixture);
  ASSERT_TRUE(before.variables.at("topography").attributes.contains("_FillValue"));
  const auto path = pm::testing::WrittenDir() / "surface_back_basic.nc";
  pm::WriteSurfaceFile(path, before);
  const auto after = pm::ReadSurfaceFile(path);
  EXPECT_EQ(after.source, before.source);
  EXPECT_EQ(after.created, before.created);
  EXPECT_EQ(after.title, before.title);
  EXPECT_EQ(after.scales, before.scales);
  EXPECT_EQ(after.time, before.time);
  EXPECT_EQ(after.instant, before.instant);
  EXPECT_EQ(after.radius, before.radius);
  EXPECT_EQ(after.citation, before.citation);
  EXPECT_EQ(after.meta, before.meta);
  EXPECT_EQ(after.grid->Kind(), before.grid->Kind());
  EXPECT_EQ(after.grid->Lmax(), before.grid->Lmax());
  EXPECT_EQ(after.grid->Weights(), before.grid->Weights());
  ASSERT_EQ(after.variables.size(), before.variables.size());
  for (const auto& [name, var] : before.variables) {
    SCOPED_TRACE(name);
    ASSERT_TRUE(after.variables.contains(name));
    const auto& got = after.variables.at(name);
    EXPECT_EQ(got.dimensions, var.dimensions);
    EXPECT_EQ(got.shape, var.shape);
    ASSERT_EQ(got.values.size(), var.values.size());
    for (auto n = std::size_t{0}; n < var.values.size(); ++n) {
      if (std::isnan(var.values[n])) {
        EXPECT_TRUE(std::isnan(got.values[n])) << n;
      } else {
        EXPECT_EQ(got.values[n], var.values[n]) << n;
      }
    }
    for (const auto& [key, value] : var.attributes) {
      if (key.starts_with('_')) continue;
      ASSERT_TRUE(got.attributes.contains(key)) << key;
      EXPECT_EQ(got.attributes.at(key), value) << key;
    }
  }
}

}  // namespace

#if defined(__unix__) || defined(__APPLE__)
TEST(SurfaceFile, GivesTheFileThePermissionsAPlainOpenWould) {
  const auto file = WrittenSurface();
  const auto dir = pm::testing::WrittenDir() / "permissions";
  std::filesystem::create_directories(dir);
  pm::testing::ExpectPlainPermissions(dir / "surface.nc", [&file](const std::filesystem::path& p) {
    pm::WriteSurfaceFile(p, file);
  });
}
#endif
