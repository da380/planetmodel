// The surface file: the fixture read and held to its expectations; a file
// built from known fields written, read back and left for the Python to
// read; and the refusals, before anything is written.

#include "planetmodel/surfacefile.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/grid.hpp"
#include "planetmodel/units.hpp"

namespace {

namespace pm = planetmodel;

constexpr auto kNaN = std::numeric_limits<double>::quiet_NaN();

/// The flat index of an expectation's sample in a variable's values.
std::size_t FlatIndex(const std::string& tag, std::size_t i, std::size_t j, std::size_t ntheta,
                      std::size_t nphi) {
  const auto k = tag == "-" ? std::size_t{0} : static_cast<std::size_t>(std::stoul(tag));
  return (k * ntheta + i) * nphi + j;
}

TEST(SurfaceFile, ReadsTheFixtureToItsExpectations) {
  PLANETMODEL_FIXTURE(path, "surface_basic.nc");
  PLANETMODEL_FIXTURE(expect, "surface_basic.expect");
  const auto file = pm::ReadSurfaceFile(path);
  EXPECT_EQ(file.schema, pm::SURFACE_SCHEMA);
  EXPECT_FALSE(file.source.empty());
  EXPECT_FALSE(file.created.empty());
  ASSERT_TRUE(file.grid && file.time && file.instant);
  const auto ntheta = file.grid->NTheta();
  const auto nphi = file.grid->NPhi();
  EXPECT_EQ(pm::detail::TextOf(file.meta, "epoch").value_or(""), "present");
  EXPECT_EQ(file.citation, std::vector<std::string>{"a work"});

  auto in = std::ifstream(expect);
  ASSERT_TRUE(in) << "cannot open " << expect;
  auto line = std::string{};
  auto checked = 0;
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
      const auto& axis = what == "time" ? *file.time : *file.instant;
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
    } else {
      FAIL() << "unknown expectation: " << line;
    }
    ++checked;
  }
  EXPECT_GT(checked, 30);
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
                       {.support = "node", .time_interpolation = "linear",
                        .dimensions = pm::units::LENGTH}));
  file.variables.emplace("rate", pm::SurfaceField("rate", on_grid(2, [](auto k, auto i, auto j) {
                                                    return 10.0 * k + 1.0 * i - 0.1 * j;
                                                  }),
                                                  {.support = "interval"}));
  file.variables.emplace(
      "instantaneous",
      pm::SurfaceField("instantaneous",
                       on_grid(2, [](auto k, auto i, auto j) { return 1.0 * k + 2.0 * i + 0.2 * j; }),
                       {.support = "instant", .lateral_interpolation = "cubic"}));
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
  expect_refusal([](pm::SurfaceFile& f) { f.source = "  "; }, "an empty source");
  expect_refusal([](pm::SurfaceFile& f) { f.radius = -1.0; }, "a negative radius");
  expect_refusal([](pm::SurfaceFile& f) { f.time = std::vector<double>{1.0, 1.0, 2.0}; },
                 "a time axis that does not increase");
  expect_refusal(
      [](pm::SurfaceFile& f) {
        f.variables.emplace("time", pm::SurfaceField("time", f.variables.at("rate").values,
                                                     {.support = "interval"}));
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

}  // namespace
