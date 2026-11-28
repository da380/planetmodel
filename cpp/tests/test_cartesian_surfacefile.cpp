// The flat surface file: the fixtures of a box and of a rectangle read and
// held to their expectations, the fields as functions of (x, y) among
// them, and written back; files built from known fields written, read
// back and left for the Python to read; each reader refusing the other's
// files by their schema; the malformed copies refused at their places;
// and the refusals of the writer, before anything is written.
#include "planetmodel/cartesian/surfacefile.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/surfacefile.hpp"
#include "planetmodel/time.hpp"
#include "planetmodel/units.hpp"
#include "planetmodel/version.hpp"

namespace {

namespace pm = planetmodel;
namespace flat = planetmodel::cartesian;

constexpr auto kNaN = std::numeric_limits<double>::quiet_NaN();

/// One fixture against its expectations, line by line.
void ExpectAsThePython(const std::string& label, double& worst_function) {
  PLANETMODEL_FIXTURE(path, std::format("flat_surface_{}.nc", label));
  PLANETMODEL_FIXTURE(expect, std::format("flat_surface_{}.expect", label));
  SCOPED_TRACE(label);
  const auto file = flat::ReadSurfaceFile(path);
  EXPECT_EQ(file.schema, flat::SURFACE_SCHEMA);
  EXPECT_EQ(file.source, "planetmodel fixtures");
  EXPECT_FALSE(file.created.empty());
  ASSERT_TRUE(file.grid && file.time && file.instant);
  const auto& grid = *file.grid;
  const auto ny = grid.NY();
  const auto cells = grid.Size();
  EXPECT_EQ(pm::detail::TextOf(file.meta, "epoch").value_or(""), "present");
  EXPECT_EQ(file.citation, std::vector<std::string>{"a work"});
  const auto history = flat::SurfaceHistory(file);

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
    } else if (what == "depth") {
      auto text = std::string{};
      words >> text;
      EXPECT_EQ(file.depth, text == "-" ? std::optional<double>{} : std::optional{std::stod(text)})
          << line;
    } else if (what == "domain") {
      auto kind = std::string{};
      words >> kind;
      EXPECT_EQ(grid.GetDomain().Kind(), kind) << line;
      auto numbers = std::vector<double>{};
      for (auto v = 0.0; words >> v;) numbers.push_back(v);
      auto stated = grid.GetDomain().BoundsList();
      const auto padding = grid.GetDomain().PaddingList();
      stated.insert(stated.end(), padding.begin(), padding.end());
      EXPECT_EQ(stated, numbers) << line;
    } else if (what == "grid") {
      auto shape = std::vector<std::size_t>{};
      for (auto n = std::size_t{0}; words >> n;) shape.push_back(n);
      EXPECT_EQ(grid.Shape(), shape) << line;
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
      const auto k = tag == "-" ? std::size_t{0} : static_cast<std::size_t>(std::stoul(tag));
      const auto index = k * cells + i * ny + j;
      ASSERT_LT(index, values.size()) << line;
      if (what == "absent") {
        EXPECT_TRUE(std::isnan(values[index])) << line;
      } else {
        auto value = 0.0;
        words >> value;
        EXPECT_EQ(values[index], value) << line;
      }
    } else if (what == "sample") {
      auto name = std::string{}, tag = std::string{}, extrap = std::string{};
      auto i = std::size_t{0}, j = std::size_t{0};
      auto value = 0.0;
      words >> name >> tag >> extrap >> i >> j >> value;
      const auto date = tag == "-" ? std::optional<double>{} : std::optional{std::stod(tag)};
      const auto at = history.Values(
          name, date,
          {.extrapolation = extrap == "hold" ? pm::Extrapolation::Hold
                                             : pm::Extrapolation::Refuse});
      ASSERT_LT(i * ny + j, at.size()) << line;
      EXPECT_DOUBLE_EQ(at[i * ny + j], value) << line;
    } else if (what == "function") {
      auto name = std::string{}, tag = std::string{};
      auto x = 0.0, y = 0.0, value = 0.0, dx = 0.0, dy = 0.0;
      words >> name >> tag >> x >> y >> value >> dx >> dy;
      const auto date = tag == "-" ? std::optional<double>{} : std::optional{std::stod(tag)};
      const auto f = history.Function(name, date);
      const auto at = history.Values(name, date);
      auto scale = 0.0;
      for (const auto v : at) scale = std::max(scale, std::abs(v));
      const auto length = grid.GetDomain().Length(0);
      EXPECT_NEAR(f.Value(x, y), value, 1e-12 * scale) << line;
      const auto [gx, gy] = f.Gradient(x, y);
      EXPECT_NEAR(gx, dx, 1e-9 * scale / length) << line;
      EXPECT_NEAR(gy, dy, 1e-9 * scale / length) << line;
      worst_function = std::max(worst_function, std::abs(f.Value(x, y) - value) / scale);
    } else {
      FAIL() << "unknown expectation: " << line;
    }
    ++checked;
  }
  EXPECT_GT(checked, 60);
}

/// The file the writer's test leaves for the Python, from formulaic
/// fields `test_cpp_written_cartesian.py` rebuilds: over a box, or over a
/// rectangle.
flat::SurfaceFile WrittenSurface(bool box = true) {
  auto file = flat::SurfaceFile{};
  file.scales = pm::Scales({.length = 2.0, .mass = 3.0, .time = 5.0});
  file.grid = box ? flat::Grid(flat::Domain::Box({0.0, 4.0}, {-1.0, 2.0}, {0.5, 0.0}), 6, 4)
                  : flat::Grid(flat::Domain::Rectangle({0.0, 4.0}, 0.5), 6);
  file.time = std::vector<double>{0.0, 1.0, 2.5};
  file.instant = std::vector<double>{1.5, 2.5};
  if (box) file.depth = 1.25;
  file.source = "planetmodel cpp tests";
  file.title = "written flat surface";
  file.meta.emplace("epoch", std::string{"present"});
  file.meta.emplace("count", std::vector<long long>{4});
  file.citation = {"one", "two"};
  const auto nx = file.grid->NX();
  const auto ny = file.grid->NY();
  auto on_grid = [&](std::size_t leading, auto value) {
    auto out = std::vector<double>{};
    for (auto k = std::size_t{0}; k < leading; ++k) {
      for (auto i = std::size_t{0}; i < nx; ++i) {
        for (auto j = std::size_t{0}; j < ny; ++j) out.push_back(value(k, i, j));
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

}  // namespace

TEST(CartesianSurfaceFile, ReadsTheFixturesToTheirExpectations) {
  auto worst = 0.0;
  ExpectAsThePython("box", worst);
  ExpectAsThePython("rectangle", worst);
  RecordProperty("worst_relative_function", std::format("{:.3e}", worst));
}

TEST(CartesianSurfaceFile, WritesBackTheFixturesItReads) {
  // a variable read carries netCDF's own attributes (`_FillValue`), which
  // are the writer's to state
  for (const auto* label : {"box", "rectangle"}) {
    SCOPED_TRACE(label);
    PLANETMODEL_FIXTURE(fixture, std::format("flat_surface_{}.nc", label));
    const auto before = flat::ReadSurfaceFile(fixture);
    ASSERT_FALSE(before.variables.empty());
    ASSERT_TRUE(before.variables.begin()->second.attributes.contains("_FillValue"));
    const auto path = pm::testing::WrittenDir() / std::format("flat_surface_back_{}.nc", label);
    flat::WriteSurfaceFile(path, before);
    const auto after = flat::ReadSurfaceFile(path);
    EXPECT_EQ(after.source, before.source);
    EXPECT_EQ(after.created, before.created);
    EXPECT_EQ(after.title, before.title);
    EXPECT_EQ(after.scales, before.scales);
    EXPECT_EQ(after.time, before.time);
    EXPECT_EQ(after.instant, before.instant);
    EXPECT_EQ(after.depth, before.depth);
    EXPECT_EQ(after.citation, before.citation);
    EXPECT_EQ(after.meta, before.meta);
    EXPECT_EQ(*after.grid, *before.grid);
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
}

TEST(CartesianSurfaceFile, EachReaderRefusesTheOthersFilesByTheirSchema) {
  PLANETMODEL_FIXTURE(box, "flat_surface_box.nc");
  try {
    static_cast<void>(pm::ReadSurfaceFile(box));
    FAIL() << "the spherical reader read a flat surface file";
  } catch (const pm::Error& e) {
    EXPECT_EQ(e.Where(), "schema");
    const auto said = std::string{e.what()};
    EXPECT_NE(said.find("planetmodel.surface/2"), std::string::npos) << said;
    EXPECT_NE(said.find("planetmodel::cartesian::ReadSurfaceFile"), std::string::npos) << said;
  }
  PLANETMODEL_FIXTURE(spherical, "surface_basic.nc");
  try {
    static_cast<void>(flat::ReadSurfaceFile(spherical));
    FAIL() << "the flat reader read a spherical surface file";
  } catch (const pm::Error& e) {
    EXPECT_EQ(e.Where(), "schema");
    const auto said = std::string{e.what()};
    EXPECT_NE(said.find("planetmodel::ReadSurfaceFile"), std::string::npos) << said;
  }
  EXPECT_EQ(flat::SURFACE_SCHEMA, "planetmodel.surface/2");
}

TEST(CartesianSurfaceFile, WritesReadsBackAndLeavesTheFilesForThePython) {
  for (const auto box : {true, false}) {
    SCOPED_TRACE(box ? "box" : "rectangle");
    const auto file = WrittenSurface(box);
    const auto path = pm::testing::WrittenDir() /
                      (box ? "flat_surface_written.nc" : "flat_surface_written_rectangle.nc");
    flat::WriteSurfaceFile(path, file);
    const auto read = flat::ReadSurfaceFile(path);
    EXPECT_EQ(read.schema, flat::SURFACE_SCHEMA);
    EXPECT_EQ(read.source, file.source);
    EXPECT_NE(read.created.find('T'), std::string::npos);
    EXPECT_EQ(read.title, file.title);
    EXPECT_EQ(read.scales.Length(), 2.0);
    EXPECT_EQ(read.time, file.time);
    EXPECT_EQ(read.instant, file.instant);
    EXPECT_EQ(read.depth, file.depth);
    EXPECT_EQ(read.citation, file.citation);
    ASSERT_TRUE(read.grid);
    EXPECT_EQ(*read.grid, *file.grid);
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
    const auto& thickness = read.variables.at("ice_thickness");
    EXPECT_EQ(thickness.Text("time_interpolation"), "linear");
    EXPECT_EQ(thickness.dimensions.front(), "time");
    EXPECT_EQ(thickness.dimensions.back(), box ? "y" : "x");
    EXPECT_TRUE(thickness.attributes.contains("units"));
  }
}

TEST(CartesianSurfaceHistory, ServesEverySupportAndTheFieldsAsFunctions) {
  const auto file = WrittenSurface();
  const auto history = flat::SurfaceHistory(file);
  const auto size = file.grid->Size();
  const auto& nodes = file.variables.at("ice_thickness").values;
  const auto mid = history.Values("ice_thickness", 0.5);
  for (auto n = std::size_t{0}; n < size; ++n) {
    EXPECT_DOUBLE_EQ(mid[n], 0.5 * nodes[n] + 0.5 * nodes[size + n]);
  }
  const auto hold = pm::DateOptions{.extrapolation = pm::Extrapolation::Hold};
  const auto held = history.Values("ice_thickness", 9.0, hold);
  for (auto n = std::size_t{0}; n < size; ++n) EXPECT_EQ(held[n], nodes[2 * size + n]);
  const auto& rate = file.variables.at("rate").values;
  EXPECT_EQ(history.Values("rate", 2.5).front(), rate[size]);
  EXPECT_EQ(history.Values("instantaneous", 1.5).front(),
            file.variables.at("instantaneous").values.front());
  EXPECT_THROW(static_cast<void>(history.Values("instantaneous", 1.6, hold)), pm::Error);
  EXPECT_TRUE(std::isnan(history.Values("topography", 7.0)[1]));
  EXPECT_THROW(static_cast<void>(history.Values("ice_thickness")), pm::Error);
  EXPECT_THROW(static_cast<void>(history.Values("ice_thickness", 9.0)), pm::Error);
  EXPECT_THROW(static_cast<void>(history.Values("missing", 0.0)), pm::Error);
  EXPECT_EQ(history.Names().size(), 4u);
  EXPECT_EQ(history.Support("rate"), pm::TimeSupport::Interval);
  EXPECT_EQ(history.GetGrid(), *file.grid);
  // a field at a date as a function of (x, y): its values at the nodes,
  // periodic over the extent
  const auto f = history.Function("ice_thickness", 0.5);
  const auto& grid = history.GetGrid();
  for (auto n = std::size_t{0}; n < size; ++n) {
    const auto [x, y] = grid.Node(n);
    EXPECT_NEAR(f.Value(x, y), mid[n], 1e-12 * 100.0);
    EXPECT_NEAR(f.Value(x + 5.0, y - 3.0), mid[n], 1e-12 * 100.0);
  }
  // held beyond the span, as the values are: the last node's
  const auto late = history.Function("ice_thickness", 9.0, hold);
  const auto [x0, y0] = grid.Node(0);
  EXPECT_NEAR(late.Value(x0, y0), held[0], 1e-12 * 200.0);
  EXPECT_THROW(static_cast<void>(history.Function("ice_thickness", 9.0)), pm::Error);
  // a field with a point it does not cover is no function
  EXPECT_THROW(static_cast<void>(history.Function("topography")), pm::Error);
  auto fourier = file;
  fourier.variables.at("rate").attributes["lateral_interpolation"] = std::string{"fourier"};
  EXPECT_NO_THROW(flat::ValidateSurfaceFile(fourier, "a file"));
  const auto series = flat::SurfaceHistory(fourier);
  EXPECT_NO_THROW(static_cast<void>(series.Values("rate", 0.5)));
  EXPECT_THROW(static_cast<void>(series.Function("rate", 0.5)), pm::Error);
}

TEST(CartesianSurfaceFile, RefusesWhatThePythonRefusesAtItsPlace) {
  PLANETMODEL_FIXTURE(index, "flat_surface_cases/bad.txt");
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
      static_cast<void>(flat::ReadSurfaceFile(path));
      ADD_FAILURE() << label << " was accepted";
    } catch (const pm::Error& e) {
      EXPECT_EQ(e.Where(), where) << label << ": " << e.what();
      EXPECT_NE(std::string{e.what()}.find(phrase), std::string::npos)
          << label << ": " << e.what();
    }
    ++cases;
  }
  EXPECT_GT(cases, 10);
}

TEST(CartesianSurfaceFile, RefusesWhatTheFormatCannotHold) {
  const auto dir = pm::testing::WrittenDir();
  const auto good = WrittenSurface();
  const auto refused = dir / "refused_flat_surface.nc";
  auto expect_refusal = [&](auto edit, std::string_view what) {
    auto file = good;
    edit(file);
    EXPECT_THROW(flat::WriteSurfaceFile(refused, file), pm::Error) << what;
    EXPECT_FALSE(std::filesystem::exists(refused)) << what;
  };
  expect_refusal([](flat::SurfaceFile& f) { f.source = "  "; }, "a blank source");
  expect_refusal([](flat::SurfaceFile& f) { f.depth = -1.0; }, "a negative depth");
  expect_refusal([](flat::SurfaceFile& f) { f.schema = std::string{pm::SURFACE_SCHEMA}; },
                 "the spherical schema");
  expect_refusal([](flat::SurfaceFile& f) { f.time = std::vector<double>{1.0, 1.0, 2.0}; },
                 "a time axis that does not increase");
  // the file's own names, and those of a spherical file's variables, which
  // a reader refuses in a flat file
  for (const auto* name : {"x", "y", "time", "instant", "radius", "latitude", "longitude",
                           "colatitude_weights", "interface_radius", "boundary_radius"}) {
    expect_refusal(
        [name](flat::SurfaceFile& f) {
          f.variables.emplace(name, pm::SurfaceField(name, f.variables.at("topography").values));
        },
        std::format("the reserved name {}", name));
  }
  expect_refusal(
      [](flat::SurfaceFile& f) {
        f.variables.emplace("topography_poles",
                            pm::FileVariable{.name = "topography_poles", .values = {1.5, -0.5}});
      },
      "polar values");
  expect_refusal(
      [](flat::SurfaceFile& f) {
        f.variables.at("rate").attributes["time_interpolation"] = std::string{"linear"};
      },
      "a time rule on an interval field");
  expect_refusal(
      [](flat::SurfaceFile& f) {
        f.variables.at("topography").attributes["lateral_interpolation"] =
            std::string{"harmonic"};
      },
      "the sphere's lateral rule");
  expect_refusal([](flat::SurfaceFile& f) { f.variables.at("rate").values.pop_back(); },
                 "too few values");
  expect_refusal([](flat::SurfaceFile& f) { f.time.reset(); },
                 "time-supported fields without a time axis");
}

TEST(CartesianSurfaceFile, RefusesTheCubicRuleOnAGridTooSmallForIt) {
  // three nodes along every axis, on writing as on reading
  const auto domain = flat::Domain::Box({0.0, 4.0}, {-1.0, 2.0}, {0.5, 0.0});
  auto on = [&domain](std::size_t nx, std::size_t ny, pm::LateralRule rule) {
    auto file = flat::SurfaceFile{};
    file.grid = flat::Grid(domain, nx, ny);
    file.source = "planetmodel cpp tests";
    file.variables.emplace("h", pm::SurfaceField("h", std::vector<double>(nx * ny, 1.0),
                                                 {.lateral_interpolation = rule}));
    return file;
  };
  const auto refused = pm::testing::WrittenDir() / "refused_flat_surface_rule.nc";
  const auto thin = on(4, 2, pm::LateralRule::Cubic);
  try {
    flat::ValidateSurfaceFile(thin, "a built file");
    ADD_FAILURE() << "the cubic rule was accepted on two nodes along y";
  } catch (const pm::Error& e) {
    EXPECT_EQ(e.Source(), "a built file");
    EXPECT_EQ(e.Where(), "h");
    EXPECT_NE(std::string{e.what()}.find("states the lateral rule cubic"), std::string::npos)
        << e.what();
    EXPECT_NE(std::string{e.what()}.find("this grid has 4 by 2"), std::string::npos) << e.what();
  }
  EXPECT_THROW(flat::WriteSurfaceFile(refused, thin), pm::Error);
  EXPECT_FALSE(std::filesystem::exists(refused));
  EXPECT_NO_THROW(flat::ValidateSurfaceFile(on(4, 2, pm::LateralRule::Linear), "a built file"));
  EXPECT_NO_THROW(flat::ValidateSurfaceFile(on(4, 3, pm::LateralRule::Cubic), "a built file"));
}

TEST(CartesianSurfaceFile, StampsThisLibraryAsItsSourceUnlessGivenOne) {
  // as the model file's writer does
  auto file = WrittenSurface();
  file.source = "";
  const auto path = pm::testing::WrittenDir() / "flat_surface_stamped.nc";
  flat::WriteSurfaceFile(path, file);
  EXPECT_EQ(flat::ReadSurfaceFile(path).source, "planetmodel C++ " + std::string{pm::Version()});
  EXPECT_THROW(flat::ValidateSurfaceFile(file, "a built file"), pm::Error);
}

TEST(CartesianSurfaceFile, WritesACitationByTheRuleOfAModels) {
  // each work trimmed of white space at its ends, an empty one or a repeat
  // dropped, and a work of more than one line refused
  auto file = WrittenSurface();
  file.citation = {" A. ", "", "B.", "A.", "\tC.\n"};
  const auto path = pm::testing::WrittenDir() / "flat_surface_cited.nc";
  flat::WriteSurfaceFile(path, file);
  EXPECT_EQ(flat::ReadSurfaceFile(path).citation, (std::vector<std::string>{"A.", "B.", "C."}));
  file.citation = {"A.\nB."};
  const auto refused = pm::testing::WrittenDir() / "refused_flat_surface_citation.nc";
  try {
    flat::WriteSurfaceFile(refused, file);
    ADD_FAILURE() << "a work of two lines was written";
  } catch (const pm::Error& e) {
    EXPECT_EQ(e.Where(), "citation");
    EXPECT_NE(std::string{e.what()}.find("more than one line"), std::string::npos) << e.what();
  }
  EXPECT_FALSE(std::filesystem::exists(refused));
}

#if defined(__unix__) || defined(__APPLE__)
TEST(CartesianSurfaceFile, GivesTheFileThePermissionsAPlainOpenWould) {
  const auto file = WrittenSurface();
  const auto dir = pm::testing::WrittenDir() / "permissions";
  std::filesystem::create_directories(dir);
  pm::testing::ExpectPlainPermissions(dir / "flat_surface.nc",
                                      [&file](const std::filesystem::path& p) {
                                        flat::WriteSurfaceFile(p, file);
                                      });
}
#endif
