// The flat model file: every file the Python writes read as the Python
// reads it (its metadata and domain, every variable on every layer in both
// frames, the scalars' gradients, the vertical stretch, the fields on
// interfaces with their gradients), over a box and over a rectangle;
// which variables are lateral; the fourier rule refused with the numbers
// still readable; each reader refusing the other's files by their schema;
// and every malformed copy refused at its place.
#include "planetmodel/cartesian/modelfile.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/cartesian/manifest.hpp"
#include "planetmodel/modelfile.hpp"

namespace {

namespace flat = planetmodel::cartesian;
using planetmodel::Error;
using flat::Frame;

double ToDouble(const std::string& text) {
  auto value = 0.0;
  std::from_chars(text.data(), text.data() + text.size(), value);
  return value;
}

std::string Unhex(const std::string& hex) {
  auto out = std::string{};
  for (auto i = std::size_t{0}; i + 1 < hex.size(); i += 2) {
    out += static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16));
  }
  return out;
}

std::vector<std::filesystem::path> Expectations() {
  auto out = std::vector<std::filesystem::path>{};
  const auto dir = planetmodel::testing::FixturesDir();
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    const auto name = entry.path().filename().string();
    if (name.starts_with("flat_model_") && entry.path().extension() == ".expect") {
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

/// The worst differences from the Python, relative to each tolerance's
/// scale, by kind of evaluation.
struct Agreement {
  double value = 0.0;
  double gradient = 0.0;
  double displacement = 0.0;
  double surface = 0.0;
  double surface_gradient = 0.0;
};

/// The flat model file against the Python's reading of it, line by line.
void ExpectAsThePython(const std::filesystem::path& expect, Agreement& worst) {
  const auto model = expect.parent_path() / (expect.stem().string() + ".nc");
  SCOPED_TRACE(model.filename().string());
  const auto file = flat::ReadModelFile(model);
  EXPECT_EQ(file.Schema(), flat::MODEL_SCHEMA);
  auto fields = std::map<std::pair<int, std::string>, flat::Field>{};
  auto field = [&](int layer, const std::string& name) -> const flat::Field& {
    const auto key = std::pair{layer, name};
    if (!fields.contains(key)) fields.emplace(key, file.FieldOn(layer, name));
    return fields.at(key);
  };
  auto surfaces = std::map<std::pair<int, std::string>, flat::LateralInterpolant>{};
  auto surface = [&](int interface, const std::string& name) -> const flat::LateralInterpolant& {
    const auto key = std::pair{interface, name};
    if (!surfaces.contains(key)) surfaces.emplace(key, file.SurfaceFieldOn(interface, name));
    return surfaces.at(key);
  };
  const auto mapping = file.Mapping();
  auto in = std::ifstream(expect);
  auto lines = std::vector<std::string>{};
  for (auto line = std::string{}; std::getline(in, line);) lines.push_back(line);
  // each variable's largest value, the scale of its tolerance
  auto largest = std::map<std::string, double>{};
  for (const auto& line : lines) {
    auto words = std::istringstream{line};
    auto key = std::string{};
    auto layer = std::string{};
    auto name = std::string{};
    auto tag = std::string{};
    if (!(words >> key) || key != "value") continue;
    words >> layer >> name >> tag;
    auto skip = 0.0;
    words >> skip >> skip >> skip;
    for (auto v = 0.0; words >> v;) largest[name] = std::max(largest[name], std::abs(v));
  }
  const auto base = file.GetSkeleton().Boundaries().back();
  const auto& domain = file.GetDomain();
  const auto lateral = std::max(domain.Length(0), domain.IsBox() ? domain.Length(1) : 0.0);
  auto checked = 0;
  for (const auto& line : lines) {
    auto words = std::istringstream{line};
    auto key = std::string{};
    words >> key;
    auto rest = std::vector<std::string>{};
    for (auto w = std::string{}; words >> w;) rest.push_back(w);
    if (key == "scales") {
      const auto& s = file.GetScales();
      EXPECT_EQ(s.Length(), ToDouble(rest[0]));
      EXPECT_EQ(s.Mass(), ToDouble(rest[1]));
      EXPECT_EQ(s.Time(), ToDouble(rest[2]));
      EXPECT_EQ(s.Temperature(), ToDouble(rest[3]));
    } else if (key == "rtol") {
      EXPECT_EQ(file.Rtol(), ToDouble(rest[0]));
    } else if (key == "citation") {
      auto joined = std::string{};
      for (const auto& work : file.Citation()) joined += (joined.empty() ? "" : "\n") + work;
      EXPECT_EQ(joined.empty() ? "-" : joined, rest[0] == "-" ? "-" : Unhex(rest[0]));
    } else if (key == "name") {
      EXPECT_EQ(file.Name().value_or("-"), rest[0] == "-" ? "-" : Unhex(rest[0]));
    } else if (key == "domain") {
      EXPECT_EQ(domain.Kind(), rest[0]) << line;
      auto numbers = std::vector<double>{};
      for (auto i = std::size_t{1}; i < rest.size(); ++i) numbers.push_back(ToDouble(rest[i]));
      auto stated = domain.BoundsList();
      const auto padding = domain.PaddingList();
      stated.insert(stated.end(), padding.begin(), padding.end());
      EXPECT_EQ(stated, numbers) << line;
    } else if (key == "layer_name" || key == "interface_name") {
      const auto& list = key == "layer_name" ? file.LayerNames() : file.InterfaceNames();
      const auto i = static_cast<std::size_t>(std::stoi(rest[0]));
      EXPECT_EQ(list.at(i).value_or("-"), rest[1] == "-" ? "-" : Unhex(rest[1])) << line;
    } else if (key == "constant") {
      const auto& c = file.Constants().at(rest[0]);
      EXPECT_EQ(c.value, ToDouble(rest[1])) << line;
      EXPECT_EQ(c.dimensions.has_value(), rest.size() == 6) << line;
    } else if (key == "absent") {
      EXPECT_FALSE(file.HasField(std::stoi(rest[0]), rest[1])) << line;
      EXPECT_THROW(static_cast<void>(file.FieldOn(std::stoi(rest[0]), rest[1])), Error) << line;
    } else if (key == "value") {
      const auto& f = field(std::stoi(rest[0]), rest[1]);
      const auto frame = rest[2] == "l" ? Frame::Local : Frame::Cartesian;
      const auto depth = ToDouble(rest[3]);
      const auto x = ToDouble(rest[4]);
      const auto y = ToDouble(rest[5]);
      auto want = std::vector<double>{};
      for (auto i = std::size_t{6}; i < rest.size(); ++i) want.push_back(ToDouble(rest[i]));
      ASSERT_EQ(want.size(), f.Size()) << line;
      const auto got = f.Evaluate(depth, x, y, frame);
      ASSERT_EQ(got.size(), want.size()) << line;
      // the same point as (x, y, z), z up
      const auto at = f.EvaluateAt({x, y, -depth}, frame);
      EXPECT_EQ(at, got) << line;
      const auto scale = std::max(Largest(want), largest[rest[1]]) + 1e-300;
      for (auto c = std::size_t{0}; c < want.size(); ++c) {
        EXPECT_NEAR(got[c], want[c], 1e-11 * scale) << line << " [" << c << "]";
        worst.value = std::max(worst.value, std::abs(got[c] - want[c]) / scale);
      }
      ++checked;
    } else if (key == "gradient") {
      const auto scalar = field(std::stoi(rest[0]), rest[1]).AsScalar();
      const auto depth = ToDouble(rest[2]);
      const auto x = ToDouble(rest[3]);
      const auto y = ToDouble(rest[4]);
      const auto want =
          std::vector<double>{ToDouble(rest[5]), ToDouble(rest[6]), ToDouble(rest[7])};
      const auto got = scalar.Gradient(depth, x, y);
      const auto [lo, hi] = scalar.Underlying().Interval();
      // each derivative against the field's size over the length it varies on
      const auto size = largest[rest[1]] + 1e-300;
      const auto lengths = std::array{hi - lo, lateral, lateral};
      for (auto d = std::size_t{0}; d < 3; ++d) {
        const auto scale = size / lengths[d] + Largest(want);
        EXPECT_NEAR(got[d], want[d], 1e-9 * scale) << line << " [" << d << "]";
        worst.gradient = std::max(worst.gradient, std::abs(got[d] - want[d]) / scale);
      }
      ++checked;
    } else if (key == "displacement") {
      const auto X = planetmodel::Vector3{ToDouble(rest[0]), ToDouble(rest[1]), ToDouble(rest[2])};
      const auto want = planetmodel::Vector3{ToDouble(rest[3]), ToDouble(rest[4]),
                                             ToDouble(rest[5])};
      const auto got = mapping.Displacement(X);
      for (auto i = std::size_t{0}; i < 3; ++i) {
        EXPECT_NEAR(got[i], want[i], 1e-12 * base) << line;
        worst.displacement = std::max(worst.displacement, std::abs(got[i] - want[i]) / base);
      }
      const auto moved = mapping(X);
      for (auto i = std::size_t{0}; i < 3; ++i) {
        EXPECT_NEAR(moved[i], X[i] + want[i], 1e-12 * (base + std::abs(X[i]))) << line;
      }
      EXPECT_EQ(mapping.Height(-X[2], X[0], X[1]), got[2]) << line;
      ++checked;
    } else if (key == "surface") {
      const auto& f = surface(std::stoi(rest[0]), rest[1]);
      const auto want = ToDouble(rest[4]);
      const auto got = f.Value(ToDouble(rest[2]), ToDouble(rest[3]));
      EXPECT_NEAR(got, want, 1e-12 * (std::abs(want) + 1.0)) << line;
      worst.surface = std::max(worst.surface, std::abs(got - want) / (std::abs(want) + 1.0));
      ++checked;
    } else if (key == "surface_gradient") {
      const auto& f = surface(std::stoi(rest[0]), rest[1]);
      const auto got = f.Gradient(ToDouble(rest[2]), ToDouble(rest[3]));
      const auto want = std::vector<double>{ToDouble(rest[4]), ToDouble(rest[5])};
      const auto scale = Largest(want) + 1.0 / lateral;
      for (auto d = std::size_t{0}; d < 2; ++d) {
        EXPECT_NEAR(got[d], want[d], 1e-9 * scale) << line;
        worst.surface_gradient =
            std::max(worst.surface_gradient, std::abs(got[d] - want[d]) / scale);
      }
      ++checked;
    } else if (key == "surface_absent") {
      EXPECT_FALSE(file.HasSurfaceField(std::stoi(rest[0]), rest[1])) << line;
      EXPECT_THROW(static_cast<void>(file.SurfaceFieldOn(std::stoi(rest[0]), rest[1])), Error)
          << line;
    }
  }
  EXPECT_GT(checked, 10);
}

}  // namespace

TEST(CartesianModelFile, ReadsEveryFlatModelFileAsThePythonDoes) {
  PLANETMODEL_FIXTURE(probe, "flat_model_layered.expect");
  const auto expectations = Expectations();
  EXPECT_GE(expectations.size(), 10u);
  auto worst = Agreement{};
  for (const auto& expect : expectations) ExpectAsThePython(expect, worst);
  RecordProperty("worst_relative_value", std::format("{:.3e}", worst.value));
  RecordProperty("worst_relative_gradient", std::format("{:.3e}", worst.gradient));
  RecordProperty("worst_relative_displacement", std::format("{:.3e}", worst.displacement));
  RecordProperty("worst_relative_surface", std::format("{:.3e}", worst.surface));
  RecordProperty("worst_relative_surface_gradient",
                 std::format("{:.3e}", worst.surface_gradient));
}

TEST(CartesianModelFile, EachReaderRefusesTheOthersFilesByTheirSchema) {
  // the flat file by the spherical reader, naming the flat one
  PLANETMODEL_FIXTURE(layered, "flat_model_layered.nc");
  try {
    static_cast<void>(planetmodel::ReadModelFile(layered));
    FAIL() << "the spherical reader read a flat model file";
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "schema");
    const auto said = std::string{e.what()};
    EXPECT_NE(said.find("planetmodel.model/3"), std::string::npos) << said;
    EXPECT_NE(said.find("planetmodel::cartesian::ReadModelFile"), std::string::npos) << said;
  }
  // the spherical files, of either schema, by the flat reader, naming theirs
  for (const auto* name : {"model_prem.nc", "model_log.nc"}) {
    PLANETMODEL_FIXTURE(spherical, name);
    try {
      static_cast<void>(flat::ReadModelFile(spherical));
      FAIL() << "the flat reader read " << name;
    } catch (const Error& e) {
      EXPECT_EQ(e.Where(), "schema");
      const auto said = std::string{e.what()};
      EXPECT_NE(said.find("planetmodel::ReadModelFile"), std::string::npos) << said;
      EXPECT_NE(said.find("planetmodel.model/3"), std::string::npos) << said;
    }
  }
  EXPECT_EQ(flat::MODEL_SCHEMA, planetmodel::MODEL_SCHEMA_FLAT);
  EXPECT_EQ(flat::MODEL_SCHEMA, "planetmodel.model/3");
}

TEST(CartesianModelFile, HoldsTheDomainTheSkeletonAndTheGrids) {
  PLANETMODEL_FIXTURE(path, "flat_model_lateral.nc");
  const auto file = flat::ReadModelFile(path);
  EXPECT_TRUE(file.GetDomain().IsBox());
  EXPECT_EQ(file.GetDomain(), flat::Domain::Box({0.0, 400e3}, {0.0, 300e3}, {50e3, 0.0}));
  // depths from the surface down, an interface the top of each layer
  EXPECT_EQ(file.GetSkeleton().Boundaries(), (std::vector{0.0, 30e3, 100e3, 200e3}));
  EXPECT_EQ(file.InterfaceDepths(), (std::vector{0.0, 30e3, 100e3}));
  EXPECT_EQ(file.LayerNames().front(), "crust");
  EXPECT_EQ(file.InterfaceNames().at(1), "moho");
  EXPECT_FALSE(file.InterfaceNames().at(2));
  ASSERT_EQ(file.Parts().size(), 2u);
  EXPECT_EQ(file.Parts()[0].layers, (std::vector{0, 1}));
  EXPECT_EQ(file.Parts()[0].grid->Shape(), (std::vector<std::size_t>{16, 12}));
  EXPECT_EQ(file.PartOf(2).grid->Shape(), (std::vector<std::size_t>{10, 6}));
  EXPECT_EQ(file.PartOf(2).depth.front(), 100e3);
  EXPECT_EQ(file.PartOf(2).depth.back(), 200e3);
  EXPECT_FALSE(file.Geometry());
  EXPECT_FALSE(file.Surfaces());
  // a rectangle's file states the one axis
  PLANETMODEL_FIXTURE(cut, "flat_model_rectangle.nc");
  const auto rectangle = flat::ReadModelFile(cut);
  EXPECT_FALSE(rectangle.GetDomain().IsBox());
  EXPECT_EQ(rectangle.GetDomain(), flat::Domain::Rectangle({0.0, 400e3}, 25e3));
  EXPECT_EQ(rectangle.Parts().front().grid->Shape(), (std::vector<std::size_t>{14}));
  const auto& rho = rectangle.PartOf(1).variables.at("rho");
  EXPECT_EQ(rho.dimensions, (std::vector<std::string>{"node", "x"}));
  // y is never read
  const auto field = rectangle.FieldOn(1, "rho").AsScalar();
  EXPECT_EQ(field.Value(50e3, 123e3, 0.0), field.Value(50e3, 123e3, 7e8));
  EXPECT_EQ(field.Gradient(50e3, 123e3, 0.0)[2], 0.0);
}

TEST(CartesianModelFile, AVariableOnTheGridIsLateral) {
  // a flat variable is lateral by its grid's dimensions after the node, x
  // and y over a box and x over a rectangle, as a spherical one is by its
  // latitude and longitude; its components' axes follow either way
  const auto of = [](std::vector<std::string> dimensions) {
    auto var = planetmodel::FileVariable{};
    var.dimensions = std::move(dimensions);
    return var;
  };
  EXPECT_TRUE(of({"node", "x", "y"}).IsLateral());
  EXPECT_TRUE(of({"node", "x", "y", "component"}).IsLateral());
  EXPECT_TRUE(of({"node", "x"}).IsLateral());
  EXPECT_TRUE(of({"node", "x", "voigt_row", "voigt_column"}).IsLateral());
  EXPECT_FALSE(of({"node"}).IsLateral());
  EXPECT_FALSE(of({"node", "component"}).IsLateral());
  EXPECT_FALSE(of({"node", "voigt"}).IsLateral());
  EXPECT_TRUE(of({"node", "latitude", "longitude"}).IsLateral());
  EXPECT_FALSE(of({"node", "latitude"}).IsLateral());

  // over a box: gridded fields of ranks 0 to 2, and fields of depth alone
  PLANETMODEL_FIXTURE(lateral, "flat_model_lateral.nc");
  const auto box = flat::ReadModelFile(lateral);
  EXPECT_TRUE(box.PartOf(1).variables.at("rho").IsLateral());
  const auto& flow = box.PartOf(1).variables.at("flow");
  EXPECT_EQ(flow.dimensions, (std::vector<std::string>{"node", "x", "y", "component"}));
  EXPECT_TRUE(flow.IsLateral());
  EXPECT_TRUE(box.PartOf(2).variables.at("stress").IsLateral());
  EXPECT_FALSE(box.PartOf(1).variables.at("stress").IsLateral());
  EXPECT_FALSE(box.PartOf(1).variables.at("vp").IsLateral());
  PLANETMODEL_FIXTURE(layered, "flat_model_layered.nc");
  const auto plain = flat::ReadModelFile(layered);
  for (const auto& part : plain.Parts()) {
    for (const auto& [name, var] : part.variables) {
      SCOPED_TRACE(name);
      EXPECT_FALSE(var.IsLateral());
    }
  }
  // over a rectangle: a gridded scalar and a gridded vector on the one
  // axis, and a field of depth alone
  PLANETMODEL_FIXTURE(cut, "flat_model_rectangle.nc");
  const auto rectangle = flat::ReadModelFile(cut);
  EXPECT_TRUE(rectangle.PartOf(1).variables.at("rho").IsLateral());
  const auto& drift = rectangle.PartOf(2).variables.at("flow");
  EXPECT_EQ(drift.dimensions, (std::vector<std::string>{"node", "x", "component"}));
  EXPECT_TRUE(drift.IsLateral());
  EXPECT_FALSE(rectangle.PartOf(1).variables.at("vp").IsLateral());
}

TEST(CartesianModelFile, HoldsTheFormsOfTheVerticalStretch) {
  PLANETMODEL_FIXTURE(layered, "flat_model_layered.nc");
  const auto identity = flat::ReadModelFile(layered).Mapping();
  EXPECT_EQ(identity.GetForm(), flat::VerticalStretch::Form::Identity);
  EXPECT_EQ(identity.Displacement({1e5, 2e5, -3e4}), (planetmodel::Vector3{0.0, 0.0, 0.0}));
  // outside the body in depth is refused; laterally the body is periodic
  EXPECT_THROW(static_cast<void>(identity.Displacement({0.0, 0.0, 1.0})), Error);
  EXPECT_THROW(static_cast<void>(identity.Displacement({0.0, 0.0, -201e3})), Error);
  EXPECT_NO_THROW(static_cast<void>(identity.Displacement({9e9, -9e9, -1e3})));
  PLANETMODEL_FIXTURE(stacked, "flat_model_stacked.nc");
  const auto file = flat::ReadModelFile(stacked);
  const auto stack = file.Mapping();
  EXPECT_EQ(stack.GetForm(), flat::VerticalStretch::Form::Stack);
  ASSERT_TRUE(file.Geometry());
  EXPECT_EQ(file.Geometry()->depths, (std::vector{0.0, 30e3, 100e3, 200e3}));
  EXPECT_EQ(file.Geometry()->between, "linear");
  // a relief is a height, positive up: the surface goes to its topography,
  // and the base, which has none, is held
  const auto top = stack.Height(0.0, 120e3, 80e3);
  EXPECT_NE(top, 0.0);
  EXPECT_EQ(stack({120e3, 80e3, 0.0}), (planetmodel::Vector3{120e3, 80e3, top}));
  EXPECT_EQ(stack.Height(200e3, 120e3, 80e3), 0.0);
  EXPECT_EQ(stack.Height(150e3, 120e3, 80e3), 0.0);
  // linear in depth between the surfaces
  const auto moho = stack.Height(30e3, 120e3, 80e3);
  EXPECT_NEAR(stack.Height(15e3, 120e3, 80e3), 0.5 * (top + moho), 1e-12 * std::abs(top));
  PLANETMODEL_FIXTURE(stretched, "flat_model_stretched.nc");
  EXPECT_EQ(flat::ReadModelFile(stretched).Mapping().GetForm(),
            flat::VerticalStretch::Form::Field);
}

TEST(CartesianModelFile, ReadsAPerturbedFieldAsItsKindAndAHeldPadding) {
  PLANETMODEL_FIXTURE(path, "flat_model_lateral.nc");
  const auto file = flat::ReadModelFile(path);
  EXPECT_EQ(file.FieldOn(2, "vs").PerturbationKind(), "relative");
  EXPECT_FALSE(file.FieldOn(2, "vs").IsRadial());
  EXPECT_FALSE(file.FieldOn(1, "vs").PerturbationKind());
  EXPECT_TRUE(file.FieldOn(1, "vs").IsRadial());
  EXPECT_EQ(file.FieldsOf("flow").size(), 1u);
  EXPECT_EQ(file.FieldsOf("stress").size(), 2u);
  EXPECT_THROW(static_cast<void>(file.FieldOn(2, "vs_perturbation")), Error);
  EXPECT_THROW(static_cast<void>(file.FieldOn(1, "nonesuch")), Error);
  EXPECT_THROW(static_cast<void>(file.FieldOn(1, "stress")), Error);
  EXPECT_THROW(static_cast<void>(file.FieldOn(0, "stress").AsScalar()), Error);
  // below the base the lowest properties are held: constant in depth,
  // whatever they are laterally
  PLANETMODEL_FIXTURE(padded, "flat_model_held.nc");
  const auto held = flat::ReadModelFile(padded);
  EXPECT_EQ(held.GetSkeleton().Boundaries(),
            (std::vector{0.0, 30e3, 100e3, 200e3, 260e3, 400e3}));
  EXPECT_EQ(held.LayerNames().back(), "deep_padding");
  for (const auto* name : {"rho", "vs", "stress"}) {
    const auto above = held.FieldOn(2, name);
    for (const auto layer : {3, 4}) {
      const auto below = held.FieldOn(layer, name);
      const auto [lo, hi] = below.Interval();
      const auto want = above.Evaluate(200e3, 130e3, 70e3);
      for (const auto depth : {lo, 0.5 * (lo + hi), hi}) {
        const auto got = below.Evaluate(depth, 130e3, 70e3);
        for (auto c = std::size_t{0}; c < want.size(); ++c) {
          EXPECT_NEAR(got[c], want[c], 1e-12 * (std::abs(want[c]) + 1.0)) << name;
        }
      }
      const auto [dd, dx, dy] = below.Derivatives(0.5 * (lo + hi), 130e3, 70e3);
      for (const auto v : dd) EXPECT_NEAR(v, 0.0, 1e-12 * Largest(want) / (hi - lo)) << name;
    }
  }
}

TEST(CartesianModelFile, ReadsAViscosityInItsLogarithmExactly) {
  PLANETMODEL_FIXTURE(path, "flat_model_log.nc");
  const auto file = flat::ReadModelFile(path);
  EXPECT_EQ(file.PartOf(2).variables.at("viscosity").Text("radial_interpolation"),
            "log-polynomial");
  EXPECT_FALSE(file.HasField(0, "viscosity"));
  // three decades down across the layer, geometric in depth
  const auto graded = file.FieldOn(2, "viscosity").AsScalar();
  for (const auto t : {0.0, 0.25, 0.5, 0.9, 1.0}) {
    const auto want = 1e21 * std::pow(10.0, -3.0 * t);
    EXPECT_NEAR(graded.Value(100e3 + t * 100e3, 1e3, 2e3), want, 1e-12 * want) << t;
    const auto slope = -3.0 * std::log(10.0) / 100e3 * want;
    EXPECT_NEAR(graded.Gradient(100e3 + t * 100e3, 1e3, 2e3)[0], slope,
                1e-11 * std::abs(slope));
  }
  const auto constant = file.FieldOn(1, "viscosity").AsScalar();
  for (const auto depth : {30e3, 51e3, 100e3}) EXPECT_EQ(constant.Value(depth, 0.0, 0.0), 1e23);
}

TEST(CartesianModelFile, TheDerivativesAreTheDifferencesOfTheValues) {
  PLANETMODEL_FIXTURE(path, "flat_model_lateral.nc");
  const auto file = flat::ReadModelFile(path);
  struct Probe {
    int layer;
    const char* name;
    double depth, x, y;
  };
  for (const auto& [layer, name, depth, x, y] :
       {Probe{0, "stress", 11e3, 101e3, 33e3}, Probe{2, "stress", 151e3, 301e3, 211e3},
        Probe{1, "rho", 47e3, 77e3, 123e3}, Probe{2, "vs", 133e3, 20e3, 250e3}}) {
    SCOPED_TRACE(std::format("{} on layer {}", name, layer));
    const auto f = file.FieldOn(layer, name);
    const auto n = f.Size();
    for (const auto frame : {Frame::Local, Frame::Cartesian}) {
      const auto [dd, dx, dy] = f.Derivatives(depth, x, y, frame);
      ASSERT_EQ(dd.size(), n);
      auto at = [&](double a, double b, double c) { return f.Evaluate(a, b, c, frame); };
      constexpr auto h = 1.0;
      const auto steps = std::array<std::array<double, 3>, 3>{{{h, 0, 0}, {0, h, 0}, {0, 0, h}}};
      const auto got = std::array<const std::vector<double>*, 3>{&dd, &dx, &dy};
      const auto scale = Largest(at(depth, x, y)) / 1e4 + Largest(dd) + Largest(dx) + Largest(dy);
      for (auto d = std::size_t{0}; d < 3; ++d) {
        const auto& s = steps[d];
        const auto plus = at(depth + s[0], x + s[1], y + s[2]);
        const auto minus = at(depth - s[0], x - s[1], y - s[2]);
        for (auto c = std::size_t{0}; c < n; ++c) {
          EXPECT_NEAR((*got[d])[c], (plus[c] - minus[c]) / (2 * h), 1e-6 * scale)
              << "d" << d << " [" << c << "]";
        }
      }
    }
  }
}

TEST(CartesianModelFile, HoldsFieldsOnInterfacesBesideTheLayers) {
  PLANETMODEL_FIXTURE(path, "flat_model_surfaces.nc");
  const auto file = flat::ReadModelFile(path);
  ASSERT_TRUE(file.Surfaces());
  EXPECT_EQ(file.Surfaces()->interfaces, (std::vector<int>{0, 1}));
  EXPECT_EQ(file.SurfaceNames(0), (std::vector<std::string>{"K_shape"}));
  EXPECT_EQ(file.SurfaceNames(1), (std::vector<std::string>{"K_shape", "shift"}));
  EXPECT_TRUE(file.SurfaceNames(2).empty());
  const auto& kernel = file.Surfaces()->variables.at("K_shape");
  EXPECT_EQ(kernel.dimensions, (std::vector<std::string>{"surface", "x", "y"}));
  EXPECT_EQ(kernel.Numbers("dimensions"), (std::vector<double>{0.0, -2.0, 0.0, 0.0}));
  EXPECT_EQ(file.Surfaces()->variables.at("shift").Text("lateral_interpolation"), "cubic");
  EXPECT_FALSE(file.HasField(1, "K_shape"));
  EXPECT_FALSE(file.HasSurfaceField(1, "rho"));
  EXPECT_THROW(static_cast<void>(file.SurfaceFieldOn(0, "shift")), Error);
  EXPECT_THROW(static_cast<void>(file.SurfaceFieldOn(3, "K_shape")), Error);
}

TEST(CartesianModelFile, RefusesTheFourierRuleAndKeepsItsNumbers) {
  PLANETMODEL_FIXTURE(path, "flat_model_fourier.nc");
  const auto file = flat::ReadModelFile(path);
  try {
    static_cast<void>(file.FieldOn(1, "vs"));
    FAIL();
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "part_0/vs_perturbation");
    EXPECT_NE(std::string{e.what()}.find("fourier"), std::string::npos);
  }
  EXPECT_NO_THROW(static_cast<void>(file.FieldOn(1, "rho")));
  const auto& vs = file.PartOf(1).variables.at("vs_perturbation");
  EXPECT_EQ(vs.Text("lateral_interpolation"), "fourier");
  EXPECT_EQ(vs.shape, (std::vector<std::size_t>{file.PartOf(1).depth.size(), 8, 6}));
  EXPECT_FALSE(vs.values.empty());
}

TEST(CartesianModelFile, RefusesAPointOffItsLayer) {
  PLANETMODEL_FIXTURE(path, "flat_model_lateral.nc");
  const auto file = flat::ReadModelFile(path);
  const auto rho = file.FieldOn(1, "rho").AsScalar();
  EXPECT_NO_THROW(static_cast<void>(rho.Value(50e3, 1.0, 1.0)));
  EXPECT_NO_THROW(static_cast<void>(rho.Value(100e3 + 1e-6, 1.0, 1.0)));
  EXPECT_THROW(static_cast<void>(rho.Value(20e3, 1.0, 1.0)), Error);
  EXPECT_THROW(static_cast<void>(rho.ValueAt({1.0, 1.0, 50e3})), Error);  // z up: above
  EXPECT_NO_THROW(static_cast<void>(rho.ValueAt({1.0, 1.0, -50e3})));
  EXPECT_THROW(static_cast<void>(rho.Value(50e3, std::nan(""), 1.0)), Error);
}

TEST(CartesianModelFile, ReadsTheModelFileAFlatManifestNames) {
  // the export of a box under topography for MFEM: the manifest beside the
  // mesh names the model file, on the mesh's reference skeleton
  PLANETMODEL_FIXTURE(path, "flat_manifests/referential/referential.json");
  const auto manifest = flat::MeshManifest(path);
  ASSERT_TRUE(manifest.Model());
  EXPECT_EQ(manifest.Model()->schema, flat::MODEL_SCHEMA);
  const auto file = flat::ReadModelFile(manifest.ModelFile());
  EXPECT_EQ(file.GetSkeleton().Boundaries(), manifest.Boundaries());
  EXPECT_TRUE(file.GetDomain().IsBox());
  // the domain's bounds are the box's sides
  EXPECT_EQ(file.GetDomain().Bounds(0),
            (std::array{manifest.FaceNamed("x_min").value, manifest.FaceNamed("x_max").value}));
  EXPECT_EQ(file.GetDomain().Bounds(1),
            (std::array{manifest.FaceNamed("y_min").value, manifest.FaceNamed("y_max").value}));
  // layer n of the file is the mesh's layer of own attribute n + 1
  for (const auto& layer : manifest.Layers()) {
    const auto n = layer.attribute - 1;
    EXPECT_EQ(file.LayerNames().at(static_cast<std::size_t>(n)), layer.name);
    for (const auto& field : manifest.Fields()) {
      if (field.name == *manifest.Displacement()) continue;
      EXPECT_TRUE(file.HasField(n, field.name)) << field.name;
    }
  }
  const auto& moho = manifest.InterfaceNamed("moho");
  const auto rho = file.FieldOn(1, "rho").AsScalar();
  EXPECT_EQ(rho.Value(moho.depth, 0.3, 0.1), moho.ValueBeside("rho", 2));
  // the mesh's nodes are reference ones: the file's stretch places them
  EXPECT_FALSE(manifest.PhysicalNodes());
  const auto mapping = file.Mapping();
  EXPECT_EQ(mapping.GetForm(), flat::VerticalStretch::Form::Stack);
  // the ridge at a node of the reliefs' grid, x = 10 / 16 and y = 0
  const auto ridge = 0.025 * (1.0 + std::cos(2.0 * std::numbers::pi * 0.025)) * 1.25;
  EXPECT_NEAR(mapping.Height(0.0, 0.625, 0.0), ridge, 1e-14);
  EXPECT_EQ(mapping.Height(manifest.BaseDepth(), 0.6, 0.0), 0.0);
}

TEST(CartesianModelFile, RefusesWhatThePythonRefusesAtItsPlace) {
  PLANETMODEL_FIXTURE(index, "flat_model_cases/bad.txt");
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
      static_cast<void>(flat::ReadModelFile(path));
      ADD_FAILURE() << label << " was accepted";
    } catch (const Error& e) {
      EXPECT_EQ(e.Where(), where) << label << ": " << e.what();
      EXPECT_NE(std::string{e.what()}.find(phrase), std::string::npos)
          << label << ": " << e.what();
    }
    ++cases;
  }
  EXPECT_GT(cases, 15);
  EXPECT_THROW(static_cast<void>(flat::ReadModelFile("/nonexistent.nc")), Error);
}
