// The model file: every file the Python writes read as the Python reads
// it (its metadata, every variable on every layer in both frames, the
// scalars' gradients, the mapping, the fields on interfaces with their
// gradients), the harmonic rule refused with the numbers still readable,
// and every malformed copy refused at its place.
#include "planetmodel/modelfile.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <format>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/gll.hpp"

namespace {

using planetmodel::Error;
using planetmodel::Frame;

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
    if (name.starts_with("model_") && entry.path().extension() == ".expect") {
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

/// The model file against the Python's reading of it, line by line.
void ExpectAsThePython(const std::filesystem::path& expect) {
  const auto model = expect.parent_path() / (expect.stem().string() + ".nc");
  SCOPED_TRACE(model.filename().string());
  const auto file = planetmodel::ReadModelFile(model);
  auto fields = std::map<std::pair<int, std::string>, planetmodel::Field>{};
  auto field = [&](int layer, const std::string& name) -> const planetmodel::Field& {
    const auto key = std::pair{layer, name};
    if (!fields.contains(key)) fields.emplace(key, file.FieldOn(layer, name));
    return fields.at(key);
  };
  auto surfaces = std::map<std::pair<int, std::string>, planetmodel::LateralInterpolant>{};
  auto surface = [&](int interface, const std::string& name)
      -> const planetmodel::LateralInterpolant& {
    const auto key = std::pair{interface, name};
    if (!surfaces.contains(key)) surfaces.emplace(key, file.SurfaceFieldOn(interface, name));
    return surfaces.at(key);
  };
  const auto mapping = file.Mapping();
  auto in = std::ifstream(expect);
  auto lines = std::vector<std::string>{};
  for (auto line = std::string{}; std::getline(in, line);) lines.push_back(line);
  // each variable's largest value, the scale of its tolerance, as the
  // Python's check of a model file has it: a field zero somewhere (a
  // pressure at the surface) is held to its size, not to that zero
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
    } else if (key == "layer_name" || key == "interface_name") {
      const auto& list = key == "layer_name" ? file.LayerNames() : file.InterfaceNames();
      const auto i = static_cast<std::size_t>(std::stoi(rest[0]));
      EXPECT_EQ(list.at(i).value_or("-"), rest[1] == "-" ? "-" : Unhex(rest[1])) << line;
    } else if (key == "constant") {
      const auto& c = file.Constants().at(rest[0]);
      EXPECT_EQ(c.value, ToDouble(rest[1])) << line;
      EXPECT_EQ(c.dimensions.has_value(), rest.size() == 6) << line;
      if (c.dimensions) {
        EXPECT_EQ(c.dimensions->mass, std::stoi(rest[2]));
        EXPECT_EQ(c.dimensions->length, std::stoi(rest[3]));
        EXPECT_EQ(c.dimensions->time, std::stoi(rest[4]));
        EXPECT_EQ(c.dimensions->temperature, std::stoi(rest[5]));
      }
    } else if (key == "absent") {
      EXPECT_FALSE(file.HasField(std::stoi(rest[0]), rest[1])) << line;
      EXPECT_THROW(static_cast<void>(file.FieldOn(std::stoi(rest[0]), rest[1])), Error) << line;
    } else if (key == "value") {
      const auto& f = field(std::stoi(rest[0]), rest[1]);
      const auto frame = rest[2] == "s" ? Frame::Spherical : Frame::Cartesian;
      const auto r = ToDouble(rest[3]);
      const auto theta = ToDouble(rest[4]);
      const auto phi = ToDouble(rest[5]);
      auto want = std::vector<double>{};
      for (auto i = std::size_t{6}; i < rest.size(); ++i) want.push_back(ToDouble(rest[i]));
      ASSERT_EQ(want.size(), f.Size()) << line;
      const auto got = f.Evaluate(r, theta, phi, frame);
      ASSERT_EQ(got.size(), want.size()) << line;
      const auto scale = std::max(Largest(want), largest[rest[1]]) + 1e-300;
      for (auto c = std::size_t{0}; c < want.size(); ++c) {
        EXPECT_NEAR(got[c], want[c], 1e-11 * scale) << line << " [" << c << "]";
      }
      ++checked;
    } else if (key == "gradient") {
      const auto scalar = field(std::stoi(rest[0]), rest[1]).AsScalar();
      const auto r = ToDouble(rest[2]);
      const auto theta = ToDouble(rest[3]);
      const auto phi = ToDouble(rest[4]);
      const auto want =
          std::vector<double>{ToDouble(rest[5]), ToDouble(rest[6]), ToDouble(rest[7])};
      const auto got = scalar.Gradient(r, theta, phi);
      const auto [lo, hi] = scalar.Underlying().Interval();
      const auto value = std::abs(scalar.Value(r, theta, phi)) + Largest(want) * (hi - lo);
      EXPECT_NEAR(got[0], want[0], 1e-9 * value / (hi - lo)) << line;
      EXPECT_NEAR(got[1], want[1], 1e-9 * (value + Largest(want))) << line;
      EXPECT_NEAR(got[2], want[2], 1e-9 * (value + Largest(want))) << line;
      ++checked;
    } else if (key == "displacement") {
      const auto X = planetmodel::Vector3{ToDouble(rest[0]), ToDouble(rest[1]), ToDouble(rest[2])};
      const auto want = planetmodel::Vector3{ToDouble(rest[3]), ToDouble(rest[4]),
                                             ToDouble(rest[5])};
      const auto got = mapping.Displacement(X);
      const auto outer = file.GetSkeleton().Boundaries().back();
      for (auto i = 0; i < 3; ++i) EXPECT_NEAR(got[i], want[i], 1e-12 * outer) << line;
      const auto moved = mapping(X);
      for (auto i = 0; i < 3; ++i) EXPECT_NEAR(moved[i], X[i] + want[i], 1e-12 * outer) << line;
      ++checked;
    } else if (key == "surface") {
      const auto& f = surface(std::stoi(rest[0]), rest[1]);
      const auto want = ToDouble(rest[4]);
      EXPECT_NEAR(f.Value(ToDouble(rest[2]), ToDouble(rest[3])), want,
                  1e-12 * (std::abs(want) + 1.0))
          << line;
      ++checked;
    } else if (key == "surface_gradient") {
      const auto& f = surface(std::stoi(rest[0]), rest[1]);
      const auto got = f.Gradient(ToDouble(rest[2]), ToDouble(rest[3]));
      const auto want = std::vector<double>{ToDouble(rest[4]), ToDouble(rest[5])};
      const auto scale = Largest(want) + 1.0;
      EXPECT_NEAR(got[0], want[0], 1e-9 * scale) << line;
      EXPECT_NEAR(got[1], want[1], 1e-9 * scale) << line;
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

TEST(ModelFile, ReadsEveryModelFileAsThePythonDoes) {
  PLANETMODEL_FIXTURE(probe, "model_prem.expect");
  const auto expectations = Expectations();
  EXPECT_GE(expectations.size(), 8u);
  for (const auto& expect : expectations) ExpectAsThePython(expect);
}

TEST(ModelFile, HoldsTheFormsOfTheMapping) {
  PLANETMODEL_FIXTURE(prem, "model_prem.nc");
  const auto identity = planetmodel::ReadModelFile(prem).Mapping();
  EXPECT_EQ(identity.GetForm(), planetmodel::ModelMapping::Form::Identity);
  EXPECT_EQ(identity.Displacement({1e6, 2e6, 3e6}), (planetmodel::Vector3{0.0, 0.0, 0.0}));
  EXPECT_THROW(static_cast<void>(identity.Displacement({7e6, 0.0, 0.0})), Error);
  PLANETMODEL_FIXTURE(stacked, "model_stacked.nc");
  EXPECT_EQ(planetmodel::ReadModelFile(stacked).Mapping().GetForm(),
            planetmodel::ModelMapping::Form::Stack);
  PLANETMODEL_FIXTURE(flattened, "model_flattened.nc");
  EXPECT_EQ(planetmodel::ReadModelFile(flattened).Mapping().GetForm(),
            planetmodel::ModelMapping::Form::Field);
}

TEST(ModelFile, ReadsAPerturbedFieldAsItsKind) {
  PLANETMODEL_FIXTURE(path, "model_kinds.nc");
  const auto file = planetmodel::ReadModelFile(path);
  EXPECT_EQ(file.FieldOn(1, "rho").PerturbationKind(), "absolute");
  EXPECT_EQ(file.FieldOn(1, "viscosity").PerturbationKind(), "log");
  EXPECT_FALSE(file.FieldOn(1, "viscosity").IsRadial());
  EXPECT_FALSE(file.FieldOn(0, "rho").PerturbationKind());
  EXPECT_TRUE(file.FieldOn(0, "rho").IsRadial());
  EXPECT_EQ(file.FieldsOf("viscosity").size(), 1u);
  EXPECT_EQ(file.FieldsOf("rho").size(), 2u);
}

TEST(ModelFile, ReadsAViscosityInItsLogarithmExactlyAndAConstantToTheBit) {
  PLANETMODEL_FIXTURE(path, "model_log.nc");
  const auto file = planetmodel::ReadModelFile(path);
  EXPECT_EQ(file.Schema(), planetmodel::MODEL_SCHEMA_WITH_LOG_RULES);
  EXPECT_EQ(file.PartOf(1).variables.at("viscosity").Text("radial_interpolation"),
            "log-polynomial");
  EXPECT_FALSE(file.HasField(0, "viscosity"));
  EXPECT_FALSE(file.HasField(3, "viscosity"));
  // three decades up and down between knots 760 km apart: geometric
  // between them, the knots at 1.2e6 + 760e3 k
  const auto graded = file.FieldOn(1, "viscosity").AsScalar();
  for (auto k = 0; k < 5; ++k) {
    const auto lo = 1.2e6 + 760e3 * k;
    const auto low = k % 2 == 0;
    for (const auto t : {0.0, 0.25, 0.5, 0.9}) {
      const auto want = 1e20 * std::pow(1e3, low ? t : 1.0 - t);
      EXPECT_NEAR(graded.Value(lo + t * 760e3, 0.3, 1.0), want, 1e-12 * want) << k << " " << t;
      const auto slope = (low ? 1.0 : -1.0) * std::log(1e3) / 760e3 * want;
      EXPECT_NEAR(graded.Gradient(lo + t * 760e3, 0.3, 1.0)[0], slope, 1e-11 * std::abs(slope));
    }
  }
  const auto constant = file.FieldOn(2, "viscosity").AsScalar();
  for (const auto r : {5.0e6, 5.13e6, 5.6e6, 6.2e6}) {
    EXPECT_EQ(constant.Value(r, 1.0, 2.0), 1e21) << r;
    EXPECT_EQ(constant.Gradient(r, 1.0, 2.0)[0], 0.0) << r;
  }
}

TEST(ModelFile, ReadsTensorsAndRefusesThemAsScalars) {
  PLANETMODEL_FIXTURE(path, "model_tensors.nc");
  const auto file = planetmodel::ReadModelFile(path);
  const auto stress = file.FieldOn(1, "stress");
  EXPECT_EQ(stress.Size(), 6u);
  EXPECT_EQ(file.FieldOn(1, "elastic").Size(), 36u);
  EXPECT_TRUE(file.FieldOn(0, "stress").IsRadial());
  EXPECT_FALSE(stress.IsRadial());
  EXPECT_THROW(static_cast<void>(stress.AsScalar()), Error);
  // the two frames are one tensor
  const auto s = stress.Evaluate(0.8, 1.1, 2.3, Frame::Spherical);
  const auto c = stress.Evaluate(0.8, 1.1, 2.3, Frame::Cartesian);
  const auto R = planetmodel::SphericalFrame(1.1, 2.3);
  const auto back = planetmodel::RotateStored(s, 2, true, std::vector{R, R});
  for (auto i = 0; i < 6; ++i) EXPECT_NEAR(back[i], c[i], 1e-13);
}

TEST(ModelFile, TheDerivativesAreTheDifferencesOfTheValues) {
  // central differences of the Cartesian components, for tensors radial
  // (whose frame turns with the angles) and lateral, Voigt and full
  PLANETMODEL_FIXTURE(path, "model_tensors.nc");
  const auto file = planetmodel::ReadModelFile(path);
  struct Probe {
    int layer;
    const char* name;
    double r, theta, phi;
  };
  for (const auto& [layer, name, r, theta, phi] :
       {Probe{0, "stress", 0.31, 1.13, 2.37}, Probe{1, "stress", 0.83, 0.61, -0.4},
        Probe{1, "elastic", 0.71, 2.05, 4.1}, Probe{1, "rho", 0.66, 1.7, 0.9}}) {
    SCOPED_TRACE(std::format("{} on layer {}", name, layer));
    const auto f = file.FieldOn(layer, name);
    const auto n = f.Size();
    const auto [dr, dt, dp] = f.Derivatives(r, theta, phi);
    ASSERT_EQ(dr.size(), n);
    auto at = [&](double rr, double tt, double pp) {
      return f.Evaluate(rr, tt, pp, Frame::Cartesian);
    };
    constexpr auto h = 1e-6;
    const auto steps = std::array<std::array<double, 3>, 3>{{{h, 0, 0}, {0, h, 0}, {0, 0, h}}};
    const auto got = std::array<const std::vector<double>*, 3>{&dr, &dt, &dp};
    const auto scale = Largest(at(r, theta, phi)) + Largest(dr) + Largest(dt) + Largest(dp);
    for (auto d = std::size_t{0}; d < 3; ++d) {
      const auto& s = steps[d];
      const auto plus = at(r + s[0], theta + s[1], phi + s[2]);
      const auto minus = at(r - s[0], theta - s[1], phi - s[2]);
      for (auto c = std::size_t{0}; c < n; ++c) {
        EXPECT_NEAR((*got[d])[c], (plus[c] - minus[c]) / (2 * h), 1e-7 * scale)
            << "d" << d << " [" << c << "]";
      }
    }
  }
}

TEST(ModelFile, HoldsFieldsOnInterfacesBesideTheLayers) {
  PLANETMODEL_FIXTURE(path, "model_surfaces.nc");
  const auto file = planetmodel::ReadModelFile(path);
  ASSERT_TRUE(file.Surfaces());
  EXPECT_EQ(file.Surfaces()->interfaces, (std::vector<int>{0, 1}));
  EXPECT_EQ(file.SurfaceNames(0), (std::vector<std::string>{"K_shape"}));
  EXPECT_EQ(file.SurfaceNames(1), (std::vector<std::string>{"K_shape", "shift"}));
  const auto& kernel = file.Surfaces()->variables.at("K_shape");
  EXPECT_EQ(kernel.Numbers("dimensions"), (std::vector<double>{0.0, -2.0, 0.0, 0.0}));
  EXPECT_EQ(kernel.Text("long_name"), "a shape kernel");
  EXPECT_EQ(file.Surfaces()->variables.at("shift").Text("lateral_interpolation"), "cubic");
  // a field on interfaces is no field of the layers, nor the other way round
  EXPECT_FALSE(file.HasField(1, "K_shape"));
  EXPECT_THROW(static_cast<void>(file.FieldOn(1, "K_shape")), Error);
  EXPECT_FALSE(file.HasSurfaceField(1, "rho"));
  EXPECT_THROW(static_cast<void>(file.SurfaceFieldOn(0, "shift")), Error);
  EXPECT_THROW(static_cast<void>(file.SurfaceFieldOn(2, "K_shape")), Error);
  // a file without the group holds none
  PLANETMODEL_FIXTURE(prem, "model_prem.nc");
  const auto plain = planetmodel::ReadModelFile(prem);
  EXPECT_FALSE(plain.Surfaces());
  EXPECT_TRUE(plain.SurfaceNames(1).empty());
}

TEST(ModelFile, RefusesWhatIsNoFieldOfTheModel) {
  PLANETMODEL_FIXTURE(lateral, "model_lateral.nc");
  const auto file = planetmodel::ReadModelFile(lateral);
  EXPECT_THROW(static_cast<void>(file.FieldOn(1, "rho_poles")), Error);
  EXPECT_THROW(static_cast<void>(file.FieldOn(1, "nonesuch")), Error);
  EXPECT_THROW(static_cast<void>(file.PartOf(7)), Error);
  PLANETMODEL_FIXTURE(kinds, "model_kinds.nc");
  const auto perturbed = planetmodel::ReadModelFile(kinds);
  EXPECT_THROW(static_cast<void>(perturbed.FieldOn(1, "rho_perturbation")), Error);
  EXPECT_FALSE(perturbed.HasField(1, "rho_perturbation"));
  PLANETMODEL_FIXTURE(flattened, "model_flattened.nc");
  try {
    static_cast<void>(planetmodel::ReadModelFile(flattened).FieldOn(0, "displacement"));
    FAIL();
  } catch (const Error& e) {
    EXPECT_NE(std::string{e.what()}.find("Mapping()"), std::string::npos);
  }
}

TEST(ModelFile, RefusesTheHarmonicRuleAndKeepsItsNumbers) {
  PLANETMODEL_FIXTURE(path, "model_harmonic.nc");
  const auto file = planetmodel::ReadModelFile(path);
  try {
    static_cast<void>(file.FieldOn(1, "rho"));
    FAIL();
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "part_0/rho");
    EXPECT_NE(std::string{e.what()}.find("harmonic"), std::string::npos);
  }
  EXPECT_NO_THROW(static_cast<void>(file.FieldOn(1, "vp")));
  const auto& rho = file.PartOf(1).variables.at("rho");
  EXPECT_EQ(rho.Text("lateral_interpolation"), "harmonic");
  EXPECT_FALSE(rho.values.empty());
  EXPECT_EQ(file.PartOf(1).grid->Kind(), "gauss_legendre");
}

TEST(ModelFile, RefusesAPointOffItsLayer) {
  PLANETMODEL_FIXTURE(path, "model_lateral.nc");
  const auto file = planetmodel::ReadModelFile(path);
  const auto rho = file.FieldOn(1, "rho").AsScalar();
  EXPECT_NO_THROW(static_cast<void>(rho.Value(0.5, 1.0, 1.0)));
  EXPECT_NO_THROW(static_cast<void>(rho.Value(1.0 + 1e-12, 1.0, 1.0)));
  EXPECT_THROW(static_cast<void>(rho.Value(0.4, 1.0, 1.0)), Error);
  EXPECT_THROW(static_cast<void>(rho.Value(0.7, -0.1, 1.0)), Error);
}

TEST(ModelFile, RefusesWhatThePythonRefusesAtItsPlace) {
  PLANETMODEL_FIXTURE(index, "model_cases/bad.txt");
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
      static_cast<void>(planetmodel::ReadModelFile(path));
      ADD_FAILURE() << label << " was accepted";
    } catch (const Error& e) {
      EXPECT_EQ(e.Where(), where) << label << ": " << e.what();
      EXPECT_NE(std::string{e.what()}.find(phrase), std::string::npos)
          << label << ": " << e.what();
    }
    ++cases;
  }
  EXPECT_GT(cases, 15);
  EXPECT_THROW(static_cast<void>(planetmodel::ReadModelFile("/nonexistent.nc")), Error);
}

TEST(ModelFile, TheGllPointsAreThePythons) {
  PLANETMODEL_FIXTURE(path, "gll.txt");
  auto in = std::ifstream(path);
  auto rows = 0;
  for (auto line = std::string{}; std::getline(in, line); ++rows) {
    auto words = std::istringstream{line};
    auto key = std::string{};
    auto n = std::size_t{0};
    words >> key >> n;
    const auto got = planetmodel::GllPoints(n);
    auto k = std::size_t{0};
    for (auto w = std::string{}; words >> w; ++k) {
      EXPECT_NEAR(got.at(k), ToDouble(w), 1e-15) << "ngll " << n << " point " << k;
    }
    EXPECT_EQ(k, n);
  }
  EXPECT_EQ(rows, 12);
  EXPECT_THROW(static_cast<void>(planetmodel::GllPoints(1)), Error);
}
