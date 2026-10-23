// The writer: the shipped grids and the vocabulary as the Python's; every
// fixture written back unchanged; files built from known fields, on every
// grid kind, and fields on interfaces, read back to those fields and left
// for the Python to read; and the refusals, before anything is written.
#include "planetmodel/modelfile.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <numbers>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "fixtures.hpp"

namespace {

namespace pm = planetmodel;
using pm::Error;
using pm::Frame;
using pm::LayerSource;
using pm::Vector3;

constexpr auto pi = std::numbers::pi;

double ToDouble(const std::string& text) {
  auto value = 0.0;
  std::from_chars(text.data(), text.data() + text.size(), value);
  return value;
}

std::vector<std::string> Split(const std::string& text, char by) {
  auto out = std::vector<std::string>{};
  auto in = std::istringstream{text};
  for (auto part = std::string{}; std::getline(in, part, by);) out.push_back(part);
  return out;
}

std::string Unhex(const std::string& hex) {
  auto out = std::string{};
  for (auto i = std::size_t{0}; i + 1 < hex.size(); i += 2) {
    out += static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16));
  }
  return out;
}

/// A shipped grid by the name grids.txt gives it.
pm::AngularGrid GridNamed(const std::string& label) {
  const auto p = Split(label, ':');
  if (p[0] == "gl") {
    return p.size() == 3 ? pm::GaussLegendre(std::stoi(p[1]), std::stoi(p[2]))
                         : pm::GaussLegendre(std::stoi(p[1]));
  }
  if (p[0] == "dh") return pm::DriscollHealy(std::stoi(p[1]), std::stoi(p[2]));
  return pm::Equiangular(std::stoi(p[1]), std::stoi(p[2]), p[3] == "1");
}

/// A known field: its components in `frame` at a point.
using Components = std::function<std::vector<double>(double r, double theta, double phi)>;

/// What a written file holds of a field on a layer, as a function.
struct Known {
  int layer;
  std::string name;
  Frame frame;
  Components f;
};

double Largest(const std::vector<double>& v) {
  auto out = 0.0;
  for (const auto x : v) out = std::max(out, std::abs(x));
  return out;
}

/// Each known field read back from the file at every point of its layer,
/// grid nodes and poles, to rounding.
void ExpectReadsBack(const pm::ModelFile& file, const std::vector<Known>& known) {
  for (const auto& k : known) {
    SCOPED_TRACE(std::format("{} on layer {}", k.name, k.layer));
    const auto field = file.FieldOn(k.layer, k.name);
    const auto points = file.PointsOf(k.layer);
    for (auto i = std::size_t{0}; i < points.Size(); ++i) {
      const auto at = points.At(i);
      const auto want = k.f(at.r, at.theta, at.phi);
      const auto got = field.Evaluate(at.r, at.theta, at.phi, k.frame);
      ASSERT_EQ(got.size(), want.size());
      const auto scale = std::max(1.0, Largest(want));
      for (auto c = std::size_t{0}; c < want.size(); ++c) {
        ASSERT_NEAR(got[c], want[c], 1e-12 * scale)
            << "at (" << at.r << ", " << at.theta << ", " << at.phi << ") [" << c << "]";
      }
    }
  }
}

/// The expectations the Python is held to for a file the C++ wrote, from
/// the C++'s own reading of it: `value` lines for every field on every
/// layer at points between the nodes, in both frames, `displacement`
/// lines for the mapping, and `surface` lines for every field on every
/// interface, as `make_fixtures.py` writes them for its own.
void WriteExpect(const std::filesystem::path& path) {
  const auto file = pm::ReadModelFile(path);
  auto out = std::ofstream(path.parent_path() / (path.stem().string() + ".expect"));
  const auto nlayers = static_cast<int>(file.GetSkeleton().NLayers());
  for (auto layer = 0; layer < nlayers; ++layer) {
    const auto [lo, hi] = file.GetSkeleton().Interval(static_cast<std::size_t>(layer));
    for (const auto& name : file.Names(layer)) {
      const auto field = file.FieldOn(layer, name);
      for (const auto& [f, theta, phi] : {std::tuple{0.13, 0.0, -1.0}, std::tuple{0.41, 0.7, 0.3},
                                          std::tuple{0.77, pi, 4.0}, std::tuple{0.96, 2.2, 9.5}}) {
        const auto r = lo + f * (hi - lo);
        for (const auto& [frame, tag] :
             {std::pair{Frame::Spherical, "s"}, std::pair{Frame::Cartesian, "c"}}) {
          out << std::format("value {} {} {} {} {} {}", layer, name, tag, r, theta, phi);
          for (const auto v : field.Evaluate(r, theta, phi, frame)) out << std::format(" {}", v);
          out << "\n";
        }
      }
    }
  }
  const auto mapping = file.Mapping();
  if (mapping.GetForm() != pm::ModelMapping::Form::Identity) {
    const auto& b = file.GetSkeleton().Boundaries();
    for (const auto& [f, theta, phi] : {std::tuple{0.0, 0.4, 1.0}, std::tuple{0.21, 1.1, -2.0},
                                        std::tuple{0.5, 2.0, 0.7}, std::tuple{0.83, 2.9, 3.3},
                                        std::tuple{1.0, 0.0, 0.0}, std::tuple{1.0, 1.4, 5.9}}) {
      const auto X = pm::CartesianPoint(b.front() + f * (b.back() - b.front()), theta, phi);
      const auto u = mapping.Displacement(X);
      out << std::format("displacement {} {} {} {} {} {}\n", X[0], X[1], X[2], u[0], u[1], u[2]);
    }
  }
  for (auto k = 0; k < static_cast<int>(file.InterfaceRadii().size()); ++k) {
    for (const auto& name : file.SurfaceNames(k)) {
      const auto f = file.SurfaceFieldOn(k, name);
      for (const auto& [theta, phi] : {std::pair{0.0, -1.0}, std::pair{0.7, 0.3},
                                       std::pair{pi, 4.0}, std::pair{2.2, 9.5}}) {
        out << std::format("surface {} {} {} {} {}\n", k, name, theta, phi, f.Value(theta, phi));
      }
    }
  }
}

/// Written beside the others, read back, its expectations written.
std::filesystem::path Written(const std::string& name, const pm::ModelFile& file) {
  const auto path = pm::testing::WrittenDir() / name;
  pm::WriteModelFile(path, file);
  WriteExpect(path);
  return path;
}

// -- known fields -----------------------------------------------------------

/// A rotation and stretch, linear in X: Cartesian components.
Vector3 Flow(const Vector3& x) { return {x[1], -x[0], 0.5 * x[2]}; }

std::vector<double> AsVector(const Vector3& v) { return {v[0], v[1], v[2]}; }

/// A displacement smooth through the centre: Cartesian components.
Vector3 Swell(const Vector3& x) {
  constexpr auto e = 1e-3;
  return {e * x[0] * x[2], e * x[1] * x[2], -0.5 * e * (x[0] * x[0] + x[1] * x[1])};
}

/// A symmetric tensor in Voigt order, Cartesian, of the point.
std::vector<double> Stress(const Vector3& x) {
  return {1.0 + x[0] * x[1], 2.0 + x[2], 3.0 - x[0], 0.5 * x[1] * x[2], 0.2 * x[0], 0.1 + x[1]};
}

/// A Voigt elasticity matrix, Cartesian, of the point.
std::vector<double> Elastic(const Vector3& x) {
  auto out = std::vector<double>(36);
  const auto bump = 0.1 * (x[0] + 2.0 * x[1] * x[2]);
  for (auto i = 0; i < 6; ++i) {
    for (auto j = 0; j < 6; ++j) {
      out[static_cast<std::size_t>(6 * i + j)] = (i == j ? 3.0 : 0.5) + bump * (i + j) / 10.0;
    }
  }
  return out;
}

/// Components of rank `rank` turned from Cartesian to spherical at a point.
std::vector<double> ToSphericalStored(const std::vector<double>& cartesian, int rank, double theta,
                                      double phi) {
  const auto R = pm::SphericalFrame(theta, phi);
  auto Rt = pm::Matrix3{};
  for (auto i = std::size_t{0}; i < 3; ++i) {
    for (auto j = std::size_t{0}; j < 3; ++j) Rt[i][j] = R[j][i];
  }
  return pm::RotateStored(cartesian, rank, true,
                          std::vector<pm::Matrix3>(static_cast<std::size_t>(rank), Rt));
}

/// A source's values at the points of a layer, from a function.
std::vector<double> AtPoints(const pm::LayerPoints& points, const Components& f) {
  auto out = std::vector<double>{};
  for (auto i = std::size_t{0}; i < points.Size(); ++i) {
    const auto at = points.At(i);
    const auto v = f(at.r, at.theta, at.phi);
    out.insert(out.end(), v.begin(), v.end());
  }
  return out;
}

Components Scalar(std::function<double(double, double, double)> f) {
  return [f](double r, double t, double p) { return std::vector<double>{f(r, t, p)}; };
}

// The fields of the SI model: a scalar on every layer, of the radius alone
// or not; one given by its values; a vector in either frame and of the
// radius alone; the three kinds of perturbation.
double Rho0(double r) { return 13.0 - 2.0 * r * r; }
double Rho1(double r, double t, double p) {
  return 5.0 + 0.3 * r * std::cos(t) + 0.1 * r * std::sin(t) * std::cos(p);
}
double Rho2(double r) { return 4.0 - r; }
double Vs(double r, double t, double p) { return 3.0 + 0.2 * r * std::sin(t) * std::sin(p); }
double Vsv(double r) { return 4.0 + r; }
double VsvRelative1(double t) { return 0.02 * std::cos(t); }
double VsvRelative2(double r) { return 0.01 * r; }
double Eta(double r) { return 1e21 * (1.0 + r); }
double EtaLog(double t, double p) { return 0.5 * std::sin(t) * std::sin(t) * std::cos(p); }
double Temperature(double r) { return 3000.0 - 1000.0 * r; }
double TemperatureAbsolute(double t) { return 50.0 * std::cos(t); }
double Relief1(double t, double) { return 0.002 * std::cos(t); }
double Relief3(double t, double p) { return 0.001 * std::sin(t) * std::sin(t) * std::cos(2.0 * p); }

/// The SI model: three layers, each a part on a grid of its own kind
/// (Driscoll-Healy, with the north pole a node; Gauss-Legendre, with no
/// pole; equiangular with both), and the mapping a stack of reliefs.
pm::ModelFile SiModel() {
  auto file = pm::ModelFile(pm::Skeleton({0.0, 0.3, 0.6, 1.0}));
  const auto& sk = file.GetSkeleton();
  file.SetName("written si");
  file.SetCitation({"A. Author (2026).  A written model.", "B. Author (2026).  Its source."});
  file.AddPart(0, 0, pm::ElementEdges(sk, 0, 0, 0.1), 5, pm::DriscollHealy(6));
  file.AddPart(1, 1, pm::ElementEdges(sk, 1, 1, 0.15), 4, pm::GaussLegendre(5));
  file.AddPart(2, 2, pm::ElementEdges(sk, 2, 2, 0.2), 3, pm::Equiangular(9, 16, true));
  file.AddField("rho", {{0, LayerSource::Radial(Rho0)}, {1, LayerSource::Lateral(Rho1)},
                        {2, LayerSource::Radial(Rho2)}});
  auto vs = std::map<int, LayerSource>{};
  for (auto layer = 0; layer < 3; ++layer) {
    vs.emplace(layer, LayerSource::LateralValues(AtPoints(file.PointsOf(layer), Scalar(Vs)),
                                                 Frame::Spherical));
  }
  file.AddField("vs", vs, {.lateral = pm::LateralRule::Cubic});
  const auto flow = pm::FieldSpec{.character = pm::VECTOR, .dimensions = pm::units::VELOCITY,
                                  .meaning = "a flow"};
  file.AddField("flow", flow,
                {{0, LayerSource::Radial([](double r, std::span<double> out) {
                    out[0] = 0.1 * r;
                    out[1] = out[2] = 0.0;
                  })},
                 {1, LayerSource::Lateral(
                         [](double r, double t, double p, std::span<double> out) {
                           const auto u = Flow(pm::CartesianPoint(r, t, p));
                           std::ranges::copy(u, out.begin());
                         },
                         Frame::Cartesian)},
                 {2, LayerSource::Lateral(
                         [](double r, double t, double p, std::span<double> out) {
                           const auto u = pm::ToSpherical(Flow(pm::CartesianPoint(r, t, p)), t, p);
                           std::ranges::copy(u, out.begin());
                         },
                         Frame::Spherical)}});
  file.AddPerturbedField(
      "vsv",
      {{0, LayerSource::Radial(Vsv)}, {1, LayerSource::Radial(Vsv)}, {2, LayerSource::Radial(Vsv)}},
      {.kind = "relative",
       .layers = {
           {1, LayerSource::Lateral([](double, double t, double) { return VsvRelative1(t); })},
           {2, LayerSource::Radial(VsvRelative2)}}});
  file.AddPerturbedField(
      "viscosity", {{1, LayerSource::Radial(Eta)}, {2, LayerSource::Radial(Eta)}},
      {.kind = "log",
       .layers = {{1,
                   LayerSource::Lateral([](double, double t, double p) { return EtaLog(t, p); })}},
       .rules = {.lateral = pm::LateralRule::Cubic}});
  file.AddPerturbedField("temperature",
                         {{0, LayerSource::Radial(Temperature)},
                          {1, LayerSource::Radial(Temperature)},
                          {2, LayerSource::Radial(Temperature)}},
                         {.kind = "absolute",
                          .layers = {{0, LayerSource::Lateral([](double, double t, double) {
                                        return TemperatureAbsolute(t);
                                      })},
                                     {2, LayerSource::Lateral([](double, double t, double) {
                                        return TemperatureAbsolute(t);
                                      })}}});
  file.SetStack(
      {0.0, 0.3, 0.6, 1.0}, pm::GaussLegendre(8),
      {std::nullopt, LayerSource::Lateral([](double, double t, double p) { return Relief1(t, p); }),
       std::nullopt,
       LayerSource::Lateral([](double, double t, double p) { return Relief3(t, p); })});
  return file;
}

std::vector<Known> SiKnown() {
  auto out = std::vector<Known>{};
  out.push_back(
      {0, "rho", Frame::Spherical, Scalar([](double r, double, double) { return Rho0(r); })});
  out.push_back({1, "rho", Frame::Spherical, Scalar(Rho1)});
  out.push_back(
      {2, "rho", Frame::Spherical, Scalar([](double r, double, double) { return Rho2(r); })});
  out.push_back(
      {0, "vsv", Frame::Spherical, Scalar([](double r, double, double) { return Vsv(r); })});
  out.push_back({1, "vsv", Frame::Spherical, Scalar([](double r, double t, double) {
                   return Vsv(r) * (1.0 + VsvRelative1(t));
                 })});
  out.push_back({2, "vsv", Frame::Spherical, Scalar([](double r, double, double) {
                   return Vsv(r) * (1.0 + VsvRelative2(r));
                 })});
  for (auto layer = 0; layer < 3; ++layer) {
    out.push_back({layer, "vs", Frame::Spherical, Scalar(Vs)});
    out.push_back(
        {layer, "temperature", Frame::Spherical, Scalar([layer](double r, double t, double) {
           return Temperature(r) + (layer == 1 ? 0.0 : TemperatureAbsolute(t));
         })});
  }
  for (const auto layer : {1, 2}) {
    out.push_back(
        {layer, "viscosity", Frame::Spherical, Scalar([layer](double r, double t, double p) {
           return Eta(r) * (layer == 1 ? std::exp(EtaLog(t, p)) : 1.0);
         })});
  }
  out.push_back({0, "flow", Frame::Spherical,
                 [](double r, double, double) { return std::vector<double>{0.1 * r, 0.0, 0.0}; }});
  for (const auto layer : {1, 2}) {
    for (const auto frame : {Frame::Cartesian, Frame::Spherical}) {
      out.push_back({layer, "flow", frame, [frame](double r, double t, double p) {
                       const auto u = Flow(pm::CartesianPoint(r, t, p));
                       return AsVector(frame == Frame::Cartesian ? u : pm::ToSpherical(u, t, p));
                     }});
    }
  }
  return out;
}

/// The non-SI model: a hollow body of two layers in one part on a custom
/// grid, a field of the radius alone on one layer and not on the other,
/// the mapping a displacement field, with constants, meta and names.
pm::ModelFile CustomModel() {
  auto file = pm::ModelFile(pm::Skeleton({0.2, 0.55, 1.0}),
                            pm::Scales({.length = 6.371e6, .mass = 5.97e24, .time = 3600.0,
                                        .temperature = 1000.0}),
                            1e-8);
  const auto& sk = file.GetSkeleton();
  file.SetName("written custom");
  file.SetLayerName(0, "lower");
  file.SetLayerName(1, "upper");
  file.SetInterfaceName(0, "inner");
  file.SetInterfaceName(2, "surface");
  file.SetConstant("G", {.value = 1.25, .dimensions = pm::units::GRAVITATIONAL_CONSTANT,
                         .meaning = "the gravitational constant in the model's units"});
  file.SetMeta("iteration", 12);
  file.SetMeta("code", std::string{"planetmodel tests"});
  file.SetMeta("misfit", 0.25);
  const auto grid = pm::AngularGrid({0.4, 0.9, 1.5, 2.1, 2.7}, {0.1, 1.0, 2.2, 3.5, 4.4, 5.6});
  file.AddPart(0, 1, pm::ElementEdges(sk, 0, 1, 0.1), 5, grid);
  file.AddField("rho", {{0, LayerSource::Radial([](double r) { return 5.5 - r; })},
                        {1, LayerSource::Lateral([](double r, double t, double) {
                           return 3.3 + 0.1 * r * std::cos(t);
                         })}});
  file.SetDisplacement(
      {{0, LayerSource::Lateral(
               [](double r, double t, double p, std::span<double> out) {
                 std::ranges::copy(Swell(pm::CartesianPoint(r, t, p)), out.begin());
               },
               Frame::Cartesian)},
       {1, LayerSource::Lateral(
               [](double r, double t, double p, std::span<double> out) {
                 std::ranges::copy(pm::ToSpherical(Swell(pm::CartesianPoint(r, t, p)), t, p),
                                   out.begin());
               },
               Frame::Spherical)}});
  return file;
}

/// Tensors on equiangular midpoints: a stress given in the spherical
/// frame, an elasticity matrix in the Cartesian.
pm::ModelFile TensorModel() {
  auto file = pm::ModelFile(pm::Skeleton({0.0, 1.0}));
  file.AddPart(0, 0, pm::ElementEdges(file.GetSkeleton(), 0, 0, 0.25), 4, pm::Equiangular(6, 12));
  file.AddField("stress",
                {.character = pm::STRESS, .dimensions = pm::units::MODULUS, .meaning = "a stress"},
                {{0, LayerSource::Lateral(
                         [](double r, double t, double p, std::span<double> out) {
                           std::ranges::copy(
                               ToSphericalStored(Stress(pm::CartesianPoint(r, t, p)), 2, t, p),
                               out.begin());
                         },
                         Frame::Spherical)}});
  file.AddField("elastic_moduli",
                {{0, LayerSource::Lateral(
                         [](double r, double t, double p, std::span<double> out) {
                           std::ranges::copy(Elastic(pm::CartesianPoint(r, t, p)), out.begin());
                         },
                         Frame::Cartesian)}},
                {.lateral = pm::LateralRule::Cubic});
  return file;
}

}  // namespace

TEST(Writer, TheShippedGridsAreThePythons) {
  PLANETMODEL_FIXTURE(path, "grids.txt");
  auto in = std::ifstream(path);
  auto grids = std::map<std::string, pm::AngularGrid>{};
  auto checked = 0;
  for (auto line = std::string{}; std::getline(in, line);) {
    auto words = std::istringstream{line};
    auto key = std::string{};
    auto label = std::string{};
    words >> key >> label;
    SCOPED_TRACE(label);
    if (!grids.contains(label)) grids.emplace(label, GridNamed(label));
    const auto& g = grids.at(label);
    if (key == "band") {
      auto kind = std::string{};
      auto lmax = 0;
      words >> kind >> lmax;
      EXPECT_EQ(g.Kind(), kind);
      EXPECT_EQ(g.Lmax(), lmax);
      continue;
    }
    auto want = std::vector<double>{};
    for (auto w = std::string{}; words >> w;) want.push_back(ToDouble(w));
    const auto& got = key == "colatitudes" ? g.Colatitudes()
                      : key == "longitudes" ? g.Longitudes()
                                            : *g.Weights();
    ASSERT_EQ(got.size(), want.size()) << key;
    for (auto i = std::size_t{0}; i < want.size(); ++i) {
      EXPECT_NEAR(got[i], want[i],
                  (key == "weights" ? 1e-13 : 2e-14) * std::max(1.0, std::abs(want[i])))
          << key << " " << i;
    }
    ++checked;
  }
  EXPECT_EQ(checked, 3 * 16);
  EXPECT_TRUE(grids.at("dh:6:1").NorthPoleNode());
  EXPECT_FALSE(grids.at("dh:6:1").SouthPoleNode());
  EXPECT_TRUE(grids.at("eq:9:16:1").SouthPoleNode());
  EXPECT_THROW(static_cast<void>(pm::GaussLegendre(5, 10)), Error);
  EXPECT_THROW(static_cast<void>(pm::DriscollHealy(3, 3)), Error);
  EXPECT_THROW(static_cast<void>(pm::Equiangular(1, 4, true)), Error);
}

TEST(Writer, TheVocabularyIsThePythons) {
  PLANETMODEL_FIXTURE(path, "vocabulary.txt");
  auto in = std::ifstream(path);
  auto names = 0u;
  auto families = 0u;
  for (auto line = std::string{}; std::getline(in, line);) {
    auto words = std::istringstream{line};
    auto name = std::string{};
    auto rank = 0, weight = 0, voigt = 0, mass = 0, length = 0, time = 0, temperature = 0;
    auto meaning = std::string{};
    words >> name;
    const auto family = name == "family";
    if (family) words >> name;
    words >> rank >> weight >> voigt >> mass >> length >> time >> temperature >> meaning;
    if (family) {
      ++families;
      ASSERT_TRUE(pm::Families().contains(name)) << name;
      EXPECT_EQ(pm::Member(name + "_12"), (std::pair{name, 12})) << name;
      EXPECT_FALSE(pm::Member(name + "_0").has_value()) << name;
      EXPECT_FALSE(pm::Member(name + "_07").has_value()) << name;
      EXPECT_FALSE(pm::Member(name).has_value()) << name;
      name += "_3";
    } else {
      ++names;
    }
    const auto spec = pm::VocabularySpec(name);
    ASSERT_TRUE(spec.has_value()) << name;
    EXPECT_EQ(spec->character, pm::Character(rank, weight, voigt != 0)) << name;
    EXPECT_EQ(spec->dimensions, (pm::Dimensions{.mass = mass, .length = length, .time = time,
                                                .temperature = temperature}))
        << name;
    EXPECT_EQ(spec->meaning, Unhex(meaning)) << name;
  }
  EXPECT_EQ(names, pm::Vocabulary().size());
  EXPECT_EQ(families, pm::Families().size());
  EXPECT_FALSE(pm::VocabularySpec("relaxation_time_").has_value());
  EXPECT_FALSE(pm::VocabularySpec("viscosity_1").has_value());
}

TEST(Writer, WritesBackWhatItReads) {
  PLANETMODEL_FIXTURE(probe, "model_prem.nc");
  auto files = 0;
  for (const auto& entry : std::filesystem::directory_iterator(pm::testing::FixturesDir())) {
    const auto name = entry.path().filename().string();
    if (!name.starts_with("model_") || entry.path().extension() != ".nc") continue;
    SCOPED_TRACE(name);
    const auto before = pm::ReadModelFile(entry.path());
    const auto path = pm::testing::WrittenDir() / ("roundtrip_" + name);
    pm::WriteModelFile(path, before);
    const auto after = pm::ReadModelFile(path);
    EXPECT_EQ(after.Schema(), before.Schema());
    EXPECT_EQ(after.Name(), before.Name());
    EXPECT_EQ(after.Citation(), before.Citation());
    EXPECT_EQ(after.GetScales(), before.GetScales());
    EXPECT_EQ(after.Rtol(), before.Rtol());
    EXPECT_EQ(after.Meta(), before.Meta());
    EXPECT_EQ(after.GetSkeleton().Boundaries(), before.GetSkeleton().Boundaries());
    EXPECT_EQ(after.LayerNames(), before.LayerNames());
    EXPECT_EQ(after.InterfaceNames(), before.InterfaceNames());
    EXPECT_EQ(after.InterfaceRadii(), before.InterfaceRadii());
    ASSERT_EQ(after.Constants().size(), before.Constants().size());
    for (const auto& [cname, c] : before.Constants()) {
      EXPECT_EQ(after.Constants().at(cname).value, c.value);
      EXPECT_EQ(after.Constants().at(cname).dimensions, c.dimensions);
      EXPECT_EQ(after.Constants().at(cname).meaning, c.meaning);
    }
    ASSERT_EQ(after.Parts().size(), before.Parts().size());
    for (auto k = std::size_t{0}; k < before.Parts().size(); ++k) {
      const auto& a = after.Parts()[k];
      const auto& b = before.Parts()[k];
      EXPECT_EQ(a.layers, b.layers);
      EXPECT_EQ(a.ngll, b.ngll);
      EXPECT_EQ(a.element_layer, b.element_layer);
      EXPECT_EQ(a.element_bounds, b.element_bounds);
      EXPECT_EQ(a.radius, b.radius);
      EXPECT_EQ(a.grid->Kind(), b.grid->Kind());
      EXPECT_EQ(a.grid->Lmax(), b.grid->Lmax());
      EXPECT_EQ(a.grid->Weights(), b.grid->Weights());
      // the angles go through degrees, both ways, as the Python's do
      for (auto i = std::size_t{0}; i < b.grid->NTheta(); ++i) {
        EXPECT_NEAR(a.grid->Colatitudes()[i], b.grid->Colatitudes()[i], 1e-15);
      }
      for (auto j = std::size_t{0}; j < b.grid->NPhi(); ++j) {
        EXPECT_NEAR(a.grid->Longitudes()[j], b.grid->Longitudes()[j], 1e-15);
      }
      ASSERT_EQ(a.variables.size(), b.variables.size());
      for (const auto& [vname, v] : b.variables) {
        SCOPED_TRACE(vname);
        const auto& w = a.variables.at(vname);
        EXPECT_EQ(w.dimensions, v.dimensions);
        EXPECT_EQ(w.shape, v.shape);
        ASSERT_EQ(w.values.size(), v.values.size());
        for (auto i = std::size_t{0}; i < v.values.size(); ++i) {
          if (std::isnan(v.values[i])) {
            ASSERT_TRUE(std::isnan(w.values[i])) << i;
          } else {
            ASSERT_EQ(w.values[i], v.values[i]) << i;
          }
        }
        for (const auto& [key, value] : v.attributes) {
          if (key.starts_with('_')) continue;
          ASSERT_TRUE(w.attributes.contains(key)) << key;
          EXPECT_EQ(w.attributes.at(key), value) << key;
        }
      }
    }
    EXPECT_EQ(after.Geometry().has_value(), before.Geometry().has_value());
    if (before.Geometry()) {
      EXPECT_EQ(after.Geometry()->radii, before.Geometry()->radii);
      EXPECT_EQ(after.Geometry()->values, before.Geometry()->values);
      EXPECT_EQ(after.Geometry()->poles, before.Geometry()->poles);
      EXPECT_EQ(after.Geometry()->lateral, before.Geometry()->lateral);
    }
    EXPECT_EQ(after.Surfaces().has_value(), before.Surfaces().has_value());
    if (before.Surfaces()) {
      const auto& a = *after.Surfaces();
      const auto& b = *before.Surfaces();
      EXPECT_EQ(a.interfaces, b.interfaces);
      EXPECT_EQ(a.grid->Kind(), b.grid->Kind());
      ASSERT_EQ(a.variables.size(), b.variables.size());
      for (const auto& [vname, v] : b.variables) {
        SCOPED_TRACE(vname);
        const auto& w = a.variables.at(vname);
        EXPECT_EQ(w.dimensions, v.dimensions);
        ASSERT_EQ(w.values.size(), v.values.size());
        for (auto i = std::size_t{0}; i < v.values.size(); ++i) {
          if (std::isnan(v.values[i])) {
            ASSERT_TRUE(std::isnan(w.values[i])) << i;
          } else {
            ASSERT_EQ(w.values[i], v.values[i]) << i;
          }
        }
        for (const auto& [key, value] : v.attributes) {
          if (!key.starts_with('_')) {
            EXPECT_EQ(w.attributes.at(key), value) << key;
          }
        }
      }
    }
    ++files;
  }
  EXPECT_GE(files, 9);
}

TEST(Writer, WritesFieldsOnEveryShippedGridThatReadBackToThemselves) {
  const auto file = SiModel();
  // the built file reads as it will be read
  ExpectReadsBack(file, SiKnown());
  const auto path = Written("written_si.nc", file);
  const auto read = pm::ReadModelFile(path);
  ExpectReadsBack(read, SiKnown());
  EXPECT_EQ(read.Name(), "written si");
  EXPECT_EQ(read.WrittenBy(), "planetmodel C++ " + std::string{pm::Version()});
  EXPECT_NE(read.Created().find('T'), std::string::npos);
  EXPECT_EQ(read.Citation(), (std::vector<std::string>{"A. Author (2026).  A written model.",
                                                        "B. Author (2026).  Its source."}));
  EXPECT_EQ(read.Parts()[0].grid->Kind(), "driscoll_healy");
  EXPECT_EQ(read.Parts()[1].grid->Kind(), "gauss_legendre");
  EXPECT_EQ(read.Parts()[2].grid->Kind(), "equiangular");
  EXPECT_TRUE(read.FieldOn(0, "rho").IsRadial());
  EXPECT_FALSE(read.FieldOn(1, "rho").IsRadial());
  EXPECT_EQ(read.FieldOn(1, "vsv").PerturbationKind(), "relative");
  EXPECT_FALSE(read.FieldOn(0, "vsv").PerturbationKind());
  EXPECT_EQ(read.FieldOn(1, "viscosity").PerturbationKind(), "log");
  EXPECT_EQ(read.FieldOn(2, "temperature").PerturbationKind(), "absolute");
  EXPECT_FALSE(read.HasField(0, "viscosity"));
  // a radial polynomial of degree below ngll is read exactly anywhere
  EXPECT_NEAR(read.FieldOn(0, "rho").AsScalar().Value(0.1234, 1.0, 2.0), Rho0(0.1234), 1e-13);
  // the stack: each surface moves by its relief, along its radius
  const auto mapping = read.Mapping();
  EXPECT_EQ(mapping.GetForm(), pm::ModelMapping::Form::Stack);
  const auto grid = pm::GaussLegendre(8);
  for (const auto& [radius, relief] :
       {std::pair{0.3, &Relief1}, std::pair{1.0, &Relief3}}) {
    for (const auto t : grid.Colatitudes()) {
      for (const auto p : grid.Longitudes()) {
        const auto u = mapping.Displacement(pm::CartesianPoint(radius, t, p));
        const auto want = pm::ToCartesian({relief(t, p), 0.0, 0.0}, t, p);
        for (auto i = 0; i < 3; ++i) EXPECT_NEAR(u[i], want[i], 1e-15);
      }
    }
  }
  const auto u = mapping.Displacement(pm::CartesianPoint(0.6, 1.0, 2.0));
  for (auto i = 0; i < 3; ++i) EXPECT_EQ(u[i], 0.0);
}

TEST(Writer, WritesANonSiModelWithADisplacementOnACustomGrid) {
  const auto path = Written("written_custom.nc", CustomModel());
  const auto read = pm::ReadModelFile(path);
  EXPECT_EQ(read.GetScales().Length(), 6.371e6);
  EXPECT_EQ(read.GetScales().Temperature(), 1000.0);
  EXPECT_EQ(read.Rtol(), 1e-8);
  EXPECT_EQ(read.LayerNames()[1], "upper");
  EXPECT_EQ(read.InterfaceNames()[0], "inner");
  EXPECT_FALSE(read.InterfaceNames()[1]);
  EXPECT_EQ(read.InterfaceRadii(), (std::vector<double>{0.2, 0.55, 1.0}));
  EXPECT_EQ(read.Constants().at("G").value, 1.25);
  EXPECT_EQ(read.Constants().at("G").dimensions, pm::units::GRAVITATIONAL_CONSTANT);
  EXPECT_EQ(read.Meta().at("iteration"), pm::netcdf::Attribute{std::vector<long long>{12}});
  EXPECT_EQ(read.Meta().at("code"), pm::netcdf::Attribute{std::string{"planetmodel tests"}});
  EXPECT_EQ(read.Meta().at("misfit"), pm::netcdf::Attribute{std::vector<double>{0.25}});
  EXPECT_EQ(read.Parts()[0].grid->Kind(), "custom");
  const auto& rho = read.Parts()[0].variables.at("rho");
  EXPECT_TRUE(rho.IsLateral());
  EXPECT_EQ(rho.Numbers("radial_layers"), (std::vector<double>{0.0}));
  EXPECT_EQ(rho.Text("units"), pm::units::ScaledUnit(pm::units::DENSITY, read.GetScales()));
  EXPECT_NE(rho.Text("units"), "kg m-3");
  ExpectReadsBack(
      read, {{0, "rho", Frame::Spherical, Scalar([](double r, double, double) { return 5.5 - r; })},
             {1, "rho", Frame::Spherical,
              Scalar([](double r, double t, double) { return 3.3 + 0.1 * r * std::cos(t); })}});
  EXPECT_TRUE(read.FieldOn(0, "rho").IsRadial());
  const auto mapping = read.Mapping();
  EXPECT_EQ(mapping.GetForm(), pm::ModelMapping::Form::Field);
  for (auto layer = 0; layer < 2; ++layer) {
    const auto points = read.PointsOf(layer);
    for (auto i = std::size_t{0}; i < points.Size(); ++i) {
      const auto at = points.At(i);
      const auto X = pm::CartesianPoint(at.r, at.theta, at.phi);
      const auto u = mapping.Displacement(X);
      const auto want = Swell(X);
      for (auto c = 0; c < 3; ++c) ASSERT_NEAR(u[c], want[c], 1e-15) << i;
    }
  }
}

TEST(Writer, WritesTensorsInEitherFrameOnEquiangularMidpoints) {
  const auto path = Written("written_tensors.nc", TensorModel());
  const auto read = pm::ReadModelFile(path);
  ExpectReadsBack(
      read, {{0, "stress", Frame::Cartesian,
              [](double r, double t, double p) { return Stress(pm::CartesianPoint(r, t, p)); }},
             {0, "elastic_moduli", Frame::Cartesian,
              [](double r, double t, double p) { return Elastic(pm::CartesianPoint(r, t, p)); }},
             {0, "elastic_moduli", Frame::Spherical, [](double r, double t, double p) {
                return ToSphericalStored(Elastic(pm::CartesianPoint(r, t, p)), 4, t, p);
              }}});
  const auto& elastic = read.Parts()[0].variables.at("elastic_moduli");
  EXPECT_EQ(elastic.dimensions, (std::vector<std::string>{"node", "latitude", "longitude",
                                                          "voigt_row", "voigt_column"}));
  EXPECT_EQ(elastic.Text("lateral_interpolation"), "cubic");
  EXPECT_EQ(elastic.Text("long_name"), pm::VocabularySpec("elastic_moduli")->meaning);
}

TEST(Writer, WritesFieldsOnInterfacesThatReadBackToThemselves) {
  auto file = pm::ModelFile(pm::Skeleton({0.0, 0.4, 1.0}));
  file.AddPart(0, 1, pm::ElementEdges(file.GetSkeleton(), 0, 1, 0.2), 3, pm::GaussLegendre(4));
  file.AddField("rho", {{0, LayerSource::Radial([](double r) { return 5.0 - r; })},
                        {1, LayerSource::Radial([](double r) { return 3.0 - r; })}});
  const auto kernel = pm::FieldSpec{.character = pm::SCALAR,
                                    .dimensions = pm::Dimensions{.length = -2},
                                    .meaning = "a shape kernel"};
  const auto shift = pm::FieldSpec{.character = pm::SCALAR, .dimensions = pm::units::LENGTH,
                                   .meaning = "the shift of the interface"};
  auto k = [](double, double t, double p) {
    return std::sin(t) * std::sin(t) * std::cos(2.0 * p) + 0.3 * std::cos(t);
  };
  auto h = [](double t, double p) {
    return 0.01 * std::cos(t) * std::cos(t) + 0.002 * std::sin(t) * std::sin(p);
  };
  EXPECT_THROW(file.AddSurfaceField("K_shape", kernel, {{1, LayerSource::Lateral(k)}}), Error);
  EXPECT_THROW(static_cast<void>(file.SurfacePointsOf(0)), Error);
  file.SetSurfaceGrid(pm::Equiangular(9, 16, true));
  file.AddSurfaceField("K_shape", kernel, {{1, LayerSource::Lateral(k)}}, pm::LateralRule::Cubic);
  // values at the points the writer gives, on an interface new to the group
  const auto points = file.SurfacePointsOf(0);
  EXPECT_EQ(points.radii, (std::vector<double>{0.4}));
  auto values = std::vector<double>{};
  for (auto i = std::size_t{0}; i < points.Size(); ++i) {
    const auto at = points.At(i);
    values.push_back(h(at.theta, at.phi));
  }
  file.AddSurfaceField("shift", shift,
                       {{0, LayerSource::LateralValues(values, Frame::Spherical)},
                        {1, LayerSource::Lateral([&h](double, double t, double p) {
                           return 2.0 * h(t, p);
                         })}});
  EXPECT_EQ(file.Surfaces()->interfaces, (std::vector<int>{0, 1}));
  EXPECT_EQ(file.SurfaceNames(0), (std::vector<std::string>{"shift"}));
  // the refusals
  EXPECT_THROW(file.SetSurfaceGrid(pm::GaussLegendre(8)), Error);
  EXPECT_THROW(file.AddSurfaceField("rho", kernel, {{0, LayerSource::Lateral(k)}}), Error);
  EXPECT_THROW(file.AddSurfaceField("shift", kernel, {{0, LayerSource::Lateral(k)}}), Error);
  EXPECT_THROW(file.AddField("K_shape", kernel, {{0, LayerSource::Lateral(k)}}), Error);
  EXPECT_THROW(file.AddSurfaceField("K_other", kernel, {{2, LayerSource::Lateral(k)}}), Error);
  EXPECT_THROW(file.AddSurfaceField("K_other", kernel, {}), Error);
  EXPECT_THROW(file.AddSurfaceField("surface_interface", kernel, {{0, LayerSource::Lateral(k)}}),
               Error);
  EXPECT_THROW(file.AddSurfaceField("wind", {.character = pm::VECTOR},
                                    {{0, LayerSource::Lateral(k)}}),
               Error);
  const auto path = Written("written_surfaces.nc", file);
  const auto read = pm::ReadModelFile(path);
  EXPECT_EQ(read.SurfaceNames(1), (std::vector<std::string>{"K_shape", "shift"}));
  EXPECT_EQ(read.Surfaces()->variables.at("shift").Numbers("interfaces"),
            (std::vector<double>{0.0, 1.0}));
  EXPECT_EQ(read.Surfaces()->variables.at("K_shape").Numbers("interfaces"),
            (std::vector<double>{1.0}));
  const auto& grid = *read.Surfaces()->grid;
  const auto kernel_read = read.SurfaceFieldOn(1, "K_shape");
  const auto inner = read.SurfaceFieldOn(0, "shift");
  const auto outer = read.SurfaceFieldOn(1, "shift");
  // the grid read back is the written one through degrees
  for (const auto t : grid.Colatitudes()) {
    for (const auto p : grid.Longitudes()) {
      EXPECT_NEAR(kernel_read.Value(t, p), k(1.0, t, p), 1e-13);
      EXPECT_NEAR(inner.Value(t, p), h(t, p), 1e-13);
      EXPECT_NEAR(outer.Value(t, p), 2.0 * h(t, p), 1e-13);
    }
  }
}

TEST(Writer, PointsComeInTheOrderTheValuesFollow) {
  auto file = pm::ModelFile(pm::Skeleton({0.0, 1.0}));
  file.AddPart(0, 0, {0.0, 0.5, 1.0}, 2, pm::GaussLegendre(1));
  const auto points = file.PointsOf(0);
  EXPECT_EQ(points.radii, (std::vector<double>{0.0, 0.5, 0.5, 1.0}));
  EXPECT_TRUE(points.poles);
  EXPECT_EQ(points.Size(), 4u * (2u * 3u + 2u));
  const auto& g = points.grid;
  const auto second = points.At(1);
  EXPECT_EQ(second.r, 0.0);
  EXPECT_EQ(second.theta, g.Colatitudes()[0]);
  EXPECT_EQ(second.phi, g.Longitudes()[1]);
  const auto seventh = points.At(6);
  EXPECT_EQ(seventh.r, 0.5);
  EXPECT_EQ(seventh.theta, g.Colatitudes()[0]);
  EXPECT_EQ(points.At(24).r, 0.0);
  EXPECT_EQ(points.At(24).theta, 0.0);
  EXPECT_EQ(points.At(25).theta, pi);
  EXPECT_EQ(points.At(31).r, 1.0);
  // a source of values and one of a function make one file
  const auto f = Scalar([](double r, double t, double p) { return r + std::cos(t) * std::sin(p); });
  auto by_values = file;
  by_values.AddField("vp",
                     {{0, LayerSource::LateralValues(AtPoints(points, f), Frame::Spherical)}});
  file.AddField("vp", {{0, LayerSource::Lateral([](double r, double t, double p) {
                           return r + std::cos(t) * std::sin(p);
                         })}});
  EXPECT_EQ(by_values.Parts()[0].variables.at("vp").values,
            file.Parts()[0].variables.at("vp").values);
  EXPECT_EQ(by_values.Parts()[0].variables.at("vp_poles").values,
            file.Parts()[0].variables.at("vp_poles").values);
}

TEST(Writer, RefusesNamesTheFormatKeepsAndSpecsTheVocabularyStatesOtherwise) {
  auto file = pm::ModelFile(pm::Skeleton({0.0, 1.0}));
  file.AddPart(0, 0, {0.0, 1.0}, 3, pm::GaussLegendre(2));
  const auto one = std::map<int, LayerSource>{{0, LayerSource::Radial([](double) { return 1.0; })}};
  const auto scalar = pm::FieldSpec{.dimensions = pm::units::DIMENSIONLESS};
  for (const auto* name :
       {"radius", "displacement", "layer_part", "q_poles", "q_perturbation", ""}) {
    EXPECT_THROW(file.AddField(name, scalar, one), Error) << name;
  }
  EXPECT_THROW(file.AddField("rho", scalar, one), Error);
  EXPECT_THROW(
      file.AddField("vs", {.character = pm::SCALAR, .dimensions = pm::units::DENSITY}, one), Error);
  EXPECT_THROW(file.AddField("nonesuch", one), Error);
  EXPECT_THROW(file.AddField("twisted", {.character = pm::Character(3)}, one), Error);
  file.AddField("vs", one);
  EXPECT_THROW(file.AddField("vs", one), Error);
  EXPECT_EQ(file.Parts()[0].variables.at("vs").Text("units"), "m s-1");
}

TEST(Writer, CitesEachWorkOnceOnALineOfItsOwn) {
  auto file = pm::ModelFile(pm::Skeleton({0.0, 1.0}));
  EXPECT_TRUE(file.Citation().empty());
  file.SetCitation({" A. ", "", "B.", "A.", "\tC.\n"});
  EXPECT_EQ(file.Citation(), (std::vector<std::string>{"A.", "B.", "C."}));
  EXPECT_THROW(file.SetCitation({"A.\nB."}), Error);
  EXPECT_EQ(file.Citation(), (std::vector<std::string>{"A.", "B.", "C."}));
  file.SetCitation({});
  EXPECT_TRUE(file.Citation().empty());
}

TEST(Writer, RefusesSourcesThatDoNotFitTheirLayers) {
  auto file = pm::ModelFile(pm::Skeleton({0.0, 0.5, 1.0}));
  const auto& sk = file.GetSkeleton();
  EXPECT_THROW(file.AddPart(1, 1, {0.5, 1.0}, 3, pm::GaussLegendre(2)), Error);
  EXPECT_THROW(file.AddPart(0, 1, {0.0, 0.3, 1.0}, 3, pm::GaussLegendre(2)), Error);
  EXPECT_THROW(file.AddPart(0, 0, {0.0, 0.6}, 3, pm::GaussLegendre(2)), Error);
  EXPECT_THROW(file.AddPart(0, 0, {0.0, 0.5}, 1, pm::GaussLegendre(2)), Error);
  file.AddPart(0, 0, {0.0, 0.25, 0.5 - 1e-12}, 3, pm::GaussLegendre(2));
  EXPECT_EQ(file.Parts()[0].element_bounds.back(), 0.5);
  const auto lateral = LayerSource::Lateral([](double r, double, double) { return r; });
  EXPECT_THROW(file.AddField("vs", {{1, lateral}}), Error);
  EXPECT_THROW(file.AddField("vs", {{0, LayerSource::RadialValues({1.0, 2.0})}}), Error);
  EXPECT_THROW(file.AddField("vs", {{0, LayerSource::LateralValues({1.0, 2.0}, Frame::Spherical)}}),
               Error);
  EXPECT_THROW(
      file.AddField(
          "vs", {{0, LayerSource::Lateral([](double, double, double) { return std::nan(""); })}}),
      Error);
  EXPECT_THROW(file.AddField("flow", {.character = pm::VECTOR}, {{0, lateral}}), Error);
  // layer 1 has no part yet, and the field is refused whole
  EXPECT_THROW(file.AddField("vp", {{0, lateral}, {1, lateral}}), Error);
  EXPECT_FALSE(file.Parts()[0].variables.contains("vp"));
  file.AddField("vs", {{0, lateral}}, {.lateral = pm::LateralRule::Linear});
  auto thin = pm::ModelFile(pm::Skeleton({0.0, 1.0}));
  thin.AddPart(0, 0, {0.0, 1.0}, 2, pm::AngularGrid({0.5, 1.5}, {0.0, 3.0}));
  EXPECT_THROW(thin.AddField("vs", {{0, lateral}}, {.lateral = pm::LateralRule::Cubic}), Error);
  // perturbations
  file.AddPart(1, 1, pm::ElementEdges(sk, 1, 1, 0.25), 3, pm::GaussLegendre(3));
  const auto radial = LayerSource::Radial([](double r) { return 1.0 + r; });
  EXPECT_THROW(
      file.AddPerturbedField("vp", {{0, radial}}, {.kind = "squared", .layers = {{0, lateral}}}),
      Error);
  EXPECT_THROW(
      file.AddPerturbedField("vp", {{0, lateral}}, {.kind = "log", .layers = {{0, lateral}}}),
      Error);
  EXPECT_THROW(
      file.AddPerturbedField("vp", {{0, radial}}, {.kind = "log", .layers = {{1, lateral}}}),
      Error);
  EXPECT_THROW(file.AddPerturbedField("flow", {.character = pm::VECTOR}, {{0, radial}},
                                      {.kind = "log", .layers = {{0, lateral}}}),
               Error);
  file.AddPerturbedField("vp", {{0, radial}, {1, radial}},
                         {.kind = "log", .layers = {{1, lateral}}});
  EXPECT_FALSE(file.Parts()[0].variables.contains("vp_perturbation"));
  EXPECT_EQ(file.Parts()[1].variables.at("vp_perturbation").Text("units"), "1");
  EXPECT_EQ(file.Parts()[1].variables.at("vp").Text("perturbation_kind"), "log");
  // the mapping, once, and as a field only on one grid with every layer
  const auto still =
      LayerSource::Radial([](double, std::span<double> out) { std::ranges::fill(out, 0.0); });
  EXPECT_THROW(file.SetDisplacement({{0, still}, {1, still}}), Error) << "two grids";
  EXPECT_THROW(file.SetStack({0.1, 1.0}, pm::GaussLegendre(4), {std::nullopt, std::nullopt}),
               Error);
  EXPECT_THROW(file.SetStack({0.0, 1.0}, pm::GaussLegendre(4), {std::nullopt}), Error);
  file.SetStack({0.0, 1.0}, pm::GaussLegendre(4), {std::nullopt, std::nullopt});
  EXPECT_THROW(file.SetStack({0.0, 1.0}, pm::GaussLegendre(4), {std::nullopt, std::nullopt}),
               Error);
  auto one_grid = pm::ModelFile(pm::Skeleton({0.0, 0.5, 1.0}));
  one_grid.AddPart(0, 0, {0.0, 0.5}, 3, pm::GaussLegendre(4));
  EXPECT_THROW(one_grid.SetDisplacement({{0, still}, {1, still}}), Error);
  one_grid.AddPart(1, 1, {0.5, 1.0}, 3, pm::GaussLegendre(4));
  EXPECT_THROW(one_grid.SetDisplacement({{0, still}}), Error);
  one_grid.SetDisplacement({{0, still}, {1, still}});
  EXPECT_THROW(one_grid.SetStack({0.0, 1.0}, pm::GaussLegendre(4), {std::nullopt, std::nullopt}),
               Error);
}

TEST(Writer, RefusesWhatTheReaderWouldRefuseBeforeWritingAnything) {
  const auto dir = pm::testing::WrittenDir() / "refused";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto path = dir / "model.nc";
  // a layer no part holds
  auto partial = pm::ModelFile(pm::Skeleton({0.0, 0.5, 1.0}));
  partial.AddPart(0, 0, {0.0, 0.5}, 3, pm::GaussLegendre(2));
  try {
    pm::WriteModelFile(path, partial);
    FAIL();
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "layer_part");
  }
  EXPECT_TRUE(std::filesystem::is_empty(dir));
  // a vector whose Cartesian components at a pole node are not one value
  auto torn = pm::ModelFile(pm::Skeleton({0.0, 1.0}));
  torn.AddPart(0, 0, {0.0, 1.0}, 3, pm::DriscollHealy(3));
  torn.AddField("flow", {.character = pm::VECTOR},
                {{0, LayerSource::Lateral(
                         [](double, double, double p, std::span<double> out) {
                           out[0] = std::cos(p);
                           out[1] = out[2] = 0.0;
                         },
                         Frame::Cartesian)}});
  try {
    pm::WriteModelFile(path, torn);
    FAIL() << "a torn pole was written";
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "part_0/flow");
  }
  EXPECT_TRUE(std::filesystem::is_empty(dir));
}

TEST(Writer, AFailedWriteLeavesNoFileAndAnExistingOneStanding) {
  const auto dir = pm::testing::WrittenDir() / "failed";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir / "occupied");
  auto file = pm::ModelFile(pm::Skeleton({0.0, 1.0}));
  file.AddPart(0, 0, {0.0, 1.0}, 3, pm::GaussLegendre(2));
  file.AddField("vs", {{0, LayerSource::Radial([](double r) { return 1.0 + r; })}});
  // a directory where the file would go
  EXPECT_THROW(pm::WriteModelFile(dir / "occupied", file), Error);
  // a directory that is not there
  EXPECT_THROW(pm::WriteModelFile(dir / "missing" / "model.nc", file), Error);
  auto left = std::vector<std::string>{};
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    left.push_back(entry.path().filename().string());
  }
  EXPECT_EQ(left, (std::vector<std::string>{"occupied"}));
  EXPECT_TRUE(std::filesystem::is_empty(dir / "occupied"));
  // a file that stands until a good one replaces it
  const auto path = dir / "model.nc";
  pm::WriteModelFile(path, file);
  auto bytes = [&path] {
    auto in = std::ifstream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
  };
  const auto before = bytes();
  auto broken = pm::ModelFile(pm::Skeleton({0.0, 0.5, 1.0}));
  broken.AddPart(0, 0, {0.0, 0.5}, 3, pm::GaussLegendre(2));
  EXPECT_THROW(pm::WriteModelFile(path, broken), Error);
  EXPECT_EQ(bytes(), before);
  auto again = file;
  again.SetName("again");
  pm::WriteModelFile(path, again);
  EXPECT_EQ(pm::ReadModelFile(path).Name(), "again");
}
