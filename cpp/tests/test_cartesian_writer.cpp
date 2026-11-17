// The flat model file's writer: every flat fixture read and written back
// to the same numbers; models built here, over a box and over a rectangle,
// with fields of every rank from sources of every kind in either frame,
// perturbed fields, a viscosity in its logarithm, the vertical stretch as
// a stack of reliefs and as a displacement, and fields on interfaces, each
// reading back to its sources at the nodes and left for the Python to
// read; and the refusals, before anything is written.
#include "planetmodel/cartesian/modelfile.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <numbers>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "fixtures.hpp"

namespace {

namespace pm = planetmodel;
namespace flat = planetmodel::cartesian;
using flat::Frame;
using flat::LayerSource;
using pm::Error;

constexpr auto pi = std::numbers::pi;

/// A known field: its components in `frame` at a point.
using Components = std::function<std::vector<double>(double depth, double x, double y)>;

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
/// to rounding.
void ExpectReadsBack(const flat::ModelFile& file, const std::vector<Known>& known) {
  for (const auto& k : known) {
    SCOPED_TRACE(std::format("{} on layer {}", k.name, k.layer));
    const auto field = file.FieldOn(k.layer, k.name);
    const auto points = file.PointsOf(k.layer);
    for (auto i = std::size_t{0}; i < points.Size(); ++i) {
      const auto at = points.At(i);
      const auto want = k.f(at.depth, at.x, at.y);
      const auto got = field.Evaluate(at.depth, at.x, at.y, k.frame);
      ASSERT_EQ(got.size(), want.size());
      const auto scale = std::max(1.0, Largest(want));
      for (auto c = std::size_t{0}; c < want.size(); ++c) {
        ASSERT_NEAR(got[c], want[c], 1e-12 * scale)
            << "at (" << at.depth << ", " << at.x << ", " << at.y << ") [" << c << "]";
      }
    }
  }
}

/// The expectations the Python is held to for a flat file the C++ wrote,
/// from the C++'s own reading of it: `value` lines for every field on
/// every layer at points between the nodes, in both frames, `displacement`
/// lines for the mapping, and `surface` lines for every field on every
/// interface, as `make_fixtures.py` writes them for its own.
void WriteExpect(const std::filesystem::path& path) {
  const auto file = flat::ReadModelFile(path);
  auto out = std::ofstream(path.parent_path() / (path.stem().string() + ".expect"));
  const auto& domain = file.GetDomain();
  const auto x0 = domain.Extent(0)[0];
  const auto lx = domain.Length(0);
  const auto y0 = domain.IsBox() ? domain.Extent(1)[0] : 0.0;
  const auto ly = domain.IsBox() ? domain.Length(1) : 1.0;
  const auto places = {std::tuple{0.13, -0.2, 0.1}, std::tuple{0.41, 0.31, -0.45},
                       std::tuple{0.77, 1.0, 0.62}, std::tuple{0.96, 1.9, 2.2}};
  const auto nlayers = static_cast<int>(file.GetSkeleton().NLayers());
  for (auto layer = 0; layer < nlayers; ++layer) {
    const auto [lo, hi] = file.GetSkeleton().Interval(static_cast<std::size_t>(layer));
    for (const auto& name : file.Names(layer)) {
      const auto field = file.FieldOn(layer, name);
      for (const auto& [f, a, b] : places) {
        const auto depth = lo + f * (hi - lo);
        const auto x = x0 + a * lx;
        const auto y = y0 + b * ly;
        for (const auto& [frame, tag] :
             {std::pair{Frame::Local, "l"}, std::pair{Frame::Cartesian, "c"}}) {
          out << std::format("value {} {} {} {} {} {}", layer, name, tag, depth, x, y);
          for (const auto v : field.Evaluate(depth, x, y, frame)) out << std::format(" {}", v);
          out << "\n";
        }
      }
    }
  }
  const auto mapping = file.Mapping();
  if (mapping.GetForm() != flat::VerticalStretch::Form::Identity) {
    const auto& b = file.GetSkeleton().Boundaries();
    for (const auto& [f, a, c] : {std::tuple{0.0, 0.4, 1.0}, std::tuple{0.21, 1.1, -2.0},
                                  std::tuple{0.5, -0.3, 0.7}, std::tuple{0.83, 0.9, 0.3},
                                  std::tuple{1.0, 0.0, 0.0}, std::tuple{0.999, 0.4, 0.9}}) {
      const auto X = pm::Vector3{x0 + a * lx, y0 + c * ly,
                                 -(b.front() + f * (b.back() - b.front()))};
      const auto u = mapping.Displacement(X);
      out << std::format("displacement {} {} {} {} {} {}\n", X[0], X[1], X[2], u[0], u[1], u[2]);
    }
  }
  for (auto k = 0; k < static_cast<int>(file.InterfaceDepths().size()); ++k) {
    for (const auto& name : file.SurfaceNames(k)) {
      const auto f = file.SurfaceFieldOn(k, name);
      for (const auto& [unused, a, b] : places) {
        static_cast<void>(unused);
        const auto x = x0 + a * lx;
        const auto y = y0 + b * ly;
        out << std::format("surface {} {} {} {} {}\n", k, name, x, y, f.Value(x, y));
      }
    }
  }
}

/// Written beside the others, read back, its expectations written.
std::filesystem::path Written(const std::string& name, const flat::ModelFile& file) {
  const auto path = pm::testing::WrittenDir() / name;
  flat::WriteModelFile(path, file);
  WriteExpect(path);
  return path;
}

// -- known fields, periodic over the box of the models built here ------------

constexpr auto LX = 500e3;  // the extent along x: [-50, 450] km
constexpr auto LY = 300e3;

double Wave(double x, double y) {
  return std::sin(2.0 * pi * x / LX) + 0.5 * std::cos(2.0 * pi * y / LY + 0.3);
}

std::vector<double> Flow(double depth, double x, double y) {
  // Cartesian (x, y, z)
  return {1.0 + Wave(x, y), 2e-5 * depth, std::cos(2.0 * pi * x / LX) - 1e-5 * depth};
}

std::vector<double> Stress(double depth, double x, double y) {
  // Voigt (11, 22, 33, 23, 13, 12), local: 1 is up
  const auto w = Wave(x, y);
  return {1.0 + w, 2.0 + 1e-5 * depth, 3.0 - w, 0.5 * w, 0.2 + 1e-6 * depth * w, 0.1 * w};
}

double Relief(double x, double y) { return 1.5e3 * Wave(x, y); }

flat::Domain Box() { return flat::Domain::Box({0.0, 400e3}, {0.0, 300e3}, {50e3, 0.0}); }

/// A source's values at the points of a layer, from a function.
std::vector<double> AtPoints(const flat::LayerPoints& points, const Components& f) {
  auto out = std::vector<double>{};
  for (auto i = std::size_t{0}; i < points.Size(); ++i) {
    const auto at = points.At(i);
    const auto v = f(at.depth, at.x, at.y);
    out.insert(out.end(), v.begin(), v.end());
  }
  return out;
}

/// The box model: SI, a crust over a lithosphere over an asthenosphere in
/// two parts on two grids, fields of every rank from every kind of source
/// in either frame, a perturbed field, the vertical stretch a stack of
/// reliefs, and fields on two interfaces.
flat::ModelFile BoxModel() {
  auto file = flat::ModelFile(pm::Skeleton({0.0, 30e3, 100e3, 200e3}), Box());
  const auto& sk = file.GetSkeleton();
  file.SetName("a box built in C++");
  file.SetWrittenBy("planetmodel C++ tests");
  file.SetCitation({"A. Author (2026).  A written box.", "B. Author (2026).  Its source."});
  file.SetLayerName(0, "crust");
  file.SetLayerName(2, "asthenosphere");
  file.SetInterfaceName(0, "surface");
  file.SetInterfaceName(1, "moho");
  file.SetConstant("g", {.value = 9.8, .dimensions = pm::units::GRAVITY,
                         .meaning = "a prescribed gravity"});
  file.SetMeta("run", std::string{"test"});
  file.SetMeta("iteration", 7);
  file.AddPart(0, 1, pm::ElementEdges(sk, 0, 1, 20e3), 4, flat::Grid(Box(), 12, 8));
  file.AddPart(2, 2, pm::ElementEdges(sk, 2, 2, 50e3), 3, flat::Grid(Box(), 9, 6));
  // a density of depth alone in the crust and lateral below, by the cubic rule
  file.AddField("rho",
                {{0, LayerSource::Depth([](double d) { return 2800.0 + 1e-3 * d; })},
                 {1, LayerSource::Lateral([](double d, double x, double y) {
                    return 3300.0 * (1.0 + 1e-7 * d * Wave(x, y));
                  })},
                 {2, LayerSource::DepthValues(std::vector<double>(
                         file.PointsOf(2).depths.size(), 3400.0))}},
                {.lateral = pm::LateralRule::Cubic});
  // a vector given in Cartesian components, stored in the local frame
  file.AddField("flow", {.character = pm::VECTOR, .meaning = "a flow"},
                {{1, LayerSource::Lateral(
                         [](double d, double x, double y, std::span<double> out) {
                           std::ranges::copy(Flow(d, x, y), out.begin());
                         },
                         Frame::Cartesian)}});
  // a stress given at the points, in the local frame, and of depth alone above
  file.AddField("stress", {.character = pm::STRESS, .dimensions = pm::units::MODULUS},
                {{0, LayerSource::Depth([](double d, std::span<double> out) {
                    std::ranges::copy(Stress(d, 0.0, 0.0), out.begin());
                  })},
                 {2, LayerSource::LateralValues(AtPoints(file.PointsOf(2), Stress))}},
                {.lateral = pm::LateralRule::Cubic});
  // a relative perturbation of a reference of depth alone
  const auto reference = LayerSource::Depth([](double d) { return 4500.0 + 2e-3 * d; });
  file.AddPerturbedField(
      "vs", {{0, reference}, {1, reference}, {2, reference}},
      {.kind = "relative",
       .layers = {{2, LayerSource::Lateral([](double d, double x, double y) {
                    return 0.02 * Wave(x, y) * (200e3 - d) / 100e3;
                  })}},
       .rules = {.lateral = pm::LateralRule::Cubic}});
  // topography: the surface raised to a relief, the moho to its root given
  // at the grid's nodes, the rest held
  const auto reliefs = flat::Grid(Box(), 16, 12);
  auto root = std::vector<double>{};
  for (auto i = std::size_t{0}; i < reliefs.Size(); ++i) {
    const auto [x, y] = reliefs.Node(i);
    root.push_back(-4.0 * Relief(x, y));
  }
  file.SetStack({0.0, 30e3, 100e3, 200e3}, reliefs,
                {LayerSource::Lateral([](double, double x, double y) { return Relief(x, y); }),
                 LayerSource::LateralValues(std::move(root)), std::nullopt, std::nullopt});
  // fields on interfaces
  file.SetSurfaceGrid(flat::Grid(Box(), 10, 6));
  file.AddSurfaceField("K_shape",
                       {.character = pm::SCALAR,
                        .dimensions = pm::Dimensions{.length = -2},
                        .meaning = "a shape kernel"},
                       {{0, LayerSource::Lateral([](double, double x, double y) {
                           return 1.0 + Wave(x, y);
                         })},
                        {2, LayerSource::Lateral([](double d, double x, double y) {
                           return 1e-5 * d - Wave(x, y);
                         })}});
  file.AddSurfaceField("shift", {.character = pm::SCALAR, .dimensions = pm::units::LENGTH},
                       {{1, LayerSource::Lateral([](double, double x, double y) {
                           return 30.0 * Wave(x, y);
                         })}},
                       pm::LateralRule::Cubic);
  return file;
}

std::vector<Known> BoxKnown() {
  auto scalar = [](auto f) {
    return Components{[f](double d, double x, double y) { return std::vector{f(d, x, y)}; }};
  };
  return {{0, "rho", Frame::Local, scalar([](double d, double, double) { return 2800.0 + 1e-3 * d; })},
          {1, "rho", Frame::Local,
           scalar([](double d, double x, double y) { return 3300.0 * (1.0 + 1e-7 * d * Wave(x, y)); })},
          {2, "rho", Frame::Cartesian, scalar([](double, double, double) { return 3400.0; })},
          {1, "flow", Frame::Cartesian, Flow},
          {1, "flow", Frame::Local,
           [](double d, double x, double y) {
             const auto v = Flow(d, x, y);
             return std::vector{v[2], v[0], v[1]};  // (up, x, y)
           }},
          {0, "stress", Frame::Local, [](double d, double, double) { return Stress(d, 0.0, 0.0); }},
          {2, "stress", Frame::Local, Stress},
          {2, "stress", Frame::Cartesian,
           [](double d, double x, double y) {
             const auto s = Stress(d, x, y);
             // xx = l22, yy = l33, zz = l11, yz = l13, xz = l12, xy = l23
             return std::vector{s[1], s[2], s[0], s[4], s[5], s[3]};
           }},
          {1, "vs", Frame::Local, scalar([](double d, double, double) { return 4500.0 + 2e-3 * d; })},
          {2, "vs", Frame::Local, scalar([](double d, double x, double y) {
             return (4500.0 + 2e-3 * d) * (1.0 + 0.02 * Wave(x, y) * (200e3 - d) / 100e3);
           })}};
}

/// The rectangle's model, in kilometres, one part, the mapping its upward
/// displacement at the nodes.
flat::ModelFile RectangleModel() {
  const auto domain = flat::Domain::Rectangle({0.0, 400.0}, 50.0);
  auto file = flat::ModelFile(pm::Skeleton({5.0, 30.0, 200.0}), domain,
                              pm::Scales({.length = 1e3, .mass = 1e12, .time = 3.15576e7}), 1e-8);
  const auto& sk = file.GetSkeleton();
  file.SetName("a rectangle built in C++");
  file.AddPart(0, 1, pm::ElementEdges(sk, 0, 1, 40.0), 5, flat::Grid(domain, 14));
  file.AddField("rho", {{0, LayerSource::Depth([](double d) { return 2.8 + 1e-3 * d; })},
                        {1, LayerSource::Lateral([](double d, double x, double) {
                           return 3.3 + 1e-3 * d * std::sin(2.0 * pi * x / 500.0);
                         })}});
  file.AddField("flow", {.character = pm::VECTOR},
                {{0, LayerSource::Lateral(
                         [](double d, double x, double, std::span<double> out) {
                           out[0] = 1e-2 * d;                       // up
                           out[1] = std::cos(2.0 * pi * x / 500.0); // x
                           out[2] = 0.5;                            // y
                         },
                         Frame::Local)}},
                {.lateral = pm::LateralRule::Cubic});
  auto w = [](double d, double x, double) {
    return 0.8 * std::sin(2.0 * pi * x / 500.0) * (200.0 - d) / 195.0;
  };
  file.SetDisplacement({{0, LayerSource::Lateral(w)}, {1, LayerSource::Lateral(w)}});
  return file;
}

}  // namespace

TEST(CartesianWriter, WritesBackWhatItReads) {
  PLANETMODEL_FIXTURE(probe, "flat_model_layered.nc");
  auto files = 0;
  for (const auto& entry : std::filesystem::directory_iterator(pm::testing::FixturesDir())) {
    const auto name = entry.path().filename().string();
    if (!name.starts_with("flat_model_") || entry.path().extension() != ".nc") continue;
    SCOPED_TRACE(name);
    const auto before = flat::ReadModelFile(entry.path());
    const auto path = pm::testing::WrittenDir() / ("flat_roundtrip_" + name.substr(5));
    flat::WriteModelFile(path, before);
    const auto after = flat::ReadModelFile(path);
    EXPECT_EQ(after.Schema(), before.Schema());
    EXPECT_EQ(after.Name(), before.Name());
    EXPECT_EQ(after.WrittenBy(), before.WrittenBy());
    EXPECT_EQ(after.Created(), before.Created());
    EXPECT_EQ(after.Citation(), before.Citation());
    EXPECT_EQ(after.GetScales(), before.GetScales());
    EXPECT_EQ(after.Rtol(), before.Rtol());
    EXPECT_EQ(after.Meta(), before.Meta());
    EXPECT_EQ(after.GetDomain(), before.GetDomain());
    EXPECT_EQ(after.GetSkeleton().Boundaries(), before.GetSkeleton().Boundaries());
    EXPECT_EQ(after.LayerNames(), before.LayerNames());
    EXPECT_EQ(after.InterfaceNames(), before.InterfaceNames());
    EXPECT_EQ(after.InterfaceDepths(), before.InterfaceDepths());
    ASSERT_EQ(after.Constants().size(), before.Constants().size());
    for (const auto& [cname, c] : before.Constants()) {
      EXPECT_EQ(after.Constants().at(cname).value, c.value);
      EXPECT_EQ(after.Constants().at(cname).dimensions, c.dimensions);
      EXPECT_EQ(after.Constants().at(cname).meaning, c.meaning);
    }
    auto same = [](const std::map<std::string, pm::FileVariable>& a,
                   const std::map<std::string, pm::FileVariable>& b) {
      ASSERT_EQ(a.size(), b.size());
      for (const auto& [vname, v] : b) {
        SCOPED_TRACE(vname);
        const auto& w = a.at(vname);
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
    };
    ASSERT_EQ(after.Parts().size(), before.Parts().size());
    for (auto k = std::size_t{0}; k < before.Parts().size(); ++k) {
      const auto& a = after.Parts()[k];
      const auto& b = before.Parts()[k];
      EXPECT_EQ(a.layers, b.layers);
      EXPECT_EQ(a.ngll, b.ngll);
      EXPECT_EQ(a.element_layer, b.element_layer);
      EXPECT_EQ(a.element_bounds, b.element_bounds);
      EXPECT_EQ(a.depth, b.depth);
      EXPECT_EQ(*a.grid, *b.grid);
      same(a.variables, b.variables);
    }
    EXPECT_EQ(after.Geometry().has_value(), before.Geometry().has_value());
    if (before.Geometry()) {
      EXPECT_EQ(after.Geometry()->depths, before.Geometry()->depths);
      EXPECT_EQ(*after.Geometry()->grid, *before.Geometry()->grid);
      EXPECT_EQ(after.Geometry()->values, before.Geometry()->values);
      EXPECT_EQ(after.Geometry()->lateral, before.Geometry()->lateral);
      EXPECT_EQ(after.Geometry()->between, before.Geometry()->between);
    }
    EXPECT_EQ(after.Surfaces().has_value(), before.Surfaces().has_value());
    if (before.Surfaces()) {
      EXPECT_EQ(after.Surfaces()->interfaces, before.Surfaces()->interfaces);
      EXPECT_EQ(*after.Surfaces()->grid, *before.Surfaces()->grid);
      same(after.Surfaces()->variables, before.Surfaces()->variables);
    }
    ++files;
  }
  EXPECT_GE(files, 11);
}

TEST(CartesianWriter, WritesABoxOfEveryKindOfFieldThatReadsBackToItsSources) {
  const auto built = BoxModel();
  const auto path = Written("flat_written_box.nc", built);
  const auto file = flat::ReadModelFile(path);
  EXPECT_EQ(file.Schema(), flat::MODEL_SCHEMA);
  EXPECT_EQ(file.Name(), "a box built in C++");
  EXPECT_EQ(file.WrittenBy(), "planetmodel C++ tests");
  EXPECT_NE(file.Created().find('T'), std::string::npos);
  EXPECT_EQ(file.Citation().size(), 2u);
  EXPECT_EQ(file.GetDomain(), Box());
  EXPECT_EQ(file.LayerNames(),
            (std::vector<std::optional<std::string>>{"crust", std::nullopt, "asthenosphere"}));
  EXPECT_EQ(file.InterfaceDepths(), (std::vector{0.0, 30e3, 100e3}));
  EXPECT_EQ(file.Constants().at("g").value, 9.8);
  EXPECT_EQ(file.Constants().at("g").dimensions, pm::units::GRAVITY);
  EXPECT_EQ(pm::detail::TextOf(file.Meta(), "run"), "test");
  ExpectReadsBack(file, BoxKnown());
  // what the format states of each variable
  const auto& rho = file.PartOf(1).variables.at("rho");
  EXPECT_EQ(rho.dimensions, (std::vector<std::string>{"node", "x", "y"}));
  EXPECT_EQ(rho.Numbers("radial_layers"), (std::vector<double>{0.0}));
  EXPECT_EQ(rho.Text("lateral_interpolation"), "cubic");
  EXPECT_EQ(rho.Text("coordinates"), "depth");
  EXPECT_EQ(file.PartOf(2).variables.at("rho").dimensions, (std::vector<std::string>{"node"}));
  EXPECT_EQ(file.PartOf(1).variables.at("flow").dimensions,
            (std::vector<std::string>{"node", "x", "y", "component"}));
  EXPECT_FALSE(file.HasField(0, "flow"));
  EXPECT_FALSE(file.HasField(1, "stress"));
  EXPECT_EQ(file.FieldOn(2, "vs").PerturbationKind(), "relative");
  EXPECT_FALSE(file.PartOf(0).variables.contains("vs_perturbation"));
  EXPECT_EQ(file.PartOf(2).variables.at("vs_perturbation").Text("units"), "1");
  // the stretch: each surface to its relief, linear in depth between
  const auto mapping = file.Mapping();
  EXPECT_EQ(mapping.GetForm(), flat::VerticalStretch::Form::Stack);
  const auto& grid = *file.Geometry()->grid;
  for (auto i = std::size_t{0}; i < grid.Size(); i += 7) {
    const auto [x, y] = grid.Node(i);
    EXPECT_NEAR(mapping.Height(0.0, x, y), Relief(x, y), 1e-12 * 3e3);
    EXPECT_NEAR(mapping.Height(30e3, x, y), -4.0 * Relief(x, y), 1e-12 * 12e3);
    EXPECT_NEAR(mapping.Height(15e3, x, y), -1.5 * Relief(x, y), 1e-12 * 12e3);
    EXPECT_NEAR(mapping.Height(65e3, x, y), -2.0 * Relief(x, y), 1e-12 * 12e3);
    EXPECT_EQ(mapping.Height(150e3, x, y), 0.0);
  }
  // the fields on interfaces, at the nodes of their grid
  EXPECT_EQ(file.Surfaces()->interfaces, (std::vector<int>{0, 1, 2}));
  EXPECT_EQ(file.SurfaceNames(1), (std::vector<std::string>{"shift"}));
  const auto& on = *file.Surfaces()->grid;
  const auto top = file.SurfaceFieldOn(0, "K_shape");
  const auto lab = file.SurfaceFieldOn(2, "K_shape");
  const auto shift = file.SurfaceFieldOn(1, "shift");
  for (auto i = std::size_t{0}; i < on.Size(); ++i) {
    const auto [x, y] = on.Node(i);
    EXPECT_NEAR(top.Value(x, y), 1.0 + Wave(x, y), 1e-13);
    EXPECT_NEAR(lab.Value(x, y), 1.0 - Wave(x, y), 1e-13);
    EXPECT_NEAR(shift.Value(x, y), 30.0 * Wave(x, y), 1e-12);
  }
  EXPECT_THROW(static_cast<void>(file.SurfaceFieldOn(1, "K_shape")), Error);
}

TEST(CartesianWriter, WritesARectangleWithItsDisplacementInItsOwnUnits) {
  const auto path = Written("flat_written_rectangle.nc", RectangleModel());
  const auto file = flat::ReadModelFile(path);
  EXPECT_EQ(file.GetDomain(), flat::Domain::Rectangle({0.0, 400.0}, 50.0));
  EXPECT_EQ(file.GetScales().Length(), 1e3);
  EXPECT_EQ(file.Rtol(), 1e-8);
  // a hollowed body: its top below the free surface
  EXPECT_EQ(file.InterfaceDepths(), (std::vector{5.0, 30.0}));
  EXPECT_EQ(file.PartOf(1).variables.at("rho").dimensions,
            (std::vector<std::string>{"node", "x"}));
  EXPECT_EQ(file.PartOf(0).variables.at("flow").Text("units"), "unknown");
  ExpectReadsBack(
      file, {{0, "rho", Frame::Local,
              [](double d, double, double) { return std::vector{2.8 + 1e-3 * d}; }},
             {1, "rho", Frame::Local,
              [](double d, double x, double) {
                return std::vector{3.3 + 1e-3 * d * std::sin(2.0 * pi * x / 500.0)};
              }},
             {0, "flow", Frame::Cartesian, [](double d, double x, double) {
                return std::vector{std::cos(2.0 * pi * x / 500.0), 0.5, 1e-2 * d};
              }}});
  const auto mapping = file.Mapping();
  EXPECT_EQ(mapping.GetForm(), flat::VerticalStretch::Form::Field);
  const auto& displacement = file.PartOf(0).variables.at("displacement");
  EXPECT_EQ(displacement.dimensions, (std::vector<std::string>{"node", "x"}));
  EXPECT_EQ(displacement.Text("positive"), "up");
  EXPECT_EQ(displacement.Text("lateral_interpolation"), "cubic");
  EXPECT_FALSE(displacement.attributes.contains("rank"));
  for (const auto layer : {0, 1}) {
    const auto points = file.PointsOf(layer);
    for (auto i = std::size_t{0}; i < points.Size(); ++i) {
      const auto at = points.At(i);
      const auto want = 0.8 * std::sin(2.0 * pi * at.x / 500.0) * (200.0 - at.depth) / 195.0;
      // the same at every y, the rectangle's body being invariant along it
      EXPECT_NEAR(mapping.Displacement({at.x, 1e6, -at.depth})[2], want, 1e-13);
    }
  }
  EXPECT_THROW(static_cast<void>(file.FieldOn(0, "displacement")), Error);
}

TEST(CartesianWriter, WritesAViscosityInItsLogarithmThatReadsBackExactly) {
  auto file = flat::ModelFile(pm::Skeleton({0.0, 0.3, 1.0}), flat::Domain::Rectangle({0.0, 1.0}));
  file.AddPart(0, 1, pm::ElementEdges(file.GetSkeleton(), 0, 1, 0.4), 3,
               flat::Grid(file.GetDomain(), 3));
  // exp of a quadratic in depth: three nodes an element read it exactly
  auto eta = [](double d) { return std::exp(46.0 + 8.0 * d - 5.0 * d * d); };
  file.AddField("viscosity",
                {{0, LayerSource::Depth([](double) { return 2e21; })}, {1, LayerSource::Depth(eta)}},
                {.radial = pm::RadialRule::LogPolynomial});
  const auto path = Written("flat_written_log.nc", file);
  const auto read = flat::ReadModelFile(path);
  EXPECT_EQ(read.Schema(), flat::MODEL_SCHEMA);  // the one schema, a rule in the logarithm or none
  const auto graded = read.FieldOn(1, "viscosity").AsScalar();
  for (const auto d : {0.3, 0.41, 0.65, 0.9, 1.0}) {
    EXPECT_NEAR(graded.Value(d, 0.2, 0.0), eta(d), 1e-13 * eta(d)) << d;
    EXPECT_NEAR(graded.Gradient(d, 0.2, 0.0)[0], eta(d) * (8.0 - 10.0 * d), 1e-11 * eta(d) * 8.0);
  }
  const auto constant = read.FieldOn(0, "viscosity").AsScalar();
  for (const auto d : {0.0, 0.1, 0.3}) EXPECT_EQ(constant.Value(d, 0.0, 0.0), 2e21);
  // a rule in the logarithm is a positive scalar's of depth alone
  auto refused = flat::ModelFile(pm::Skeleton({0.0, 1.0}), flat::Domain::Rectangle({0.0, 1.0}));
  refused.AddPart(0, 0, {0.0, 1.0}, 3, flat::Grid(refused.GetDomain(), 3));
  const auto log = pm::Rules{.radial = pm::RadialRule::LogLinear};
  EXPECT_THROW(refused.AddField("viscosity", {{0, LayerSource::Depth([](double d) { return d; })}},
                                log),
               Error);
  EXPECT_THROW(
      refused.AddField("viscosity",
                       {{0, LayerSource::Lateral([](double, double, double) { return 1.0; })}}, log),
      Error);
}

TEST(CartesianWriter, PointsComeInTheOrderTheValuesFollow) {
  const auto file = BoxModel();
  const auto points = file.PointsOf(2);
  const auto& part = file.PartOf(2);
  ASSERT_EQ(points.depths.size(), part.depth.size());
  EXPECT_EQ(points.Size(), part.depth.size() * 9 * 6);
  // depth by depth, x before y, y fastest
  const auto& grid = *part.grid;
  EXPECT_EQ(points.At(0).depth, 100e3);
  EXPECT_EQ(points.At(0).x, grid.X()[0]);
  EXPECT_EQ(points.At(1).y, grid.Y()[1]);
  EXPECT_EQ(points.At(6).x, grid.X()[1]);
  EXPECT_EQ(points.At(6).y, grid.Y()[0]);
  EXPECT_EQ(points.At(54).depth, part.depth[1]);
  // a rectangle's points have y = 0
  const auto rectangle = RectangleModel();
  const auto line = rectangle.PointsOf(0);
  EXPECT_EQ(line.Size(), line.depths.size() * 14);
  EXPECT_EQ(line.At(15).y, 0.0);
  EXPECT_EQ(line.At(15).x, rectangle.PartOf(0).grid->X()[1]);
  // an interface's points lie at its depth on the surface grid
  const auto on = file.SurfacePointsOf(1);
  EXPECT_EQ(on.depths, (std::vector{30e3}));
  EXPECT_EQ(on.Size(), 60u);
  EXPECT_THROW(static_cast<void>(file.SurfacePointsOf(3)), Error);
  EXPECT_THROW(static_cast<void>(rectangle.SurfacePointsOf(0)), Error);
}

TEST(CartesianWriter, RefusesSourcesThatDoNotFitTheirLayers) {
  const auto domain = flat::Domain::Box({0.0, 1.0}, {0.0, 1.0});
  const auto grid = flat::Grid(domain, 4, 4);
  auto file = flat::ModelFile(pm::Skeleton({0.0, 0.5, 1.0}), domain);
  const auto& sk = file.GetSkeleton();
  EXPECT_THROW(file.AddPart(1, 1, {0.5, 1.0}, 3, grid), Error);  // from the surface down
  EXPECT_THROW(file.AddPart(0, 1, {0.0, 0.3, 1.0}, 3, grid), Error);
  EXPECT_THROW(file.AddPart(0, 0, {0.0, 0.6}, 3, grid), Error);
  EXPECT_THROW(file.AddPart(0, 0, {0.0, 0.5}, 1, grid), Error);
  // every grid of a flat file covers the model's domain
  EXPECT_THROW(file.AddPart(0, 0, {0.0, 0.5}, 3,
                            flat::Grid(flat::Domain::Box({0.0, 2.0}, {0.0, 1.0}), 4, 4)),
               Error);
  EXPECT_THROW(file.AddPart(0, 0, {0.0, 0.5}, 3, flat::Grid(flat::Domain::Rectangle({0.0, 1.0}), 4)),
               Error);
  file.AddPart(0, 0, {0.0, 0.25, 0.5 - 1e-12}, 3, grid);
  EXPECT_EQ(file.Parts()[0].element_bounds.back(), 0.5);
  const auto lateral = LayerSource::Lateral([](double d, double, double) { return d; });
  EXPECT_THROW(file.AddField("vs", {{1, lateral}}), Error);
  EXPECT_THROW(file.AddField("vs", {{0, LayerSource::DepthValues({1.0, 2.0})}}), Error);
  EXPECT_THROW(file.AddField("vs", {{0, LayerSource::LateralValues({1.0, 2.0})}}), Error);
  EXPECT_THROW(
      file.AddField(
          "vs", {{0, LayerSource::Lateral([](double, double, double) { return std::nan(""); })}}),
      Error);
  EXPECT_THROW(file.AddField("flow", {.character = pm::VECTOR}, {{0, lateral}}), Error);
  EXPECT_THROW(file.AddField("vp", {{0, lateral}, {1, lateral}}), Error);
  EXPECT_FALSE(file.Parts()[0].variables.contains("vp"));
  // names the format keeps, and specs the vocabulary states otherwise
  for (const auto* name : {"depth", "x", "y", "displacement", "interface_depth", "a_perturbation",
                           "a_poles", ""}) {
    EXPECT_THROW(file.AddField(name, {.character = pm::SCALAR}, {{0, lateral}}), Error) << name;
  }
  EXPECT_NO_THROW(file.AddField("radius", {.character = pm::SCALAR}, {{0, lateral}}));
  EXPECT_THROW(file.AddField("vs", {.character = pm::VECTOR}, {{0, lateral}}), Error);
  EXPECT_THROW(file.AddField("nonesuch", {{0, lateral}}), Error);
  file.AddField("vs", {{0, lateral}});
  EXPECT_THROW(file.AddField("vs", {{0, lateral}}), Error);
  // the cubic rule on fewer than three nodes along an axis
  auto thin = flat::ModelFile(pm::Skeleton({0.0, 1.0}), domain);
  thin.AddPart(0, 0, {0.0, 1.0}, 2, flat::Grid(domain, 4, 2));
  EXPECT_THROW(thin.AddField("vs", {{0, lateral}}, {.lateral = pm::LateralRule::Cubic}), Error);
  EXPECT_NO_THROW(thin.AddField("vs", {{0, lateral}}, {.lateral = pm::LateralRule::Linear}));
  // perturbations
  file.AddPart(1, 1, pm::ElementEdges(sk, 1, 1, 0.25), 3, flat::Grid(domain, 5, 3));
  const auto radial = LayerSource::Depth([](double d) { return 1.0 + d; });
  EXPECT_THROW(
      file.AddPerturbedField("vp", {{0, radial}}, {.kind = "squared", .layers = {{0, lateral}}}),
      Error);
  EXPECT_THROW(
      file.AddPerturbedField("vp", {{0, lateral}}, {.kind = "log", .layers = {{0, lateral}}}),
      Error);
  EXPECT_THROW(
      file.AddPerturbedField("vp", {{0, radial}}, {.kind = "log", .layers = {{1, lateral}}}),
      Error);
  file.AddPerturbedField("vp", {{0, radial}, {1, radial}},
                         {.kind = "log", .layers = {{1, lateral}}});
  EXPECT_FALSE(file.Parts()[0].variables.contains("vp_perturbation"));
  EXPECT_EQ(file.Parts()[1].variables.at("vp").Text("perturbation_kind"), "log");
  // the mapping, once, and as a field only on one grid with every layer
  const auto still = LayerSource::Depth([](double) { return 0.0; });
  EXPECT_THROW(file.SetDisplacement({{0, still}, {1, still}}), Error) << "two grids";
  EXPECT_THROW(file.SetStack({0.1, 1.0}, grid, {std::nullopt, std::nullopt}), Error);
  EXPECT_THROW(file.SetStack({0.0, 1.0}, grid, {std::nullopt}), Error);
  EXPECT_THROW(file.SetStack({0.0, 1.0}, flat::Grid(flat::Domain::Box({0.0, 2.0}, {0.0, 1.0}), 4, 4),
                             {std::nullopt, std::nullopt}),
               Error);
  file.SetStack({0.0, 1.0}, grid, {std::nullopt, std::nullopt});
  EXPECT_THROW(file.SetStack({0.0, 1.0}, grid, {std::nullopt, std::nullopt}), Error);
  auto one_grid = flat::ModelFile(pm::Skeleton({0.0, 0.5, 1.0}), domain);
  one_grid.AddPart(0, 0, {0.0, 0.5}, 3, grid);
  EXPECT_THROW(one_grid.SetDisplacement({{0, still}, {1, still}}), Error);
  one_grid.AddPart(1, 1, {0.5, 1.0}, 3, grid);
  EXPECT_THROW(one_grid.SetDisplacement({{0, still}}), Error);
  one_grid.SetDisplacement({{0, still}, {1, still}});
  EXPECT_THROW(one_grid.SetStack({0.0, 1.0}, grid, {std::nullopt, std::nullopt}), Error);
  // fields on interfaces: a grid first, over the domain, on the skeleton's interfaces
  EXPECT_THROW(one_grid.AddSurfaceField("K", {.character = pm::SCALAR}, {{0, lateral}}), Error);
  EXPECT_THROW(one_grid.SetSurfaceGrid(flat::Grid(flat::Domain::Rectangle({0.0, 1.0}), 4)), Error);
  one_grid.SetSurfaceGrid(grid);
  EXPECT_THROW(one_grid.AddSurfaceField("K", {.character = pm::SCALAR}, {{2, lateral}}), Error);
  EXPECT_THROW(one_grid.AddSurfaceField("K", {.character = pm::VECTOR}, {{0, lateral}}), Error);
  one_grid.AddSurfaceField("K", {.character = pm::SCALAR}, {{1, lateral}});
  EXPECT_THROW(one_grid.SetSurfaceGrid(grid), Error);
  EXPECT_THROW(static_cast<void>(flat::ModelFile(pm::Skeleton({0.0, 1.0}), domain,
                                                 pm::Scales::SI(), 2.0)),
               Error);
}

TEST(CartesianWriter, RefusesWhatTheReaderWouldRefuseAndLeavesNoFile) {
  const auto dir = pm::testing::WrittenDir() / "flat_refused";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir / "occupied");
  const auto domain = flat::Domain::Rectangle({0.0, 1.0});
  // a layer no part holds
  auto partial = flat::ModelFile(pm::Skeleton({0.0, 0.5, 1.0}), domain);
  partial.AddPart(0, 0, {0.0, 0.5}, 3, flat::Grid(domain, 4));
  try {
    flat::WriteModelFile(dir / "model.nc", partial);
    FAIL();
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "layer_part");
  }
  auto file = flat::ModelFile(pm::Skeleton({0.0, 1.0}), domain);
  file.AddPart(0, 0, {0.0, 1.0}, 3, flat::Grid(domain, 4));
  file.AddField("vs", {{0, LayerSource::Depth([](double d) { return 1.0 + d; })}});
  // a directory where the file would go, and one that is not there
  EXPECT_THROW(flat::WriteModelFile(dir / "occupied", file), Error);
  EXPECT_THROW(flat::WriteModelFile(dir / "missing" / "model.nc", file), Error);
  auto left = std::vector<std::string>{};
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    left.push_back(entry.path().filename().string());
  }
  EXPECT_EQ(left, (std::vector<std::string>{"occupied"}));
  // a file that stands until a good one replaces it
  const auto path = dir / "model.nc";
  flat::WriteModelFile(path, file);
  auto bytes = [&path] {
    auto in = std::ifstream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
  };
  const auto before = bytes();
  EXPECT_THROW(flat::WriteModelFile(path, partial), Error);
  EXPECT_EQ(bytes(), before);
  auto again = file;
  again.SetName("again");
  flat::WriteModelFile(path, again);
  EXPECT_EQ(flat::ReadModelFile(path).Name(), "again");
  // the spherical reader refuses what the flat writer wrote, by its schema
  EXPECT_THROW(static_cast<void>(pm::ReadModelFile(path)), Error);
}

#if defined(__unix__) || defined(__APPLE__)
TEST(CartesianWriter, GivesTheFileThePermissionsAPlainOpenWould) {
  const auto domain = flat::Domain::Rectangle({0.0, 1.0});
  auto file = flat::ModelFile(pm::Skeleton({0.0, 1.0}), domain);
  file.AddPart(0, 0, {0.0, 1.0}, 3, flat::Grid(domain, 4));
  file.AddField("vs", {{0, LayerSource::Depth([](double d) { return 1.0 + d; })}});
  const auto dir = pm::testing::WrittenDir() / "permissions";
  std::filesystem::create_directories(dir);
  pm::testing::ExpectPlainPermissions(
      dir / "flat_model.nc",
      [&file](const std::filesystem::path& p) { flat::WriteModelFile(p, file); });
}
#endif
