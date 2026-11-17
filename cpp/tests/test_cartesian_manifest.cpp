// The flat manifest reader (schema 7): every flat manifest the Python
// writes read into the values it wrote, a box's and a rectangle's, with
// faults and without; the layers and interfaces by depth, the faces of the
// box, a fault's face edge and its four curve groups; each reader refusing
// the other's manifests by their schema; every fault the Python's
// validate_structure refuses refused at its place; and the rules of a
// fault's faces giving the Python's answers on flat meshes with faults,
// the face edge told from the tip by the faces of the box.
#include "planetmodel/cartesian/manifest.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/manifest.hpp"

namespace {

namespace flat = planetmodel::cartesian;
using flat::MeshManifest;
using planetmodel::Error;
using planetmodel::Json;
using planetmodel::MeshEdge;
using planetmodel::MeshTetrahedron;
using planetmodel::MeshTriangle;
using planetmodel::MeshVertex;

/// Whether reading `path` throws an Error at `where` saying `phrase`.
::testing::AssertionResult RefusedAt(const std::filesystem::path& path,
                                     const std::string& where, const std::string& phrase) {
  try {
    static_cast<void>(MeshManifest(path));
  } catch (const Error& e) {
    if (e.Source() != path.string()) {
      return ::testing::AssertionFailure() << "names " << e.Source() << ": " << e.what();
    }
    if (e.Where() != where) {
      return ::testing::AssertionFailure() << "at " << e.Where() << ", not " << where
                                           << ": " << e.what();
    }
    if (std::string{e.what()}.find(phrase) == std::string::npos) {
      return ::testing::AssertionFailure() << "without \"" << phrase << "\": " << e.what();
    }
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "accepted";
}

std::vector<int> Ints(const Json& array) {
  auto out = std::vector<int>{};
  for (const auto& v : array.AsArray()) out.push_back(v.AsInt());
  return out;
}

std::optional<int> OptionalInt(const Json& v) {
  return v.IsNull() ? std::nullopt : std::optional<int>{v.AsInt()};
}

/// The manifest's records against the document they were read from.
void ExpectAsWritten(const MeshManifest& m, const Json& doc, bool files = true) {
  EXPECT_EQ(m.Schema(), flat::MANIFEST_SCHEMA);
  const auto& mesh = doc.At("mesh");
  EXPECT_EQ(m.MeshFile(), m.Path().parent_path() / mesh.At("file").AsString());
  EXPECT_EQ(m.Format(), mesh.At("format").AsString());
  EXPECT_EQ(m.Nodes(), mesh.At("nodes").AsString());
  const auto& d = mesh.At("displacement");
  EXPECT_EQ(m.Displacement().has_value(), !d.IsNull());

  const auto& coordinates = doc.At("coordinates");
  ASSERT_EQ(m.Coordinates().size(), coordinates.Size());
  for (auto i = std::size_t{0}; i < coordinates.Size(); ++i) {
    EXPECT_EQ(m.Coordinates()[i], coordinates.At(i).AsString());
  }
  EXPECT_EQ(m.Dimension(), static_cast<int>(coordinates.Size()));
  EXPECT_EQ(m.IsRectangle(), coordinates.Size() == 2);

  const auto faulted = doc.Find("faults") != nullptr && doc.At("faults").Size() > 0;
  const auto& layers = doc.At("layers");
  ASSERT_EQ(m.Layers().size(), layers.Size());
  for (auto i = std::size_t{0}; i < layers.Size(); ++i) {
    const auto& l = m.Layers()[i];
    const auto& e = layers.At(i);
    if (faulted) {
      EXPECT_EQ(l.attributes, Ints(e.At("attributes")));
    } else {
      EXPECT_EQ(l.attributes, (std::vector<int>{l.attribute}));
    }
    for (const auto a : l.attributes) EXPECT_EQ(&m.LayerOf(a), &m.Layers()[i]);
    EXPECT_EQ(l.attribute, e.At("attribute").AsInt());
    EXPECT_EQ(l.name, e.At("name").AsString());
    EXPECT_EQ(l.depth_top, e.At("depth_top").AsNumber());
    EXPECT_EQ(l.depth_bottom, e.At("depth_bottom").AsNumber());
    EXPECT_TRUE(l.in_geometry);
    EXPECT_EQ(l.fluid.has_value(), !e.At("fluid").IsNull());
    EXPECT_EQ(m.Boundaries()[i], l.depth_top);
    EXPECT_EQ(m.Boundaries()[i + 1], l.depth_bottom);
    EXPECT_EQ(&m.LayerNamed(l.name), &l);
  }
  const auto& interfaces = doc.At("interfaces");
  ASSERT_EQ(m.Interfaces().size(), interfaces.Size());
  for (auto i = std::size_t{0}; i < interfaces.Size(); ++i) {
    const auto& f = m.Interfaces()[i];
    const auto& e = interfaces.At(i);
    EXPECT_EQ(f.attribute, e.At("attribute").AsInt());
    EXPECT_EQ(f.name, e.At("name").AsString());
    EXPECT_EQ(f.depth, e.At("depth").AsNumber());
    // the top of layer i, the deeper layer first
    EXPECT_EQ(f.depth, m.Layers()[i].depth_top);
    EXPECT_EQ(f.between_layers, (std::array<int, 2>{static_cast<int>(i),
                                                    static_cast<int>(i) - 1}));
    EXPECT_EQ(f.BelowAttribute(), static_cast<int>(i) + 1);
    EXPECT_EQ(f.AboveAttribute(), static_cast<int>(i));
    EXPECT_EQ(f.kind.has_value(), !e.At("kind").IsNull());
    const auto& values = e.At("values");
    EXPECT_EQ(f.values.has_value(), !values.IsNull());
    if (f.values) {
      EXPECT_EQ(f.values->size(), values.Size());
      for (const auto& [key, pair] : values.AsObject()) {
        const auto& sides = f.values->at(key);
        for (const auto side : {0, 1}) {
          const auto& want = pair.At(static_cast<std::size_t>(side));
          const auto& got = side == 0 ? sides.below : sides.above;
          EXPECT_EQ(got.has_value(), !want.IsNull()) << key;
          if (got) {
            EXPECT_EQ(*got, want.AsNumber()) << key;
          }
        }
      }
    }
  }
  EXPECT_EQ(m.SurfaceAttribute(), 1);

  // the faces: the base, then the sides the mesh has, after the interfaces
  const auto& faces = doc.At("faces");
  ASSERT_EQ(m.Faces().size(), faces.Size());
  ASSERT_EQ(m.Faces().size(), m.IsRectangle() ? 3u : 5u);
  auto attributes = std::vector<int>{};
  for (auto i = std::size_t{0}; i < faces.Size(); ++i) {
    const auto& f = m.Faces()[i];
    const auto& e = faces.At(i);
    EXPECT_EQ(f.attribute, e.At("attribute").AsInt());
    EXPECT_EQ(f.attribute, static_cast<int>(m.Interfaces().size() + i) + 1);
    EXPECT_EQ(f.name, e.At("name").AsString());
    EXPECT_EQ(f.name, flat::FACES[i].name);
    EXPECT_EQ(f.axis, e.At("axis").AsString());
    EXPECT_EQ(f.value, e.At("value").AsNumber());
    EXPECT_EQ(f.outward, e.At("outward").AsInt());
    EXPECT_EQ(&m.FaceNamed(f.name), &f);
    EXPECT_FALSE(m.IsFaultAttribute(f.attribute));
    attributes.push_back(f.attribute);
  }
  EXPECT_EQ(m.FaceAttributes(), attributes);
  EXPECT_EQ(m.FaceNamed("base").value, -m.BaseDepth());
  EXPECT_GE(m.Scale(), m.BaseDepth());

  const auto* refinements = doc.Find("refinements");
  EXPECT_EQ(m.Refinements().size(), refinements ? refinements->Size() : 0u);
  const auto& fields = doc.At("fields");
  ASSERT_EQ(m.Fields().size(), fields.Size());
  for (auto i = std::size_t{0}; i < fields.Size(); ++i) {
    const auto& f = m.Fields()[i];
    const auto& e = fields.At(i);
    EXPECT_EQ(f.name, e.At("name").AsString());
    EXPECT_EQ(f.file, e.At("file").AsString());
    EXPECT_EQ(f.fe_space, e.At("fe_space").AsString());
    EXPECT_EQ(f.vdim, e.At("vdim").AsInt());
    EXPECT_EQ(f.unit, e.At("unit").AsString());
    EXPECT_EQ(f.layers, Ints(e.At("layers")));
    EXPECT_TRUE(m.HasField(f.name));
    if (files) {
      EXPECT_TRUE(std::filesystem::exists(m.FieldFile(f.name))) << f.name;
    }
  }
  const auto* scales = doc.Find("scales");
  EXPECT_EQ(m.HasScales(), scales && !scales->IsNull());
  if (m.HasScales()) {
    EXPECT_EQ(m.GetScales().Length(), scales->At("length").AsNumber());
  }
  if (const auto* c = doc.Find("constants")) {
    EXPECT_EQ(m.Constants().size(), c->Size());
    for (const auto& [key, value] : c->AsObject()) {
      EXPECT_EQ(m.Constant(key), value.AsNumber());
      EXPECT_TRUE(m.HasConstant(key));
    }
  }
  EXPECT_EQ(m.Model().has_value(), doc.Find("model") && !doc.At("model").IsNull());

  // the faults, their attributes after the interfaces' and the faces', and
  // their curve groups four a fault
  const auto* faults = doc.Find("faults");
  ASSERT_EQ(m.Faults().size(), faults ? faults->Size() : 0u);
  for (auto i = std::size_t{0}; i < m.Faults().size(); ++i) {
    const auto& f = m.Faults()[i];
    const auto& e = faults->At(i);
    SCOPED_TRACE(f.name);
    const auto k = static_cast<int>(i) + 1;
    EXPECT_EQ(f.name, e.At("name").AsString());
    EXPECT_EQ(f.attribute, e.At("attribute").AsInt());
    EXPECT_EQ(f.attribute, static_cast<int>(m.Interfaces().size() + m.Faces().size()) + k);
    EXPECT_EQ(f.layers, Ints(e.At("layers")));
    EXPECT_EQ(f.hanging_wall, Ints(e.At("hanging_wall")));
    EXPECT_EQ(f.tip, OptionalInt(e.At("tip")));
    EXPECT_EQ(f.free_edge, OptionalInt(e.At("free_edge")));
    EXPECT_EQ(f.junction, OptionalInt(e.At("junction")));
    EXPECT_EQ(f.face_edge, OptionalInt(e.At("face_edge")));
    const auto groups = std::array{f.tip, f.free_edge, f.junction, f.face_edge};
    for (auto j = 0; j < 4; ++j) {
      if (groups[static_cast<std::size_t>(j)]) {
        EXPECT_EQ(*groups[static_cast<std::size_t>(j)], 4 * k - 3 + j) << flat::FAULT_CURVES[j];
      }
    }
    ASSERT_EQ(f.meets.size(), e.At("meets").Size());
    for (auto j = std::size_t{0}; j < f.meets.size(); ++j) {
      EXPECT_EQ(f.meets[j], e.At("meets").At(j).AsString());
      EXPECT_TRUE(f.Meets(f.meets[j]));
    }
    ASSERT_EQ(f.ends_on.size(), e.At("ends_on").Size());
    for (const auto& other : f.ends_on) EXPECT_TRUE(f.EndsOn(other));
    EXPECT_EQ(f.controls.has_value(), !e.At("controls").IsNull());
    EXPECT_EQ(f.offset, e.At("offset").AsNumber());
    EXPECT_EQ(f.sizing.has_value(), !e.At("sizing").IsNull());
    if (f.sizing) {
      EXPECT_EQ(f.sizing->size, e.At("sizing").At("size").AsNumber());
    }
    EXPECT_EQ(&m.FaultNamed(f.name), &f);
    EXPECT_EQ(&m.FaultWithAttribute(f.attribute), &f);
    EXPECT_TRUE(m.HasFault(f.name));
    EXPECT_TRUE(m.IsFaultAttribute(f.attribute));
    for (const auto a : f.hanging_wall) EXPECT_GT(a, static_cast<int>(m.Layers().size()));
  }
}

/// A mesh with faults as the fixture writes it.
struct Mesh {
  std::vector<MeshVertex> vertices;
  std::vector<MeshTetrahedron> cells;
  std::vector<int> layers;  // each cell's element attribute
  std::vector<MeshTriangle> boundary;
  std::vector<int> attributes;
};

Mesh ReadMesh(const std::filesystem::path& path) {
  auto in = std::ifstream(path);
  auto mesh = Mesh{};
  for (auto line = std::string{}; std::getline(in, line);) {
    auto words = std::istringstream(line);
    auto kind = std::string{};
    words >> kind;
    if (kind == "vertex") {
      auto& v = mesh.vertices.emplace_back();
      words >> v[0] >> v[1] >> v[2];
    } else if (kind == "cell") {
      auto& c = mesh.cells.emplace_back();
      auto attribute = 0;
      words >> c[0] >> c[1] >> c[2] >> c[3] >> attribute;
      mesh.layers.push_back(attribute);
    } else if (kind == "boundary") {
      auto& t = mesh.boundary.emplace_back();
      auto attribute = 0;
      words >> t[0] >> t[1] >> t[2] >> attribute;
      mesh.attributes.push_back(attribute);
    }
  }
  return mesh;
}

std::set<MeshEdge> AsSet(const std::vector<MeshEdge>& edges) {
  return {edges.begin(), edges.end()};
}

}  // namespace

TEST(CartesianManifest, ReadsEveryFlatManifestThePythonWrites) {
  PLANETMODEL_FIXTURE(root, "flat_manifests");
  auto read = std::map<std::string, MeshManifest>{};
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    const auto label = entry.path().filename().string();
    for (const auto& file : std::filesystem::directory_iterator(entry.path())) {
      if (file.path().extension() != ".json") continue;
      SCOPED_TRACE(file.path().string());
      const auto m = MeshManifest(file.path());
      ExpectAsWritten(m, planetmodel::ReadJsonFile(file.path()));
      EXPECT_TRUE(std::filesystem::exists(m.MeshFile()));
      read.emplace(label, m);
    }
  }
  ASSERT_EQ(read.size(), 5u);

  const auto& box = read.at("box");
  EXPECT_EQ(box.Format(), "msh");
  EXPECT_EQ(box.Coordinates(), (std::vector<std::string>{"x", "y", "z"}));
  EXPECT_FALSE(box.HasScales());
  EXPECT_FALSE(box.Model());
  EXPECT_TRUE(box.Faults().empty());
  EXPECT_THROW(static_cast<void>(box.SolidAttributes()), Error);
  EXPECT_EQ(box.LayerNamed("mantle").attribute, 2);
  EXPECT_EQ(box.Boundaries(), (std::vector{0.0, 0.4, 1.0}));
  EXPECT_EQ(box.TopDepth(), 0.0);
  EXPECT_EQ(box.BaseDepth(), 1.0);
  EXPECT_EQ(box.InterfaceNamed("moho").depth, 0.4);
  EXPECT_EQ(box.FaceAttributes(), (std::vector<int>{3, 4, 5, 6, 7}));
  EXPECT_EQ(box.FaceNamed("y_min").value, -0.5);
  EXPECT_EQ(box.FaceNamed("y_max").outward, 1);
  EXPECT_EQ(box.FaceNamed("base").axis, "z");
  EXPECT_THROW(static_cast<void>(box.FaceNamed("top")), Error);

  const auto& rectangle = read.at("rectangle");
  EXPECT_TRUE(rectangle.IsRectangle());
  EXPECT_EQ(rectangle.Dimension(), 2);
  EXPECT_EQ(rectangle.FaceAttributes(), (std::vector<int>{3, 4, 5}));
  EXPECT_THROW(static_cast<void>(rectangle.FaceNamed("y_min")), Error);

  const auto& faulted = read.at("faulted");
  ASSERT_EQ(faulted.Faults().size(), 1u);
  const auto& fault = faulted.FaultNamed("fault");
  EXPECT_EQ(fault.attribute, 8);
  EXPECT_EQ(fault.tip, 1);
  EXPECT_EQ(fault.free_edge, 2);
  EXPECT_FALSE(fault.junction);
  EXPECT_EQ(fault.face_edge, 4);
  EXPECT_EQ(faulted.LayerNamed("crust").attributes, (std::vector<int>{1, 3}));
  EXPECT_EQ(faulted.ElementAttributes(), (std::vector<int>{1, 3, 2}));
  EXPECT_EQ(MeshManifest::Marker(fault.hanging_wall, 3), (std::vector<int>{0, 0, 1}));

  const auto& referential = read.at("referential");
  EXPECT_FALSE(referential.PhysicalNodes());
  EXPECT_EQ(referential.Displacement(), "displacement");
  EXPECT_EQ(referential.Options(), planetmodel::ReadOptions{});
  EXPECT_TRUE(std::filesystem::exists(referential.ModelFile()));
  EXPECT_EQ(referential.Model()->schema, "planetmodel.model/3");
  EXPECT_EQ(referential.SolidAttributes(), (std::vector<int>{1, 2}));
  EXPECT_TRUE(referential.FluidAttributes().empty());
  EXPECT_TRUE(referential.FluidSolidInterfaces().empty());
  EXPECT_EQ(referential.Interfaces()[0].kind, "free");
  EXPECT_EQ(referential.InterfaceNamed("moho").ValueBeside("rho", 2), 3.3);
  EXPECT_EQ(referential.InterfaceNamed("moho").ValueBeside("rho", 1), 2.8);
  EXPECT_THROW(static_cast<void>(referential.InterfaceNamed("surface").ValueBeside("rho", 2)),
               Error);
  EXPECT_EQ(referential.FieldNamed("displacement").vdim, 3);
  EXPECT_TRUE(referential.GetScales().IsSI());

  const auto& cut = read.at("physical_rectangle");
  EXPECT_TRUE(cut.IsRectangle());
  EXPECT_EQ(cut.Model()->schema, "planetmodel.model/3");
  EXPECT_EQ(cut.FieldNamed("rho").fe_space, "L2_2D_P2");
}

TEST(CartesianManifest, ReadsTheBaseCaseAndItsVariants) {
  PLANETMODEL_FIXTURE(path, "flat_manifest_cases/base.json");
  const auto m = MeshManifest(path);
  ExpectAsWritten(m, planetmodel::ReadJsonFile(path), false);
  EXPECT_EQ(m.SolidAttributes(), (std::vector<int>{1, 2}));
  EXPECT_EQ(m.BodyAttributes(), (std::vector<int>{1, 2}));
  EXPECT_TRUE(m.FluidSolidInterfaces().empty());
  EXPECT_THROW(static_cast<void>(m.G()), Error);
  EXPECT_EQ(m.Scale(), 450.0);
  EXPECT_EQ(m.ModelFile(), path.parent_path() / "box.model.nc");
  EXPECT_EQ(m.Meta().At("kind").AsString(), "test");
  EXPECT_EQ(m.Refinements()[0].decay_width, 20.0);
  EXPECT_EQ(m.Constant("g"), 9.8e-3);
  EXPECT_THROW(static_cast<void>(m.Constant("G")), Error);
  EXPECT_THROW(static_cast<void>(m.FaultNamed("thrust")), Error);
  EXPECT_THROW(static_cast<void>(m.LayerOf(3)), Error);
  EXPECT_THROW(static_cast<void>(m.ElementAttributes({3})), Error);

  for (const auto* label : {"optional_keys_absent", "unknown_key", "no_model_said",
                            "rectangle", "buried_top", "faults"}) {
    SCOPED_TRACE(label);
    PLANETMODEL_FIXTURE(variant, std::format("flat_manifest_cases/accept_{}.json", label));
    ExpectAsWritten(MeshManifest(variant), planetmodel::ReadJsonFile(variant), false);
  }
  PLANETMODEL_FIXTURE(absent, "flat_manifest_cases/accept_optional_keys_absent.json");
  const auto bare = MeshManifest(absent);
  EXPECT_TRUE(bare.Refinements().empty());
  EXPECT_FALSE(bare.Model());
  EXPECT_THROW(static_cast<void>(bare.ModelFile()), Error);
  EXPECT_EQ(bare.GetScales().Temperature(), 1.0);
  PLANETMODEL_FIXTURE(silent, "flat_manifest_cases/accept_no_model_said.json");
  const auto quiet = MeshManifest(silent);
  EXPECT_FALSE(quiet.HasScales());
  try {
    static_cast<void>(quiet.FluidAttributes());
    FAIL();
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "layers[0].fluid");
  }
  // a fluid layer over a solid one: an ocean over a crust
  auto ocean = planetmodel::ReadJsonFile(path);
  {
    auto text = std::ifstream(path);
    auto json = std::string(std::istreambuf_iterator<char>(text), {});
    const auto at = json.find("\"fluid\": false");
    ASSERT_NE(at, std::string::npos);
    json.replace(at, 14, "\"fluid\": true");
    ocean = planetmodel::ReadJson(json, path.string());
  }
  const auto wet = MeshManifest(ocean, path);
  EXPECT_EQ(wet.FluidAttributes(), (std::vector<int>{1}));
  EXPECT_EQ(wet.SolidAttributes(), (std::vector<int>{2}));
  EXPECT_EQ(wet.FluidSolidInterfaces(), (std::vector<int>{2}));
  EXPECT_EQ(wet.FluidSolidInterfaces(1), (std::vector<int>{2}));
  PLANETMODEL_FIXTURE(buried, "flat_manifest_cases/accept_buried_top.json");
  EXPECT_EQ(MeshManifest(buried).TopDepth(), 5.0);

  PLANETMODEL_FIXTURE(with_faults, "flat_manifest_cases/accept_faults.json");
  const auto f = MeshManifest(with_faults);
  EXPECT_EQ(f.FaultAttributes(), (std::vector<int>{8, 9}));
  EXPECT_EQ(f.LayerNamed("crust").attributes, (std::vector<int>{1, 3, 4}));
  EXPECT_EQ(f.ElementAttributes(), (std::vector<int>{1, 3, 4, 2, 5}));
  EXPECT_EQ(f.ElementAttributes({2}), (std::vector<int>{2, 5}));
  EXPECT_EQ(f.LayerOf(5).name, "mantle");
  const auto& thrust = f.FaultNamed("thrust");
  EXPECT_EQ(thrust.face_edge, 4);
  EXPECT_EQ(*thrust.controls, (std::vector<std::array<int, 2>>{{17, 17}}));
  const auto& splay = f.FaultWithAttribute(9);
  EXPECT_EQ(splay.tip, 5);
  EXPECT_EQ(splay.junction, 7);
  EXPECT_FALSE(splay.face_edge);
  EXPECT_TRUE(splay.EndsOn("thrust"));
  EXPECT_FALSE(f.IsFaultAttribute(7));
}

TEST(CartesianManifest, EachReaderRefusesTheOthersManifestsByTheirSchema) {
  // a flat manifest by the spherical reader, naming the flat one
  PLANETMODEL_FIXTURE(base, "flat_manifest_cases/base.json");
  try {
    static_cast<void>(planetmodel::MeshManifest(base));
    FAIL() << "the spherical reader read a flat manifest";
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "schema");
    const auto said = std::string{e.what()};
    EXPECT_NE(said.find("planetmodel.mesh.manifest/7"), std::string::npos) << said;
    EXPECT_NE(said.find("planetmodel::cartesian::MeshManifest"), std::string::npos) << said;
    EXPECT_EQ(said.find("regenerate"), std::string::npos) << said;
  }
  // the spherical ones, with faults and without, by the flat reader
  for (const auto* name : {"manifest_cases/base.json", "manifest_cases/accept_faults.json"}) {
    PLANETMODEL_FIXTURE(spherical, name);
    EXPECT_TRUE(RefusedAt(spherical, "schema", "read it with planetmodel::MeshManifest")) << name;
  }
  EXPECT_EQ(flat::MANIFEST_SCHEMA, "planetmodel.mesh.manifest/7");
}

TEST(CartesianManifest, RefusesWhatThePythonRefusesAtItsPlace) {
  PLANETMODEL_FIXTURE(index, "flat_manifest_cases/bad.txt");
  auto in = std::ifstream(index);
  auto cases = 0;
  for (auto line = std::string{}; std::getline(in, line);) {
    const auto tab1 = line.find('\t');
    const auto tab2 = line.find('\t', tab1 + 1);
    const auto label = line.substr(0, tab1);
    const auto where = line.substr(tab1 + 1, tab2 - tab1 - 1);
    const auto phrase = line.substr(tab2 + 1);
    EXPECT_TRUE(RefusedAt(index.parent_path() / ("bad_" + label + ".json"), where, phrase))
        << label;
    ++cases;
  }
  EXPECT_GT(cases, 30);
  EXPECT_THROW(MeshManifest(planetmodel::ReadJson("[]"), "x.json"), Error);
  EXPECT_THROW(MeshManifest("/nonexistent/x.json"), Error);
}

TEST(CartesianManifest, TheFaceEdgeIsToldFromTheTipByTheFacesOfTheBox) {
  // a fan of two faces in the plane y = 0, one edge on the surface z = 0,
  // one on the side x = 1, the rest buried
  const auto faults = std::vector<std::vector<MeshTriangle>>{{{0, 1, 2}, {1, 3, 2}}};
  const auto surface = std::vector<MeshTriangle>{{0, 1, 4}};
  const auto sides = std::vector<MeshTriangle>{{1, 3, 5}};
  const auto found = flat::FaultCurves(faults, surface, sides);
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found[0].free_edge, (std::vector<MeshEdge>{{0, 1}}));
  EXPECT_EQ(found[0].face_edge, (std::vector<MeshEdge>{{1, 3}}));
  EXPECT_EQ(found[0].tip, (std::vector<MeshEdge>{{0, 2}, {2, 3}}));
  // the face edge is free, as the surface's: never pinned
  EXPECT_EQ(found[0].Pinned(), found[0].tip);
  // without the faces of the box it is taken for a tip, as on a sphere
  const auto blind = planetmodel::FaultCurves(faults, surface);
  EXPECT_EQ(blind[0].tip, (std::vector<MeshEdge>{{0, 2}, {1, 3}, {2, 3}}));
  // an edge on the surface and on a side is the free edge's
  const auto corner = flat::FaultCurves(faults, surface, std::vector<MeshTriangle>{{0, 1, 5}});
  EXPECT_EQ(corner[0].free_edge, (std::vector<MeshEdge>{{0, 1}}));
  EXPECT_TRUE(corner[0].face_edge.empty());
}

TEST(CartesianManifest, TheRulesGiveThePythonsAnswersOnFlatMeshesWithFaults) {
  PLANETMODEL_FIXTURE(root, "flat_fault_meshes");
  auto cases = 0;
  auto with_face_edge = 0;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    const auto label = entry.path().filename().string();
    SCOPED_TRACE(label);
    const auto path = entry.path() / (label + ".json");
    const auto m = MeshManifest(path);
    ExpectAsWritten(m, planetmodel::ReadJsonFile(path), false);
    const auto mesh = ReadMesh(entry.path() / (label + ".mesh.txt"));
    ASSERT_FALSE(mesh.vertices.empty());

    // the answers
    auto pairs = 0;
    auto expected = std::map<std::string, std::map<std::string, std::set<MeshEdge>>>{};
    auto ends = std::map<std::string, std::vector<std::string>>{};
    auto walls = std::map<std::string, std::vector<int>>{};
    auto in = std::ifstream(entry.path() / (label + ".expect"));
    for (auto line = std::string{}; std::getline(in, line);) {
      auto words = std::istringstream(line);
      auto kind = std::string{};
      words >> kind;
      if (kind == "pair") {
        auto b = std::size_t{0}, p = std::size_t{0}, q = std::size_t{0};
        auto plus = 0;
        words >> b >> p >> q >> plus;
        ASSERT_TRUE(m.IsFaultAttribute(mesh.attributes[b]));
        EXPECT_EQ(planetmodel::PlusCell(mesh.vertices, mesh.boundary[b], mesh.cells[p],
                                        mesh.cells[q]),
                  plus)
            << b;
        ++pairs;
      } else if (kind == "edge") {
        auto name = std::string{}, key = std::string{};
        auto edge = MeshEdge{};
        words >> name >> key >> edge[0] >> edge[1];
        expected[name][key].insert(edge);
      } else if (kind == "ends") {
        auto name = std::string{}, other = std::string{};
        words >> name >> other;
        ends[name].push_back(other);
      } else if (kind == "wall") {
        auto name = std::string{};
        auto cell = std::size_t{0};
        auto wall = 0;
        words >> name >> cell >> wall;
        auto& w = walls[name];
        w.resize(mesh.cells.size(), 0);
        w[cell] = wall;
      }
    }

    // each fault's faces, the free surface's and the box's
    auto faces = std::vector<std::vector<MeshTriangle>>(m.Faults().size());
    auto surface = std::vector<MeshTriangle>{};
    auto sides = std::vector<MeshTriangle>{};
    const auto face_attributes = m.FaceAttributes();
    for (auto b = std::size_t{0}; b < mesh.boundary.size(); ++b) {
      const auto a = mesh.attributes[b];
      if (m.IsFaultAttribute(a)) {
        faces[static_cast<std::size_t>(a - m.Faults().front().attribute)].push_back(
            mesh.boundary[b]);
      } else if (a == m.SurfaceAttribute()) {
        surface.push_back(mesh.boundary[b]);
      } else if (std::ranges::contains(face_attributes, a)) {
        sides.push_back(mesh.boundary[b]);
      }
    }
    auto total = std::size_t{0};
    for (const auto& f : faces) total += f.size();
    EXPECT_EQ(static_cast<std::size_t>(pairs), total);
    EXPECT_FALSE(sides.empty());
    const auto curves = flat::FaultCurves(faces, surface, sides);
    ASSERT_EQ(curves.size(), m.Faults().size());
    auto names = std::vector<std::string>{};
    for (auto k = std::size_t{0}; k < curves.size(); ++k) {
      const auto& fault = m.Faults()[k];
      names.push_back(fault.name);
      const auto& want = expected[fault.name];
      for (const auto& [key, edges, group] :
           {std::tuple{"tip", curves[k].tip, fault.tip},
            std::tuple{"free_edge", curves[k].free_edge, fault.free_edge},
            std::tuple{"junction", curves[k].junction, fault.junction},
            std::tuple{"face_edge", curves[k].face_edge, fault.face_edge}}) {
        const auto it = want.find(key);
        EXPECT_EQ(AsSet(edges), it == want.end() ? std::set<MeshEdge>{} : it->second)
            << fault.name << " " << key;
        EXPECT_TRUE(std::ranges::is_sorted(edges)) << fault.name << " " << key;
        // the manifest names a curve's group exactly where the fault has it
        EXPECT_EQ(group.has_value(), !edges.empty()) << fault.name << " " << key;
      }
      with_face_edge += !curves[k].face_edge.empty();
      const auto it = want.find("junction_end");
      EXPECT_EQ(AsSet(curves[k].junction_end),
                it == want.end() ? std::set<MeshEdge>{} : it->second)
          << fault.name;
      auto ending = std::vector<std::string>{};
      for (const auto j : curves[k].ends_on) {
        ending.push_back(m.Faults()[static_cast<std::size_t>(j)].name);
      }
      EXPECT_EQ(ending, ends[fault.name]) << fault.name;
      EXPECT_EQ(ending, fault.ends_on) << fault.name;
    }
    // the walls, and the collar attributes the mesher gave the cells
    const auto all =
        flat::WallsOfFaults(mesh.vertices, mesh.cells, faces, surface, sides, names);
    ASSERT_EQ(all.size(), m.Faults().size());
    for (auto k = std::size_t{0}; k < all.size(); ++k) {
      const auto& fault = m.Faults()[k];
      auto want_walls = walls[fault.name];
      want_walls.resize(mesh.cells.size(), 0);
      EXPECT_EQ(all[k], want_walls) << fault.name;
      auto hanging = 0;
      for (auto c = std::size_t{0}; c < mesh.cells.size(); ++c) {
        const auto in_collar = std::ranges::contains(fault.hanging_wall, mesh.layers[c]);
        EXPECT_EQ(in_collar, all[k][c] == 1) << fault.name << " cell " << c;
        hanging += all[k][c] == 1;
      }
      EXPECT_GT(hanging, 0) << fault.name;
    }
    for (const auto a : mesh.layers) EXPECT_NO_THROW(static_cast<void>(m.LayerOf(a)));
    ++cases;
  }
  EXPECT_EQ(cases, 2);
  EXPECT_GE(with_face_edge, 2);
}
