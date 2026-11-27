// The manifest reader: every manifest the Python writes read into the
// values it wrote, the keys added within the schema optional, and every
// fault the Python's validate_structure refuses refused at its place.
#include "planetmodel/manifest.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "fixtures.hpp"

namespace {

using planetmodel::Error;
using planetmodel::Json;
using planetmodel::MeshManifest;

/// Whether reading `path` throws an Error at `where` saying `phrase`.
::testing::AssertionResult RefusedAt(const std::filesystem::path& path,
                                     const std::string& where, const std::string& phrase) {
  try {
    static_cast<void>(planetmodel::ReadMeshManifest(path));
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

/// The manifest's records against the document they were read from.
void ExpectAsWritten(const MeshManifest& m, const Json& doc, bool files = true) {
  const auto& mesh = doc.At("mesh");
  EXPECT_EQ(m.MeshFile(), m.Path().parent_path() / mesh.At("file").AsString());
  EXPECT_EQ(m.Format(), mesh.At("format").AsString());
  EXPECT_EQ(m.Nodes(), mesh.At("nodes").AsString());
  const auto& d = mesh.At("displacement");
  EXPECT_EQ(m.Displacement().has_value(), !d.IsNull());
  if (!d.IsNull()) {
    EXPECT_EQ(*m.Displacement(), d.AsString());
  }

  const auto& layers = doc.At("layers");
  ASSERT_EQ(m.Layers().size(), layers.Size());
  for (auto i = std::size_t{0}; i < layers.Size(); ++i) {
    const auto& [attribute, name, r_inner, r_outer, in_geometry, fluid, attributes] =
        m.Layers()[i];
    const auto& e = layers.At(i);
    if (m.Schema() == planetmodel::MANIFEST_SCHEMA_WITH_FAULTS) {
      EXPECT_EQ(attributes, Ints(e.At("attributes")));
    } else {
      EXPECT_EQ(attributes, (std::vector<int>{attribute}));
    }
    for (const auto a : attributes) EXPECT_EQ(&m.LayerOf(a), &m.Layers()[i]);
    EXPECT_EQ(attribute, e.At("attribute").AsInt());
    EXPECT_EQ(name, e.At("name").AsString());
    EXPECT_EQ(r_inner, e.At("r_inner").AsNumber());
    EXPECT_EQ(r_outer, e.At("r_outer").AsNumber());
    EXPECT_EQ(in_geometry, e.At("in_geometry").AsBool());
    EXPECT_EQ(fluid.has_value(), !e.At("fluid").IsNull());
    if (fluid) {
      EXPECT_EQ(*fluid, e.At("fluid").AsBool());
    }
  }
  const auto& faces = doc.At("interfaces");
  ASSERT_EQ(m.Interfaces().size(), faces.Size());
  for (auto i = std::size_t{0}; i < faces.Size(); ++i) {
    const auto& f = m.Interfaces()[i];
    const auto& e = faces.At(i);
    EXPECT_EQ(f.attribute, e.At("attribute").AsInt());
    EXPECT_EQ(f.name, e.At("name").AsString());
    EXPECT_EQ(f.radius, e.At("radius").AsNumber());
    EXPECT_EQ(f.between_layers[0], e.At("between_layers").At(0).AsInt());
    EXPECT_EQ(f.between_layers[1], e.At("between_layers").At(1).AsInt());
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
    EXPECT_EQ(f.ordering, e.At("ordering").AsString());
    EXPECT_EQ(f.rank, e.At("rank").AsInt());
    EXPECT_EQ(f.weight, e.At("weight").AsInt());
    EXPECT_EQ(f.voigt, e.At("voigt").AsBool());
    EXPECT_EQ(f.unit, e.At("unit").AsString());
    EXPECT_EQ(f.layers, Ints(e.At("layers")));
    EXPECT_EQ(f.radial_degree.has_value(), !e.At("radial_degree").IsNull());
    if (files) {
      EXPECT_TRUE(std::filesystem::exists(m.FieldFile(f.name))) << f.name;
    }
  }
  const auto* scales = doc.Find("scales");
  EXPECT_EQ(m.HasScales(), scales && !scales->IsNull());
  if (m.HasScales()) {
    EXPECT_EQ(m.GetScales().Length(), scales->At("length").AsNumber());
    EXPECT_EQ(m.GetScales().Mass(), scales->At("mass").AsNumber());
    EXPECT_EQ(m.GetScales().Time(), scales->At("time").AsNumber());
  }
  if (const auto* c = doc.Find("constants")) {
    EXPECT_EQ(m.Constants().size(), c->Size());
    for (const auto& [key, value] : c->AsObject()) {
      EXPECT_EQ(m.Constant(key), value.AsNumber());
    }
  }
  EXPECT_EQ(m.Model().has_value(), doc.Find("model") && !doc.At("model").IsNull());
}

std::optional<int> OptionalInt(const Json& v) {
  return v.IsNull() ? std::nullopt : std::optional<int>{v.AsInt()};
}

/// The manifest's fault records against the document they were read from.
void ExpectFaultsAsWritten(const MeshManifest& m, const Json& doc) {
  const auto* faults = doc.Find("faults");
  ASSERT_EQ(m.Faults().size(), faults ? faults->Size() : 0u);
  for (auto i = std::size_t{0}; i < m.Faults().size(); ++i) {
    const auto& f = m.Faults()[i];
    const auto& e = faults->At(i);
    SCOPED_TRACE(f.name);
    EXPECT_EQ(f.name, e.At("name").AsString());
    EXPECT_EQ(f.attribute, e.At("attribute").AsInt());
    EXPECT_EQ(f.layers, Ints(e.At("layers")));
    EXPECT_EQ(f.hanging_wall, Ints(e.At("hanging_wall")));
    ASSERT_EQ(f.ends_on.size(), e.At("ends_on").Size());
    for (auto j = std::size_t{0}; j < f.ends_on.size(); ++j) {
      EXPECT_EQ(f.ends_on[j], e.At("ends_on").At(j).AsString());
      EXPECT_TRUE(f.EndsOn(f.ends_on[j]));
      EXPECT_TRUE(f.Meets(f.ends_on[j]));
    }
    for (const auto a : f.hanging_wall) EXPECT_GT(a, static_cast<int>(m.Layers().size()));
    EXPECT_EQ(f.tip, OptionalInt(e.At("tip")));
    EXPECT_EQ(f.free_edge, OptionalInt(e.At("free_edge")));
    EXPECT_EQ(f.junction, OptionalInt(e.At("junction")));
    ASSERT_EQ(f.meets.size(), e.At("meets").Size());
    for (auto j = std::size_t{0}; j < f.meets.size(); ++j) {
      EXPECT_EQ(f.meets[j], e.At("meets").At(j).AsString());
      EXPECT_TRUE(f.Meets(f.meets[j]));
    }
    EXPECT_EQ(f.controls.has_value(), !e.At("controls").IsNull());
    if (f.controls) {
      ASSERT_EQ(f.controls->size(), e.At("controls").Size());
      for (auto j = std::size_t{0}; j < f.controls->size(); ++j) {
        EXPECT_EQ((*f.controls)[j][0], e.At("controls").At(j).At(0).AsInt());
        EXPECT_EQ((*f.controls)[j][1], e.At("controls").At(j).At(1).AsInt());
      }
    }
    EXPECT_EQ(f.offset, e.At("offset").AsNumber());
    EXPECT_EQ(f.sizing.has_value(), !e.At("sizing").IsNull());
    if (f.sizing) {
      EXPECT_EQ(f.sizing->size, e.At("sizing").At("size").AsNumber());
      EXPECT_EQ(f.sizing->far_size, e.At("sizing").At("far_size").AsNumber());
      EXPECT_EQ(f.sizing->decay_width, e.At("sizing").At("decay_width").AsNumber());
    }
    EXPECT_EQ(&m.FaultNamed(f.name), &f);
    EXPECT_EQ(&m.FaultWithAttribute(f.attribute), &f);
    EXPECT_TRUE(m.HasFault(f.name));
    EXPECT_TRUE(m.IsFaultAttribute(f.attribute));
    EXPECT_EQ(f.attribute, static_cast<int>(m.Interfaces().size() + i) + 1);
  }
  for (const auto& face : m.Interfaces()) {
    EXPECT_FALSE(m.IsFaultAttribute(face.attribute)) << face.name;
  }
}

}  // namespace

TEST(Manifest, ReadsEveryManifestThePythonWrites) {
  PLANETMODEL_FIXTURE(root, "manifests");
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
  ASSERT_EQ(read.size(), 7u);

  const auto& mesher = read.at("mesher");
  EXPECT_EQ(mesher.Format(), "msh");
  EXPECT_EQ(mesher.Options(), planetmodel::ReadOptions{});
  EXPECT_FALSE(mesher.HasScales());
  EXPECT_FALSE(mesher.Model());
  EXPECT_THROW(static_cast<void>(mesher.SolidAttributes()), Error);
  EXPECT_EQ(mesher.LayerNamed("mantle").attribute, 2);
  EXPECT_EQ(mesher.SurfaceAttribute(), 3);

  const auto& refined = read.at("mesher_refined");
  ASSERT_EQ(refined.Refinements().size(), 1u);
  EXPECT_EQ(refined.Refinements()[0].points, 1);
  EXPECT_TRUE(refined.Refinements()[0].digest);

  const auto& hollow = read.at("hollow_shell");
  EXPECT_TRUE(hollow.IsHollow());
  EXPECT_EQ(hollow.InnerRadius(), 0.5);
  EXPECT_EQ(hollow.ShellAttributes(), (std::vector<int>{3}));
  EXPECT_EQ(hollow.Interfaces().front().between_layers, (std::array<int, 2>{-1, 0}));
  EXPECT_EQ(hollow.Interfaces().front().BelowAttribute(), 0);
  EXPECT_EQ(hollow.SurfaceAttribute(), 3);
  EXPECT_EQ(hollow.OuterAttribute(), 4);

  const auto& referential = read.at("referential");
  EXPECT_FALSE(referential.PhysicalNodes());
  EXPECT_EQ(referential.Displacement(), "displacement");
  EXPECT_EQ(referential.Options(), planetmodel::ReadOptions{});
  EXPECT_TRUE(std::filesystem::exists(referential.ModelFile()));
  EXPECT_EQ(referential.Model()->schema, "planetmodel.model/1");
  EXPECT_EQ(referential.FluidAttributes(), (std::vector<int>{1}));
  EXPECT_EQ(referential.SolidAttributes(), (std::vector<int>{2, 3}));
  EXPECT_EQ(referential.FluidSolidInterfaces(), (std::vector<int>{1}));
  EXPECT_EQ(referential.Interfaces()[0].kind, "fluid-solid");
  EXPECT_EQ(referential.Interfaces()[0].ValueBeside("rho", 1), 12.0);
  EXPECT_EQ(referential.Interfaces()[0].ValueBeside("rho", 2), 5.0);
  EXPECT_TRUE(referential.GetScales().IsSI());
  EXPECT_DOUBLE_EQ(referential.G(), planetmodel::G_SI);

  const auto& physical = read.at("physical");
  EXPECT_TRUE(physical.PhysicalNodes());
  EXPECT_FALSE(physical.Displacement());

  const auto& nd = read.at("nondimensional");
  EXPECT_FALSE(nd.GetScales().IsSI());
  EXPECT_NEAR(nd.G(), 1.0, 1e-12);
  EXPECT_NEAR(nd.G() * nd.GetScales().Factor(planetmodel::units::GRAVITATIONAL_CONSTANT),
              planetmodel::G_SI, 1e-24);
}

TEST(Manifest, ReadsTheManifestsOfCubedMeshes) {
  // the manifests the builders of a cubed sphere and of a squared circle
  // write beside their own meshes, in MFEM's format with no export
  PLANETMODEL_FIXTURE(root, "cubed_manifests");
  auto read = std::map<std::string, MeshManifest>{};
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    const auto label = entry.path().filename().string();
    for (const auto& file : std::filesystem::directory_iterator(entry.path())) {
      if (file.path().extension() != ".json") continue;
      SCOPED_TRACE(file.path().string());
      const auto m = MeshManifest(file.path());
      ExpectAsWritten(m, planetmodel::ReadJsonFile(file.path()));
      EXPECT_TRUE(std::filesystem::exists(m.MeshFile()));
      EXPECT_EQ(m.Schema(), planetmodel::MANIFEST_SCHEMA);
      EXPECT_EQ(m.Format(), "mfem");
      EXPECT_FALSE(m.PhysicalNodes());
      EXPECT_FALSE(m.Displacement());
      EXPECT_FALSE(m.Model());
      EXPECT_TRUE(m.Fields().empty());
      EXPECT_TRUE(m.Faults().empty());
      read.emplace(label, m);
    }
  }
  ASSERT_EQ(read.size(), 2u);

  const auto& sphere = read.at("cubed_sphere");
  EXPECT_FALSE(sphere.IsHollow());
  ASSERT_EQ(sphere.Layers().size(), 3u);
  EXPECT_EQ(sphere.LayerNamed("mantle").attribute, 2);
  EXPECT_EQ(sphere.SurfaceAttribute(), 3);
  EXPECT_EQ(sphere.OuterAttribute(), 3);
  EXPECT_TRUE(sphere.ShellAttributes().empty());

  const auto& circle = read.at("squared_circle");
  EXPECT_TRUE(circle.IsHollow());
  EXPECT_EQ(circle.InnerRadius(), 0.5);
  ASSERT_EQ(circle.Layers().size(), 3u);
  EXPECT_EQ(circle.ShellAttributes(), (std::vector<int>{3}));
  EXPECT_EQ(circle.SurfaceAttribute(), 3);
  EXPECT_EQ(circle.OuterAttribute(), 4);
}

TEST(Manifest, ReadsTheBaseCaseAsWritten) {
  PLANETMODEL_FIXTURE(path, "manifest_cases/base.json");
  const auto m = MeshManifest(path);
  ExpectAsWritten(m, planetmodel::ReadJsonFile(path), false);
  // the function beside the other formats' readers reads what the
  // constructor does
  const auto read = planetmodel::ReadMeshManifest(path);
  ExpectAsWritten(read, planetmodel::ReadJsonFile(path), false);
  EXPECT_EQ(read.Path(), m.Path());
  EXPECT_EQ(read.Schema(), m.Schema());
  EXPECT_EQ(read.MeshFile(), m.MeshFile());
  EXPECT_EQ(m.BodyAttributes(), (std::vector<int>{1, 2}));
  EXPECT_EQ(m.ShellAttributes(), (std::vector<int>{3}));
  EXPECT_EQ(m.FluidAttributes(), (std::vector<int>{1}));
  EXPECT_EQ(m.SolidAttributes(), (std::vector<int>{2}));
  EXPECT_EQ(m.FluidSolidInterfaces(), (std::vector<int>{1}));
  EXPECT_EQ(m.FluidSolidInterfaces(1), (std::vector<int>{1}));
  EXPECT_EQ(m.SurfaceAttribute(), 2);
  EXPECT_EQ(m.SurfaceRadius(), 1.0);
  EXPECT_EQ(m.OuterAttribute(), 3);
  EXPECT_EQ(m.OuterRadius(), 1.5);
  EXPECT_EQ(MeshManifest::Marker({1, 3}, 4), (std::vector<int>{1, 0, 1, 0}));
  const auto& cmb = m.InterfaceNamed("cmb");
  EXPECT_EQ(cmb.ValueBeside("rho", 1), 1.7157);
  EXPECT_EQ(cmb.ValueBeside("rho", 2), 0.7655);
  EXPECT_THROW(static_cast<void>(cmb.ValueBeside("rho", 3)), Error);
  EXPECT_THROW(static_cast<void>(m.InterfaceNamed("surface").ValueBeside("rho", 3)), Error);
  EXPECT_THROW(static_cast<void>(cmb.ValueBeside("vs", 1)), Error);
  EXPECT_EQ(m.FieldNamed("rho").GetCharacter(), planetmodel::Character(0, 1, false));
  EXPECT_EQ(m.FieldFile("rho"), path.parent_path() / "earth.rho.gf");
  EXPECT_EQ(m.ModelFile(), path.parent_path() / "earth.model.nc");
  EXPECT_EQ(m.Meta().At("kind").AsString(), "test");
  EXPECT_EQ(m.Refinements()[0].decay_width, 0.2);
  try {
    static_cast<void>(m.FieldNamed("vs"));
    FAIL();
  } catch (const Error& e) {
    EXPECT_NE(std::string{e.what()}.find("it has displacement, rho"), std::string::npos)
        << e.what();
  }
  EXPECT_THROW(static_cast<void>(m.Constant("omega")), Error);
}

TEST(Manifest, TakesTheKeysAddedWithinTheSchemaAsOptional) {
  PLANETMODEL_FIXTURE(absent, "manifest_cases/accept_optional_keys_absent.json");
  const auto m = MeshManifest(absent);
  EXPECT_TRUE(m.Refinements().empty());
  EXPECT_FALSE(m.Model());
  EXPECT_THROW(static_cast<void>(m.ModelFile()), Error);
  EXPECT_EQ(m.GetScales().Temperature(), 1.0);
  EXPECT_TRUE(m.Constants().empty());
  EXPECT_TRUE(m.Meta().IsObject());
  EXPECT_EQ(m.Meta().Size(), 0u);

  PLANETMODEL_FIXTURE(unknown, "manifest_cases/accept_unknown_key.json");
  EXPECT_NO_THROW(static_cast<void>(MeshManifest(unknown)));

  PLANETMODEL_FIXTURE(silent, "manifest_cases/accept_no_model_said.json");
  const auto quiet = MeshManifest(silent);
  EXPECT_FALSE(quiet.HasScales());
  EXPECT_THROW(static_cast<void>(quiet.GetScales()), Error);
  EXPECT_FALSE(quiet.Layers()[0].fluid);
  try {
    static_cast<void>(quiet.FluidAttributes());
    FAIL();
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "layers[0].fluid");
  }
  EXPECT_THROW(static_cast<void>(quiet.Interfaces()[0].ValueBeside("rho", 1)), Error);
  EXPECT_EQ(quiet.BodyAttributes(), (std::vector<int>{1, 2}));
}

TEST(Manifest, RefusesWhatThePythonRefusesAtItsPlace) {
  PLANETMODEL_FIXTURE(index, "manifest_cases/bad.txt");
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
}

TEST(Manifest, ReadsNoFaultsWhereTheKeyIsAbsent) {
  PLANETMODEL_FIXTURE(path, "manifest_cases/base.json");
  const auto m = MeshManifest(path);
  EXPECT_EQ(m.Schema(), planetmodel::MANIFEST_SCHEMA);
  EXPECT_EQ(m.ElementAttributes(), (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(m.ElementAttributes({2, 1}), (std::vector<int>{2, 1}));
  EXPECT_TRUE(m.ElementAttributes(std::vector<int>{}).empty());
  EXPECT_EQ(m.LayerOf(2).name, "mantle");
  EXPECT_THROW(static_cast<void>(m.LayerOf(4)), Error);
  EXPECT_THROW(static_cast<void>(m.ElementAttributes({4})), Error);
  EXPECT_TRUE(m.Faults().empty());
  EXPECT_TRUE(m.FaultAttributes().empty());
  EXPECT_FALSE(m.IsFaultAttribute(4));
  EXPECT_FALSE(m.HasFault("megathrust"));
  EXPECT_THROW(static_cast<void>(m.FaultNamed("megathrust")), Error);
  EXPECT_THROW(static_cast<void>(m.FaultWithAttribute(4)), Error);
}

TEST(Manifest, ReadsTheFaultsAsWritten) {
  PLANETMODEL_FIXTURE(path, "manifest_cases/accept_faults.json");
  const auto m = MeshManifest(path);
  ExpectAsWritten(m, planetmodel::ReadJsonFile(path), false);
  ExpectFaultsAsWritten(m, planetmodel::ReadJsonFile(path));
  EXPECT_EQ(m.Schema(), planetmodel::MANIFEST_SCHEMA_WITH_FAULTS);
  EXPECT_EQ(m.FaultAttributes(), (std::vector<int>{4, 5}));
  EXPECT_FALSE(m.IsFaultAttribute(6));
  // the collars: each layer's cells carry a list, and a marker takes it
  EXPECT_EQ(m.LayerNamed("mantle").attributes, (std::vector<int>{2, 5, 6, 7}));
  EXPECT_EQ(m.ElementAttributes(), (std::vector<int>{1, 4, 2, 5, 6, 7, 3}));
  EXPECT_EQ(m.ElementAttributes({2}), (std::vector<int>{2, 5, 6, 7}));
  EXPECT_EQ(m.BodyAttributes(), (std::vector<int>{1, 4, 2, 5, 6, 7}));
  EXPECT_EQ(m.ShellAttributes(), (std::vector<int>{3}));
  EXPECT_EQ(m.SolidAttributes(), (std::vector<int>{2, 5, 6, 7}));
  EXPECT_EQ(m.FluidAttributes(), (std::vector<int>{1, 4}));
  EXPECT_EQ(m.LayerOf(4).name, "core");
  EXPECT_EQ(m.LayerOf(6).name, "mantle");
  EXPECT_THROW(static_cast<void>(m.LayerOf(8)), Error);
  EXPECT_EQ(MeshManifest::Marker(m.FaultNamed("splay").hanging_wall, 7),
            (std::vector<int>{0, 0, 0, 1, 0, 1, 1}));
  const auto& mega = m.FaultNamed("megathrust");
  EXPECT_EQ(mega.attribute, 4);
  EXPECT_EQ(mega.layers, (std::vector<int>{2}));
  EXPECT_EQ(mega.hanging_wall, (std::vector<int>{5, 6}));
  EXPECT_TRUE(mega.ends_on.empty());
  EXPECT_TRUE(m.FaultNamed("splay").EndsOn("megathrust"));
  EXPECT_FALSE(mega.EndsOn("splay"));
  EXPECT_EQ(mega.tip, 1);
  EXPECT_EQ(mega.free_edge, 2);
  EXPECT_EQ(mega.junction, 3);
  EXPECT_TRUE(mega.Meets("splay"));
  EXPECT_FALSE(mega.Meets("megathrust"));
  ASSERT_TRUE(mega.controls);
  EXPECT_EQ(*mega.controls, (std::vector<std::array<int, 2>>{{17, 17}}));
  EXPECT_EQ(mega.offset, 3.3e-8);
  ASSERT_TRUE(mega.sizing);
  EXPECT_EQ(mega.sizing->size, 0.05);
  EXPECT_EQ(mega.sizing->far_size, 0.3);
  EXPECT_EQ(mega.sizing->decay_width, 0.2);
  const auto& splay = m.FaultWithAttribute(5);
  EXPECT_EQ(splay.name, "splay");
  EXPECT_EQ(splay.layers, (std::vector<int>{1, 2}));
  EXPECT_EQ(splay.tip, 4);
  EXPECT_FALSE(splay.free_edge);
  EXPECT_EQ(splay.junction, 6);
  EXPECT_EQ(*splay.controls, (std::vector<std::array<int, 2>>{{9, 9}, {5, 17}}));
  EXPECT_EQ(splay.offset, 0.0);
  EXPECT_FALSE(splay.sizing);
  try {
    static_cast<void>(m.FaultNamed("slab"));
    FAIL();
  } catch (const Error& e) {
    EXPECT_NE(std::string{e.what()}.find("it has megathrust, splay"), std::string::npos)
        << e.what();
  }
  EXPECT_THROW(static_cast<void>(m.FaultWithAttribute(3)), Error);
  EXPECT_EQ(MeshManifest::Marker(m.FaultAttributes(), 5), (std::vector<int>{0, 0, 0, 1, 1}));
}

TEST(Manifest, ReadsTheManifestsOfMeshesWithFaults) {
  PLANETMODEL_FIXTURE(root, "fault_meshes");
  auto read = 0;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    const auto path = entry.path() / (entry.path().filename().string() + ".json");
    SCOPED_TRACE(path.string());
    const auto m = MeshManifest(path);
    const auto doc = planetmodel::ReadJsonFile(path);
    ExpectAsWritten(m, doc, false);
    ExpectFaultsAsWritten(m, doc);
    EXPECT_FALSE(m.Faults().empty());
    EXPECT_EQ(m.Schema(), planetmodel::MANIFEST_SCHEMA_WITH_FAULTS);
    ++read;
  }
  EXPECT_EQ(read, 3);
}

TEST(Manifest, RefusesADocumentThatIsNoManifest) {
  EXPECT_THROW(MeshManifest(planetmodel::ReadJson("[]"), "x.json"), Error);
  EXPECT_THROW(MeshManifest("/nonexistent/x.json"), Error);
}
