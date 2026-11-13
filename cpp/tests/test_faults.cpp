// The two rules of a fault's faces, which cell of a face is the plus one
// and which edges are a fault's curves, on a few hand-made tetrahedra and
// on meshes with faults the Python built, against the answers of the
// Python's own statement of the same rules.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/manifest.hpp"

namespace {

using planetmodel::CurveEdges;
using planetmodel::Error;
using planetmodel::FaultCurves;
using planetmodel::FaultSide;
using planetmodel::MeshEdge;
using planetmodel::MeshManifest;
using planetmodel::MeshTetrahedron;
using planetmodel::MeshTriangle;
using planetmodel::MeshVertex;
using planetmodel::PlusCell;

/// A face in the plane z = 0, its normal along +z, a corner above it and
/// one below, and one in its plane.
const auto CORNERS = std::vector<MeshVertex>{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                             {0.0, 1.0, 0.0}, {0.2, 0.2, 1.0},
                                             {0.2, 0.2, -1.0}, {0.5, 0.5, 0.0}};

/// Whether `call` throws an Error from `source` at `where`.
template <class Call>
::testing::AssertionResult RefusedBy(Call call, const std::string& source,
                                     const std::string& where) {
  try {
    static_cast<void>(call());
  } catch (const Error& e) {
    if (e.Source() == source && e.Where() == where) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "refused as " << e.what();
  }
  return ::testing::AssertionFailure() << "accepted";
}

/// A mesh with faults as the fixture writes it.
struct Mesh {
  std::vector<MeshVertex> vertices;
  std::vector<MeshTetrahedron> cells;
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

TEST(Faults, ThePlusCellIsOnTheSideTheFaceVertexOrderPointsTo) {
  const auto up = MeshTetrahedron{0, 1, 2, 3};
  const auto down = MeshTetrahedron{1, 4, 2, 0};
  EXPECT_EQ(FaultSide(CORNERS, {0, 1, 2}, up), 1);
  EXPECT_EQ(FaultSide(CORNERS, {0, 1, 2}, down), -1);
  EXPECT_EQ(FaultSide(CORNERS, {0, 2, 1}, up), -1);
  EXPECT_EQ(FaultSide(CORNERS, {1, 2, 0}, up), 1);
  EXPECT_EQ(PlusCell(CORNERS, {0, 1, 2}, down, up), 1);
  EXPECT_EQ(PlusCell(CORNERS, {0, 1, 2}, up, down), 0);
  EXPECT_EQ(PlusCell(CORNERS, {0, 2, 1}, down, up), 0);
}

TEST(Faults, TheSidesAreRefusedWhereTheyCannotBeTold) {
  EXPECT_TRUE(RefusedBy([] { return FaultSide(CORNERS, {0, 1, 2}, {0, 1, 3, 4}); },
                        "FaultSide", "cell"));
  EXPECT_TRUE(RefusedBy([] { return FaultSide(CORNERS, {0, 1, 2}, {0, 1, 2, 2}); },
                        "FaultSide", "cell"));
  EXPECT_TRUE(RefusedBy([] { return FaultSide(CORNERS, {0, 1, 2}, {0, 1, 2, 5}); },
                        "FaultSide", "cell"));
  EXPECT_TRUE(RefusedBy([] { return FaultSide(CORNERS, {0, 1, 2}, {0, 1, 2, 6}); },
                        "FaultSide", "cell"));
  EXPECT_TRUE(RefusedBy([] { return FaultSide(CORNERS, {0, -1, 2}, {0, 1, 2, 3}); },
                        "FaultSide", "face"));
  EXPECT_TRUE(RefusedBy([] { return PlusCell(CORNERS, {0, 1, 2}, {0, 1, 2, 3}, {3, 2, 1, 0}); },
                        "PlusCell", ""));
}

TEST(Faults, TheCurvesAreReadFromTheFaces) {
  // a square of two faces whose edge (0, 1) is on the surface, and a second
  // fault of one face meeting it along (2, 3)
  const auto faults = std::vector<std::vector<MeshTriangle>>{{{0, 1, 2}, {1, 3, 2}}, {{3, 2, 4}}};
  const auto surface = std::vector<MeshTriangle>{{0, 1, 9}};
  const auto curves = FaultCurves(faults, surface);
  ASSERT_EQ(curves.size(), 2u);
  EXPECT_EQ(curves[0].free_edge, (std::vector<MeshEdge>{{0, 1}}));
  EXPECT_EQ(curves[0].tip, (std::vector<MeshEdge>{{0, 2}, {1, 3}}));
  EXPECT_EQ(curves[0].junction, (std::vector<MeshEdge>{{2, 3}}));
  EXPECT_TRUE(curves[1].free_edge.empty());
  EXPECT_EQ(curves[1].tip, (std::vector<MeshEdge>{{2, 4}, {3, 4}}));
  EXPECT_EQ(curves[1].junction, (std::vector<MeshEdge>{{2, 3}}));
  const auto alone = FaultCurves(std::vector<std::vector<MeshTriangle>>{{{5, 4, 6}}}, {});
  EXPECT_EQ(alone[0].tip, (std::vector<MeshEdge>{{4, 5}, {4, 6}, {5, 6}}));
  EXPECT_TRUE(FaultCurves({}, surface).empty());
}

TEST(Faults, TheRulesGiveThePythonsAnswersOnMeshesWithFaults) {
  PLANETMODEL_FIXTURE(root, "fault_meshes");
  auto cases = 0;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    const auto label = entry.path().filename().string();
    SCOPED_TRACE(label);
    const auto m = MeshManifest(entry.path() / (label + ".json"));
    const auto mesh = ReadMesh(entry.path() / (label + ".mesh.txt"));
    ASSERT_FALSE(mesh.vertices.empty());
    ASSERT_EQ(mesh.boundary.size(), mesh.attributes.size());

    // the answers
    auto pairs = 0;
    auto expected = std::map<std::string, std::map<std::string, std::set<MeshEdge>>>{};
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
        const auto& face = mesh.boundary[b];
        EXPECT_EQ(PlusCell(mesh.vertices, face, mesh.cells[p], mesh.cells[q]), plus) << b;
        EXPECT_EQ(PlusCell(mesh.vertices, face, mesh.cells[q], mesh.cells[p]), 1 - plus) << b;
        EXPECT_EQ(FaultSide(mesh.vertices, face, mesh.cells[plus == 0 ? p : q]), 1) << b;
        EXPECT_EQ(FaultSide(mesh.vertices, face, mesh.cells[plus == 0 ? q : p]), -1) << b;
        ++pairs;
      } else if (kind == "edge") {
        auto name = std::string{}, key = std::string{};
        auto edge = MeshEdge{};
        words >> name >> key >> edge[0] >> edge[1];
        expected[name][key].insert(edge);
      }
    }

    // every fault face has its answer, and the curves
    auto faces = std::vector<std::vector<MeshTriangle>>(m.Faults().size());
    auto surface = std::vector<MeshTriangle>{};
    for (auto b = std::size_t{0}; b < mesh.boundary.size(); ++b) {
      const auto a = mesh.attributes[b];
      if (m.IsFaultAttribute(a)) {
        const auto& fault = m.FaultWithAttribute(a);
        faces[static_cast<std::size_t>(fault.attribute - m.Faults().front().attribute)]
            .push_back(mesh.boundary[b]);
      } else if (a == m.SurfaceAttribute()) {
        surface.push_back(mesh.boundary[b]);
      }
    }
    auto total = std::size_t{0};
    for (const auto& f : faces) total += f.size();
    EXPECT_EQ(static_cast<std::size_t>(pairs), total);
    const auto curves = FaultCurves(faces, surface);
    ASSERT_EQ(curves.size(), m.Faults().size());
    for (auto k = std::size_t{0}; k < curves.size(); ++k) {
      const auto& fault = m.Faults()[k];
      const auto& want = expected[fault.name];
      for (const auto& [key, edges, group] :
           {std::tuple{"tip", curves[k].tip, fault.tip},
            std::tuple{"free_edge", curves[k].free_edge, fault.free_edge},
            std::tuple{"junction", curves[k].junction, fault.junction}}) {
        const auto it = want.find(key);
        EXPECT_EQ(AsSet(edges), it == want.end() ? std::set<MeshEdge>{} : it->second)
            << fault.name << " " << key;
        EXPECT_TRUE(std::ranges::is_sorted(edges)) << fault.name << " " << key;
        EXPECT_EQ(group.has_value(), !edges.empty()) << fault.name << " " << key;
      }
    }
    ++cases;
  }
  EXPECT_EQ(cases, 2);
}
