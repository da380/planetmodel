// The three rules of a fault's faces, which cell of a face is the plus
// one, which edges are a fault's curves and where it ends, and which cells
// are in its walls, on a few hand-made tetrahedra and on meshes with
// faults the Python built, against the answers of the Python's own
// statement of the same rules; and the collar attributes the mesher wrote
// against the walls.
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
#include <utility>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/manifest.hpp"

namespace {

using planetmodel::CurveEdges;
using planetmodel::Error;
using planetmodel::FaultCurves;
using planetmodel::FaultSide;
using planetmodel::FaultWalls;
using planetmodel::MeshEdge;
using planetmodel::MeshManifest;
using planetmodel::MeshTetrahedron;
using planetmodel::MeshTriangle;
using planetmodel::MeshVertex;
using planetmodel::PlusCell;
using planetmodel::WallsOfFaults;

/// A face in the plane z = 0, its normal along +z, a corner above it and
/// one below, and one in its plane.
const auto CORNERS = std::vector<MeshVertex>{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0},
                                             {0.0, 1.0, 0.0}, {0.2, 0.2, 1.0},
                                             {0.2, 0.2, -1.0}, {0.5, 0.5, 0.0}};

/// A fan of four faces in the plane z = 0 about the origin (vertex 0), its
/// rim 1..4, and the tetrahedra on them up to 5 and down to 6: the fault's
/// one slip-carrying vertex is the origin, its rim its tip.
const auto FAN = std::vector<MeshVertex>{{0.0, 0.0, 0.0},  {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
                                         {-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 1.0},
                                         {0.0, 0.0, -1.0}};
const auto FAN_FACES = std::vector<MeshTriangle>{{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 1}};
const auto FAN_CELLS = std::vector<MeshTetrahedron>{{0, 1, 2, 5}, {0, 2, 3, 5}, {0, 3, 4, 5},
                                                    {0, 4, 1, 5}, {0, 2, 1, 6}, {0, 3, 2, 6},
                                                    {0, 4, 3, 6}, {0, 1, 4, 6}};
const auto FAN_RIM = std::vector<MeshEdge>{{1, 2}, {2, 3}, {3, 4}, {1, 4}};

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
  // a square of two faces whose edge (0, 1) is on the surface, a second
  // fault of one face meeting it along (2, 3), both ending there, and a
  // third of two faces with (2, 3) inside it, which the square ends on
  const auto faults = std::vector<std::vector<MeshTriangle>>{
      {{0, 1, 2}, {1, 3, 2}}, {{3, 2, 4}}, {{3, 2, 5}, {2, 3, 6}}};
  const auto surface = std::vector<MeshTriangle>{{0, 1, 9}};
  const auto curves = FaultCurves(faults, surface);
  ASSERT_EQ(curves.size(), 3u);
  EXPECT_EQ(curves[0].free_edge, (std::vector<MeshEdge>{{0, 1}}));
  EXPECT_EQ(curves[0].tip, (std::vector<MeshEdge>{{0, 2}, {1, 3}}));
  EXPECT_EQ(curves[0].junction, (std::vector<MeshEdge>{{2, 3}}));
  EXPECT_EQ(curves[0].junction_end, (std::vector<MeshEdge>{{2, 3}}));
  EXPECT_EQ(curves[0].ends_on, (std::vector<int>{1, 2}));
  EXPECT_EQ(curves[0].Pinned(), (std::vector<MeshEdge>{{0, 2}, {1, 3}, {2, 3}}));
  EXPECT_TRUE(curves[1].free_edge.empty());
  EXPECT_EQ(curves[1].tip, (std::vector<MeshEdge>{{2, 4}, {3, 4}}));
  EXPECT_EQ(curves[1].junction, (std::vector<MeshEdge>{{2, 3}}));
  EXPECT_EQ(curves[1].ends_on, (std::vector<int>{0, 2}));
  EXPECT_EQ(curves[2].junction, (std::vector<MeshEdge>{{2, 3}}));
  EXPECT_TRUE(curves[2].junction_end.empty());
  EXPECT_TRUE(curves[2].ends_on.empty());
  EXPECT_EQ(curves[2].Pinned(), curves[2].tip);
  const auto alone = FaultCurves(std::vector<std::vector<MeshTriangle>>{{{5, 4, 6}}}, {});
  EXPECT_EQ(alone[0].tip, (std::vector<MeshEdge>{{4, 5}, {4, 6}, {5, 6}}));
  EXPECT_TRUE(FaultCurves({}, surface).empty());
}

TEST(Faults, TheWallsAreTheTwoSidesOfTheCellsHoldingTheSlip) {
  const auto walls = FaultWalls(FAN, FAN_CELLS, FAN_FACES, FAN_RIM, "fan");
  EXPECT_EQ(walls, (std::vector<int>{1, 1, 1, 1, -1, -1, -1, -1}));
  // turned over, the faces' order says the other side
  auto turned = FAN_FACES;
  for (auto& f : turned) std::swap(f[1], f[2]);
  EXPECT_EQ(FaultWalls(FAN, FAN_CELLS, turned, FAN_RIM),
            (std::vector<int>{-1, -1, -1, -1, 1, 1, 1, 1}));
  // every vertex pinned and every edge of the faces pinned: nothing slips
  const auto all = std::vector<MeshEdge>{{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2}, {2, 3}, {3, 4},
                                         {1, 4}};
  EXPECT_EQ(FaultWalls(FAN, FAN_CELLS, FAN_FACES, all), std::vector<int>(8, 0));
  // the rule over all faults at once, the rim read from the faces
  const auto faults = std::vector<std::vector<MeshTriangle>>{FAN_FACES};
  EXPECT_EQ(WallsOfFaults(FAN, FAN_CELLS, faults, {}),
            (std::vector<std::vector<int>>{walls}));
}

TEST(Faults, TheWallsAreRefusedWhereTheSidesCannotBeTold) {
  auto turned = FAN_FACES;
  std::swap(turned[0][1], turned[0][2]);
  EXPECT_TRUE(RefusedBy([&] { return FaultWalls(FAN, FAN_CELLS, turned, FAN_RIM, "fan"); },
                        "FaultWalls", "fan"));
  auto fewer = FAN_CELLS;
  fewer.pop_back();
  EXPECT_TRUE(RefusedBy([&] { return FaultWalls(FAN, fewer, FAN_FACES, FAN_RIM, "fan"); },
                        "FaultWalls", "fan"));
  const auto outside = std::vector<MeshEdge>{{1, 9}};
  EXPECT_TRUE(RefusedBy([&] { return FaultWalls(FAN, FAN_CELLS, FAN_FACES, outside, "fan"); },
                        "FaultWalls", "fan"));
  // a cell indexing no vertex is refused wherever the index stands among
  // its corners: here after a corner that carries slip, in two cells that
  // share the face holding it
  const auto vertices = std::vector<MeshVertex>{
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, -1}, {2, 2, 2}, {3, 2, 2}};
  const auto face = std::vector<MeshTriangle>{{0, 1, 2}};
  const auto cells = std::vector<MeshTetrahedron>{
      {0, 1, 2, 3}, {0, 1, 2, 4}, {0, 5, 6, 7}, {1, 5, 6, 7}};
  try {
    static_cast<void>(FaultWalls(vertices, cells, face, {}, "plane"));
    ADD_FAILURE() << "a cell indexing vertex 7 of 7 was accepted";
  } catch (const Error& e) {
    EXPECT_EQ(e.Where(), "plane");
    EXPECT_NE(std::string{e.what()}.find("indexes vertex 7, outside the 7 given"),
              std::string::npos)
        << e.what();
  }
  try {
    static_cast<void>(FaultWalls(FAN, FAN_CELLS, turned, FAN_RIM, "fan"));
  } catch (const Error& e) {
    EXPECT_NE(std::string{e.what()}.find("on both sides of its faces"), std::string::npos)
        << e.what();
  }
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

      // the walls, and the collar attributes the mesher gave the cells
      const auto pinned = curves[k].Pinned();
      const auto found = FaultWalls(mesh.vertices, mesh.cells, faces[k], pinned, fault.name);
      auto want_walls = walls[fault.name];
      want_walls.resize(mesh.cells.size(), 0);
      EXPECT_EQ(found, want_walls) << fault.name;
      auto hanging = 0;
      for (auto c = std::size_t{0}; c < mesh.cells.size(); ++c) {
        const auto in = std::ranges::contains(fault.hanging_wall, mesh.layers[c]);
        EXPECT_EQ(in, found[c] == 1) << fault.name << " cell " << c;
        hanging += found[c] == 1;
      }
      EXPECT_GT(hanging, 0) << fault.name;
    }
    auto names = std::vector<std::string>{};
    for (const auto& f : m.Faults()) names.push_back(f.name);
    const auto all = WallsOfFaults(mesh.vertices, mesh.cells, faces, surface, names);
    ASSERT_EQ(all.size(), m.Faults().size());
    for (auto k = std::size_t{0}; k < all.size(); ++k) {
      auto want_walls = walls[m.Faults()[k].name];
      want_walls.resize(mesh.cells.size(), 0);
      EXPECT_EQ(all[k], want_walls);
    }
    for (const auto a : mesh.layers) EXPECT_NO_THROW(static_cast<void>(m.LayerOf(a)));
    ++cases;
  }
  EXPECT_EQ(cases, 3);
}
