#ifndef PLANETMODEL_MANIFEST_HPP
#define PLANETMODEL_MANIFEST_HPP

/**
 * @file
 * @brief The mesh manifest: the JSON that travels beside every mesh the
 * Python's 3D mesher writes, as `planetmodel/mesh3d/manifest.py` defines
 * it and the format page describes it.
 *
 * A mesh file carries numbered attributes and nothing else; the manifest
 * says which attribute is which layer and interface, where each sits,
 * whether the nodes carry the mapping, how the mesh must be opened, and
 * which files beside it hold which fields.  `ReadMeshManifest(path)`, or
 * the constructor `MeshManifest(path)` it calls, reads one and holds it
 * to the structure the Python's `validate_structure` checks,
 * refusing the first fault with an `Error` naming the file and the JSON
 * path of the fault; schemas 5 (a mesh without faults) and 6 (a mesh with
 * them) are read, and schema 7, a flat mesh's, is refused by name:
 * `planetmodel::cartesian::MeshManifest`
 * (`planetmodel/cartesian/manifest.hpp`) reads it.  Keys a schema added within itself are optional
 * (`refinements` empty, `scales.temperature` a kelvin, `model` null where
 * absent), and keys it does not know are ignored.  Null is kept as null: a
 * manifest written without a model says nothing of fluidity, kinds or
 * interface values, and the queries that need them say so.
 *
 * A fault built into the mesh is an interior surface of boundary elements
 * of its own attribute, after the interfaces' (`MeshManifest::Fault`),
 * with no node split: splitting them is the code's.  The cells of its
 * hanging wall, its collar, carry element attributes of their own after
 * the layers' (`Fault::hanging_wall`), so that a layer's cells carry a
 * list of attributes (`Layer::attributes`, `ElementAttributes`,
 * `LayerOf`).  Three rules of the format page turn a fault's faces into
 * what a split-node code needs, stated here over plain arrays (vertex
 * coordinates, corners as indices) as the Python's
 * `planetmodel.mesh3d.manifest` states them: which of a face's two cells
 * is the hanging-wall (plus) one (`FaultSide`, `PlusCell`); which edges of
 * a fault's faces are its tip, free edge and junction, and of its
 * junction where it ends, read from the faces alone where the mesh format
 * keeps no curves (`FaultCurves`); and which cells are in its hanging
 * wall and which in its footwall (`FaultWalls`, `WallsOfFaults`), which
 * the mesher wrote as the collar attributes.
 *
 * Nothing here knows MFEM.  A code opens the mesh and its fields itself,
 * with what the manifest gives:
 *
 * @code
 * const auto m = planetmodel::ReadMeshManifest(path);
 * const auto o = m.Options();
 * auto mesh = mfem::Mesh(m.MeshFile().string(), o.generate_edges, o.refine,
 *                        o.fix_orientation);
 * const auto& rho = m.FieldNamed("rho");
 * auto in = std::ifstream(m.FieldFile("rho"));
 * auto gf = mfem::GridFunction(&mesh, in);   // in rho.fe_space, vdim
 * @endcode
 *
 * builds the indicator of a fault's hanging wall from the element
 * attributes alone, a marker over `attributes.Max()` entries (the
 * attributes survive partitioning and uniform refinement):
 *
 * @code
 * const auto& fault = m.FaultNamed("megathrust");
 * const auto chi = planetmodel::MeshManifest::Marker(fault.hanging_wall,
 *                                                     mesh.attributes.Max());
 * const auto mantle = planetmodel::MeshManifest::Marker(
 *     m.LayerNamed("mantle").attributes, mesh.attributes.Max());
 * @endcode
 *
 * and finds the two sides of a fault's faces the same way:
 *
 * @code
 * for (auto be = 0; be < mesh.GetNBE(); ++be) {
 *   if (mesh.GetBdrAttribute(be) != fault.attribute) continue;
 *   auto e1 = 0, e2 = 0;
 *   mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(be), &e1, &e2);
 *   // the corners of be, e1 and e2 as MeshTriangle and MeshTetrahedron
 *   const auto plus = planetmodel::PlusCell(vertices, face, first, second) == 0 ? e1 : e2;
 * }
 * @endcode
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <format>
#include <iterator>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "planetmodel/character.hpp"
#include "planetmodel/error.hpp"
#include "planetmodel/json.hpp"
#include "planetmodel/units.hpp"

namespace planetmodel {

/**
 * @brief The schema of a manifest whose mesh has no fault, the cells of
 * layer n all carrying attribute n; bumped only for an incompatible change.
 */
inline constexpr std::string_view MANIFEST_SCHEMA = "planetmodel.mesh.manifest/5";

/**
 * @brief The schema of a manifest whose mesh has faults: the cells of the
 * faults' collars carry attributes beyond the layers', listed by each
 * layer (`attributes`) and each fault (`hanging_wall`).
 */
inline constexpr std::string_view MANIFEST_SCHEMA_WITH_FAULTS = "planetmodel.mesh.manifest/6";

/**
 * @brief The schema of a manifest whose mesh is flat-layered, a box or a
 * rectangle layered in depth, with faults or without, which `MeshManifest`
 * refuses by name: `planetmodel::cartesian::MeshManifest`
 * (`planetmodel/cartesian/manifest.hpp`) reads it.
 */
inline constexpr std::string_view MANIFEST_SCHEMA_FLAT = "planetmodel.mesh.manifest/7";

/** @brief The formats a manifest's mesh may be in: the mesher's MSH, or MFEM's. */
inline constexpr auto MANIFEST_FORMATS = std::array<std::string_view, 2>{"msh", "mfem"};

/**
 * @brief What a mesh's node coordinates are: reference ones, or physical
 * ones that carry the mapping.
 */
inline constexpr auto MANIFEST_NODES =
    std::array<std::string_view, 2>{"reference", "physical"};

/** @brief What an interface's kind may be, where a model said. */
inline constexpr auto INTERFACE_KINDS = std::array<std::string_view, 5>{
    "solid-solid", "fluid-solid", "fluid-fluid", "free", "outer"};

/**
 * @brief The `mfem::Mesh` constructor arguments the dof numbering of a
 * manifest's fields was written under, the documented ones where it gives
 * none.
 */
struct ReadOptions {
  int generate_edges = 1;       /**< Whether MFEM generates the edges. */
  int refine = 0;               /**< MFEM's refine argument. */
  bool fix_orientation = false; /**< Whether MFEM fixes the elements' orientation. */

  /** @brief Options are equal when all three are. */
  friend bool operator==(const ReadOptions&, const ReadOptions&) = default;
};

namespace detail {

/// What a fault's `controls` must be.
inline constexpr std::string_view MANIFEST_CONTROLS_RULE =
    "is not [n_u, n_v] per patch, each at least 4";

/// The items of a list, comma separated.
template <class Item>
std::string JoinedList(const std::vector<Item>& items) {
  auto out = std::string{};
  for (auto first = true; const auto& item : items) {
    out += std::format("{}{}", first ? "" : ", ", item);
    first = false;
  }
  return out;
}

/// A list block of a manifest's document: absent is empty.
inline const std::vector<Json>& ManifestList(const Json& document, std::string_view key) {
  static const auto none = std::vector<Json>{};
  const auto* v = document.Find(key);
  return v ? v->AsArray() : none;
}

inline std::optional<double> OptionalNumber(const Json& v) {
  return v.IsNull() ? std::nullopt : std::optional<double>{v.AsNumber()};
}

inline std::optional<int> OptionalInt(const Json& v) {
  return v.IsNull() ? std::nullopt : std::optional<int>{v.AsInt()};
}

/// The `mesh` block of a manifest, as every schema has it.
struct ManifestMesh {
  std::string file{};
  std::string format{};
  std::string nodes{};
  ReadOptions options{};
  std::optional<std::string> displacement{};
};

inline ManifestMesh ReadManifestMesh(const Json& mesh) {
  auto out = ManifestMesh{};
  out.file = mesh.At("file").AsString();
  out.format = mesh.At("format").AsString();
  out.nodes = mesh.At("nodes").AsString();
  const auto& options = mesh.At("read_options");
  if (const auto* v = options.Find("generate_edges")) out.options.generate_edges = v->AsInt();
  if (const auto* v = options.Find("refine")) out.options.refine = v->AsInt();
  if (const auto* v = options.Find("fix_orientation")) out.options.fix_orientation = v->AsBool();
  if (const auto& v = mesh.At("displacement"); !v.IsNull()) out.displacement = v.AsString();
  return out;
}

/// What an interface record says beside its place, into a record with
/// `between_layers`, `kind` and `values`: the same keys under every schema.
template <class Interface, class Sides>
void ReadManifestInterface(const Json& e, Interface& face, const std::string& source) {
  const auto& between = e.At("between_layers");
  if (between.Size() != 2) throw Error(source, between.Path(), "is not [below, above]");
  face.between_layers = {between.At(0).AsInt(), between.At(1).AsInt()};
  if (const auto& k = e.At("kind"); !k.IsNull()) face.kind = k.AsString();
  if (const auto& v = e.At("values"); !v.IsNull()) {
    auto values = std::map<std::string, Sides>{};
    for (const auto& [key, pair] : v.AsObject()) {
      if (!pair.IsArray() || pair.Size() != 2 ||
          !std::ranges::all_of(pair.AsArray(), [](const Json& x) {
            return x.IsNull() || x.IsNumber();
          })) {
        throw Error(source, pair.Path(), "is not [below, above] as numbers or nulls");
      }
      values[key] = {OptionalNumber(pair.At(0)), OptionalNumber(pair.At(1))};
    }
    face.values = std::move(values);
  }
}

/// The `refinements[]` records.
template <class Refinement>
std::vector<Refinement> ReadManifestRefinements(const Json& document) {
  auto out = std::vector<Refinement>{};
  for (const auto& e : ManifestList(document, "refinements")) {
    auto r = Refinement{.name = e.At("name").AsString(),
                        .size = e.At("size").AsNumber(),
                        .far_size = e.At("far_size").AsNumber(),
                        .decay_width = OptionalNumber(e.At("decay_width"))};
    if (const auto& p = e.At("points"); !p.IsNull()) r.points = p.AsInt();
    if (const auto& d = e.At("digest"); !d.IsNull()) r.digest = d.AsString();
    out.push_back(std::move(r));
  }
  return out;
}

/// What a `faults[]` record says under every schema that has faults, into
/// a record with those members.
template <class Fault, class Sizing>
void ReadManifestFault(const Json& e, Fault& f, const std::string& source) {
  f.name = e.At("name").AsString();
  f.attribute = e.At("attribute").AsInt();
  for (const auto& a : e.At("layers").AsArray()) f.layers.push_back(a.AsInt());
  for (const auto& a : e.At("hanging_wall").AsArray()) f.hanging_wall.push_back(a.AsInt());
  f.tip = OptionalInt(e.At("tip"));
  f.free_edge = OptionalInt(e.At("free_edge"));
  f.junction = OptionalInt(e.At("junction"));
  for (const auto& other : e.At("meets").AsArray()) f.meets.push_back(other.AsString());
  for (const auto& other : e.At("ends_on").AsArray()) f.ends_on.push_back(other.AsString());
  if (const auto& c = e.At("controls"); !c.IsNull()) {
    auto controls = std::vector<std::array<int, 2>>{};
    for (const auto& pair : c.AsArray()) {
      if (!pair.IsArray() || pair.Size() != 2) {
        throw Error(source, c.Path(), MANIFEST_CONTROLS_RULE);
      }
      controls.push_back({pair.At(0).AsInt(), pair.At(1).AsInt()});
    }
    f.controls = std::move(controls);
  }
  f.offset = e.At("offset").AsNumber();
  if (const auto& s = e.At("sizing"); !s.IsNull()) {
    static_cast<void>(s.AsObject());
    f.sizing = Sizing{.size = s.At("size").AsNumber(),
                      .far_size = s.At("far_size").AsNumber(),
                      .decay_width = s.At("decay_width").AsNumber()};
  }
}

/// The `fields[]` records.
template <class Field>
std::vector<Field> ReadManifestFields(const Json& document) {
  auto out = std::vector<Field>{};
  for (const auto& e : ManifestList(document, "fields")) {
    auto f = Field{.name = e.At("name").AsString(),
                   .file = e.At("file").AsString(),
                   .fe_space = e.At("fe_space").AsString(),
                   .vdim = e.At("vdim").AsInt(),
                   .ordering = e.At("ordering").AsString(),
                   .rank = e.At("rank").AsInt(),
                   .weight = e.At("weight").AsInt(),
                   .voigt = e.At("voigt").AsBool(),
                   .unit = e.At("unit").AsString()};
    for (const auto& a : e.At("layers").AsArray()) f.layers.push_back(a.AsInt());
    if (const auto& d = e.At("radial_degree"); !d.IsNull()) f.radial_degree = d.AsInt();
    out.push_back(std::move(f));
  }
  return out;
}

/// The `scales`, `constants`, `model` and `meta` blocks.
template <class ModelRecord>
void ReadManifestModelBlocks(const Json& doc, const std::string& source,
                             std::optional<::planetmodel::Scales>& scales,
                             std::map<std::string, double>& constants,
                             std::optional<ModelRecord>& model, Json& meta) {
  if (const auto* s = doc.Find("scales"); s && !s->IsNull()) {
    const auto* t = s->Find("temperature");
    try {
      scales = ::planetmodel::Scales(
          {.length = s->At("length").AsNumber(),
           .mass = s->At("mass").AsNumber(),
           .time = s->At("time").AsNumber(),
           .temperature = t ? t->AsNumber() : 1.0});
    } catch (const Error& e) {
      if (e.Source() != "Scales") throw;
      throw Error(source, std::format("scales.{}", e.Where()),
                  "is not a positive finite number");
    }
  }
  if (const auto* c = doc.Find("constants")) {
    for (const auto& [key, value] : c->AsObject()) constants[key] = value.AsNumber();
  }
  if (const auto* m = doc.Find("model"); m && !m->IsNull()) {
    model = ModelRecord{m->At("file").AsString(), m->At("schema").AsString()};
  }
  if (const auto* found = doc.Find("meta")) {
    static_cast<void>(found->AsObject());
    meta = *found;
  } else {
    meta = ReadJson("{}", source);
  }
}

/// The `mesh` block's format and nodes.
inline void CheckManifestMesh(const ManifestMesh& mesh, const std::string& source) {
  if (std::ranges::find(MANIFEST_FORMATS, std::string_view{mesh.format}) ==
      MANIFEST_FORMATS.end()) {
    throw Error(source, "mesh.format", std::format("is {}, not msh or mfem", mesh.format));
  }
  if (std::ranges::find(MANIFEST_NODES, std::string_view{mesh.nodes}) == MANIFEST_NODES.end()) {
    throw Error(source, "mesh.nodes",
                std::format("is {}, not reference or physical", mesh.nodes));
  }
}

/// The `kind` of interface `i`, one of `INTERFACE_KINDS` where a model said.
template <class Interface>
void CheckManifestKind(const Interface& f, int i, const std::string& source) {
  if (f.kind && std::ranges::find(INTERFACE_KINDS, std::string_view{*f.kind}) ==
                    INTERFACE_KINDS.end()) {
    throw Error(source, std::format("interfaces[{}].kind", i),
                std::format("is {}, not a kind of interface or null", *f.kind));
  }
}

/// The `refinements[]` records: sizes, widths and counts.
template <class Refinement>
void CheckManifestRefinements(const std::vector<Refinement>& refinements,
                              const std::string& source) {
  for (auto i = std::size_t{0}; i < refinements.size(); ++i) {
    const auto& r = refinements[i];
    const auto where = std::format("refinements[{}]", i);
    if (!(r.size > 0.0) || r.far_size < r.size) {
      throw Error(source, where,
                  std::format("has size {} and far_size {}, not 0 < size <= far_size", r.size,
                              r.far_size));
    }
    if (r.decay_width && !(*r.decay_width > 0.0)) {
      throw Error(source, where + ".decay_width",
                  std::format("is {}, not positive", *r.decay_width));
    }
    if (r.points && *r.points < 1) {
      throw Error(source, where + ".points", std::format("is {}, not a count", *r.points));
    }
  }
}

/// The `fields[]` records against the layers, the displacement the `mesh`
/// block names, and the fields the interfaces' `values` name.
template <class Field, class Interface>
void CheckManifestFields(const std::vector<Field>& fields,
                         const std::vector<Interface>& interfaces, int n_layers,
                         const ManifestMesh& mesh, const std::string& source) {
  auto names = std::vector<std::string>{};
  for (const auto& f : fields) names.push_back(f.name);
  for (auto i = std::size_t{0}; i < fields.size(); ++i) {
    const auto& f = fields[i];
    const auto where = std::format("fields[{}]", i);
    if (std::ranges::count(names, f.name) > 1) {
      throw Error(source, where + ".name", std::format("{} is repeated", f.name));
    }
    for (const auto a : f.layers) {
      if (a < 1 || a > n_layers) {
        throw Error(source, where + ".layers",
                    std::format("holds {}, not an attribute in 1..{}", a, n_layers));
      }
    }
    if (f.radial_degree && *f.radial_degree < 0) {
      throw Error(source, where + ".radial_degree",
                  std::format("is {}, not a degree", *f.radial_degree));
    }
    auto stored = std::size_t{0};
    try {
      stored = f.GetCharacter().StoredSize();
    } catch (const Error& e) {
      throw Error(source, where, std::format("has no character: {}", e.what()));
    }
    // a vector of the mesh's own space, the displacement in 2D, has two
    if (static_cast<std::size_t>(f.vdim) != stored && !(f.rank == 1 && f.vdim == 2)) {
      throw Error(source, where + ".vdim",
                  std::format("is {}, but a field of rank {}{} stores {} components", f.vdim,
                              f.rank, f.voigt ? "" : " without Voigt form", stored));
    }
  }
  if (mesh.displacement) {
    if (!std::ranges::contains(names, *mesh.displacement)) {
      throw Error(source, "mesh.displacement",
                  std::format("names {}, which fields holds no record for",
                              *mesh.displacement));
    }
    if (mesh.nodes == "physical") {
      throw Error(source, "mesh.displacement",
                  "is recorded with physical nodes; a mesh whose nodes are already moved has "
                  "no displacement to apply");
    }
  }
  for (auto i = std::size_t{0}; i < interfaces.size(); ++i) {
    if (!interfaces[i].values) continue;
    for (const auto& [key, sides] : *interfaces[i].values) {
      if (!std::ranges::contains(names, key)) {
        throw Error(source, std::format("interfaces[{}].values", i),
                    std::format("names {}, which fields holds no record for", key));
      }
    }
  }
}

/// The collars of a mesh with faults: each layer's attributes its own and
/// then collar attributes above N in increasing order, none for a shell;
/// the collar attributes N + 1..N + C, each one layer's and in some fault's
/// hanging wall; each hanging wall collar attributes in increasing order;
/// and the collars numbered in the order of their layer and then of their
/// faults.
template <class Layer, class Fault>
void CheckManifestCollars(const std::vector<Layer>& layers, const std::vector<Fault>& faults,
                          const std::string& source) {
  const auto n = static_cast<int>(layers.size());
  auto owner = std::map<int, int>{};  // collar attribute -> the layer's own
  for (auto i = 0; i < n; ++i) {
    const auto& l = layers[static_cast<std::size_t>(i)];
    const auto where = std::format("layers[{}].attributes", i);
    const auto& a = l.attributes;
    auto ordered = !a.empty() && a.front() == l.attribute;
    for (auto j = std::size_t{1}; ordered && j < a.size(); ++j) {
      ordered = a[j] > n && (j == 1 || a[j] > a[j - 1]);
    }
    if (!ordered) {
      throw Error(source, where,
                  std::format("is [{}], not its own attribute {} and then collar attributes "
                              "above {} in increasing order",
                              JoinedList(a), l.attribute, n));
    }
    if (a.size() > 1 && !l.in_geometry) {
      throw Error(source, where, "lists collars, and a shell holds no fault");
    }
    for (auto j = std::size_t{1}; j < a.size(); ++j) {
      if (owner.contains(a[j])) {
        throw Error(source, where,
                    std::format("lists collar {}, which layers[{}] lists too", a[j],
                                owner[a[j]] - 1));
      }
      owner[a[j]] = l.attribute;
    }
  }
  const auto c = static_cast<int>(owner.size());
  if (!owner.empty() && (owner.begin()->first != n + 1 || owner.rbegin()->first != n + c)) {
    auto listed = std::vector<int>{};
    for (const auto& [a, _] : owner) listed.push_back(a);
    throw Error(source, "layers",
                std::format("list the collar attributes [{}], not {}..{}", JoinedList(listed),
                            n + 1, n + c));
  }
  auto walls = std::map<int, std::vector<int>>{};
  for (auto k = std::size_t{0}; k < faults.size(); ++k) {
    const auto& hw = faults[k].hanging_wall;
    auto ordered = true;
    for (auto j = std::size_t{0}; ordered && j < hw.size(); ++j) {
      ordered = owner.contains(hw[j]) && (j == 0 || hw[j] > hw[j - 1]);
    }
    if (!ordered) {
      throw Error(source, std::format("faults[{}].hanging_wall", k),
                  std::format("is [{}], not collar attributes in increasing order",
                              JoinedList(hw)));
    }
    for (const auto a : hw) walls[a].push_back(static_cast<int>(k));
  }
  auto previous = std::optional<std::pair<int, std::vector<int>>>{};
  for (const auto& [a, layer] : owner) {
    const auto where = std::format("layers[{}].attributes", layer - 1);
    if (!walls.contains(a)) {
      throw Error(source, where,
                  std::format("lists collar {}, which is in no fault's hanging wall", a));
    }
    auto key = std::pair{layer, walls[a]};
    if (previous && key <= *previous) {
      throw Error(source, where,
                  std::format("lists collar {} (layer {}, faults [{}]) out of the order of "
                              "layer and faults",
                              a, layer, JoinedList(walls[a])));
    }
    previous = std::move(key);
  }
}

/// The faults against each other and the skeleton: names unique and not an
/// interface's (nor one of `others`, a flat mesh's faces, named `other_kind`),
/// the attributes running on from `before` (the boundary attributes that
/// come first, described by `after`), layers of the geometry, the curve
/// groups fault k's (`curves(fault)` its groups in the order of the mesh's
/// curves, with their keys), meetings mutual and with a junction to show
/// for them, the pull-back and sizing well formed, and the collars.
template <class Layer, class Interface, class Fault, class Curves>
void CheckManifestFaults(const std::vector<Layer>& layers,
                         const std::vector<Interface>& interfaces,
                         const std::vector<Fault>& faults, int before, std::string_view after,
                         const std::vector<std::string>& others, std::string_view other_kind,
                         Curves curves, const std::string& source) {
  auto body = std::vector<int>{};  // the geometry's layers, by their own attributes
  for (const auto& l : layers) {
    if (l.in_geometry) body.push_back(l.attribute);
  }
  auto named = [&](std::string_view name) -> const Fault* {
    for (const auto& f : faults) {
      if (f.name == name) return &f;
    }
    return nullptr;
  };
  for (auto i = std::size_t{0}; i < faults.size(); ++i) {
    const auto& name = faults[i].name;
    if (std::ranges::count(faults, name, &Fault::name) > 1) {
      throw Error(source, std::format("faults[{}].name", i),
                  std::format("{} is repeated", name));
    }
  }
  for (auto i = std::size_t{0}; i < faults.size(); ++i) {
    const auto& name = faults[i].name;
    if (std::ranges::count(interfaces, name, &Interface::name) > 0) {
      throw Error(source, std::format("faults[{}].name", i),
                  std::format("{} shares its name with an interface", name));
    }
    if (std::ranges::contains(others, name)) {
      throw Error(source, std::format("faults[{}].name", i),
                  std::format("{} shares its name with {}", name, other_kind));
    }
  }
  for (auto i = std::size_t{0}; i < faults.size(); ++i) {
    const auto& f = faults[i];
    const auto k = static_cast<int>(i) + 1;
    const auto where = std::format("faults[{}]", i);
    if (f.attribute != before + k) {
      throw Error(source, where + ".attribute",
                  std::format("is {}, expected {}, after {}", f.attribute, before + k, after));
    }
    if (f.layers.empty() ||
        !std::ranges::all_of(f.layers, [&](int a) { return std::ranges::contains(body, a); })) {
      throw Error(source, where + ".layers",
                  std::format("is [{}], not attributes of layers of the geometry, [{}]",
                              JoinedList(f.layers), JoinedList(body)));
    }
    const auto groups = curves(f);
    const auto count = static_cast<int>(groups.size());
    for (auto j = 0; j < count; ++j) {
      const auto& [key, group] = groups[static_cast<std::size_t>(j)];
      if (const auto want = count * (k - 1) + 1 + j; group && *group != want) {
        throw Error(source, std::format("{}.{}", where, key),
                    std::format("is {}, expected {} or null", *group, want));
      }
    }
    if (!std::ranges::all_of(f.meets, [&](const std::string& other) {
          return other != f.name && named(other) != nullptr;
        })) {
      throw Error(source, where + ".meets",
                  std::format("is [{}], not the names of other faults", JoinedList(f.meets)));
    }
    if (!std::ranges::all_of(f.meets, [&](const std::string& other) {
          return named(other)->Meets(f.name);
        })) {
      throw Error(source, where + ".meets",
                  std::format("is [{}], and not every one of them meets {}",
                              JoinedList(f.meets), f.name));
    }
    if (f.meets.empty() == f.junction.has_value()) {
      throw Error(source, where + ".junction",
                  std::format("is {} but meets is [{}]: a junction is where it meets another "
                              "fault",
                              f.junction ? std::to_string(*f.junction) : "null",
                              JoinedList(f.meets)));
    }
    if (!std::ranges::all_of(f.ends_on, [&](const std::string& other) {
          return f.Meets(other) && std::ranges::count(f.ends_on, other) == 1;
        })) {
      throw Error(source, where + ".ends_on",
                  std::format("is [{}], not faults it meets, each once", JoinedList(f.ends_on)));
    }
    if (f.controls && !std::ranges::all_of(*f.controls, [](const std::array<int, 2>& c) {
          return c[0] >= 4 && c[1] >= 4;
        })) {
      throw Error(source, where + ".controls", MANIFEST_CONTROLS_RULE);
    }
    if (!(f.offset >= 0.0)) {
      throw Error(source, where + ".offset", std::format("is {}, not a distance", f.offset));
    }
    if (const auto& s = f.sizing;
        s && !(0.0 < s->size && s->size <= s->far_size && s->decay_width > 0.0)) {
      throw Error(source, where + ".sizing",
                  std::format("is {{size {}, far_size {}, decay_width {}}}, not 0 < size <= "
                              "far_size with a positive decay_width",
                              s->size, s->far_size, s->decay_width));
    }
  }
  if (!faults.empty()) CheckManifestCollars(layers, faults, source);
}

}  // namespace detail

/** @brief A mesh manifest, read and checked. */
class MeshManifest {
 public:
  /** @brief One `layers[]` record: a layer of the mesh, centre outward. */
  struct Layer {
    /** @brief The layer's own attribute, from one, by which records name it. */
    int attribute = 0;
    std::string name;      /**< The layer's name. */
    double r_inner = 0.0;  /**< Its inner radius. */
    double r_outer = 0.0;  /**< Its outer radius. */
    /** @brief False for a shell the mesh spec added outside the geometry. */
    bool in_geometry = true;
    /**
     * @brief Whether shear vanishes throughout the model's layer; empty
     * where no model said.
     */
    std::optional<bool> fluid{};
    /**
     * @brief The element attributes its cells carry: its own first, then
     * its collars' (schema 6); its own alone under schema 5.
     */
    std::vector<int> attributes{};
  };

  /**
   * @brief The values just inside each side of an interface, empty on a
   * side holding none.
   */
  struct Sides {
    std::optional<double> below{}; /**< Just inside the layer below. */
    std::optional<double> above{}; /**< Just inside the layer above. */
  };

  /** @brief One `interfaces[]` record: a boundary of the mesh, centre outward. */
  struct Interface {
    int attribute = 0;    /**< The boundary attribute, from one. */
    std::string name;     /**< The interface's name. */
    double radius = 0.0;  /**< Its radius. */
    /** @brief The layers below and above as 0-based indices, -1 for the outside. */
    std::array<int, 2> between_layers{};
    /** @brief One of `INTERFACE_KINDS`; empty where no model said. */
    std::optional<std::string> kind{};
    /**
     * @brief {field: sides} for the exported real scalar radial fields;
     * empty where no model said.
     */
    std::optional<std::map<std::string, Sides>> values{};

    /** @brief The attribute of the layer below, 0 for the outside. */
    [[nodiscard]] int BelowAttribute() const { return between_layers[0] + 1; }
    /** @brief The attribute of the layer above, 0 for the outside. */
    [[nodiscard]] int AboveAttribute() const { return between_layers[1] + 1; }

    /**
     * @brief A field's value just inside the side of a layer.
     * @param field The field.
     * @param attribute The layer's attribute, below or above.
     * @throws planetmodel::Error where that layer is not a side or the
     *         manifest holds no such value.
     */
    [[nodiscard]] double ValueBeside(std::string_view field, int attribute) const {
      if (attribute < 1 || (attribute != BelowAttribute() && attribute != AboveAttribute())) {
        throw Error("MeshManifest", name,
                    std::format("layer {} is not a side of this interface", attribute));
      }
      auto value = std::optional<double>{};
      if (values) {
        if (const auto it = values->find(std::string{field}); it != values->end()) {
          value = attribute == BelowAttribute() ? it->second.below : it->second.above;
        }
      }
      if (!value) {
        throw Error("MeshManifest", name,
                    std::format("holds no value of {} on the side of layer {}; the "
                                "values are written by an export with a model",
                                field, attribute));
      }
      return *value;
    }
  };

  /**
   * @brief One `refinements[]` record: an extra size field the mesh was
   * sized under, for provenance alone.
   */
  struct Refinement {
    std::string name;                     /**< The refinement's name. */
    double size = 0.0;                    /**< The element size it asked for. */
    double far_size = 0.0;                /**< The size far from it. */
    std::optional<double> decay_width{};  /**< How fast the size relaxes, where given. */
    std::optional<int> points{};          /**< Its number of points, where given. */
    std::optional<std::string> digest{};  /**< A digest of what it was sized by, where given. */
  };

  /** @brief One `fields[]` record: a file beside the mesh and what it holds. */
  struct Field {
    std::string name{};   /**< The field's name. */
    std::string file{};   /**< Its file, beside the manifest. */
    /** @brief MFEM's collection name, `L2_3D_P2` say. */
    std::string fe_space;
    int vdim = 1;         /**< The components per dof. */
    /** @brief `byNODES` or `byVDIM`. */
    std::string ordering;
    int rank = 0;         /**< The character's rank. */
    int weight = 0;       /**< The character's weight. */
    bool voigt = true;    /**< Whether the character is in Voigt form. */
    std::string unit;     /**< The unit of the stored values. */
    /**
     * @brief The layers on which the values mean anything, by their own
     * attributes (their cells carry `ElementAttributes` of them).
     */
    std::vector<int> layers{};
    /**
     * @brief The degree in r reproducing a scalar field within each layer
     * holding it; empty where no single degree does or no model said.
     */
    std::optional<int> radial_degree{};

    /** @brief The field's character. */
    [[nodiscard]] Character GetCharacter() const { return Character(rank, weight, voigt); }
  };

  /** @brief The `model` record: a model file beside the manifest. */
  struct ModelRecord {
    std::string file;    /**< The model file, beside the manifest. */
    std::string schema;  /**< The schema it was written under. */
  };

  /** @brief A fault's `sizing`: the refinement towards it the mesh was sized under. */
  struct FaultSizing {
    double size = 0.0;         /**< The element size on the fault. */
    double far_size = 0.0;     /**< The element size far from it. */
    double decay_width = 0.0;  /**< The distance over which the size grows. */
  };

  /**
   * @brief One `faults[]` record: a fault built into the mesh as an
   * interior surface, one conforming sheet of boundary elements with no
   * node split.
   *
   * Every face is written with its corners in the order that makes
   * (v1 - v0) x (v2 - v0) point from the fault's footwall into its
   * hanging wall, so the cell on that side of a face is the plus one
   * (`PlusCell`).  The curve groups are those of the MSH file, fault k's
   * numbered 3k - 2, 3k - 1 and 3k; a mesh format without curves gives
   * the same edges from the faces (`FaultCurves`).  The cells of its
   * hanging wall (`FaultWalls`) carry the attributes `hanging_wall`.
   */
  struct Fault {
    /** @brief The fault's name, its boundary group's in the MSH file. */
    std::string name;
    /** @brief The boundary attribute of its faces, after the interfaces'. */
    int attribute = 0;
    /** @brief The layers its faces lie in, by their own attributes. */
    std::vector<int> layers{};
    /** @brief The element attributes of the cells of its hanging wall, its collar. */
    std::vector<int> hanging_wall{};
    /**
     * @brief The MSH file's curve group of its buried tip, where its sides
     * are joined; empty where it has none.
     */
    std::optional<int> tip{};
    /**
     * @brief The curve group of its edge on the model's outer surface,
     * where its sides are free; empty where it has none.
     */
    std::optional<int> free_edge{};
    /**
     * @brief The curve group of its curves shared with other faults;
     * empty where it meets none.
     */
    std::optional<int> junction{};
    /** @brief The names of the faults it meets along its junction. */
    std::vector<std::string> meets{};
    /**
     * @brief The names of the faults it ends on, of those it meets: its
     * slip is pinned along its junction with them, which lies on its own
     * boundary.
     */
    std::vector<std::string> ends_on{};
    /**
     * @brief Each patch's pull-back control grid [n_u, n_v]; empty where
     * the mapping is the identity and the fault was built in as given.
     */
    std::optional<std::vector<std::array<int, 2>>> controls{};
    /**
     * @brief The largest distance found between the fault and its
     * pull-back mapped forward, in the mesh's lengths: the scale of how
     * far a node of its faces lies off the fault once mapped.
     */
    double offset = 0.0;
    /** @brief The refinement towards it; empty where there was none. */
    std::optional<FaultSizing> sizing{};

    /**
     * @brief Whether it meets another fault along its junction.
     * @param other The other fault's name.
     */
    [[nodiscard]] bool Meets(std::string_view other) const {
      return std::ranges::any_of(meets, [&](const std::string& m) { return m == other; });
    }

    /**
     * @brief Whether it ends on another fault, its slip pinned along their
     * junction.
     * @param other The other fault's name.
     */
    [[nodiscard]] bool EndsOn(std::string_view other) const {
      return std::ranges::any_of(ends_on, [&](const std::string& m) { return m == other; });
    }
  };

  /**
   * @brief The manifest in a file, read and checked.
   * @param path The manifest.
   * @throws planetmodel::Error naming the file and the JSON path of the
   *         first fault.
   */
  explicit MeshManifest(const std::filesystem::path& path)
      : MeshManifest(ReadJsonFile(path), path) {}

  /**
   * @brief The manifest in a document already read.
   * @param document The document.
   * @param path Where it lies; the files it names are beside it.
   * @throws planetmodel::Error naming the file and the JSON path of the
   *         first fault.
   */
  MeshManifest(const Json& document, std::filesystem::path path)
      : path_(std::move(path)), document_(document) {
    Read();
    Check();
  }

  /** @brief The manifest's own path. */
  [[nodiscard]] const std::filesystem::path& Path() const noexcept { return path_; }

  /** @brief The schema it declares: `MANIFEST_SCHEMA` or `MANIFEST_SCHEMA_WITH_FAULTS`. */
  [[nodiscard]] const std::string& Schema() const noexcept { return schema_; }

  /** @brief The mesh file beside the manifest. */
  [[nodiscard]] std::filesystem::path MeshFile() const {
    return path_.parent_path() / mesh_.file;
  }
  /** @brief `msh` from the mesher, `mfem` from the exporter and from the
   * builder of a cubed sphere. */
  [[nodiscard]] const std::string& Format() const noexcept { return mesh_.format; }
  /** @brief `reference` or `physical`. */
  [[nodiscard]] const std::string& Nodes() const noexcept { return mesh_.nodes; }
  /** @brief Whether the mesh's nodes carry the mapping already. */
  [[nodiscard]] bool PhysicalNodes() const noexcept { return mesh_.nodes == "physical"; }
  /** @brief How the mesh must be opened for its fields' dof numbering to hold. */
  [[nodiscard]] const ReadOptions& Options() const noexcept { return mesh_.options; }
  /**
   * @brief The name of the displacement field, where one was written
   * beside a reference mesh.
   */
  [[nodiscard]] const std::optional<std::string>& Displacement() const noexcept {
    return mesh_.displacement;
  }

  /** @brief The layers, centre outward. */
  [[nodiscard]] const std::vector<Layer>& Layers() const noexcept { return layers_; }
  /** @brief The interfaces, centre outward. */
  [[nodiscard]] const std::vector<Interface>& Interfaces() const noexcept {
    return interfaces_;
  }
  /** @brief The refinements the mesh was sized under. */
  [[nodiscard]] const std::vector<Refinement>& Refinements() const noexcept {
    return refinements_;
  }
  /** @brief The fields written beside the mesh. */
  [[nodiscard]] const std::vector<Field>& Fields() const noexcept { return fields_; }
  /**
   * @brief The faults built into the mesh, in the order of their
   * attributes; empty for a mesh without one.
   */
  [[nodiscard]] const std::vector<Fault>& Faults() const noexcept { return faults_; }

  /**
   * @brief A layer by name.
   * @param name The layer's name.
   * @throws planetmodel::Error if the manifest has no such layer.
   */
  [[nodiscard]] const Layer& LayerNamed(std::string_view name) const {
    return Named(layers_, name, "layer");
  }
  /**
   * @brief An interface by name.
   * @param name The interface's name.
   * @throws planetmodel::Error if the manifest has no such interface.
   */
  [[nodiscard]] const Interface& InterfaceNamed(std::string_view name) const {
    return Named(interfaces_, name, "interface");
  }
  /**
   * @brief A field by name.
   * @param name The field's name.
   * @throws planetmodel::Error if the manifest has no such field.
   */
  [[nodiscard]] const Field& FieldNamed(std::string_view name) const {
    return Named(fields_, name, "field");
  }
  /**
   * @brief Whether a field was written beside the mesh.
   * @param name The field's name.
   */
  [[nodiscard]] bool HasField(std::string_view name) const {
    return std::ranges::any_of(fields_, [&](const auto& f) { return f.name == name; });
  }
  /**
   * @brief A field's file, beside the manifest.
   * @param name The field's name.
   * @throws planetmodel::Error if the manifest has no such field.
   */
  [[nodiscard]] std::filesystem::path FieldFile(std::string_view name) const {
    return path_.parent_path() / FieldNamed(name).file;
  }

  /**
   * @brief A fault by name.
   * @param name The fault's name.
   * @throws planetmodel::Error if the mesh has no such fault.
   */
  [[nodiscard]] const Fault& FaultNamed(std::string_view name) const {
    return Named(faults_, name, "fault");
  }
  /**
   * @brief Whether a fault of that name was built into the mesh.
   * @param name The fault's name.
   */
  [[nodiscard]] bool HasFault(std::string_view name) const {
    return std::ranges::any_of(faults_, [&](const auto& f) { return f.name == name; });
  }
  /** @brief The boundary attributes of the faults' faces, in order. */
  [[nodiscard]] std::vector<int> FaultAttributes() const {
    auto out = std::vector<int>{};
    for (const auto& f : faults_) out.push_back(f.attribute);
    return out;
  }
  /**
   * @brief Whether a boundary attribute is a fault's rather than an
   * interface's.
   * @param attribute The boundary attribute.
   */
  [[nodiscard]] bool IsFaultAttribute(int attribute) const noexcept {
    return std::ranges::any_of(faults_, [&](const auto& f) { return f.attribute == attribute; });
  }
  /**
   * @brief The fault whose faces carry a boundary attribute.
   * @param attribute The boundary attribute.
   * @throws planetmodel::Error if no fault's faces carry it.
   */
  [[nodiscard]] const Fault& FaultWithAttribute(int attribute) const {
    for (const auto& f : faults_) {
      if (f.attribute == attribute) return f;
    }
    throw Error(path_.string(), "faults",
                std::format("lists no fault of boundary attribute {}; the faults' are {}",
                            attribute, Join(FaultAttributes())));
  }

  /** @brief Zero for a full mesh, the inner boundary's radius for a hollow one. */
  [[nodiscard]] double InnerRadius() const { return layers_.front().r_inner; }
  /** @brief The outer radius of the mesh, shells included. */
  [[nodiscard]] double OuterRadius() const { return layers_.back().r_outer; }
  /** @brief Whether the mesh has a hole at the centre. */
  [[nodiscard]] bool IsHollow() const { return InnerRadius() > 0.0; }

  /**
   * @brief The layer whose cells carry an element attribute: its own, or
   * one of its collars'.
   * @param attribute The element attribute.
   * @throws planetmodel::Error where no layer's cells carry it.
   */
  [[nodiscard]] const Layer& LayerOf(int attribute) const {
    for (const auto& l : layers_) {
      if (std::ranges::contains(l.attributes, attribute)) return l;
    }
    throw Error(path_.string(), "layers",
                std::format("lists no layer whose cells carry element attribute {}; they "
                            "carry {}",
                            attribute, Join(ElementAttributes())));
  }
  /** @brief The element attributes every layer's cells carry, layer by layer. */
  [[nodiscard]] std::vector<int> ElementAttributes() const {
    return Select([](const Layer&) { return true; });
  }
  /**
   * @brief The element attributes the cells of some layers carry, each
   * layer's own and then its collars', in the order the layers are given.
   * The layers are named by their own attributes, as a field's `layers`
   * and a fault's name them: an MFEM marker for their cells is made from
   * this list.
   * @param layers The layers' own attributes.
   * @throws planetmodel::Error for a number that is no layer's own attribute.
   */
  [[nodiscard]] std::vector<int> ElementAttributes(const std::vector<int>& layers) const {
    auto out = std::vector<int>{};
    for (const auto a : layers) {
      if (a < 1 || a > static_cast<int>(layers_.size())) {
        throw Error(path_.string(), "layers",
                    std::format("has no layer of own attribute {}", a));
      }
      const auto& l = layers_[static_cast<std::size_t>(a - 1)];
      out.insert(out.end(), l.attributes.begin(), l.attributes.end());
    }
    return out;
  }

  /**
   * @brief The element attributes of the geometry's layers, their collars'
   * among them: a marker's list.
   */
  [[nodiscard]] std::vector<int> BodyAttributes() const {
    return Select([](const Layer& l) { return l.in_geometry; });
  }
  /** @brief The element attributes of the shells outside the geometry. */
  [[nodiscard]] std::vector<int> ShellAttributes() const {
    return Select([](const Layer& l) { return !l.in_geometry; });
  }
  /**
   * @brief The element attributes of the geometry's solid layers, their
   * collars' among them.
   * @throws planetmodel::Error where a layer's fluidity is unknown (no
   *         model said).
   */
  [[nodiscard]] std::vector<int> SolidAttributes() const {
    RequireFluidity();
    return Select([](const Layer& l) { return l.in_geometry && !*l.fluid; });
  }
  /**
   * @brief The element attributes of the geometry's fluid layers, their
   * collars' among them.
   * @throws planetmodel::Error where a layer's fluidity is unknown.
   */
  [[nodiscard]] std::vector<int> FluidAttributes() const {
    RequireFluidity();
    return Select([](const Layer& l) { return l.in_geometry && *l.fluid; });
  }
  /**
   * @brief The interfaces between a fluid and a solid layer of the
   * geometry.
   * @throws planetmodel::Error where a layer's fluidity is unknown.
   */
  [[nodiscard]] std::vector<int> FluidSolidInterfaces() const {
    RequireFluidity();
    auto out = std::vector<int>{};
    for (const auto& f : interfaces_) {
      const auto below = BodyLayer(f.BelowAttribute());
      const auto above = BodyLayer(f.AboveAttribute());
      if (below && above && *below->fluid != *above->fluid) out.push_back(f.attribute);
    }
    return out;
  }
  /**
   * @brief The interfaces between a fluid layer and a solid one.
   * @param fluid_attribute The fluid layer's attribute.
   * @throws planetmodel::Error where a layer's fluidity is unknown.
   */
  [[nodiscard]] std::vector<int> FluidSolidInterfaces(int fluid_attribute) const {
    RequireFluidity();
    auto solid = [&](int attribute) {
      const auto* layer = BodyLayer(attribute);
      return layer && !*layer->fluid;
    };
    auto out = std::vector<int>{};
    for (const auto& f : interfaces_) {
      if ((f.BelowAttribute() == fluid_attribute && solid(f.AboveAttribute())) ||
          (f.AboveAttribute() == fluid_attribute && solid(f.BelowAttribute()))) {
        out.push_back(f.attribute);
      }
    }
    return out;
  }
  /**
   * @brief The interface on the outside of the geometry's outermost layer.
   * @throws planetmodel::Error if the manifest lists none.
   */
  [[nodiscard]] int SurfaceAttribute() const {
    auto top = 0;  // the geometry's outermost layer, by its own attribute
    for (const auto& l : layers_) {
      if (l.in_geometry) top = l.attribute;
    }
    for (const auto& f : interfaces_) {
      if (top > 0 && f.BelowAttribute() == top) return f.attribute;
    }
    throw Error(path_.string(), "interfaces", "lists no surface of the geometry");
  }
  /**
   * @brief The radius of the surface, as `SurfaceAttribute` gives it.
   * @throws planetmodel::Error if the manifest lists none.
   */
  [[nodiscard]] double SurfaceRadius() const {
    return interfaces_[static_cast<std::size_t>(SurfaceAttribute() - 1)].radius;
  }
  /** @brief The outer boundary of the mesh. */
  [[nodiscard]] int OuterAttribute() const { return interfaces_.back().attribute; }

  /**
   * @brief MFEM's marker array for a list of attributes: one where the
   * attribute is listed, zero elsewhere.
   * @param attributes The attributes, from one.
   * @param size The number of entries.
   * @throws planetmodel::Error if an attribute is below one.
   */
  [[nodiscard]] static std::vector<int> Marker(const std::vector<int>& attributes, int size) {
    auto marker = std::vector<int>(static_cast<std::size_t>(std::max(size, 0)), 0);
    for (const auto a : attributes) {
      if (a < 1) throw Error("MeshManifest", "Marker", "attributes start at one");
      if (a <= size) marker[static_cast<std::size_t>(a - 1)] = 1;
    }
    return marker;
  }

  /** @brief Whether a model's fields were exported, and so the scales stated. */
  [[nodiscard]] bool HasScales() const noexcept { return scales_.has_value(); }
  /**
   * @brief The scales the field values and constants are in.
   * @throws planetmodel::Error where no model was exported.
   */
  [[nodiscard]] const ::planetmodel::Scales& GetScales() const {
    if (!scales_) {
      throw Error(path_.string(), "scales",
                  "is null: no model's fields were exported beside this mesh");
    }
    return *scales_;
  }
  /** @brief The model's constants, in its units. */
  [[nodiscard]] const std::map<std::string, double>& Constants() const noexcept {
    return constants_;
  }
  /**
   * @brief Whether the manifest holds a constant.
   * @param name The constant.
   */
  [[nodiscard]] bool HasConstant(std::string_view name) const {
    return constants_.contains(std::string{name});
  }
  /**
   * @brief A constant, in the model's units.
   * @param name The constant.
   * @throws planetmodel::Error where the manifest holds none of that name.
   */
  [[nodiscard]] double Constant(std::string_view name) const {
    if (const auto it = constants_.find(std::string{name}); it != constants_.end()) {
      return it->second;
    }
    throw Error(path_.string(), "constants",
                std::format("holds no constant {}", name));
  }
  /**
   * @brief The gravitational constant in the model's units.
   * @throws planetmodel::Error where the manifest holds none.
   */
  [[nodiscard]] double G() const { return Constant("G"); }

  /** @brief The model file beside the mesh, where one was written. */
  [[nodiscard]] const std::optional<ModelRecord>& Model() const noexcept { return model_; }
  /**
   * @brief The model file's path, beside the manifest.
   * @throws planetmodel::Error where none was written.
   */
  [[nodiscard]] std::filesystem::path ModelFile() const {
    if (!model_) throw Error(path_.string(), "model", "is null: no model file was written");
    return path_.parent_path() / model_->file;
  }

  /**
   * @brief The mesh spec's `meta`, as the document gives it; an empty
   * object where it gives none.
   */
  [[nodiscard]] const Json& Meta() const { return meta_; }

 private:
  // -- reading --------------------------------------------------------------

  void Read() {
    const auto& doc = document_;
    const auto source = path_.string();
    if (!doc.IsObject()) Fail("", "is not an object");
    ReadSchema(doc.At("schema"));
    const auto faulted = schema_ == MANIFEST_SCHEMA_WITH_FAULTS;
    if (!faulted && !detail::ManifestList(doc, "faults").empty()) {
      Fail("faults", std::format("are listed under schema {}; a manifest with faults is {}",
                                 schema_, MANIFEST_SCHEMA_WITH_FAULTS));
    }
    if (faulted && detail::ManifestList(doc, "faults").empty()) {
      Fail("faults", std::format("lists none, and a manifest of schema {} has faults",
                                 MANIFEST_SCHEMA_WITH_FAULTS));
    }

    mesh_ = detail::ReadManifestMesh(doc.At("mesh"));

    for (const auto& e : detail::ManifestList(doc, "layers")) {
      const auto& fluid = e.At("fluid");
      auto layer = Layer{.attribute = e.At("attribute").AsInt(),
                         .name = e.At("name").AsString(),
                         .r_inner = e.At("r_inner").AsNumber(),
                         .r_outer = e.At("r_outer").AsNumber(),
                         .in_geometry = e.At("in_geometry").AsBool(),
                         .fluid = fluid.IsNull() ? std::nullopt
                                                 : std::optional<bool>{fluid.AsBool()}};
      if (faulted) {
        for (const auto& a : e.At("attributes").AsArray()) layer.attributes.push_back(a.AsInt());
      } else {
        layer.attributes = {layer.attribute};
      }
      layers_.push_back(std::move(layer));
    }
    for (const auto& e : detail::ManifestList(doc, "interfaces")) {
      auto face = Interface{.attribute = e.At("attribute").AsInt(),
                            .name = e.At("name").AsString(),
                            .radius = e.At("radius").AsNumber()};
      detail::ReadManifestInterface<Interface, Sides>(e, face, source);
      interfaces_.push_back(std::move(face));
    }
    refinements_ = detail::ReadManifestRefinements<Refinement>(doc);
    for (const auto& e : detail::ManifestList(doc, "faults")) {
      auto f = Fault{};
      detail::ReadManifestFault<Fault, FaultSizing>(e, f, source);
      faults_.push_back(std::move(f));
    }
    fields_ = detail::ReadManifestFields<Field>(doc);
    detail::ReadManifestModelBlocks(doc, source, scales_, constants_, model_, meta_);
  }

  void ReadSchema(const Json& schema) {
    const auto& declared = schema.AsString();
    if (declared == MANIFEST_SCHEMA || declared == MANIFEST_SCHEMA_WITH_FAULTS) {
      schema_ = declared;
      return;
    }
    if (declared == MANIFEST_SCHEMA_FLAT) {
      Fail("schema", std::format("is {}, a flat mesh's, and this reader reads {} and {}: read "
                                 "it with planetmodel::cartesian::MeshManifest "
                                 "(planetmodel/cartesian/manifest.hpp)",
                                 declared, MANIFEST_SCHEMA, MANIFEST_SCHEMA_WITH_FAULTS));
    }
    if (declared.starts_with("planetmodel.mesh.manifest/")) {
      Fail("schema", std::format("is {}, and this reader reads {} and {}: regenerate the "
                                 "manifest with this planetmodel",
                                 declared, MANIFEST_SCHEMA, MANIFEST_SCHEMA_WITH_FAULTS));
    }
    Fail("schema", std::format("is {}, not a mesh manifest's", declared));
  }

  // -- checking, as validate_structure does ----------------------------------

  void Check() const {
    const auto source = path_.string();
    detail::CheckManifestMesh(mesh_, source);
    const auto n = static_cast<int>(layers_.size());
    if (n == 0) Fail("layers", "lists no layers");
    for (auto i = 0; i < n; ++i) {
      const auto& l = layers_[static_cast<std::size_t>(i)];
      const auto where = std::format("layers[{}]", i);
      if (l.attribute != i + 1) {
        Fail(where + ".attribute", std::format("is {}, expected {}", l.attribute, i + 1));
      }
      if (!(0.0 <= l.r_inner && l.r_inner < l.r_outer)) {
        Fail(where, std::format("runs from {} to {}, not 0 <= r_inner < r_outer",
                                l.r_inner, l.r_outer));
      }
      if (i > 0 && l.r_inner != layers_[static_cast<std::size_t>(i - 1)].r_outer) {
        Fail(where, std::format("starts at {} but the layer below ends at {}", l.r_inner,
                                layers_[static_cast<std::size_t>(i - 1)].r_outer));
      }
    }
    const auto m = static_cast<int>(interfaces_.size());
    if (m != n && m != n + 1) {
      Fail("interfaces", std::format("lists {} for {} layers; a full mesh has one per layer "
                                     "and a hollow mesh one more", m, n));
    }
    const auto first = m - n;  // 1 where the inner boundary is a face
    if ((layers_.front().r_inner > 0.0) != (first == 1)) {
      Fail("interfaces", std::format("lists {} for {} layers starting at r = {}: a full mesh "
                                     "starts at the centre and a hollow one has a face on "
                                     "its inner boundary",
                                     m, n, layers_.front().r_inner));
    }
    for (auto i = 0; i < m; ++i) {
      const auto& f = interfaces_[static_cast<std::size_t>(i)];
      const auto where = std::format("interfaces[{}]", i);
      if (f.attribute != i + 1) {
        Fail(where + ".attribute", std::format("is {}, expected {}", f.attribute, i + 1));
      }
      const auto j = i + 1 - first;  // the skeleton boundary the face sits on
      const auto want = std::array<int, 2>{j - 1, j < n ? j : -1};
      if (f.between_layers != want) {
        Fail(where + ".between_layers",
             std::format("is [{}, {}], expected [{}, {}]", f.between_layers[0],
                         f.between_layers[1], want[0], want[1]));
      }
      detail::CheckManifestKind(f, i, source);
    }
    detail::CheckManifestRefinements(refinements_, source);
    detail::CheckManifestFaults(
        layers_, interfaces_, faults_, m, std::format("the {} interfaces", m), {}, "",
        [](const Fault& f) {
          return std::array{std::pair{"tip", f.tip}, std::pair{"free_edge", f.free_edge},
                            std::pair{"junction", f.junction}};
        },
        source);
    detail::CheckManifestFields(fields_, interfaces_, n, mesh_, source);
  }

  // -- helpers -----------------------------------------------------------------

  [[noreturn]] void Fail(std::string_view where, std::string_view message) const {
    throw Error(path_.string(), where, message);
  }

  /// The items of a list, comma separated.
  template <class Item>
  static std::string Join(const std::vector<Item>& items) {
    return detail::JoinedList(items);
  }

  template <class Record>
  const Record& Named(const std::vector<Record>& records, std::string_view name,
                      std::string_view what) const {
    for (const auto& r : records) {
      if (r.name == name) return r;
    }
    auto known = std::string{};
    for (const auto& r : records) known += (known.empty() ? "" : ", ") + r.name;
    throw Error(path_.string(), "",
                std::format("has no {} named {}; it has {}", what, name, known));
  }

  template <class Predicate>
  std::vector<int> Select(Predicate keep) const {
    auto out = std::vector<int>{};
    for (const auto& l : layers_) {
      if (keep(l)) out.insert(out.end(), l.attributes.begin(), l.attributes.end());
    }
    return out;
  }

  /// The geometry's layer of `attribute`, or null for the outside or a
  /// shell.
  const Layer* BodyLayer(int attribute) const {
    if (attribute < 1 || attribute > static_cast<int>(layers_.size())) return nullptr;
    const auto& l = layers_[static_cast<std::size_t>(attribute - 1)];
    return l.in_geometry ? &l : nullptr;
  }

  void RequireFluidity() const {
    for (const auto& l : layers_) {
      if (l.in_geometry && !l.fluid) {
        throw Error(path_.string(), std::format("layers[{}].fluid", l.attribute - 1),
                    "is null: the mesh was exported without a model, so nothing says "
                    "which layers are fluid");
      }
    }
  }

  std::filesystem::path path_;
  std::string schema_;
  Json document_;
  Json meta_;
  detail::ManifestMesh mesh_{};
  std::vector<Layer> layers_;
  std::vector<Interface> interfaces_;
  std::vector<Refinement> refinements_;
  std::vector<Field> fields_;
  std::vector<Fault> faults_;
  std::optional<::planetmodel::Scales> scales_;
  std::map<std::string, double> constants_;
  std::optional<ModelRecord> model_;
};

/**
 * @brief Read a mesh manifest and hold it to the structure, as
 * `MeshManifest(path)` does: the function beside `ReadModelFile`,
 * `ReadSurfaceFile` and `ReadFaultFile`.
 * @param path The manifest.
 * @return The manifest, read and checked.
 * @throws planetmodel::Error naming the file and the JSON path of the
 *         first fault.
 */
[[nodiscard]] inline MeshManifest ReadMeshManifest(const std::filesystem::path& path) {
  return MeshManifest(path);
}

/** @brief A vertex of a mesh: its three coordinates. */
using MeshVertex = std::array<double, 3>;

/**
 * @brief A triangle of a mesh, a face of a fault say: its three corners as
 * vertex indices, in the order the mesh holds them.
 */
using MeshTriangle = std::array<int, 3>;

/** @brief A tetrahedron of a mesh: its four corners as vertex indices. */
using MeshTetrahedron = std::array<int, 4>;

/** @brief An edge of a mesh: its two corners as vertex indices, the smaller first. */
using MeshEdge = std::array<int, 2>;

/**
 * @brief The side of a fault's face a cell lies on: +1 the hanging wall
 * (plus), -1 the footwall (minus).
 *
 * With v0, v1, v2 the face's corners in the order the mesh holds them and
 * x the cell's corner not on the face, the cell lies on the hanging-wall
 * side where (x - v0) . ((v1 - v0) x (v2 - v0)) > 0 and on the footwall
 * side where it is negative: the format page's rule, which the Python
 * states as `planetmodel.mesh3d.manifest.fault_side`.  It holds for
 * reference and physical coordinates alike, the mapping preserving
 * orientation.
 *
 * @param vertices The mesh's vertices.
 * @param face The face's corners, in the order the mesh holds them (in
 *        MFEM, `GetBdrElementVertices` of its boundary element).
 * @param cell The corners of a tetrahedron holding the face, in any order.
 * @throws planetmodel::Error where an index lies outside `vertices`, where
 *         the cell does not hold the face's three corners, or where its
 *         fourth corner lies in the face's plane.
 */
[[nodiscard]] inline int FaultSide(std::span<const MeshVertex> vertices, const MeshTriangle& face,
                                   const MeshTetrahedron& cell) {
  const auto n = vertices.size();
  auto vertex = [&](int i, std::string_view what) -> const MeshVertex& {
    if (i < 0 || static_cast<std::size_t>(i) >= n) {
      throw Error("FaultSide", what,
                  std::format("indexes vertex {}, outside the {} given", i, n));
    }
    return vertices[static_cast<std::size_t>(i)];
  };
  for (const auto i : face) static_cast<void>(vertex(i, "face"));
  auto fourth = -1;
  auto off = 0;
  for (const auto i : cell) {
    static_cast<void>(vertex(i, "cell"));
    if (!std::ranges::contains(face, i)) {
      fourth = i;
      ++off;
    }
  }
  if (off != 1 || !std::ranges::all_of(face, [&](int i) { return std::ranges::contains(cell, i); })) {
    throw Error("FaultSide", "cell",
                std::format("{{{}, {}, {}, {}}} does not hold the face's corners {{{}, {}, {}}}",
                            cell[0], cell[1], cell[2], cell[3], face[0], face[1], face[2]));
  }
  const auto& v0 = vertex(face[0], "face");
  const auto& v1 = vertex(face[1], "face");
  const auto& v2 = vertex(face[2], "face");
  const auto& x = vertex(fourth, "cell");
  auto less = [&](const MeshVertex& p) {
    return MeshVertex{p[0] - v0[0], p[1] - v0[1], p[2] - v0[2]};
  };
  const auto a = less(v1);
  const auto b = less(v2);
  const auto c = less(x);
  const auto s = c[0] * (a[1] * b[2] - a[2] * b[1]) + c[1] * (a[2] * b[0] - a[0] * b[2]) +
                 c[2] * (a[0] * b[1] - a[1] * b[0]);
  if (s == 0.0) {
    throw Error("FaultSide", "cell",
                std::format("{{{}, {}, {}, {}}} has its fourth corner in the face's plane",
                            cell[0], cell[1], cell[2], cell[3]));
  }
  return s > 0.0 ? 1 : -1;
}

/**
 * @brief Which of a fault face's two cells is the plus (hanging-wall)
 * one: 0 for `first`, 1 for `second`; the other is the minus (footwall)
 * cell.
 *
 * The rule is `FaultSide`'s; the Python states it as
 * `planetmodel.mesh3d.manifest.plus_cell`.
 *
 * @param vertices The mesh's vertices.
 * @param face The face's corners, in the order the mesh holds them.
 * @param first The corners of one cell sharing the face (in MFEM, of an
 *        element `GetFaceElements` gives for the face
 *        `GetBdrElementFaceIndex` gives).
 * @param second The corners of the other.
 * @throws planetmodel::Error where `FaultSide` refuses either cell, or
 *         where the two lie on one side of the face.
 */
[[nodiscard]] inline int PlusCell(std::span<const MeshVertex> vertices, const MeshTriangle& face,
                                  const MeshTetrahedron& first, const MeshTetrahedron& second) {
  const auto a = FaultSide(vertices, face, first);
  if (FaultSide(vertices, face, second) == a) {
    throw Error("PlusCell", "",
                std::format("both cells lie on the {} side of the face {{{}, {}, {}}}",
                            a > 0 ? "hanging-wall" : "footwall", face[0], face[1], face[2]));
  }
  return a > 0 ? 0 : 1;
}

/**
 * @brief The edges of one fault's faces that lie on its curves
 * (`FaultCurves`), each list in increasing order, and the faults it ends
 * on.
 */
struct CurveEdges {
  std::vector<MeshEdge> tip{};        /**< Its buried tip, where its sides are joined. */
  std::vector<MeshEdge> free_edge{};  /**< Its edge on the outer surface, its sides free. */
  std::vector<MeshEdge> junction{};   /**< Where it meets another fault. */
  /**
   * @brief The part of its junction on its own boundary, where it ends on
   * another fault: its slip is pinned there as on its tip.
   */
  std::vector<MeshEdge> junction_end{};
  /** @brief The faults it ends on, by their indices in the list given. */
  std::vector<int> ends_on{};

  /**
   * @brief The edges along which its slip is pinned to zero, in increasing
   * order: its tip, and its junction where it ends.
   */
  [[nodiscard]] std::vector<MeshEdge> Pinned() const {
    auto out = tip;
    out.insert(out.end(), junction_end.begin(), junction_end.end());
    std::ranges::sort(out);
    out.erase(std::ranges::unique(out).begin(), out.end());
    return out;
  }
};

namespace detail {

/// The curves of each fault read from the faces alone (`FaultCurves`), into
/// records `Edges` with the members of `CurveEdges`, and with `face_edge`
/// where they have one: of the edges a fault counts once that it shares
/// with no other fault, those on the outer surface are its free edge, those
/// of the rest on `box`, the faces of a flat mesh's base and sides, its
/// face edge, and the others its tip.
template <class Edges>
std::vector<Edges> CurvesOfFaults(std::span<const std::vector<MeshTriangle>> faults,
                                  std::span<const MeshTriangle> surface,
                                  std::span<const MeshTriangle> box) {
  auto count = [](std::span<const MeshTriangle> faces) {
    auto out = std::map<MeshEdge, int>{};
    for (const auto& t : faces) {
      for (const auto& [i, j] : {std::pair{0, 1}, std::pair{1, 2}, std::pair{2, 0}}) {
        const auto a = t[static_cast<std::size_t>(i)];
        const auto b = t[static_cast<std::size_t>(j)];
        ++out[MeshEdge{std::min(a, b), std::max(a, b)}];
      }
    }
    return out;
  };
  auto edges = std::vector<std::map<MeshEdge, int>>{};
  for (const auto& f : faults) edges.push_back(count(f));
  const auto outer = count(surface);
  const auto sides = count(box);
  auto out = std::vector<Edges>(faults.size());
  for (auto k = std::size_t{0}; k < edges.size(); ++k) {
    auto ends = std::vector<char>(edges.size(), 0);
    for (const auto& [edge, uses] : edges[k]) {
      auto shared = false;
      for (auto j = std::size_t{0}; j < edges.size(); ++j) {
        if (j != k && edges[j].contains(edge)) {
          shared = true;
          if (uses == 1) ends[j] = 1;
        }
      }
      if (shared) {
        out[k].junction.push_back(edge);
        if (uses == 1) out[k].junction_end.push_back(edge);
      } else if (uses == 1) {
        if (outer.contains(edge)) {
          out[k].free_edge.push_back(edge);
        } else if (sides.contains(edge)) {
          if constexpr (requires { out[k].face_edge; }) out[k].face_edge.push_back(edge);
        } else {
          out[k].tip.push_back(edge);
        }
      }
    }
    for (auto j = std::size_t{0}; j < edges.size(); ++j) {
      if (ends[j]) out[k].ends_on.push_back(static_cast<int>(j));
    }
  }
  return out;
}

}  // namespace detail

/**
 * @brief The tip, free-edge and junction edges of each fault, and the part
 * of its junction where it ends, read from the faces alone, for a mesh
 * format that keeps no curves (MFEM's): the edges the MSH file's curve
 * groups hold.
 *
 * An edge is a pair of corners, and a fault's count of it the number of
 * its faces that use it: one on the fault's boundary, two inside it.  The
 * edges of a fault's faces that are also edges of another fault's faces
 * are its junction; those of them it counts once, on its own boundary, are
 * where it ends on that other fault (`junction_end`, the faults
 * `ends_on`), and those it counts twice where the other ends on it or the
 * two cross.  Of the rest, an edge counted once is in its free edge where
 * it is also an edge of the geometry's outer surface, and in its tip
 * otherwise; an edge counted twice lies inside it.  The format page's
 * rule, which the Python states as
 * `planetmodel.mesh3d.manifest.fault_curves`.
 *
 * @param faults Each fault's faces (in MFEM, the boundary elements of its
 *        attribute), all in one numbering of the vertices.
 * @param surface The faces of the geometry's outer surface (the boundary
 *        elements of `MeshManifest::SurfaceAttribute`).
 * @return One `CurveEdges` a fault, in the order given.
 */
[[nodiscard]] inline std::vector<CurveEdges> FaultCurves(
    std::span<const std::vector<MeshTriangle>> faults, std::span<const MeshTriangle> surface) {
  return detail::CurvesOfFaults<CurveEdges>(faults, surface, {});
}

/**
 * @brief The wall of one fault each cell is in: +1 its hanging wall, -1 its
 * footwall, 0 neither.
 *
 * The vertices and edges of the fault's faces that are not on a pinned
 * edge carry its slip.  The cells holding one of them are joined across
 * every face that is not one of the fault's and holds one of them too; so
 * joined, they fall into groups each lying on one side of the fault, the
 * side of the faces of the fault its cells hold (the hanging wall where a
 * face's vertex order points to the cell, `FaultSide`).  An edge carries
 * slip in its own right: an edge of the fault joining two pinned vertices
 * (across a corner of the fault, say) has slip at its interior nodes in a
 * mesh of order two or more, and at its midpoint once the mesh is refined.
 * Every other cell touches the fault, if at all, along its pinned edges
 * alone, where its slip is zero, and is in neither wall.  The format page's
 * rule, which the Python states as `planetmodel.mesh3d.manifest.fault_walls`;
 * the mesher wrote its answers as the element attributes of the fault's
 * `hanging_wall`, so a code need not apply it.
 *
 * @param vertices The mesh's vertices.
 * @param cells The tetrahedra of the body the fault is in: the geometry's
 *        layers, without the shells a mesh spec adds outside them.
 * @param faces The fault's faces, in the vertex order the mesh holds.
 * @param pinned The edges along which its slip is pinned (`CurveEdges::Pinned`).
 * @param name The fault's name, for a refusal.
 * @return One wall a cell, in the order of `cells`.
 * @throws planetmodel::Error, where its fault's name, where an index lies
 *         outside `vertices`, where a face of the fault is not a face of
 *         two cells, where a group of cells holds cells on both sides of
 *         its faces (a face oriented against the others, or a tip or
 *         junction missing from `pinned`), or where a group is next to none
 *         of its faces.
 */
[[nodiscard]] inline std::vector<int> FaultWalls(std::span<const MeshVertex> vertices,
                                                 std::span<const MeshTetrahedron> cells,
                                                 std::span<const MeshTriangle> faces,
                                                 std::span<const MeshEdge> pinned,
                                                 std::string_view name = "the fault") {
  using Face = std::array<int, 3>;
  const auto n = static_cast<long long>(vertices.size());
  auto check = [&](int i) {
    if (i < 0 || i >= n) {
      throw Error("FaultWalls", name, std::format("indexes vertex {}, outside the {} given", i, n));
    }
    return static_cast<std::size_t>(i);
  };
  auto key = [&](int a, int b) {
    return static_cast<long long>(std::min(a, b)) * n + std::max(a, b);
  };
  auto sorted = [](Face f) {
    std::ranges::sort(f);
    return f;
  };
  auto walls = std::vector<int>(cells.size(), 0);
  if (faces.empty()) return walls;

  // what carries slip, and the cells holding any of it
  auto carries = std::vector<char>(vertices.size(), 0);
  auto slip_edges = std::vector<long long>{};
  for (const auto& f : faces) {
    for (const auto i : f) carries[check(i)] = 1;
    slip_edges.insert(slip_edges.end(), {key(f[0], f[1]), key(f[1], f[2]), key(f[0], f[2])});
  }
  auto pinned_keys = std::vector<long long>{};
  for (const auto& e : pinned) {
    carries[check(e[0])] = carries[check(e[1])] = 0;
    pinned_keys.push_back(key(e[0], e[1]));
  }
  std::ranges::sort(slip_edges);
  slip_edges.erase(std::ranges::unique(slip_edges).begin(), slip_edges.end());
  std::ranges::sort(pinned_keys);
  auto difference = std::vector<long long>{};
  std::ranges::set_difference(slip_edges, pinned_keys, std::back_inserter(difference));
  slip_edges = std::move(difference);
  auto is_slip_edge = [&](int a, int b) {
    return std::ranges::binary_search(slip_edges, key(a, b));
  };
  auto touching = std::vector<char>(cells.size(), 0);
  for (auto c = std::size_t{0}; c < cells.size(); ++c) {
    const auto& t = cells[c];
    // every corner checked, whichever of them carry slip
    for (const auto i : t) {
      if (carries[check(i)]) touching[c] = 1;
    }
    for (auto a = 0; a < 4 && !touching[c]; ++a) {
      for (auto b = a + 1; b < 4; ++b) {
        if (is_slip_edge(t[static_cast<std::size_t>(a)], t[static_cast<std::size_t>(b)])) {
          touching[c] = 1;
        }
      }
    }
  }

  // every face of every cell, and the cells of each
  auto sides = std::vector<std::pair<Face, int>>{};
  sides.reserve(4 * cells.size());
  for (auto c = std::size_t{0}; c < cells.size(); ++c) {
    const auto& t = cells[c];
    for (const auto& [a, b, d] : {Face{1, 2, 3}, Face{0, 2, 3}, Face{0, 1, 3}, Face{0, 1, 2}}) {
      sides.emplace_back(sorted({t[static_cast<std::size_t>(a)], t[static_cast<std::size_t>(b)],
                                 t[static_cast<std::size_t>(d)]}),
                         static_cast<int>(c));
    }
  }
  std::ranges::sort(sides);
  auto fault_faces = std::vector<Face>{};
  for (const auto& f : faces) fault_faces.push_back(sorted(f));
  std::ranges::sort(fault_faces);

  // the groups: cells joined across faces not the fault's holding slip
  auto parent = std::vector<int>(cells.size());
  for (auto c = std::size_t{0}; c < cells.size(); ++c) parent[c] = static_cast<int>(c);
  auto root = [&](int c) {
    while (parent[static_cast<std::size_t>(c)] != c) {
      c = parent[static_cast<std::size_t>(c)] =
          parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(c)])];
    }
    return c;
  };
  for (auto i = std::size_t{0}; i < sides.size();) {
    auto j = i;
    while (j < sides.size() && sides[j].first == sides[i].first) ++j;
    if (j - i > 2) {
      throw Error("FaultWalls", name, "a face of the mesh is held by more than two cells");
    }
    const auto& f = sides[i].first;
    if (j - i == 2 && !std::ranges::binary_search(fault_faces, f)) {
      const auto holds = carries[static_cast<std::size_t>(f[0])] ||
                         carries[static_cast<std::size_t>(f[1])] ||
                         carries[static_cast<std::size_t>(f[2])] || is_slip_edge(f[0], f[1]) ||
                         is_slip_edge(f[1], f[2]) || is_slip_edge(f[0], f[2]);
      if (holds) parent[static_cast<std::size_t>(root(sides[i].second))] = root(sides[i + 1].second);
    }
    i = j;
  }

  // each group's side, from the faces of the fault its cells hold
  auto plus = std::vector<char>(cells.size(), 0);
  auto minus = std::vector<char>(cells.size(), 0);
  auto lonely = 0;
  for (const auto& f : faces) {
    const auto at = std::ranges::lower_bound(sides, std::pair{sorted(f), -1});
    if (at == sides.end() || at->first != sorted(f) || std::next(at) == sides.end() ||
        std::next(at)->first != at->first) {
      ++lonely;
      continue;
    }
    for (const auto c : {at->second, std::next(at)->second}) {
      if (!touching[static_cast<std::size_t>(c)]) continue;
      const auto side = FaultSide(vertices, f, cells[static_cast<std::size_t>(c)]);
      (side > 0 ? plus : minus)[static_cast<std::size_t>(root(c))] = 1;
    }
  }
  if (lonely) {
    throw Error("FaultWalls", name,
                std::format("{} of its {} faces are not a face of two cells", lonely,
                            faces.size()));
  }
  auto groups = 0, both = 0, none = 0;
  for (auto c = std::size_t{0}; c < cells.size(); ++c) {
    if (!touching[c] || root(static_cast<int>(c)) != static_cast<int>(c)) continue;
    ++groups;
    both += plus[c] && minus[c];
    none += !plus[c] && !minus[c];
  }
  if (both) {
    throw Error("FaultWalls", name,
                std::format("{} of the {} groups of cells touching its slip-carrying vertices and "
                            "edges hold cells on both sides of its faces, joining its hanging "
                            "wall to its footwall: a face oriented against the others, or a tip "
                            "or junction missing from its pinned edges",
                            both, groups));
  }
  if (none) {
    throw Error("FaultWalls", name,
                std::format("{} of the {} groups of cells touching its slip-carrying vertices and "
                            "edges are next to none of its faces",
                            none, groups));
  }
  for (auto c = std::size_t{0}; c < cells.size(); ++c) {
    if (touching[c]) walls[c] = plus[static_cast<std::size_t>(root(static_cast<int>(c)))] ? 1 : -1;
  }
  return walls;
}

/**
 * @brief Every fault's walls (`FaultWalls`), its pinned edges read from the
 * faces (`FaultCurves`, `CurveEdges::Pinned`); the Python states it as
 * `planetmodel.mesh3d.manifest.walls_of_faults`.
 *
 * @param vertices The mesh's vertices.
 * @param cells The tetrahedra of the body, as `FaultWalls` takes them.
 * @param faults Each fault's faces, as `FaultCurves` takes them.
 * @param surface The faces of the geometry's outer surface.
 * @param names The faults' names, for a refusal; `fault <k>` where none
 *        are given.
 * @return One list of walls a fault, in the order given.
 * @throws planetmodel::Error where `FaultWalls` refuses a fault.
 */
[[nodiscard]] inline std::vector<std::vector<int>> WallsOfFaults(
    std::span<const MeshVertex> vertices, std::span<const MeshTetrahedron> cells,
    std::span<const std::vector<MeshTriangle>> faults, std::span<const MeshTriangle> surface,
    std::span<const std::string> names = {}) {
  const auto curves = FaultCurves(faults, surface);
  auto out = std::vector<std::vector<int>>{};
  for (auto k = std::size_t{0}; k < faults.size(); ++k) {
    const auto pinned = curves[k].Pinned();
    out.push_back(FaultWalls(vertices, cells, faults[k], pinned,
                             k < names.size() ? names[k] : std::format("fault {}", k)));
  }
  return out;
}

}  // namespace planetmodel

#endif  // PLANETMODEL_MANIFEST_HPP
