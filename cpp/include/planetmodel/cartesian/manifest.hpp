#ifndef PLANETMODEL_CARTESIAN_MANIFEST_HPP
#define PLANETMODEL_CARTESIAN_MANIFEST_HPP

/**
 * @file
 * @brief The manifest of a flat mesh: the JSON beside a mesh of a
 * flat-layered geometry, a box or a rectangle (the cross-section of a body
 * invariant along y), of schema `planetmodel.mesh.manifest/7`, as
 * `planetmodel/mesh3d/manifest.py` defines it and the format page describes
 * it.
 *
 * A flat mesh is layered in depth below its free surface: points are
 * (x, y, z) with z up, the free surface z = 0, and the layered coordinate
 * is the depth -z.  `coordinates` names the physical axes of the mesh's
 * own coordinates, (x, y, z) for a box and (x, z) for a rectangle, whose
 * mesh is two-dimensional with z its second coordinate.  The layers run
 * from the surface down, attribute 1 the top one; the interfaces are the
 * tops of the layers, numbered the same way, interface k (from one)
 * between layers k - 1 and k - 2 as 0-based indices, the deeper first and
 * -1 for the outside above the free surface.  The base and the sides are
 * no interfaces: they are the box's faces (`faces[]`), whose boundary
 * attributes follow the interfaces' in the order of `FACES`, each the
 * plane of constant `axis` at `value` with the body on the side opposite
 * `outward`.  A box stands for an unbounded body and its faces carry no
 * physics: they are labelled for a solver's own conditions.
 *
 * A flat mesh with faults is schema 7 as well: its manifest has the
 * records of the faults and of their collars as the spherical schema 6 has
 * them (manifest.hpp), the faults' boundary attributes after the
 * interfaces' and the faces', and one more curve of a fault, `face_edge`,
 * where it leaves the body through its base or a side, an artificial
 * boundary where its sides are free as on the surface and which a consumer
 * may pin or leave free; fault k's curve groups are 4k - 3 to 4k, in the
 * order of `FAULT_CURVES`.  A flat mesh without faults has neither the
 * faults' records nor its layers' `attributes`.
 *
 * `cartesian::MeshManifest(path)` reads a manifest and holds it to the
 * structure the Python's `validate_structure` checks; a spherical manifest
 * (schemas 5 and 6) is refused by its schema, naming its reader.  The
 * rules that turn a fault's faces into what a split-node code needs are
 * the spherical ones over plain arrays (`FaultSide`, `PlusCell`,
 * `FaultWalls`), with the face edge told from the tip by the faces of the
 * box (`cartesian::FaultCurves`, `cartesian::WallsOfFaults`).
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "planetmodel/error.hpp"
#include "planetmodel/json.hpp"
#include "planetmodel/manifest.hpp"
#include "planetmodel/units.hpp"

namespace planetmodel::cartesian {

/**
 * @brief The schema of a flat mesh's manifest, with faults or without;
 * bumped only for an incompatible change.
 */
inline constexpr std::string_view MANIFEST_SCHEMA = MANIFEST_SCHEMA_FLAT;

/**
 * @brief The curves of a fault in a flat mesh, in the order of its
 * dimension-1 groups: its buried tip, its free edge on the free surface,
 * its junction with other faults, and its face edge, where it leaves the
 * box through its base or a side.
 */
inline constexpr auto FAULT_CURVES =
    std::array<std::string_view, 4>{"tip", "free_edge", "junction", "face_edge"};

/** @brief A face of a flat mesh as the format names it: a row of `FACES`. */
struct FaceKind {
  std::string_view name; /**< `base`, `x_min`, `x_max`, `y_min` or `y_max`. */
  std::string_view axis; /**< The axis normal to it: `x`, `y` or `z`. */
  int outward;           /**< The sign of its outward normal along the axis. */
};

/**
 * @brief The faces of a flat mesh, in the order their boundary attributes
 * take after the interfaces': the base, then x_min, x_max, y_min and
 * y_max.  A mesh has those whose axis is among its coordinates: a box all
 * five, a rectangle the base and the x faces.
 */
inline constexpr auto FACES = std::array<FaceKind, 5>{{{"base", "z", -1},
                                                       {"x_min", "x", -1},
                                                       {"x_max", "x", 1},
                                                       {"y_min", "y", -1},
                                                       {"y_max", "y", 1}}};

/** @brief A flat mesh's manifest, read and checked. */
class MeshManifest {
 public:
  /** @brief The values just inside each side of an interface. */
  using Sides = ::planetmodel::MeshManifest::Sides;
  /** @brief One `refinements[]` record. */
  using Refinement = ::planetmodel::MeshManifest::Refinement;
  /** @brief One `fields[]` record: a file beside the mesh and what it holds. */
  using Field = ::planetmodel::MeshManifest::Field;
  /** @brief The `model` record: a model file beside the manifest. */
  using ModelRecord = ::planetmodel::MeshManifest::ModelRecord;
  /** @brief A fault's `sizing`. */
  using FaultSizing = ::planetmodel::MeshManifest::FaultSizing;

  /** @brief One `layers[]` record: a layer of the mesh, from the surface down. */
  struct Layer {
    /** @brief The layer's own attribute, from one at the top, by which records name it. */
    int attribute = 0;
    std::string name;          /**< The layer's name. */
    double depth_top = 0.0;    /**< The depth of its top. */
    double depth_bottom = 0.0; /**< The depth of its bottom. */
    /** @brief True: every layer of a flat mesh is the geometry's. */
    bool in_geometry = true;
    /**
     * @brief Whether shear vanishes throughout the model's layer; empty
     * where no model said.
     */
    std::optional<bool> fluid{};
    /**
     * @brief The element attributes its cells carry: its own first, then
     * its collars' where the mesh has faults; its own alone otherwise.
     */
    std::vector<int> attributes{};
  };

  /** @brief One `interfaces[]` record: the top of a layer, from the surface down. */
  struct Interface {
    int attribute = 0;  /**< The boundary attribute, from one at the free surface. */
    std::string name;   /**< The interface's name. */
    double depth = 0.0; /**< Its depth. */
    /** @brief The layers below and above as 0-based indices, the deeper
     * first, -1 for the outside above the free surface. */
    std::array<int, 2> between_layers{};
    /** @brief One of `INTERFACE_KINDS`; empty where no model said. */
    std::optional<std::string> kind{};
    /**
     * @brief {field: sides} for the exported real scalar fields of depth
     * alone; empty where no model said.
     */
    std::optional<std::map<std::string, Sides>> values{};

    /** @brief The attribute of the layer below, the deeper one. */
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
   * @brief One `faces[]` record: a face of the body that is no interface,
   * its base or a side, labelled for a solver's own conditions.
   */
  struct Face {
    int attribute = 0;  /**< The boundary attribute of its elements, after the interfaces'. */
    std::string name;   /**< `base`, `x_min`, `x_max`, `y_min` or `y_max`. */
    std::string axis;   /**< The axis normal to it: `x`, `y` or `z`. */
    double value = 0.0; /**< The coordinate along `axis` of its plane, z up. */
    int outward = 0;    /**< The sign of its outward normal along `axis`, -1 or 1. */
  };

  /**
   * @brief One `faults[]` record of a flat mesh: the spherical record
   * (`planetmodel::MeshManifest::Fault`) and its face edge.  Its boundary
   * attribute follows the interfaces' and the faces', and its curve groups
   * are 4k - 3 to 4k for fault k, in the order of `FAULT_CURVES`.
   */
  struct Fault : ::planetmodel::MeshManifest::Fault {
    /**
     * @brief The MSH file's curve group of its curves on the faces of the
     * box, its base and sides, where it leaves the body through an
     * artificial boundary and its sides are free; empty where it has none.
     */
    std::optional<int> face_edge{};
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
  /** @brief The schema it declares: `MANIFEST_SCHEMA`. */
  [[nodiscard]] const std::string& Schema() const noexcept { return schema_; }
  /** @brief The mesh file beside the manifest. */
  [[nodiscard]] std::filesystem::path MeshFile() const {
    return path_.parent_path() / mesh_.file;
  }
  /** @brief `msh` from the mesher, `mfem` from the exporter. */
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

  /**
   * @brief The physical axes of the mesh's own coordinates, in order:
   * (x, y, z) for a box, (x, z) for a rectangle.
   */
  [[nodiscard]] const std::vector<std::string>& Coordinates() const noexcept {
    return coordinates_;
  }
  /** @brief The number of coordinates of the mesh: three for a box, two for a rectangle. */
  [[nodiscard]] int Dimension() const noexcept { return static_cast<int>(coordinates_.size()); }
  /** @brief Whether the mesh is a rectangle's, two-dimensional with z its second coordinate. */
  [[nodiscard]] bool IsRectangle() const noexcept { return coordinates_.size() == 2; }

  /** @brief The layers, from the surface down. */
  [[nodiscard]] const std::vector<Layer>& Layers() const noexcept { return layers_; }
  /** @brief The interfaces, the tops of the layers, from the surface down. */
  [[nodiscard]] const std::vector<Interface>& Interfaces() const noexcept {
    return interfaces_;
  }
  /** @brief The faces of the body that are no interfaces: its base, then its sides. */
  [[nodiscard]] const std::vector<Face>& Faces() const noexcept { return faces_; }
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
   * @brief A face by name: `base`, `x_min`, `x_max`, `y_min` or `y_max`.
   * @param name The face's name.
   * @throws planetmodel::Error if the mesh has no such face.
   */
  [[nodiscard]] const Face& FaceNamed(std::string_view name) const {
    return Named(faces_, name, "face");
  }
  /** @brief The boundary attributes of the faces, its base and sides, in order. */
  [[nodiscard]] std::vector<int> FaceAttributes() const {
    auto out = std::vector<int>{};
    for (const auto& f : faces_) out.push_back(f.attribute);
    return out;
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
   * interface's or a face's.
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
                            attribute, detail::JoinedList(FaultAttributes())));
  }

  /** @brief The depths of the ends of the layers, the top first: one more than the layers. */
  [[nodiscard]] std::vector<double> Boundaries() const {
    auto out = std::vector<double>{layers_.front().depth_top};
    for (const auto& l : layers_) out.push_back(l.depth_bottom);
    return out;
  }
  /** @brief The depth of the top of the mesh: zero at the free surface. */
  [[nodiscard]] double TopDepth() const { return layers_.front().depth_top; }
  /** @brief The depth of the base of the mesh. */
  [[nodiscard]] double BaseDepth() const { return layers_.back().depth_bottom; }
  /**
   * @brief The length the mesh's tolerances are relative to: its largest
   * coordinate in magnitude, the base's depth or a side's place.
   */
  [[nodiscard]] double Scale() const {
    auto out = std::abs(BaseDepth());
    for (const auto& f : faces_) out = std::max(out, std::abs(f.value));
    return out;
  }

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
                            attribute, detail::JoinedList(ElementAttributes())));
  }
  /** @brief The element attributes every layer's cells carry, layer by layer. */
  [[nodiscard]] std::vector<int> ElementAttributes() const {
    return Select([](const Layer&) { return true; });
  }
  /**
   * @brief The element attributes the cells of some layers carry, each
   * layer's own and then its collars', in the order the layers are given.
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
   * @brief The element attributes of the body's layers, their collars'
   * among them: every layer's, a flat mesh having no shells.
   */
  [[nodiscard]] std::vector<int> BodyAttributes() const { return ElementAttributes(); }
  /**
   * @brief The element attributes of the solid layers, their collars'
   * among them.
   * @throws planetmodel::Error where a layer's fluidity is unknown (no
   *         model said).
   */
  [[nodiscard]] std::vector<int> SolidAttributes() const {
    RequireFluidity();
    return Select([](const Layer& l) { return !*l.fluid; });
  }
  /**
   * @brief The element attributes of the fluid layers, their collars'
   * among them.
   * @throws planetmodel::Error where a layer's fluidity is unknown.
   */
  [[nodiscard]] std::vector<int> FluidAttributes() const {
    RequireFluidity();
    return Select([](const Layer& l) { return *l.fluid; });
  }
  /**
   * @brief The interfaces between a fluid and a solid layer.
   * @throws planetmodel::Error where a layer's fluidity is unknown.
   */
  [[nodiscard]] std::vector<int> FluidSolidInterfaces() const {
    RequireFluidity();
    auto out = std::vector<int>{};
    for (const auto& f : interfaces_) {
      const auto* below = LayerWith(f.BelowAttribute());
      const auto* above = LayerWith(f.AboveAttribute());
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
      const auto* layer = LayerWith(attribute);
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
   * @brief The attribute of the top of the mesh, its free surface, where a
   * fault's free edge lies: the interface with the outside above it.
   */
  [[nodiscard]] int SurfaceAttribute() const { return interfaces_.front().attribute; }

  /**
   * @brief MFEM's marker array for a list of attributes: one where the
   * attribute is listed, zero elsewhere.
   * @param attributes The attributes, from one.
   * @param size The number of entries.
   * @throws planetmodel::Error if an attribute is below one.
   */
  [[nodiscard]] static std::vector<int> Marker(const std::vector<int>& attributes, int size) {
    return ::planetmodel::MeshManifest::Marker(attributes, size);
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
    throw Error(path_.string(), "constants", std::format("holds no constant {}", name));
  }
  /**
   * @brief The gravitational constant in the model's units, where the
   * model declares it.
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
  void Read() {
    const auto& doc = document_;
    const auto source = path_.string();
    if (!doc.IsObject()) Fail("", "is not an object");
    const auto& declared = doc.At("schema").AsString();
    if (declared == ::planetmodel::MANIFEST_SCHEMA || declared == MANIFEST_SCHEMA_WITH_FAULTS) {
      Fail("schema", std::format("is {}, a spherical mesh's, and this reader reads {}: read it "
                                 "with planetmodel::MeshManifest (planetmodel/manifest.hpp)",
                                 declared, MANIFEST_SCHEMA));
    }
    if (declared != MANIFEST_SCHEMA) {
      Fail("schema", declared.starts_with("planetmodel.mesh.manifest/")
                         ? std::format("is {}, and this reader reads {}: regenerate the "
                                       "manifest with this planetmodel",
                                       declared, MANIFEST_SCHEMA)
                         : std::format("is {}, not a mesh manifest's", declared));
    }
    schema_ = declared;
    const auto faulted = !detail::ManifestList(doc, "faults").empty();

    mesh_ = detail::ReadManifestMesh(doc.At("mesh"));
    const auto& coordinates = doc.At("coordinates");
    for (const auto& axis : coordinates.AsArray()) coordinates_.push_back(axis.AsString());

    for (const auto& e : detail::ManifestList(doc, "layers")) {
      const auto& fluid = e.At("fluid");
      auto layer = Layer{.attribute = e.At("attribute").AsInt(),
                         .name = e.At("name").AsString(),
                         .depth_top = e.At("depth_top").AsNumber(),
                         .depth_bottom = e.At("depth_bottom").AsNumber(),
                         .in_geometry = e.At("in_geometry").AsBool(),
                         .fluid = fluid.IsNull() ? std::nullopt
                                                 : std::optional<bool>{fluid.AsBool()}};
      if (!faulted && e.Find("attributes") != nullptr) {
        Fail(std::format("layers[{}].attributes", layers_.size()),
             "is listed, and faults lists none: a flat manifest lists its layers' attributes "
             "with its faults or neither");
      }
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
                            .depth = e.At("depth").AsNumber()};
      detail::ReadManifestInterface<Interface, Sides>(e, face, source);
      interfaces_.push_back(std::move(face));
    }
    for (const auto& e : doc.At("faces").AsArray()) {
      faces_.push_back(Face{.attribute = e.At("attribute").AsInt(),
                            .name = e.At("name").AsString(),
                            .axis = e.At("axis").AsString(),
                            .value = e.At("value").AsNumber(),
                            .outward = e.At("outward").AsInt()});
    }
    refinements_ = detail::ReadManifestRefinements<Refinement>(doc);
    for (const auto& e : detail::ManifestList(doc, "faults")) {
      auto f = Fault{};
      detail::ReadManifestFault<Fault, FaultSizing>(e, f, source);
      f.face_edge = detail::OptionalInt(e.At("face_edge"));
      faults_.push_back(std::move(f));
    }
    fields_ = detail::ReadManifestFields<Field>(doc);
    detail::ReadManifestModelBlocks(doc, source, scales_, constants_, model_, meta_);
  }

  /// As the Python's `validate_structure` checks a manifest of schema 7.
  void Check() const {
    const auto source = path_.string();
    detail::CheckManifestMesh(mesh_, source);
    const auto n = static_cast<int>(layers_.size());
    if (n == 0) Fail("layers", "lists no layers");
    const auto box = coordinates_ == std::vector<std::string>{"x", "y", "z"};
    if (!box && coordinates_ != std::vector<std::string>{"x", "z"}) {
      Fail("coordinates", std::format("is [{}], not [x, y, z] or [x, z]",
                                      detail::JoinedList(coordinates_)));
    }
    for (auto i = 0; i < n; ++i) {
      const auto& l = layers_[static_cast<std::size_t>(i)];
      const auto where = std::format("layers[{}]", i);
      if (l.attribute != i + 1) {
        Fail(where + ".attribute", std::format("is {}, expected {}", l.attribute, i + 1));
      }
      if (!(0.0 <= l.depth_top && l.depth_top < l.depth_bottom)) {
        Fail(where, std::format("runs from depth {} to {}, not 0 <= depth_top < depth_bottom",
                                l.depth_top, l.depth_bottom));
      }
      if (i > 0 && l.depth_top != layers_[static_cast<std::size_t>(i - 1)].depth_bottom) {
        Fail(where, std::format("starts at depth {} but the layer above ends at {}", l.depth_top,
                                layers_[static_cast<std::size_t>(i - 1)].depth_bottom));
      }
      if (!l.in_geometry) {
        Fail(where + ".in_geometry", "is false: every layer of a flat mesh is the geometry's");
      }
    }
    const auto m = static_cast<int>(interfaces_.size());
    if (m != n) {
      Fail("interfaces", std::format("lists {} for {} layers; a flat mesh has one per layer, "
                                     "its top",
                                     m, n));
    }
    for (auto i = 0; i < m; ++i) {
      const auto& f = interfaces_[static_cast<std::size_t>(i)];
      const auto where = std::format("interfaces[{}]", i);
      if (f.attribute != i + 1) {
        Fail(where + ".attribute", std::format("is {}, expected {}", f.attribute, i + 1));
      }
      if (f.depth != layers_[static_cast<std::size_t>(i)].depth_top) {
        Fail(where + ".depth",
             std::format("is {}, not the top of layers[{}], {}", f.depth, i,
                         layers_[static_cast<std::size_t>(i)].depth_top));
      }
      const auto want = std::array<int, 2>{i, i - 1};
      if (f.between_layers != want) {
        Fail(where + ".between_layers",
             std::format("is [{}, {}], expected [{}, {}]", f.between_layers[0],
                         f.between_layers[1], want[0], want[1]));
      }
      detail::CheckManifestKind(f, i, source);
    }

    // the faces of FACES whose axes the mesh has, in that order
    auto rows = std::vector<FaceKind>{};
    for (const auto& row : FACES) {
      if (std::ranges::contains(coordinates_, std::string{row.axis})) rows.push_back(row);
    }
    auto got = std::vector<std::string>{};
    auto want = std::vector<std::string>{};
    for (const auto& f : faces_) got.push_back(f.name);
    for (const auto& row : rows) want.emplace_back(row.name);
    if (got != want) {
      Fail("faces", std::format("are [{}], not [{}] for coordinates [{}]",
                                detail::JoinedList(got), detail::JoinedList(want),
                                detail::JoinedList(coordinates_)));
    }
    for (auto k = std::size_t{0}; k < faces_.size(); ++k) {
      const auto& f = faces_[k];
      const auto where = std::format("faces[{}]", k);
      if (f.attribute != m + static_cast<int>(k) + 1) {
        Fail(where + ".attribute",
             std::format("is {}, expected {}, after the {} interfaces", f.attribute,
                         m + static_cast<int>(k) + 1, m));
      }
      if (f.axis != rows[k].axis || f.outward != rows[k].outward) {
        Fail(where, std::format("({}) is normal to {} with outward {}, not {} with {}", f.name,
                                f.axis, f.outward, rows[k].axis, rows[k].outward));
      }
    }
    if (faces_.front().value != -layers_.back().depth_bottom) {
      Fail("faces[0].value", std::format("is z = {}, not the last layer's bottom, z = {}",
                                         faces_.front().value, -layers_.back().depth_bottom));
    }
    for (auto k = std::size_t{1}; k + 1 < faces_.size(); k += 2) {
      if (!(faces_[k].value < faces_[k + 1].value)) {
        Fail(std::format("faces[{}].value", k),
             std::format("{} and {} lie at {} and {}, not the first below the second",
                         faces_[k].name, faces_[k + 1].name, faces_[k].value,
                         faces_[k + 1].value));
      }
    }
    for (auto k = std::size_t{0}; k < faces_.size(); ++k) {
      if (std::ranges::count(interfaces_, faces_[k].name, &Interface::name) > 0) {
        Fail(std::format("faces[{}].name", k),
             std::format("{} shares its name with an interface", faces_[k].name));
      }
    }

    detail::CheckManifestRefinements(refinements_, source);
    const auto before = m + static_cast<int>(faces_.size());
    detail::CheckManifestFaults(
        layers_, interfaces_, faults_, before,
        std::format("the {} interfaces and the {} faces", m, faces_.size()), got,
        "a face of the box",
        [](const Fault& f) {
          return std::array{std::pair{"tip", f.tip}, std::pair{"free_edge", f.free_edge},
                            std::pair{"junction", f.junction},
                            std::pair{"face_edge", f.face_edge}};
        },
        source);
    detail::CheckManifestFields(fields_, interfaces_, n, mesh_, source);
  }

  [[noreturn]] void Fail(std::string_view where, std::string_view message) const {
    throw Error(path_.string(), where, message);
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

  /// The layer of own attribute `attribute`, or null for the outside.
  const Layer* LayerWith(int attribute) const {
    if (attribute < 1 || attribute > static_cast<int>(layers_.size())) return nullptr;
    return &layers_[static_cast<std::size_t>(attribute - 1)];
  }

  void RequireFluidity() const {
    for (const auto& l : layers_) {
      if (!l.fluid) {
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
  std::vector<std::string> coordinates_;
  std::vector<Layer> layers_;
  std::vector<Interface> interfaces_;
  std::vector<Face> faces_;
  std::vector<Refinement> refinements_;
  std::vector<Field> fields_;
  std::vector<Fault> faults_;
  std::optional<::planetmodel::Scales> scales_;
  std::map<std::string, double> constants_;
  std::optional<ModelRecord> model_;
};

/**
 * @brief The edges of one fault's faces that lie on its curves in a flat
 * mesh (`cartesian::FaultCurves`): those of `planetmodel::CurveEdges`, and
 * its face edge.
 */
struct CurveEdges : ::planetmodel::CurveEdges {
  /**
   * @brief Its edge on the faces of the box, its base and sides, where it
   * leaves the body through an artificial boundary: its sides free, as on
   * the surface, and not pinned (`Pinned` leaves it out).
   */
  std::vector<MeshEdge> face_edge{};
};

/**
 * @brief The tip, free-edge, face-edge and junction edges of each fault of
 * a flat mesh, and the part of its junction where it ends, read from the
 * faces alone, for a mesh format that keeps no curves: the edges the MSH
 * file's curve groups hold.
 *
 * The rule is `planetmodel::FaultCurves`', with one more curve: of the
 * edges a fault counts once and shares with no other fault, those that
 * are edges of the free surface are its free edge, those of the rest that
 * are edges of the box's faces its face edge, and the others its tip.  The
 * Python states it as `planetmodel.mesh3d.manifest.fault_curves` with
 * `faces`.
 *
 * @param faults Each fault's faces (in MFEM, the boundary elements of its
 *        attribute), all in one numbering of the vertices.
 * @param surface The faces of the free surface (the boundary elements of
 *        `MeshManifest::SurfaceAttribute`).
 * @param faces The faces of the box's base and sides (the boundary
 *        elements of `MeshManifest::FaceAttributes`).
 * @return One `CurveEdges` a fault, in the order given.
 */
[[nodiscard]] inline std::vector<CurveEdges> FaultCurves(
    std::span<const std::vector<MeshTriangle>> faults, std::span<const MeshTriangle> surface,
    std::span<const MeshTriangle> faces) {
  return detail::CurvesOfFaults<CurveEdges>(faults, surface, faces);
}

/**
 * @brief Every fault's walls in a flat mesh (`planetmodel::FaultWalls`),
 * its pinned edges read from the faces (`cartesian::FaultCurves`,
 * `CurveEdges::Pinned`): its tip and its junction where it ends, never its
 * free or face edge.
 *
 * @param vertices The mesh's vertices.
 * @param cells The tetrahedra of the body, as `FaultWalls` takes them.
 * @param faults Each fault's faces, as `FaultCurves` takes them.
 * @param surface The faces of the free surface.
 * @param faces The faces of the box's base and sides.
 * @param names The faults' names, for a refusal; `fault <k>` where none
 *        are given.
 * @return One list of walls a fault, in the order given.
 * @throws planetmodel::Error where `FaultWalls` refuses a fault.
 */
[[nodiscard]] inline std::vector<std::vector<int>> WallsOfFaults(
    std::span<const MeshVertex> vertices, std::span<const MeshTetrahedron> cells,
    std::span<const std::vector<MeshTriangle>> faults, std::span<const MeshTriangle> surface,
    std::span<const MeshTriangle> faces, std::span<const std::string> names = {}) {
  const auto curves = FaultCurves(faults, surface, faces);
  auto out = std::vector<std::vector<int>>{};
  for (auto k = std::size_t{0}; k < faults.size(); ++k) {
    const auto pinned = curves[k].Pinned();
    out.push_back(FaultWalls(vertices, cells, faults[k], pinned,
                             k < names.size() ? names[k] : std::format("fault {}", k)));
  }
  return out;
}

}  // namespace planetmodel::cartesian

#endif  // PLANETMODEL_CARTESIAN_MANIFEST_HPP
