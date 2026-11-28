#ifndef PLANETMODEL_CARTESIAN_MODELFILE_HPP
#define PLANETMODEL_CARTESIAN_MODELFILE_HPP

/**
 * @file
 * @brief The flat model file, as `planetmodel.cartesian.modelfile` defines
 * it and the format page describes it: a model of `planetmodel.cartesian`,
 * layered in depth over a box or a rectangle, in one netCDF-4 file of schema
 * `planetmodel.model/3`, read and written.
 *
 * Everything the model file holds on the sphere (modelfile.hpp) it holds
 * here, but for what the geometry makes different.  The root states the
 * system (`coordinate_system` `flat`, `local_frame` `up x y`) and the
 * lateral domain (`domain`, `domain_bounds`, `domain_padding`); the
 * skeleton is in depth from the free surface down, layer 0 the top one
 * and interface k the top of layer k, the base a face of the box and no
 * interface; a part holds `depth(node)` and the horizontal grid `x(x)` and,
 * over a box, `y(y)`; a field is `name(node)` of depth alone or
 * `name(node, x, y)` (`name(node, x)` over a rectangle), its components in
 * the local frame (e_up, e_x, e_y), the same at every point; the lateral
 * rules are the doubly periodic `linear` and `cubic` (cartesian/rules.hpp),
 * `fourier` the Python's alone and refused by name where a field stating
 * it is built; and the mapping is the vertical stretch, held as the
 * heights of surfaces (the `geometry` group) or as its upward displacement
 * at the nodes (`displacement` in every part), heights positive up.
 *
 * `ReadModelFile(path)` reads a file whole and holds it to the structure
 * the Python's `cartesian.modelfile.read_model_file` checks, its schema
 * first: a spherical file is refused by its schema, naming its reader.
 * `ModelFile::FieldOn(layer, name)` builds a field of one variable on one
 * layer, evaluated at (depth, x, y) or at a point (x, y, z), z up, with
 * components in the local or the Cartesian frame;
 * `ModelFile::Mapping()` gives the vertical stretch;
 * `ModelFile::SurfaceFieldOn(interface, name)` a field of the `surfaces`
 * group as a function of (x, y); `ModelFile::Behaviours()` the names of
 * the model's behaviours, which a file written from a `ModelFile` that
 * was read states again unchanged.  A code that holds fields of its own
 * makes a `ModelFile` of its skeleton of depths, its domain and scales,
 * cuts the layers into parts from the surface down, adds its fields from
 * `LayerSource`s, and `WriteModelFile` writes the file after holding it to
 * the reader's checks:
 *
 * @code
 * namespace pm = planetmodel;
 * namespace flat = planetmodel::cartesian;
 * const auto file = flat::ReadModelFile("box.nc");
 * const auto vs = file.FieldOn(1, "vs").AsScalar();
 * const auto value = vs.Value(depth, x, y);
 * const auto u = file.Mapping().Displacement({x, y, z});
 *
 * const auto domain = flat::Domain::Box({0.0, 400e3}, {0.0, 300e3});
 * auto built = flat::ModelFile(pm::Skeleton({0.0, 30e3, 200e3}), domain);
 * built.AddPart(0, 1, pm::ElementEdges(built.GetSkeleton(), 0, 1, 20e3), 5,
 *               flat::Grid(domain, 64, 48));
 * built.AddField("viscosity", {{0, flat::LayerSource::Depth(crust)},
 *                              {1, flat::LayerSource::Lateral(mantle)}});
 * flat::WriteModelFile("viscosity.nc", built);
 * @endcode
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "planetmodel/cartesian/coordinates.hpp"
#include "planetmodel/cartesian/domain.hpp"
#include "planetmodel/cartesian/grid.hpp"
#include "planetmodel/cartesian/rules.hpp"
#include "planetmodel/character.hpp"
#include "planetmodel/error.hpp"
#include "planetmodel/gll.hpp"
#include "planetmodel/modelfile.hpp"
#include "planetmodel/netcdf.hpp"
#include "planetmodel/rules.hpp"
#include "planetmodel/skeleton.hpp"
#include "planetmodel/units.hpp"
#include "planetmodel/version.hpp"
#include "planetmodel/vocabulary.hpp"

namespace planetmodel::cartesian {

/**
 * @brief The schema of a flat model file, whether a variable states a
 * radial rule in the logarithm or none does; bumped only for an
 * incompatible change.
 */
inline constexpr std::string_view MODEL_SCHEMA = MODEL_SCHEMA_FLAT;

/** @brief What a flat file states as its `coordinate_system`. */
inline constexpr std::string_view COORDINATE_SYSTEM = "flat";

/** @brief What a flat model file states as its `local_frame`: (e_up, e_x, e_y). */
inline constexpr std::string_view LOCAL_FRAME_AXES = "up x y";

namespace detail {

using namespace ::planetmodel::detail;

/// The statements of a spherical file a flat one never makes.
inline constexpr auto SPHERICAL_DIMENSIONS =
    std::array<std::string_view, 3>{"latitude", "longitude", "pole"};
inline constexpr auto SPHERICAL_ATTRIBUTES =
    std::array<std::string_view, 4>{"grid_kind", "grid_lmax", "poles", "radius"};
inline constexpr auto SPHERICAL_VARIABLES = std::array<std::string_view, 6>{
    "latitude", "longitude", "colatitude_weights", "radius", "interface_radius",
    "boundary_radius"};

/// The names of `own` and then those of a spherical file's variables,
/// `SPHERICAL_VARIABLES`, which a reader refuses in a flat file.
template <std::size_t N>
constexpr auto WithSphericalVariables(const std::array<std::string_view, N>& own) {
  auto out = std::array<std::string_view, N + SPHERICAL_VARIABLES.size()>{};
  std::ranges::copy(SPHERICAL_VARIABLES, std::ranges::copy(own, out.begin()).out);
  return out;
}

}  // namespace detail

/**
 * @brief Names no field of a flat model file may take: those the file gives
 * its own variables, and those of a spherical file's variables (`radius`,
 * `latitude`, `longitude`, `colatitude_weights`, `interface_radius`,
 * `boundary_radius`), which a reader refuses in a flat file.
 */
inline constexpr auto RESERVED_NAMES =
    detail::WithSphericalVariables(std::array<std::string_view, 12>{
        "depth",           "element_layer",  "element_bounds", "x",
        "y",               "displacement",   "layer_bounds",   "layer_name",
        "interface_depth", "interface_name", "layer_part",     "surface_interface"});

namespace detail {

/// `where` and a name under it, or the name alone at the root.
inline std::string Under(std::string_view where, std::string_view name) {
  return where.empty() ? std::string{name} : std::format("{}/{}", where, name);
}

/// Refuse, by name, a statement of a spherical file in a flat file's group,
/// and any polar values.
inline void RefuseSpherical(int group, const netcdf::Attributes& attrs,
                            const std::string& source, std::string_view where) {
  auto refuse = [&](std::string_view name) {
    Fail(source, Under(where, name),
         std::format("is a spherical file's, and a flat file states no {}", name));
  };
  const auto dims = netcdf::GroupDimensionNames(group, source);
  for (const auto name : SPHERICAL_DIMENSIONS) {
    if (std::ranges::contains(dims, std::string{name})) refuse(name);
  }
  for (const auto name : SPHERICAL_ATTRIBUTES) {
    if (attrs.contains(std::string{name})) refuse(name);
  }
  for (const auto name : SPHERICAL_VARIABLES) {
    if (netcdf::HasVariable(group, name)) refuse(name);
  }
  for (const auto& name : netcdf::VariableNames(group, source)) {
    if (name.ends_with("_poles")) {
      Fail(source, Under(where, name), "holds polar values, which a flat grid has none of");
    }
  }
}

/// The lateral domain a group's attributes state, as the Python's
/// `read_domain` reads it.
inline Domain ReadDomain(const netcdf::Attributes& attrs, const std::string& source) {
  for (const auto* key : {"domain", "domain_bounds", "domain_padding"}) {
    if (!attrs.contains(key)) {
      Fail(source, key, "is missing: a flat file states its lateral domain");
    }
  }
  const auto kind = TextOf(attrs, "domain");
  if (!kind || !std::ranges::contains(DOMAIN_KINDS, std::string_view{*kind})) {
    Fail(source, "domain", std::format("is {}, not box or rectangle", kind.value_or("a number")));
  }
  const auto bounds = NumbersOf(attrs, "domain_bounds");
  const auto padding = NumbersOf(attrs, "domain_padding");
  if (!bounds || !padding) {
    Fail(source, "domain_bounds", "and domain_padding are numbers");
  }
  const auto box = *kind == DOMAIN_KINDS[0];
  if (bounds->size() != (box ? 4u : 2u) || padding->size() != (box ? 2u : 1u)) {
    Fail(source, "domain_bounds",
         std::format("a {} states {} bounds and {} padding, got {} and {}", *kind, box ? 4 : 2,
                     box ? 2 : 1, bounds->size(), padding->size()));
  }
  try {
    const auto& b = *bounds;
    const auto& p = *padding;
    return box ? Domain::Box({b[0], b[1]}, {b[2], b[3]}, {p[0], p[1]})
               : Domain::Rectangle({b[0], b[1]}, p[0]);
  } catch (const Error& e) {
    Fail(source, "domain", std::format("is malformed: {}", e.what()));
  }
}

/// The statements of the lateral domain, written.
inline void WriteDomain(int group, const Domain& domain, const std::string& source) {
  netcdf::WriteAttribute(group, netcdf::GLOBAL, "domain", std::string{domain.Kind()}, source);
  netcdf::WriteAttribute(group, netcdf::GLOBAL, "domain_bounds", domain.BoundsList(), source);
  netcdf::WriteAttribute(group, netcdf::GLOBAL, "domain_padding", domain.PaddingList(), source);
}

/// The names of a domain's lateral axes: (x, y) of a box, (x) of a rectangle.
inline std::vector<std::string> AxesOf(const Domain& domain) {
  return domain.IsBox() ? std::vector<std::string>{"x", "y"} : std::vector<std::string>{"x"};
}

/// The grid a group's dimensions hold over `domain`, its coordinate
/// variables checked to be the grid's nodes (to 1e-12 of the period), as
/// the Python's `check_grid_variables` does; a rectangle's group with a `y`
/// is refused.
inline Grid GridOf(int group, const Domain& domain, const std::string& source,
                   std::string_view where) {
  const auto dims = netcdf::GroupDimensionNames(group, source);
  if (!domain.IsBox() &&
      (std::ranges::contains(dims, std::string{"y"}) || netcdf::HasVariable(group, "y"))) {
    Fail(source, Under(where, "y"),
         "a rectangle's grid has the one axis x, and the group states y");
  }
  auto shape = std::vector<std::size_t>{};
  for (const auto& axis : AxesOf(domain)) {
    if (!std::ranges::contains(dims, axis) || !netcdf::HasVariable(group, axis)) {
      Fail(source, Under(where, axis), "is no coordinate of the group");
    }
    if (netcdf::DimensionNames(group, axis, source) != std::vector<std::string>{axis}) {
      Fail(source, Under(where, axis), "does not run over its own dimension");
    }
    shape.push_back(*netcdf::DimensionLength(group, axis, source));
  }
  auto grid = [&]() {
    try {
      return Grid(domain, shape);
    } catch (const Error& e) {
      Fail(source, where, std::format("the grid is malformed: {}", e.what()));
    }
  }();
  for (auto axis = std::size_t{0}; axis < shape.size(); ++axis) {
    const auto name = AxesOf(domain)[axis];
    const auto stored = netcdf::ReadDoubles(group, name, source).values;
    const auto& nodes = axis == 0 ? grid.X() : grid.Y();
    const auto tolerance = 1e-12 * domain.Length(axis);
    for (auto i = std::size_t{0}; i < nodes.size(); ++i) {
      if (!(std::abs(stored[i] - nodes[i]) <= tolerance)) {
        Fail(source, Under(where, name),
             std::format("is not the nodes of a grid of {} over the domain's extent",
                         nodes.size()));
      }
    }
  }
  return grid;
}

/// A grid's dimensions and CF coordinate variables, `x(x)` and over a box
/// `y(y)`, in the unit `length`.
inline void WriteGrid(int group, const Grid& grid, const std::string& length,
                      const std::string& source) {
  const auto axes = AxesOf(grid.GetDomain());
  for (auto axis = std::size_t{0}; axis < axes.size(); ++axis) {
    const auto& name = axes[axis];
    netcdf::Dimension(group, name, grid.Shape()[axis], source);
    const auto var =
        netcdf::WriteDoubles(group, name, {name}, axis == 0 ? grid.X() : grid.Y(), source);
    netcdf::WriteAttribute(group, var, "units", length, source);
    netcdf::WriteAttribute(group, var, "long_name", name, source);
    netcdf::WriteAttribute(group, var, "axis", std::string{axis == 0 ? "X" : "Y"}, source);
  }
}

/// The scales, written as the `scale_<base>` attributes.
inline void WriteScales(int root, const Scales& scales, const std::string& source) {
  const auto put = [&](std::string_view key, double value) {
    netcdf::WriteAttribute(root, netcdf::GLOBAL, key, std::vector<double>{value}, source);
  };
  put("scale_length", scales.Length());
  put("scale_mass", scales.Mass());
  put("scale_time", scales.Time());
  put("scale_temperature", scales.Temperature());
}

/// The depths of a skeleton's interfaces, from the surface down: every
/// boundary but the base.
inline std::vector<double> InterfaceDepths(const Skeleton& skeleton) {
  const auto& b = skeleton.Boundaries();
  return std::vector<double>(b.begin(), b.end() - 1);
}

/// Refuse a variable stating a lateral rule the grid of its group cannot
/// carry, as the Python's `cartesian.lateral.check_rule` does: `cubic` on a
/// grid too small for it (`CheckRule`).
inline void CheckCarried(const Grid& grid, std::string_view rule, const std::string& source,
                         std::string_view where) {
  if (rule != "cubic") return;
  if (const auto why = CubicRefusal(grid)) {
    Fail(source, where, std::format("states the lateral rule cubic, and {}", *why));
  }
}

}  // namespace detail

/** @brief One part of a flat model file: a run of layers on one grid. */
struct FilePart {
  int index = 0;                        /**< `part_<index>`. */
  std::vector<int> layers{};            /**< The layers it holds, contiguous. */
  std::size_t ngll = 0;                 /**< The nodes per element. */
  std::vector<int> element_layer{};     /**< The layer of every element. */
  std::vector<double> element_bounds{}; /**< Each element's top and bottom, two per element. */
  std::vector<double> depth{};          /**< Every node's depth, element by element. */
  std::optional<Grid> grid{};           /**< The horizontal grid. */
  /** @brief The variables by name: `name(node)` of depth alone, or
   * `name(node, x, y)` (`name(node, x)` over a rectangle) where it varies
   * laterally (`FileVariable::IsLateral`), then its components' axes. */
  std::map<std::string, FileVariable> variables{};

  /**
   * @brief The elements of a layer of the part, as [begin, end).
   * @param layer The layer.
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> Elements(int layer) const {
    const auto first = std::ranges::find(element_layer, layer) - element_layer.begin();
    const auto last = std::ranges::find_last(element_layer, layer).begin() -
                      element_layer.begin();
    return {static_cast<std::size_t>(first), static_cast<std::size_t>(last) + 1};
  }
};

/**
 * @brief The `geometry` group of a flat file: the vertical stretch as the
 * heights of surfaces, linear in depth between them.
 */
struct FileGeometry {
  std::vector<double> depths{}; /**< The surfaces' reference depths, increasing. */
  std::optional<Grid> grid{};   /**< The grid the heights are on. */
  std::vector<double> values{}; /**< (boundary, nx, ny), C order; heights, positive up. */
  std::string lateral{};        /**< The lateral rule stated. */
  std::string between{};        /**< The rule between surfaces: `linear`. */
};

/**
 * @brief The `surfaces` group of a flat file: fields on some of the
 * skeleton's interfaces, each a number at every node of the group's grid.
 */
struct FileSurfaces {
  std::vector<int> interfaces{}; /**< The interfaces holding any, increasing. */
  std::optional<Grid> grid{};    /**< The grid the fields are on. */
  /** @brief The variables by name: (surface, nx, ny); NaN where an
   * interface lacks it. */
  std::map<std::string, FileVariable> variables{};

  /**
   * @brief The row of an interface, or empty where it holds no field.
   * @param interface The interface, from the surface down.
   */
  [[nodiscard]] std::optional<std::size_t> Row(int interface) const {
    const auto it = std::ranges::find(interfaces, interface);
    if (it == interfaces.end()) return std::nullopt;
    return static_cast<std::size_t>(it - interfaces.begin());
  }
};

/**
 * @brief The points a layer's values are given at.
 *
 * A source of depth alone is given at each of `depths`, the layer's nodes
 * element by element, an element boundary twice.  A source that varies
 * laterally is given at (depth, x_i, y_j) for each depth and each node of
 * the grid, x before y and y fastest; over a rectangle at (depth, x_i, 0).
 */
struct LayerPoints {
  std::vector<double> depths; /**< The depths, from the top of the layer down. */
  Grid grid;                  /**< The horizontal grid. */

  /** @brief The number of points a source that varies laterally is given at. */
  [[nodiscard]] std::size_t Size() const { return depths.size() * grid.Size(); }

  /**
   * @brief A point of a source that varies laterally, by its place in the
   * order.
   * @param index The place, below `Size()`.
   */
  [[nodiscard]] Point At(std::size_t index) const {
    const auto cells = grid.Size();
    const auto [x, y] = grid.Node(index % cells);
    return {.depth = depths[index / cells], .x = x, .y = y};
  }
};

/**
 * @brief What a layer holds of a variable: a function of depth, or of the
 * point, or the values at the points the layer's `LayerPoints` give.
 *
 * Every function and every value gives the stored components, `Size()` of
 * them in the variable's layout (a Voigt vector or matrix for ranks 2 and
 * 4 in Voigt form, without engineering factors).  A source of depth alone
 * gives its components in the local frame (e_up, e_x, e_y), which is the
 * same at every point; a source that varies laterally gives them in the
 * frame it names.  Every value must be finite.
 */
class LayerSource {
 public:
  /** @brief A function of depth alone, writing the components. */
  using OfDepth = std::function<void(double depth, std::span<double> out)>;
  /** @brief A function of the point, writing the components. */
  using OfPoint = std::function<void(double depth, double x, double y, std::span<double> out)>;

  /**
   * @brief A field of depth alone.
   * @param f Its local components at a depth.
   */
  [[nodiscard]] static LayerSource Depth(OfDepth f) {
    auto out = LayerSource{};
    out.of_depth_ = std::move(f);
    return out;
  }

  /**
   * @brief A scalar of depth alone.
   * @param f Its value at a depth.
   */
  [[nodiscard]] static LayerSource Depth(std::function<double(double depth)> f) {
    auto out = LayerSource{};
    out.scalar_ = true;
    out.of_depth_ = [f = std::move(f)](double depth, std::span<double> v) { v[0] = f(depth); };
    return out;
  }

  /**
   * @brief A field of depth alone, given at the layer's nodes.
   * @param values Its local components at each of the layer's depths,
   *        depth by depth.
   */
  [[nodiscard]] static LayerSource DepthValues(std::vector<double> values) {
    auto out = LayerSource{};
    out.values_ = std::move(values);
    return out;
  }

  /**
   * @brief A field that varies laterally.
   * @param f Its components at a point, in `frame`.
   * @param frame The frame `f` gives the components in.
   */
  [[nodiscard]] static LayerSource Lateral(OfPoint f, Frame frame) {
    auto out = LayerSource{};
    out.radial_ = false;
    out.frame_ = frame;
    out.of_point_ = std::move(f);
    return out;
  }

  /**
   * @brief A scalar that varies laterally.
   * @param f Its value at a point.
   */
  [[nodiscard]] static LayerSource Lateral(
      std::function<double(double depth, double x, double y)> f) {
    auto out = LayerSource{};
    out.radial_ = false;
    out.scalar_ = true;
    out.of_point_ = [f = std::move(f)](double depth, double x, double y, std::span<double> v) {
      v[0] = f(depth, x, y);
    };
    return out;
  }

  /**
   * @brief A field that varies laterally, given at the layer's points.
   * @param values Its components at each of the points `LayerPoints::At`
   *        gives, point by point, in `frame`.
   * @param frame The frame the components are in.
   */
  [[nodiscard]] static LayerSource LateralValues(std::vector<double> values,
                                                 Frame frame = Frame::Local) {
    auto out = LayerSource{};
    out.radial_ = false;
    out.frame_ = frame;
    out.values_ = std::move(values);
    return out;
  }

  /** @brief Whether the field is of depth alone. */
  [[nodiscard]] bool IsRadial() const noexcept { return radial_; }

 private:
  friend class ModelFile;

  LayerSource() = default;

  /// The stored values of the source at `points` for a variable of
  /// `character`, in the local frame: (depth, components) where it is of
  /// depth alone and `lateral` is false; otherwise (depth, nx, ny,
  /// components).
  std::vector<double> Sample(const LayerPoints& points, const Character& character,
                             bool lateral, const std::string& source,
                             std::string_view where) const {
    const auto n = character.StoredSize();
    const auto rank = character.Rank();
    const auto voigt = character.Voigt();
    const auto cells = points.grid.Size();
    const auto count = points.depths.size();
    if (scalar_ && n != 1) {
      detail::Fail(source, where,
                   std::format("is given by a scalar source, and stores {} components", n));
    }
    auto finite = [&](std::span<const double> v, const Point& at) {
      for (const auto x : v) {
        if (!std::isfinite(x)) {
          detail::Fail(source, where,
                       std::format("is not finite at (depth, x, y) = ({}, {}, {})", at.depth,
                                   at.x, at.y));
        }
      }
    };
    if (radial_) {
      auto local = std::vector<double>(count * n);
      if (values_) {
        if (values_->size() != count * n) {
          detail::Fail(source, where,
                       std::format("is given {} values, and its {} depths take {}",
                                   values_->size(), count, count * n));
        }
        local = *values_;
      } else {
        for (auto k = std::size_t{0}; k < count; ++k) {
          of_depth_(points.depths[k], std::span<double>(local).subspan(k * n, n));
        }
      }
      for (auto k = std::size_t{0}; k < count; ++k) {
        finite(std::span<const double>(local).subspan(k * n, n), {points.depths[k], 0.0, 0.0});
      }
      if (!lateral) return local;
      auto out = std::vector<double>(count * cells * n);
      for (auto k = std::size_t{0}; k < count; ++k) {
        for (auto cell = std::size_t{0}; cell < cells; ++cell) {
          std::copy_n(local.begin() + static_cast<std::ptrdiff_t>(k * n), n,
                      out.begin() + static_cast<std::ptrdiff_t>((k * cells + cell) * n));
        }
      }
      return out;
    }
    const auto size = points.Size();
    if (values_ && values_->size() != size * n) {
      detail::Fail(source, where,
                   std::format("is given {} values, and its {} points take {}", values_->size(),
                               size, size * n));
    }
    auto out = std::vector<double>{};
    out.reserve(size * n);
    auto buffer = std::vector<double>(n);
    for (auto index = std::size_t{0}; index < size; ++index) {
      const auto at = points.At(index);
      if (values_) {
        std::copy_n(values_->begin() + static_cast<std::ptrdiff_t>(index * n), n, buffer.begin());
      } else {
        of_point_(at.depth, at.x, at.y, buffer);
      }
      finite(buffer, at);
      const auto v = frame_ == Frame::Cartesian ? ToLocal(buffer, rank, voigt) : buffer;
      out.insert(out.end(), v.begin(), v.end());
    }
    return out;
  }

  bool radial_ = true;
  bool scalar_ = false;
  Frame frame_ = Frame::Local;
  OfDepth of_depth_{};
  OfPoint of_point_{};
  std::optional<std::vector<double>> values_{};
};

/** @brief A perturbation of a reference of depth alone, on some of its layers. */
struct Perturbation {
  std::string kind{};                  /**< One of `PERTURBATION_KINDS`. */
  std::map<int, LayerSource> layers{}; /**< The perturbed layers, each a layer of the reference. */
  Rules rules{};                       /**< The rules the perturbation states. */
};

class ModelFile;
class ScalarField;

/**
 * @brief A variable of a flat model file on one layer as a function of the
 * point, by the rules the file states.
 */
class Field {
 public:
  /** @brief The field's name. */
  [[nodiscard]] const std::string& Name() const noexcept { return name_; }
  /** @brief The layer it is on, from the surface down. */
  [[nodiscard]] int Layer() const noexcept { return layer_; }
  /** @brief The depths of the layer's top and bottom. */
  [[nodiscard]] std::pair<double, double> Interval() const noexcept { return interval_; }
  /** @brief The field's character, its law under a mapping. */
  [[nodiscard]] const Character& GetCharacter() const noexcept { return character_; }
  /** @brief The number of stored components. */
  [[nodiscard]] std::size_t Size() const { return character_.StoredSize(); }
  /** @brief Its dimensions, where the file states them. */
  [[nodiscard]] const std::optional<Dimensions>& GetDimensions() const noexcept {
    return dimensions_;
  }
  /** @brief Whether it depends on depth alone on its layer. */
  [[nodiscard]] bool IsRadial() const noexcept {
    return reference_->radial && (!perturbation_ || perturbation_->radial);
  }
  /** @brief The kind of its perturbation, where it is a perturbed field. */
  [[nodiscard]] const std::optional<std::string>& PerturbationKind() const noexcept {
    return kind_;
  }

  /**
   * @brief The stored components at a point by its flat coordinates.
   * @param depth The depth, within the layer to the file's tolerance.
   * @param x The first lateral coordinate, of any range: the field is
   *        periodic over the domain's extent.
   * @param y The second lateral coordinate, of any range; not read over a
   *        rectangle.
   * @param frame The frame the components are given in.
   * @return The components, `Size()` of them.
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] std::vector<double> Evaluate(double depth, double x, double y,
                                             Frame frame = Frame::Local) const {
    auto out = Read(*reference_, depth, x, y).value;
    if (perturbation_) {
      const auto delta = Read(*perturbation_, depth, x, y).value[0];
      out[0] = *kind_ == "absolute"   ? out[0] + delta
               : *kind_ == "relative" ? out[0] * (1.0 + delta)
                                      : out[0] * std::exp(delta);
    }
    return Framed(std::move(out), frame);
  }

  /**
   * @brief The stored components at a Cartesian point.
   * @param X The point (x, y, z), z up, its depth -z within the layer to
   *        the file's tolerance.
   * @param frame The frame the components are given in.
   * @return The components, `Size()` of them.
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] std::vector<double> EvaluateAt(const Vector3& X,
                                               Frame frame = Frame::Cartesian) const {
    const auto at = FlatPoint(X);
    return Evaluate(at.depth, at.x, at.y, frame);
  }

  /**
   * @brief The stored components' derivatives in depth, x and y at a point;
   * the derivative in depth is downward, d/d depth = -d/dz.
   * @param depth The depth, within the layer to the file's tolerance.
   * @param x The first lateral coordinate, of any range.
   * @param y The second lateral coordinate, of any range; not read over a
   *        rectangle, where the derivative in y is zero.
   * @param frame The frame the components are given in.
   * @return The derivatives in depth, x and y, `Size()` of each.
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] std::array<std::vector<double>, 3> Derivatives(
      double depth, double x, double y, Frame frame = Frame::Local) const {
    auto ref = Read(*reference_, depth, x, y, true);
    if (perturbation_) {
      // a perturbed field is of rank 0, its perturbation too
      const auto dp = Read(*perturbation_, depth, x, y, true);
      const auto value = ref.value[0];
      const auto delta = dp.value[0];
      for (auto d = std::size_t{0}; d < 3; ++d) {
        const auto d_ref = ref.derivative[d][0];
        const auto d_delta = dp.derivative[d][0];
        ref.derivative[d][0] = *kind_ == "absolute" ? d_ref + d_delta
                               : *kind_ == "relative"
                                   ? d_ref * (1.0 + delta) + value * d_delta
                                   : std::exp(delta) * (d_ref + value * d_delta);
      }
    }
    for (auto& d : ref.derivative) d = Framed(std::move(d), frame);
    return std::move(ref.derivative);
  }

  /**
   * @brief The field read as a number, for a field of rank zero.
   * @throws planetmodel::Error if the field's rank is not zero.
   */
  [[nodiscard]] ScalarField AsScalar() const;

 private:
  friend class ModelFile;

  /// What one variable is on one layer, at the nodes of its elements: the
  /// stored local components where it is of depth alone, or the lateral
  /// rule's interpolants of them.  Under a rule in the logarithm, a scalar
  /// of depth alone: `nodal` holds log(y / y_e) and `scale` each element's
  /// first value y_e.
  struct Piece {
    bool radial = true;
    Character character{0};
    RadialRule rule = RadialRule::Polynomial;
    std::optional<RadialLayout> layout{};
    std::vector<double> nodal{};               // [(node * ncomp) + c], of depth alone
    std::vector<double> scale{};               // [element], a rule in the logarithm
    std::vector<LateralInterpolant> lateral{}; // [(node * ncomp) + c], lateral
  };

  struct Reading {
    std::vector<double> value{};
    std::array<std::vector<double>, 3> derivative{};
  };

  Field() = default;

  /// Local components in `frame`.
  std::vector<double> Framed(std::vector<double> local, Frame frame) const {
    if (frame == Frame::Local || character_.Rank() == 0) return local;
    return ToCartesian(local, character_.Rank(), character_.Voigt());
  }

  /// A piece's local components at a point, and with `derivatives` their
  /// derivatives in depth, x and y.
  static Reading Read(const Piece& p, double depth, double x, double y,
                      bool derivatives = false) {
    const auto n = p.character.StoredSize();
    auto out = Reading{};
    out.value.assign(n, 0.0);
    if (derivatives) {
      for (auto& d : out.derivative) d.assign(n, 0.0);
    }
    // the element's weights in depth, once for every component
    const auto& layout = *p.layout;
    const auto e = layout.ElementOf(depth);
    const auto ngll = layout.Ngll();
    auto w = std::vector<double>(ngll);
    auto dw = std::vector<double>(derivatives ? ngll : 0);
    layout.Weights(e, depth, p.rule, w, dw);
    if (p.radial) {
      for (auto k = std::size_t{0}; k < ngll; ++k) {
        const auto* at = &p.nodal[(e * ngll + k) * n];
        for (auto c = std::size_t{0}; c < n; ++c) {
          out.value[c] += w[k] * at[c];
          if (derivatives) out.derivative[0][c] += dw[k] * at[c];
        }
      }
      if (IsLogRule(p.rule)) {
        // a scalar: the element's first value times the exponential of
        // the logarithm read, and its derivative that times the slope
        out.value[0] = p.scale[e] * std::exp(out.value[0]);
        if (derivatives) out.derivative[0][0] *= out.value[0];
      }
      return out;
    }
    for (auto k = std::size_t{0}; k < ngll; ++k) {
      for (auto c = std::size_t{0}; c < n; ++c) {
        const auto& f = p.lateral[((e * ngll) + k) * n + c];
        const auto v = f.Value(x, y);
        out.value[c] += w[k] * v;
        if (derivatives) {
          const auto [gx, gy] = f.Gradient(x, y);
          out.derivative[0][c] += dw[k] * v;
          out.derivative[1][c] += w[k] * gx;
          out.derivative[2][c] += w[k] * gy;
        }
      }
    }
    return out;
  }

  std::string source_{};
  std::string name_{};
  int layer_ = 0;
  std::pair<double, double> interval_{};
  Character character_{0};
  std::optional<Dimensions> dimensions_{};
  std::shared_ptr<const Piece> reference_{};
  std::shared_ptr<const Piece> perturbation_{};
  std::optional<std::string> kind_{};
};

/** @brief A field of a flat model file of rank zero, read as a number. */
class ScalarField {
 public:
  /**
   * @brief The value at a point by its flat coordinates.
   * @param depth The depth, within the layer to the file's tolerance.
   * @param x The first lateral coordinate, of any range.
   * @param y The second lateral coordinate, of any range; not read over a
   *        rectangle.
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] double Value(double depth, double x, double y) const {
    return field_.Evaluate(depth, x, y, Frame::Local)[0];
  }

  /**
   * @brief The value at a Cartesian point.
   * @param X The point (x, y, z), z up.
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] double ValueAt(const Vector3& X) const {
    return field_.EvaluateAt(X, Frame::Local)[0];
  }

  /**
   * @brief The derivatives in depth, x and y at a point; the derivative in
   * depth is downward, d/d depth = -d/dz.
   * @param depth The depth, within the layer to the file's tolerance.
   * @param x The first lateral coordinate, of any range.
   * @param y The second lateral coordinate, of any range; not read over a
   *        rectangle.
   * @return (d/d depth, d/dx, d/dy).
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] std::array<double, 3> Gradient(double depth, double x, double y) const {
    const auto d = field_.Derivatives(depth, x, y);
    return {d[0][0], d[1][0], d[2][0]};
  }

  /** @brief The field it reads. */
  [[nodiscard]] const Field& Underlying() const noexcept { return field_; }

 private:
  friend class Field;
  explicit ScalarField(Field field) : field_(std::move(field)) {}
  Field field_;
};

/**
 * @brief The field read as a number, for a field of rank zero.
 * @throws planetmodel::Error if the field's rank is not zero.
 */
inline ScalarField Field::AsScalar() const {
  if (character_.Rank() != 0) {
    throw Error(source_, name_,
                std::format("is of rank {}, not a scalar: read its components with Evaluate",
                            character_.Rank()));
  }
  return ScalarField(*this);
}

/**
 * @brief The mapping a flat model file holds, the vertical stretch
 * m(X) = X + w e_z of `planetmodel.cartesian.VerticalStretch`: every point
 * of the reference body rises by w(depth, x, y), a height, positive up, so
 * that x and y are kept and the point at depth d goes to z = w - d.  The
 * file holds w as the heights of surfaces, linear in depth between them,
 * or at the nodes of every part, or not at all (the identity).
 */
class VerticalStretch {
 public:
  /** @brief The form the file holds the mapping in. */
  enum class Form {
    Identity, /**< The file holds no mapping. */
    Stack,    /**< The `geometry` group: surfaces' heights, linear in depth between. */
    Field     /**< The `displacement` field of every part. */
  };

  /** @brief The form the mapping is held in. */
  [[nodiscard]] Form GetForm() const noexcept { return form_; }

  /**
   * @brief The upward displacement w at a reference point by its flat
   * coordinates: how far the point rises.
   * @param depth The depth, within the reference body to the file's
   *        tolerance.
   * @param x The first lateral coordinate, of any range.
   * @param y The second lateral coordinate, of any range; not read over a
   *        rectangle.
   * @throws planetmodel::Error if the depth lies outside the body.
   */
  [[nodiscard]] double Height(double depth, double x, double y) const {
    if (!(depth >= top_ - tolerance_ && depth <= base_ + tolerance_)) {
      throw Error(source_, "mapping",
                  std::format("depth = {} lies outside the body's [{}, {}]", depth, top_, base_));
    }
    if (form_ == Form::Identity) return 0.0;
    if (form_ == Form::Field) {
      return fields_[LayerOf(boundaries_, depth)].Evaluate(std::clamp(depth, top_, base_), x, y,
                                                           Frame::Local)[0];
    }
    // the heights of the surfaces above and below, joined linearly in depth
    const auto i = LayerOf(depths_, depth);
    const auto upper = Relief(i, x, y);
    const auto lower = Relief(i + 1, x, y);
    const auto t = (depth - depths_[i]) / (depths_[i + 1] - depths_[i]);
    return (1.0 - t) * upper + t * lower;
  }

  /**
   * @brief The displacement u(X) = m(X) - X = w e_z, Cartesian, at a
   * reference point.
   * @param X The point (x, y, z), z up, within the reference body to the
   *        file's tolerance.
   * @throws planetmodel::Error if the point lies outside the body.
   */
  [[nodiscard]] Vector3 Displacement(const Vector3& X) const {
    return {0.0, 0.0, Height(-X[2], X[0], X[1])};
  }

  /**
   * @brief The mapped point m(X) = X + u(X), Cartesian.
   * @param X The reference point (x, y, z), z up, within the body.
   * @throws planetmodel::Error if the point lies outside the body.
   */
  [[nodiscard]] Vector3 operator()(const Vector3& X) const {
    return {X[0], X[1], X[2] + Height(-X[2], X[0], X[1])};
  }

 private:
  friend class ModelFile;

  /// The layer of a depth among boundaries, the deeper one on a boundary,
  /// the end layers beyond them.
  static std::size_t LayerOf(const std::vector<double>& boundaries, double depth) {
    const auto i = std::upper_bound(boundaries.begin(), boundaries.end(), depth) -
                   boundaries.begin() - 1;
    return static_cast<std::size_t>(
        std::clamp<std::ptrdiff_t>(i, 0, static_cast<std::ptrdiff_t>(boundaries.size()) - 2));
  }

  double Relief(std::size_t k, double x, double y) const {
    const auto& f = reliefs_[k];
    return f ? f->Value(x, y) : 0.0;
  }

  Form form_ = Form::Identity;
  std::string source_{};
  double top_ = 0.0;
  double base_ = 0.0;
  double tolerance_ = 0.0;
  std::vector<double> boundaries_{};
  std::vector<::planetmodel::cartesian::Field> fields_{};
  std::vector<double> depths_{};
  std::vector<std::optional<LateralInterpolant>> reliefs_{};
};

/**
 * @brief A flat model file: read and checked, its numbers as stored, or
 * built to be written.
 */
class ModelFile {
 public:
  /**
   * @brief A flat model file to build: a skeleton of depths over a lateral
   * domain, in `scales`, the geometry's tolerance `rtol`, with no parts
   * yet.
   * @param skeleton The reference skeleton: the depths of the boundaries,
   *        from the free surface (or a hollowed body's top) down to the
   *        base.
   * @param domain The lateral domain, a box or a rectangle.
   * @param scales The scales the numbers are in.
   * @param rtol The geometry's tolerance, relative to the skeleton's span.
   * @throws planetmodel::Error if `rtol` is not in (0, 1).
   */
  ModelFile(::planetmodel::Skeleton skeleton, Domain domain,
            ::planetmodel::Scales scales = ::planetmodel::Scales::SI(),
            double rtol = DEFAULT_RTOL)
      : source_("ModelFile"),
        schema_(MODEL_SCHEMA),
        scales_(scales),
        rtol_(rtol),
        skeleton_(std::move(skeleton)),
        domain_(std::move(domain)) {
    if (!(rtol_ > 0.0 && rtol_ < 1.0)) Fail("rtol", "is not a tolerance in (0, 1)");
    layer_names_.assign(skeleton_->NLayers(), std::nullopt);
    interface_depths_ = detail::InterfaceDepths(*skeleton_);
    interface_names_.assign(interface_depths_.size(), std::nullopt);
  }

  /** @brief The path it was read from, as errors name it; `ModelFile` for one being built. */
  [[nodiscard]] const std::string& Source() const noexcept { return source_; }
  /** @brief The schema it declares: `MODEL_SCHEMA`. */
  [[nodiscard]] const std::string& Schema() const noexcept { return schema_; }
  /** @brief The code that wrote the file (the attribute `source`). */
  [[nodiscard]] const std::string& WrittenBy() const noexcept { return written_by_; }
  /** @brief When the file was written, ISO 8601 (the attribute `created`). */
  [[nodiscard]] const std::string& Created() const noexcept { return created_; }
  /** @brief The model's name, where it has one. */
  [[nodiscard]] const std::optional<std::string>& Name() const noexcept { return name_; }
  /** @brief The works the model cites, one a string, in the order stated; empty for none. */
  [[nodiscard]] const std::vector<std::string>& Citation() const noexcept { return citation_; }
  /**
   * @brief The names of the model's behaviours (the attribute `behaviours`),
   * in the order stated; empty for none.  They say what kind of model the
   * numbers are, and are written again as they were read.
   */
  [[nodiscard]] const std::vector<std::string>& Behaviours() const noexcept {
    return behaviours_;
  }
  /** @brief The scales the numbers are in. */
  [[nodiscard]] const ::planetmodel::Scales& GetScales() const noexcept { return scales_; }
  /** @brief The geometry's tolerance, relative to the skeleton's span. */
  [[nodiscard]] double Rtol() const noexcept { return rtol_; }
  /** @brief The model's constants by name, in its units. */
  [[nodiscard]] const std::map<std::string, FileConstant>& Constants() const noexcept {
    return constants_;
  }
  /** @brief The writer's `meta`, by key. */
  [[nodiscard]] const netcdf::Attributes& Meta() const noexcept { return meta_; }
  /** @brief The reference skeleton: the boundaries' depths, the top first. */
  [[nodiscard]] const ::planetmodel::Skeleton& GetSkeleton() const noexcept { return *skeleton_; }
  /** @brief The lateral domain the body extends over. */
  [[nodiscard]] const Domain& GetDomain() const noexcept { return *domain_; }
  /** @brief Each layer's name, from the surface down, empty where it has none. */
  [[nodiscard]] const std::vector<std::optional<std::string>>& LayerNames() const noexcept {
    return layer_names_;
  }
  /** @brief Each interface's name, from the surface down, empty where it has none. */
  [[nodiscard]] const std::vector<std::optional<std::string>>& InterfaceNames() const noexcept {
    return interface_names_;
  }
  /** @brief Each interface's depth: interface k is the top of layer k, the
   * base being a face of the box and no interface. */
  [[nodiscard]] const std::vector<double>& InterfaceDepths() const noexcept {
    return interface_depths_;
  }
  /** @brief The parts, from the surface down. */
  [[nodiscard]] const std::vector<FilePart>& Parts() const noexcept { return parts_; }
  /** @brief The `geometry` group, where the file holds the mapping so. */
  [[nodiscard]] const std::optional<FileGeometry>& Geometry() const noexcept { return geometry_; }
  /** @brief The `surfaces` group, where the file holds fields on interfaces. */
  [[nodiscard]] const std::optional<FileSurfaces>& Surfaces() const noexcept { return surfaces_; }

  /**
   * @brief The names of the fields an interface holds.
   * @param interface The interface, from the surface down.
   */
  [[nodiscard]] std::vector<std::string> SurfaceNames(int interface) const {
    auto out = std::vector<std::string>{};
    const auto row = surfaces_ ? surfaces_->Row(interface) : std::nullopt;
    if (!row) return out;
    for (const auto& [name, var] : surfaces_->variables) {
      if (HoldsRow(var, *row)) out.push_back(name);
    }
    return out;
  }

  /**
   * @brief Whether an interface holds a field.
   * @param interface The interface, from the surface down.
   * @param name The field.
   */
  [[nodiscard]] bool HasSurfaceField(int interface, std::string_view name) const {
    return std::ranges::contains(SurfaceNames(interface), std::string{name});
  }

  /**
   * @brief A field on an interface, a function of (x, y) by the lateral
   * rule the file states.
   * @param interface The interface, from the surface down.
   * @param name The field.
   * @return Its interpolant: `Value(x, y)` and `Gradient(x, y)`.
   * @throws planetmodel::Error if the interface does not hold the field, or
   *         its stated lateral rule is `fourier`, which the C++ does not
   *         read.
   */
  [[nodiscard]] LateralInterpolant SurfaceFieldOn(int interface, std::string_view name) const {
    const auto where = std::format("surfaces/{}", name);
    if (!HasSurfaceField(interface, name)) {
      throw Error(source_, where, std::format("is not held on interface {}", interface));
    }
    const auto& var = surfaces_->variables.at(std::string{name});
    const auto row = *surfaces_->Row(interface);
    const auto& grid = *surfaces_->grid;
    const auto cells = grid.Size();
    const auto rule = LateralRuleNamed(*var.Text("lateral_interpolation"), source_, where);
    return LateralInterpolant(grid, std::span<const double>(var.values).subspan(row * cells, cells),
                              rule, where);
  }

  /**
   * @brief The part holding a layer.
   * @param layer The layer.
   * @throws planetmodel::Error if no part holds it.
   */
  [[nodiscard]] const FilePart& PartOf(int layer) const {
    for (const auto& p : parts_) {
      if (std::ranges::contains(p.layers, layer)) return p;
    }
    throw Error(source_, "", std::format("no part holds layer {}", layer));
  }

  /**
   * @brief The names of the fields a layer holds.
   * @param layer The layer.
   * @throws planetmodel::Error if no part holds the layer.
   */
  [[nodiscard]] std::vector<std::string> Names(int layer) const {
    const auto& part = PartOf(layer);
    auto out = std::vector<std::string>{};
    for (const auto& [name, var] : part.variables) {
      if (IsFieldName(name) && Holds(part, layer, var)) out.push_back(name);
    }
    return out;
  }

  /**
   * @brief Whether a layer holds a field.
   * @param layer The layer.
   * @param name The field.
   */
  [[nodiscard]] bool HasField(int layer, std::string_view name) const {
    return std::ranges::contains(Names(layer), std::string{name});
  }

  /**
   * @brief A field of one variable on one layer, by the rules the file
   * states.
   * @param layer The layer, from the surface down.
   * @param name The variable.
   * @throws planetmodel::Error if the layer does not hold the variable as
   *         a field of the model, or its stated lateral rule is `fourier`,
   *         which the C++ does not read.
   */
  [[nodiscard]] ::planetmodel::cartesian::Field FieldOn(int layer, std::string_view name) const {
    const auto& part = PartOf(layer);
    const auto it = part.variables.find(std::string{name});
    if (name == "displacement") {
      throw Error(source_, std::format("part_{}/displacement", part.index),
                  "is the mapping, not a field of the model: read it with Mapping()");
    }
    if (it == part.variables.end() || !IsFieldName(it->first) ||
        !Holds(part, layer, it->second)) {
      throw Error(source_, std::format("part_{}/{}", part.index, name),
                  std::format("is not held on layer {}", layer));
    }
    const auto& var = it->second;
    auto out = ::planetmodel::cartesian::Field{};
    out.source_ = source_;
    out.name_ = std::string{name};
    out.layer_ = layer;
    out.interval_ = skeleton_->Interval(static_cast<std::size_t>(layer));
    out.character_ = CharacterOf(var);
    out.dimensions_ = detail::DimensionsFrom(var.Numbers("dimensions"));
    out.reference_ = std::make_shared<const ::planetmodel::cartesian::Field::Piece>(
        Piece(part, layer, var, out.character_));
    if (const auto other = var.Text("perturbation")) {
      const auto& pert = part.variables.at(*other);
      if (out.character_.Rank() != 0 || CharacterOf(pert).Rank() != 0) {
        throw Error(source_, std::format("part_{}/{}", part.index, name),
                    "is perturbed, and a perturbed field and its perturbation are of "
                    "rank 0");
      }
      if (Holds(part, layer, pert)) {
        out.kind_ = *var.Text("perturbation_kind");
        out.perturbation_ = std::make_shared<const ::planetmodel::cartesian::Field::Piece>(
            Piece(part, layer, pert, CharacterOf(pert)));
      }
    }
    return out;
  }

  /**
   * @brief A field per layer that holds a variable, from the surface down.
   * @param name The variable.
   * @throws planetmodel::Error as `FieldOn` does.
   */
  [[nodiscard]] std::vector<::planetmodel::cartesian::Field> FieldsOf(
      std::string_view name) const {
    auto out = std::vector<::planetmodel::cartesian::Field>{};
    for (auto layer = 0; layer < static_cast<int>(skeleton_->NLayers()); ++layer) {
      if (HasField(layer, name)) out.push_back(FieldOn(layer, name));
    }
    return out;
  }

  /**
   * @brief The mapping the file holds, a vertical stretch, the identity
   * where it holds none.
   * @throws planetmodel::Error if the mapping's lateral rule is `fourier`.
   */
  [[nodiscard]] VerticalStretch Mapping() const {
    auto out = VerticalStretch{};
    out.source_ = source_;
    out.top_ = skeleton_->Boundaries().front();
    out.base_ = skeleton_->Boundaries().back();
    out.tolerance_ = rtol_ * skeleton_->Span();
    out.boundaries_ = skeleton_->Boundaries();
    if (geometry_) {
      out.form_ = VerticalStretch::Form::Stack;
      out.depths_ = geometry_->depths;
      const auto& g = *geometry_->grid;
      const auto rule = LateralRuleNamed(geometry_->lateral, source_, "geometry");
      const auto cells = g.Size();
      for (auto k = std::size_t{0}; k < geometry_->depths.size(); ++k) {
        const auto values = std::span<const double>(geometry_->values).subspan(k * cells, cells);
        if (std::ranges::all_of(values, [](double v) { return v == 0.0; })) {
          out.reliefs_.emplace_back(std::nullopt);
        } else {
          out.reliefs_.emplace_back(LateralInterpolant(g, values, rule, "geometry"));
        }
      }
      return out;
    }
    if (!parts_.empty() && parts_.front().variables.contains("displacement")) {
      out.form_ = VerticalStretch::Form::Field;
      for (auto layer = 0; layer < static_cast<int>(skeleton_->NLayers()); ++layer) {
        out.fields_.push_back(Displacement(layer));
      }
    }
    return out;
  }

  /**
   * @brief Name the model.
   * @param name The name, or empty for none.
   */
  void SetName(std::string name) {
    name_ = name.empty() ? std::nullopt : std::optional<std::string>{std::move(name)};
  }

  /**
   * @brief Name the writing code and version, the `source` attribute of
   * the written file; this library and its version unless set.
   * @param written_by The code and version.
   */
  void SetWrittenBy(std::string written_by) { written_by_ = std::move(written_by); }

  /**
   * @brief State the works the model cites, in place of any stated before.
   * @param works The works, one a string, each trimmed of white space at its ends, an
   *              empty one or a repeat dropped.
   * @throws planetmodel::Error if a work is more than one line.
   */
  void SetCitation(const std::vector<std::string>& works) {
    citation_ = detail::Cited(works, source_);
  }

  /**
   * @brief State the behaviours of the model, in place of any stated before:
   * the names a reader of the file gives the model's methods by.
   * @param names The names, each one word, in the order the model has them.
   * @throws planetmodel::Error if a name is empty, holds white space or is given twice.
   */
  void SetBehaviours(const std::vector<std::string>& names) {
    behaviours_ = detail::BehavioursNamed(names, source_);
  }

  /**
   * @brief Name a layer.
   * @param layer The layer, from the surface down.
   * @param name The name, or empty for none.
   * @throws planetmodel::Error if there is no such layer.
   */
  void SetLayerName(int layer, std::string name) {
    if (layer < 0 || layer >= static_cast<int>(layer_names_.size())) {
      Fail("layer_name", std::format("has no layer {}", layer));
    }
    layer_names_[static_cast<std::size_t>(layer)] =
        name.empty() ? std::nullopt : std::optional<std::string>{std::move(name)};
  }

  /**
   * @brief Name an interface.
   * @param interface The interface, from the surface down.
   * @param name The name, or empty for none.
   * @throws planetmodel::Error if there is no such interface.
   */
  void SetInterfaceName(int interface, std::string name) {
    if (interface < 0 || interface >= static_cast<int>(interface_names_.size())) {
      Fail("interface_name", std::format("has no interface {}", interface));
    }
    interface_names_[static_cast<std::size_t>(interface)] =
        name.empty() ? std::nullopt : std::optional<std::string>{std::move(name)};
  }

  /**
   * @brief A constant of the model.
   * @param name Its name.
   * @param constant Its value in the model's units, its dimensions and
   *        meaning.
   * @throws planetmodel::Error if the value is not finite.
   */
  void SetConstant(const std::string& name, FileConstant constant) {
    if (!std::isfinite(constant.value)) Fail("constant_" + name, "is not finite");
    constants_.insert_or_assign(name, std::move(constant));
  }

  /**
   * @brief An entry of the writer's `meta`, text.
   * @param key The key, written as `meta_<key>`.
   * @param value The text.
   */
  void SetMeta(const std::string& key, std::string value) {
    meta_.insert_or_assign(key, std::move(value));
  }

  /**
   * @brief An entry of the writer's `meta`, a real number.
   * @param key The key, written as `meta_<key>`.
   * @param value The number.
   */
  void SetMeta(const std::string& key, double value) {
    meta_.insert_or_assign(key, std::vector<double>{value});
  }

  /**
   * @brief An entry of the writer's `meta`, an integer.
   * @tparam T An integer type other than `bool`.
   * @param key The key, written as `meta_<key>`.
   * @param value The integer.
   */
  template <std::integral T>
    requires(!std::same_as<T, bool>)
  void SetMeta(const std::string& key, T value) {
    meta_.insert_or_assign(key, std::vector<long long>{static_cast<long long>(value)});
  }

  /**
   * @brief A part: the layers `first` to `last`, laid out in elements of
   * `ngll` Gauss-Lobatto-Legendre nodes between `edges`, on `grid`.
   *
   * Parts are added from the surface down, each starting at the layer
   * after the last one's.  An edge within the file's tolerance of a
   * boundary is taken as on it.
   *
   * @param first The top layer of the part.
   * @param last The bottom layer of the part.
   * @param edges The depths of the elements' ends, strictly increasing,
   *        from the top of `first` to the bottom of `last` and at every
   *        boundary between (`ElementEdges` gives equal elements).
   * @param ngll The nodes per element, at least two.
   * @param grid The part's horizontal grid, over the file's domain.
   * @throws planetmodel::Error if the part does not follow the last, the
   *         edges or `ngll` are not as above, or the grid lies over another
   *         domain.
   */
  void AddPart(int first, int last, std::vector<double> edges, std::size_t ngll, Grid grid) {
    const auto where = std::format("part_{}", parts_.size());
    const auto next = parts_.empty() ? 0 : parts_.back().layers.back() + 1;
    const auto nlayers = static_cast<int>(skeleton_->NLayers());
    if (first != next) {
      Fail(where, std::format("starts at layer {}, and parts are added from the surface down "
                              "from layer {}",
                              first, next));
    }
    if (last < first || last >= nlayers) {
      Fail(where, std::format("ends at layer {}, outside [{}, {}]", last, first, nlayers - 1));
    }
    if (ngll < 2) Fail(where, std::format("ngll = {}; an element has two nodes or more", ngll));
    if (grid.GetDomain() != *domain_) {
      Fail(where, "its grid lies over another domain: every grid of a flat file covers the "
                  "model's domain");
    }
    const auto& b = skeleton_->Boundaries();
    const auto tolerance = rtol_ * skeleton_->Span();
    for (auto& e : edges) {
      for (auto i = first; i <= last + 1; ++i) {
        if (std::abs(e - b[static_cast<std::size_t>(i)]) <= tolerance) {
          e = b[static_cast<std::size_t>(i)];
        }
      }
    }
    if (edges.size() < 2 || edges.front() != b[static_cast<std::size_t>(first)] ||
        edges.back() != b[static_cast<std::size_t>(last) + 1]) {
      Fail(where, "its edges do not run from the top of its first layer to the bottom of its "
                  "last");
    }
    for (auto k = std::size_t{1}; k < edges.size(); ++k) {
      if (!(edges[k] > edges[k - 1])) Fail(where, "its edges do not increase strictly");
    }
    for (auto i = first + 1; i <= last; ++i) {
      if (!std::ranges::contains(edges, b[static_cast<std::size_t>(i)])) {
        Fail(where, std::format("its edges leave out the boundary between layers {} and {}",
                                i - 1, i));
      }
    }
    auto part = FilePart{.index = static_cast<int>(parts_.size()), .ngll = ngll};
    for (auto i = first; i <= last; ++i) part.layers.push_back(i);
    for (auto e = std::size_t{0}; e + 1 < edges.size(); ++e) {
      const auto middle = 0.5 * (edges[e] + edges[e + 1]);
      const auto layer = std::upper_bound(b.begin(), b.end(), middle) - b.begin() - 1;
      part.element_layer.push_back(static_cast<int>(layer));
      part.element_bounds.push_back(edges[e]);
      part.element_bounds.push_back(edges[e + 1]);
      const auto nodes = GllNodes(edges[e], edges[e + 1], ngll);
      part.depth.insert(part.depth.end(), nodes.begin(), nodes.end());
    }
    part.grid = std::move(grid);
    parts_.push_back(std::move(part));
  }

  /**
   * @brief The points a layer's values are given at, for a `LayerSource` of
   * values.
   * @param layer The layer, which a part holds.
   * @throws planetmodel::Error if no part holds the layer.
   */
  [[nodiscard]] LayerPoints PointsOf(int layer) const {
    const auto& part = PartOf(layer);
    const auto [e0, e1] = part.Elements(layer);
    auto depths = std::vector<double>(
        part.depth.begin() + static_cast<std::ptrdiff_t>(e0 * part.ngll),
        part.depth.begin() + static_cast<std::ptrdiff_t>(e1 * part.ngll));
    return {std::move(depths), *part.grid};
  }

  /**
   * @brief A field of a name the vocabulary holds, its spec from there.
   * @param name The name.
   * @param layers Each layer holding it, and its source there.
   * @param rules The rules the variable states.
   * @throws planetmodel::Error if the vocabulary has no such name, or as
   *         the other `AddField` does.
   */
  void AddField(const std::string& name, const std::map<int, LayerSource>& layers,
                const Rules& rules = {}) {
    AddField(name, KnownSpec(name), layers, rules);
  }

  /**
   * @brief A field: the variable `name` on each layer of `layers`, from its
   * source there, in the part holding the layer.
   *
   * In a part where every layer holding it has a source of depth alone the
   * variable is `name(node)`; otherwise it varies laterally, the layers of
   * depth alone named in its `radial_layers`.
   *
   * @param name The name, which the format does not keep for itself and the
   *        file does not yet hold.
   * @param spec Its character, dimensions and meaning; a vocabulary name's
   *        must be the vocabulary's.
   * @param layers Each layer holding it, and its source there.
   * @param rules The rules the variable states; a radial rule in the
   *        logarithm for a scalar of depth alone, positive at every node.
   * @throws planetmodel::Error if the name or the spec is refused, a layer
   *         is in no part, the grid cannot carry the lateral rule, a radial
   *         rule in the logarithm is stated by a variable that varies
   *         laterally or has components, or meets a value that is not
   *         positive, or a source gives values that do not fit or are not
   *         finite.
   */
  void AddField(const std::string& name, const FieldSpec& spec,
                const std::map<int, LayerSource>& layers, const Rules& rules = {}) {
    CheckNewName(name, spec);
    auto made = std::vector<std::pair<std::size_t, FileVariable>>{};
    for (const auto& [index, holders] : ByPart(name, layers)) {
      auto radial = true;
      for (const auto layer : holders) radial = radial && layers.at(layer).IsRadial();
      made.emplace_back(index, Variable(parts_[index], name, spec, spec.character, holders,
                                        layers, rules, !radial));
    }
    Commit(std::move(made));
  }

  /**
   * @brief A perturbed field of a name the vocabulary holds, its spec from
   * there.
   * @param name The name.
   * @param reference Each layer of the reference, of depth alone.
   * @param perturbation The perturbation and the layers it perturbs.
   * @param reference_rule The radial rule the reference states.
   * @throws planetmodel::Error if the vocabulary has no such name, or as
   *         the other `AddPerturbedField` does.
   */
  void AddPerturbedField(const std::string& name, const std::map<int, LayerSource>& reference,
                         const Perturbation& perturbation,
                         RadialRule reference_rule = RadialRule::Polynomial) {
    AddPerturbedField(name, KnownSpec(name), reference, perturbation, reference_rule);
  }

  /**
   * @brief A perturbed field: a reference of depth alone and a
   * perturbation of it on some of its layers, combined by the
   * perturbation's kind.
   *
   * In each part holding a perturbed layer the field is two variables,
   * `name(node)`, the reference, and `name_perturbation`, of the field's
   * dimensions where the kind is `absolute` and a pure number otherwise; in
   * a part with no perturbed layer it is the reference alone.
   *
   * @param name The name, which the format does not keep for itself and the
   *        file does not yet hold.
   * @param spec Its spec, of rank 0; a vocabulary name's must be the
   *        vocabulary's.
   * @param reference Each layer of the reference, of depth alone.
   * @param perturbation The perturbation and the layers it perturbs.
   * @param reference_rule The radial rule the reference states.
   * @throws planetmodel::Error as `AddField` does, or if the field is not
   *         of rank 0, the kind is unknown, a reference layer's source
   *         varies laterally, or a perturbed layer is not one of the
   *         reference's.
   */
  void AddPerturbedField(const std::string& name, const FieldSpec& spec,
                         const std::map<int, LayerSource>& reference,
                         const Perturbation& perturbation,
                         RadialRule reference_rule = RadialRule::Polynomial) {
    CheckNewName(name, spec);
    if (spec.character.Rank() != 0) {
      Fail(name, "is perturbed, and a perturbed field is of rank 0");
    }
    if (!std::ranges::contains(PERTURBATION_KINDS, std::string_view{perturbation.kind})) {
      Fail(name, std::format("states the perturbation_kind {}, which is not one of absolute, "
                             "relative and log",
                             perturbation.kind));
    }
    for (const auto& [layer, source] : reference) {
      if (!source.IsRadial()) {
        Fail(name, std::format("has a reference that varies laterally on layer {}; a reference "
                               "is of depth alone",
                               layer));
      }
    }
    if (perturbation.layers.empty()) Fail(name, "is perturbed on no layer");
    for (const auto& [layer, source] : perturbation.layers) {
      if (!reference.contains(layer)) {
        Fail(name, std::format("is perturbed on layer {}, where it has no reference", layer));
      }
    }
    const auto pert_name = name + "_perturbation";
    const auto absolute = perturbation.kind == "absolute";
    auto pert_spec = FieldSpec{.character = absolute ? spec.character : SCALAR,
                               .dimensions = absolute ? spec.dimensions : units::DIMENSIONLESS,
                               .meaning = std::format("{} perturbation of {}", perturbation.kind,
                                                      name)};
    auto made = std::vector<std::pair<std::size_t, FileVariable>>{};
    for (const auto& [index, holders] : ByPart(name, reference)) {
      const auto& part = parts_[index];
      auto perturbed = std::vector<int>{};
      for (const auto layer : holders) {
        if (perturbation.layers.contains(layer)) perturbed.push_back(layer);
      }
      auto ref = Variable(part, name, spec, spec.character, holders, reference,
                          {.radial = reference_rule}, false);
      if (!perturbed.empty()) {
        ref.attributes.insert_or_assign("perturbation", pert_name);
        ref.attributes.insert_or_assign("perturbation_kind", perturbation.kind);
        auto radial = true;
        for (const auto layer : perturbed) {
          radial = radial && perturbation.layers.at(layer).IsRadial();
        }
        auto pert = Variable(part, pert_name, pert_spec, pert_spec.character, perturbed,
                             perturbation.layers, perturbation.rules, !radial);
        if (!absolute) pert.attributes.insert_or_assign("units", std::string{"1"});
        pert.attributes.insert_or_assign("long_name", pert_spec.meaning);
        made.emplace_back(index, std::move(pert));
      }
      made.emplace_back(index, std::move(ref));
    }
    Commit(std::move(made));
  }

  /**
   * @brief The mapping as the heights of surfaces, linear in depth between
   * them: the `geometry` group.
   * @param depths The surfaces' reference depths, strictly increasing,
   *        from the skeleton's top boundary to its base, none negative;
   *        they may lie inside layers.
   * @param grid The grid the heights are held on, over the file's domain.
   * @param reliefs Each surface's height above its reference place,
   *        positive up, a scalar source (evaluated at the surface's
   *        depth), or empty where it stays flat.
   * @param lateral The lateral rule the heights state.
   * @throws planetmodel::Error if the file holds a mapping already, the
   *         surfaces are not as above, the grid lies over another domain
   *         or cannot carry the rule, or a relief gives values that do not
   *         fit or are not finite.
   */
  void SetStack(std::vector<double> depths, Grid grid,
                const std::vector<std::optional<LayerSource>>& reliefs,
                LateralRule lateral = LateralRule::Cubic) {
    const auto where = std::string{"geometry/boundary_displacement"};
    if (HoldsMapping()) Fail("geometry", "the file holds a mapping already");
    if (depths.size() < 2 || reliefs.size() != depths.size()) {
      Fail(where, std::format("is given {} reliefs for {} surfaces; a stack is two surfaces "
                              "or more, each with a relief",
                              reliefs.size(), depths.size()));
    }
    for (auto k = std::size_t{1}; k < depths.size(); ++k) {
      if (!(depths[k] > depths[k - 1])) {
        Fail(where, "its surfaces' depths do not increase strictly");
      }
    }
    if (!(depths.front() >= 0.0)) {
      Fail(where, "its first surface lies above the free surface, at a negative depth");
    }
    const auto tolerance = rtol_ * skeleton_->Span();
    if (depths.front() > skeleton_->Boundaries().front() + tolerance ||
        depths.back() < skeleton_->Boundaries().back() - tolerance) {
      Fail(where, "its surfaces do not span the skeleton");
    }
    if (grid.GetDomain() != *domain_) {
      Fail(where, "its grid lies over another domain: the geometry group holds the heights on "
                  "a grid over the model's domain");
    }
    CheckRule(grid, lateral, where);
    const auto cells = grid.Size();
    auto geometry = FileGeometry{.depths = depths,
                                 .values = std::vector<double>(depths.size() * cells),
                                 .lateral = std::string{RuleName(lateral)},
                                 .between = "linear"};
    for (auto k = std::size_t{0}; k < depths.size(); ++k) {
      if (!reliefs[k]) continue;
      const auto points = LayerPoints{{depths[k]}, grid};
      const auto sampled = reliefs[k]->Sample(points, SCALAR, true, source_, where);
      std::ranges::copy(sampled, geometry.values.begin() + static_cast<std::ptrdiff_t>(k * cells));
    }
    geometry.grid = std::move(grid);
    geometry_ = std::move(geometry);
  }

  /**
   * @brief The mapping as its upward displacement w, a height, positive
   * up, on every layer, in every part: the `displacement` variable.
   *
   * Every part is on one grid, so that the parts' displacements agree where
   * they meet; the parts are added first.
   *
   * @param layers Each layer of the skeleton, and the displacement's source
   *        there, a scalar.
   * @param rules The rules the displacement states.
   * @throws planetmodel::Error if the file holds a mapping already, its
   *         parts do not hold every layer on one grid, a layer has no
   *         source, the grid cannot carry the rule, or a source gives values
   *         that do not fit or are not finite.
   */
  void SetDisplacement(const std::map<int, LayerSource>& layers,
                       const Rules& rules = {.lateral = LateralRule::Cubic}) {
    if (HoldsMapping()) Fail("displacement", "the file holds a mapping already");
    const auto nlayers = static_cast<int>(skeleton_->NLayers());
    if (parts_.empty() || parts_.back().layers.back() != nlayers - 1) {
      Fail("displacement", "is set once the parts hold every layer");
    }
    for (const auto& part : parts_) {
      if (!(*part.grid == *parts_.front().grid)) {
        Fail("displacement",
             "is held in every part, read by the part's own rule, so the parts must share "
             "one grid: give every layer one grid, or the mapping as the heights of "
             "surfaces with SetStack");
      }
    }
    for (auto layer = 0; layer < nlayers; ++layer) {
      if (!layers.contains(layer)) {
        Fail("displacement", std::format("has no source on layer {}", layer));
      }
    }
    const auto spec = FieldSpec{.character = SCALAR, .dimensions = units::LENGTH,
                                .meaning = "upward displacement of the vertical stretch, a "
                                           "height above the node's reference place"};
    auto made = std::vector<std::pair<std::size_t, FileVariable>>{};
    for (auto index = std::size_t{0}; index < parts_.size(); ++index) {
      auto var = Variable(parts_[index], "displacement", spec, SCALAR, parts_[index].layers,
                          layers, rules, true);
      for (const auto* key : {"rank", "weight", "voigt", "layers", "radial_layers"}) {
        var.attributes.erase(key);
      }
      var.attributes.insert_or_assign("positive", std::string{"up"});
      made.emplace_back(index, std::move(var));
    }
    Commit(std::move(made));
  }

  /**
   * @brief The grid the fields on interfaces are held on, set before the
   * first of them is added.
   * @param grid The grid, over the file's domain.
   * @throws planetmodel::Error if the file holds a field on an interface
   *         already, or the grid lies over another domain.
   */
  void SetSurfaceGrid(Grid grid) {
    if (surfaces_ && !surfaces_->variables.empty()) {
      Fail("surfaces", "holds fields already, on the grid set before them");
    }
    if (grid.GetDomain() != *domain_) {
      Fail("surfaces", "its grid lies over another domain: every grid of a flat file covers "
                       "the model's domain");
    }
    surfaces_ = FileSurfaces{.grid = std::move(grid)};
  }

  /**
   * @brief The points the values of a field on an interface are given at,
   * for a `LayerSource` of values: the interface's depth on the surface
   * grid.
   * @param interface The interface, from the surface down.
   * @throws planetmodel::Error if the skeleton has no such interface, or the
   *         surface grid is not set.
   */
  [[nodiscard]] LayerPoints SurfacePointsOf(int interface) const {
    CheckInterface(interface, "surfaces");
    if (!surfaces_) Fail("surfaces", "has no grid: set it with SetSurfaceGrid first");
    return {{interface_depths_[static_cast<std::size_t>(interface)]}, *surfaces_->grid};
  }

  /**
   * @brief A field on interfaces: the variable `name` of the `surfaces`
   * group, from its source on each of `interfaces`, evaluated at the
   * interface's depth on the surface grid, NaN on the other interfaces the
   * group holds.
   * @param name The name, which the format does not keep for itself and the
   *        file does not yet hold.
   * @param spec Its dimensions and meaning, of rank 0; a vocabulary name's
   *        must be the vocabulary's.
   * @param interfaces Each interface holding it, from the surface down, and
   *        its source there, a scalar.
   * @param lateral The lateral rule the variable states.
   * @throws planetmodel::Error if the name or the spec is refused, the spec
   *         is not of rank 0, the surface grid is not set, an interface is
   *         not the skeleton's, the grid cannot carry the rule, or a source
   *         gives values that do not fit or are not finite.
   */
  void AddSurfaceField(const std::string& name, const FieldSpec& spec,
                       const std::map<int, LayerSource>& interfaces,
                       LateralRule lateral = LateralRule::Linear) {
    const auto where = "surfaces/" + name;
    CheckNewName(name, spec);
    if (spec.character.Rank() != 0) {
      Fail(where, std::format("is of rank {} by its spec, and a surface field is a number at "
                              "each point",
                              spec.character.Rank()));
    }
    if (!surfaces_) Fail("surfaces", "has no grid: set it with SetSurfaceGrid first");
    if (interfaces.empty()) Fail(where, "is held on no interface");
    for (const auto& [k, source] : interfaces) CheckInterface(k, where);
    const auto& grid = *surfaces_->grid;
    CheckRule(grid, lateral, where);
    // every interface holding anything once this is added, increasing
    auto rows = surfaces_->interfaces;
    for (const auto& [k, source] : interfaces) {
      if (!std::ranges::contains(rows, k)) rows.push_back(k);
    }
    std::ranges::sort(rows);
    auto held = FileSurfaces{.interfaces = rows, .grid = grid};
    for (const auto& [vname, var] : surfaces_->variables) {
      held.variables.emplace(vname, Rowed(var, surfaces_->interfaces, rows));
    }
    const auto cells = grid.Size();
    auto var = FileVariable{.name = name};
    var.dimensions = {"surface"};
    var.shape = {rows.size()};
    AppendGrid(var, grid);
    var.values.assign(rows.size() * cells, std::numeric_limits<double>::quiet_NaN());
    auto holders = std::vector<int>{};
    for (const auto& [k, source] : interfaces) {
      const auto row = static_cast<std::size_t>(std::ranges::find(rows, k) - rows.begin());
      const auto sampled = source.Sample(SurfacePointsOf(k), SCALAR, true, source_,
                                         std::format("{} on interface {}", where, k));
      std::ranges::copy(sampled, var.values.begin() + static_cast<std::ptrdiff_t>(row * cells));
      holders.push_back(k);
    }
    auto& a = var.attributes;
    a.emplace("long_name", spec.meaning.empty() ? name : spec.meaning);
    a.emplace("units", units::ScaledUnit(spec.dimensions, scales_));
    if (spec.dimensions) a.emplace("dimensions", detail::Exponents(*spec.dimensions));
    a.emplace("lateral_interpolation", std::string{RuleName(lateral)});
    a.emplace("interfaces", detail::Integers(holders));
    held.variables.emplace(name, std::move(var));
    surfaces_ = std::move(held);
  }

 private:
  friend ModelFile ReadModelFile(const std::filesystem::path& path);
  friend void ValidateModelFile(const ModelFile& file, const std::string& source);
  friend void WriteModelFile(const std::filesystem::path& path, const ModelFile& file);

  ModelFile() = default;

  /// The file's groups, variables and attributes, written into a new file.
  void WriteTo(int root, const std::string& source) const;

  /// A refusal of what is being built.
  [[noreturn]] void Fail(std::string_view where, std::string_view message) const {
    detail::Fail(source_, where, message);
  }

  /// The grid's dimensions after a variable's leading ones.
  static void AppendGrid(FileVariable& var, const Grid& grid) {
    const auto axes = detail::AxesOf(grid.GetDomain());
    var.dimensions.insert(var.dimensions.end(), axes.begin(), axes.end());
    var.shape.insert(var.shape.end(), grid.Shape().begin(), grid.Shape().end());
  }

  /// Whether the file holds a mapping, as the geometry group or a
  /// displacement field.
  bool HoldsMapping() const {
    if (geometry_) return true;
    return std::ranges::any_of(parts_, [](const FilePart& p) {
      return p.variables.contains("displacement");
    });
  }

  /// The vocabulary's spec of a name, or the refusal.
  FieldSpec KnownSpec(const std::string& name) const {
    const auto spec = VocabularySpec(name);
    if (!spec) {
      Fail(name, "is not a name of the vocabulary: give its spec, its character, dimensions "
                 "and meaning");
    }
    return *spec;
  }

  /// Refuse a name the format keeps, one the file holds, a vocabulary name
  /// stated otherwise, or a stored shape the format has no axes for.
  void CheckNewName(const std::string& name, const FieldSpec& spec) const {
    if (name.empty() || std::ranges::contains(RESERVED_NAMES, std::string_view{name}) ||
        name.ends_with("_perturbation") || name.ends_with("_poles")) {
      Fail(name, "is a name the flat model file keeps for itself");
    }
    for (const auto& part : parts_) {
      if (part.variables.contains(name)) Fail(name, "is a variable of the file already");
    }
    if (surfaces_ && surfaces_->variables.contains(name)) {
      Fail(name, "is a variable of the file already");
    }
    if (const auto known = VocabularySpec(name)) {
      if (known->character != spec.character ||
          (spec.dimensions && spec.dimensions != known->dimensions)) {
        Fail(name, "is a vocabulary name, stated with another character or other dimensions "
                   "than the vocabulary's");
      }
    }
    static_cast<void>(AxesOf(spec.character, name));
  }

  /// The names of the trailing axes of a character's stored shape.
  std::vector<std::string> AxesOf(const Character& character, std::string_view where) const {
    const auto shape = character.StoredShape();
    if (shape.empty()) return {};
    if (shape == std::vector<std::size_t>{3}) return {"component"};
    if (shape == std::vector<std::size_t>{6}) return {"voigt"};
    if (shape == std::vector<std::size_t>{3, 3}) return {"component_row", "component_column"};
    if (shape == std::vector<std::size_t>{6, 6}) return {"voigt_row", "voigt_column"};
    Fail(where, std::format("is of rank {}, and the model file holds no field of that stored "
                            "shape",
                            character.Rank()));
  }

  /// The layers of `layers` in each part, by the part's place, every layer
  /// in one, or the refusal.
  std::map<std::size_t, std::vector<int>> ByPart(
      const std::string& name, const std::map<int, LayerSource>& layers) const {
    if (layers.empty()) Fail(name, "is held on no layer");
    auto out = std::map<std::size_t, std::vector<int>>{};
    for (const auto& [layer, source] : layers) {
      auto found = false;
      for (auto index = std::size_t{0}; index < parts_.size(); ++index) {
        if (std::ranges::contains(parts_[index].layers, layer)) {
          out[index].push_back(layer);
          found = true;
        }
      }
      if (!found) Fail(name, std::format("is given on layer {}, which no part holds", layer));
    }
    return out;
  }

  /// The variable `name` of a part on its layers `holders`: the values from
  /// the sources, NaN on the part's other layers, and the attributes the
  /// format asks for.
  FileVariable Variable(const FilePart& part, const std::string& name, const FieldSpec& spec,
                        const Character& character, const std::vector<int>& holders,
                        const std::map<int, LayerSource>& sources, const Rules& rules,
                        bool lateral) const {
    const auto where = std::format("part_{}/{}", part.index, name);
    const auto& grid = *part.grid;
    if (lateral) CheckRule(grid, rules.lateral, where);
    if (IsLogRule(rules.radial) && (lateral || character.Rank() != 0)) {
      Fail(where, std::format("states {}, a rule in the logarithm, which a variable of rank 0 "
                              "of depth alone states, and {}",
                              RuleName(rules.radial),
                              lateral ? "this one varies laterally" : "this one has components"));
    }
    const auto n = character.StoredSize();
    const auto axes = AxesOf(character, where);
    const auto stored = character.StoredShape();
    const auto nnode = part.depth.size();
    auto var = FileVariable{.name = name};
    var.dimensions = {"node"};
    var.shape = {nnode};
    if (lateral) AppendGrid(var, grid);
    var.dimensions.insert(var.dimensions.end(), axes.begin(), axes.end());
    var.shape.insert(var.shape.end(), stored.begin(), stored.end());
    const auto per_node = (lateral ? grid.Size() : 1) * n;
    var.values.assign(nnode * per_node, std::numeric_limits<double>::quiet_NaN());
    auto radial_layers = std::vector<int>{};
    for (const auto layer : holders) {
      const auto& source = sources.at(layer);
      if (lateral && source.IsRadial()) radial_layers.push_back(layer);
      const auto [e0, e1] = part.Elements(layer);
      const auto n0 = e0 * part.ngll;
      const auto sampled = source.Sample(PointsOf(layer), character, lateral, source_,
                                         std::format("{} on layer {}", where, layer));
      if (IsLogRule(rules.radial)) {
        for (auto k = std::size_t{0}; k < sampled.size(); ++k) {
          if (!(sampled[k] > 0.0)) {
            Fail(where, std::format("is {} at depth = {} on layer {}; a variable read by {}, "
                                    "its logarithm interpolated, is positive at every node of "
                                    "a layer holding it",
                                    sampled[k], part.depth[n0 + k], layer,
                                    RuleName(rules.radial)));
          }
        }
      }
      std::ranges::copy(sampled, var.values.begin() + static_cast<std::ptrdiff_t>(n0 * per_node));
    }
    auto& a = var.attributes;
    a.emplace("long_name", spec.meaning.empty() ? name : spec.meaning);
    a.emplace("units", units::ScaledUnit(spec.dimensions, scales_));
    a.emplace("rank", detail::Integer(character.Rank()));
    a.emplace("weight", detail::Integer(character.Weight()));
    a.emplace("voigt", detail::Integer(character.Voigt() ? 1 : 0));
    a.emplace("layers", detail::Integers(holders));
    a.emplace("radial_interpolation", std::string{RuleName(rules.radial)});
    if (spec.dimensions) a.emplace("dimensions", detail::Exponents(*spec.dimensions));
    if (lateral) a.emplace("lateral_interpolation", std::string{RuleName(rules.lateral)});
    a.emplace("coordinates", std::string{"depth"});
    if (!radial_layers.empty()) a.emplace("radial_layers", detail::Integers(radial_layers));
    return var;
  }

  /// Variables made for parts, put into them once all are made.
  void Commit(std::vector<std::pair<std::size_t, FileVariable>> made) {
    for (auto& [index, var] : made) {
      auto name = var.name;
      parts_[index].variables.insert_or_assign(std::move(name), std::move(var));
    }
  }

  /// Refuse an interface the skeleton lacks.
  void CheckInterface(int interface, std::string_view where) const {
    if (interface < 0 || interface >= static_cast<int>(interface_depths_.size())) {
      Fail(where, std::format("is given on interface {}, which the skeleton lacks", interface));
    }
  }

  /// A variable of the surfaces group laid over the rows `to` from the
  /// rows `from`, by interface, NaN on the rows it did not have.
  static FileVariable Rowed(const FileVariable& var, const std::vector<int>& from,
                            const std::vector<int>& to) {
    auto out = var;
    const auto per_row = var.values.size() / var.shape.front();
    out.shape.front() = to.size();
    out.values.assign(to.size() * per_row, std::numeric_limits<double>::quiet_NaN());
    for (auto i = std::size_t{0}; i < from.size(); ++i) {
      const auto j = static_cast<std::size_t>(std::ranges::find(to, from[i]) - to.begin());
      std::copy_n(var.values.begin() + static_cast<std::ptrdiff_t>(i * per_row), per_row,
                  out.values.begin() + static_cast<std::ptrdiff_t>(j * per_row));
    }
    return out;
  }

  /// Whether a variable of the surfaces group has a value on a row.
  static bool HoldsRow(const FileVariable& var, std::size_t row) {
    const auto per_row = var.values.size() / var.shape.front();
    const auto begin = var.values.begin() + static_cast<std::ptrdiff_t>(row * per_row);
    return std::any_of(begin, begin + static_cast<std::ptrdiff_t>(per_row),
                       [](double v) { return !std::isnan(v); });
  }

  /// Refuse what the reading of the fields would refuse, and what the
  /// Python's `read_model` would: a vocabulary name stated otherwise.
  void CheckReadable(const std::string& source) const {
    for (const auto& part : parts_) {
      for (const auto& [name, var] : part.variables) {
        const auto where = std::format("part_{}/{}", part.index, name);
        if (IsFieldName(name)) {
          if (const auto known = VocabularySpec(name)) {
            const auto dims = detail::DimensionsFrom(var.Numbers("dimensions"));
            if (known->character != CharacterOf(var) || (dims && dims != known->dimensions)) {
              detail::Fail(source, where,
                           "is a vocabulary name, stated with another character or other "
                           "dimensions than the vocabulary's");
            }
          }
        }
      }
    }
    if (surfaces_) {
      for (const auto& [name, var] : surfaces_->variables) {
        const auto where = "surfaces/" + name;
        if (const auto known = VocabularySpec(name)) {
          const auto dims = detail::DimensionsFrom(var.Numbers("dimensions"));
          if (known->character.Rank() != 0 || (dims && dims != known->dimensions)) {
            detail::Fail(source, where,
                         "is a vocabulary name, stated with another character or other "
                         "dimensions than the vocabulary's");
          }
        }
      }
    }
  }

  /// Whether a variable is a field of the model, not a field's
  /// perturbation or the mapping.
  static bool IsFieldName(std::string_view name) {
    return !name.ends_with("_perturbation") && name != "displacement";
  }

  /// Whether a variable has a value on a layer: a field's fill is NaN.
  static bool Holds(const FilePart& part, int layer, const FileVariable& var) {
    const auto [e0, e1] = part.Elements(layer);
    const auto per_node = var.values.size() / var.shape.front();
    const auto begin = e0 * part.ngll * per_node;
    const auto end = e1 * part.ngll * per_node;
    return std::any_of(var.values.begin() + static_cast<std::ptrdiff_t>(begin),
                       var.values.begin() + static_cast<std::ptrdiff_t>(end),
                       [](double v) { return !std::isnan(v); });
  }

  /// The character a variable states; a displacement is a height, a scalar.
  static Character CharacterOf(const FileVariable& var) {
    if (var.name == "displacement") return SCALAR;
    auto integer = [&var](std::string_view key) {
      const auto v = var.Numbers(key);
      if (!v || v->size() != 1) throw Error("", var.name, std::format("{} is not a number", key));
      return static_cast<int>((*v)[0]);
    };
    return Character(integer("rank"), integer("weight"), integer("voigt") != 0);
  }

  /// The displacement field of a layer, a scalar of the stated rules.
  ::planetmodel::cartesian::Field Displacement(int layer) const {
    const auto& part = PartOf(layer);
    const auto& var = part.variables.at("displacement");
    auto out = ::planetmodel::cartesian::Field{};
    out.source_ = source_;
    out.name_ = "displacement";
    out.layer_ = layer;
    out.interval_ = skeleton_->Interval(static_cast<std::size_t>(layer));
    out.character_ = SCALAR;
    out.reference_ = std::make_shared<const ::planetmodel::cartesian::Field::Piece>(
        Piece(part, layer, var, out.character_));
    return out;
  }

  /// One variable on one layer as a piece of a field.
  ::planetmodel::cartesian::Field::Piece Piece(const FilePart& part, int layer,
                                               const FileVariable& var,
                                               const Character& character) const {
    const auto where = std::format("part_{}/{}", part.index, var.name);
    auto out = ::planetmodel::cartesian::Field::Piece{};
    out.character = character;
    out.rule = RadialRuleNamed(*var.Text("radial_interpolation"), source_, where);
    const auto [e0, e1] = part.Elements(layer);
    const auto ngll = part.ngll;
    const auto n0 = e0 * ngll;
    const auto n1 = e1 * ngll;
    const auto ncomp = character.StoredSize();
    auto depths = std::vector<double>(part.depth.begin() + static_cast<std::ptrdiff_t>(n0),
                                      part.depth.begin() + static_cast<std::ptrdiff_t>(n1));
    out.layout = RadialLayout(std::move(depths), ngll, rtol_ * skeleton_->Span());
    auto radial_layers = std::vector<int>{};
    if (const auto listed = var.Numbers("radial_layers")) {
      for (const auto x : *listed) radial_layers.push_back(static_cast<int>(x));
    }
    const auto lateral = var.IsLateral();
    const auto cells = part.grid->Size();
    out.radial = !lateral || std::ranges::contains(radial_layers, layer);
    if (out.radial) {
      // the stored components at every node, a lateral variable's from its
      // first column, all of whose columns are the same
      const auto stride = lateral ? cells * ncomp : ncomp;
      for (auto node = n0; node < n1; ++node) {
        const auto* at = &var.values[node * stride];
        out.nodal.insert(out.nodal.end(), at, at + ncomp);
      }
      if (IsLogRule(out.rule)) {
        // a scalar, positive (as the file was checked): the logarithms
        // over each element's first value
        for (auto e = std::size_t{0}; e < e1 - e0; ++e) {
          const auto first = out.nodal[e * ngll];
          out.scale.push_back(first);
          for (auto k = std::size_t{0}; k < ngll; ++k) {
            out.nodal[e * ngll + k] = std::log(out.nodal[e * ngll + k] / first);
          }
        }
      }
      return out;
    }
    // the local components are the same frame at every node of the grid:
    // each is read as it stands
    const auto& grid = *part.grid;
    const auto rule = LateralRuleNamed(*var.Text("lateral_interpolation"), source_, where);
    auto component = std::vector<double>(cells);
    for (auto node = n0; node < n1; ++node) {
      for (auto c = std::size_t{0}; c < ncomp; ++c) {
        for (auto cell = std::size_t{0}; cell < cells; ++cell) {
          component[cell] = var.values[(node * cells + cell) * ncomp + c];
        }
        out.lateral.emplace_back(grid, component, rule, where);
      }
    }
    return out;
  }

  std::string source_{};
  std::string schema_{};
  std::string written_by_{};
  std::string created_{};
  std::optional<std::string> name_{};
  std::vector<std::string> citation_{};
  std::vector<std::string> behaviours_{};
  ::planetmodel::Scales scales_{};
  double rtol_ = DEFAULT_RTOL;
  std::map<std::string, FileConstant> constants_{};
  netcdf::Attributes meta_{};
  std::optional<::planetmodel::Skeleton> skeleton_{};
  std::optional<Domain> domain_{};
  std::vector<std::optional<std::string>> layer_names_{};
  std::vector<std::optional<std::string>> interface_names_{};
  std::vector<double> interface_depths_{};
  std::vector<FilePart> parts_{};
  std::optional<FileGeometry> geometry_{};
  std::optional<FileSurfaces> surfaces_{};
};

/**
 * @brief Hold a flat model file to the structure the readers check, the
 * Python's `cartesian.modelfile.read_model_file` among them: the schema
 * and tolerance, the interfaces those of the skeleton (every boundary but
 * the base, to 1e-12 of its span), every layer in one part of contiguous
 * layers, each part's elements tiling its layers at their
 * Gauss-Lobatto-Legendre nodes on a grid over the file's domain, every
 * variable running over the nodes (and the part's grid where it varies
 * laterally) with the attributes and trailing axes its character asks for,
 * known rules, a lateral rule the grid of its group can carry (`cubic` as
 * `CheckRule` holds it), a perturbation that is there and of a known kind,
 * the mapping held once, and the fields on interfaces on the skeleton's
 * interfaces, over the group's grid, under names no part holds.
 * @param file The file, read or built.
 * @param source The file's path, as the errors name it.
 * @throws planetmodel::Error naming the place of the first fault.
 */
inline void ValidateModelFile(const ModelFile& file, const std::string& source) {
  auto fail = [&source](std::string_view where, std::string_view message) {
    detail::Fail(source, where, message);
  };
  if (file.schema_ != MODEL_SCHEMA) {
    fail("schema", std::format("is {}, and this reader reads {}", file.schema_, MODEL_SCHEMA));
  }
  if (!(file.rtol_ > 0.0 && file.rtol_ < 1.0)) fail("rtol", "is not a tolerance in (0, 1)");
  const auto& skeleton = *file.skeleton_;
  const auto& domain = *file.domain_;
  const auto& b = skeleton.Boundaries();
  const auto nlayers = skeleton.NLayers();
  const auto axes = detail::AxesOf(domain);
  if (file.layer_names_.size() != nlayers) fail("layer_name", "is not one name per layer");
  if (const auto interfaces = detail::InterfaceDepths(skeleton);
      !detail::HoldsInterfaces(file.interface_depths_, interfaces, skeleton.Span())) {
    fail("interface_depth",
         std::format("does not hold the {} interfaces of the skeleton that layer_bounds make",
                     interfaces.size()));
  }
  if (file.interface_names_.size() != file.interface_depths_.size()) {
    fail("interface_name", "is not one name per interface");
  }

  // every layer in one part, each part's layers contiguous
  auto owner = std::vector<int>(nlayers, -1);
  for (const auto& part : file.parts_) {
    const auto& layers = part.layers;
    if (layers.empty() || layers.back() - layers.front() + 1 != static_cast<int>(layers.size()) ||
        !std::ranges::is_sorted(layers)) {
      fail("layer_part", std::format("the layers of part {} are not contiguous", part.index));
    }
    for (const auto layer : layers) {
      if (layer < 0 || layer >= static_cast<int>(nlayers) ||
          owner[static_cast<std::size_t>(layer)] >= 0) {
        fail("layer_part", std::format("part {} holds layer {}, which is not a layer of its "
                                       "own",
                                       part.index, layer));
      }
      owner[static_cast<std::size_t>(layer)] = part.index;
    }
  }
  for (auto layer = std::size_t{0}; layer < nlayers; ++layer) {
    if (owner[layer] < 0) fail("layer_part", std::format("no part holds layer {}", layer));
  }

  // a variable's leading dimensions and shape where it runs over a grid
  auto on_grid = [&](const FileVariable& var, std::string_view lead, std::size_t length,
                     const Grid& grid) {
    auto dims = std::vector<std::string>{std::string{lead}};
    dims.insert(dims.end(), axes.begin(), axes.end());
    auto shape = std::vector<std::size_t>{length};
    shape.insert(shape.end(), grid.Shape().begin(), grid.Shape().end());
    return var.dimensions.size() >= dims.size() && var.shape.size() >= shape.size() &&
           std::equal(dims.begin(), dims.end(), var.dimensions.begin()) &&
           std::equal(shape.begin(), shape.end(), var.shape.begin());
  };

  // the parts
  for (const auto& part : file.parts_) {
    const auto where = std::format("part_{}", part.index);
    const auto& grid = *part.grid;
    if (grid.GetDomain() != domain) fail(where, "its grid does not lie over the file's domain");
    const auto nelem = part.element_layer.size();
    const auto nnode = part.depth.size();
    if (part.ngll < 2 || nelem * part.ngll != nnode) {
      fail(where, std::format("{} nodes are not {} elements of ngll = {}", nnode, nelem,
                              part.ngll));
    }
    if (part.element_bounds.size() != 2 * nelem) {
      fail(where, "element_bounds are not two per element");
    }
    auto held = part.element_layer;
    if (!std::ranges::is_sorted(held)) fail(where, "its elements do not lie in its layers");
    held.erase(std::unique(held.begin(), held.end()), held.end());
    if (held != part.layers) fail(where, "its elements do not lie in its layers");
    const auto& eb = part.element_bounds;
    // the part's ends and every element's ends meet to 1e-12 of its span
    const auto top = b[static_cast<std::size_t>(part.layers.front())];
    const auto bottom = b[static_cast<std::size_t>(part.layers.back()) + 1];
    const auto tile = 1e-12 * (bottom - top);
    auto tiles = std::abs(eb.front() - top) <= tile && std::abs(eb.back() - bottom) <= tile;
    for (auto e = std::size_t{1}; e < nelem; ++e) {
      tiles = tiles && std::abs(eb[2 * e] - eb[2 * e - 1]) <= tile;
    }
    if (!tiles) fail(where, "element_bounds do not tile its layers");
    // each element within the layer `element_layer` gives it, to that tolerance
    for (auto e = std::size_t{0}; e < nelem; ++e) {
      const auto layer = static_cast<std::size_t>(part.element_layer[e]);
      if (!(eb[2 * e] >= b[layer] - tile && eb[2 * e + 1] <= b[layer + 1] + tile)) {
        fail(where + "/element_layer",
             std::format("gives element {} to layer {}, which its bounds do not lie in", e,
                         layer));
      }
    }
    for (auto e = std::size_t{0}; e < nelem; ++e) {
      const auto nodes = GllNodes(eb[2 * e], eb[2 * e + 1], part.ngll);
      for (auto k = std::size_t{0}; k < part.ngll; ++k) {
        const auto stored = part.depth[e * part.ngll + k];
        if (std::abs(stored - nodes[k]) > 1e-12 * std::abs(nodes[k])) {
          fail(where + "/depth", "is not the Gauss-Lobatto-Legendre nodes of its elements");
        }
      }
    }
    for (const auto& [vname, var] : part.variables) {
      const auto vwhere = std::format("{}/{}", where, vname);
      if (var.dimensions.empty() || var.dimensions.front() != "node" || var.shape.empty() ||
          var.shape.front() != nnode) {
        fail(vwhere, "does not run over the nodes");
      }
      auto size = std::size_t{1};
      for (const auto extent : var.shape) size *= extent;
      if (var.dimensions.size() != var.shape.size() || var.values.size() != size) {
        fail(vwhere, "has values that do not fill its shape");
      }
      const auto lateral = var.IsLateral();
      if (lateral && !on_grid(var, "node", nnode, grid)) fail(vwhere, "is not on the part's grid");
      const auto need = vname == "displacement"
                            ? std::vector<std::string>{"radial_interpolation"}
                            : std::vector<std::string>{"radial_interpolation", "rank", "weight",
                                                       "voigt"};
      for (const auto& key : need) {
        if (!var.attributes.contains(key)) fail(vwhere, std::format("lacks the attribute {}", key));
      }
      auto stored = std::vector<std::size_t>{};
      try {
        stored = ModelFile::CharacterOf(var).StoredShape();
      } catch (const Error& e) {
        fail(vwhere, std::format("states no character: {}", e.what()));
      }
      const auto lead = lateral ? 1 + axes.size() : std::size_t{1};
      const auto tail = std::vector<std::size_t>(
          var.shape.begin() + static_cast<std::ptrdiff_t>(lead), var.shape.end());
      if (tail != stored) fail(vwhere, "has trailing axes its character does not store");
      const auto radial_rule = var.Text("radial_interpolation");
      if (!radial_rule || !std::ranges::contains(RADIAL_RULES, std::string_view{*radial_rule})) {
        fail(vwhere, "states an unknown radial rule");
      }
      if (IsLogRule(RadialRuleNamed(*radial_rule))) {
        if (lateral || !stored.empty() || vname == "displacement") {
          fail(vwhere, std::format("states {}, a rule in the logarithm, which a variable of rank "
                                   "0 of depth alone states",
                                   *radial_rule));
        }
        for (const auto v : var.values) {
          if (!std::isnan(v) && !(std::isfinite(v) && v > 0.0)) {
            fail(vwhere, std::format("states {}, a rule in the logarithm, and holds a value that "
                                     "is not positive",
                                     *radial_rule));
          }
        }
      }
      if (var.attributes.contains("radial_layers")) {
        auto holders = std::vector<int>{};
        if (const auto l = var.Numbers("layers")) {
          for (const auto x : *l) holders.push_back(static_cast<int>(x));
        } else {
          holders = part.layers;
        }
        for (const auto x : var.Numbers("radial_layers").value_or(std::vector<double>{-1.0})) {
          if (!lateral || !std::ranges::contains(holders, static_cast<int>(x))) {
            fail(vwhere, "names radial_layers which must be layers a lateral variable holds");
          }
        }
      }
      if (lateral) {
        const auto rule = var.Text("lateral_interpolation");
        if (!rule || !std::ranges::contains(LATERAL_RULES, std::string_view{*rule})) {
          fail(vwhere, "states no known lateral rule");
        }
        detail::CheckCarried(grid, *rule, source, vwhere);
      }
      if (var.attributes.contains("perturbation")) {
        const auto other = var.Text("perturbation");
        if (!other || !part.variables.contains(*other)) {
          fail(vwhere, std::format("names its perturbation {}, which the part lacks",
                                   other.value_or("by a number")));
        }
        const auto kind = var.Text("perturbation_kind");
        if (!kind || !std::ranges::contains(PERTURBATION_KINDS, std::string_view{*kind})) {
          fail(vwhere, "states no known perturbation_kind");
        }
      }
    }
  }

  // the mapping
  auto holding = std::size_t{0};
  for (const auto& part : file.parts_) holding += part.variables.contains("displacement");
  if (holding != 0 && holding != file.parts_.size()) {
    fail("displacement", "is held in some parts only; a mapping held as a field is held in "
                         "every part");
  }
  if (file.geometry_ && holding != 0) {
    fail("geometry", "the mapping is held twice, as the geometry group and as a displacement "
                     "field");
  }
  if (file.geometry_) {
    const auto& geometry = *file.geometry_;
    const auto where = std::string{"geometry/boundary_displacement"};
    if (!std::ranges::contains(LATERAL_RULES, std::string_view{geometry.lateral})) {
      fail(where, "states no known lateral rule");
    }
    if (geometry.between != "linear") fail(where, "states an unknown rule between surfaces");
    detail::CheckCarried(*geometry.grid, geometry.lateral, source, where);
    const auto& depths = geometry.depths;
    for (auto k = std::size_t{1}; k < depths.size(); ++k) {
      if (!(depths[k] > depths[k - 1])) {
        fail("geometry/boundary_depth", "does not increase strictly");
      }
    }
    if (depths.size() < 2) fail("geometry/boundary_depth", "is fewer than two surfaces");
    if (!(depths.front() >= 0.0)) {
      fail("geometry/boundary_depth", "starts above the free surface, at a negative depth");
    }
    if (geometry.grid->GetDomain() != domain) {
      fail("geometry", "its grid does not lie over the file's domain");
    }
    if (geometry.values.size() != depths.size() * geometry.grid->Size()) {
      fail(where, "has values that do not fill its shape");
    }
  }

  // the fields on interfaces
  if (file.surfaces_) {
    const auto& surfaces = *file.surfaces_;
    const auto& faces = surfaces.interfaces;
    const auto& grid = *surfaces.grid;
    if (grid.GetDomain() != domain) {
      fail("surfaces", "its grid does not lie over the file's domain");
    }
    const auto ninterface = static_cast<int>(file.interface_depths_.size());
    for (auto i = std::size_t{0}; i < faces.size(); ++i) {
      if (faces[i] < 0 || faces[i] >= ninterface || (i > 0 && faces[i] <= faces[i - 1])) {
        fail("surfaces/surface_interface", "is not interfaces of the file, increasing");
      }
    }
    for (const auto& [vname, var] : surfaces.variables) {
      const auto vwhere = "surfaces/" + vname;
      if (var.dimensions.size() != 1 + axes.size() ||
          !on_grid(var, "surface", faces.size(), grid) ||
          var.values.size() != faces.size() * grid.Size()) {
        fail(vwhere, "does not run over the surfaces and the grid");
      }
      for (const auto& part : file.parts_) {
        if (part.variables.contains(vname)) {
          fail(vwhere, "is a layer field's name too; a name is a layer field's or a surface "
                       "field's, never both");
        }
      }
      const auto rule = var.Text("lateral_interpolation");
      if (!rule || !std::ranges::contains(LATERAL_RULES, std::string_view{*rule})) {
        fail(vwhere, "states no known lateral rule");
      }
      detail::CheckCarried(grid, *rule, source, vwhere);
      if (var.attributes.contains("interfaces")) {
        for (const auto x : var.Numbers("interfaces").value_or(std::vector<double>{-1.0})) {
          if (!std::ranges::contains(faces, static_cast<int>(x))) {
            fail(vwhere, "names interfaces that are not surfaces of the group");
          }
        }
      }
    }
  }
}

/**
 * @brief A flat model file, read whole and checked as the Python's
 * `cartesian.modelfile.read_model_file` checks it, its schema first: a
 * spherical model file is refused by its schema, naming its reader.
 * @param path The file.
 * @return Its numbers and metadata, as stored.
 * @throws planetmodel::Error naming the file and the place of the first
 *         fault, or netCDF-C's message where the file cannot be read.
 */
[[nodiscard]] inline ModelFile ReadModelFile(const std::filesystem::path& path) {
  const auto file = netcdf::File(path);
  const auto root = file.Id();
  const auto& source = file.Path();
  auto fail = [&source](std::string_view where, std::string_view message) {
    detail::Fail(source, where, message);
  };
  auto out = ModelFile{};
  out.source_ = source;
  const auto attrs = netcdf::ReadAttributes(root, netcdf::GLOBAL, source);
  auto text = [&attrs](std::string_view key) { return detail::TextOf(attrs, key); };
  auto number = [&attrs](std::string_view key) { return detail::NumberOf(attrs, key); };

  // the schema first
  const auto schema = text("schema");
  if (!schema) fail("schema", "is missing: not a planetmodel model file");
  out.schema_ = *schema;
  if (out.schema_ != MODEL_SCHEMA) {
    const auto spherical = out.schema_ == ::planetmodel::MODEL_SCHEMA ||
                           out.schema_ == MODEL_SCHEMA_WITH_LOG_RULES;
    fail("schema", std::format("is {}, and this reader reads {}{}", out.schema_, MODEL_SCHEMA,
                               spherical ? "; a file of that schema is a spherical model's, "
                                           "read by planetmodel::ReadModelFile "
                                           "(planetmodel/modelfile.hpp)"
                                         : ""));
  }

  // the root
  out.name_ = text("name");
  out.written_by_ = text("source").value_or("");
  out.created_ = text("created").value_or("");
  out.citation_ = detail::CitationOf(attrs);
  out.behaviours_ = detail::BehavioursOf(attrs, source);
  out.scales_ = detail::ScalesOf(attrs, source);
  if (attrs.contains("rtol")) {
    const auto rtol = number("rtol");
    if (!rtol) fail("rtol", "is not a tolerance in (0, 1)");
    out.rtol_ = *rtol;
  }
  for (const auto& [key, value] : attrs) {
    if (key.starts_with("meta_")) out.meta_.emplace(key.substr(5), value);
    if (!key.starts_with("constant_") || key.ends_with("_dimensions") ||
        key.ends_with("_meaning")) {
      continue;
    }
    const auto cname = key.substr(9);
    const auto v = number(key);
    if (!v) fail(key, "is not a number");
    auto constant = FileConstant{.value = *v};
    if (attrs.contains(key + "_dimensions")) {
      constant.dimensions = detail::DimensionsFrom(detail::NumbersOf(attrs, key + "_dimensions"));
      if (!constant.dimensions) fail(key + "_dimensions", "is not four exponents");
    }
    constant.meaning = text(key + "_meaning").value_or("");
    out.constants_.emplace(cname, constant);
  }
  for (const auto& [key, want] : {std::pair{"coordinate_system", COORDINATE_SYSTEM},
                                  std::pair{"local_frame", LOCAL_FRAME_AXES}}) {
    if (text(key) != want) {
      fail(key, std::format("is {}, and a flat model file states {}",
                            text(key).value_or("missing or a number"), want));
    }
  }
  detail::RefuseSpherical(root, attrs, source, "");
  out.domain_ = detail::ReadDomain(attrs, source);
  const auto& domain = *out.domain_;
  const auto axes = detail::AxesOf(domain);

  // the skeleton
  for (const auto& [name, dims] : std::vector<std::pair<std::string, std::vector<std::string>>>{
           {"layer_bounds", {"layer", "bound"}},
           {"layer_name", {"layer"}},
           {"interface_depth", {"interface"}},
           {"interface_name", {"interface"}},
           {"layer_part", {"layer"}}}) {
    if (!netcdf::HasVariable(root, name)) fail(name, "is not a variable at the root");
    if (netcdf::DimensionNames(root, name, source) != dims) {
      fail(name, "has the wrong dimensions");
    }
  }
  const auto bounds = netcdf::ReadDoubles(root, "layer_bounds", source).values;
  const auto nlayers = bounds.size() / 2;
  auto boundaries = std::vector<double>{};
  for (auto i = std::size_t{0}; i < nlayers; ++i) {
    if (!(bounds[2 * i + 1] > bounds[2 * i]) || (i > 0 && bounds[2 * i] != bounds[2 * i - 1])) {
      fail("layer_bounds", "do not make a skeleton of contiguous layers");
    }
    boundaries.push_back(bounds[2 * i]);
  }
  if (nlayers == 0) fail("layer_bounds", "do not make a skeleton of contiguous layers");
  boundaries.push_back(bounds.back());
  try {
    out.skeleton_ = Skeleton(boundaries);
  } catch (const Error& e) {
    fail("layer_bounds", std::format("do not make a skeleton: {}", e.what()));
  }
  for (const auto& n : netcdf::ReadStrings(root, "layer_name", source)) {
    out.layer_names_.push_back(n.empty() ? std::nullopt : std::optional<std::string>{n});
  }
  for (const auto& n : netcdf::ReadStrings(root, "interface_name", source)) {
    out.interface_names_.push_back(n.empty() ? std::nullopt : std::optional<std::string>{n});
  }
  out.interface_depths_ = netcdf::ReadDoubles(root, "interface_depth", source).values;
  const auto layer_part_values = netcdf::ReadDoubles(root, "layer_part", source).values;
  auto layer_part = std::vector<int>{};
  for (const auto x : layer_part_values) layer_part.push_back(detail::IndexOf(x));
  const auto groups = netcdf::GroupNames(root, source);
  auto part_numbers = layer_part;
  std::ranges::sort(part_numbers);
  part_numbers.erase(std::unique(part_numbers.begin(), part_numbers.end()), part_numbers.end());
  for (const auto k : part_numbers) {
    const auto gname = std::format("part_{}", k);
    if (!std::ranges::contains(groups, gname)) {
      fail("layer_part", std::format("names {}, which the file lacks", gname));
    }
    auto layers = std::vector<int>{};
    for (auto i = std::size_t{0}; i < nlayers; ++i) {
      if (layer_part[i] == k) layers.push_back(static_cast<int>(i));
    }
    out.parts_.push_back(FilePart{.index = k, .layers = layers});
  }

  auto own = [&axes](const std::string& name, std::initializer_list<const char*> names) {
    return std::ranges::contains(axes, name) ||
           std::ranges::any_of(names, [&](const char* n) { return name == n; });
  };

  // the parts, as stored
  for (auto& part : out.parts_) {
    const auto where = std::format("part_{}", part.index);
    const auto g = netcdf::Group(root, where, source);
    const auto dims = netcdf::GroupDimensionNames(g, source);
    for (const auto* dim : {"node", "element", "bound"}) {
      if (!std::ranges::contains(dims, std::string{dim})) {
        fail(where, std::format("has no dimension {}", dim));
      }
    }
    for (const auto* name : {"depth", "element_layer", "element_bounds"}) {
      if (!netcdf::HasVariable(g, name)) fail(where, std::format("has no variable {}", name));
    }
    const auto gattrs = netcdf::ReadAttributes(g, netcdf::GLOBAL, source);
    if (!gattrs.contains("ngll")) fail(where, "lacks the attribute ngll");
    detail::RefuseSpherical(g, gattrs, source, where);
    part.grid = detail::GridOf(g, domain, source, where);
    const auto ngll = detail::NgllOf(gattrs);
    if (!ngll) fail(where + ".ngll", "is not a whole number of nodes, two or more");
    part.ngll = *ngll;
    for (const auto x : netcdf::ReadDoubles(g, "element_layer", source).values) {
      part.element_layer.push_back(detail::IndexOf(x));
    }
    part.element_bounds = netcdf::ReadDoubles(g, "element_bounds", source).values;
    part.depth = netcdf::ReadDoubles(g, "depth", source).values;
    for (const auto& vname : netcdf::VariableNames(g, source)) {
      if (own(vname, {"depth", "element_layer", "element_bounds"})) continue;
      part.variables.emplace(vname, detail::ReadVariable(g, vname, source));
    }
  }

  // the geometry group, as stored
  if (std::ranges::contains(groups, std::string{"geometry"})) {
    const auto g = netcdf::Group(root, "geometry", source);
    const auto gattrs = netcdf::ReadAttributes(g, netcdf::GLOBAL, source);
    detail::RefuseSpherical(g, gattrs, source, "geometry");
    for (const auto* name : {"boundary_depth", "boundary_displacement"}) {
      if (!netcdf::HasVariable(g, name)) fail("geometry", std::format("has no variable {}", name));
    }
    auto geometry = FileGeometry{};
    geometry.grid = detail::GridOf(g, domain, source, "geometry");
    auto want = std::vector<std::string>{"boundary"};
    want.insert(want.end(), axes.begin(), axes.end());
    if (netcdf::DimensionNames(g, "boundary_displacement", source) != want) {
      fail("geometry/boundary_displacement", "has the wrong dimensions");
    }
    const auto var = detail::ReadVariable(g, "boundary_displacement", source);
    geometry.depths = netcdf::ReadDoubles(g, "boundary_depth", source).values;
    geometry.values = var.values;
    geometry.lateral = var.Text("lateral_interpolation").value_or("");
    geometry.between = var.Text("between").value_or("");
    if (var.Text("positive") != "up") {
      fail("geometry/boundary_displacement",
           "is a height, positive up, and states otherwise");
    }
    out.geometry_ = std::move(geometry);
  }

  // the surfaces group, as stored
  if (std::ranges::contains(groups, std::string{"surfaces"})) {
    const auto g = netcdf::Group(root, "surfaces", source);
    const auto gattrs = netcdf::ReadAttributes(g, netcdf::GLOBAL, source);
    if (!std::ranges::contains(netcdf::GroupDimensionNames(g, source), std::string{"surface"})) {
      fail("surfaces", "has no dimension surface");
    }
    if (!netcdf::HasVariable(g, "surface_interface")) {
      fail("surfaces", "has no variable surface_interface");
    }
    detail::RefuseSpherical(g, gattrs, source, "surfaces");
    auto surfaces = FileSurfaces{};
    surfaces.grid = detail::GridOf(g, domain, source, "surfaces");
    if (netcdf::DimensionNames(g, "surface_interface", source) !=
        std::vector<std::string>{"surface"}) {
      fail("surfaces/surface_interface", "is not interfaces of the file, increasing");
    }
    for (const auto x : netcdf::ReadDoubles(g, "surface_interface", source).values) {
      if (x != std::round(x)) {
        fail("surfaces/surface_interface", "is not interfaces of the file, increasing");
      }
      surfaces.interfaces.push_back(static_cast<int>(x));
    }
    for (const auto& vname : netcdf::VariableNames(g, source)) {
      if (own(vname, {"surface_interface"})) continue;
      surfaces.variables.emplace(vname, detail::ReadVariable(g, vname, source));
    }
    out.surfaces_ = std::move(surfaces);
  }
  ValidateModelFile(out, source);
  return out;
}

inline void ModelFile::WriteTo(int root, const std::string& source) const {
  namespace nc = netcdf;
  constexpr auto global = nc::GLOBAL;
  auto put = [&source](int group, int var, std::string_view key, const nc::Attribute& value) {
    nc::WriteAttribute(group, var, key, value, source);
  };
  const auto length = units::ScaledUnit(units::LENGTH, scales_);

  // the root
  put(root, global, "schema", schema_);
  put(root, global, "Conventions", std::string{MODEL_CONVENTIONS});
  put(root, global, "title", name_ ? "planetmodel model " + *name_ : "planetmodel model");
  put(root, global, "source", detail::StampedSource(written_by_));
  put(root, global, "created", created_.empty() ? detail::CreatedNow() : created_);
  if (name_) put(root, global, "name", *name_);
  detail::WriteCitation(root, citation_, source);
  detail::WriteBehaviours(root, behaviours_, source);
  detail::WriteScales(root, scales_, source);
  put(root, global, "rtol", std::vector<double>{rtol_});
  for (const auto& [cname, constant] : constants_) {
    put(root, global, "constant_" + cname, std::vector<double>{constant.value});
    if (constant.dimensions) {
      put(root, global, "constant_" + cname + "_dimensions",
          detail::Exponents(*constant.dimensions));
    }
    if (!constant.meaning.empty()) {
      put(root, global, "constant_" + cname + "_meaning", constant.meaning);
    }
  }
  for (const auto& [key, value] : meta_) put(root, global, "meta_" + key, value);
  put(root, global, "coordinate_system", std::string{COORDINATE_SYSTEM});
  put(root, global, "local_frame", std::string{LOCAL_FRAME_AXES});
  detail::WriteDomain(root, *domain_, source);
  const auto nlayers = skeleton_->NLayers();
  nc::Dimension(root, "layer", nlayers, source);
  nc::Dimension(root, "interface", interface_depths_.size(), source);
  nc::Dimension(root, "bound", 2, source);
  const auto& b = skeleton_->Boundaries();
  auto bounds = std::vector<double>{};
  for (auto i = std::size_t{0}; i < nlayers; ++i) {
    bounds.push_back(b[i]);
    bounds.push_back(b[i + 1]);
  }
  auto var = nc::WriteDoubles(root, "layer_bounds", {"layer", "bound"}, bounds, source);
  put(root, var, "long_name",
      std::string{"depths of the layer's top and bottom below the free surface"});
  put(root, var, "units", length);
  auto names = std::vector<std::string>{};
  for (const auto& n : layer_names_) names.push_back(n.value_or(""));
  nc::WriteStrings(root, "layer_name", "layer", names, source);
  var = nc::WriteDoubles(root, "interface_depth", {"interface"}, interface_depths_, source);
  put(root, var, "long_name", std::string{"depth of the interface below the free surface"});
  put(root, var, "units", length);
  names.clear();
  for (const auto& n : interface_names_) names.push_back(n.value_or(""));
  nc::WriteStrings(root, "interface_name", "interface", names, source);
  auto layer_part = std::vector<int>(nlayers);
  for (const auto& part : parts_) {
    for (const auto layer : part.layers) layer_part[static_cast<std::size_t>(layer)] = part.index;
  }
  var = nc::WriteIntegers(root, "layer_part", {"layer"}, layer_part, source);
  put(root, var, "long_name", std::string{"the part, part_<k>, holding the layer"});

  auto write_variables = [&](int g, const std::map<std::string, FileVariable>& variables) {
    for (const auto& [vname, variable] : variables) {
      for (auto i = std::size_t{0}; i < variable.dimensions.size(); ++i) {
        nc::Dimension(g, variable.dimensions[i], variable.shape[i], source);
      }
      const auto v = nc::WriteDoubles(g, vname, variable.dimensions, variable.values, source);
      for (const auto& [key, value] : variable.attributes) {
        if (!key.starts_with('_')) put(g, v, key, value);
      }
    }
  };

  // the parts
  for (const auto& part : parts_) {
    const auto g = nc::CreateGroup(root, std::format("part_{}", part.index), source);
    nc::Dimension(g, "node", part.depth.size(), source);
    nc::Dimension(g, "element", part.element_layer.size(), source);
    nc::Dimension(g, "bound", 2, source);
    put(g, global, "ngll", detail::Integer(static_cast<long long>(part.ngll)));
    put(g, global, "layers", detail::Integers(part.layers));
    var = nc::WriteDoubles(g, "depth", {"node"}, part.depth, source);
    put(g, var, "long_name",
        std::string{"depth of the node below the free surface; an element boundary is two "
                    "nodes"});
    put(g, var, "units", length);
    put(g, var, "standard_name", std::string{"depth"});
    put(g, var, "positive", std::string{"down"});
    var = nc::WriteIntegers(g, "element_layer", {"element"}, part.element_layer, source);
    put(g, var, "long_name", std::string{"skeleton layer of the element"});
    var = nc::WriteDoubles(g, "element_bounds", {"element", "bound"}, part.element_bounds, source);
    put(g, var, "long_name", std::string{"depths of the element's top and bottom"});
    put(g, var, "units", length);
    detail::WriteGrid(g, *part.grid, length, source);
    write_variables(g, part.variables);
  }

  // the geometry group
  if (geometry_) {
    const auto& geometry = *geometry_;
    const auto g = nc::CreateGroup(root, "geometry", source);
    nc::Dimension(g, "boundary", geometry.depths.size(), source);
    var = nc::WriteDoubles(g, "boundary_depth", {"boundary"}, geometry.depths, source);
    put(g, var, "long_name",
        std::string{"reference depth of the surface the displacement belongs to"});
    put(g, var, "units", length);
    detail::WriteGrid(g, *geometry.grid, length, source);
    auto dims = std::vector<std::string>{"boundary"};
    const auto axes = detail::AxesOf(*domain_);
    dims.insert(dims.end(), axes.begin(), axes.end());
    var = nc::WriteDoubles(g, "boundary_displacement", dims, geometry.values, source);
    put(g, var, "long_name",
        std::string{"upward displacement of the surface, its height above its reference "
                    "place"});
    put(g, var, "units", length);
    put(g, var, "positive", std::string{"up"});
    put(g, var, "lateral_interpolation", geometry.lateral);
    put(g, var, "between", geometry.between);
  }

  // the surfaces group
  if (surfaces_ && !surfaces_->variables.empty()) {
    const auto& surfaces = *surfaces_;
    const auto g = nc::CreateGroup(root, "surfaces", source);
    nc::Dimension(g, "surface", surfaces.interfaces.size(), source);
    var = nc::WriteIntegers(g, "surface_interface", {"surface"}, surfaces.interfaces, source);
    put(g, var, "long_name", std::string{"the interface the surface is, by its index at the root"});
    detail::WriteGrid(g, *surfaces.grid, length, source);
    write_variables(g, surfaces.variables);
  }
}

/**
 * @brief Write a flat model file, after holding it to the readers' checks.
 *
 * The file is checked as `ValidateModelFile` checks it, and as the reading
 * of its fields and the Python's `read_model` would (a vocabulary name
 * stated otherwise), before anything is written; it is written under a
 * name of its own beside `path` and put in place when whole, so that a
 * failed write leaves no file and an existing file at `path` stands until
 * it is replaced.  The file has the permissions a plain open for writing
 * would give it: an existing file's, kept, or the umask's.
 *
 * @param path The file to write; one there is replaced.
 * @param file The flat model file, read or built.
 * @throws planetmodel::Error naming the place of the first fault, or what
 *         kept the file from being written.
 */
inline void WriteModelFile(const std::filesystem::path& path, const ModelFile& file) {
  const auto source = path.string();
  ValidateModelFile(file, source);
  file.CheckReadable(source);
  const auto partial = detail::PartialName(path);
  try {
    auto nc = netcdf::File::Create(partial);
    file.WriteTo(nc.Id(), source);
    nc.Close();
    detail::PutInPlace(partial, path, source);
  } catch (...) {
    auto error = std::error_code{};
    std::filesystem::remove(partial, error);
    throw;
  }
}

}  // namespace planetmodel::cartesian

#endif  // PLANETMODEL_CARTESIAN_MODELFILE_HPP
