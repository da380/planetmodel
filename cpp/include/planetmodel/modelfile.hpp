#ifndef PLANETMODEL_MODELFILE_HPP
#define PLANETMODEL_MODELFILE_HPP

/**
 * @file
 * @brief The model file, as `planetmodel.modelfile` defines it and the
 * format page describes it: a model in one netCDF-4 file, on its
 * reference skeleton, read and written.
 *
 * `ReadModelFile(path)` reads a file whole and holds it to the structure
 * the Python's `read_model_file` checks, refusing the first fault with an
 * `Error` naming the file and the place; the numbers are kept as stored,
 * in the file's scales.  `ModelFile::FieldOn(layer, name)` builds a field
 * of one variable on one layer, which evaluates anywhere on its layer by
 * the rules the file states: a radial variable, or a layer a lateral
 * variable names in `radial_layers`, as a function of the radius; a
 * lateral one by its radial and lateral rules, its components of rank one
 * and above interpolated in Cartesian components; a reference with a
 * perturbation by the perturbation's kind.  `ModelFile::Mapping()` gives
 * the mapping, from the boundary displacements of the `geometry` group or
 * from the `displacement` field.  `ModelFile::SurfaceFieldOn(interface,
 * name)` reads a field of the `surfaces` group, a number at each point of
 * one of the skeleton's interfaces (a shape kernel, a boundary's shift), as
 * a function of the angles by its stated lateral rule.
 *
 * A field holds the interpolants it was built with and nothing mutable,
 * so it is cheap to copy and safe to evaluate from many threads; the
 * `ModelFile` holds only numbers.  A code reads the file, builds the
 * fields it wants, samples them onto its own discretisation and lets them
 * go:
 *
 * @code
 * const auto file = planetmodel::ReadModelFile("earth.nc");
 * const auto vs = file.FieldOn(3, "vsv").AsScalar();
 * const auto value = vs.Value(r, theta, phi);
 * const auto u = file.Mapping().Displacement({x, y, z});
 * @endcode
 *
 * **Writing.**  A code that holds fields of its own makes a `ModelFile`
 * of its skeleton and scales, cuts the layers into parts, each a run of
 * layers laid out in elements of GLL nodes on an angular grid of its own,
 * and adds its fields, each layer's from a `LayerSource`: a function of the
 * radius, or of the point in a frame the code names, or the values at the
 * points `PointsOf` gives.  The library evaluates, stores the components
 * in the format's frames (spherical at the grid's nodes, Cartesian at the
 * poles a grid lacks) with every attribute the format asks for, and
 * `WriteModelFile` writes the file after holding it to the reader's
 * checks, under a name of its own until it is whole.  Fields on interfaces
 * go on a grid set once by `SetSurfaceGrid`, each added by
 * `AddSurfaceField` from a source per interface, evaluated at the
 * interface's radius:
 *
 * @code
 * auto file = planetmodel::ModelFile(planetmodel::Skeleton({3480e3, 5701e3, 6371e3}));
 * const auto& skeleton = file.GetSkeleton();
 * file.AddPart(0, 1, planetmodel::ElementEdges(skeleton, 0, 1, 100e3), 5,
 *              planetmodel::GaussLegendre(64));
 * file.AddField("viscosity", {{0, planetmodel::LayerSource::Lateral(lower)},
 *                             {1, planetmodel::LayerSource::Lateral(upper)}});
 * file.SetSurfaceGrid(planetmodel::GaussLegendre(64));
 * file.AddSurfaceField("K_moho", spec, {{1, planetmodel::LayerSource::LateralValues(
 *                                              values, planetmodel::Frame::Spherical)}});
 * planetmodel::WriteModelFile("viscosity.nc", file);
 * @endcode
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "planetmodel/character.hpp"
#include "planetmodel/error.hpp"
#include "planetmodel/frames.hpp"
#include "planetmodel/gll.hpp"
#include "planetmodel/grid.hpp"
#include "planetmodel/netcdf.hpp"
#include "planetmodel/rules.hpp"
#include "planetmodel/skeleton.hpp"
#include "planetmodel/units.hpp"
#include "planetmodel/version.hpp"
#include "planetmodel/vocabulary.hpp"

namespace planetmodel {

/** @brief The schema this library reads and writes; bumped only for an incompatible change. */
inline constexpr std::string_view MODEL_SCHEMA = "planetmodel.model/1";

/** @brief The CF conventions the file follows; groups are admitted from 1.8. */
inline constexpr std::string_view MODEL_CONVENTIONS = "CF-1.8";

/** @brief How a perturbation combines with its reference. */
inline constexpr auto PERTURBATION_KINDS =
    std::array<std::string_view, 3>{"absolute", "relative", "log"};

/** @brief Names the format gives its own variables, which no field may take. */
inline constexpr auto RESERVED_NAMES = std::array<std::string_view, 14>{
    "radius",         "element_layer",      "element_bounds", "latitude",
    "longitude",      "colatitude_weights", "displacement",   "displacement_poles",
    "layer_bounds",   "layer_name",         "interface_radius", "interface_name",
    "layer_part",     "surface_interface"};

/** @brief The frame components are given in. */
enum class Frame {
  Spherical, /**< The local frame (e_r, e_theta, e_phi) at the point. */
  Cartesian  /**< The Cartesian frame of the coordinates. */
};

namespace detail {

/// The refusal of a file, naming the place of the fault.
[[noreturn]] inline void Fail(const std::string& source, std::string_view where,
                              std::string_view message) {
  throw Error(source, where, message);
}

/// An attribute's text, or empty where it is absent or a number.
inline std::optional<std::string> TextOf(const netcdf::Attributes& attrs, std::string_view key) {
  const auto it = attrs.find(std::string{key});
  if (it == attrs.end() || !std::holds_alternative<std::string>(it->second)) return std::nullopt;
  return std::get<std::string>(it->second);
}

/// An attribute's numbers, or empty where it is absent or text.
inline std::optional<std::vector<double>> NumbersOf(const netcdf::Attributes& attrs,
                                                   std::string_view key) {
  const auto it = attrs.find(std::string{key});
  if (it == attrs.end()) return std::nullopt;
  return netcdf::AsNumbers(it->second);
}

/// An attribute's one number, or empty where it is absent, text or many.
inline std::optional<double> NumberOf(const netcdf::Attributes& attrs, std::string_view key) {
  const auto v = NumbersOf(attrs, key);
  return v && v->size() == 1 ? std::optional<double>{(*v)[0]} : std::nullopt;
}

/// Four exponents as dimensions, in the order `units.BASE` gives them.
inline std::optional<Dimensions> DimensionsFrom(const std::optional<std::vector<double>>& d) {
  if (!d || d->size() != 4) return std::nullopt;
  return Dimensions{.mass = static_cast<int>((*d)[0]),
                    .length = static_cast<int>((*d)[1]),
                    .time = static_cast<int>((*d)[2]),
                    .temperature = static_cast<int>((*d)[3])};
}

/// Dimensions as their four exponents, an integer attribute.
inline netcdf::Attribute Exponents(const Dimensions& d) {
  return std::vector<long long>{d.mass, d.length, d.time, d.temperature};
}

/// Integers as an integer attribute.
inline netcdf::Attribute Integers(std::span<const int> values) {
  return std::vector<long long>(values.begin(), values.end());
}

/// One integer as an integer attribute.
inline netcdf::Attribute Integer(long long value) { return std::vector<long long>{value}; }

/// The writing time in UTC, ISO 8601: the `created` stamp of the writers.
inline std::string CreatedNow() {
  const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
  return std::format("{:%FT%T}+00:00", now);
}

}  // namespace detail

/** @brief One variable of a part, its values as stored. */
struct FileVariable {
  std::string name{};                    /**< The variable's name. */
  std::vector<std::string> dimensions{}; /**< Its dimensions' names. */
  std::vector<std::size_t> shape{};      /**< Its dimensions' lengths. */
  std::vector<double> values{};          /**< Its values, C order, NaN for its fill. */
  netcdf::Attributes attributes{};       /**< Its attributes. */

  /**
   * @brief A text attribute.
   * @param key The attribute.
   * @return Its text, or empty where there is no such text attribute.
   */
  [[nodiscard]] std::optional<std::string> Text(std::string_view key) const {
    return detail::TextOf(attributes, key);
  }

  /**
   * @brief A numeric attribute.
   * @param key The attribute.
   * @return Its numbers, or empty where there is no such numeric attribute.
   */
  [[nodiscard]] std::optional<std::vector<double>> Numbers(std::string_view key) const {
    return detail::NumbersOf(attributes, key);
  }

  /** @brief Whether the variable varies with direction. */
  [[nodiscard]] bool IsLateral() const {
    return dimensions.size() >= 3 && dimensions[1] == "latitude" &&
           dimensions[2] == "longitude";
  }
};

/** @brief One part of a model file: a run of layers on one grid. */
struct FilePart {
  int index = 0;                         /**< `part_<index>`. */
  std::vector<int> layers{};             /**< The layers it holds, contiguous. */
  std::size_t ngll = 0;                  /**< The nodes per element. */
  std::vector<int> element_layer{};      /**< The layer of every element. */
  std::vector<double> element_bounds{};  /**< Each element's ends, two per element. */
  std::vector<double> radius{};          /**< Every node's radius, element by element. */
  std::optional<AngularGrid> grid{};     /**< The angular grid, with its weights. */
  std::map<std::string, FileVariable> variables{}; /**< The variables by name. */

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
 * @brief The `geometry` group: the mapping as displacements of surfaces,
 * linear in r between them.
 */
struct FileGeometry {
  std::vector<double> radii{};          /**< The surfaces' radii, increasing. */
  std::optional<AngularGrid> grid{};    /**< The grid the displacements are on. */
  std::vector<double> values{};         /**< (boundary, ntheta, nphi), C order. */
  std::optional<std::vector<double>> poles{}; /**< (boundary, 2), where the grid lacks a pole. */
  std::string lateral{};                /**< The lateral rule stated. */
  std::string between{};                /**< The rule between surfaces: `linear`. */
};

/**
 * @brief The `surfaces` group: fields on some of the skeleton's
 * interfaces, each a number at every point of the group's grid.
 */
struct FileSurfaces {
  std::vector<int> interfaces{};      /**< The interfaces holding any, increasing. */
  std::optional<AngularGrid> grid{};  /**< The grid the fields are on. */
  /** @brief The variables by name: (surface, ntheta, nphi), and `<name>_poles`
   * (surface, 2) where the grid lacks a pole; NaN where an interface lacks it. */
  std::map<std::string, FileVariable> variables{};

  /**
   * @brief The row of an interface, or empty where it holds no field.
   * @param interface The interface, from the centre.
   */
  [[nodiscard]] std::optional<std::size_t> Row(int interface) const {
    const auto it = std::ranges::find(interfaces, interface);
    if (it == interfaces.end()) return std::nullopt;
    return static_cast<std::size_t>(it - interfaces.begin());
  }
};

/** @brief A constant of the model, in its units. */
struct FileConstant {
  double value = 0.0;                     /**< In the file's scales. */
  std::optional<Dimensions> dimensions{}; /**< Where the file states them. */
  std::string meaning{};                  /**< What it is, where the file says. */
};

/** @brief A point by its spherical coordinates: radius, colatitude, longitude. */
struct Point {
  double r = 0.0;     /**< The radius. */
  double theta = 0.0; /**< The colatitude, in [0, pi]. */
  double phi = 0.0;   /**< The longitude. */
};

/**
 * @brief The points a layer's values are given at.
 *
 * A source of the radius alone is given at each of `radii`, the layer's
 * nodes element by element, an element boundary twice.  A source that
 * varies with direction is given at (r, theta_i, phi_j) for each radius,
 * colatitude and longitude of the grid, the longitude fastest, and then,
 * where the grid lacks a pole, at (r, 0, 0) and (r, pi, 0) for each
 * radius, north then south: its components in the spherical frame there
 * are those of the frame along the meridian phi = 0.
 */
struct LayerPoints {
  std::vector<double> radii; /**< The radii, centre outward. */
  AngularGrid grid;          /**< The angular grid. */
  bool poles = false;        /**< Whether values at both poles follow the grid's. */

  /** @brief The number of points a source that varies with direction is given at. */
  [[nodiscard]] std::size_t Size() const {
    return radii.size() * (grid.NTheta() * grid.NPhi() + (poles ? 2 : 0));
  }

  /**
   * @brief A point of a source that varies with direction, by its place in
   * the order.
   * @param index The place, below `Size()`.
   */
  [[nodiscard]] Point At(std::size_t index) const {
    const auto cells = grid.NTheta() * grid.NPhi();
    const auto on_grid = radii.size() * cells;
    if (index < on_grid) {
      const auto cell = index % cells;
      return {radii[index / cells], grid.Colatitudes()[cell / grid.NPhi()],
              grid.Longitudes()[cell % grid.NPhi()]};
    }
    const auto pole = index - on_grid;
    return {radii[pole / 2], pole % 2 == 0 ? 0.0 : std::numbers::pi, 0.0};
  }
};

/**
 * @brief What a layer holds of a variable: a function of the radius, or of
 * the point, or the values at the points the layer's `LayerPoints` give.
 *
 * Every function and every value gives the stored components, `Size()` of
 * them in the variable's layout (a Voigt vector or matrix for ranks 2 and
 * 4 in Voigt form, without engineering factors).  A source of the radius
 * alone gives its components in the spherical frame, which are the same
 * at every point of a sphere; a source that varies with direction gives
 * them in the frame it names.  Every value must be finite.
 */
class LayerSource {
 public:
  /** @brief A function of the radius alone, writing the components. */
  using OfRadius = std::function<void(double r, std::span<double> out)>;
  /** @brief A function of the point, writing the components. */
  using OfPoint = std::function<void(double r, double theta, double phi, std::span<double> out)>;

  /**
   * @brief A field of the radius alone.
   * @param f Its spherical components at a radius.
   */
  [[nodiscard]] static LayerSource Radial(OfRadius f) {
    auto out = LayerSource{};
    out.of_radius_ = std::move(f);
    return out;
  }

  /**
   * @brief A scalar of the radius alone.
   * @param f Its value at a radius.
   */
  [[nodiscard]] static LayerSource Radial(std::function<double(double r)> f) {
    auto out = LayerSource{};
    out.scalar_ = true;
    out.of_radius_ = [f = std::move(f)](double r, std::span<double> v) { v[0] = f(r); };
    return out;
  }

  /**
   * @brief A field of the radius alone, given at the layer's nodes.
   * @param values Its spherical components at each of the layer's radii,
   *        radius by radius.
   */
  [[nodiscard]] static LayerSource RadialValues(std::vector<double> values) {
    auto out = LayerSource{};
    out.values_ = std::move(values);
    return out;
  }

  /**
   * @brief A field that varies with direction.
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
   * @brief A scalar that varies with direction.
   * @param f Its value at a point.
   */
  [[nodiscard]] static LayerSource Lateral(
      std::function<double(double r, double theta, double phi)> f) {
    auto out = LayerSource{};
    out.radial_ = false;
    out.scalar_ = true;
    out.of_point_ = [f = std::move(f)](double r, double theta, double phi, std::span<double> v) {
      v[0] = f(r, theta, phi);
    };
    return out;
  }

  /**
   * @brief A field that varies with direction, given at the layer's points.
   * @param values Its components at each of the points `LayerPoints::At`
   *        gives, point by point, in `frame`.
   * @param frame The frame the components are in.
   */
  [[nodiscard]] static LayerSource LateralValues(std::vector<double> values, Frame frame) {
    auto out = LayerSource{};
    out.radial_ = false;
    out.frame_ = frame;
    out.values_ = std::move(values);
    return out;
  }

  /** @brief Whether the field is of the radius alone. */
  [[nodiscard]] bool IsRadial() const noexcept { return radial_; }

 private:
  friend class ModelFile;

  LayerSource() = default;

  /// The stored values of the source at `points` for a variable of
  /// `character`: (radius, components) in the spherical frame where it is
  /// of the radius alone and `lateral` is false; otherwise (radius, ntheta,
  /// nphi, components) in the spherical frame at the grid's nodes, with
  /// (radius, 2, components) Cartesian at the poles where the grid lacks
  /// one.
  struct Sampled {
    std::vector<double> values{};
    std::vector<double> poles{};
  };

  Sampled Sample(const LayerPoints& points, const Character& character, bool lateral,
                 const std::string& source, std::string_view where) const {
    const auto n = character.StoredSize();
    const auto rank = character.Rank();
    const auto voigt = character.Voigt();
    const auto& grid = points.grid;
    const auto cells = grid.NTheta() * grid.NPhi();
    const auto count = points.radii.size();
    if (scalar_ && n != 1) {
      detail::Fail(source, where, std::format("is given by a scalar source, and stores {} "
                                              "components",
                                              n));
    }
    auto finite = [&](std::span<const double> v, const Point& at) {
      for (const auto x : v) {
        if (!std::isfinite(x)) {
          detail::Fail(source, where,
                       std::format("is not finite at (r, theta, phi) = ({}, {}, {})", at.r,
                                   at.theta, at.phi));
        }
      }
    };
    auto turned = [&](std::span<const double> v, const Matrix3& m) {
      return rank == 0 ? std::vector<double>(v.begin(), v.end())
                       : RotateStored(v, rank, voigt,
                                      std::vector<Matrix3>(static_cast<std::size_t>(rank), m));
    };
    auto transposed = [](const Matrix3& m) {
      auto t = Matrix3{};
      for (auto i = std::size_t{0}; i < 3; ++i) {
        for (auto j = std::size_t{0}; j < 3; ++j) t[i][j] = m[j][i];
      }
      return t;
    };
    auto out = Sampled{};
    if (radial_) {
      auto sph = std::vector<double>(count * n);
      if (values_) {
        if (values_->size() != count * n) {
          detail::Fail(source, where,
                       std::format("is given {} values, and its {} radii take {}", values_->size(),
                                   count, count * n));
        }
        sph = *values_;
      } else {
        for (auto k = std::size_t{0}; k < count; ++k) {
          of_radius_(points.radii[k], std::span<double>(sph).subspan(k * n, n));
        }
      }
      for (auto k = std::size_t{0}; k < count; ++k) {
        finite(std::span<const double>(sph).subspan(k * n, n), {points.radii[k], 0.0, 0.0});
      }
      if (!lateral) {
        out.values = std::move(sph);
        return out;
      }
      out.values.resize(count * cells * n);
      for (auto k = std::size_t{0}; k < count; ++k) {
        for (auto cell = std::size_t{0}; cell < cells; ++cell) {
          std::copy_n(sph.begin() + static_cast<std::ptrdiff_t>(k * n), n,
                      out.values.begin() + static_cast<std::ptrdiff_t>((k * cells + cell) * n));
        }
      }
      if (points.poles) {
        for (auto k = std::size_t{0}; k < count; ++k) {
          for (const auto theta : {0.0, std::numbers::pi}) {
            const auto c = turned(std::span<const double>(sph).subspan(k * n, n),
                                  SphericalFrame(theta, 0.0));
            out.poles.insert(out.poles.end(), c.begin(), c.end());
          }
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
    out.values.reserve(count * cells * n);
    auto buffer = std::vector<double>(n);
    for (auto index = std::size_t{0}; index < size; ++index) {
      const auto at = points.At(index);
      if (values_) {
        std::copy_n(values_->begin() + static_cast<std::ptrdiff_t>(index * n), n, buffer.begin());
      } else {
        of_point_(at.r, at.theta, at.phi, buffer);
      }
      finite(buffer, at);
      const auto R = SphericalFrame(at.theta, at.phi);
      if (index < count * cells) {
        // spherical at the grid's nodes
        const auto v = frame_ == Frame::Cartesian ? turned(buffer, transposed(R)) : buffer;
        out.values.insert(out.values.end(), v.begin(), v.end());
      } else {
        // Cartesian at the poles
        const auto v = frame_ == Frame::Spherical ? turned(buffer, R) : buffer;
        out.poles.insert(out.poles.end(), v.begin(), v.end());
      }
    }
    return out;
  }

  bool radial_ = true;
  bool scalar_ = false;
  Frame frame_ = Frame::Spherical;
  OfRadius of_radius_{};
  OfPoint of_point_{};
  std::optional<std::vector<double>> values_{};
};

/** @brief The rules a variable states, radially and laterally. */
struct Rules {
  RadialRule radial = RadialRule::Polynomial; /**< Between the radial nodes. */
  LateralRule lateral = LateralRule::Linear;  /**< Between the grid's nodes. */
};

/** @brief A perturbation of a radial reference, on some of its layers. */
struct Perturbation {
  std::string kind{};            /**< One of `PERTURBATION_KINDS`. */
  std::map<int, LayerSource> layers{}; /**< The perturbed layers, each a layer of the reference. */
  Rules rules{};                 /**< The rules the perturbation states. */
};

class ModelFile;
class ScalarField;

/**
 * @brief A variable on one layer as a function of the point, by the rules
 * the file states.
 */
class Field {
 public:
  /** @brief The field's name. */
  [[nodiscard]] const std::string& Name() const noexcept { return name_; }
  /** @brief The layer it is on. */
  [[nodiscard]] int Layer() const noexcept { return layer_; }
  /** @brief The layer's radii. */
  [[nodiscard]] std::pair<double, double> Interval() const noexcept { return interval_; }
  /** @brief The field's character, its law under a mapping. */
  [[nodiscard]] const Character& GetCharacter() const noexcept { return character_; }
  /** @brief The number of stored components. */
  [[nodiscard]] std::size_t Size() const { return character_.StoredSize(); }
  /** @brief Its dimensions, where the file states them. */
  [[nodiscard]] const std::optional<Dimensions>& GetDimensions() const noexcept {
    return dimensions_;
  }
  /** @brief Whether it depends on the radius alone on its layer. */
  [[nodiscard]] bool IsRadial() const noexcept {
    return reference_->radial && (!perturbation_ || perturbation_->radial);
  }
  /** @brief The kind of its perturbation, where it is a perturbed field. */
  [[nodiscard]] const std::optional<std::string>& PerturbationKind() const noexcept {
    return kind_;
  }

  /**
   * @brief The stored components at a point.
   * @param r The radius, within the layer to the file's tolerance.
   * @param theta The colatitude, in [0, pi].
   * @param phi The longitude, of any range.
   * @param frame The frame the components are given in.
   * @return The components, `Size()` of them.
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] std::vector<double> Evaluate(double r, double theta, double phi,
                                             Frame frame) const {
    auto out = Read(*reference_, r, theta, phi, frame).value;
    if (perturbation_) {
      const auto delta = Read(*perturbation_, r, theta, phi, frame).value[0];
      out[0] = *kind_ == "absolute"   ? out[0] + delta
               : *kind_ == "relative" ? out[0] * (1.0 + delta)
                                      : out[0] * std::exp(delta);
    }
    return out;
  }

  /**
   * @brief The stored components' derivatives in r, theta and phi at a
   * point: of the Cartesian components for a field of rank one and above.
   * @param r The radius, within the layer to the file's tolerance.
   * @param theta The colatitude, in [0, pi].
   * @param phi The longitude, of any range.
   * @return The derivatives in r, theta and phi, `Size()` of each.
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] std::array<std::vector<double>, 3> Derivatives(double r, double theta,
                                                               double phi) const {
    auto ref = Read(*reference_, r, theta, phi, Frame::Cartesian, true);
    if (!perturbation_) return std::move(ref.derivative);
    // a perturbed field is of rank 0, its perturbation too
    const auto dp = Read(*perturbation_, r, theta, phi, Frame::Cartesian, true);
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
  /// stored components in the spherical frame where it is radial, or the
  /// lateral rule's interpolants of the Cartesian components.
  struct Piece {
    bool radial = true;
    Character character{0};
    RadialRule rule = RadialRule::Polynomial;
    std::optional<RadialLayout> layout{};
    std::vector<double> nodal{};               // [(node * ncomp) + c], radial
    std::vector<LateralInterpolant> lateral{};  // [(node * ncomp) + c], lateral
  };

  struct Reading {
    std::vector<double> value{};
    std::array<std::vector<double>, 3> derivative{};
  };

  Field() = default;

  /// A piece's components at a point in `frame`, and with `derivatives`
  /// their Cartesian derivatives.
  static Reading Read(const Piece& p, double r, double theta, double phi, Frame frame,
                      bool derivatives = false) {
    const auto n = p.character.StoredSize();
    const auto rank = p.character.Rank();
    const auto voigt = p.character.Voigt();
    auto out = Reading{};
    out.value.assign(n, 0.0);
    if (derivatives) {
      for (auto& d : out.derivative) d.assign(n, 0.0);
    }
    // the element's radial weights, once for every component
    const auto& layout = *p.layout;
    const auto e = layout.ElementOf(r);
    const auto ngll = layout.Ngll();
    auto w = std::vector<double>(ngll);
    auto dw = std::vector<double>(derivatives ? ngll : 0);
    layout.Weights(e, r, p.rule, w, dw);
    if (p.radial) {
      auto sph = std::vector<double>(n, 0.0);
      auto dsph = std::vector<double>(n, 0.0);
      for (auto k = std::size_t{0}; k < ngll; ++k) {
        const auto* at = &p.nodal[(e * ngll + k) * n];
        for (auto c = std::size_t{0}; c < n; ++c) {
          sph[c] += w[k] * at[c];
          if (derivatives) dsph[c] += dw[k] * at[c];
        }
      }
      if (rank == 0 || frame == Frame::Spherical) out.value = sph;
      const auto R = SphericalFrame(theta, phi);
      const auto Rs = std::vector<Matrix3>(static_cast<std::size_t>(rank), R);
      if (rank > 0 && frame == Frame::Cartesian) out.value = RotateStored(sph, rank, voigt, Rs);
      if (!derivatives) return out;
      out.derivative[0] = rank == 0 ? dsph : RotateStored(dsph, rank, voigt, Rs);
      if (rank == 0) return out;
      // the frame's own derivatives, one factor at a time
      const auto dR = SphericalFrameDerivatives(theta, phi);
      for (auto d = 0; d < 2; ++d) {
        for (auto s = 0; s < rank; ++s) {
          auto Ms = Rs;
          Ms[static_cast<std::size_t>(s)] = dR[static_cast<std::size_t>(d)];
          const auto term = RotateStored(sph, rank, voigt, Ms);
          for (auto c = std::size_t{0}; c < n; ++c) {
            out.derivative[static_cast<std::size_t>(d) + 1][c] += term[c];
          }
        }
      }
      return out;
    }
    auto cart = std::vector<double>(n, 0.0);
    for (auto k = std::size_t{0}; k < ngll; ++k) {
      for (auto c = std::size_t{0}; c < n; ++c) {
        const auto& f = p.lateral[((e * ngll) + k) * n + c];
        const auto v = f.Value(theta, phi);
        cart[c] += w[k] * v;
        if (derivatives) {
          const auto [gt, gp] = f.Gradient(theta, phi);
          out.derivative[0][c] += dw[k] * v;
          out.derivative[1][c] += w[k] * gt;
          out.derivative[2][c] += w[k] * gp;
        }
      }
    }
    if (rank == 0 || frame == Frame::Cartesian) {
      out.value = cart;
    } else {
      const auto R = SphericalFrame(theta, phi);
      auto Rt = Matrix3{};
      for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
          Rt[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
              R[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)];
        }
      }
      out.value = RotateStored(cart, rank, voigt,
                               std::vector<Matrix3>(static_cast<std::size_t>(rank), Rt));
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

/** @brief A field of rank zero, read as a number. */
class ScalarField {
 public:
  /**
   * @brief The value at a point.
   * @param r The radius, within the layer to the file's tolerance.
   * @param theta The colatitude, in [0, pi].
   * @param phi The longitude, of any range.
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] double Value(double r, double theta, double phi) const {
    return field_.Evaluate(r, theta, phi, Frame::Spherical)[0];
  }

  /**
   * @brief The derivatives in r, theta and phi at a point.
   * @param r The radius, within the layer to the file's tolerance.
   * @param theta The colatitude, in [0, pi].
   * @param phi The longitude, of any range.
   * @return (d/dr, d/dtheta, d/dphi).
   * @throws planetmodel::Error if the point lies off the layer.
   */
  [[nodiscard]] std::array<double, 3> Gradient(double r, double theta, double phi) const {
    const auto d = field_.Derivatives(r, theta, phi);
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
 * @brief The mapping a model file holds: m(X) = X + u(X), from the
 * displacements of surfaces or from a displacement field, or the
 * identity.
 */
class ModelMapping {
 public:
  /** @brief The form the file holds the mapping in. */
  enum class Form {
    Identity, /**< The file holds no mapping. */
    Stack,    /**< The `geometry` group: surfaces' displacements, linear in r between. */
    Field     /**< The `displacement` field of every part. */
  };

  /** @brief The form the mapping is held in. */
  [[nodiscard]] Form GetForm() const noexcept { return form_; }

  /**
   * @brief The displacement u(X) = m(X) - X, Cartesian, at a reference
   * point.
   * @param X The point, Cartesian, within the reference body to the
   *        file's tolerance.
   * @throws planetmodel::Error if the point lies outside the body.
   */
  [[nodiscard]] Vector3 Displacement(const Vector3& X) const {
    const auto r = std::hypot(X[0], X[1], X[2]);
    if (!(r >= inner_ - tolerance_ && r <= outer_ + tolerance_)) {
      throw Error(source_, "mapping",
                  std::format("|X| = {} lies outside the body's [{}, {}]", r, inner_, outer_));
    }
    if (form_ == Form::Identity) return {0.0, 0.0, 0.0};
    const auto theta = std::atan2(std::hypot(X[0], X[1]), X[2]);
    const auto phi = std::atan2(X[1], X[0]);
    if (form_ == Form::Field) {
      const auto u =
          fields_[LayerOf(r)].Evaluate(std::clamp(r, inner_, outer_), theta, phi, Frame::Cartesian);
      return {u[0], u[1], u[2]};
    }
    // a radial stretch: h along e_r, X h / r, the limit dh/dr X at the centre
    const auto i = StackLayer(r);
    const auto lo = radii_[i];
    const auto hi = radii_[i + 1];
    const auto below = Relief(i, theta, phi);
    const auto above = Relief(i + 1, theta, phi);
    const auto t = (r - lo) / (hi - lo);
    const auto h = (1.0 - t) * below + t * above;
    const auto ratio = r > R_FLOOR * outer_ ? h / r : (above - below) / (hi - lo);
    return {X[0] * ratio, X[1] * ratio, X[2] * ratio};
  }

  /**
   * @brief The mapped point m(X) = X + u(X), Cartesian.
   * @param X The reference point, Cartesian, within the body.
   * @throws planetmodel::Error if the point lies outside the body.
   */
  [[nodiscard]] Vector3 operator()(const Vector3& X) const {
    const auto u = Displacement(X);
    return {X[0] + u[0], X[1] + u[1], X[2] + u[2]};
  }

 private:
  friend class ModelFile;

  /// A point closer to the origin than this fraction of the outer radius
  /// is the origin, as `mapping.R_FLOOR_FRACTION` takes it.
  static constexpr auto R_FLOOR = 1e-9;

  std::size_t LayerOf(double r) const {
    const auto i = std::upper_bound(boundaries_.begin(), boundaries_.end(), r) -
                   boundaries_.begin() - 1;
    return static_cast<std::size_t>(
        std::clamp<std::ptrdiff_t>(i, 0, static_cast<std::ptrdiff_t>(boundaries_.size()) - 2));
  }

  std::size_t StackLayer(double r) const {
    const auto i = std::upper_bound(radii_.begin(), radii_.end(), r) - radii_.begin() - 1;
    return static_cast<std::size_t>(
        std::clamp<std::ptrdiff_t>(i, 0, static_cast<std::ptrdiff_t>(radii_.size()) - 2));
  }

  double Relief(std::size_t k, double theta, double phi) const {
    const auto& f = reliefs_[k];
    return f ? f->Value(theta, phi) : 0.0;
  }

  Form form_ = Form::Identity;
  std::string source_{};
  double inner_ = 0.0;
  double outer_ = 0.0;
  double tolerance_ = 0.0;
  std::vector<double> boundaries_{};
  std::vector<Field> fields_{};
  std::vector<double> radii_{};
  std::vector<std::optional<LateralInterpolant>> reliefs_{};
};

/**
 * @brief A model file: read and checked, its numbers as stored, or built
 * to be written.
 */
class ModelFile {
 public:
  /**
   * @brief A model file to build: a skeleton in `scales`, the geometry's
   * tolerance `rtol`, with no parts yet.
   * @param skeleton The reference skeleton.
   * @param scales The scales the numbers are in.
   * @param rtol The geometry's tolerance, relative to the skeleton's span.
   * @throws planetmodel::Error if `rtol` is not in (0, 1).
   */
  explicit ModelFile(::planetmodel::Skeleton skeleton,
                     ::planetmodel::Scales scales = ::planetmodel::Scales::SI(),
                     double rtol = DEFAULT_RTOL)
      : source_("ModelFile"),
        schema_(MODEL_SCHEMA),
        scales_(scales),
        rtol_(rtol),
        skeleton_(std::move(skeleton)) {
    if (!(rtol_ > 0.0 && rtol_ < 1.0)) Fail("rtol", "is not a tolerance in (0, 1)");
    const auto& b = skeleton_->Boundaries();
    layer_names_.assign(skeleton_->NLayers(), std::nullopt);
    for (auto j = skeleton_->IsHollow() ? std::size_t{0} : std::size_t{1}; j < b.size(); ++j) {
      interface_radii_.push_back(b[j]);
    }
    interface_names_.assign(interface_radii_.size(), std::nullopt);
  }

  /** @brief The path it was read from, as errors name it; `ModelFile` for one being built. */
  [[nodiscard]] const std::string& Source() const noexcept { return source_; }
  /** @brief The schema it declares, `MODEL_SCHEMA`. */
  [[nodiscard]] const std::string& Schema() const noexcept { return schema_; }
  /** @brief The code that wrote the file (the attribute `source`); empty in
   * files from before the writers stamped it. */
  [[nodiscard]] const std::string& WrittenBy() const noexcept { return written_by_; }
  /** @brief When the file was written, ISO 8601 (the attribute `created`);
   * empty in files from before the writers stamped it. */
  [[nodiscard]] const std::string& Created() const noexcept { return created_; }
  /** @brief The model's name, where it has one. */
  [[nodiscard]] const std::optional<std::string>& Name() const noexcept { return name_; }
  /** @brief The works the model cites, one a string, in the order stated; empty for none. */
  [[nodiscard]] const std::vector<std::string>& Citation() const noexcept { return citation_; }
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
  /** @brief The reference skeleton. */
  [[nodiscard]] const ::planetmodel::Skeleton& GetSkeleton() const noexcept { return *skeleton_; }
  /** @brief Each layer's name, empty where it has none. */
  [[nodiscard]] const std::vector<std::optional<std::string>>& LayerNames() const noexcept {
    return layer_names_;
  }
  /** @brief Each interface's name, empty where it has none. */
  [[nodiscard]] const std::vector<std::optional<std::string>>& InterfaceNames() const noexcept {
    return interface_names_;
  }
  /** @brief Each interface's radius, centre outward. */
  [[nodiscard]] const std::vector<double>& InterfaceRadii() const noexcept {
    return interface_radii_;
  }
  /** @brief The parts, centre outward. */
  [[nodiscard]] const std::vector<FilePart>& Parts() const noexcept { return parts_; }
  /** @brief The `geometry` group, where the file holds the mapping so. */
  [[nodiscard]] const std::optional<FileGeometry>& Geometry() const noexcept { return geometry_; }
  /** @brief The `surfaces` group, where the file holds fields on interfaces. */
  [[nodiscard]] const std::optional<FileSurfaces>& Surfaces() const noexcept { return surfaces_; }

  /**
   * @brief The names of the fields an interface holds.
   * @param interface The interface, from the centre.
   */
  [[nodiscard]] std::vector<std::string> SurfaceNames(int interface) const {
    auto out = std::vector<std::string>{};
    const auto row = surfaces_ ? surfaces_->Row(interface) : std::nullopt;
    if (!row) return out;
    for (const auto& [name, var] : surfaces_->variables) {
      if (!name.ends_with("_poles") && HoldsRow(var, *row)) out.push_back(name);
    }
    return out;
  }

  /**
   * @brief Whether an interface holds a field.
   * @param interface The interface, from the centre.
   * @param name The field.
   */
  [[nodiscard]] bool HasSurfaceField(int interface, std::string_view name) const {
    return std::ranges::contains(SurfaceNames(interface), std::string{name});
  }

  /**
   * @brief A field on an interface, a function of the angles by the
   * lateral rule the file states.
   * @param interface The interface, from the centre.
   * @param name The field.
   * @return Its interpolant: `Value(theta, phi)` and `Gradient(theta, phi)`.
   * @throws planetmodel::Error if the interface does not hold the field, or
   *         its stated lateral rule is `harmonic`, which the C++ does not
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
    const auto cells = grid.NTheta() * grid.NPhi();
    const auto rule = LateralRuleNamed(*var.Text("lateral_interpolation"), source_, where);
    return LateralInterpolant(grid, std::span<const double>(var.values).subspan(row * cells, cells),
                              SurfacePoles(var.name, row), rule, where);
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
   * @param layer The layer.
   * @param name The variable.
   * @throws planetmodel::Error if the layer does not hold the variable as
   *         a field of the model, or its stated lateral rule is `harmonic`,
   *         which the C++ does not read.
   */
  [[nodiscard]] ::planetmodel::Field FieldOn(int layer, std::string_view name) const {
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
    auto out = ::planetmodel::Field{};
    out.source_ = source_;
    out.name_ = std::string{name};
    out.layer_ = layer;
    out.interval_ = skeleton_->Interval(static_cast<std::size_t>(layer));
    out.character_ = CharacterOf(var);
    out.dimensions_ = detail::DimensionsFrom(var.Numbers("dimensions"));
    out.reference_ = std::make_shared<const ::planetmodel::Field::Piece>(
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
        out.perturbation_ = std::make_shared<const ::planetmodel::Field::Piece>(
            Piece(part, layer, pert, CharacterOf(pert)));
      }
    }
    return out;
  }

  /**
   * @brief A field per layer that holds a variable, centre outward.
   * @param name The variable.
   * @throws planetmodel::Error as `FieldOn` does.
   */
  [[nodiscard]] std::vector<::planetmodel::Field> FieldsOf(std::string_view name) const {
    auto out = std::vector<::planetmodel::Field>{};
    for (auto layer = 0; layer < static_cast<int>(skeleton_->NLayers()); ++layer) {
      if (HasField(layer, name)) out.push_back(FieldOn(layer, name));
    }
    return out;
  }

  /**
   * @brief The mapping the file holds, the identity where it holds none.
   * @throws planetmodel::Error if a displacement field's lateral rule is
   *         `harmonic`.
   */
  [[nodiscard]] ModelMapping Mapping() const {
    auto out = ModelMapping{};
    out.source_ = source_;
    out.inner_ = skeleton_->Boundaries().front();
    out.outer_ = skeleton_->Boundaries().back();
    out.tolerance_ = rtol_ * skeleton_->Span();
    out.boundaries_ = skeleton_->Boundaries();
    if (geometry_) {
      out.form_ = ModelMapping::Form::Stack;
      out.radii_ = geometry_->radii;
      const auto& g = *geometry_->grid;
      const auto rule = LateralRuleNamed(geometry_->lateral, source_, "geometry");
      const auto cells = g.NTheta() * g.NPhi();
      for (auto k = std::size_t{0}; k < geometry_->radii.size(); ++k) {
        const auto values = std::span<const double>(geometry_->values).subspan(k * cells, cells);
        const auto poles = geometry_->poles
                               ? std::span<const double>(*geometry_->poles).subspan(2 * k, 2)
                               : std::span<const double>{};
        const auto zero = std::ranges::all_of(values, [](double v) { return v == 0.0; }) &&
                          std::ranges::all_of(poles, [](double v) { return v == 0.0; });
        if (zero) {
          out.reliefs_.emplace_back(std::nullopt);
        } else {
          out.reliefs_.emplace_back(LateralInterpolant(g, values, poles, rule, "geometry"));
        }
      }
      return out;
    }
    if (!parts_.empty() && parts_.front().variables.contains("displacement")) {
      out.form_ = ModelMapping::Form::Field;
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
    constexpr auto space = std::string_view{" \t\n\r\f\v"};
    auto out = std::vector<std::string>{};
    for (const auto& work : works) {
      const auto first = work.find_first_not_of(space);
      if (first == std::string::npos) continue;
      auto w = work.substr(first, work.find_last_not_of(space) - first + 1);
      if (w.find_first_of("\r\n") != std::string::npos) {
        Fail("citation", "holds a work of more than one line");
      }
      if (std::ranges::find(out, w) == out.end()) out.push_back(std::move(w));
    }
    citation_ = std::move(out);
  }

  /**
   * @brief Name a layer.
   * @param layer The layer, from the centre.
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
   * @param interface The interface, from the centre.
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
   * Parts are added centre outward, each starting at the layer after the
   * last one's.  An edge within the file's tolerance of a boundary is
   * taken as on it.
   *
   * @param first The innermost layer of the part.
   * @param last The outermost layer of the part.
   * @param edges The ends of the elements, strictly increasing, from the
   *        bottom of `first` to the top of `last` and at every boundary
   *        between (`ElementEdges` gives equal elements).
   * @param ngll The nodes per element, at least two.
   * @param grid The part's angular grid.
   * @throws planetmodel::Error if the part does not follow the last, or the
   *         edges or `ngll` are not as above.
   */
  void AddPart(int first, int last, std::vector<double> edges, std::size_t ngll,
               AngularGrid grid) {
    const auto where = std::format("part_{}", parts_.size());
    const auto next = parts_.empty() ? 0 : parts_.back().layers.back() + 1;
    const auto nlayers = static_cast<int>(skeleton_->NLayers());
    if (first != next) {
      Fail(where, std::format("starts at layer {}, and parts are added centre outward from "
                              "layer {}",
                              first, next));
    }
    if (last < first || last >= nlayers) {
      Fail(where, std::format("ends at layer {}, outside [{}, {}]", last, first, nlayers - 1));
    }
    if (ngll < 2) Fail(where, std::format("ngll = {}; an element has two nodes or more", ngll));
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
      Fail(where, "its edges do not run from the bottom of its first layer to the top of its "
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
      part.radius.insert(part.radius.end(), nodes.begin(), nodes.end());
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
    auto radii = std::vector<double>(
        part.radius.begin() + static_cast<std::ptrdiff_t>(e0 * part.ngll),
        part.radius.begin() + static_cast<std::ptrdiff_t>(e1 * part.ngll));
    return {std::move(radii), *part.grid, part.grid->LacksPole()};
  }

  /**
   * @brief A field of a name the vocabulary holds, its spec from there.
   * @param name The name.
   * @param layers Each layer holding it, and its source there.
   * @param rules The rules the variable states.
   * @throws planetmodel::Error if the vocabulary has no such name, or as
   *         the other `AddField` does.
   */
  void AddField(const std::string& name, const std::map<int, ::planetmodel::LayerSource>& layers,
                const Rules& rules = {}) {
    AddField(name, KnownSpec(name), layers, rules);
  }

  /**
   * @brief A field: the variable `name` on each layer of `layers`, from its
   * source there, in the part holding the layer.
   *
   * In a part where every layer holding it has a source of the radius
   * alone the variable is `name(node)`; otherwise it varies with direction,
   * the layers of the radius alone named in its `radial_layers`.
   *
   * @param name The name, which the format does not keep for itself and the
   *        file does not yet hold.
   * @param spec Its character, dimensions and meaning; a vocabulary name's
   *        must be the vocabulary's.
   * @param layers Each layer holding it, and its source there.
   * @param rules The rules the variable states.
   * @throws planetmodel::Error if the name or the spec is refused, a layer
   *         is in no part, the grid cannot carry the lateral rule, or a
   *         source gives values that do not fit or are not finite.
   */
  void AddField(const std::string& name, const FieldSpec& spec,
                const std::map<int, ::planetmodel::LayerSource>& layers, const Rules& rules = {}) {
    CheckNewName(name, spec);
    auto made = std::vector<std::pair<std::size_t, std::vector<FileVariable>>>{};
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
   * @param reference Each layer of the reference, of the radius alone.
   * @param perturbation The perturbation and the layers it perturbs.
   * @param reference_rule The radial rule the reference states.
   * @throws planetmodel::Error if the vocabulary has no such name, or as
   *         the other `AddPerturbedField` does.
   */
  void AddPerturbedField(const std::string& name,
                         const std::map<int, ::planetmodel::LayerSource>& reference,
                         const Perturbation& perturbation,
                         RadialRule reference_rule = RadialRule::Polynomial) {
    AddPerturbedField(name, KnownSpec(name), reference, perturbation, reference_rule);
  }

  /**
   * @brief A perturbed field: a reference of the radius alone and a
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
   * @param reference Each layer of the reference, of the radius alone.
   * @param perturbation The perturbation and the layers it perturbs.
   * @param reference_rule The radial rule the reference states.
   * @throws planetmodel::Error as `AddField` does, or if the field is not
   *         of rank 0, the kind is unknown, a reference layer's source
   *         varies with direction, or a perturbed layer is not one of the
   *         reference's.
   */
  void AddPerturbedField(const std::string& name, const FieldSpec& spec,
                         const std::map<int, ::planetmodel::LayerSource>& reference,
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
        Fail(name, std::format("has a reference that varies with direction on layer {}; a "
                               "reference is of the radius alone",
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
    auto made = std::vector<std::pair<std::size_t, std::vector<FileVariable>>>{};
    for (const auto& [index, holders] : ByPart(name, reference)) {
      const auto& part = parts_[index];
      auto perturbed = std::vector<int>{};
      for (const auto layer : holders) {
        if (perturbation.layers.contains(layer)) perturbed.push_back(layer);
      }
      auto variables = Variable(part, name, spec, spec.character, holders, reference,
                                {.radial = reference_rule}, false);
      if (!perturbed.empty()) {
        auto& ref = variables.front();
        ref.attributes.insert_or_assign("perturbation", pert_name);
        ref.attributes.insert_or_assign("perturbation_kind", perturbation.kind);
        auto radial = true;
        for (const auto layer : perturbed) {
          radial = radial && perturbation.layers.at(layer).IsRadial();
        }
        auto pert = Variable(part, pert_name, pert_spec, pert_spec.character, perturbed,
                             perturbation.layers, perturbation.rules, !radial);
        if (!absolute) pert.front().attributes.insert_or_assign("units", std::string{"1"});
        pert.front().attributes.insert_or_assign("long_name", pert_spec.meaning);
        if (pert.size() > 1) {
          pert.back().attributes.insert_or_assign(
              "long_name", std::format("{} at the poles", pert_spec.meaning));
          pert.back().attributes.insert_or_assign("units", pert.front().attributes.at("units"));
        }
        variables.insert(variables.end(), pert.begin(), pert.end());
      }
      made.emplace_back(index, std::move(variables));
    }
    Commit(std::move(made));
  }

  /**
   * @brief The mapping as the radial displacements of surfaces, linear in
   * r between them: the `geometry` group.
   * @param radii The surfaces' reference radii, strictly increasing, from
   *        the skeleton's inner boundary to its outer one; they may lie
   *        inside layers.
   * @param grid The grid the displacements are held on.
   * @param reliefs Each surface's radial displacement, a scalar source
   *        (evaluated at the surface's radius), or empty where it stays
   *        spherical.
   * @param lateral The lateral rule the displacements state.
   * @throws planetmodel::Error if the file holds a mapping already, the
   *         surfaces are not as above, the grid cannot carry the rule, or a
   *         relief gives values that do not fit or are not finite.
   */
  void SetStack(std::vector<double> radii, AngularGrid grid,
                const std::vector<std::optional<::planetmodel::LayerSource>>& reliefs,
                LateralRule lateral = LateralRule::Cubic) {
    const auto where = std::string{"geometry/boundary_displacement"};
    if (HoldsMapping()) Fail("geometry", "the file holds a mapping already");
    if (radii.size() < 2 || reliefs.size() != radii.size()) {
      Fail(where, std::format("is given {} reliefs for {} surfaces; a stack is two surfaces "
                              "or more, each with a relief",
                              reliefs.size(), radii.size()));
    }
    for (auto k = std::size_t{1}; k < radii.size(); ++k) {
      if (!(radii[k] > radii[k - 1])) Fail(where, "its surfaces' radii do not increase strictly");
    }
    const auto tolerance = rtol_ * skeleton_->Span();
    if (radii.front() > skeleton_->Boundaries().front() + tolerance ||
        radii.back() < skeleton_->Boundaries().back() - tolerance) {
      Fail(where, "its surfaces do not span the skeleton");
    }
    CheckRule(grid, lateral, where);
    const auto cells = grid.NTheta() * grid.NPhi();
    auto geometry = FileGeometry{.radii = radii,
                                 .values = std::vector<double>(radii.size() * cells),
                                 .lateral = std::string{RuleName(lateral)},
                                 .between = "linear"};
    auto poles = std::vector<double>(radii.size() * 2);
    for (auto k = std::size_t{0}; k < radii.size(); ++k) {
      if (!reliefs[k]) continue;
      const auto points = LayerPoints{{radii[k]}, grid, grid.LacksPole()};
      const auto sampled = reliefs[k]->Sample(points, SCALAR, true, source_, where);
      std::ranges::copy(sampled.values,
                        geometry.values.begin() + static_cast<std::ptrdiff_t>(k * cells));
      if (!sampled.poles.empty()) {
        std::ranges::copy(sampled.poles, poles.begin() + static_cast<std::ptrdiff_t>(2 * k));
      }
    }
    if (grid.LacksPole()) geometry.poles = std::move(poles);
    geometry.grid = std::move(grid);
    geometry_ = std::move(geometry);
  }

  /**
   * @brief The mapping as a displacement field, m(X) - X, on every layer,
   * in every part.
   *
   * Every part is on one grid, so that the parts' displacements agree where
   * they meet; the parts are added first.
   *
   * @param layers Each layer of the skeleton, and the displacement's source
   *        there, a vector.
   * @param rules The rules the displacement states.
   * @throws planetmodel::Error if the file holds a mapping already, its
   *         parts do not hold every layer on one grid, a layer has no
   *         source, the grid cannot carry the rule, or a source gives values
   *         that do not fit or are not finite.
   */
  void SetDisplacement(const std::map<int, ::planetmodel::LayerSource>& layers,
                       const Rules& rules = {.lateral = LateralRule::Cubic}) {
    if (HoldsMapping()) Fail("displacement", "the file holds a mapping already");
    const auto nlayers = static_cast<int>(skeleton_->NLayers());
    if (parts_.empty() || parts_.back().layers.back() != nlayers - 1) {
      Fail("displacement", "is set once the parts hold every layer");
    }
    for (const auto& part : parts_) {
      if (*part.grid != *parts_.front().grid) {
        Fail("displacement",
             "is held in every part, read by the part's own rule, so the parts must share "
             "one grid: give every layer one grid, or the mapping as the displacements of "
             "surfaces with SetStack");
      }
    }
    for (auto layer = 0; layer < nlayers; ++layer) {
      if (!layers.contains(layer)) {
        Fail("displacement", std::format("has no source on layer {}", layer));
      }
    }
    const auto spec = FieldSpec{.character = VECTOR, .dimensions = units::LENGTH,
                                .meaning = "displacement of the mapping, m(X) - X, spherical "
                                           "components at X"};
    auto made = std::vector<std::pair<std::size_t, std::vector<FileVariable>>>{};
    for (auto index = std::size_t{0}; index < parts_.size(); ++index) {
      auto variables = Variable(parts_[index], "displacement", spec, VECTOR,
                                parts_[index].layers, layers, rules, true);
      for (auto& var : variables) {
        for (const auto* key : {"rank", "weight", "voigt", "layers"}) var.attributes.erase(key);
      }
      if (variables.size() > 1) {
        variables.back().attributes.insert_or_assign(
            "long_name",
            std::string{"displacement at the north and south poles, Cartesian components"});
      }
      made.emplace_back(index, std::move(variables));
    }
    Commit(std::move(made));
  }

  /**
   * @brief The grid the fields on interfaces are held on, set before the
   * first of them is added.
   * @param grid The grid.
   * @throws planetmodel::Error if the file holds a field on an interface
   *         already.
   */
  void SetSurfaceGrid(AngularGrid grid) {
    if (surfaces_ && !surfaces_->variables.empty()) {
      Fail("surfaces", "holds fields already, on the grid set before them");
    }
    surfaces_ = FileSurfaces{.grid = std::move(grid)};
  }

  /**
   * @brief The points the values of a field on an interface are given at,
   * for a `LayerSource` of values: the interface's radius on the surface
   * grid, and both poles where the grid lacks one.
   * @param interface The interface, from the centre.
   * @throws planetmodel::Error if the skeleton has no such interface, or the
   *         surface grid is not set.
   */
  [[nodiscard]] LayerPoints SurfacePointsOf(int interface) const {
    CheckInterface(interface, "surfaces");
    if (!surfaces_) Fail("surfaces", "has no grid: set it with SetSurfaceGrid first");
    const auto& grid = *surfaces_->grid;
    return {{interface_radii_[static_cast<std::size_t>(interface)]}, grid, grid.LacksPole()};
  }

  /**
   * @brief A field on interfaces: the variable `name` of the `surfaces`
   * group, from its source on each of `interfaces`, evaluated at the
   * interface's radius on the surface grid, NaN on the other interfaces
   * the group holds.
   * @param name The name, which the format does not keep for itself and the
   *        file does not yet hold.
   * @param spec Its dimensions and meaning, of rank 0; a vocabulary name's
   *        must be the vocabulary's.
   * @param interfaces Each interface holding it, from the centre, and its
   *        source there, a scalar.
   * @param lateral The lateral rule the variable states.
   * @throws planetmodel::Error if the name or the spec is refused, the spec
   *         is not of rank 0, the surface grid is not set, an interface is
   *         not the skeleton's, the grid cannot carry the rule, or a source
   *         gives values that do not fit or are not finite.
   */
  void AddSurfaceField(const std::string& name, const FieldSpec& spec,
                       const std::map<int, ::planetmodel::LayerSource>& interfaces,
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
    const auto cells = grid.NTheta() * grid.NPhi();
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    auto var = FileVariable{.name = name,
                            .dimensions = {"surface", "latitude", "longitude"},
                            .shape = {rows.size(), grid.NTheta(), grid.NPhi()},
                            .values = std::vector<double>(rows.size() * cells, nan)};
    auto poles = FileVariable{.name = name + "_poles",
                              .dimensions = {"surface", "pole"},
                              .shape = {rows.size(), 2},
                              .values = std::vector<double>(rows.size() * 2, nan)};
    auto holders = std::vector<int>{};
    for (const auto& [k, source] : interfaces) {
      const auto row = static_cast<std::size_t>(std::ranges::find(rows, k) - rows.begin());
      const auto sampled = source.Sample(SurfacePointsOf(k), SCALAR, true, source_,
                                         std::format("{} on interface {}", where, k));
      std::ranges::copy(sampled.values,
                        var.values.begin() + static_cast<std::ptrdiff_t>(row * cells));
      if (!sampled.poles.empty()) {
        std::ranges::copy(sampled.poles, poles.values.begin() + static_cast<std::ptrdiff_t>(2 * row));
      }
      holders.push_back(k);
    }
    const auto units = units::ScaledUnit(spec.dimensions, scales_);
    auto& a = var.attributes;
    a.emplace("long_name", spec.meaning.empty() ? name : spec.meaning);
    a.emplace("units", units);
    if (spec.dimensions) a.emplace("dimensions", detail::Exponents(*spec.dimensions));
    a.emplace("lateral_interpolation", std::string{RuleName(lateral)});
    a.emplace("interfaces", detail::Integers(holders));
    held.variables.emplace(name, std::move(var));
    if (grid.LacksPole()) {
      poles.attributes.emplace("long_name", std::format("{} at the north and south poles", name));
      poles.attributes.emplace("units", units);
      held.variables.emplace(poles.name, std::move(poles));
    }
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
      Fail(name, "is a name the model file keeps for itself");
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
      const std::string& name, const std::map<int, ::planetmodel::LayerSource>& layers) const {
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

  /// The variable `name` of a part on its layers `holders`, and its poles
  /// where it varies with direction on a grid that lacks a pole: the
  /// values from the sources, NaN on the part's other layers, and the
  /// attributes the format asks for.
  std::vector<FileVariable> Variable(const FilePart& part, const std::string& name,
                                     const FieldSpec& spec, const Character& character,
                                     const std::vector<int>& holders,
                                     const std::map<int, ::planetmodel::LayerSource>& sources,
                                     const Rules& rules, bool lateral) const {
    const auto where = std::format("part_{}/{}", part.index, name);
    const auto& grid = *part.grid;
    if (lateral) CheckRule(grid, rules.lateral, where);
    const auto n = character.StoredSize();
    const auto axes = AxesOf(character, where);
    const auto stored = character.StoredShape();
    const auto nnode = part.radius.size();
    const auto cells = grid.NTheta() * grid.NPhi();
    auto var = FileVariable{.name = name};
    var.dimensions = lateral ? std::vector<std::string>{"node", "latitude", "longitude"}
                             : std::vector<std::string>{"node"};
    var.shape = lateral ? std::vector<std::size_t>{nnode, grid.NTheta(), grid.NPhi()}
                        : std::vector<std::size_t>{nnode};
    var.dimensions.insert(var.dimensions.end(), axes.begin(), axes.end());
    var.shape.insert(var.shape.end(), stored.begin(), stored.end());
    const auto per_node = (lateral ? cells : 1) * n;
    var.values.assign(nnode * per_node, std::numeric_limits<double>::quiet_NaN());
    const auto with_poles = lateral && grid.LacksPole();
    auto poles = FileVariable{.name = name + "_poles"};
    if (with_poles) {
      poles.dimensions = {"node", "pole"};
      poles.dimensions.insert(poles.dimensions.end(), axes.begin(), axes.end());
      poles.shape = {nnode, 2};
      poles.shape.insert(poles.shape.end(), stored.begin(), stored.end());
      poles.values.assign(nnode * 2 * n, std::numeric_limits<double>::quiet_NaN());
    }
    auto radial_layers = std::vector<int>{};
    for (const auto layer : holders) {
      const auto& source = sources.at(layer);
      if (lateral && source.IsRadial()) radial_layers.push_back(layer);
      const auto [e0, e1] = part.Elements(layer);
      const auto n0 = e0 * part.ngll;
      const auto sampled = source.Sample(PointsOf(layer), character, lateral, source_,
                                         std::format("{} on layer {}", where, layer));
      std::ranges::copy(sampled.values,
                        var.values.begin() + static_cast<std::ptrdiff_t>(n0 * per_node));
      if (with_poles) {
        std::ranges::copy(sampled.poles,
                          poles.values.begin() + static_cast<std::ptrdiff_t>(n0 * 2 * n));
      }
    }
    const auto units = units::ScaledUnit(spec.dimensions, scales_);
    auto& a = var.attributes;
    a.emplace("long_name", spec.meaning.empty() ? name : spec.meaning);
    a.emplace("units", units);
    a.emplace("rank", detail::Integer(character.Rank()));
    a.emplace("weight", detail::Integer(character.Weight()));
    a.emplace("voigt", detail::Integer(character.Voigt() ? 1 : 0));
    a.emplace("layers", detail::Integers(holders));
    a.emplace("radial_interpolation", std::string{RuleName(rules.radial)});
    if (spec.dimensions) a.emplace("dimensions", detail::Exponents(*spec.dimensions));
    if (lateral) a.emplace("lateral_interpolation", std::string{RuleName(rules.lateral)});
    a.emplace("coordinates", std::string{"radius"});
    if (!radial_layers.empty()) a.emplace("radial_layers", detail::Integers(radial_layers));
    auto out = std::vector<FileVariable>{std::move(var)};
    if (with_poles) {
      poles.attributes.emplace(
          "long_name", std::format("{} at the north and south poles, Cartesian components", name));
      poles.attributes.emplace("units", units);
      out.push_back(std::move(poles));
    }
    return out;
  }

  /// Variables made for parts, put into them once all are made.
  void Commit(std::vector<std::pair<std::size_t, std::vector<FileVariable>>> made) {
    for (auto& [index, variables] : made) {
      for (auto& var : variables) {
        auto name = var.name;
        parts_[index].variables.insert_or_assign(std::move(name), std::move(var));
      }
    }
  }

  /// Refuse an interface the skeleton lacks.
  void CheckInterface(int interface, std::string_view where) const {
    if (interface < 0 || interface >= static_cast<int>(interface_radii_.size())) {
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

  /// A surface field's values at both poles on a row, or nothing where
  /// the group holds no poles for it.
  std::span<const double> SurfacePoles(const std::string& name, std::size_t row) const {
    const auto it = surfaces_->variables.find(name + "_poles");
    if (it == surfaces_->variables.end()) return {};
    return std::span<const double>(it->second.values).subspan(2 * row, 2);
  }

  /// The frames at a grid's nodes, `rank` copies of each, for turning
  /// components of rank one and above.
  static std::vector<std::vector<Matrix3>> GridFrames(const AngularGrid& grid, int rank) {
    auto out = std::vector<std::vector<Matrix3>>{};
    if (rank == 0) return out;
    for (const auto theta : grid.Colatitudes()) {
      for (const auto phi : grid.Longitudes()) {
        out.emplace_back(static_cast<std::size_t>(rank), SphericalFrame(theta, phi));
      }
    }
    return out;
  }

  /// A lateral variable's values at one node in Cartesian components, laid
  /// out component by component, each a grid of values.
  static std::vector<double> NodeCartesian(const FilePart& part, const FileVariable& var,
                                           std::size_t node, const Character& character,
                                           const std::vector<std::vector<Matrix3>>& frames) {
    const auto cells = part.grid->NTheta() * part.grid->NPhi();
    const auto ncomp = character.StoredSize();
    const auto rank = character.Rank();
    auto cart = std::vector<double>(ncomp * cells);
    for (auto cell = std::size_t{0}; cell < cells; ++cell) {
      const auto at = std::span<const double>(var.values).subspan((node * cells + cell) * ncomp,
                                                                   ncomp);
      const auto turned = rank > 0 ? RotateStored(at, rank, character.Voigt(), frames[cell])
                                   : std::vector<double>(at.begin(), at.end());
      for (auto c = std::size_t{0}; c < ncomp; ++c) cart[c * cells + cell] = turned[c];
    }
    return cart;
  }

  /// A lateral variable's Cartesian component `c` at both poles at one
  /// node, or nothing where the part holds no poles for it.
  static std::vector<double> NodePoles(const FilePart& part, const FileVariable& var,
                                       std::size_t node, std::size_t c, std::size_t ncomp) {
    const auto it = part.variables.find(var.name + "_poles");
    if (it == part.variables.end()) return {};
    return {it->second.values[(node * 2) * ncomp + c],
            it->second.values[(node * 2 + 1) * ncomp + c]};
  }

  /// Refuse what the reading of the fields would refuse, and what the
  /// Python's `read_model` would: a vocabulary name stated otherwise, the
  /// cubic rule on a grid that cannot carry it, and a pole node whose
  /// values are not one value; a variable stating the harmonic rule is
  /// left to the Python.
  void CheckReadable(const std::string& source) const {
    for (const auto& part : parts_) {
      const auto& grid = *part.grid;
      for (const auto& [name, var] : part.variables) {
        if (name.ends_with("_poles")) continue;
        const auto where = std::format("part_{}/{}", part.index, name);
        const auto character = CharacterOf(var);
        if (IsFieldName(name)) {
          if (const auto known = VocabularySpec(name)) {
            const auto dims = detail::DimensionsFrom(var.Numbers("dimensions"));
            if (known->character != character || (dims && dims != known->dimensions)) {
              detail::Fail(source, where,
                           "is a vocabulary name, stated with another character or other "
                           "dimensions than the vocabulary's");
            }
          }
        }
        if (!var.IsLateral()) continue;
        const auto stated = *var.Text("lateral_interpolation");
        if (stated == "harmonic") continue;
        const auto rule = LateralRuleNamed(stated, source, where);
        CheckRule(grid, rule, where);
        if (!grid.NorthPoleNode() && !grid.SouthPoleNode()) continue;
        auto radial_layers = std::vector<int>{};
        if (const auto listed = var.Numbers("radial_layers")) {
          for (const auto x : *listed) radial_layers.push_back(static_cast<int>(x));
        }
        const auto frames = GridFrames(grid, character.Rank());
        const auto ncomp = character.StoredSize();
        const auto cells = grid.NTheta() * grid.NPhi();
        for (const auto layer : part.layers) {
          if (std::ranges::contains(radial_layers, layer) || !Holds(part, layer, var)) continue;
          const auto [e0, e1] = part.Elements(layer);
          for (auto node = e0 * part.ngll; node < e1 * part.ngll; ++node) {
            const auto cart = NodeCartesian(part, var, node, character, frames);
            for (auto c = std::size_t{0}; c < ncomp; ++c) {
              static_cast<void>(PolarValues(grid,
                                            std::span<const double>(cart).subspan(c * cells, cells),
                                            NodePoles(part, var, node, c, ncomp), 1e-9, where));
            }
          }
        }
      }
    }
    if (geometry_) {
      const auto where = std::string{"geometry/boundary_displacement"};
      const auto& grid = *geometry_->grid;
      if (geometry_->lateral == "harmonic") return;
      CheckRule(grid, LateralRuleNamed(geometry_->lateral, source, where), where);
      const auto cells = grid.NTheta() * grid.NPhi();
      for (auto k = std::size_t{0}; k < geometry_->radii.size(); ++k) {
        const auto values = std::span<const double>(geometry_->values).subspan(k * cells, cells);
        const auto poles = geometry_->poles
                               ? std::span<const double>(*geometry_->poles).subspan(2 * k, 2)
                               : std::span<const double>{};
        static_cast<void>(PolarValues(grid, values, poles, 1e-9, where));
      }
    }
    if (surfaces_) {
      const auto& grid = *surfaces_->grid;
      const auto cells = grid.NTheta() * grid.NPhi();
      for (const auto& [name, var] : surfaces_->variables) {
        if (name.ends_with("_poles")) continue;
        const auto where = "surfaces/" + name;
        if (const auto known = VocabularySpec(name)) {
          const auto dims = detail::DimensionsFrom(var.Numbers("dimensions"));
          if (known->character.Rank() != 0 || (dims && dims != known->dimensions)) {
            detail::Fail(source, where,
                         "is a vocabulary name, stated with another character or other "
                         "dimensions than the vocabulary's");
          }
        }
        const auto stated = *var.Text("lateral_interpolation");
        if (stated == "harmonic") continue;
        CheckRule(grid, LateralRuleNamed(stated, source, where), where);
        for (auto row = std::size_t{0}; row < surfaces_->interfaces.size(); ++row) {
          if (!HoldsRow(var, row)) continue;
          static_cast<void>(PolarValues(
              grid, std::span<const double>(var.values).subspan(row * cells, cells),
              SurfacePoles(name, row), 1e-9, where));
        }
      }
    }
  }

  /// Whether a variable is a field of the model, not a field's poles, a
  /// field's perturbation or the mapping.
  static bool IsFieldName(std::string_view name) {
    return !name.ends_with("_poles") && !name.ends_with("_perturbation") &&
           name != "displacement";
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

  /// The character a variable states; a displacement is a vector.
  static Character CharacterOf(const FileVariable& var) {
    if (var.name == "displacement") return Character(1, 0, false);
    auto integer = [&var](std::string_view key) {
      const auto v = var.Numbers(key);
      if (!v || v->size() != 1) throw Error("", var.name, std::format("{} is not a number", key));
      return static_cast<int>((*v)[0]);
    };
    return Character(integer("rank"), integer("weight"), integer("voigt") != 0);
  }


  /// The displacement field of a layer, a vector of the stated rules.
  ::planetmodel::Field Displacement(int layer) const {
    const auto& part = PartOf(layer);
    const auto& var = part.variables.at("displacement");
    auto out = ::planetmodel::Field{};
    out.source_ = source_;
    out.name_ = "displacement";
    out.layer_ = layer;
    out.interval_ = skeleton_->Interval(static_cast<std::size_t>(layer));
    out.character_ = Character(1, 0, false);
    out.reference_ = std::make_shared<const ::planetmodel::Field::Piece>(
        Piece(part, layer, var, out.character_));
    return out;
  }

  /// One variable on one layer as a piece of a field.
  ::planetmodel::Field::Piece Piece(const FilePart& part, int layer, const FileVariable& var,
                                    const Character& character) const {
    const auto where = std::format("part_{}/{}", part.index, var.name);
    auto out = ::planetmodel::Field::Piece{};
    out.character = character;
    out.rule = RadialRuleNamed(*var.Text("radial_interpolation"), source_, where);
    const auto [e0, e1] = part.Elements(layer);
    const auto ngll = part.ngll;
    const auto n0 = e0 * ngll;
    const auto n1 = e1 * ngll;
    const auto ncomp = character.StoredSize();
    auto radii = std::vector<double>(part.radius.begin() + static_cast<std::ptrdiff_t>(n0),
                                     part.radius.begin() + static_cast<std::ptrdiff_t>(n1));
    out.layout = RadialLayout(std::move(radii), ngll, rtol_ * skeleton_->Span());
    auto radial_layers = std::vector<int>{};
    if (const auto listed = var.Numbers("radial_layers")) {
      for (const auto x : *listed) radial_layers.push_back(static_cast<int>(x));
    }
    const auto lateral = var.IsLateral();
    out.radial = !lateral || std::ranges::contains(radial_layers, layer);
    if (out.radial) {
      // the stored components at every node, a lateral variable's from its
      // first column, all of whose columns are the same
      const auto stride = lateral ? part.grid->NTheta() * part.grid->NPhi() * ncomp : ncomp;
      for (auto node = n0; node < n1; ++node) {
        const auto* at = &var.values[node * stride];
        out.nodal.insert(out.nodal.end(), at, at + ncomp);
      }
      return out;
    }
    const auto& grid = *part.grid;
    const auto rule = LateralRuleNamed(*var.Text("lateral_interpolation"), source_, where);
    const auto nt = grid.NTheta();
    const auto np = grid.NPhi();
    const auto frames = GridFrames(grid, character.Rank());
    for (auto node = n0; node < n1; ++node) {
      const auto cart = NodeCartesian(part, var, node, character, frames);
      for (auto c = std::size_t{0}; c < ncomp; ++c) {
        out.lateral.emplace_back(grid, std::span<const double>(cart).subspan(c * nt * np, nt * np),
                                 NodePoles(part, var, node, c, ncomp), rule, where);
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
  ::planetmodel::Scales scales_{};
  double rtol_ = DEFAULT_RTOL;
  std::map<std::string, FileConstant> constants_{};
  netcdf::Attributes meta_{};
  std::optional<::planetmodel::Skeleton> skeleton_{};
  std::vector<std::optional<std::string>> layer_names_{};
  std::vector<std::optional<std::string>> interface_names_{};
  std::vector<double> interface_radii_{};
  std::vector<FilePart> parts_{};
  std::optional<FileGeometry> geometry_{};
  std::optional<FileSurfaces> surfaces_{};
};

namespace detail {

/// The angular grid a group's coordinates and attributes hold, as the
/// Python's `_grid_of` reads it: colatitude 90 - latitude in radians,
/// clipped to [0, pi]; longitude in radians reduced into [0, 2 pi); the
/// colatitude weights where the group has them.
inline AngularGrid GridOf(int group, const netcdf::Attributes& attrs, const std::string& source) {
  constexpr auto degree = std::numbers::pi / 180.0;
  auto theta = netcdf::ReadDoubles(group, "latitude", source).values;
  for (auto& t : theta) t = std::clamp((90.0 - t) * degree, 0.0, std::numbers::pi);
  auto phi = netcdf::ReadDoubles(group, "longitude", source).values;
  for (auto& p : phi) p = Wrapped(p * degree);
  auto lmax = std::optional<int>{};
  if (attrs.contains("grid_lmax")) {
    const auto l = NumberOf(attrs, "grid_lmax");
    if (!l) Fail(source, "grid_lmax", "is not a degree");
    lmax = static_cast<int>(*l);
  }
  auto weights = std::optional<std::vector<double>>{};
  if (netcdf::HasVariable(group, "colatitude_weights")) {
    weights = netcdf::ReadDoubles(group, "colatitude_weights", source).values;
  }
  return AngularGrid(std::move(theta), std::move(phi),
                     TextOf(attrs, "grid_kind").value_or("custom"), lmax, std::move(weights));
}

inline FileVariable ReadVariable(int group, const std::string& name, const std::string& source) {
  auto out = FileVariable{.name = name};
  auto array = netcdf::ReadDoubles(group, name, source);
  out.values = std::move(array.values);
  out.shape = std::move(array.shape);
  out.dimensions = netcdf::DimensionNames(group, name, source);
  out.attributes = netcdf::ReadAttributes(group, netcdf::Variable(group, name, source), source);
  return out;
}

/// A grid's latitudes in degrees, as the Python's `geographic` gives them.
inline std::vector<double> Latitudes(const AngularGrid& grid) {
  constexpr auto degrees = 180.0 / std::numbers::pi;
  auto out = std::vector<double>{};
  for (const auto theta : grid.Colatitudes()) out.push_back(90.0 - theta * degrees);
  return out;
}

/// A grid's longitudes in degrees east, in [0, 360), as `geographic`
/// gives them.
inline std::vector<double> Longitudes(const AngularGrid& grid) {
  constexpr auto degrees = 180.0 / std::numbers::pi;
  auto out = std::vector<double>{};
  for (const auto phi : grid.Longitudes()) out.push_back(Wrapped(phi * degrees, 0.0, 360.0));
  return out;
}

/// A name beside `path` that nothing has, for a file until it is whole.
inline std::filesystem::path PartialName(const std::filesystem::path& path) {
  auto device = std::random_device{};
  for (auto attempt = 0; attempt < 100; ++attempt) {
    auto candidate = path;
    candidate.replace_filename(std::format(".{}.{:08x}", path.filename().string(), device()));
    auto error = std::error_code{};
    if (!std::filesystem::exists(candidate, error) && !error) return candidate;
  }
  Fail(path.string(), "", "no name is free beside the file to write it under");
}

}  // namespace detail

/**
 * @brief Hold a model file to the structure the readers check, the
 * Python's `read_model_file` among them: the schema and tolerance, every
 * layer in one part of contiguous layers, each part's elements tiling its
 * layers at their Gauss-Lobatto-Legendre nodes, every variable running
 * over the nodes on the part's grid with the attributes and trailing axes
 * its character asks for, known rules, the poles where the grid lacks
 * one, a perturbation that is there and of a known kind, the mapping
 * held once, and the fields on interfaces on the skeleton's interfaces,
 * over the group's grid, under names no part holds.
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
  const auto& b = skeleton.Boundaries();
  const auto nlayers = skeleton.NLayers();
  if (file.layer_names_.size() != nlayers) fail("layer_name", "is not one name per layer");
  if (file.interface_names_.size() != file.interface_radii_.size()) {
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

  // the parts
  for (const auto& part : file.parts_) {
    const auto where = std::format("part_{}", part.index);
    const auto& grid = *part.grid;
    const auto nelem = part.element_layer.size();
    const auto nnode = part.radius.size();
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
    const auto bottom = b[static_cast<std::size_t>(part.layers.front())];
    const auto top = b[static_cast<std::size_t>(part.layers.back()) + 1];
    const auto tile = 1e-12 * (top - bottom);
    auto tiles = std::abs(eb.front() - bottom) <= tile && std::abs(eb.back() - top) <= tile;
    for (auto e = std::size_t{1}; e < nelem; ++e) {
      tiles = tiles && std::abs(eb[2 * e] - eb[2 * e - 1]) <= tile;
    }
    if (!tiles) fail(where, "element_bounds do not tile its layers");
    for (auto e = std::size_t{0}; e < nelem; ++e) {
      const auto nodes = GllNodes(eb[2 * e], eb[2 * e + 1], part.ngll);
      for (auto k = std::size_t{0}; k < part.ngll; ++k) {
        const auto stored = part.radius[e * part.ngll + k];
        if (std::abs(stored - nodes[k]) > 1e-12 * std::abs(nodes[k])) {
          fail(where + "/radius", "is not the Gauss-Lobatto-Legendre nodes of its elements");
        }
      }
    }
    for (const auto& [vname, var] : part.variables) {
      if (vname.ends_with("_poles")) continue;
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
      if (lateral && (var.shape[1] != grid.NTheta() || var.shape[2] != grid.NPhi())) {
        fail(vwhere, "is not on the part's grid");
      }
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
      const auto tail = std::vector<std::size_t>(var.shape.begin() + (lateral ? 3 : 1),
                                                 var.shape.end());
      if (tail != stored) fail(vwhere, "has trailing axes its character does not store");
      const auto radial_rule = var.Text("radial_interpolation");
      if (!radial_rule || !std::ranges::contains(RADIAL_RULES, std::string_view{*radial_rule})) {
        fail(vwhere, "states an unknown radial rule");
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
        if (grid.LacksPole()) {
          const auto it = part.variables.find(vname + "_poles");
          if (it == part.variables.end()) {
            fail(vwhere, std::format("varies with direction on a grid without its poles, and "
                                     "{}_poles is missing",
                                     vname));
          }
          if (it->second.values.size() != nnode * 2 * ModelFile::CharacterOf(var).StoredSize()) {
            fail(vwhere + "_poles", "does not hold both poles at every node");
          }
        }
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
    const auto& radii = geometry.radii;
    for (auto k = std::size_t{1}; k < radii.size(); ++k) {
      if (!(radii[k] > radii[k - 1])) {
        fail("geometry/boundary_radius", "does not increase strictly");
      }
    }
    if (radii.size() < 2) fail("geometry/boundary_radius", "is fewer than two surfaces");
    const auto& grid = *geometry.grid;
    if (geometry.values.size() != radii.size() * grid.NTheta() * grid.NPhi()) {
      fail(where, "has values that do not fill its shape");
    }
    if (grid.LacksPole() && (!geometry.poles || geometry.poles->size() != 2 * radii.size())) {
      fail(where, "varies with direction on a grid without its poles, and "
                  "boundary_displacement_poles is missing");
    }
  }

  // the fields on interfaces
  if (file.surfaces_) {
    const auto& surfaces = *file.surfaces_;
    const auto& faces = surfaces.interfaces;
    const auto& grid = *surfaces.grid;
    const auto ninterface = static_cast<int>(file.interface_radii_.size());
    for (auto i = std::size_t{0}; i < faces.size(); ++i) {
      if (faces[i] < 0 || faces[i] >= ninterface || (i > 0 && faces[i] <= faces[i - 1])) {
        fail("surfaces/surface_interface", "is not interfaces of the file, increasing");
      }
    }
    const auto cells = grid.NTheta() * grid.NPhi();
    for (const auto& [vname, var] : surfaces.variables) {
      if (vname.ends_with("_poles")) continue;
      const auto vwhere = "surfaces/" + vname;
      if (var.dimensions != std::vector<std::string>{"surface", "latitude", "longitude"} ||
          var.shape != std::vector<std::size_t>{faces.size(), grid.NTheta(), grid.NPhi()} ||
          var.values.size() != faces.size() * cells) {
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
      if (var.attributes.contains("interfaces")) {
        for (const auto x : var.Numbers("interfaces").value_or(std::vector<double>{-1.0})) {
          if (!std::ranges::contains(faces, static_cast<int>(x))) {
            fail(vwhere, "names interfaces that are not surfaces of the group");
          }
        }
      }
      if (grid.LacksPole()) {
        const auto it = surfaces.variables.find(vname + "_poles");
        if (it == surfaces.variables.end() ||
            it->second.dimensions != std::vector<std::string>{"surface", "pole"} ||
            it->second.values.size() != 2 * faces.size()) {
          fail(vwhere, std::format("is on a grid without its poles, and {}_poles is missing",
                                   vname));
        }
      }
    }
  }
}

/**
 * @brief A model file, read whole and checked as the Python's
 * `read_model_file` checks it.
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

  // the root
  const auto schema = text("schema");
  if (!schema) fail("schema", "is missing: not a planetmodel model file");
  out.schema_ = *schema;
  out.name_ = text("name");
  out.written_by_ = text("source").value_or("");
  out.created_ = text("created").value_or("");
  if (const auto citation = text("citation")) {
    for (auto start = std::size_t{0}; start < citation->size();) {
      const auto end = std::min(citation->find('\n', start), citation->size());
      out.citation_.push_back(citation->substr(start, end - start));
      start = end + 1;
    }
  }
  auto scales = std::array<double, 4>{};
  constexpr auto bases = std::array<std::string_view, 4>{"length", "mass", "time", "temperature"};
  for (auto b = std::size_t{0}; b < 4; ++b) {
    const auto key = std::format("scale_{}", bases[b]);
    const auto value = number(key);
    if (!value || !(*value > 0.0)) fail(key, "is missing or not positive");
    scales[b] = *value;
  }
  out.scales_ = Scales({.length = scales[0], .mass = scales[1], .time = scales[2],
                        .temperature = scales[3]});
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

  // the skeleton
  for (const auto& [name, dims] : std::vector<std::pair<std::string, std::vector<std::string>>>{
           {"layer_bounds", {"layer", "bound"}},
           {"layer_name", {"layer"}},
           {"interface_radius", {"interface"}},
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
  boundaries.push_back(bounds.back());
  out.skeleton_ = Skeleton(boundaries);
  for (const auto& n : netcdf::ReadStrings(root, "layer_name", source)) {
    out.layer_names_.push_back(n.empty() ? std::nullopt : std::optional<std::string>{n});
  }
  for (const auto& n : netcdf::ReadStrings(root, "interface_name", source)) {
    out.interface_names_.push_back(n.empty() ? std::nullopt : std::optional<std::string>{n});
  }
  out.interface_radii_ = netcdf::ReadDoubles(root, "interface_radius", source).values;
  const auto layer_part_values = netcdf::ReadDoubles(root, "layer_part", source).values;
  auto layer_part = std::vector<int>{};
  for (const auto x : layer_part_values) layer_part.push_back(static_cast<int>(x));
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

  // the parts, as stored
  for (auto& part : out.parts_) {
    const auto where = std::format("part_{}", part.index);
    const auto g = netcdf::Group(root, where, source);
    for (const auto* dim : {"node", "element", "bound", "latitude", "longitude"}) {
      if (!netcdf::DimensionLength(g, dim, source)) {
        fail(where, std::format("has no dimension {}", dim));
      }
    }
    for (const auto* name :
         {"radius", "element_layer", "element_bounds", "latitude", "longitude"}) {
      if (!netcdf::HasVariable(g, name)) fail(where, std::format("has no variable {}", name));
    }
    const auto gattrs = netcdf::ReadAttributes(g, netcdf::GLOBAL, source);
    if (!gattrs.contains("ngll") || !gattrs.contains("grid_kind")) {
      fail(where, "lacks the attribute ngll or grid_kind");
    }
    const auto kind = detail::TextOf(gattrs, "grid_kind");
    if (!kind || !std::ranges::contains(GRID_KINDS, std::string_view{*kind})) {
      fail(where + ".grid_kind",
           std::format("is {}, not a kind of grid", kind.value_or("a number")));
    }
    const auto ngll = detail::NumberOf(gattrs, "ngll");
    if (!ngll || !(*ngll >= 2.0)) fail(where + ".ngll", "is not a number of nodes, two or more");
    part.ngll = static_cast<std::size_t>(*ngll);
    for (const auto x : netcdf::ReadDoubles(g, "element_layer", source).values) {
      part.element_layer.push_back(static_cast<int>(x));
    }
    part.element_bounds = netcdf::ReadDoubles(g, "element_bounds", source).values;
    part.radius = netcdf::ReadDoubles(g, "radius", source).values;
    part.grid = detail::GridOf(g, gattrs, source);
    for (const auto& vname : netcdf::VariableNames(g, source)) {
      if (vname == "radius" || vname == "element_layer" || vname == "element_bounds" ||
          vname == "latitude" || vname == "longitude" || vname == "colatitude_weights") {
        continue;
      }
      part.variables.emplace(vname, detail::ReadVariable(g, vname, source));
    }
  }

  // the geometry group, as stored
  if (std::ranges::contains(groups, std::string{"geometry"})) {
    const auto g = netcdf::Group(root, "geometry", source);
    for (const auto* name : {"boundary_radius", "boundary_displacement", "latitude", "longitude"}) {
      if (!netcdf::HasVariable(g, name)) fail("geometry", std::format("has no variable {}", name));
    }
    if (netcdf::DimensionNames(g, "boundary_displacement", source) !=
        std::vector<std::string>{"boundary", "latitude", "longitude"}) {
      fail("geometry/boundary_displacement", "has the wrong dimensions");
    }
    const auto var = detail::ReadVariable(g, "boundary_displacement", source);
    auto geometry = FileGeometry{};
    geometry.radii = netcdf::ReadDoubles(g, "boundary_radius", source).values;
    geometry.grid = detail::GridOf(g, netcdf::ReadAttributes(g, netcdf::GLOBAL, source), source);
    geometry.values = var.values;
    if (netcdf::HasVariable(g, "boundary_displacement_poles")) {
      geometry.poles = netcdf::ReadDoubles(g, "boundary_displacement_poles", source).values;
    }
    geometry.lateral = var.Text("lateral_interpolation").value_or("");
    geometry.between = var.Text("between").value_or("");
    out.geometry_ = std::move(geometry);
  }

  // the surfaces group, as stored
  if (std::ranges::contains(groups, std::string{"surfaces"})) {
    const auto g = netcdf::Group(root, "surfaces", source);
    for (const auto* dim : {"surface", "latitude", "longitude"}) {
      if (!netcdf::DimensionLength(g, dim, source)) {
        fail("surfaces", std::format("has no dimension {}", dim));
      }
    }
    for (const auto* name : {"surface_interface", "latitude", "longitude"}) {
      if (!netcdf::HasVariable(g, name)) fail("surfaces", std::format("has no variable {}", name));
    }
    const auto gattrs = netcdf::ReadAttributes(g, netcdf::GLOBAL, source);
    const auto kind = detail::TextOf(gattrs, "grid_kind");
    if (!kind || !std::ranges::contains(GRID_KINDS, std::string_view{*kind})) {
      fail("surfaces.grid_kind", std::format("is {}, not a kind of grid",
                                             kind.value_or("missing or a number")));
    }
    auto surfaces = FileSurfaces{};
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
    surfaces.grid = detail::GridOf(g, gattrs, source);
    for (const auto& vname : netcdf::VariableNames(g, source)) {
      if (vname == "surface_interface" || vname == "latitude" || vname == "longitude" ||
          vname == "colatitude_weights") {
        continue;
      }
      surfaces.variables.emplace(vname, detail::ReadVariable(g, vname, source));
    }
    out.surfaces_ = std::move(surfaces);
  }
  ValidateModelFile(out, source);
  return out;
}

/**
 * @brief The ends of elements that cut each of a run of layers into equal
 * elements no longer than `spacing`, for `ModelFile::AddPart`.
 * @param skeleton The skeleton.
 * @param first The innermost layer of the run.
 * @param last The outermost layer of the run.
 * @param spacing The longest an element may be, positive.
 * @throws planetmodel::Error if the layers or the spacing are not as above.
 */
[[nodiscard]] inline std::vector<double> ElementEdges(const Skeleton& skeleton, int first,
                                                      int last, double spacing) {
  if (first < 0 || last < first || last >= static_cast<int>(skeleton.NLayers())) {
    throw Error("ElementEdges", "",
                std::format("layers {} to {} are not a run of the skeleton's", first, last));
  }
  if (!(spacing > 0.0)) throw Error("ElementEdges", "", "spacing must be positive");
  const auto& b = skeleton.Boundaries();
  auto out = std::vector<double>{b[static_cast<std::size_t>(first)]};
  for (auto i = static_cast<std::size_t>(first); i <= static_cast<std::size_t>(last); ++i) {
    const auto lo = b[i];
    const auto hi = b[i + 1];
    const auto count = std::max(1.0, std::ceil((hi - lo) / spacing));
    for (auto k = 1.0; k < count; ++k) out.push_back(lo + (hi - lo) * k / count);
    out.push_back(hi);
  }
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
  put(root, global, "schema", std::string{MODEL_SCHEMA});
  put(root, global, "Conventions", std::string{MODEL_CONVENTIONS});
  put(root, global, "title", name_ ? "planetmodel model " + *name_ : "planetmodel model");
  put(root, global, "source",
      written_by_.empty() ? "planetmodel C++ " + std::string{Version()} : written_by_);
  put(root, global, "created", created_.empty() ? detail::CreatedNow() : created_);
  if (name_) put(root, global, "name", *name_);
  if (!citation_.empty()) {
    auto joined = citation_.front();
    for (auto k = std::size_t{1}; k < citation_.size(); ++k) joined += "\n" + citation_[k];
    put(root, global, "citation", joined);
  }
  put(root, global, "scale_length", std::vector<double>{scales_.Length()});
  put(root, global, "scale_mass", std::vector<double>{scales_.Mass()});
  put(root, global, "scale_time", std::vector<double>{scales_.Time()});
  put(root, global, "scale_temperature", std::vector<double>{scales_.Temperature()});
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
  const auto nlayers = skeleton_->NLayers();
  nc::Dimension(root, "layer", nlayers, source);
  nc::Dimension(root, "interface", interface_radii_.size(), source);
  nc::Dimension(root, "bound", 2, source);
  const auto& b = skeleton_->Boundaries();
  auto bounds = std::vector<double>{};
  for (auto i = std::size_t{0}; i < nlayers; ++i) {
    bounds.push_back(b[i]);
    bounds.push_back(b[i + 1]);
  }
  auto var = nc::WriteDoubles(root, "layer_bounds", {"layer", "bound"}, bounds, source);
  put(root, var, "long_name", std::string{"radii of the layer's inner and outer boundaries"});
  put(root, var, "units", length);
  auto names = std::vector<std::string>{};
  for (const auto& n : layer_names_) names.push_back(n.value_or(""));
  nc::WriteStrings(root, "layer_name", "layer", names, source);
  var = nc::WriteDoubles(root, "interface_radius", {"interface"}, interface_radii_, source);
  put(root, var, "long_name", std::string{"radius of the interface"});
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

  // a group's grid, as coordinates, attributes and weights
  auto write_grid = [&](int group, const AngularGrid& grid, bool long_names) {
    nc::Dimension(group, "latitude", grid.NTheta(), source);
    nc::Dimension(group, "longitude", grid.NPhi(), source);
    put(group, global, "grid_kind", grid.Kind());
    if (grid.Lmax()) put(group, global, "grid_lmax", detail::Integer(*grid.Lmax()));
    auto v = nc::WriteDoubles(group, "latitude", {"latitude"}, detail::Latitudes(grid), source);
    put(group, v, "units", std::string{"degrees_north"});
    put(group, v, "standard_name", std::string{"latitude"});
    if (long_names) put(group, v, "long_name", std::string{"latitude"});
    v = nc::WriteDoubles(group, "longitude", {"longitude"}, detail::Longitudes(grid), source);
    put(group, v, "units", std::string{"degrees_east"});
    put(group, v, "standard_name", std::string{"longitude"});
    if (long_names) put(group, v, "long_name", std::string{"longitude"});
    if (grid.Weights()) {
      v = nc::WriteDoubles(group, "colatitude_weights", {"latitude"}, *grid.Weights(), source);
      if (long_names) {
        put(group, v, "long_name",
            std::string{"colatitude quadrature weights against sin(theta) dtheta"});
      }
      put(group, v, "units", std::string{"1"});
    }
    if (grid.LacksPole()) nc::Dimension(group, "pole", 2, source);
  };

  // the parts
  for (const auto& part : parts_) {
    const auto g = nc::CreateGroup(root, std::format("part_{}", part.index), source);
    nc::Dimension(g, "node", part.radius.size(), source);
    nc::Dimension(g, "element", part.element_layer.size(), source);
    nc::Dimension(g, "bound", 2, source);
    put(g, global, "ngll", detail::Integer(static_cast<long long>(part.ngll)));
    put(g, global, "layers", detail::Integers(part.layers));
    write_grid(g, *part.grid, true);
    if (part.grid->LacksPole()) put(g, global, "poles", std::string{"north south"});
    var = nc::WriteDoubles(g, "radius", {"node"}, part.radius, source);
    put(g, var, "long_name", std::string{"radius of the node; an element boundary is two nodes"});
    put(g, var, "units", length);
    var = nc::WriteIntegers(g, "element_layer", {"element"}, part.element_layer, source);
    put(g, var, "long_name", std::string{"skeleton layer of the element"});
    var = nc::WriteDoubles(g, "element_bounds", {"element", "bound"}, part.element_bounds, source);
    put(g, var, "long_name", std::string{"radii of the element's ends"});
    put(g, var, "units", length);
    for (const auto& [vname, variable] : part.variables) {
      for (auto i = std::size_t{0}; i < variable.dimensions.size(); ++i) {
        nc::Dimension(g, variable.dimensions[i], variable.shape[i], source);
      }
      var = nc::WriteDoubles(g, vname, variable.dimensions, variable.values, source);
      for (const auto& [key, value] : variable.attributes) {
        if (!key.starts_with('_')) put(g, var, key, value);
      }
    }
  }

  // the geometry group
  if (geometry_) {
    const auto& geometry = *geometry_;
    const auto g = nc::CreateGroup(root, "geometry", source);
    nc::Dimension(g, "boundary", geometry.radii.size(), source);
    write_grid(g, *geometry.grid, false);
    var = nc::WriteDoubles(g, "boundary_radius", {"boundary"}, geometry.radii, source);
    put(g, var, "long_name", std::string{"radius of the surface the displacement belongs to"});
    put(g, var, "units", length);
    var = nc::WriteDoubles(g, "boundary_displacement", {"boundary", "latitude", "longitude"},
                           geometry.values, source);
    put(g, var, "long_name", std::string{"radial displacement of the surface"});
    put(g, var, "units", length);
    put(g, var, "lateral_interpolation", geometry.lateral);
    put(g, var, "between", geometry.between);
    if (geometry.poles) {
      var = nc::WriteDoubles(g, "boundary_displacement_poles", {"boundary", "pole"},
                             *geometry.poles, source);
      put(g, var, "long_name", std::string{"radial displacement at the north and south poles"});
      put(g, var, "units", length);
    }
  }

  // the surfaces group
  if (surfaces_ && !surfaces_->variables.empty()) {
    const auto& surfaces = *surfaces_;
    const auto g = nc::CreateGroup(root, "surfaces", source);
    nc::Dimension(g, "surface", surfaces.interfaces.size(), source);
    write_grid(g, *surfaces.grid, true);
    if (surfaces.grid->LacksPole()) put(g, global, "poles", std::string{"north south"});
    var = nc::WriteIntegers(g, "surface_interface", {"surface"}, surfaces.interfaces, source);
    put(g, var, "long_name", std::string{"the interface the surface is, by its index at the root"});
    for (const auto& [vname, variable] : surfaces.variables) {
      for (auto i = std::size_t{0}; i < variable.dimensions.size(); ++i) {
        nc::Dimension(g, variable.dimensions[i], variable.shape[i], source);
      }
      var = nc::WriteDoubles(g, vname, variable.dimensions, variable.values, source);
      for (const auto& [key, value] : variable.attributes) {
        if (!key.starts_with('_')) put(g, var, key, value);
      }
    }
  }
}

/**
 * @brief Write a model file, after holding it to the readers' checks.
 *
 * The file is checked as `ValidateModelFile` checks it, and as the reading
 * of its fields and the Python's `read_model` would (a vocabulary name
 * stated otherwise, the cubic rule on a grid that cannot carry it, a pole
 * node whose values are not one value), before anything is written; it is
 * written under a name of its own beside `path` and put in place when
 * whole, so that a failed write leaves no file and an existing file at
 * `path` stands until it is replaced.
 *
 * @param path The file to write; one there is replaced.
 * @param file The model file, read or built.
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
    auto error = std::error_code{};
    std::filesystem::rename(partial, path, error);
    if (error) detail::Fail(source, "", std::format("cannot be put in place: {}", error.message()));
  } catch (...) {
    auto error = std::error_code{};
    std::filesystem::remove(partial, error);
    throw;
  }
}

}  // namespace planetmodel

#endif  // PLANETMODEL_MODELFILE_HPP
