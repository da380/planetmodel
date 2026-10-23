#ifndef PLANETMODEL_SURFACEFILE_HPP
#define PLANETMODEL_SURFACEFILE_HPP

/**
 * @file
 * @brief The surface file: fields on one spherical surface, static or in
 * time, in one netCDF file.
 *
 * The format is the Python's (`planetmodel.surfacefile`, and
 * `docs/source/formats/surface_file.rst` for consumers): one angular
 * grid and one surface at the root, the optional strictly increasing
 * coordinates `time` and `instant`, the derived dimension `interval`
 * (one shorter than `time`), and one float64 variable per field, NaN
 * marking the points it does not cover, its dimensions fixed by its
 * `time_support` attribute — `static` `(latitude, longitude)`, `node`
 * `(time, latitude, longitude)`, `interval` and `instant` likewise over
 * their own dimensions.  `ReadSurfaceFile(path)` reads a file whole and
 * holds it to the structure; `WriteSurfaceFile(path, file)` writes one
 * from a code's own fields, whole or not at all, for the Python and the
 * other consumers to read; `SurfaceField` makes a field's variable with
 * the attributes the format asks for.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "planetmodel/grid.hpp"
#include "planetmodel/modelfile.hpp"
#include "planetmodel/netcdf.hpp"
#include "planetmodel/units.hpp"

namespace planetmodel {

/** @brief The schema every reader checks; bumped only for an incompatible
 * change. */
inline constexpr std::string_view SURFACE_SCHEMA = "planetmodel.surface/1";

/** @brief How a field sits in time; its dimensions follow (file comment). */
inline constexpr auto TIME_SUPPORTS =
    std::array<std::string_view, 4>{"static", "node", "interval", "instant"};

/** @brief The time rules a `node` field may state; every other support
 * states `none`. */
inline constexpr auto TIME_METHODS =
    std::array<std::string_view, 3>{"linear", "previous", "none"};

/** @brief The lateral rules a surface file may state: the model file's,
 * with `harmonic`, which this library reads as data and does not
 * evaluate. */
inline constexpr auto SURFACE_LATERAL_RULES =
    std::array<std::string_view, 3>{"linear", "cubic", "harmonic"};

namespace detail {

/// Names the format gives its own variables, which no field may take.
inline constexpr auto SURFACE_RESERVED = std::array<std::string_view, 5>{
    "latitude", "longitude", "colatitude_weights", "time", "instant"};

}  // namespace detail

/** @brief What a field of `SurfaceField` states beside its values. */
struct SurfaceFieldOptions {
  std::string support = "static";              /**< One of `TIME_SUPPORTS`. */
  std::string time_interpolation = "none";     /**< One of `TIME_METHODS`. */
  std::string lateral_interpolation = "linear"; /**< One of `SURFACE_LATERAL_RULES`. */
  std::optional<Dimensions> dimensions{};      /**< The field's, where declared. */
  std::string meaning{};                       /**< Its `long_name`; the name, empty. */
};

/**
 * @brief A field's variable, its attributes as the format states them.
 * @param name The field.
 * @param values Its values, C order, the support's leading shape then
 *        (ntheta, nphi); NaN marks the points it does not cover.
 * @param options What it states beside them.
 * @return The variable, for `SurfaceFile::variables`; the writer adds its
 *         `units` from the dimensions and the file's scales.
 */
[[nodiscard]] inline FileVariable SurfaceField(std::string name, std::vector<double> values,
                                               const SurfaceFieldOptions& options = {}) {
  auto out = FileVariable{.name = std::move(name), .values = std::move(values)};
  out.attributes.emplace("long_name", options.meaning.empty() ? out.name : options.meaning);
  out.attributes.emplace("time_support", options.support);
  out.attributes.emplace("time_interpolation", options.time_interpolation);
  out.attributes.emplace("lateral_interpolation", options.lateral_interpolation);
  if (options.dimensions) out.attributes.emplace("dimensions", detail::Exponents(*options.dimensions));
  return out;
}

/**
 * @brief A surface file's numbers and metadata, read or built: nothing is
 * interpolated or converted.
 */
struct SurfaceFile {
  std::string schema{SURFACE_SCHEMA};   /**< `SURFACE_SCHEMA`. */
  Scales scales{};                      /**< What one stored unit is in SI. */
  std::optional<AngularGrid> grid{};    /**< The angular grid, with its weights. */
  std::optional<std::vector<double>> time{};    /**< The time nodes, strictly increasing. */
  std::optional<std::vector<double>> instant{}; /**< The observation times, strictly increasing. */
  std::optional<double> radius{};       /**< The surface's radius, where stated. */
  std::string source{};                 /**< The writing code and version; required. */
  std::string created{};                /**< When, ISO 8601; the writer stamps it when empty. */
  std::optional<std::string> title{};   /**< A name for the contents, where given. */
  netcdf::Attributes meta{};            /**< The free `meta_<key>` attributes, by key. */
  std::vector<std::string> citation{};  /**< The works cited, one a line. */
  /** @brief The fields by name, and `<name>_poles` where a field carries
   * polar values on a grid lacking a pole. */
  std::map<std::string, FileVariable> variables{};
};

namespace detail {

/// The leading length of a support's variables; empty for `static`.
inline std::optional<std::size_t> LeadingOf(std::string_view support,
                                            const std::optional<std::vector<double>>& time,
                                            const std::optional<std::vector<double>>& instant) {
  if (support == "node" && time) return time->size();
  if (support == "interval" && time && time->size() >= 2) return time->size() - 1;
  if (support == "instant" && instant) return instant->size();
  return std::nullopt;
}

/// Strictly increasing and finite, as the axes must be.
inline bool Increasing(const std::vector<double>& values) {
  if (values.empty()) return false;
  auto last = -std::numeric_limits<double>::infinity();
  for (const auto v : values) {
    if (!std::isfinite(v) || !(v > last)) return false;
    last = v;
  }
  return true;
}

}  // namespace detail

/**
 * @brief Hold a surface file to the structure the readers check, the
 * Python's `read_surface_file` among them: the schema, a source and (read
 * back) a created stamp, a grid, increasing axes, a positive radius, and
 * every field of a known support with its dimensions following it, known
 * rules, `none` in time unless it is a `node` field, and polar values
 * only beside a field and only on a grid lacking a pole.
 * @param file The file, read or built.
 * @param source The file's path, as the errors name it.
 * @throws planetmodel::Error naming the place of the first fault.
 */
inline void ValidateSurfaceFile(const SurfaceFile& file, const std::string& source) {
  auto fail = [&source](std::string_view where, std::string_view message) {
    detail::Fail(source, where, message);
  };
  if (file.schema != SURFACE_SCHEMA) {
    fail("schema", std::format("is {}, this reader reads {}", file.schema, SURFACE_SCHEMA));
  }
  if (file.source.find_first_not_of(" \t\n") == std::string::npos) {
    fail("source", "is missing or empty");
  }
  if (!file.grid) fail("grid", "is missing");
  for (const auto& [name, axis] :
       {std::pair{"time", &file.time}, std::pair{"instant", &file.instant}}) {
    if (*axis && !detail::Increasing(**axis)) {
      fail(name, "is not finite and strictly increasing");
    }
  }
  if (file.radius && !(std::isfinite(*file.radius) && *file.radius > 0.0)) {
    fail("radius", "is not a positive number");
  }
  const auto ntheta = file.grid->NTheta();
  const auto nphi = file.grid->NPhi();
  for (const auto& [name, var] : file.variables) {
    if (std::ranges::contains(detail::SURFACE_RESERVED, std::string_view{name})) {
      fail(name, "is a name the surface file keeps for itself");
    }
    if (name.ends_with("_poles")) {
      const auto base = name.substr(0, name.size() - 6);
      const auto it = file.variables.find(base);
      if (it == file.variables.end()) fail(name, "accompanies no field of the file");
      if (!file.grid->LacksPole()) fail(name, "gives polar values on a grid holding both poles");
      const auto support = detail::TextOf(it->second.attributes, "time_support").value_or("");
      const auto leading = support == "static"
                               ? std::optional<std::size_t>{1}
                               : detail::LeadingOf(support, file.time, file.instant);
      if (!leading || var.values.size() != *leading * 2) {
        fail(name, "does not hold two polar values on its field's leading shape");
      }
      continue;
    }
    const auto support = detail::TextOf(var.attributes, "time_support");
    if (!support || !std::ranges::contains(TIME_SUPPORTS, std::string_view{*support})) {
      fail(name, "states no time support: static, node, interval or instant");
    }
    const auto in_time = detail::TextOf(var.attributes, "time_interpolation");
    if (!in_time || !std::ranges::contains(TIME_METHODS, std::string_view{*in_time}) ||
        (*support != "node" && *in_time != "none")) {
      fail(name, std::format("is {} and states the time rule {}", *support,
                             in_time.value_or("none at all")));
    }
    const auto lateral = detail::TextOf(var.attributes, "lateral_interpolation");
    if (!lateral || !std::ranges::contains(SURFACE_LATERAL_RULES, std::string_view{*lateral})) {
      fail(name, "states no known lateral rule");
    }
    auto size = ntheta * nphi;
    if (*support != "static") {
      const auto leading = detail::LeadingOf(*support, file.time, file.instant);
      if (!leading) fail(name, std::format("is {} and the file lacks its axis", *support));
      size *= *leading;
    }
    if (var.values.size() != size) {
      fail(name, std::format("holds {} values, not the {} of its support and the grid",
                             var.values.size(), size));
    }
    if (const auto dims = detail::NumbersOf(var.attributes, "dimensions");
        dims && !detail::DimensionsFrom(dims)) {
      fail(name, "declares dimensions that are not four exponents");
    }
  }
}

/**
 * @brief Read a surface file whole.
 * @param path The file.
 * @return Its numbers and metadata, held to the structure first.
 * @throws planetmodel::Error naming the place of the first fault.
 */
[[nodiscard]] inline SurfaceFile ReadSurfaceFile(const std::filesystem::path& path) {
  namespace nc = netcdf;
  const auto file = nc::File(path);
  const auto root = file.Id();
  const auto& source = file.Path();
  auto fail = [&source](std::string_view where, std::string_view message) {
    detail::Fail(source, where, message);
  };
  auto out = SurfaceFile{};
  const auto attrs = nc::ReadAttributes(root, nc::GLOBAL, source);
  const auto schema = detail::TextOf(attrs, "schema");
  if (!schema) fail("schema", "is missing: not a planetmodel surface file");
  out.schema = *schema;
  if (out.schema != SURFACE_SCHEMA) {
    fail("schema", std::format("is {}, this reader reads {}", out.schema, SURFACE_SCHEMA));
  }
  out.source = detail::TextOf(attrs, "source").value_or("");
  out.created = detail::TextOf(attrs, "created").value_or("");
  if (out.created.empty()) fail("created", "is missing or empty");
  out.title = detail::TextOf(attrs, "title");
  auto scales = std::array<double, 4>{};
  constexpr auto bases = std::array<std::string_view, 4>{"length", "mass", "time", "temperature"};
  for (auto b = std::size_t{0}; b < 4; ++b) {
    const auto key = std::format("scale_{}", bases[b]);
    const auto value = detail::NumberOf(attrs, key);
    if (!value || !(*value > 0.0)) fail(key, "is missing or not positive");
    scales[b] = *value;
  }
  out.scales = Scales({.length = scales[0], .mass = scales[1], .time = scales[2],
                       .temperature = scales[3]});
  if (attrs.contains("radius")) {
    out.radius = detail::NumberOf(attrs, "radius");
    if (!out.radius) fail("radius", "is not a positive number");
  }
  if (const auto citation = detail::TextOf(attrs, "citation")) {
    for (auto start = std::size_t{0}; start < citation->size();) {
      const auto end = std::min(citation->find('\n', start), citation->size());
      out.citation.push_back(citation->substr(start, end - start));
      start = end + 1;
    }
  }
  for (const auto& [key, value] : attrs) {
    if (key.starts_with("meta_")) out.meta.emplace(key.substr(5), value);
  }
  const auto kind = detail::TextOf(attrs, "grid_kind");
  if (!kind || !std::ranges::contains(GRID_KINDS, std::string_view{*kind})) {
    fail("grid_kind", "is missing or not a kind of grid");
  }
  out.grid = detail::GridOf(root, attrs, source);
  for (const auto name : {"time", "instant"}) {
    if (!nc::HasVariable(root, name)) continue;
    if (nc::DimensionNames(root, name, source) != std::vector<std::string>{name}) {
      fail(name, "does not run over its own dimension");
    }
    auto values = nc::ReadDoubles(root, name, source).values;
    (name == std::string_view{"time"} ? out.time : out.instant) = std::move(values);
  }
  if (const auto interval = nc::DimensionLength(root, "interval", source)) {
    if (!out.time || *interval != out.time->size() - 1) {
      fail("interval", "is not one shorter than time");
    }
  }
  for (const auto& name : nc::VariableNames(root, source)) {
    if (std::ranges::contains(detail::SURFACE_RESERVED, std::string_view{name})) continue;
    out.variables.emplace(name, detail::ReadVariable(root, name, source));
  }
  ValidateSurfaceFile(out, source);
  return out;
}

/**
 * @brief Write a surface file, whole or not at all.
 * @param path The file to write; one there is replaced.
 * @param file The surface file, built or read; an empty `created` is
 *        stamped with the writing time in UTC, and each field's `units`
 *        comes from its `dimensions` and the file's scales where the
 *        field does not carry one.
 * @throws planetmodel::Error naming the place of the first fault, or what
 *         kept the file from being written.
 */
inline void WriteSurfaceFile(const std::filesystem::path& path, const SurfaceFile& file) {
  namespace nc = netcdf;
  const auto source = path.string();
  auto stamped = file;
  if (stamped.created.empty()) {
    const auto now = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    stamped.created = std::format("{:%FT%T}+00:00", now);
  }
  ValidateSurfaceFile(stamped, source);
  const auto partial = detail::PartialName(path);
  try {
    auto out = nc::File::Create(partial);
    const auto root = out.Id();
    constexpr auto global = nc::GLOBAL;
    auto put = [&source](int group, int var, std::string_view key, const nc::Attribute& value) {
      nc::WriteAttribute(group, var, key, value, source);
    };
    put(root, global, "schema", std::string{SURFACE_SCHEMA});
    put(root, global, "Conventions", std::string{MODEL_CONVENTIONS});
    if (stamped.title) put(root, global, "title", *stamped.title);
    put(root, global, "source", stamped.source);
    put(root, global, "created", stamped.created);
    put(root, global, "scale_length", std::vector<double>{stamped.scales.Length()});
    put(root, global, "scale_mass", std::vector<double>{stamped.scales.Mass()});
    put(root, global, "scale_time", std::vector<double>{stamped.scales.Time()});
    put(root, global, "scale_temperature", std::vector<double>{stamped.scales.Temperature()});
    if (stamped.radius) put(root, global, "radius", std::vector<double>{*stamped.radius});
    if (!stamped.citation.empty()) {
      auto joined = stamped.citation.front();
      for (auto k = std::size_t{1}; k < stamped.citation.size(); ++k) {
        joined += "\n" + stamped.citation[k];
      }
      put(root, global, "citation", joined);
    }
    for (const auto& [key, value] : stamped.meta) put(root, global, "meta_" + key, value);
    const auto& grid = *stamped.grid;
    nc::Dimension(root, "latitude", grid.NTheta(), source);
    nc::Dimension(root, "longitude", grid.NPhi(), source);
    put(root, global, "grid_kind", grid.Kind());
    if (grid.Lmax()) put(root, global, "grid_lmax", detail::Integer(*grid.Lmax()));
    auto var = nc::WriteDoubles(root, "latitude", {"latitude"}, detail::Latitudes(grid), source);
    put(root, var, "units", std::string{"degrees_north"});
    put(root, var, "standard_name", std::string{"latitude"});
    put(root, var, "long_name", std::string{"latitude"});
    var = nc::WriteDoubles(root, "longitude", {"longitude"}, detail::Longitudes(grid), source);
    put(root, var, "units", std::string{"degrees_east"});
    put(root, var, "standard_name", std::string{"longitude"});
    put(root, var, "long_name", std::string{"longitude"});
    if (grid.Weights()) {
      var = nc::WriteDoubles(root, "colatitude_weights", {"latitude"}, *grid.Weights(), source);
      put(root, var, "long_name",
          std::string{"colatitude quadrature weights against sin(theta) dtheta"});
      put(root, var, "units", std::string{"1"});
    }
    const auto time_unit = units::ScaledUnit(units::TIME, stamped.scales);
    for (const auto& [name, axis, long_name] :
         {std::tuple{"time", &stamped.time, "time"},
          std::tuple{"instant", &stamped.instant, "observation times"}}) {
      if (!*axis) continue;
      nc::Dimension(root, name, (*axis)->size(), source);
      var = nc::WriteDoubles(root, name, {name}, **axis, source);
      put(root, var, "long_name", std::string{long_name});
      put(root, var, "units", time_unit);
      put(root, var, "dimensions", detail::Exponents(units::TIME));
    }
    auto needs = [&stamped](std::string_view what) {
      return std::ranges::any_of(stamped.variables, [&what](const auto& entry) {
        return detail::TextOf(entry.second.attributes, "time_support") == what;
      });
    };
    if (needs("interval")) nc::Dimension(root, "interval", stamped.time->size() - 1, source);
    const auto needs_pole = std::ranges::any_of(
        stamped.variables, [](const auto& entry) { return entry.first.ends_with("_poles"); });
    if (needs_pole) {
      nc::Dimension(root, "pole", 2, source);
      put(root, global, "poles", std::string{"north south"});
    }
    for (const auto& [name, field] : stamped.variables) {
      auto leading = std::vector<std::string>{};
      std::optional<Dimensions> declared{};
      if (name.ends_with("_poles")) {
        const auto& base = stamped.variables.at(name.substr(0, name.size() - 6));
        const auto support = detail::TextOf(base.attributes, "time_support").value_or("static");
        if (support != "static") leading.push_back(support == "interval" ? "interval"
                                                   : support == "node"   ? "time"
                                                                         : "instant");
        leading.emplace_back("pole");
        declared = detail::DimensionsFrom(detail::NumbersOf(base.attributes, "dimensions"));
      } else {
        const auto support = *detail::TextOf(field.attributes, "time_support");
        if (support != "static") leading.push_back(support == "interval" ? "interval"
                                                   : support == "node"   ? "time"
                                                                         : "instant");
        leading.emplace_back("latitude");
        leading.emplace_back("longitude");
        declared = detail::DimensionsFrom(detail::NumbersOf(field.attributes, "dimensions"));
      }
      var = nc::WriteDoubles(root, name, leading, field.values, source);
      for (const auto& [key, value] : field.attributes) put(root, var, key, value);
      if (!field.attributes.contains("units")) {
        put(root, var, "units", units::ScaledUnit(declared, stamped.scales));
      }
      if (!field.attributes.contains("long_name")) put(root, var, "long_name", name);
    }
    out.Close();
    auto error = std::error_code{};
    std::filesystem::rename(partial, path, error);
    if (error) {
      detail::Fail(source, "", std::format("cannot be put in place: {}", error.message()));
    }
  } catch (...) {
    auto error = std::error_code{};
    std::filesystem::remove(partial, error);
    throw;
  }
}

/** @brief What a field does with a date outside its span. */
enum class Extrapolation {
  Refuse, /**< The date is refused. */
  Hold    /**< The nearest value of the span is served. */
};

/**
 * @brief A surface file as a function of time: each field's values at a
 * date, by the field's own time rule.
 *
 * A `node` field is interpolated between its nodes (`linear`), read from
 * the latest node not after the date (`previous`), or served at its
 * stored nodes alone (`none`); an `interval` field from the interval
 * holding the date, the last closed at its end; an `instant` field at
 * its stored instants alone; a `static` field at any date.  Outside its
 * span a field refuses the date, or holds its nearest value under
 * `Extrapolation::Hold`.  The arithmetic is the Python
 * `SurfaceHistory`'s, so the two read a file identically.
 */
class SurfaceHistory {
 public:
  /**
   * @brief Hold a file as a history.
   * @param file The file, read or built; a built one should pass
   *        `ValidateSurfaceFile` first.
   */
  explicit SurfaceHistory(SurfaceFile file) : file_(std::move(file)) {
    if (!file_.grid) detail::Fail("surface history", "grid", "is missing");
  }

  /** @brief The file's numbers and metadata, as read. */
  [[nodiscard]] const SurfaceFile& File() const noexcept { return file_; }

  /** @brief The fields, polar values not counted among them. */
  [[nodiscard]] std::vector<std::string> Names() const {
    auto out = std::vector<std::string>{};
    for (const auto& [name, var] : file_.variables) {
      if (!name.ends_with("_poles")) out.push_back(name);
    }
    return out;
  }

  /**
   * @brief How a field sits in time, one of `TIME_SUPPORTS`.
   * @param name The field.
   * @throws planetmodel::Error where the file holds no such field.
   */
  [[nodiscard]] std::string Support(std::string_view name) const {
    return *detail::TextOf(VariableOf(name).attributes, "time_support");
  }

  /**
   * @brief A field's values on the grid at a date, (ntheta, nphi) in C
   * order, by its own time rule; NaN stays where the field does not
   * cover a point.
   * @param name The field.
   * @param date The date, in the file's units; a `static` field takes
   *        none.
   * @param extrapolation What to do outside the field's span: refuse
   *        (the default), or hold its nearest value (`linear`,
   *        `previous` and `interval` fields).
   * @throws planetmodel::Error where the file holds no such field, the
   *         date is refused, or a dated field is given no date.
   */
  [[nodiscard]] std::vector<double> Values(
      std::string_view name, std::optional<double> date = {},
      Extrapolation extrapolation = Extrapolation::Refuse) const {
    const auto& var = VariableOf(name);
    const auto size = file_.grid->NTheta() * file_.grid->NPhi();
    const auto rows = Rows(name, var, date, extrapolation);
    if (!rows) return var.values;
    const auto [first, second] = *rows;
    const auto [k0, w0] = first;
    const auto [k1, w1] = second;
    auto out = std::vector<double>(size);
    for (auto n = std::size_t{0}; n < size; ++n) {
      out[n] = w0 * var.values[k0 * size + n] + w1 * var.values[k1 * size + n];
    }
    return out;
  }

 private:
  using Row = std::pair<std::size_t, double>;

  [[nodiscard]] const FileVariable& VariableOf(std::string_view name) const {
    const auto it = file_.variables.find(std::string{name});
    if (name.ends_with("_poles") || it == file_.variables.end()) {
      detail::Fail("surface history", name, "is not a field of the file");
    }
    return it->second;
  }

  /// The leading rows a date reads, with their weights; empty for a
  /// static field, whose values are the answer whole.  The selection and
  /// the weights are the Python `SurfaceHistory._rows`'s.
  [[nodiscard]] std::optional<std::pair<Row, Row>> Rows(
      std::string_view name, const FileVariable& var, std::optional<double> date,
      Extrapolation extrapolation) const {
    auto fail = [&name](std::string_view message) {
      detail::Fail("surface history", name, message);
    };
    const auto support = *detail::TextOf(var.attributes, "time_support");
    if (support == "static") return std::nullopt;
    if (!date) fail(std::format("is a {} field and needs a date", support));
    auto exact = [&fail](const std::vector<double>& axis, double at,
                         std::string_view what) {
      const auto it = std::ranges::find(axis, at);
      if (it == axis.end()) fail(what);
      const auto k = static_cast<std::size_t>(it - axis.begin());
      return std::pair{Row{k, 1.0}, Row{k, 0.0}};
    };
    if (support == "instant") {
      return exact(*file_.instant, *date,
                   "is held at its stored instants alone, and the date is not one");
    }
    const auto& axis = *file_.time;
    const auto rule = *detail::TextOf(var.attributes, "time_interpolation");
    if (support == "node" && rule == "none") {
      return exact(axis, *date,
                   "states no time rule and is held at its stored nodes alone");
    }
    auto at = *date;
    if (at < axis.front() || at > axis.back()) {
      if (extrapolation != Extrapolation::Hold) {
        fail(std::format("is not sampled at {}, outside its span [{}, {}]", at,
                         axis.front(), axis.back()));
      }
      at = std::clamp(at, axis.front(), axis.back());
    }
    const auto after =
        static_cast<std::size_t>(std::ranges::upper_bound(axis, at) - axis.begin());
    if (support == "interval" || rule == "previous") {
      const auto last = support == "interval" ? axis.size() - 2 : axis.size() - 1;
      return std::pair{Row{std::min(after - 1, last), 1.0},
                       Row{std::min(after - 1, last), 0.0}};
    }
    const auto k = std::min(after - 1, axis.size() - 2);
    const auto w = (at - axis[k]) / (axis[k + 1] - axis[k]);
    if (w == 0.0) return std::pair{Row{k, 1.0}, Row{k, 0.0}};
    if (w == 1.0) return std::pair{Row{k + 1, 1.0}, Row{k + 1, 0.0}};
    return std::pair{Row{k, 1.0 - w}, Row{k + 1, w}};
  }

  SurfaceFile file_;
};

}  // namespace planetmodel

#endif  // PLANETMODEL_SURFACEFILE_HPP
