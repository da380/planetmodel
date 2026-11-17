#ifndef PLANETMODEL_CARTESIAN_SURFACEFILE_HPP
#define PLANETMODEL_CARTESIAN_SURFACEFILE_HPP

/**
 * @file
 * @brief The flat surface file: fields on a horizontal plane of a flat
 * body, static or in time, in one netCDF file of schema
 * `planetmodel.surface/2`.
 *
 * The format is the Python's (`planetmodel.cartesian.surfacefile`, and
 * `docs/source/formats/surface_file.rst` for consumers), the counterpart
 * of the surface file on the sphere (surfacefile.hpp) and the same in
 * everything but the surface: `coordinate_system` is `flat`; the lateral
 * domain is stated as the flat model file states it (`domain`,
 * `domain_bounds`, `domain_padding`) and the grid is the horizontal grid
 * over it, its nodes the coordinate variables `x(x)` and, over a box,
 * `y(y)`; `depth`, where stated, is the depth of the plane in the
 * reference body, and absent the fields lie on the free surface; a field
 * is `name(x, y)` (`name(x)` over a rectangle) after its support's leading
 * dimension, its lateral rule one of the flat rules.  A flat grid has no
 * poles: polar values and every other statement of a spherical file are
 * refused.  `ReadSurfaceFile(path)` reads a file whole and holds it to the
 * structure, its schema first (a spherical surface file is refused by its
 * schema, naming its reader); `WriteSurfaceFile(path, file)` writes one
 * from a code's own fields (`planetmodel::SurfaceField` makes a field's
 * variable), whole or not at all; `SurfaceHistory` serves a field's
 * values at any date by the field's own time rule, and the field there as
 * a function of (x, y) by its lateral rule.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "planetmodel/cartesian/grid.hpp"
#include "planetmodel/cartesian/modelfile.hpp"
#include "planetmodel/cartesian/rules.hpp"
#include "planetmodel/netcdf.hpp"
#include "planetmodel/surfacefile.hpp"
#include "planetmodel/time.hpp"
#include "planetmodel/units.hpp"

namespace planetmodel::cartesian {

/** @brief The schema of a flat surface file; bumped only for an
 * incompatible change. */
inline constexpr std::string_view SURFACE_SCHEMA = SURFACE_SCHEMA_FLAT;

namespace detail {

/// Names the flat surface file gives its own variables, which no field
/// may take.
inline constexpr auto SURFACE_RESERVED =
    std::array<std::string_view, 4>{"x", "y", "time", "instant"};

}  // namespace detail

/**
 * @brief A flat surface file's numbers and metadata, read or built:
 * nothing is interpolated or converted.
 */
struct SurfaceFile {
  std::string schema{SURFACE_SCHEMA};           /**< `SURFACE_SCHEMA`. */
  Scales scales{};                              /**< What one stored unit is in SI. */
  std::optional<Grid> grid{};                   /**< The horizontal grid, over the file's domain. */
  std::optional<std::vector<double>> time{};    /**< The time nodes, strictly increasing. */
  std::optional<std::vector<double>> instant{}; /**< The observation times, strictly increasing. */
  /** @brief The depth of the plane in the reference body, where stated; the
   * fields lie on the free surface where it is not. */
  std::optional<double> depth{};
  std::string source{};                         /**< The writing code and version; required. */
  /** @brief When, ISO 8601; the writer stamps it when empty. */
  std::string created{};
  std::optional<std::string> title{};           /**< A name for the contents, where given. */
  netcdf::Attributes meta{};                    /**< The free `meta_<key>` attributes, by key. */
  std::vector<std::string> citation{};          /**< The works cited, one a line. */
  /** @brief The fields by name, each (nx, ny) in C order after its
   * support's leading dimension (`planetmodel::SurfaceField` makes one). */
  std::map<std::string, FileVariable> variables{};
};

/**
 * @brief Hold a flat surface file to the structure the readers check, the
 * Python's `cartesian.surfacefile.read_surface_file` among them: the
 * schema, a source, a grid, increasing axes, a depth that is finite and
 * not negative, no polar values, and every field of a known support with
 * its values filling it and the grid, known rules, and `none` in time
 * unless it is a `node` field.
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
  if (file.depth && !(std::isfinite(*file.depth) && *file.depth >= 0.0)) {
    fail("depth", "is not a depth, finite and not negative");
  }
  const auto cells = file.grid->Size();
  for (const auto& [name, var] : file.variables) {
    if (std::ranges::contains(detail::SURFACE_RESERVED, std::string_view{name})) {
      fail(name, "is a name the flat surface file keeps for itself");
    }
    if (name.ends_with("_poles")) {
      fail(name, "holds polar values, which a flat grid has none of");
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
    if (!lateral || !std::ranges::contains(LATERAL_RULES, std::string_view{*lateral})) {
      fail(name, "states no known lateral rule");
    }
    auto size = cells;
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
 * @brief Read a flat surface file whole, its schema first: a spherical
 * surface file is refused by its schema, naming its reader.
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
    fail("schema",
         std::format("is {}, this reader reads {}{}", out.schema, SURFACE_SCHEMA,
                     out.schema == ::planetmodel::SURFACE_SCHEMA
                         ? "; a file of that schema is a spherical surface's, read by "
                           "planetmodel::ReadSurfaceFile (planetmodel/surfacefile.hpp)"
                         : ""));
  }
  out.source = detail::TextOf(attrs, "source").value_or("");
  out.created = detail::TextOf(attrs, "created").value_or("");
  if (out.created.empty()) fail("created", "is missing or empty");
  out.title = detail::TextOf(attrs, "title");
  out.scales = detail::ScalesOf(attrs, source);
  if (attrs.contains("depth")) {
    out.depth = detail::NumberOf(attrs, "depth");
    if (!out.depth) fail("depth", "is not a depth, finite and not negative");
  }
  out.citation = detail::CitationOf(attrs);
  for (const auto& [key, value] : attrs) {
    if (key.starts_with("meta_")) out.meta.emplace(key.substr(5), value);
  }
  if (detail::TextOf(attrs, "coordinate_system") != COORDINATE_SYSTEM) {
    fail("coordinate_system",
         std::format("is {}, and a flat surface file states {}",
                     detail::TextOf(attrs, "coordinate_system").value_or("missing or a number"),
                     COORDINATE_SYSTEM));
  }
  detail::RefuseSpherical(root, attrs, source, "");
  const auto domain = detail::ReadDomain(attrs, source);
  out.grid = detail::GridOf(root, domain, source, "");
  const auto axes = detail::AxesOf(domain);
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
    auto var = detail::ReadVariable(root, name, source);
    // the support's leading dimension, then the grid's
    const auto support = detail::TextOf(var.attributes, "time_support").value_or("");
    auto want = std::vector<std::string>{};
    if (support == "node") want.emplace_back("time");
    if (support == "interval") want.emplace_back("interval");
    if (support == "instant") want.emplace_back("instant");
    want.insert(want.end(), axes.begin(), axes.end());
    if (std::ranges::contains(TIME_SUPPORTS, std::string_view{support}) &&
        var.dimensions != want) {
      fail(name, std::format("is {} and does not run over its support and the grid", support));
    }
    out.variables.emplace(name, std::move(var));
  }
  ValidateSurfaceFile(out, source);
  return out;
}

/**
 * @brief Write a flat surface file, whole or not at all, with the
 * permissions a plain open for writing would give it: an existing file's,
 * kept, or the umask's.
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
  if (stamped.created.empty()) stamped.created = detail::CreatedNow();
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
    detail::WriteScales(root, stamped.scales, source);
    if (stamped.depth) put(root, global, "depth", std::vector<double>{*stamped.depth});
    detail::WriteCitation(root, stamped.citation, source);
    for (const auto& [key, value] : stamped.meta) put(root, global, "meta_" + key, value);
    const auto& grid = *stamped.grid;
    put(root, global, "coordinate_system", std::string{COORDINATE_SYSTEM});
    detail::WriteDomain(root, grid.GetDomain(), source);
    detail::WriteGrid(root, grid, units::ScaledUnit(units::LENGTH, stamped.scales), source);
    const auto axes = detail::AxesOf(grid.GetDomain());
    const auto time_unit = units::ScaledUnit(units::TIME, stamped.scales);
    for (const auto& [name, axis, long_name] :
         {std::tuple{"time", &stamped.time, "time"},
          std::tuple{"instant", &stamped.instant, "observation times"}}) {
      if (!*axis) continue;
      nc::Dimension(root, name, (*axis)->size(), source);
      const auto var = nc::WriteDoubles(root, name, {name}, **axis, source);
      put(root, var, "long_name", std::string{long_name});
      put(root, var, "units", time_unit);
      put(root, var, "dimensions", detail::Exponents(units::TIME));
    }
    const auto needs_interval = std::ranges::any_of(stamped.variables, [](const auto& entry) {
      return detail::TextOf(entry.second.attributes, "time_support") == "interval";
    });
    if (needs_interval) nc::Dimension(root, "interval", stamped.time->size() - 1, source);
    for (const auto& [name, field] : stamped.variables) {
      const auto support = *detail::TextOf(field.attributes, "time_support");
      auto dims = std::vector<std::string>{};
      if (support != "static") {
        dims.push_back(support == "interval" ? "interval" : support == "node" ? "time" : "instant");
      }
      dims.insert(dims.end(), axes.begin(), axes.end());
      const auto var = nc::WriteDoubles(root, name, dims, field.values, source);
      for (const auto& [key, value] : field.attributes) put(root, var, key, value);
      if (!field.attributes.contains("units")) {
        put(root, var, "units",
            units::ScaledUnit(
                detail::DimensionsFrom(detail::NumbersOf(field.attributes, "dimensions")),
                stamped.scales));
      }
      if (!field.attributes.contains("long_name")) put(root, var, "long_name", name);
    }
    out.Close();
    detail::PutInPlace(partial, path, source);
  } catch (...) {
    auto error = std::error_code{};
    std::filesystem::remove(partial, error);
    throw;
  }
}

/**
 * @brief A flat surface file as a function of time: each field's values at
 * a date, by the field's own time rule, and the field there as a function
 * of (x, y) by its lateral rule.
 *
 * The time rules, their tolerance and their extrapolation are the
 * spherical surface file's (`planetmodel::SurfaceHistory`, `TimeRowsAt`),
 * and the Python's, so the two languages read a file identically.
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

  /** @brief The horizontal grid every field is on. */
  [[nodiscard]] const Grid& GetGrid() const noexcept { return *file_.grid; }

  /** @brief The fields. */
  [[nodiscard]] std::vector<std::string> Names() const {
    auto out = std::vector<std::string>{};
    for (const auto& [name, var] : file_.variables) out.push_back(name);
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
   * @brief A field's values on the grid at a date, (nx, ny) in C order, by
   * its own time rule; NaN stays where the field does not cover a point.
   * @param name The field.
   * @param date The date, in the file's units; a `static` field takes
   *        none.
   * @param extrapolation What to do outside the field's span: refuse
   *        (the default), or hold its nearest value (`linear`,
   *        `previous` and `interval` fields).
   * @param time_rtol The date's tolerance on a stored time and on an end
   *        of the span, relative to the scale of the axis (`TimeRowsAt`);
   *        zero matches exactly.
   * @throws planetmodel::Error where the file holds no such field, the
   *         date is refused, or a dated field is given no date.
   */
  [[nodiscard]] std::vector<double> Values(std::string_view name, std::optional<double> date = {},
                                           Extrapolation extrapolation = Extrapolation::Refuse,
                                           double time_rtol = TIME_RTOL) const {
    const auto& var = VariableOf(name);
    const auto size = file_.grid->Size();
    const auto rows = TimeRowsAt(*detail::TextOf(var.attributes, "time_support"),
                                 *detail::TextOf(var.attributes, "time_interpolation"), date,
                                 file_.time, file_.instant, extrapolation, "surface history",
                                 name, time_rtol);
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

  /**
   * @brief A field at a date as a function of (x, y), by its stated
   * lateral rule on the file's grid.
   * @param name The field.
   * @param date The date, as `Values` takes it.
   * @param extrapolation What to do outside the field's span, as `Values`
   *        takes it.
   * @param time_rtol The date's tolerance, as `Values` takes it.
   * @return Its interpolant: `Value(x, y)` and `Gradient(x, y)`.
   * @throws planetmodel::Error as `Values` does, or if the field's lateral
   *         rule is `fourier`, which the C++ does not read, or its values
   *         at the date hold a NaN, which no rule reads.
   */
  [[nodiscard]] LateralInterpolant Function(std::string_view name,
                                            std::optional<double> date = {},
                                            Extrapolation extrapolation = Extrapolation::Refuse,
                                            double time_rtol = TIME_RTOL) const {
    const auto rule = LateralRuleNamed(
        *detail::TextOf(VariableOf(name).attributes, "lateral_interpolation"),
        "surface history", name);
    return LateralInterpolant(*file_.grid, Values(name, date, extrapolation, time_rtol), rule,
                              name);
  }

 private:
  [[nodiscard]] const FileVariable& VariableOf(std::string_view name) const {
    const auto it = file_.variables.find(std::string{name});
    if (it == file_.variables.end()) {
      detail::Fail("surface history", name, "is not a field of the file");
    }
    return it->second;
  }

  SurfaceFile file_;
};

}  // namespace planetmodel::cartesian

#endif  // PLANETMODEL_CARTESIAN_SURFACEFILE_HPP
