#ifndef PLANETMODEL_FAULTFILE_HPP
#define PLANETMODEL_FAULTFILE_HPP

/**
 * @file
 * @brief The fault file: a fault, its patches of surface and the slip on
 * them in time, in one netCDF file.
 *
 * The format is the Python's (`planetmodel.faultfile`, and
 * `docs/source/formats/fault_file.rst` for consumers): the frame, the
 * time axis and the scales at the root, one group a patch, `patch_<k>`,
 * holding a plane's parameters as attributes or a structured patch's
 * control positions as variables, and the slip's fault components on its
 * own grid.  `ReadFaultFile(path)` reads a file whole into a `Fault`,
 * making the Python reader's checks in its order and refusing with its
 * places and words: anything the schema does not name is refused, so
 * that an extension is a schema bump.  `WriteFaultFile(path, file)`
 * writes one as the Python's `write_fault` does, whole or not at all,
 * with the permissions a plain open for writing would give the file.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "planetmodel/fault.hpp"
#include "planetmodel/modelfile.hpp"
#include "planetmodel/netcdf.hpp"
#include "planetmodel/time.hpp"
#include "planetmodel/units.hpp"

/** @brief Defined where this header provides `WriteFaultFile` beside
 * `ReadFaultFile`. */
#define PLANETMODEL_HAS_FAULT_FILE_WRITER 1

namespace planetmodel {

/** @brief The schema every reader checks; bumped for any change to what a
 * file may hold. */
inline constexpr std::string_view FAULT_SCHEMA = "planetmodel.fault/1";

/** @brief The frames a fault file states. */
inline constexpr auto FAULT_FRAMES = std::array<std::string_view, 2>{"geographic", "cartesian"};

/** @brief The kinds of patch. */
inline constexpr auto PATCH_KINDS = std::array<std::string_view, 2>{"plane", "structured"};

/**
 * @brief A fault file as read, or to write: the fault, the patches' names
 * (empty where a patch has none) and the file's metadata.
 */
struct FaultFile {
  std::string schema{}; /**< `FAULT_SCHEMA`; empty in one built to write. */
  Fault fault;          /**< The fault. */
  std::vector<std::optional<std::string>> names{}; /**< Each patch's name, where it has one. */
  /** @brief The writing code and version, which a file read states; the
   * writer stamps this library and its version when empty. */
  std::string source{};
  std::string created{};                           /**< When, ISO 8601; stamped where empty. */
  std::optional<std::string> title{};              /**< A name for the contents, where given. */
  netcdf::Attributes meta{};                       /**< The free `meta_<key>` attributes, by key. */
  std::vector<std::string> citation{};             /**< The works cited, one a line. */
};

namespace detail {

/// The global attributes the format names, besides `meta_<key>`.
inline constexpr auto FAULT_ROOT_ATTRIBUTES = std::array<std::string_view, 16>{"schema",
                                                                               "Conventions",
                                                                               "source",
                                                                               "created",
                                                                               "title",
                                                                               "citation",
                                                                               "scale_length",
                                                                               "scale_mass",
                                                                               "scale_time",
                                                                               "scale_temperature",
                                                                               "frame",
                                                                               "reference_radius",
                                                                               "reference_surface",
                                                                               "time_support",
                                                                               "time_interpolation",
                                                                               "patches"};

/// A structured patch's position variables in a frame.
inline std::array<std::string_view, 3> FaultPositions(std::string_view frame) {
  if (frame == "geographic") return {"latitude", "longitude", "radius"};
  return {"x", "y", "z"};
}

/// The attributes a variable of the format may carry.
inline std::vector<std::string_view> FaultVariableAttributes(std::string_view name) {
  if (name == "latitude" || name == "longitude") {
    return {"long_name", "units", "standard_name", "_FillValue"};
  }
  if (name == "slip") {
    return {"long_name", "units", "dimensions", "rule", "components", "_FillValue"};
  }
  return {"long_name", "units", "dimensions", "_FillValue"};
}

/// Names as the messages list them: `a, b, c`.
template <typename Names>
std::string Listed(const Names& names) {
  auto out = std::string{};
  for (const auto& name : names) {
    if (!out.empty()) out += ", ";
    out += name;
  }
  return out;
}

/// Names as the Python writes a tuple of them: `('a', 'b', 'c')`, one of
/// them `('a',)`.
template <typename Names>
std::string TupleRepr(const Names& names) {
  auto out = std::string{"("};
  for (const auto& name : names) {
    if (out.size() > 1) out += ", ";
    out += std::format("'{}'", name);
  }
  return out + (std::size(names) == 1 ? ",)" : ")");
}

/// The reader of one file: its source and the refusals in its words.
class FaultReader {
 public:
  explicit FaultReader(std::string source) : source_(std::move(source)) {}

  [[noreturn]] void Fail(std::string_view where, std::string_view message) const {
    throw Error(source_, where, message);
  }

  /// Refuse the first of `present` that is not among `named`.
  template <typename Named>
  void Unnamed(std::string_view where, const std::vector<std::string>& present, const Named& named,
               std::string_view what) const {
    for (const auto& key : present) {
      if (!std::ranges::contains(named, std::string_view{key})) {
        Fail(std::format("{}{}", where, key),
             std::format("is not {}: {} does not name it", what, FAULT_SCHEMA));
      }
    }
  }

  [[nodiscard]] const std::string& Source() const noexcept { return source_; }

 private:
  std::string source_;
};

/// An attribute's one integer, or empty where it is not one.
inline std::optional<long long> CountOf(const netcdf::Attributes& attrs, std::string_view key) {
  const auto it = attrs.find(std::string{key});
  if (it == attrs.end()) return std::nullopt;
  const auto* integers = std::get_if<std::vector<long long>>(&it->second);
  if (integers == nullptr || integers->size() != 1) return std::nullopt;
  return integers->front();
}

/// Whether a position has a latitude in [-90, 90] and a positive radius.
inline bool GeographicPosition(const Vector3& c) { return std::abs(c[0]) <= 90.0 && c[2] > 0.0; }

/// A plane's parameters, checked.
inline PlaneParameters PlaneParametersOf(const FaultReader& reader, const std::string& where,
                                         const netcdf::Attributes& attrs, std::string_view frame) {
  auto out = PlaneParameters{};
  const auto centroid = NumbersOf(attrs, "centroid");
  if (!centroid || centroid->size() != 3 ||
      !std::ranges::all_of(*centroid, [](double x) { return std::isfinite(x); })) {
    reader.Fail(where + ".centroid", "is missing or not three finite numbers");
  }
  out.centroid = {(*centroid)[0], (*centroid)[1], (*centroid)[2]};
  if (frame == "geographic" && !GeographicPosition(out.centroid)) {
    reader.Fail(where + ".centroid",
                "has a latitude outside [-90, 90] or a radius that is not positive");
  }
  const auto strike = NumberOf(attrs, "strike");
  if (!strike || !std::isfinite(*strike)) {
    reader.Fail(where + ".strike", "is missing or not a finite number");
  }
  const auto dip = NumberOf(attrs, "dip");
  if (!dip || !(*dip >= 0.0 && *dip <= 90.0)) {
    reader.Fail(where + ".dip", "is missing or not in [0, 90] degrees");
  }
  out.strike = *strike;
  out.dip = *dip;
  for (const auto key : {"length", "width"}) {
    const auto value = NumberOf(attrs, key);
    if (!value || !(std::isfinite(*value) && *value > 0.0)) {
      reader.Fail(std::format("{}.{}", where, key), "is missing or not a positive number");
    }
    (std::string_view{key} == "length" ? out.length : out.width) = *value;
  }
  return out;
}

/// A structured patch's control positions, checked.
inline ControlGrid ControlGridOf(const FaultReader& reader, int group, const std::string& where,
                                 std::string_view frame, const std::string& interpolation) {
  namespace nc = netcdf;
  const auto& source = reader.Source();
  auto grid = ControlGrid{.interpolation = interpolation};
  grid.n_u = nc::DimensionLength(group, "nu", source).value_or(0);
  grid.n_v = nc::DimensionLength(group, "nv", source).value_or(0);
  const auto least = MinNodes(interpolation);
  if (std::min(grid.n_u, grid.n_v) < least) {
    reader.Fail(where, std::format("has {} by {} control nodes, and the {} rule needs at "
                                   "least {} each way",
                                   grid.n_u, grid.n_v, interpolation, least));
  }
  auto columns = std::vector<std::vector<double>>{};
  for (const auto name : FaultPositions(frame)) {
    const auto at = std::format("{}/{}", where, name);
    if (!nc::HasVariable(group, name)) reader.Fail(at, "is missing");
    if (nc::DimensionNames(group, name, source) != std::vector<std::string>{"nu", "nv"}) {
      reader.Fail(at, "does not run over (nu, nv)");
    }
    reader.Unnamed(at + ".", nc::AttributeNames(group, nc::Variable(group, name, source), source),
                   FaultVariableAttributes(name), "an attribute of a control position");
    columns.push_back(nc::ReadDoubles(group, name, source).values);
  }
  grid.positions.resize(grid.n_u * grid.n_v);
  for (auto k = std::size_t{0}; k < grid.positions.size(); ++k) {
    grid.positions[k] = {columns[0][k], columns[1][k], columns[2][k]};
  }
  for (const auto& p : grid.positions) {
    if (!std::ranges::all_of(p, [](double x) { return std::isfinite(x); })) {
      reader.Fail(where, "has control positions that are not finite");
    }
  }
  if (frame == "geographic" && !std::ranges::all_of(grid.positions, [](const Vector3& p) {
        return GeographicPosition(p);
      })) {
    reader.Fail(where,
                "has a control position with a latitude outside [-90, 90] or a "
                "radius that is not positive");
  }
  return grid;
}

/// A patch's slip, checked.
inline FaultSlip SlipOf(const FaultReader& reader, int group, const std::string& where,
                        const std::vector<std::string>& leading, std::size_t nslices) {
  namespace nc = netcdf;
  const auto& source = reader.Source();
  const auto at = where + "/slip";
  if (!nc::HasVariable(group, "slip")) reader.Fail(at, "is missing");
  auto want = leading;
  for (const auto name : {"su", "sv", "component"}) want.emplace_back(name);
  if (nc::DimensionNames(group, "slip", source) != want) {
    reader.Fail(at, std::format("does not run over ({})", Listed(want)));
  }
  const auto components = nc::DimensionLength(group, "component", source).value_or(0);
  if (components != 3) reader.Fail(at, std::format("has {} components, not 3", components));
  const auto var = nc::Variable(group, "slip", source);
  reader.Unnamed(at + ".", nc::AttributeNames(group, var, source), FaultVariableAttributes("slip"),
                 "an attribute of the slip");
  const auto attrs = nc::ReadAttributes(group, var, source);
  auto spaced = std::string{};
  for (const auto name : SLIP_COMPONENTS) {
    if (!spaced.empty()) spaced += " ";
    spaced += name;
  }
  if (TextOf(attrs, "components") != spaced) {
    reader.Fail(at + ".components", std::format("is missing or not \"{}\"", spaced));
  }
  const auto rule = TextOf(attrs, "rule");
  if (!rule || !std::ranges::contains(SLIP_RULES, std::string_view{*rule})) {
    reader.Fail(at + ".rule", std::format("is missing or not one of {}", Listed(SLIP_RULES)));
  }
  const auto n_u = *nc::DimensionLength(group, "su", source);
  const auto n_v = *nc::DimensionLength(group, "sv", source);
  const auto least = MinNodes(*rule);
  if (std::min(n_u, n_v) < least) {
    reader.Fail(at, std::format("has {} by {} values, and the {} rule needs at least {} {} "
                                "each way",
                                n_u, n_v, *rule, least, *rule == "cell" ? "cells" : "nodes"));
  }
  auto values = nc::ReadDoubles(group, "slip", source).values;
  if (!std::ranges::all_of(values, [](double x) { return std::isfinite(x); })) {
    reader.Fail(at, "has values that are not finite: slip is defined everywhere on its patch");
  }
  return FaultSlip(std::move(values), nslices, n_u, n_v, *rule);
}

/// The frame the root states, its reference checked and the other frame's
/// refused.
inline FaultFrame FaultFrameOf(const FaultReader& reader, std::string_view name,
                               const netcdf::Attributes& attrs) {
  const auto key = name == "geographic" ? "reference_radius" : "reference_surface";
  const auto other = name == "geographic" ? "reference_surface" : "reference_radius";
  if (attrs.contains(other)) {
    reader.Fail(other, std::format("is not an attribute of a {} fault file", name));
  }
  const auto value = NumberOf(attrs, key);
  if (name == "geographic") {
    if (!value || !(std::isfinite(*value) && *value > 0.0)) {
      reader.Fail(key, "is missing or not a positive number");
    }
    return FaultFrame::Geographic(*value);
  }
  if (!value || !std::isfinite(*value)) reader.Fail(key, "is missing or not a finite number");
  return FaultFrame::Cartesian(*value);
}

/// One patch group read and checked: the patch and its name.
inline std::pair<FaultPatch, std::optional<std::string>> FaultPatchOf(
    const FaultReader& reader, int group, const std::string& where, const FaultFrame& frame,
    const std::vector<std::string>& leading, std::size_t nslices) {
  namespace nc = netcdf;
  const auto& source = reader.Source();
  reader.Unnamed(where + "/", nc::GroupNames(group, source), std::array<std::string_view, 0>{},
                 "a group of a patch");
  const auto attrs = nc::ReadAttributes(group, nc::GLOBAL, source);
  const auto kind = TextOf(attrs, "kind");
  if (!kind || !std::ranges::contains(PATCH_KINDS, std::string_view{*kind})) {
    reader.Fail(where + ".kind", std::format("is missing or not one of {}", Listed(PATCH_KINDS)));
  }
  const auto plane = *kind == "plane";
  const auto named = plane ? std::vector<std::string_view>{"kind", "name",   "centroid", "strike",
                                                           "dip",  "length", "width"}
                           : std::vector<std::string_view>{"kind", "name", "interpolation"};
  reader.Unnamed(where + ".", nc::AttributeNames(group, nc::GLOBAL, source), named,
                 std::format("an attribute of a {} patch", *kind));
  auto name = std::optional<std::string>{};
  if (attrs.contains("name")) {
    name = TextOf(attrs, "name");
    if (!name) reader.Fail(where + ".name", "is not text");
  }
  auto dims = std::vector<std::string_view>{"su", "sv", "component"};
  if (!plane) dims.insert(dims.end(), {"nu", "nv"});
  reader.Unnamed(where + "/", nc::GroupDimensionNames(group, source), dims,
                 std::format("a dimension of a {} patch", *kind));
  auto variables = std::vector<std::string_view>{"slip"};
  if (!plane) {
    for (const auto position : FaultPositions(frame.Name())) variables.push_back(position);
  }
  reader.Unnamed(where + "/", nc::VariableNames(group, source), variables,
                 std::format("a variable of a {} patch", *kind));
  auto parameters = PlaneParameters{};
  auto grid = ControlGrid{};
  if (plane) {
    parameters = PlaneParametersOf(reader, where, attrs, frame.Name());
  } else {
    const auto interpolation = TextOf(attrs, "interpolation");
    if (!interpolation ||
        !std::ranges::contains(FAULT_INTERPOLATIONS, std::string_view{*interpolation})) {
      reader.Fail(where + ".interpolation",
                  std::format("is missing or not one of {}", Listed(FAULT_INTERPOLATIONS)));
    }
    grid = ControlGridOf(reader, group, where, frame.Name(), *interpolation);
  }
  auto slip = SlipOf(reader, group, where, leading, nslices);
  if (plane) return {FaultPatch::Plane(frame, parameters, std::move(slip)), name};
  try {
    return {FaultPatch::Structured(frame, std::move(grid), std::move(slip)), name};
  } catch (const Error&) {
    reader.Fail(where, "is degenerate: its area element is not positive at every control node");
  }
}

}  // namespace detail

/**
 * @brief Read a fault file whole, every check of the format made first,
 * in the Python reader's order.
 * @param path The file.
 * @return The fault and the file's metadata.
 * @throws planetmodel::Error naming the place of the first fault: the
 *         path, then the attribute (`patch_<k>.<name>` within a patch),
 *         the variable (`patch_<k>/<name>`) or its attribute
 *         (`patch_<k>/<name>.<attribute>`).
 */
[[nodiscard]] inline FaultFile ReadFaultFile(const std::filesystem::path& path) {
  namespace nc = netcdf;
  const auto file = nc::File(path);
  const auto root = file.Id();
  const auto reader = detail::FaultReader(file.Path());
  const auto& source = reader.Source();
  auto fail = [&reader](std::string_view where, std::string_view message) {
    reader.Fail(where, message);
  };
  const auto attrs = nc::ReadAttributes(root, nc::GLOBAL, source);
  const auto schema = detail::TextOf(attrs, "schema");
  if (!schema) fail("schema", "is missing: not a planetmodel fault file");
  if (*schema != FAULT_SCHEMA) {
    fail("schema", std::format("is {}, this reader reads {}", *schema, FAULT_SCHEMA));
  }
  auto plain = std::vector<std::string>{};
  for (auto& key : nc::AttributeNames(root, nc::GLOBAL, source)) {
    if (!key.starts_with("meta_")) plain.push_back(std::move(key));
  }
  reader.Unnamed("", plain, detail::FAULT_ROOT_ATTRIBUTES, "an attribute of a fault file");
  for (const auto key : {"source", "created"}) {
    const auto text = detail::TextOf(attrs, key);
    if (!text || text->find_first_not_of(" \t\n\r") == std::string::npos) {
      fail(key, "is missing or empty");
    }
  }
  for (const auto key : {"title", "citation"}) {
    if (attrs.contains(key) && !detail::TextOf(attrs, key)) fail(key, "is not text");
  }
  const auto scales = detail::ScalesOf(attrs, source);
  const auto frame_name = detail::TextOf(attrs, "frame");
  if (!frame_name || !std::ranges::contains(FAULT_FRAMES, std::string_view{*frame_name})) {
    fail("frame", std::format("is missing or not one of {}", detail::Listed(FAULT_FRAMES)));
  }
  const auto frame = detail::FaultFrameOf(reader, *frame_name, attrs);
  const auto support = detail::TextOf(attrs, "time_support");
  if (support == "interval" || support == "instant") {
    fail("time_support", std::format("is {}, which is not part of {}: its supports are {}",
                                     *support, FAULT_SCHEMA, detail::Listed(FAULT_SUPPORTS)));
  }
  if (!support || !std::ranges::contains(FAULT_SUPPORTS, std::string_view{*support})) {
    fail("time_support",
         std::format("is missing or not one of {}", detail::Listed(FAULT_SUPPORTS)));
  }
  const auto rule = detail::TextOf(attrs, "time_interpolation").value_or("");
  if (!std::ranges::contains(TIME_METHODS, std::string_view{rule})) {
    fail("time_interpolation",
         std::format("'slip': time_interpolation '{}' is not one of {}", rule,
                     detail::TupleRepr(TIME_METHODS)));
  }
  if (*support != "node" && rule != "none") {
    fail("time_interpolation", std::format("'slip': a '{}' field is not sampled between times; its "
                                           "time_interpolation is 'none'",
                                           *support));
  }
  const auto count = detail::CountOf(attrs, "patches");
  if (!count || *count < 1) fail("patches", "is missing or not a positive integer");
  const auto leading =
      *support == "node" ? std::vector<std::string>{"time"} : std::vector<std::string>{};
  reader.Unnamed("", nc::GroupDimensionNames(root, source), leading,
                 std::format("a dimension of a {} fault file", *support));
  reader.Unnamed("", nc::VariableNames(root, source), leading,
                 std::format("a variable of a {} fault file", *support));
  auto time = std::optional<std::vector<double>>{};
  if (*support == "node") {
    if (!nc::HasVariable(root, "time")) fail("time", "is missing: a node slip needs its time axis");
    if (nc::DimensionNames(root, "time", source) != std::vector<std::string>{"time"}) {
      fail("time", "does not run over its own dimension");
    }
    reader.Unnamed("time.", nc::AttributeNames(root, nc::Variable(root, "time", source), source),
                   detail::FaultVariableAttributes("time"), "an attribute of the time axis");
    time = nc::ReadDoubles(root, "time", source).values;
    auto last = -std::numeric_limits<double>::infinity();
    for (const auto t : *time) {
      if (!(std::isfinite(t) && t > last)) fail("time", "is not finite and strictly increasing");
      last = t;
    }
  }
  const auto npatches = static_cast<std::size_t>(*count);
  auto expected = std::vector<std::string>{};
  for (auto k = std::size_t{0}; k < npatches; ++k) expected.push_back(std::format("patch_{}", k));
  reader.Unnamed("", nc::GroupNames(root, source), expected,
                 std::format("one of the {} patches", npatches));
  const auto groups = nc::GroupNames(root, source);
  for (const auto& name : expected) {
    if (!std::ranges::contains(groups, name)) {
      fail(name, std::format("is missing: the file states {} patches", npatches));
    }
  }
  auto patches = std::vector<FaultPatch>{};
  auto names = std::vector<std::optional<std::string>>{};
  const auto nslices = time ? time->size() : std::size_t{1};
  for (const auto& name : expected) {
    auto [patch, label] =
        detail::FaultPatchOf(reader, nc::Group(root, name, source), name, frame, leading, nslices);
    patches.push_back(std::move(patch));
    names.push_back(std::move(label));
  }
  auto meta = nc::Attributes{};
  for (const auto& [key, value] : attrs) {
    if (key.starts_with("meta_")) meta.emplace(key.substr(5), value);
  }
  auto citation = detail::CitationOf(attrs);
  return FaultFile{.schema = *schema,
                   .fault = Fault(std::move(patches), *TimeSupportNamed(*support),
                                  *TimeRuleNamed(rule), std::move(time), scales),
                   .names = std::move(names),
                   .source = *detail::TextOf(attrs, "source"),
                   .created = *detail::TextOf(attrs, "created"),
                   .title = detail::TextOf(attrs, "title"),
                   .meta = std::move(meta),
                   .citation = std::move(citation)};
}

namespace detail {

/// The slip variable's `long_name`.
inline constexpr std::string_view FAULT_SLIP_MEANING =
    "displacement of the hanging wall relative to the footwall";

/// The attributes of a structured patch's position variable.
inline std::vector<std::pair<std::string, netcdf::Attribute>> FaultPositionAttributes(
    std::string_view name, const std::string& length) {
  using A = std::vector<std::pair<std::string, netcdf::Attribute>>;
  if (name == "latitude") {
    return A{{"long_name", std::string{"latitude of the control positions"}},
             {"units", std::string{"degrees_north"}},
             {"standard_name", std::string{"latitude"}}};
  }
  if (name == "longitude") {
    return A{{"long_name", std::string{"longitude of the control positions"}},
             {"units", std::string{"degrees_east"}},
             {"standard_name", std::string{"longitude"}}};
  }
  const auto meaning = name == "radius" ? std::string{"radius"}
                       : name == "x"    ? std::string{"x (east)"}
                       : name == "y"    ? std::string{"y (north)"}
                                        : std::string{"z (up)"};
  return A{{"long_name", meaning + " of the control positions"},
           {"units", length},
           {"dimensions", Exponents(units::LENGTH)}};
}

/// The checks the Python's `write_fault` makes before anything is written.
inline void CheckFaultToWrite(const FaultFile& file, const std::string& source) {
  if (!file.schema.empty() && file.schema != FAULT_SCHEMA) {
    Fail(source, "schema", std::format("is {}; this writer writes {}", file.schema, FAULT_SCHEMA));
  }
  if (!file.source.empty() && file.source.find_first_not_of(" \t\n\r") == std::string::npos) {
    Fail(source, "source", "names the writing code and version, and is blank");
  }
  const auto npatches = file.fault.Patches().size();
  if (!file.names.empty() && file.names.size() != npatches) {
    Fail(source, "names",
         std::format("gives {} names for {} patches", file.names.size(), npatches));
  }
  for (const auto& [key, value] : file.meta) {
    const auto* text = std::get_if<std::string>(&value);
    const auto count = text ? std::size_t{1}
                       : std::holds_alternative<std::vector<double>>(value)
                           ? std::get<std::vector<double>>(value).size()
                           : std::get<std::vector<long long>>(value).size();
    if (key.empty() || count != 1) {
      Fail(source, "meta_" + key, "a free attribute is named, and is text or one number");
    }
  }
}

}  // namespace detail

/**
 * @brief Write a fault file, whole or not at all, as the Python's
 * `write_fault` writes it: the frame, the time axis and the scales at the
 * root, one group a patch with a plane's parameters or a structured
 * patch's control positions, and each patch's slip in fault components
 * on its own grid, every number in the fault's scales.
 *
 * The file is written under a name of its own beside `path` and put in
 * place when whole, so that a failed write leaves no file and an existing
 * file at `path` stands until it is replaced; it has the permissions a
 * plain open for writing would give it, an existing file's kept.
 *
 * @param path The file to write; one there is replaced.
 * @param file The fault and the file's metadata: `schema` empty or
 *        `FAULT_SCHEMA`; `source`, the writing code and version, this
 *        library and its version where it is empty, as the model file's
 *        is; `names` empty (no patch named) or one a patch, each
 *        written where it is given; an empty `created` stamped with the
 *        writing time in UTC; `title` written where given and not empty,
 *        `citation` one work a line (each trimmed of white space at its
 *        ends, an empty one or a repeat dropped, a work of more than one
 *        line refused), and `meta` as `meta_<key>` attributes, each text
 *        or one number.
 * @throws planetmodel::Error naming what is wrong, before anything is
 *         written, or what kept the file from being written.
 */
inline void WriteFaultFile(const std::filesystem::path& path, const FaultFile& file) {
  namespace nc = netcdf;
  const auto source = path.string();
  detail::CheckFaultToWrite(file, source);
  const auto citation = detail::Cited(file.citation, source);
  const auto& fault = file.fault;
  const auto& scales = fault.GetScales();
  const auto& frame = fault.Frame();
  const auto length = units::ScaledUnit(units::LENGTH, scales);
  const auto partial = detail::PartialName(path);
  try {
    auto out = nc::File::Create(partial);
    const auto root = out.Id();
    auto put = [&source](int group, int var, std::string_view key, const nc::Attribute& value) {
      nc::WriteAttribute(group, var, key, value, source);
    };
    auto real = [](double x) { return nc::Attribute{std::vector<double>{x}}; };
    constexpr auto global = nc::GLOBAL;
    put(root, global, "schema", std::string{FAULT_SCHEMA});
    put(root, global, "Conventions", std::string{MODEL_CONVENTIONS});
    if (file.title && !file.title->empty()) put(root, global, "title", *file.title);
    put(root, global, "source", detail::StampedSource(file.source));
    put(root, global, "created", file.created.empty() ? detail::CreatedNow() : file.created);
    put(root, global, "scale_length", real(scales.Length()));
    put(root, global, "scale_mass", real(scales.Mass()));
    put(root, global, "scale_time", real(scales.Time()));
    put(root, global, "scale_temperature", real(scales.Temperature()));
    detail::WriteCitation(root, citation, source);
    for (const auto& [key, value] : file.meta) put(root, global, "meta_" + key, value);
    put(root, global, "frame", std::string{frame.Name()});
    put(root, global,
        frame.GetKind() == FaultFrame::Kind::Geographic ? "reference_radius" : "reference_surface",
        real(frame.Reference()));
    put(root, global, "time_support", std::string{fault.SupportName()});
    put(root, global, "time_interpolation", std::string{fault.TimeInterpolationName()});
    put(root, global, "patches", detail::Integer(static_cast<long long>(fault.Patches().size())));
    auto leading = std::vector<std::string>{};
    if (fault.Support() == TimeSupport::Node) {
      const auto& time = *fault.Time();
      nc::Dimension(root, "time", time.size(), source);
      const auto var = nc::WriteDoubles(root, "time", {"time"}, time, source);
      put(root, var, "long_name", std::string{"time"});
      put(root, var, "units", units::ScaledUnit(units::TIME, scales));
      put(root, var, "dimensions", detail::Exponents(units::TIME));
      leading.emplace_back("time");
    }
    for (auto k = std::size_t{0}; k < fault.Patches().size(); ++k) {
      const auto& patch = fault.Patches()[k];
      const auto group = nc::CreateGroup(root, std::format("patch_{}", k), source);
      put(group, global, "kind", std::string{patch.KindName()});
      if (k < file.names.size() && file.names[k]) put(group, global, "name", *file.names[k]);
      if (patch.GetKind() == FaultPatch::Kind::Plane) {
        const auto& plane = patch.PlaneOf();
        put(group, global, "centroid",
            nc::Attribute{
                std::vector<double>{plane.centroid[0], plane.centroid[1], plane.centroid[2]}});
        put(group, global, "strike", real(plane.strike));
        put(group, global, "dip", real(plane.dip));
        put(group, global, "length", real(plane.length));
        put(group, global, "width", real(plane.width));
      } else {
        const auto& grid = patch.Grid();
        nc::Dimension(group, "nu", grid.n_u, source);
        nc::Dimension(group, "nv", grid.n_v, source);
        put(group, global, "interpolation", grid.interpolation);
        const auto names = detail::FaultPositions(frame.Name());
        for (auto c = std::size_t{0}; c < 3; ++c) {
          auto column = std::vector<double>{};
          column.reserve(grid.positions.size());
          for (const auto& position : grid.positions) column.push_back(position[c]);
          const auto var = nc::WriteDoubles(group, names[c], {"nu", "nv"}, column, source);
          for (const auto& [key, value] : detail::FaultPositionAttributes(names[c], length)) {
            put(group, var, key, value);
          }
        }
      }
      const auto& slip = patch.Slip();
      nc::Dimension(group, "su", slip.NU(), source);
      nc::Dimension(group, "sv", slip.NV(), source);
      nc::Dimension(group, "component", 3, source);
      auto dims = leading;
      dims.insert(dims.end(), {"su", "sv", "component"});
      const auto var = nc::WriteDoubles(group, "slip", dims, slip.Values(), source);
      put(group, var, "long_name", std::string{detail::FAULT_SLIP_MEANING});
      put(group, var, "units", length);
      put(group, var, "dimensions", detail::Exponents(units::LENGTH));
      put(group, var, "rule", slip.Rule());
      auto components = std::string{};
      for (const auto name : SLIP_COMPONENTS) {
        if (!components.empty()) components += " ";
        components += name;
      }
      put(group, var, "components", components);
    }
    out.Close();
    detail::PutInPlace(partial, path, source);
  } catch (...) {
    auto error = std::error_code{};
    std::filesystem::remove(partial, error);
    throw;
  }
}

}  // namespace planetmodel

#endif  // PLANETMODEL_FAULTFILE_HPP
