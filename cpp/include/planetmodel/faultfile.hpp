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
 * that an extension is a schema bump.  There is no writer here.
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
#include <utility>
#include <variant>
#include <vector>

#include "planetmodel/fault.hpp"
#include "planetmodel/modelfile.hpp"
#include "planetmodel/netcdf.hpp"
#include "planetmodel/time.hpp"
#include "planetmodel/units.hpp"

namespace planetmodel {

/** @brief The schema every reader checks; bumped for any change to what a
 * file may hold. */
inline constexpr std::string_view FAULT_SCHEMA = "planetmodel.fault/1";

/** @brief The frames a fault file states. */
inline constexpr auto FAULT_FRAMES = std::array<std::string_view, 2>{"geographic", "cartesian"};

/** @brief The time supports of this version. */
inline constexpr auto FAULT_SUPPORTS = std::array<std::string_view, 2>{"static", "node"};

/** @brief The kinds of patch. */
inline constexpr auto PATCH_KINDS = std::array<std::string_view, 2>{"plane", "structured"};

/**
 * @brief A fault file as read: the fault, the patches' names (empty where
 * a patch has none) and the file's metadata.
 */
struct FaultFile {
  std::string schema{};                            /**< `FAULT_SCHEMA`. */
  Fault fault;                                     /**< The fault. */
  std::vector<std::optional<std::string>> names{}; /**< Each patch's name, where it has one. */
  std::string source{};                            /**< The writing code and version. */
  std::string created{};                           /**< When, ISO 8601. */
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
  auto scales = std::array<double, 4>{};
  constexpr auto bases = std::array<std::string_view, 4>{"length", "mass", "time", "temperature"};
  for (auto b = std::size_t{0}; b < 4; ++b) {
    const auto key = std::format("scale_{}", bases[b]);
    const auto value = detail::NumberOf(attrs, key);
    if (!value || !(*value > 0.0)) fail(key, "is missing or not positive");
    scales[b] = *value;
  }
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
         std::format("'slip': time_interpolation '{}' is not one of ('linear', 'previous', "
                     "'none')",
                     rule));
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
  auto citation = std::vector<std::string>{};
  if (const auto text = detail::TextOf(attrs, "citation")) {
    for (auto start = std::size_t{0}; start < text->size();) {
      const auto end = std::min(text->find('\n', start), text->size());
      citation.push_back(text->substr(start, end - start));
      start = end + 1;
    }
  }
  return FaultFile{.schema = *schema,
                   .fault = Fault(std::move(patches), *support, rule, std::move(time),
                                  Scales({.length = scales[0],
                                          .mass = scales[1],
                                          .time = scales[2],
                                          .temperature = scales[3]})),
                   .names = std::move(names),
                   .source = *detail::TextOf(attrs, "source"),
                   .created = *detail::TextOf(attrs, "created"),
                   .title = detail::TextOf(attrs, "title"),
                   .meta = std::move(meta),
                   .citation = std::move(citation)};
}

}  // namespace planetmodel

#endif  // PLANETMODEL_FAULTFILE_HPP
