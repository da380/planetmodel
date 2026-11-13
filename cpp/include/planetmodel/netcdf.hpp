#ifndef PLANETMODEL_NETCDF_HPP
#define PLANETMODEL_NETCDF_HPP

/**
 * @file
 * @brief netCDF-C, over which the model file is read and written.
 *
 * It is there when the library was configured with
 * PLANETMODEL_WITH_NETCDF, the default; a code built without it cannot
 * include this header, nor any that includes it.  `netcdf::File` opens a
 * file for reading, or creates one for writing, and closes it when it
 * goes; a group of it, the file itself included, is an `int` id as
 * netCDF-C has it.  The readers and writers take a group, a name and the
 * source the errors name, and refuse with an `Error` naming the variable
 * or attribute.  Every real variable is written with NaN as its fill and
 * compressed as the Python writes it (deflate level 4, shuffled).
 */

#ifndef PLANETMODEL_WITH_NETCDF
#error "planetmodel was configured without netCDF-C, so without the model file: configure it with PLANETMODEL_WITH_NETCDF=ON"
#endif

#include <netcdf.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "planetmodel/error.hpp"

namespace planetmodel::netcdf {

/**
 * @brief A netCDF-C status turned into an `Error` where it is a fault.
 * @param status What a netCDF-C call returned.
 * @param source The file, as the error names it.
 * @param where The variable or attribute, as the error names it.
 * @throws planetmodel::Error with netCDF-C's message where `status` is a
 *         fault.
 */
inline void Check(int status, const std::string& source, std::string_view where) {
  if (status != NC_NOERR) throw Error(source, where, nc_strerror(status));
}

/** @brief A file open for reading or writing, closed when it goes. */
class File {
 public:
  /**
   * @brief Open a file for reading.
   * @param path The file.
   * @throws planetmodel::Error if netCDF-C cannot open it.
   */
  explicit File(const std::filesystem::path& path) : path_(path.string()) {
    Check(nc_open(path_.c_str(), NC_NOWRITE, &id_), path_, "");
  }

  /**
   * @brief Create a netCDF-4 file for writing.
   * @param path The file, which must not exist.
   * @throws planetmodel::Error if netCDF-C cannot create it.
   */
  [[nodiscard]] static File Create(const std::filesystem::path& path) {
    auto out = File{};
    out.path_ = path.string();
    Check(nc_create(out.path_.c_str(), NC_NETCDF4 | NC_NOCLOBBER, &out.id_), out.path_, "");
    return out;
  }
  /** @brief A file is not copied, its handle being its own. */
  File(const File&) = delete;
  /** @brief A file is not copied, its handle being its own. */
  File& operator=(const File&) = delete;
  /**
   * @brief The handle moves with the file.
   * @param other The file moved from, left closed.
   */
  File(File&& other) noexcept : path_(std::move(other.path_)), id_(std::exchange(other.id_, -1)) {}
  /**
   * @brief The handles are exchanged.
   * @param other The file exchanged with.
   */
  File& operator=(File&& other) noexcept {
    std::swap(path_, other.path_);
    std::swap(id_, other.id_);
    return *this;
  }
  /** @brief Close the file, quietly where it is already closed. */
  ~File() {
    if (id_ >= 0) nc_close(id_);
  }

  /**
   * @brief Close the file, finishing what was written to it.
   * @throws planetmodel::Error if netCDF-C cannot finish the file.
   */
  void Close() {
    if (id_ >= 0) Check(nc_close(std::exchange(id_, -1)), path_, "");
  }

  /** @brief The file's own id, the root group. */
  [[nodiscard]] int Id() const noexcept { return id_; }
  /** @brief The path it was opened from, as errors name it. */
  [[nodiscard]] const std::string& Path() const noexcept { return path_; }

 private:
  File() = default;

  std::string path_;
  int id_ = -1;
};

/** @brief The values of a variable, flattened in C order, and its shape. */
struct Array {
  std::vector<double> values{};    /**< The values, last index fastest. */
  std::vector<std::size_t> shape{}; /**< The length of each dimension. */
};

/**
 * @brief Whether a group holds a variable.
 * @param group The group, or the file's id.
 * @param name The variable.
 */
[[nodiscard]] inline bool HasVariable(int group, std::string_view name) {
  auto id = 0;
  return nc_inq_varid(group, std::string{name}.c_str(), &id) == NC_NOERR;
}

/**
 * @brief The id of a subgroup.
 * @param group The group, or the file's id.
 * @param name The subgroup.
 * @param source The file, as an error names it.
 * @throws planetmodel::Error if the group has no such subgroup.
 */
[[nodiscard]] inline int Group(int group, std::string_view name, const std::string& source) {
  auto id = 0;
  Check(nc_inq_grp_ncid(group, std::string{name}.c_str(), &id), source, name);
  return id;
}

/**
 * @brief The id of a variable.
 * @param group The group, or the file's id.
 * @param name The variable.
 * @param source The file, as an error names it.
 * @throws planetmodel::Error naming the variable where the group has none.
 */
[[nodiscard]] inline int Variable(int group, std::string_view name, const std::string& source) {
  auto id = 0;
  if (nc_inq_varid(group, std::string{name}.c_str(), &id) != NC_NOERR) {
    throw Error(source, name, "is not a variable of the file");
  }
  return id;
}

/**
 * @brief A variable's values as doubles, and its shape.
 * @param group The group, or the file's id.
 * @param name The variable.
 * @param source The file, as an error names it.
 * @throws planetmodel::Error if there is no such variable or it cannot be
 *         read as doubles.
 */
[[nodiscard]] inline Array ReadDoubles(int group, std::string_view name,
                                       const std::string& source) {
  const auto var = Variable(group, name, source);
  auto ndims = 0;
  Check(nc_inq_varndims(group, var, &ndims), source, name);
  auto dims = std::vector<int>(static_cast<std::size_t>(ndims));
  Check(nc_inq_vardimid(group, var, dims.data()), source, name);
  auto out = Array{};
  auto size = std::size_t{1};
  for (const auto dim : dims) {
    auto length = std::size_t{0};
    Check(nc_inq_dimlen(group, dim, &length), source, name);
    out.shape.push_back(length);
    size *= length;
  }
  out.values.resize(size);
  if (size > 0) Check(nc_get_var_double(group, var, out.values.data()), source, name);
  return out;
}

/** @brief The variable id of a group's own attributes. */
inline constexpr auto GLOBAL = NC_GLOBAL;

/**
 * @brief Whether an attribute exists.
 * @param group The group, or the file's id.
 * @param var The variable's id, or `GLOBAL`.
 * @param name The attribute.
 */
[[nodiscard]] inline bool HasAttribute(int group, int var, std::string_view name) {
  auto type = nc_type{};
  auto length = std::size_t{0};
  return nc_inq_att(group, var, std::string{name}.c_str(), &type, &length) == NC_NOERR;
}

/**
 * @brief An attribute as text, from a character or a string attribute.
 * @param group The group, or the file's id.
 * @param var The variable's id, or `GLOBAL`.
 * @param name The attribute.
 * @param source The file, as an error names it.
 * @return The text, or empty where there is no such attribute.
 * @throws planetmodel::Error if the attribute is not text.
 */
[[nodiscard]] inline std::optional<std::string> TextAttribute(int group, int var,
                                                              std::string_view name,
                                                              const std::string& source) {
  auto type = nc_type{};
  auto length = std::size_t{0};
  const auto key = std::string{name};
  if (nc_inq_att(group, var, key.c_str(), &type, &length) != NC_NOERR) return std::nullopt;
  if (type == NC_CHAR) {
    auto text = std::string(length, '\0');
    Check(nc_get_att_text(group, var, key.c_str(), text.data()), source, name);
    return text;
  }
  if (type == NC_STRING && length == 1) {
    char* text = nullptr;
    Check(nc_get_att_string(group, var, key.c_str(), &text), source, name);
    auto out = std::string{text ? text : ""};
    nc_free_string(1, &text);
    return out;
  }
  throw Error(source, name, "is not a text attribute");
}

/**
 * @brief An attribute as numbers.
 * @param group The group, or the file's id.
 * @param var The variable's id, or `GLOBAL`.
 * @param name The attribute.
 * @param source The file, as an error names it.
 * @return The numbers, or empty where there is no such attribute.
 * @throws planetmodel::Error if the attribute is text.
 */
[[nodiscard]] inline std::optional<std::vector<double>> NumberAttribute(
    int group, int var, std::string_view name, const std::string& source) {
  auto type = nc_type{};
  auto length = std::size_t{0};
  const auto key = std::string{name};
  if (nc_inq_att(group, var, key.c_str(), &type, &length) != NC_NOERR) return std::nullopt;
  if (type == NC_CHAR || type == NC_STRING) {
    throw Error(source, name, "is text, not a number");
  }
  auto out = std::vector<double>(length);
  Check(nc_get_att_double(group, var, key.c_str(), out.data()), source, name);
  return out;
}

/**
 * @brief An attribute's value: text, real numbers, or integers, as the
 * file types it.
 */
using Attribute = std::variant<std::string, std::vector<double>, std::vector<long long>>;

/** @brief Attributes by name. */
using Attributes = std::map<std::string, Attribute>;

/**
 * @brief A numeric attribute's numbers as doubles.
 * @param value The attribute.
 * @return Its numbers, or empty where it is text.
 */
[[nodiscard]] inline std::optional<std::vector<double>> AsNumbers(const Attribute& value) {
  if (const auto* reals = std::get_if<std::vector<double>>(&value)) return *reals;
  if (const auto* integers = std::get_if<std::vector<long long>>(&value)) {
    return std::vector<double>(integers->begin(), integers->end());
  }
  return std::nullopt;
}

/**
 * @brief Whether a netCDF type is an integer type.
 * @param type The type.
 */
[[nodiscard]] constexpr bool IsIntegerType(nc_type type) noexcept {
  return type == NC_BYTE || type == NC_UBYTE || type == NC_SHORT || type == NC_USHORT ||
         type == NC_INT || type == NC_UINT || type == NC_INT64 || type == NC_UINT64;
}

/**
 * @brief Every attribute of a variable, or of a group with `GLOBAL`.
 * @param group The group, or the file's id.
 * @param var The variable's id, or `GLOBAL`.
 * @param source The file, as an error names it.
 * @throws planetmodel::Error if netCDF-C cannot read one.
 */
[[nodiscard]] inline Attributes ReadAttributes(int group, int var, const std::string& source) {
  auto count = 0;
  Check(var == GLOBAL ? nc_inq_natts(group, &count) : nc_inq_varnatts(group, var, &count),
        source, "");
  auto out = Attributes{};
  for (auto i = 0; i < count; ++i) {
    auto name = std::string(NC_MAX_NAME + 1, '\0');
    Check(nc_inq_attname(group, var, i, name.data()), source, "");
    name.resize(std::char_traits<char>::length(name.c_str()));
    auto type = nc_type{};
    auto length = std::size_t{0};
    Check(nc_inq_att(group, var, name.c_str(), &type, &length), source, name);
    if (type == NC_CHAR || type == NC_STRING) {
      out.emplace(name, *TextAttribute(group, var, name, source));
    } else if (IsIntegerType(type)) {
      auto values = std::vector<long long>(length);
      Check(nc_get_att_longlong(group, var, name.c_str(), values.data()), source, name);
      out.emplace(name, std::move(values));
    } else {
      out.emplace(name, *NumberAttribute(group, var, name, source));
    }
  }
  return out;
}

/**
 * @brief The names of a group's variables, in the file's order.
 * @param group The group, or the file's id.
 * @param source The file, as an error names it.
 */
[[nodiscard]] inline std::vector<std::string> VariableNames(int group, const std::string& source) {
  auto count = 0;
  Check(nc_inq_nvars(group, &count), source, "");
  auto ids = std::vector<int>(static_cast<std::size_t>(count));
  Check(nc_inq_varids(group, &count, ids.data()), source, "");
  auto out = std::vector<std::string>{};
  for (const auto id : ids) {
    auto name = std::string(NC_MAX_NAME + 1, '\0');
    Check(nc_inq_varname(group, id, name.data()), source, "");
    name.resize(std::char_traits<char>::length(name.c_str()));
    out.push_back(std::move(name));
  }
  return out;
}

/**
 * @brief The names of a variable's dimensions, in order.
 * @param group The group, or the file's id.
 * @param name The variable.
 * @param source The file, as an error names it.
 * @throws planetmodel::Error if there is no such variable.
 */
[[nodiscard]] inline std::vector<std::string> DimensionNames(int group, std::string_view name,
                                                            const std::string& source) {
  const auto var = Variable(group, name, source);
  auto ndims = 0;
  Check(nc_inq_varndims(group, var, &ndims), source, name);
  auto dims = std::vector<int>(static_cast<std::size_t>(ndims));
  Check(nc_inq_vardimid(group, var, dims.data()), source, name);
  auto out = std::vector<std::string>{};
  for (const auto dim : dims) {
    auto text = std::string(NC_MAX_NAME + 1, '\0');
    Check(nc_inq_dimname(group, dim, text.data()), source, name);
    text.resize(std::char_traits<char>::length(text.c_str()));
    out.push_back(std::move(text));
  }
  return out;
}

/**
 * @brief The length of a group's dimension, or empty where it has none.
 * @param group The group, or the file's id.
 * @param name The dimension.
 * @param source The file, as an error names it.
 */
[[nodiscard]] inline std::optional<std::size_t> DimensionLength(int group, std::string_view name,
                                                               const std::string& source) {
  auto id = 0;
  if (nc_inq_dimid(group, std::string{name}.c_str(), &id) != NC_NOERR) return std::nullopt;
  auto length = std::size_t{0};
  Check(nc_inq_dimlen(group, id, &length), source, name);
  return length;
}

/**
 * @brief The names of a group's subgroups.
 * @param group The group, or the file's id.
 * @param source The file, as an error names it.
 */
[[nodiscard]] inline std::vector<std::string> GroupNames(int group, const std::string& source) {
  auto count = 0;
  Check(nc_inq_grps(group, &count, nullptr), source, "");
  auto ids = std::vector<int>(static_cast<std::size_t>(count));
  if (count > 0) Check(nc_inq_grps(group, &count, ids.data()), source, "");
  auto out = std::vector<std::string>{};
  for (const auto id : ids) {
    auto name = std::string(NC_MAX_NAME + 1, '\0');
    Check(nc_inq_grpname(id, name.data()), source, "");
    name.resize(std::char_traits<char>::length(name.c_str()));
    out.push_back(std::move(name));
  }
  return out;
}

/**
 * @brief The names of a group's own dimensions, its parents' left out, in
 * the file's order.
 * @param group The group, or the file's id.
 * @param source The file, as an error names it.
 */
[[nodiscard]] inline std::vector<std::string> GroupDimensionNames(int group,
                                                                const std::string& source) {
  auto count = 0;
  Check(nc_inq_dimids(group, &count, nullptr, 0), source, "");
  auto ids = std::vector<int>(static_cast<std::size_t>(count));
  if (count > 0) Check(nc_inq_dimids(group, &count, ids.data(), 0), source, "");
  auto out = std::vector<std::string>{};
  for (const auto id : ids) {
    auto name = std::string(NC_MAX_NAME + 1, '\0');
    Check(nc_inq_dimname(group, id, name.data()), source, "");
    name.resize(std::char_traits<char>::length(name.c_str()));
    out.push_back(std::move(name));
  }
  return out;
}

/**
 * @brief The names of a variable's attributes, or of a group's with
 * `GLOBAL`, in the file's order.
 * @param group The group, or the file's id.
 * @param var The variable's id, or `GLOBAL`.
 * @param source The file, as an error names it.
 */
[[nodiscard]] inline std::vector<std::string> AttributeNames(int group, int var,
                                                           const std::string& source) {
  auto count = 0;
  Check(var == GLOBAL ? nc_inq_natts(group, &count) : nc_inq_varnatts(group, var, &count),
        source, "");
  auto out = std::vector<std::string>{};
  for (auto i = 0; i < count; ++i) {
    auto name = std::string(NC_MAX_NAME + 1, '\0');
    Check(nc_inq_attname(group, var, i, name.data()), source, "");
    name.resize(std::char_traits<char>::length(name.c_str()));
    out.push_back(std::move(name));
  }
  return out;
}

/**
 * @brief A string variable's values, of one dimension.
 * @param group The group, or the file's id.
 * @param name The variable.
 * @param source The file, as an error names it.
 * @throws planetmodel::Error if there is no such variable or it is not a
 *         string variable of one dimension.
 */
[[nodiscard]] inline std::vector<std::string> ReadStrings(int group, std::string_view name,
                                                         const std::string& source) {
  const auto var = Variable(group, name, source);
  auto type = nc_type{};
  auto ndims = 0;
  Check(nc_inq_vartype(group, var, &type), source, name);
  Check(nc_inq_varndims(group, var, &ndims), source, name);
  if (type != NC_STRING || ndims != 1) {
    throw Error(source, name, "is not a string variable of one dimension");
  }
  auto dim = 0;
  Check(nc_inq_vardimid(group, var, &dim), source, name);
  auto length = std::size_t{0};
  Check(nc_inq_dimlen(group, dim, &length), source, name);
  auto raw = std::vector<char*>(length, nullptr);
  if (length > 0) Check(nc_get_var_string(group, var, raw.data()), source, name);
  auto out = std::vector<std::string>{};
  for (auto* text : raw) out.emplace_back(text ? text : "");
  if (length > 0) nc_free_string(length, raw.data());
  return out;
}

/**
 * @brief A subgroup, created.
 * @param group The group, or the file's id.
 * @param name The subgroup.
 * @param source The file, as an error names it.
 * @return The subgroup's id.
 * @throws planetmodel::Error if netCDF-C cannot create it.
 */
[[nodiscard]] inline int CreateGroup(int group, std::string_view name, const std::string& source) {
  auto id = 0;
  Check(nc_def_grp(group, std::string{name}.c_str(), &id), source, name);
  return id;
}

/**
 * @brief A dimension of a group, created where the group has none of the
 * name.
 * @param group The group, or the file's id.
 * @param name The dimension.
 * @param length Its length.
 * @param source The file, as an error names it.
 * @throws planetmodel::Error if the group's dimension of the name has
 *         another length, or netCDF-C cannot create it.
 */
inline void Dimension(int group, std::string_view name, std::size_t length,
                      const std::string& source) {
  auto id = 0;
  const auto key = std::string{name};
  // the group's own dimensions only: a parent's of the name is another
  auto count = 0;
  Check(nc_inq_dimids(group, &count, nullptr, 0), source, name);
  auto ids = std::vector<int>(static_cast<std::size_t>(count));
  if (count > 0) Check(nc_inq_dimids(group, &count, ids.data(), 0), source, name);
  for (const auto dim : ids) {
    auto text = std::string(NC_MAX_NAME + 1, '\0');
    Check(nc_inq_dimname(group, dim, text.data()), source, name);
    text.resize(std::char_traits<char>::length(text.c_str()));
    if (text != key) continue;
    auto have = std::size_t{0};
    Check(nc_inq_dimlen(group, dim, &have), source, name);
    if (have != length) {
      throw Error(source, name,
                  "is a dimension of " + std::to_string(have) + ", not of " +
                      std::to_string(length));
    }
    return;
  }
  Check(nc_def_dim(group, key.c_str(), length, &id), source, name);
}

namespace detail {

/// A variable of `type` over the named dimensions of `group`, created.
inline int DefineVariable(int group, std::string_view name, nc_type type,
                          const std::vector<std::string>& dimensions, const std::string& source) {
  auto ids = std::vector<int>{};
  for (const auto& dim : dimensions) {
    auto id = 0;
    Check(nc_inq_dimid(group, dim.c_str(), &id), source, name);
    ids.push_back(id);
  }
  auto var = 0;
  Check(nc_def_var(group, std::string{name}.c_str(), type, static_cast<int>(ids.size()),
                   ids.data(), &var),
        source, name);
  return var;
}

}  // namespace detail

/**
 * @brief A real variable, written whole, with NaN as its fill and
 * compressed.
 * @param group The group, or the file's id.
 * @param name The variable.
 * @param dimensions The names of its dimensions, each of the group or a
 *        parent.
 * @param values Its values, C order.
 * @param source The file, as an error names it.
 * @return The variable's id.
 * @throws planetmodel::Error if netCDF-C cannot write it.
 */
inline int WriteDoubles(int group, std::string_view name,
                        const std::vector<std::string>& dimensions, std::span<const double> values,
                        const std::string& source) {
  const auto var = detail::DefineVariable(group, name, NC_DOUBLE, dimensions, source);
  const auto fill = std::numeric_limits<double>::quiet_NaN();
  Check(nc_def_var_fill(group, var, NC_FILL, &fill), source, name);
  if (!dimensions.empty()) Check(nc_def_var_deflate(group, var, 1, 1, 4), source, name);
  if (!values.empty()) Check(nc_put_var_double(group, var, values.data()), source, name);
  return var;
}

/**
 * @brief An integer variable, written whole.
 * @param group The group, or the file's id.
 * @param name The variable.
 * @param dimensions The names of its dimensions, each of the group or a
 *        parent.
 * @param values Its values, C order.
 * @param source The file, as an error names it.
 * @return The variable's id.
 * @throws planetmodel::Error if netCDF-C cannot write it.
 */
inline int WriteIntegers(int group, std::string_view name,
                         const std::vector<std::string>& dimensions, std::span<const int> values,
                         const std::string& source) {
  const auto var = detail::DefineVariable(group, name, NC_INT, dimensions, source);
  if (!values.empty()) Check(nc_put_var_int(group, var, values.data()), source, name);
  return var;
}

/**
 * @brief A string variable of one dimension, written whole.
 * @param group The group, or the file's id.
 * @param name The variable.
 * @param dimension The name of its dimension.
 * @param values Its values.
 * @param source The file, as an error names it.
 * @return The variable's id.
 * @throws planetmodel::Error if netCDF-C cannot write it.
 */
inline int WriteStrings(int group, std::string_view name, const std::string& dimension,
                        std::span<const std::string> values, const std::string& source) {
  const auto var = detail::DefineVariable(group, name, NC_STRING, {dimension}, source);
  auto raw = std::vector<const char*>{};
  for (const auto& v : values) raw.push_back(v.c_str());
  if (!raw.empty()) Check(nc_put_var_string(group, var, raw.data()), source, name);
  return var;
}

/**
 * @brief An attribute, written: text as characters, real numbers as
 * doubles, integers as 32-bit integers where every one fits and 64-bit
 * ones otherwise.
 * @param group The group, or the file's id.
 * @param var The variable's id, or `GLOBAL`.
 * @param name The attribute.
 * @param value Its value.
 * @param source The file, as an error names it.
 * @throws planetmodel::Error if netCDF-C cannot write it.
 */
inline void WriteAttribute(int group, int var, std::string_view name, const Attribute& value,
                           const std::string& source) {
  const auto key = std::string{name};
  if (const auto* text = std::get_if<std::string>(&value)) {
    Check(nc_put_att_text(group, var, key.c_str(), text->size(), text->data()), source, name);
  } else if (const auto* reals = std::get_if<std::vector<double>>(&value)) {
    Check(nc_put_att_double(group, var, key.c_str(), NC_DOUBLE, reals->size(), reals->data()),
          source, name);
  } else {
    const auto& integers = std::get<std::vector<long long>>(value);
    auto narrow = true;
    for (const auto i : integers) {
      narrow =
          narrow && i >= std::numeric_limits<int>::min() && i <= std::numeric_limits<int>::max();
    }
    if (narrow) {
      const auto values = std::vector<int>(integers.begin(), integers.end());
      Check(nc_put_att_int(group, var, key.c_str(), NC_INT, values.size(), values.data()),
            source, name);
    } else {
      Check(nc_put_att_longlong(group, var, key.c_str(), NC_INT64, integers.size(),
                                integers.data()),
            source, name);
    }
  }
}

}  // namespace planetmodel::netcdf

#endif  // PLANETMODEL_NETCDF_HPP
