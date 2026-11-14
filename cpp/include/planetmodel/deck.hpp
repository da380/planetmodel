#ifndef PLANETMODEL_DECK_HPP
#define PLANETMODEL_DECK_HPP

/**
 * @file
 * @brief Decks, as `planetmodel.deck` reads and writes them: tables of
 * knots, radius first and one column per named field, in which a repeated
 * radius marks a boundary.
 *
 * A `DeckFormat` names the columns (by row width, where a format admits
 * more than one), says how many header lines precede the table and which
 * `#` lines may follow them as part of the header, what the header must
 * say and how it is written, and what is written for a column a layer
 * lacks.  `MINEOS()` is the format of the mineos and PREM decks, three
 * header lines (a title; `ifanis tref ifdeck`; `nknot nic noc`) and nine
 * or six columns, an absent quality factor written as 0, mineos's no loss.
 * `MAXWELL()` is that of a Maxwell body, the unrelaxed elastic values
 * beside a viscosity: a title, the line naming the columns `r rho vpv vsv
 * vph vsh eta viscosity`, then optionally `# layers` and `# interfaces`
 * lines naming them centre outward (`-` for none), SI, NaN where a layer
 * lacks a field.  `PRONY()` is that of a generalised Maxwell body: the
 * same header and elastic columns, and in place of the viscosity one pair
 * of columns per Prony branch, `relaxation_strength_k relaxation_time_k`
 * for k = 1 to N, N read from the column line and the same throughout the
 * deck (`PronyColumns`, `PronyBranches`); a layer is elastic, NaN in every
 * branch column, or holds a number in each, a strength of 0 in a branch it
 * does not use, which the writer completes with the time of the layer's
 * last branch.  `BURGERS()` is the reduced form of two branches, as
 * `MAXWELL()` is of one: the Maxwell element's viscosity beside the Kelvin
 * element's viscosity and shear modulus, the latter a fraction of the
 * layer's, a Maxwell layer NaN in the two Kelvin columns.  Both formats
 * check the values of every deck they read or write, as the Python's do,
 * with the same messages.  `ND()` is TauP's named discontinuities, depth first from
 * the surface in km, km/s and g/cm^3 with label lines naming the regions
 * below them, and `TVEL()` its tabled velocities under two header lines;
 * a format whose table is not radius-first SI rows reads and writes it
 * through hooks of its own, the deck being SI and radius first whatever
 * the file.  `ReadDeck` gives a `Deck`, the numbers, the header lines and
 * a table's labels, nothing else; the table's blank lines and `#`
 * comments are skipped and its numbers read as Python reads them, `nan`
 * and a leading `+` included.  `WriteDeck` writes a deck as the Python's
 * `write_deck` does, character for character.  `MineosNames`,
 * `MaxwellNames` (of a Maxwell, prony or burgers deck) and `NdNames` give
 * the layer and interface names a header or a table's labels imply, and
 * `DeckLayers` each layer's columns as functions of the radius, joined by
 * the kind asked for, by column or by a family's stem where asked, on the
 * deck's skeleton.  A kind is `cubic`, `natural` or `linear`, a piecewise
 * polynomial through the values, or the same with `log-` before it, the
 * polynomial through the logarithms of values positive and finite at every
 * knot of the layer and the value its exponential, `log-linear` being
 * geometric between knots.  A Maxwell deck's viscosity is joined by
 * `log-linear` (`MAXWELL_KINDS`), which keeps it positive and follows a
 * viscosity changing by orders of magnitude within a layer; so are a prony
 * deck's relaxation times, its strengths by `linear` (`PRONY_KINDS`), and
 * a burgers deck's two viscosities, its Kelvin modulus fraction by
 * `linear` (`BURGERS_KINDS`).  A map of kinds of one's own joins any of
 * them otherwise: `{{"viscosity", "linear"}}` joins a viscosity's values
 * by straight lines.  A burgers deck's Burgers body is its two Prony
 * branches by `BurgersTerms` (`prony.hpp`).
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#include "planetmodel/error.hpp"
#include "planetmodel/radial.hpp"
#include "planetmodel/skeleton.hpp"
#include "planetmodel/vocabulary.hpp"

/** @brief Defined where this header provides the prony and burgers decks
 * beside the Maxwell deck (`PRONY()`, `BURGERS()`, `PRONY_KINDS`,
 * `BURGERS_KINDS`, `PronyColumns`, `PronyBranches`, `PronyHeader`,
 * `BurgersHeader`), and `MAXWELL_KINDS` joining a Maxwell deck's viscosity
 * by `log-linear`. */
#define PLANETMODEL_HAS_VISCOELASTIC_DECKS 1

namespace planetmodel {

/** @brief The columns of a transversely isotropic mineos deck, after the radius. */
inline const auto MINEOS_COLUMNS = std::vector<std::string>{
    "rho", "vpv", "vsv", "qkappa", "qmu", "vph", "vsh", "eta"};

/** @brief The columns of an isotropic mineos deck, after the radius. */
inline const auto MINEOS_ISOTROPIC_COLUMNS =
    std::vector<std::string>{"rho", "vp", "vs", "qkappa", "qmu"};

/** @brief The columns of a Maxwell deck, after the radius. */
inline const auto MAXWELL_COLUMNS = std::vector<std::string>{
    "rho", "vpv", "vsv", "vph", "vsh", "eta", "viscosity"};

/**
 * @brief The kinds a Maxwell deck's columns are joined by where not the
 * default: the viscosity in its logarithm by straight lines, geometric
 * between knots, which keeps it positive and follows a viscosity that
 * changes by orders of magnitude within a layer; `{{"viscosity",
 * "linear"}}` joins the values themselves by straight lines instead.
 */
inline const auto MAXWELL_KINDS =
    std::map<std::string, std::string, std::less<>>{{"viscosity", "log-linear"}};

/**
 * @brief The stems of a Prony branch's two names, its strength and its time,
 * as the Python's `rheology.BRANCH_STEMS`: families of the vocabulary
 * (`Families`).
 */
inline const auto BRANCH_STEMS =
    std::vector<std::string>{"relaxation_strength", "relaxation_time"};

/**
 * @brief The columns of a prony deck after the radius and before its
 * branches: a Maxwell deck's, without the viscosity.
 */
inline const auto PRONY_COLUMNS =
    std::vector<std::string>{"rho", "vpv", "vsv", "vph", "vsh", "eta"};

/**
 * @brief The kinds a prony deck's columns are joined by where not the
 * default, by the stems of the branches' families: the strengths by
 * straight lines, which keep every one in [0, 1] and their sum at most one
 * between knots; the times in their logarithm by straight lines, positive
 * and geometric between knots, as a Maxwell deck's viscosity is joined.
 */
inline const auto PRONY_KINDS = std::map<std::string, std::string, std::less<>>{
    {"relaxation_strength", "linear"}, {"relaxation_time", "log-linear"}};

/**
 * @brief The columns of a burgers deck, after the radius: a Maxwell deck's,
 * then the Kelvin element's viscosity and its shear modulus as a fraction
 * of the layer's elastic (Voigt) shear modulus.
 */
inline const auto BURGERS_COLUMNS =
    std::vector<std::string>{"rho", "vpv", "vsv", "vph", "vsh", "eta", "viscosity",
                             "kelvin_viscosity", "kelvin_modulus_fraction"};

/**
 * @brief The kinds a burgers deck's columns are joined by where not the
 * default: the two viscosities in their logarithm by straight lines, as a
 * Maxwell deck's viscosity is joined, and the Kelvin element's modulus
 * fraction by straight lines; all three stay positive between knots.
 */
inline const auto BURGERS_KINDS = std::map<std::string, std::string, std::less<>>{
    {"viscosity", "log-linear"},
    {"kelvin_viscosity", "log-linear"},
    {"kelvin_modulus_fraction", "linear"}};

/**
 * @brief How far, per branch, a prony deck's strengths may stand from what
 * they were before they were written: the rounding of the eight significant
 * digits a value is written with, at most 5e-9 on a strength below one,
 * doubled.  A layer's N strengths may sum to one plus N times this.
 */
inline constexpr double STRENGTH_ROUNDING = 1e-8;

namespace detail {

inline std::vector<std::string> Tokens(std::string_view line) {
  auto in = std::istringstream{std::string{line}};
  auto out = std::vector<std::string>{};
  for (auto token = std::string{}; in >> token;) out.push_back(token);
  return out;
}

inline std::string Trimmed(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) return {};
  const auto last = text.find_last_not_of(" \t\r\n");
  return std::string{text.substr(first, last - first + 1)};
}

/// A number as Python's float reads a deck's: a leading sign, `nan` and
/// `inf` included; empty where the token is not one.
inline std::optional<double> Number(std::string_view token) {
  if (!token.empty() && token.front() == '+') token.remove_prefix(1);
  auto value = 0.0;
  const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
  if (error != std::errc{} || end != token.data() + token.size()) return std::nullopt;
  return value;
}

inline std::optional<int> Integer(std::string_view token) {
  if (!token.empty() && token.front() == '+') token.remove_prefix(1);
  auto value = 0;
  const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
  if (error != std::errc{} || end != token.data() + token.size()) return std::nullopt;
  return value;
}

/// The first word after the `#` of a comment line; empty where the line is
/// no comment, or a comment of no words.
inline std::optional<std::string> Keyword(std::string_view line) {
  const auto text = Trimmed(line);
  if (text.empty() || text.front() != '#') return std::nullopt;
  const auto words = Tokens(std::string_view{text}.substr(1));
  if (words.empty()) return std::nullopt;
  return words.front();
}

/// Words as the Python writes a list of them: `['a', 'b']`.
inline std::string ListRepr(const std::vector<std::string>& words) {
  auto out = std::string{"["};
  for (const auto& word : words) out += (out.size() > 1 ? ", '" : "'") + word + "'";
  return out + "]";
}

/// Words joined by single spaces.
inline std::string Joined(const std::vector<std::string>& words) {
  auto out = std::string{};
  for (const auto& word : words) out += (out.empty() ? "" : " ") + word;
  return out;
}

/// The members of the families `stems` for k = 1 to n, one of each stem
/// for each index, in order.
inline std::vector<std::string> Members(const std::vector<std::string>& stems, std::size_t n) {
  auto out = std::vector<std::string>{};
  for (auto k = std::size_t{1}; k <= n; ++k) {
    for (const auto& stem : stems) out.push_back(std::format("{}_{}", stem, k));
  }
  return out;
}

/// N where the column `names` hold every member of the families `stems`
/// for k = 1 to N and no other, N at least one; refused, naming the members
/// held, otherwise.
inline std::size_t Groups(const std::vector<std::string>& stems,
                          const std::vector<std::string>& names, std::string_view what,
                          const std::string& source) {
  auto held = std::map<int, std::set<std::string>>{};
  for (const auto& name : names) {
    const auto found = Member(name);
    if (found && std::ranges::find(stems, found->first) != stems.end()) {
      held[found->second].insert(found->first);
    }
  }
  auto whole = !held.empty();
  auto k = 1;
  for (const auto& [index, got] : held) {
    if (index != k++ || got.size() != stems.size()) whole = false;
  }
  if (whole) return held.size();
  auto rule = std::string{};
  for (const auto& stem : stems) rule += (rule.empty() ? "" : ", ") + stem + "_k";
  auto got = std::vector<std::string>{};
  for (const auto& [index, members] : held) {
    for (const auto& stem : stems) {
      if (members.contains(stem)) got.push_back(std::format("{}_{}", stem, index));
    }
  }
  throw Error(source, "",
              std::format("{} holds {} for k = 1 to N, N at least one, each of them; it has {}",
                          what, rule, got.empty() ? std::string{"none"} : ListRepr(got)));
}

}  // namespace detail

/**
 * @brief A table of knots: radii, one column per name, the header lines,
 * and the labels a table carries.
 */
class Deck {
 public:
  /**
   * @brief A deck of knots.
   * @param radius The knots' radii, non-decreasing, every repeat a
   *        boundary.
   * @param columns Each column's name and its value at every knot, NaN
   *        where it is absent.
   * @param header The header lines.
   * @param source Where the deck was read from, as errors name it.
   * @param labels Each label's radius and the label: a TauP table's
   *        labels, each naming the region below its radius.
   * @throws planetmodel::Error if the radii decrease, a column has not one
   *         value per knot, or a layer has fewer than two knots.
   */
  Deck(std::vector<double> radius,
       std::vector<std::pair<std::string, std::vector<double>>> columns,
       std::vector<std::string> header = {}, std::string source = "the deck",
       std::vector<std::pair<double, std::string>> labels = {})
      : radius_(std::move(radius)),
        columns_(std::move(columns)),
        header_(std::move(header)),
        source_(std::move(source)),
        labels_(std::move(labels)) {
    if (radius_.size() < 2) Fail("needs at least two knot radii");
    for (auto i = std::size_t{1}; i < radius_.size(); ++i) {
      if (!(radius_[i] >= radius_[i - 1])) Fail("knot radii must be non-decreasing");
    }
    for (const auto& [name, values] : columns_) {
      if (values.size() != radius_.size()) {
        Fail(std::format("column {} has {} values for {} radii", name, values.size(),
                         radius_.size()));
      }
    }
    static_cast<void>(Layers());
  }

  /** @brief The knots' radii. */
  [[nodiscard]] const std::vector<double>& Radius() const noexcept { return radius_; }
  /** @brief The number of knots. */
  [[nodiscard]] std::size_t NKnots() const noexcept { return radius_.size(); }
  /** @brief The header lines, as read. */
  [[nodiscard]] const std::vector<std::string>& Header() const noexcept { return header_; }
  /** @brief Where the deck was read from. */
  [[nodiscard]] const std::string& Source() const noexcept { return source_; }
  /** @brief Each label's radius and the label, as the table carried them. */
  [[nodiscard]] const std::vector<std::pair<double, std::string>>& Labels() const noexcept {
    return labels_;
  }

  /** @brief The column names, in the deck's order. */
  [[nodiscard]] std::vector<std::string> Names() const {
    auto out = std::vector<std::string>{};
    for (const auto& [name, values] : columns_) out.push_back(name);
    return out;
  }
  /**
   * @brief Whether the deck has a column.
   * @param name The column.
   */
  [[nodiscard]] bool HasColumn(std::string_view name) const {
    for (const auto& [n, values] : columns_) {
      if (n == name) return true;
    }
    return false;
  }
  /**
   * @brief A column's values at every knot.
   * @param name The column.
   * @throws planetmodel::Error if the deck has no such column.
   */
  [[nodiscard]] const std::vector<double>& Column(std::string_view name) const {
    for (const auto& [n, values] : columns_) {
      if (n == name) return values;
    }
    throw Error(source_, "", std::format("has no column {}", name));
  }

  /**
   * @brief The rows of every layer, centre outward, as [begin, end): one
   * span between repeated radii, each of at least two strictly increasing
   * knots.
   * @throws planetmodel::Error if a layer has fewer than two knots.
   */
  [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>> Layers() const {
    auto out = std::vector<std::pair<std::size_t, std::size_t>>{};
    auto start = std::size_t{0};
    for (auto i = std::size_t{1}; i <= radius_.size(); ++i) {
      if (i < radius_.size() && radius_[i] != radius_[i - 1]) continue;
      if (i - start < 2) {
        if (i < radius_.size() && i >= 2 && radius_[i - 1] == radius_[i - 2]) {
          Fail("a radius repeats more than twice: a layer of no thickness");
        }
        Fail(std::format("the layer starting at r = {} has one knot; every layer needs "
                         "at least two",
                         radius_[start]));
      }
      out.emplace_back(start, i);
      start = i;
    }
    return out;
  }
  /** @brief The number of layers. */
  [[nodiscard]] std::size_t NLayers() const { return Layers().size(); }

  /** @brief The skeleton's boundary radii: the ends and every repeated radius. */
  [[nodiscard]] std::vector<double> Boundaries() const {
    auto out = std::vector<double>{radius_.front()};
    for (auto i = std::size_t{1}; i < radius_.size(); ++i) {
      if (radius_[i] == radius_[i - 1]) out.push_back(radius_[i]);
    }
    out.push_back(radius_.back());
    return out;
  }

 private:
  [[noreturn]] void Fail(std::string_view message) const { throw Error(source_, "", message); }

  std::vector<double> radius_;
  std::vector<std::pair<std::string, std::vector<double>>> columns_;
  std::vector<std::string> header_;
  std::string source_;
  std::vector<std::pair<double, std::string>> labels_;
};

/**
 * @brief What a format's own table reader gives: the knot radii and the
 * columns at them, SI and radius first, and the labels the table carries.
 */
struct DeckTable {
  /** @brief The knots' radii, non-decreasing. */
  std::vector<double> radius{};
  /** @brief Each column's name and its value at every knot. */
  std::vector<std::pair<std::string, std::vector<double>>> columns{};
  /** @brief Each label's radius and the label, as `Deck::Labels` holds them. */
  std::vector<std::pair<double, std::string>> labels{};
};

/** @brief What a family of deck files looks like. */
struct DeckFormat {
  /** @brief The format's name, as errors give it. */
  std::string name = "custom";
  /**
   * @brief The column names after the radius, by the number of them a row
   * has; a deck is written with the widest set it holds.  A format with
   * `families` has one set, those before the groups.
   */
  std::map<std::size_t, std::vector<std::string>> columns{};
  /**
   * @brief Stems of the vocabulary's families (`Families`) whose members
   * follow the columns in groups, one member of each stem for each index
   * k = 1 to N in order, N at least one and the same throughout a deck,
   * read from the width of the rows; a deck is written with every group it
   * holds.  Empty for a format of fixed columns.
   */
  std::vector<std::string> families{};
  /** @brief How many lines precede the table. */
  std::size_t header_lines = 0;
  /**
   * @brief The keywords of the `#` lines that may follow the header lines
   * and are read as part of the header, each such line a `#`, its keyword
   * and its words.
   */
  std::vector<std::string> header_comments{};
  /**
   * @brief What the header lines must say: throws `Error` naming `source`
   * where they do not; none for a header kept verbatim.
   */
  std::function<void(const std::vector<std::string>& lines, const std::string& source)>
      check_header{};
  /**
   * @brief The header lines a deck is written with; none to write the
   * deck's own header lines as they are.
   */
  std::function<std::vector<std::string>(const Deck& deck)> write_header{};
  /**
   * @brief The value written for a column where a layer lacks it; NaN for
   * a column not named here.
   */
  std::map<std::string, double, std::less<>> fill{};
  /**
   * @brief The reader of a table that is not rows of radius and SI values:
   * the lines after the header to the radii, columns and labels, SI and
   * radius first; none for rows of radius and SI values.
   */
  std::function<DeckTable(const std::vector<std::string>& lines, const std::string& source)>
      read_table{};
  /**
   * @brief The writer of such a table: a deck and the columns to write to
   * the table's lines; none for rows of radius and SI values.
   */
  std::function<std::vector<std::string>(const Deck& deck,
                                         const std::vector<std::string>& names)>
      write_table{};
  /**
   * @brief What the format requires of a deck's values beyond their
   * layering: throws `Error` where a deck breaks it.  Applied to every deck
   * read, and to every deck written, as it is written: completed, its
   * header the lines it is written with, its columns the format's.  None
   * for no more.
   */
  std::function<void(const Deck& deck)> check_table{};
  /**
   * @brief A deck as it is written, the values a layer lacks completed
   * where `fill` does not say what they are; none to write the deck as it
   * is.
   */
  std::function<Deck(const Deck& deck)> complete{};

  /**
   * @brief The column names of a row of a width.
   * @param width The number of values after the radius.
   * @param source The deck, as an error names it.
   * @throws planetmodel::Error if the format has no row of that width.
   */
  [[nodiscard]] std::vector<std::string> Names(std::size_t width,
                                               const std::string& source) const {
    if (!families.empty()) {
      const auto& lead = columns.begin()->second;
      const auto size = families.size();
      if (width < lead.size() + size || (width - lead.size()) % size != 0) {
        auto stems = std::vector<std::string>{};
        for (const auto& stem : families) stems.push_back(stem + "_k");
        throw Error(source, "",
                    std::format("a {} deck has {} columns after the radius, {}, then {} for "
                                "each index k = 1 to N, {}, N at least one; the table has {}",
                                name, lead.size(), detail::ListRepr(lead), size,
                                detail::ListRepr(stems), width));
      }
      auto out = lead;
      for (auto& member : detail::Members(families, (width - lead.size()) / size)) {
        out.push_back(std::move(member));
      }
      return out;
    }
    if (const auto it = columns.find(width); it != columns.end()) return it->second;
    auto widths = std::string{};
    for (const auto& [w, names] : columns) widths += (widths.empty() ? "" : " or ") +
                                                     std::to_string(w);
    throw Error(source, "",
                std::format("a {} deck has {} columns after the radius, not {}", name,
                            widths, width));
  }

  /**
   * @brief Whether a line is a `#` line of one of `header_comments`.
   * @param line The line.
   */
  [[nodiscard]] bool IsHeaderComment(std::string_view line) const {
    const auto keyword = detail::Keyword(line);
    return keyword && std::ranges::find(header_comments, *keyword) != header_comments.end();
  }
};

/** @brief The three header lines of a mineos deck, read. */
struct MineosHeader {
  std::string name;    /**< The title, trimmed. */
  int ifanis = 0;      /**< Whether the deck is anisotropic. */
  double tref = 0.0;   /**< The reference period of the velocities, in seconds. */
  int ifdeck = 0;      /**< Whether the deck is a table of knots. */
  int nknot = 0;       /**< The number of knots the header states. */
  int nic = 0;         /**< The knots of the inner core. */
  int noc = 0;         /**< The knots to the top of the outer core. */
  std::vector<std::string> extras2{}; /**< The tokens after the second line's three. */
  std::vector<std::string> extras3{}; /**< The tokens after the third line's three. */
};

namespace detail {

/// A mineos header, or empty where the lines are not one.
inline std::optional<MineosHeader> ParsedMineosHeader(const std::vector<std::string>& lines) {
  if (lines.size() < 3) return std::nullopt;
  const auto h2 = Tokens(lines[1]);
  const auto h3 = Tokens(lines[2]);
  if (h2.size() < 3 || h3.size() < 3) return std::nullopt;
  const auto ifanis = Integer(h2[0]);
  const auto tref = Number(h2[1]);
  const auto ifdeck = Integer(h2[2]);
  const auto nknot = Integer(h3[0]);
  const auto nic = Integer(h3[1]);
  const auto noc = Integer(h3[2]);
  if (!ifanis || !tref || !ifdeck || !nknot || !nic || !noc) return std::nullopt;
  return MineosHeader{.name = Trimmed(lines[0]),
                      .ifanis = *ifanis,
                      .tref = *tref,
                      .ifdeck = *ifdeck,
                      .nknot = *nknot,
                      .nic = *nic,
                      .noc = *noc,
                      .extras2 = std::vector<std::string>(h2.begin() + 3, h2.end()),
                      .extras3 = std::vector<std::string>(h3.begin() + 3, h3.end())};
}

}  // namespace detail

/**
 * @brief A mineos deck's header, read.
 * @param lines The header lines, three at least.
 * @param source The deck, as an error names it.
 * @throws planetmodel::Error saying what the second and third lines must
 *         carry, where they do not.
 */
[[nodiscard]] inline MineosHeader ReadMineosHeader(const std::vector<std::string>& lines,
                                                   const std::string& source) {
  if (lines.size() < 3) {
    throw Error(source, "header", "a mineos deck has three header lines");
  }
  if (auto header = detail::ParsedMineosHeader(lines)) return *std::move(header);
  throw Error(source, "header",
              "a mineos deck's second and third header lines carry `ifanis tref "
              "ifdeck` and `nknot nic noc`");
}

/**
 * @brief A deck from its text, held to the format's `check_table`.
 * @param text The deck.
 * @param format Its format.
 * @param source What the deck is, as errors name it.
 * @throws planetmodel::Error naming the line of the first fault, or the
 *         header where it is not as the format says, or as the format's
 *         `check_table` does.
 */
[[nodiscard]] inline Deck ReadDeckText(std::string_view text, const DeckFormat& format,
                                       std::string source = "the deck") {
  auto lines = std::vector<std::string>{};
  {
    auto in = std::istringstream{std::string{text}};
    for (auto line = std::string{}; std::getline(in, line);) lines.push_back(line);
  }
  if (lines.size() < format.header_lines) {
    throw Error(source, "",
                std::format("{} lines, fewer than the {} header lines of a {} deck",
                            lines.size(), format.header_lines, format.name));
  }
  auto n = format.header_lines;
  while (n < lines.size() && format.IsHeaderComment(lines[n])) ++n;
  const auto header =
      std::vector<std::string>(lines.begin(), lines.begin() + static_cast<std::ptrdiff_t>(n));
  if (format.check_header) format.check_header(header, source);
  if (format.read_table) {
    auto table = format.read_table(
        std::vector<std::string>(lines.begin() + static_cast<std::ptrdiff_t>(n), lines.end()),
        source);
    auto deck = Deck(std::move(table.radius), std::move(table.columns), header,
                     std::move(source), std::move(table.labels));
    if (format.check_table) format.check_table(deck);
    return deck;
  }

  auto rows = std::vector<std::vector<double>>{};
  for (auto i = n; i < lines.size(); ++i) {
    auto line = std::string_view{lines[i]};
    line = line.substr(0, line.find('#'));
    const auto tokens = detail::Tokens(line);
    if (tokens.empty()) continue;
    auto row = std::vector<double>{};
    for (const auto& token : tokens) {
      const auto value = detail::Number(token);
      if (!value) {
        throw Error(source, std::format("line {}", i + 1),
                    std::format("{} is not a number", token));
      }
      row.push_back(*value);
    }
    if (!rows.empty() && row.size() != rows.front().size()) {
      throw Error(source, std::format("line {}", i + 1),
                  std::format("has {} values, the rows before it {}", row.size(),
                              rows.front().size()));
    }
    rows.push_back(std::move(row));
  }
  if (rows.empty()) throw Error(source, "", "no knots after the header");
  const auto names = format.Names(rows.front().size() - 1, source);
  auto radius = std::vector<double>{};
  auto columns = std::vector<std::pair<std::string, std::vector<double>>>{};
  for (const auto& name : names) columns.emplace_back(name, std::vector<double>{});
  for (const auto& row : rows) {
    radius.push_back(row[0]);
    for (auto j = std::size_t{0}; j < names.size(); ++j) columns[j].second.push_back(row[j + 1]);
  }
  auto deck = Deck(std::move(radius), std::move(columns), header, std::move(source));
  if (format.check_table) format.check_table(deck);
  return deck;
}

/**
 * @brief A deck from a file.
 * @param path The file.
 * @param format Its format.
 * @throws planetmodel::Error if the file cannot be opened, or as
 *         `ReadDeckText` does.
 */
[[nodiscard]] inline Deck ReadDeck(const std::filesystem::path& path, const DeckFormat& format) {
  auto in = std::ifstream(path, std::ios::binary);
  if (!in) throw Error(path.string(), "", "cannot be opened");
  const auto text = std::string(std::istreambuf_iterator<char>(in), {});
  return ReadDeckText(text, format, path.string());
}

/**
 * @brief A format of named columns, its header lines kept as they are.
 * @param columns The column names after the radius.
 * @param header_lines How many lines precede the table.
 */
[[nodiscard]] inline DeckFormat ColumnsFormat(std::vector<std::string> columns,
                                              std::size_t header_lines = 0) {
  const auto width = columns.size();
  return DeckFormat{.columns = {{width, std::move(columns)}}, .header_lines = header_lines};
}

/** @brief The names a deck's header implies, and what in it did not hold. */
struct DeckNames {
  /** @brief One per layer, empty where the header names none. */
  std::vector<std::optional<std::string>> layers{};
  /** @brief One per interface, centre outward. */
  std::vector<std::optional<std::string>> interfaces{};
  /** @brief What of the header was ignored, and why, as the Python warns it. */
  std::vector<std::string> ignored{};
};

/**
 * @brief The layer and interface names a mineos header implies.
 *
 * `nic` and `noc` count the knots of the inner core and to the top of the
 * outer core, so where each sits on a repeated radius the first two layers
 * are `inner_core` and `outer_core` under `icb` and `cmb`; the outermost
 * interface is `surface`.  A count of zero says there is no such core; a
 * count that does not sit on a repeated radius, or a knot count other than
 * the table's, is ignored and said so in `ignored`.
 *
 * @param deck A mineos deck.
 * @throws planetmodel::Error if the deck's header is not a mineos header.
 */
[[nodiscard]] inline DeckNames MineosNames(const Deck& deck) {
  const auto h = ReadMineosHeader(deck.Header(), deck.Source());
  const auto& r = deck.Radius();
  const auto n = deck.NLayers();
  const auto first = r.front() > 0.0 ? std::size_t{1} : std::size_t{0};
  auto out = DeckNames{.layers = std::vector<std::optional<std::string>>(n),
                       .interfaces = std::vector<std::optional<std::string>>(n + first)};
  out.interfaces.back() = "surface";
  const auto boundaries = deck.Boundaries();
  const auto cores = std::array<std::tuple<int, std::string_view, std::string_view,
                                           std::string_view>, 2>{
      {{h.nic, "nic", "inner_core", "icb"}, {h.noc, "noc", "outer_core", "cmb"}}};
  for (const auto& [count, key, layer, face] : cores) {
    const auto i = static_cast<std::size_t>(count < 0 ? 0 : count);
    if (!(count >= 1 && i < r.size() && r[i - 1] == r[i])) {
      if (count != 0) {
        out.ignored.push_back(std::format("the header's {} = {} does not sit on a repeated "
                                          "radius; the layer is left unnamed",
                                          key, count));
      }
      continue;
    }
    const auto k = static_cast<std::size_t>(
        std::lower_bound(boundaries.begin(), boundaries.end(), r[i]) - boundaries.begin());
    if (k > 0 && k < n) {
      out.layers[k - 1] = std::string{layer};
      out.interfaces[k - 1 + first] = std::string{face};
    }
  }
  if (h.nknot != static_cast<int>(deck.NKnots())) {
    out.ignored.push_back(std::format("the header says {} knots, the table has {}; the table "
                                      "is taken as the truth",
                                      h.nknot, deck.NKnots()));
  }
  return out;
}

namespace detail {

/// The words of the header's `# keyword` line after the keyword, `-` read
/// as no name; empty where the header has no such line.
inline std::optional<std::vector<std::optional<std::string>>> NamesOn(
    const std::vector<std::string>& header, std::string_view keyword) {
  for (auto i = std::size_t{2}; i < header.size(); ++i) {
    if (Keyword(header[i]) != keyword) continue;
    const auto words = Tokens(std::string_view{Trimmed(header[i])}.substr(1));
    auto out = std::vector<std::optional<std::string>>{};
    for (auto w = std::next(words.begin()); w != words.end(); ++w) {
      out.push_back(*w == "-" ? std::nullopt : std::optional{*w});
    }
    return out;
  }
  return std::nullopt;
}

/// One `# layers` or `# interfaces` line, `-` for no name; `what` is the
/// deck, as an error names it.
inline std::string NameLine(std::string_view keyword,
                            const std::vector<std::optional<std::string>>& names,
                            std::string_view what = "a Maxwell deck") {
  auto out = std::format("# {}", keyword);
  for (const auto& name : names) {
    if (name && (name->empty() || *name == "-" || name->find('#') != std::string::npos ||
                 std::ranges::any_of(*name, [](unsigned char c) { return std::isspace(c); }))) {
      throw Error(what, keyword,
                  std::format("a name in a deck is one word, not `-` and without `#`; "
                              "got '{}'",
                              *name));
    }
    out += " ";
    out += name.value_or("-");
  }
  return out;
}

inline bool AnyNamed(const std::vector<std::optional<std::string>>& names) {
  return std::ranges::any_of(names, [](const auto& name) { return name.has_value(); });
}

/// The line naming `r` and the columns.
inline std::string ColumnLine(const std::vector<std::string>& columns) {
  auto out = std::string{"r"};
  for (const auto& name : columns) out += " " + name;
  return out;
}

/// The line naming a Maxwell deck's columns.
inline std::string MaxwellColumnLine() { return ColumnLine(MAXWELL_COLUMNS); }

/// The header lines of a deck of a title, a column line and naming lines:
/// the title, the line naming `r` and `columns`, and a naming line for the
/// layers and one for the interfaces where any of them is named.
inline std::vector<std::string> NamedHeader(
    std::string_view title, const std::vector<std::string>& columns,
    const std::vector<std::optional<std::string>>& layers,
    const std::vector<std::optional<std::string>>& interfaces, std::string_view what) {
  auto out = std::vector<std::string>{std::string{title}, ColumnLine(columns)};
  if (AnyNamed(layers)) out.push_back(NameLine("layers", layers, what));
  if (AnyNamed(interfaces)) out.push_back(NameLine("interfaces", interfaces, what));
  return out;
}

/// Refuse a header of a title, a column line and naming lines that has more
/// than one `# layers` or `# interfaces` line.
inline void OneNamingLineEach(const std::vector<std::string>& lines, const std::string& source,
                              std::string_view what) {
  for (const auto* keyword : {"layers", "interfaces"}) {
    const auto count = std::ranges::count_if(
        lines.begin() + 2, lines.end(), [&](const auto& l) { return Keyword(l) == keyword; });
    if (count > 1) {
      throw Error(source, "header",
                  std::format("{} has one `# {}` line; it has {}", what, keyword, count));
    }
  }
}

}  // namespace detail

/**
 * @brief The layer and interface names a Maxwell header gives, or a prony
 * or burgers header, which name them the same way.
 *
 * Those of its `# layers` and `# interfaces` lines, centre outward, empty
 * for `-`.  Without a line the layers are unnamed, and the interfaces too
 * but for `surface`, the outermost.
 *
 * @param deck A Maxwell, prony or burgers deck.
 * @throws planetmodel::Error if a line names other than the deck's layers,
 *         or its interfaces (one per layer, and one more for a hollow
 *         body's inner boundary).
 */
[[nodiscard]] inline DeckNames MaxwellNames(const Deck& deck) {
  const auto n = deck.NLayers();
  const auto nfaces = n + (deck.Radius().front() > 0.0 ? std::size_t{1} : std::size_t{0});
  auto out = DeckNames{.layers = std::vector<std::optional<std::string>>(n),
                       .interfaces = std::vector<std::optional<std::string>>(nfaces)};
  out.interfaces.back() = "surface";
  if (auto layers = detail::NamesOn(deck.Header(), "layers")) out.layers = *std::move(layers);
  if (auto faces = detail::NamesOn(deck.Header(), "interfaces")) {
    out.interfaces = *std::move(faces);
  }
  for (const auto& [what, size, want] :
       {std::tuple{"layers", out.layers.size(), n},
        std::tuple{"interfaces", out.interfaces.size(), nfaces}}) {
    if (size != want) {
      throw Error(deck.Source(), "header",
                  std::format("names {} {}; its table has {}", size, what, want));
    }
  }
  return out;
}

/**
 * @brief The header lines of a Maxwell deck: its title, the line naming
 * the columns, and a naming line for the layers and one for the interfaces
 * where any of them is named.
 * @param title The deck's title.
 * @param layers The layers' names, centre outward, empty where unnamed.
 * @param interfaces The interfaces' names, centre outward.
 * @throws planetmodel::Error if a name is not one word, is `-`, or holds a
 *         `#`.
 */
[[nodiscard]] inline std::vector<std::string> MaxwellHeader(
    std::string_view title, const std::vector<std::optional<std::string>>& layers = {},
    const std::vector<std::optional<std::string>>& interfaces = {}) {
  return detail::NamedHeader(title, MAXWELL_COLUMNS, layers, interfaces, "a Maxwell deck");
}

namespace detail {

/// The header lines a deck of a title, a column line and naming lines is
/// written with: its title, the line naming `r` and `columns`, and the
/// naming lines its header has where any of their names is given, their
/// counts checked against the deck's (`MaxwellNames`).
inline std::vector<std::string> NamedHeaderLines(const Deck& deck,
                                                 const std::vector<std::string>& columns,
                                                 std::string_view what) {
  const auto names = MaxwellNames(deck);
  const auto& header = deck.Header();
  auto out = std::vector<std::string>{header.empty() ? std::string{} : Trimmed(header.front()),
                                      ColumnLine(columns)};
  if (NamesOn(header, "layers") && AnyNamed(names.layers)) {
    out.push_back(NameLine("layers", names.layers, what));
  }
  if (NamesOn(header, "interfaces") && AnyNamed(names.interfaces)) {
    out.push_back(NameLine("interfaces", names.interfaces, what));
  }
  return out;
}

/// A deck's title where its header gives none of its own: its first header
/// line, else where it was read from, else `deck`, as the Python's writers
/// take it.
inline std::string DeckTitle(const Deck& deck) {
  if (!deck.Header().empty()) return Trimmed(deck.Header().front());
  return deck.Source() == "the deck" ? std::string{"deck"} : deck.Source();
}

/// The three header lines of a mineos deck as the Python writes them: the
/// title and flags of the deck's own mineos header where it has one (else
/// its `DeckTitle`, `ifanis` from the columns, `tref` 1 and `ifdeck` 1),
/// the knots counted from the table, `nic` and `noc` the
/// header's or the first two repeated radii's, and the trailing tokens the
/// header was read with.
inline std::vector<std::string> MineosHeaderLines(const Deck& deck) {
  const auto& r = deck.Radius();
  auto cuts = std::vector<std::size_t>{};
  for (auto i = std::size_t{1}; i < r.size(); ++i) {
    if (r[i] == r[i - 1]) cuts.push_back(i);
  }
  auto h = MineosHeader{.name = DeckTitle(deck),
                        .ifanis = deck.HasColumn("vph") ? 1 : 0,
                        .tref = 1.0,
                        .ifdeck = 1};
  h.nic = cuts.empty() ? static_cast<int>(deck.NKnots()) : static_cast<int>(cuts[0]);
  h.noc = cuts.size() > 1 ? static_cast<int>(cuts[1]) : h.nic;
  if (auto read = ParsedMineosHeader(deck.Header())) h = *std::move(read);
  auto line2 = std::format("{} {:g} {}", h.ifanis, h.tref, h.ifdeck);
  for (const auto& token : h.extras2) line2 += " " + token;
  auto line3 = std::format("{} {} {}", deck.NKnots(), h.nic, h.noc);
  for (const auto& token : h.extras3) line3 += " " + token;
  return {h.name, line2, line3};
}

}  // namespace detail

/**
 * @brief The mineos format: three header lines, then `r` and the eight
 * columns of `MINEOS_COLUMNS` or the five of `MINEOS_ISOTROPIC_COLUMNS`;
 * an absent quality factor is written as 0, which mineos reads as no loss.
 */
[[nodiscard]] inline const DeckFormat& MINEOS() {
  static const auto format = DeckFormat{
      .name = "mineos",
      .columns = {{8, MINEOS_COLUMNS}, {5, MINEOS_ISOTROPIC_COLUMNS}},
      .header_lines = 3,
      .check_header = [](const std::vector<std::string>& lines, const std::string& source) {
        static_cast<void>(ReadMineosHeader(lines, source));
      },
      .write_header = [](const Deck& deck) { return detail::MineosHeaderLines(deck); },
      .fill = {{"qkappa", 0.0}, {"qmu", 0.0}}};
  return format;
}

/**
 * @brief The Maxwell format: a title line, the line naming the columns
 * (a `#` comment may follow them), optionally `# layers` and `# interfaces`
 * lines, then `r` and the columns of `MAXWELL_COLUMNS`, NaN where a layer
 * lacks a field.
 */
[[nodiscard]] inline const DeckFormat& MAXWELL() {
  static const auto format = DeckFormat{
      .name = "maxwell",
      .columns = {{7, MAXWELL_COLUMNS}},
      .header_lines = 2,
      .header_comments = {"layers", "interfaces"},
      .check_header =
          [](const std::vector<std::string>& lines, const std::string& source) {
            const auto line = std::string_view{lines.at(1)};
            if (detail::Tokens(line.substr(0, line.find('#'))) !=
                detail::Tokens(detail::MaxwellColumnLine())) {
              throw Error(source, "header",
                          std::format("a Maxwell deck's second line names its columns, "
                                      "{}; got '{}'",
                                      detail::MaxwellColumnLine(),
                                      detail::Trimmed(lines.at(1))));
            }
            detail::OneNamingLineEach(lines, source, "a Maxwell deck");
          },
      .write_header =
          [](const Deck& deck) {
            return detail::NamedHeaderLines(deck, MAXWELL_COLUMNS, "a Maxwell deck");
          }};
  return format;
}

/**
 * @brief The columns after the radius of a prony deck of `n` branches:
 * `PRONY_COLUMNS`, then `relaxation_strength_k` and `relaxation_time_k`
 * for k = 1 to n.
 * @param n The number of branches, at least one.
 * @throws planetmodel::Error if `n` is zero.
 */
[[nodiscard]] inline std::vector<std::string> PronyColumns(std::size_t n) {
  if (n < 1) {
    throw Error("PronyColumns", "",
                std::format("a prony deck has a whole number of branches, at least one; got {}",
                            n));
  }
  auto out = PRONY_COLUMNS;
  for (auto& member : detail::Members(BRANCH_STEMS, n)) out.push_back(std::move(member));
  return out;
}

/**
 * @brief The number of branches N of a prony deck, the same on every layer:
 * the pairs of branch columns it holds, both of each pair for k = 1 to N.
 * @param deck A prony deck.
 * @throws planetmodel::Error if the deck holds other branch columns, or
 *         none.
 */
[[nodiscard]] inline std::size_t PronyBranches(const Deck& deck) {
  return detail::Groups(BRANCH_STEMS, deck.Names(), "a prony deck", deck.Source());
}

/**
 * @brief The header lines of a prony deck of `n` branches: its title, the
 * line naming the columns, and a naming line for the layers and one for the
 * interfaces where any of them is named.  A deck is written with the column
 * line of the branches it holds, whatever `n` its header lines were made
 * with.
 * @param title The deck's title.
 * @param n The number of branches, at least one.
 * @param layers The layers' names, centre outward, empty where unnamed.
 * @param interfaces The interfaces' names, centre outward.
 * @throws planetmodel::Error if `n` is zero, or a name is not one word, is
 *         `-`, or holds a `#`.
 */
[[nodiscard]] inline std::vector<std::string> PronyHeader(
    std::string_view title, std::size_t n,
    const std::vector<std::optional<std::string>>& layers = {},
    const std::vector<std::optional<std::string>>& interfaces = {}) {
  return detail::NamedHeader(title, PronyColumns(n), layers, interfaces, "a prony deck");
}

/**
 * @brief The header lines of a burgers deck: its title, the line naming the
 * columns, and a naming line for the layers and one for the interfaces
 * where any of them is named.
 * @param title The deck's title.
 * @param layers The layers' names, centre outward, empty where unnamed.
 * @param interfaces The interfaces' names, centre outward.
 * @throws planetmodel::Error if a name is not one word, is `-`, or holds a
 *         `#`.
 */
[[nodiscard]] inline std::vector<std::string> BurgersHeader(
    std::string_view title, const std::vector<std::optional<std::string>>& layers = {},
    const std::vector<std::optional<std::string>>& interfaces = {}) {
  return detail::NamedHeader(title, BURGERS_COLUMNS, layers, interfaces, "a burgers deck");
}

namespace detail {

/// Refuse a prony column line other than `r`, `PRONY_COLUMNS` and the pairs
/// of branch columns for k = 1 to N in order, N at least one, saying what is
/// wrong, as the Python does.
inline void CheckPronyColumnLine(std::string_view line, const std::string& source) {
  const auto named = Tokens(line.substr(0, line.find('#')));
  auto lead = std::vector<std::string>{"r"};
  lead.insert(lead.end(), PRONY_COLUMNS.begin(), PRONY_COLUMNS.end());
  if (named.size() < lead.size() || !std::equal(lead.begin(), lead.end(), named.begin())) {
    throw Error(source, "header",
                std::format("a prony deck's second line names its columns, {} then "
                            "relaxation_strength_k relaxation_time_k for each branch k = 1 to "
                            "N; got '{}'",
                            Joined(lead), Trimmed(line)));
  }
  const auto words = std::vector<std::string>(
      named.begin() + static_cast<std::ptrdiff_t>(lead.size()), named.end());
  if (words.empty()) {
    throw Error(source, "header",
                "a prony deck has at least one branch; its column line names none after eta");
  }
  const auto want = Members(BRANCH_STEMS, (words.size() + 1) / 2);
  constexpr auto rule = std::string_view{
      "a prony deck's branch columns come in pairs, relaxation_strength_k relaxation_time_k "
      "for k = 1, 2, ... in order"};
  for (auto i = std::size_t{0}; i < words.size(); ++i) {
    if (words[i] != want[i]) {
      throw Error(source, "header",
                  std::format("{}; '{}' stands where '{}' belongs", rule, words[i], want[i]));
    }
  }
  if (words.size() % 2 != 0) {
    throw Error(source, "header",
                std::format("{}; branch {} has no '{}'", rule, want.size() / 2, want.back()));
  }
}

/// A layer's values of one column, the rows [begin, end).
inline std::vector<double> Rows(const std::vector<double>& values, std::size_t begin,
                                std::size_t end) {
  return {values.begin() + static_cast<std::ptrdiff_t>(begin),
          values.begin() + static_cast<std::ptrdiff_t>(end)};
}

/// Refuse a prony deck as the Python's `PRONY` does: a column line (the
/// header's second, where it has one) naming other columns than its
/// branches'; a layer neither NaN in every branch
/// column (elastic) nor a number in each; a strength outside [0, 1], or
/// strengths summing to more than one past `STRENGTH_ROUNDING`; a time not
/// positive and finite.  Naming the layer, the column and the radius.
inline void CheckPronyTable(const Deck& deck) {
  const auto n = PronyBranches(deck);
  const auto& source = deck.Source();
  if (const auto& header = deck.Header(); header.size() >= 2) {
    const auto line = std::string_view{header[1]};
    const auto named = Tokens(line.substr(0, line.find('#')));
    const auto columns = PronyColumns(n);
    if (named.empty() || !std::ranges::equal(named | std::views::drop(1), columns)) {
      throw Error(source, "",
                  std::format("the column line names {} columns after the radius; the rows "
                              "have {}",
                              named.empty() ? 0 : named.size() - 1, columns.size()));
    }
  }
  const auto& r = deck.Radius();
  const auto layers = deck.Layers();
  for (auto i = std::size_t{0}; i < layers.size(); ++i) {
    const auto [begin, end] = layers[i];
    const auto where = std::format("layer {}", i);
    auto w = std::vector<std::vector<double>>{};
    auto t = std::vector<std::vector<double>>{};
    auto absent = std::size_t{0};
    for (auto k = std::size_t{1}; k <= n; ++k) {
      w.push_back(Rows(deck.Column(std::format("{}_{}", BRANCH_STEMS[0], k)), begin, end));
      t.push_back(Rows(deck.Column(std::format("{}_{}", BRANCH_STEMS[1], k)), begin, end));
      for (const auto* values : {&w.back(), &t.back()}) {
        absent += static_cast<std::size_t>(
            std::ranges::count_if(*values, [](double v) { return std::isnan(v); }));
      }
    }
    if (absent == 2 * n * (end - begin)) continue;
    if (absent > 0) {
      throw Error(source, where,
                  "holds nan in some of its branch columns and numbers in others; a layer of "
                  "a prony deck is nan in every branch column (elastic) or holds a number in "
                  "each, a strength of 0 in a branch it does not use");
    }
    for (auto k = std::size_t{0}; k < n; ++k) {
      for (auto j = std::size_t{0}; j < end - begin; ++j) {
        if (!(w[k][j] >= 0.0 && w[k][j] <= 1.0)) {
          throw Error(source, where,
                      std::format("{}_{} is {:g} at r = {:g}; a strength lies in [0, 1]",
                                  BRANCH_STEMS[0], k + 1, w[k][j], r[begin + j]));
        }
      }
    }
    for (auto k = std::size_t{0}; k < n; ++k) {
      for (auto j = std::size_t{0}; j < end - begin; ++j) {
        if (!(std::isfinite(t[k][j]) && t[k][j] > 0.0)) {
          throw Error(source, where,
                      std::format("{}_{} is {:g} at r = {:g}; a time is positive and finite",
                                  BRANCH_STEMS[1], k + 1, t[k][j], r[begin + j]));
        }
      }
    }
    for (auto j = std::size_t{0}; j < end - begin; ++j) {
      auto total = 0.0;
      for (auto k = std::size_t{0}; k < n; ++k) total += w[k][j];
      if (total > 1.0 + static_cast<double>(n) * STRENGTH_ROUNDING) {
        throw Error(source, where,
                    std::format("the strengths sum to {:.9g} at r = {:g}, more than one", total,
                                r[begin + j]));
      }
    }
  }
}

/// A prony deck as written, as the Python's `PRONY` completes it: on a layer
/// holding branches 1 to m and NaN throughout in the others, each other
/// branch is given a strength of 0 and the time of branch m at every knot.
/// Any other layer is left as it is, for `CheckPronyTable` to hold.
inline Deck CompletePronyDeck(const Deck& deck) {
  const auto n = PronyBranches(deck);
  auto columns = std::vector<std::pair<std::string, std::vector<double>>>{};
  for (const auto& name : deck.Names()) columns.emplace_back(name, deck.Column(name));
  const auto column = [&](std::string_view stem, std::size_t k) -> std::vector<double>& {
    const auto name = std::format("{}_{}", stem, k);
    return std::ranges::find(columns, name, [](const auto& c) { return c.first; })->second;
  };
  const auto finite = [](const std::vector<double>& v, std::size_t b, std::size_t e) {
    return std::none_of(v.begin() + static_cast<std::ptrdiff_t>(b),
                        v.begin() + static_cast<std::ptrdiff_t>(e),
                        [](double x) { return std::isnan(x); });
  };
  const auto missing = [](const std::vector<double>& v, std::size_t b, std::size_t e) {
    return std::all_of(v.begin() + static_cast<std::ptrdiff_t>(b),
                       v.begin() + static_cast<std::ptrdiff_t>(e),
                       [](double x) { return std::isnan(x); });
  };
  for (const auto& [begin, end] : deck.Layers()) {
    auto held = std::size_t{0};
    while (held < n && finite(column(BRANCH_STEMS[0], held + 1), begin, end) &&
           finite(column(BRANCH_STEMS[1], held + 1), begin, end)) {
      ++held;
    }
    auto rest = held > 0;
    for (auto k = held + 1; k <= n; ++k) {
      rest = rest && missing(column(BRANCH_STEMS[0], k), begin, end) &&
             missing(column(BRANCH_STEMS[1], k), begin, end);
    }
    if (!rest) continue;
    for (auto k = held + 1; k <= n; ++k) {
      auto& strength = column(BRANCH_STEMS[0], k);
      auto& time = column(BRANCH_STEMS[1], k);
      const auto& last = column(BRANCH_STEMS[1], held);
      for (auto j = begin; j < end; ++j) {
        strength[j] = 0.0;
        time[j] = last[j];
      }
    }
  }
  return Deck(deck.Radius(), std::move(columns), deck.Header(), deck.Source(), deck.Labels());
}

/// Refuse a burgers deck as the Python's `BURGERS` does: a layer holding
/// other than none of the viscosity and the Kelvin columns (elastic), the
/// viscosity alone (a Maxwell body) or all three (a Burgers body), each NaN
/// throughout the layer or nowhere in it, or holding one that is not
/// positive and finite.  Naming the layer, the column and the radius.
inline void CheckBurgersTable(const Deck& deck) {
  const auto& source = deck.Source();
  const auto& r = deck.Radius();
  const auto names = std::vector<std::string>(BURGERS_COLUMNS.end() - 3, BURGERS_COLUMNS.end());
  const auto layers = deck.Layers();
  for (auto i = std::size_t{0}; i < layers.size(); ++i) {
    const auto [begin, end] = layers[i];
    const auto where = std::format("layer {}", i);
    auto held = std::vector<std::string>{};
    for (const auto& name : names) {
      const auto values = Rows(deck.Column(name), begin, end);
      const auto absent = std::ranges::count_if(values, [](double v) { return std::isnan(v); });
      if (absent > 0 && absent < static_cast<std::ptrdiff_t>(values.size())) {
        throw Error(source, where,
                    std::format("{} is nan on part of the layer; a column is nan throughout a "
                                "layer or nowhere in it",
                                name));
      }
      if (absent == 0) held.push_back(name);
    }
    if (!(held.empty() || held == std::vector<std::string>{names.front()} || held == names)) {
      throw Error(source, where,
                  std::format("holds {}; a layer of a burgers deck holds a viscosity alone (a "
                              "Maxwell body), a viscosity and both Kelvin columns (a Burgers "
                              "body), or none of the three (elastic)",
                              ListRepr(held)));
    }
    for (const auto& name : held) {
      const auto& values = deck.Column(name);
      for (auto j = begin; j < end; ++j) {
        if (!(std::isfinite(values[j]) && values[j] > 0.0)) {
          throw Error(source, where,
                      std::format("{} is {:g} at r = {:g}; it is positive and finite", name,
                                  values[j], r[j]));
        }
      }
    }
  }
}

}  // namespace detail

/**
 * @brief The prony format: the Maxwell format's title, naming lines and
 * layering, its columns with the viscosity replaced by the pairs
 * `relaxation_strength_k relaxation_time_k` for k = 1 to N, N read from the
 * column line and the same throughout the deck (`PronyBranches`).  A layer
 * is NaN in every branch column (elastic) or a number in each, a strength
 * of 0 in a branch it does not use; every strength lies in [0, 1], their
 * sum is at most one (to `STRENGTH_ROUNDING` a branch) and every time is
 * positive and finite, on reading and on writing.  The writer completes a
 * layer holding branches 1 to m and NaN in the others with strengths of 0
 * and the time of branch m.
 */
[[nodiscard]] inline const DeckFormat& PRONY() {
  static const auto format = DeckFormat{
      .name = "prony",
      .columns = {{PRONY_COLUMNS.size(), PRONY_COLUMNS}},
      .families = BRANCH_STEMS,
      .header_lines = 2,
      .header_comments = {"layers", "interfaces"},
      .check_header =
          [](const std::vector<std::string>& lines, const std::string& source) {
            detail::CheckPronyColumnLine(lines.at(1), source);
            detail::OneNamingLineEach(lines, source, "a prony deck");
          },
      .write_header =
          [](const Deck& deck) {
            return detail::NamedHeaderLines(deck, PronyColumns(PronyBranches(deck)),
                                            "a prony deck");
          },
      .check_table = detail::CheckPronyTable,
      .complete = detail::CompletePronyDeck};
  return format;
}

/**
 * @brief The burgers format: the Maxwell format's title, naming lines and
 * layering, then `r` and the columns of `BURGERS_COLUMNS`.  A layer holds
 * none of the viscosity and the Kelvin element's two (elastic), the
 * viscosity alone (a Maxwell body) or all three (a Burgers body), each
 * positive and finite, on reading and on writing.
 */
[[nodiscard]] inline const DeckFormat& BURGERS() {
  static const auto format = DeckFormat{
      .name = "burgers",
      .columns = {{BURGERS_COLUMNS.size(), BURGERS_COLUMNS}},
      .header_lines = 2,
      .header_comments = {"layers", "interfaces"},
      .check_header =
          [](const std::vector<std::string>& lines, const std::string& source) {
            const auto line = std::string_view{lines.at(1)};
            if (detail::Tokens(line.substr(0, line.find('#'))) !=
                detail::Tokens(detail::ColumnLine(BURGERS_COLUMNS))) {
              throw Error(source, "header",
                          std::format("a burgers deck's second line names its columns, {}; "
                                      "got '{}'",
                                      detail::ColumnLine(BURGERS_COLUMNS),
                                      detail::Trimmed(lines.at(1))));
            }
            detail::OneNamingLineEach(lines, source, "a burgers deck");
          },
      .write_header =
          [](const Deck& deck) {
            return detail::NamedHeaderLines(deck, BURGERS_COLUMNS, "a burgers deck");
          },
      .check_table = detail::CheckBurgersTable};
  return format;
}

/**
 * @brief The columns of a TauP `.nd` or `.tvel` row after the depth, by
 * how many there are: TauP's Qs is the shear quality factor `qmu`, its Qp
 * `qp`.
 */
inline const auto ND_COLUMNS = std::map<std::size_t, std::vector<std::string>>{
    {5, {"vp", "vs", "rho", "qp", "qmu"}},
    {4, {"vp", "vs", "rho", "qp"}},
    {3, {"vp", "vs", "rho"}},
    {2, {"vp", "vs"}}};

/**
 * @brief TauP's labels of the boundaries every planetary model has, and
 * the interface names they give: a label names the region below it.
 */
inline const auto ND_LABELS = std::map<std::string, std::string, std::less<>>{
    {"mantle", "moho"},     {"moho", "moho"},         {"outer-core", "cmb"},
    {"cmb", "cmb"},         {"inner-core", "icb"},    {"icocb", "icb"}};

namespace detail {

/// The SI size of one unit of an `.nd` column: km/s, g/cm^3, pure numbers.
inline double NdUnit(std::string_view name) {
  return name == "qp" || name == "qmu" ? 1.0 : 1e3;
}

/// A label line in TauP's normal form: lower case, words joined by `-`.
inline std::string NdLabel(std::string_view line) {
  auto out = std::string{};
  for (const auto& word : Tokens(line)) {
    auto lower = word;
    std::ranges::transform(lower, lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    out += (out.empty() ? "" : "-") + lower;
  }
  return out;
}

/// The rows of an `.nd` table as the Python reads them: block comments
/// removed across lines, `#` and `//` comments dropped, rows `depth vp vs
/// [rho [qp [qs]]]` of one width, a label line before the first row or
/// between the two rows of a repeated depth; radii from the deepest depth,
/// the body's radius, in SI, from the centre.
inline DeckTable NdTable(const std::vector<std::string>& lines, const std::string& source) {
  auto text = std::string{};
  for (const auto& line : lines) text += line + "\n";
  for (auto open = text.find("/*"); open != std::string::npos; open = text.find("/*", open)) {
    const auto close = text.find("*/", open + 2);
    if (close == std::string::npos) break;
    text.replace(open, close + 2 - open, " ");
  }
  auto rows = std::vector<std::vector<double>>{};
  auto at = std::vector<std::pair<std::size_t, std::string>>{};
  auto in = std::istringstream{text};
  for (auto raw = std::string{}; std::getline(in, raw);) {
    auto line = std::string_view{raw};
    line = line.substr(0, std::min(line.find('#'), line.find("//")));
    const auto trimmed = Trimmed(line);
    if (trimmed.empty()) continue;
    const auto tokens = Tokens(trimmed);
    if (!Number(tokens.front())) {
      at.emplace_back(rows.size(), NdLabel(trimmed));
      continue;
    }
    auto row = std::vector<double>{};
    for (const auto& token : tokens) {
      const auto value = Number(token);
      if (!value) {
        throw Error(source, "",
                    std::format("cannot read the row '{}': a row is numbers, a label "
                                "starts with a letter",
                                trimmed));
      }
      row.push_back(*value);
    }
    rows.push_back(std::move(row));
  }
  auto widths = std::vector<std::size_t>{};
  for (const auto& row : rows) {
    if (std::ranges::find(widths, row.size()) == widths.end()) widths.push_back(row.size());
  }
  std::ranges::sort(widths);
  if (rows.empty() || widths.size() != 1 || !ND_COLUMNS.contains(widths.front() - 1)) {
    auto got = std::string{};
    for (const auto w : widths) got += (got.empty() ? "" : ", ") + std::to_string(w);
    throw Error(source, "",
                std::format("the rows of an .nd table are depth vp vs [rho [qp [qs]]], three "
                            "to six numbers, the same on every row; got widths {}",
                            got.empty() ? std::string{"none"} : "[" + got + "]"));
  }
  const auto n = rows.size();
  for (auto i = std::size_t{1}; i < n; ++i) {
    if (rows[i][0] < rows[i - 1][0]) {
      throw Error(source, "", "the depths of an .nd table must not decrease");
    }
  }
  const auto surface = rows.back()[0];
  auto out = DeckTable{};
  for (const auto& [k, label] : at) {
    if (!(k == 0 || (k < n && rows[k - 1][0] == rows[k][0]))) {
      throw Error(source, "",
                  std::format("the label '{}' sits at no boundary: a label comes before "
                              "the first row or between the two rows of a repeated depth",
                              label));
    }
    out.labels.emplace_back((surface - rows[k < n ? k : n - 1][0]) * 1e3, label);
  }
  const auto& names = ND_COLUMNS.at(widths.front() - 1);
  for (auto i = n; i-- > 0;) out.radius.push_back((surface - rows[i][0]) * 1e3);
  for (auto j = std::size_t{0}; j < names.size(); ++j) {
    auto values = std::vector<double>{};
    for (auto i = n; i-- > 0;) values.push_back(rows[i][j + 1] * NdUnit(names[j]));
    out.columns.emplace_back(names[j], std::move(values));
  }
  return out;
}

/// The rows of an `.nd` table from a deck as the Python writes them: from
/// the surface down, its labels where they sit, depth and values as
/// `%.10g`; an absent value is refused.
inline std::vector<std::string> NdTableLines(const Deck& deck,
                                             const std::vector<std::string>& names) {
  const auto& radius = deck.Radius();
  const auto top = radius.back();
  auto labels = std::map<double, std::string>{};
  for (const auto& [r, label] : deck.Labels()) labels.insert_or_assign(r, label);
  auto columns = std::vector<std::pair<const std::vector<double>*, double>>{};
  for (const auto& name : names) columns.emplace_back(&deck.Column(name), NdUnit(name));
  for (const auto& [values, unit] : columns) {
    if (std::ranges::any_of(*values, [](double v) { return std::isnan(v); })) {
      throw Error(deck.Source(), "",
                  "an .nd table holds every column on every row; the deck has absent "
                  "values");
    }
  }
  auto out = std::vector<std::string>{};
  if (const auto it = labels.find(top); it != labels.end()) {
    out.push_back(it->second);
    labels.erase(it);
  }
  for (auto i = radius.size(); i-- > 0;) {
    const auto r = radius[i];
    if (i + 1 < radius.size() && r == radius[i + 1]) {
      if (const auto it = labels.find(r); it != labels.end()) out.push_back(it->second);
    }
    auto line = std::format("{:.10g}", (top - r) / 1e3);
    for (const auto& [values, unit] : columns) line += std::format(" {:.10g}", (*values)[i] / unit);
    out.push_back(std::move(line));
  }
  return out;
}

}  // namespace detail

/**
 * @brief TauP's named discontinuities: no header, rows `depth vp vs [rho
 * [qp [qs]]]` from the surface down in km, km/s and g/cm^3, a repeated
 * depth a boundary, a label line naming the region below it, `#`, `//`
 * and block comments; read radius first in SI from the centre.
 */
[[nodiscard]] inline const DeckFormat& ND() {
  static const auto format = DeckFormat{.name = "nd",
                                        .columns = ND_COLUMNS,
                                        .read_table = detail::NdTable,
                                        .write_table = detail::NdTableLines};
  return format;
}

/**
 * @brief TauP's tabled velocities: two header lines, kept as they are (a
 * deck without two is written under its title twice), then `.nd` rows.
 */
[[nodiscard]] inline const DeckFormat& TVEL() {
  static const auto format = DeckFormat{
      .name = "tvel",
      .columns = ND_COLUMNS,
      .header_lines = 2,
      .write_header =
          [](const Deck& deck) {
            if (deck.Header().size() == 2) return deck.Header();
            const auto title = detail::DeckTitle(deck);
            return std::vector<std::string>{title, title};
          },
      .read_table = detail::NdTable,
      .write_table = detail::NdTableLines};
  return format;
}

/**
 * @brief The layer and interface names a TauP table's labels imply.
 *
 * Each label names the layer below it, its words joined by `_`, and the
 * labels of `ND_LABELS` name their interfaces too (`moho`, `cmb`, `icb`);
 * the outermost interface is `surface`.  Every other name is empty.
 *
 * @param deck A deck read as `ND()` or `TVEL()`.
 */
[[nodiscard]] inline DeckNames NdNames(const Deck& deck) {
  const auto b = deck.Boundaries();
  const auto n = deck.NLayers();
  const auto first = b.front() > 0.0 ? std::size_t{1} : std::size_t{0};
  auto out = DeckNames{.layers = std::vector<std::optional<std::string>>(n),
                       .interfaces = std::vector<std::optional<std::string>>(n + first)};
  out.interfaces.back() = "surface";
  for (const auto& [r, label] : deck.Labels()) {
    auto k = std::size_t{0};
    for (auto i = std::size_t{1}; i < b.size(); ++i) {
      if (std::abs(b[i] - r) < std::abs(b[k] - r)) k = i;
    }
    auto name = label;
    std::ranges::replace(name, '-', '_');
    out.layers[k == 0 ? n - 1 : k - 1] = name;
    if (const auto it = ND_LABELS.find(label); it != ND_LABELS.end() && k + 1 < b.size()) {
      out.interfaces[k - 1 + first] = it->second;
    }
  }
  return out;
}

/**
 * @brief A deck's text in a format, as the Python's `write_deck` writes
 * it: the header lines, then one row per knot, the radius as `%14.3f` and
 * the format's columns in order as `%16.8g`, an absent value as the
 * format's `fill` for its column, else `nan`.  The deck is completed by the
 * format's `complete` and held to its `check_table` as it is written.
 * @param given The deck.
 * @param format The format; where it admits several sets of columns, the
 *        widest the deck holds is written, and where it has families, every
 *        group the deck holds.
 * @throws planetmodel::Error if the deck lacks a column the format needs,
 *         or the header is not the format's lines and `#` lines, or as the
 *         format's `check_table` does.
 */
[[nodiscard]] inline std::string WriteDeckText(const Deck& given, const DeckFormat& format) {
  auto names = std::vector<std::string>{};
  auto found = false;
  if (!format.families.empty()) {
    names = format.columns.begin()->second;
    const auto n = detail::Groups(format.families, given.Names(),
                                  std::format("a {} deck", format.name), given.Source());
    for (auto& member : detail::Members(format.families, n)) names.push_back(std::move(member));
    found = std::ranges::all_of(names, [&](const auto& c) { return given.HasColumn(c); });
  } else {
    for (auto it = format.columns.rbegin(); it != format.columns.rend(); ++it) {
      if (std::ranges::all_of(it->second, [&](const auto& c) { return given.HasColumn(c); })) {
        names = it->second;
        found = true;
        break;
      }
    }
  }
  if (!found) {
    auto wanted = std::string{};
    for (const auto& [width, set] : format.columns) {
      wanted += (wanted.empty() ? "" : " or ") + detail::Joined(set);
    }
    throw Error(given.Source(), "",
                std::format("a {} deck needs columns {}; the deck has {}", format.name,
                            wanted, detail::Joined(given.Names())));
  }
  const auto deck = format.complete ? format.complete(given) : given;
  const auto header = format.write_header ? format.write_header(deck) : deck.Header();
  const auto extra = header.size() >= format.header_lines
                         ? std::vector<std::string>(header.begin() + static_cast<std::ptrdiff_t>(
                                                                         format.header_lines),
                                                    header.end())
                         : std::vector<std::string>{};
  if (header.size() < format.header_lines || extra.size() > format.header_comments.size() ||
      !std::ranges::all_of(extra, [&](const auto& l) { return format.IsHeaderComment(l); })) {
    throw Error(deck.Source(), "header",
                std::format("a {} deck has {} header lines{}; {} were produced", format.name,
                            format.header_lines,
                            format.header_comments.empty()
                                ? std::string{}
                                : std::format(" and up to {} `#` lines of its own",
                                              format.header_comments.size()),
                            header.size()));
  }
  if (format.check_table) {
    auto written = std::vector<std::pair<std::string, std::vector<double>>>{};
    for (const auto& name : names) written.emplace_back(name, deck.Column(name));
    format.check_table(
        Deck(deck.Radius(), std::move(written), header, deck.Source(), deck.Labels()));
  }
  auto out = std::string{};
  for (const auto& line : header) {
    auto text = std::string_view{line};
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.remove_suffix(1);
    out += text;
    out += '\n';
  }
  if (format.write_table) {
    for (const auto& line : format.write_table(deck, names)) {
      out += line;
      out += '\n';
    }
    return out;
  }
  auto columns = std::vector<std::pair<const std::vector<double>*, std::optional<double>>>{};
  for (const auto& name : names) {
    const auto it = format.fill.find(name);
    columns.emplace_back(&deck.Column(name),
                         it == format.fill.end() ? std::nullopt : std::optional{it->second});
  }
  const auto& r = deck.Radius();
  for (auto i = std::size_t{0}; i < r.size(); ++i) {
    out += std::format("{:14.3f}", r[i]);
    for (const auto& [values, fill] : columns) {
      auto v = (*values)[i];
      if (std::isnan(v) && fill) v = *fill;
      out += std::isnan(v) ? std::format("{:>16}", "nan") : std::format("{:16.8g}", v);
    }
    out += '\n';
  }
  return out;
}

/**
 * @brief Write a deck in a format, as `WriteDeckText` gives it, making the
 * file's directory where it is missing.
 * @param path The file.
 * @param deck The deck.
 * @param format The format.
 * @throws planetmodel::Error as `WriteDeckText` does, or if the file
 *         cannot be written.
 */
inline void WriteDeck(const std::filesystem::path& path, const Deck& deck,
                      const DeckFormat& format) {
  const auto text = WriteDeckText(deck, format);
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  auto file = std::ofstream(path, std::ios::binary);
  if (!file) throw Error(path.string(), "", "cannot be written");
  file << text;
  if (!file) throw Error(path.string(), "", "cannot be written");
}

/** @brief A deck's skeleton and each layer's columns as functions of the radius. */
struct DeckFunctions {
  /** @brief The deck's skeleton. */
  Skeleton skeleton;
  /**
   * @brief One map of name to function per layer; a column NaN throughout
   * a layer is absent from it.
   */
  std::vector<std::map<std::string, RadialFunction>> layers{};
};

/**
 * @brief A kind a deck's knots may be joined by in C++, by name.
 * @param kind `cubic`, `natural` or `linear`, or one of them after `log-`.
 * @throws planetmodel::Error if the kind is the Python's alone (`pchip`,
 *         `akima`, with or without `log-`) or no kind.
 */
[[nodiscard]] inline KnotKind KnotKindNamed(std::string_view kind) {
  if (kind == "cubic") return KnotKind::Cubic;
  if (kind == "natural") return KnotKind::Natural;
  if (kind == "linear") return KnotKind::Linear;
  if (kind == "log-cubic") return KnotKind::LogCubic;
  if (kind == "log-natural") return KnotKind::LogNatural;
  if (kind == "log-linear") return KnotKind::LogLinear;
  constexpr auto kinds = std::string_view{
      "cubic, natural and linear, each also after log-"};
  if (kind == "pchip" || kind == "akima" || kind == "log-pchip" || kind == "log-akima") {
    throw Error("DeckLayers", "kind",
                std::format("{} is the Python's alone; in C++ the kinds are {}", kind, kinds));
  }
  throw Error("DeckLayers", "kind", std::format("{} is not a kind; they are {}", kind, kinds));
}

/**
 * @brief The skeleton of a deck and every layer's columns, each the
 * function through the layer's knots of the kind `kinds` gives the column,
 * or else the stem of its family (`Member`), or else of `kind`, on its
 * layer with the skeleton's default tolerance: a piecewise polynomial, or
 * for a kind after `log-` the exponential of one through the logarithms
 * (`ThroughKnots`).
 * @param deck The deck.
 * @param kind How the knots are joined: `cubic`, `natural` or `linear`, or
 *        one of them after `log-`.
 * @param kinds The kind of a column joined otherwise, by its name or by
 *        its family's stem; `MAXWELL_KINDS` for a Maxwell deck,
 *        `PRONY_KINDS` for a prony deck, `BURGERS_KINDS` for a burgers deck.
 * @throws planetmodel::Error if a kind is refused, `kinds` names neither a
 *         column of the deck nor the stem of a family it holds a member of,
 *         a column is NaN on part of a layer, or a column joined after
 *         `log-` is not positive and finite at every knot of a layer
 *         holding it.
 */
[[nodiscard]] inline DeckFunctions DeckLayers(
    const Deck& deck, std::string_view kind = "cubic",
    const std::map<std::string, std::string, std::less<>>& kinds = {}) {
  const auto knot_kind = KnotKindNamed(kind);
  auto column_kinds = std::map<std::string, KnotKind, std::less<>>{};
  const auto names = deck.Names();
  const auto stem_held = [&](std::string_view stem) {
    return std::ranges::any_of(names, [&](const auto& n) {
      const auto found = Member(n);
      return found && found->first == stem;
    });
  };
  for (const auto& [name, k] : kinds) {
    column_kinds.emplace(name, KnotKindNamed(k));
    if (!deck.HasColumn(name) && !stem_held(name)) {
      throw Error("DeckLayers", "kinds",
                  std::format("names {}, a column the deck lacks", name));
    }
  }
  auto out = DeckFunctions{.skeleton = Skeleton(deck.Boundaries()), .layers = {}};
  const auto tolerance = DEFAULT_RTOL * out.skeleton.Span();
  const auto& r = deck.Radius();
  const auto slices = deck.Layers();
  for (auto i = std::size_t{0}; i < slices.size(); ++i) {
    const auto [begin, end] = slices[i];
    auto fields = std::map<std::string, RadialFunction>{};
    for (const auto& name : deck.Names()) {
      const auto& values = deck.Column(name);
      auto knots = std::vector<double>(r.begin() + static_cast<std::ptrdiff_t>(begin),
                                       r.begin() + static_cast<std::ptrdiff_t>(end));
      auto y = std::vector<double>(values.begin() + static_cast<std::ptrdiff_t>(begin),
                                   values.begin() + static_cast<std::ptrdiff_t>(end));
      const auto missing = std::ranges::count_if(y, [](double v) { return std::isnan(v); });
      if (missing == static_cast<std::ptrdiff_t>(y.size())) continue;
      if (missing > 0) {
        throw Error(deck.Source(), name,
                    std::format("is NaN on part of layer {}; a column is absent from a "
                                "layer only when NaN throughout it",
                                i));
      }
      auto joined = knot_kind;
      auto joined_name = std::string{kind};
      if (const auto own = column_kinds.find(name); own != column_kinds.end()) {
        joined = own->second;
        joined_name = kinds.find(name)->second;
      } else if (const auto found = Member(name)) {
        if (const auto stem = column_kinds.find(found->first); stem != column_kinds.end()) {
          joined = stem->second;
          joined_name = kinds.find(found->first)->second;
        }
      }
      if (IsLogarithmic(joined)) {
        const auto off =
            std::ranges::find_if(y, [](double v) { return !(std::isfinite(v) && v > 0.0); });
        if (off != y.end()) {
          const auto j = static_cast<std::size_t>(std::distance(y.begin(), off));
          throw Error(deck.Source(), name,
                      std::format("is {:g} at r = {:g} on layer {}; a column joined by {}, its "
                                  "logarithm interpolated, is positive and finite at every "
                                  "knot of a layer holding it",
                                  *off, knots[j], i, joined_name));
        }
      }
      fields.emplace(name, ThroughKnots(std::move(knots), std::move(y), joined, tolerance));
    }
    out.layers.push_back(std::move(fields));
  }
  return out;
}

}  // namespace planetmodel

#endif  // PLANETMODEL_DECK_HPP
