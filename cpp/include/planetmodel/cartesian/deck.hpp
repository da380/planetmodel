#ifndef PLANETMODEL_CARTESIAN_DECK_HPP
#define PLANETMODEL_CARTESIAN_DECK_HPP

/**
 * @file
 * @brief Decks of depths, as `planetmodel.cartesian.deck` reads and writes
 * them: the tables a flat model is read from and written to.
 *
 * A flat model's deck is a table of knots whose first column is the
 * depth, increasing down the file from the free surface at 0, one column
 * per named field, a repeated depth a boundary between two layers; the
 * layers run from the surface down and the last knot is the base, a face
 * of the box and no interface.  It is a `planetmodel::Deck` like any
 * other, its `Radius()` holding the depths, read by `ReadDeck` and written
 * by `WriteDeck` (deck.hpp), character for character as the Python writes
 * it.
 *
 * The formats are those of the radius-first decks stated for depth: a
 * title line, a line naming the columns, `depth` first, optionally
 * `# layers` and `# interfaces` lines naming them from the surface down
 * (`-` for none), then the rows in SI, NaN where a layer lacks a field.
 * `ELASTIC()` holds `depth rho vpv vsv qkappa qmu vph vsh eta` or
 * `depth rho vp vs qkappa qmu`; `MAXWELL()`, `PRONY()` and `BURGERS()` the
 * columns, the completion and the checks of `planetmodel::MAXWELL()`,
 * `PRONY()` and `BURGERS()`, a message naming a knot's depth.  A header
 * names one interface per layer, the top first (`DepthNames`), where a
 * deck of radii names one more for a hollow body's inner boundary.
 * `DeckLayers` gives each layer's columns as functions of depth, joined by
 * the kinds of deck.hpp (`MAXWELL_KINDS`, `PRONY_KINDS` and
 * `BURGERS_KINDS` join the viscosities and times in their logarithm).  A
 * deck holds no lateral extent: the box a deck's model is built over is
 * given beside it.
 */

#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "planetmodel/deck.hpp"
#include "planetmodel/error.hpp"

namespace planetmodel::cartesian {

/**
 * @brief The layer and interface names a header of depths gives: those of
 * its `# layers` and `# interfaces` lines, from the surface down, empty
 * for `-`.  Without a line the layers are unnamed, and the interfaces too
 * but for the top, `surface` where the deck starts at depth 0.
 * @param deck A deck of depths of named columns: elastic, Maxwell, prony
 *        or burgers.
 * @throws planetmodel::Error if a line names other than the deck's layers,
 *         or its interfaces (one per layer, the base being none).
 */
[[nodiscard]] inline DeckNames DepthNames(const Deck& deck) {
  const auto n = deck.NLayers();
  auto out = DeckNames{.layers = std::vector<std::optional<std::string>>(n),
                       .interfaces = std::vector<std::optional<std::string>>(n)};
  if (deck.Radius().front() == 0.0) out.interfaces.front() = "surface";
  if (auto layers = detail::NamesOn(deck.Header(), "layers")) out.layers = *std::move(layers);
  if (auto faces = detail::NamesOn(deck.Header(), "interfaces")) {
    out.interfaces = *std::move(faces);
  }
  for (const auto& [what, size] : {std::tuple{"layers", out.layers.size()},
                                   std::tuple{"interfaces", out.interfaces.size()}}) {
    if (size != n) {
      throw Error(deck.Source(), "header",
                  std::format("names {} {}; its table has {}", size, what, n));
    }
  }
  return out;
}

/**
 * @brief The header lines of an elastic deck of depths: its title, the
 * line naming the columns, and a naming line for the layers and one for
 * the interfaces where any of them is named.
 * @param title The deck's title.
 * @param isotropic Whether the columns are the isotropic ones, `rho vp vs
 *        qkappa qmu`, not the transversely isotropic `MINEOS_COLUMNS`.
 * @param layers The layers' names, from the surface down, empty where
 *        unnamed.
 * @param interfaces The interfaces' names, from the surface down.
 * @throws planetmodel::Error if a name is not one word, is `-`, or holds a
 *         `#`.
 */
[[nodiscard]] inline std::vector<std::string> ElasticHeader(
    std::string_view title, bool isotropic = false,
    const std::vector<std::optional<std::string>>& layers = {},
    const std::vector<std::optional<std::string>>& interfaces = {}) {
  return detail::NamedHeader(title, isotropic ? MINEOS_ISOTROPIC_COLUMNS : MINEOS_COLUMNS,
                             layers, interfaces, "an elastic deck", "depth");
}

/**
 * @brief The header lines of a Maxwell deck of depths.
 * @param title The deck's title.
 * @param layers The layers' names, from the surface down, empty where
 *        unnamed.
 * @param interfaces The interfaces' names, from the surface down.
 * @throws planetmodel::Error if a name is not one word, is `-`, or holds a
 *         `#`.
 */
[[nodiscard]] inline std::vector<std::string> MaxwellHeader(
    std::string_view title, const std::vector<std::optional<std::string>>& layers = {},
    const std::vector<std::optional<std::string>>& interfaces = {}) {
  return detail::NamedHeader(title, MAXWELL_COLUMNS, layers, interfaces, "a Maxwell deck",
                             "depth");
}

/**
 * @brief The header lines of a prony deck of depths of `n` branches.
 * @param title The deck's title.
 * @param n The number of branches, at least one.
 * @param layers The layers' names, from the surface down, empty where
 *        unnamed.
 * @param interfaces The interfaces' names, from the surface down.
 * @throws planetmodel::Error if `n` is zero, or a name is not one word, is
 *         `-`, or holds a `#`.
 */
[[nodiscard]] inline std::vector<std::string> PronyHeader(
    std::string_view title, std::size_t n,
    const std::vector<std::optional<std::string>>& layers = {},
    const std::vector<std::optional<std::string>>& interfaces = {}) {
  return detail::NamedHeader(title, PronyColumns(n), layers, interfaces, "a prony deck",
                             "depth");
}

/**
 * @brief The header lines of a burgers deck of depths.
 * @param title The deck's title.
 * @param layers The layers' names, from the surface down, empty where
 *        unnamed.
 * @param interfaces The interfaces' names, from the surface down.
 * @throws planetmodel::Error if a name is not one word, is `-`, or holds a
 *         `#`.
 */
[[nodiscard]] inline std::vector<std::string> BurgersHeader(
    std::string_view title, const std::vector<std::optional<std::string>>& layers = {},
    const std::vector<std::optional<std::string>>& interfaces = {}) {
  return detail::NamedHeader(title, BURGERS_COLUMNS, layers, interfaces, "a burgers deck",
                             "depth");
}

/**
 * @brief The elastic format of depths: a title line, the line naming the
 * columns (`depth`, then `MINEOS_COLUMNS` or `MINEOS_ISOTROPIC_COLUMNS`; a
 * `#` comment may follow them), optionally `# layers` and `# interfaces`
 * lines, then the rows in SI, NaN where a layer lacks a field (a quality
 * factor where it attenuates nothing).  The column line and the width of
 * the rows agree; a deck is written with the widest set it holds.
 */
[[nodiscard]] inline const DeckFormat& ELASTIC() {
  static const auto format = DeckFormat{
      .name = "elastic",
      .columns = {{8, MINEOS_COLUMNS}, {5, MINEOS_ISOTROPIC_COLUMNS}},
      .header_lines = 2,
      .header_comments = {"layers", "interfaces"},
      .check_header =
          [](const std::vector<std::string>& lines, const std::string& source) {
            detail::CheckColumnLine(lines, source, "an elastic deck",
                                    {MINEOS_COLUMNS, MINEOS_ISOTROPIC_COLUMNS}, "depth");
            detail::OneNamingLineEach(lines, source, "an elastic deck");
          },
      .write_header =
          [](const Deck& deck) {
            const auto full = std::ranges::all_of(
                MINEOS_COLUMNS, [&](const auto& name) { return deck.HasColumn(name); });
            return detail::NamedHeaderLines(deck,
                                            full ? MINEOS_COLUMNS : MINEOS_ISOTROPIC_COLUMNS,
                                            "an elastic deck", DepthNames(deck), "depth");
          },
      .check_table =
          [](const Deck& deck) {
            // the column line, where the header has one, names the rows' columns
            const auto& header = deck.Header();
            if (header.size() < 2) return;
            const auto line = std::string_view{header[1]};
            const auto named = detail::Tokens(line.substr(0, line.find('#')));
            const auto columns = deck.Names();
            if (named.empty() || !std::ranges::equal(named | std::views::drop(1), columns)) {
              throw Error(deck.Source(), "",
                          std::format("the column line names {} columns after the depth; the "
                                      "rows have {}",
                                      named.empty() ? 0 : named.size() - 1, columns.size()));
            }
          },
      .coordinate = "depth"};
  return format;
}

/**
 * @brief The Maxwell format of depths: `planetmodel::MAXWELL()` with
 * `depth` for `r`, its naming lines from the surface down.
 */
[[nodiscard]] inline const DeckFormat& MAXWELL() {
  static const auto format = DeckFormat{
      .name = "maxwell",
      .columns = {{7, MAXWELL_COLUMNS}},
      .header_lines = 2,
      .header_comments = {"layers", "interfaces"},
      .check_header =
          [](const std::vector<std::string>& lines, const std::string& source) {
            detail::CheckColumnLine(lines, source, "a Maxwell deck", {MAXWELL_COLUMNS}, "depth");
            detail::OneNamingLineEach(lines, source, "a Maxwell deck");
          },
      .write_header =
          [](const Deck& deck) {
            return detail::NamedHeaderLines(deck, MAXWELL_COLUMNS, "a Maxwell deck",
                                            DepthNames(deck), "depth");
          },
      .coordinate = "depth"};
  return format;
}

/**
 * @brief The prony format of depths: `planetmodel::PRONY()` with `depth`
 * for `r`, its naming lines from the surface down, its checks and its
 * completion the same.
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
            detail::CheckPronyColumnLine(lines.at(1), source, "depth");
            detail::OneNamingLineEach(lines, source, "a prony deck");
          },
      .write_header =
          [](const Deck& deck) {
            const auto columns = PronyColumns(PronyBranches(deck));
            return detail::NamedHeaderLines(deck, columns, "a prony deck", DepthNames(deck),
                                            "depth");
          },
      .check_table = [](const Deck& deck) { detail::CheckPronyTable(deck, "depth", "depth"); },
      .complete = detail::CompletePronyDeck,
      .coordinate = "depth"};
  return format;
}

/**
 * @brief The burgers format of depths: `planetmodel::BURGERS()` with
 * `depth` for `r`, its naming lines from the surface down, its checks the
 * same.
 */
[[nodiscard]] inline const DeckFormat& BURGERS() {
  static const auto format = DeckFormat{
      .name = "burgers",
      .columns = {{BURGERS_COLUMNS.size(), BURGERS_COLUMNS}},
      .header_lines = 2,
      .header_comments = {"layers", "interfaces"},
      .check_header =
          [](const std::vector<std::string>& lines, const std::string& source) {
            detail::CheckColumnLine(lines, source, "a burgers deck", {BURGERS_COLUMNS}, "depth");
            detail::OneNamingLineEach(lines, source, "a burgers deck");
          },
      .write_header =
          [](const Deck& deck) {
            return detail::NamedHeaderLines(deck, BURGERS_COLUMNS, "a burgers deck",
                                            DepthNames(deck), "depth");
          },
      .check_table = [](const Deck& deck) { detail::CheckBurgersTable(deck, "depth"); },
      .coordinate = "depth"};
  return format;
}

/**
 * @brief The skeleton of a deck of depths and every layer's columns, from
 * the surface down, each the function of depth through the layer's knots:
 * `planetmodel::DeckLayers`, a message naming a knot's depth.
 * @param deck The deck of depths.
 * @param kind How the knots are joined: `cubic`, `natural` or `linear`, or
 *        one of them after `log-`.
 * @param kinds The kind of a column joined otherwise, by its name or by
 *        its family's stem; `MAXWELL_KINDS` for a Maxwell deck,
 *        `PRONY_KINDS` for a prony deck, `BURGERS_KINDS` for a burgers deck.
 * @throws planetmodel::Error as `planetmodel::DeckLayers` does.
 */
[[nodiscard]] inline DeckFunctions DeckLayers(
    const Deck& deck, std::string_view kind = "cubic",
    const std::map<std::string, std::string, std::less<>>& kinds = {}) {
  return detail::DeckLayersOf(deck, kind, kinds, "depth");
}

}  // namespace planetmodel::cartesian

#endif  // PLANETMODEL_CARTESIAN_DECK_HPP
