#ifndef PLANETMODEL_TIME_HPP
#define PLANETMODEL_TIME_HPP

/**
 * @file
 * @brief Slices in time: the supports, the rules between nodes, and the
 * rows a date reads, as `planetmodel.surfacefile` defines them.
 *
 * Whatever holds values in slices on a time axis (a surface file's
 * fields, a fault's slip) states a support and a rule and reads them by
 * `TimeRowsAt`: a `node` value interpolated between its nodes
 * (`linear`), read from the latest node not after the date
 * (`previous`), or served at its stored nodes alone (`none`); an
 * `interval` value from the interval holding the date, the last closed
 * at its end; an `instant` value at its stored instants alone; a
 * `static` value at any date.  Outside its span a value refuses the
 * date, or holds its nearest value under `Extrapolation::Hold`.  The
 * arithmetic is the Python `time_rows`'s, so the two read slices
 * identically.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "planetmodel/error.hpp"

namespace planetmodel {

/** @brief How values sit in time; their leading dimension follows. */
inline constexpr auto TIME_SUPPORTS =
    std::array<std::string_view, 4>{"static", "node", "interval", "instant"};

/** @brief The time rules a `node` value may state; every other support
 * states `none`. */
inline constexpr auto TIME_METHODS = std::array<std::string_view, 3>{"linear", "previous", "none"};

/** @brief What a value does with a date outside its span. */
enum class Extrapolation {
  Refuse, /**< The date is refused. */
  Hold    /**< The nearest value of the span is served. */
};

/** @brief One leading row a date reads, and its weight. */
using TimeRow = std::pair<std::size_t, double>;

/** @brief The two rows a date reads: the value at the date is the first
 * row's weight times its row plus the second's. */
using TimeRows = std::pair<TimeRow, TimeRow>;

/**
 * @brief The leading rows a date reads, with their weights, by a
 * support's time rule (file comment); a date on a stored node reads that
 * node alone, its weight one and the other row's zero.
 * @param support One of `TIME_SUPPORTS`.
 * @param rule One of `TIME_METHODS`, `none` for every support but `node`.
 * @param date The date, in the values' units; a `static` value takes none.
 * @param time The axis of `node` and `interval` values.
 * @param instant The axis of `instant` values.
 * @param extrapolation What to do outside the span.
 * @param source What the error names as the source.
 * @param name What the error names as the place: the value read.
 * @return The rows, or empty for a `static` value, whose values are the
 *         answer whole.
 * @throws planetmodel::Error where a dated value is given no date, or the
 *         date is refused.
 */
[[nodiscard]] inline std::optional<TimeRows> TimeRowsAt(
    std::string_view support, std::string_view rule, std::optional<double> date,
    const std::optional<std::vector<double>>& time,
    const std::optional<std::vector<double>>& instant, Extrapolation extrapolation,
    std::string_view source, std::string_view name) {
  auto fail = [&source, &name](std::string_view message) { throw Error(source, name, message); };
  if (support == "static") return std::nullopt;
  if (!date) fail(std::format("is a {} field and needs a date", support));
  auto exact = [&fail](const std::vector<double>& axis, double at, std::string_view what) {
    const auto it = std::ranges::find(axis, at);
    if (it == axis.end()) fail(what);
    const auto k = static_cast<std::size_t>(it - axis.begin());
    return TimeRows{TimeRow{k, 1.0}, TimeRow{k, 0.0}};
  };
  if (support == "instant") {
    if (!instant) fail("is held at instants, and no instant is given");
    return exact(*instant, *date, "is held at its stored instants alone, and the date is not one");
  }
  if (!time) fail("is on the time axis, and no time is given");
  const auto& axis = *time;
  if (support == "node" && rule == "none") {
    return exact(axis, *date, "states no time rule and is held at its stored nodes alone");
  }
  auto at = *date;
  if (at < axis.front() || at > axis.back()) {
    if (extrapolation != Extrapolation::Hold) {
      fail(std::format("is not sampled at {}, outside its span [{}, {}]", at, axis.front(),
                       axis.back()));
    }
    at = std::clamp(at, axis.front(), axis.back());
  }
  const auto after = static_cast<std::size_t>(std::ranges::upper_bound(axis, at) - axis.begin());
  if (support == "interval" || rule == "previous") {
    const auto last = support == "interval" ? axis.size() - 2 : axis.size() - 1;
    return TimeRows{TimeRow{std::min(after - 1, last), 1.0},
                    TimeRow{std::min(after - 1, last), 0.0}};
  }
  if (axis.size() == 1) return TimeRows{TimeRow{0, 1.0}, TimeRow{0, 0.0}};
  const auto k = std::min(after - 1, axis.size() - 2);
  const auto w = (at - axis[k]) / (axis[k + 1] - axis[k]);
  if (w == 0.0) return TimeRows{TimeRow{k, 1.0}, TimeRow{k, 0.0}};
  if (w == 1.0) return TimeRows{TimeRow{k + 1, 1.0}, TimeRow{k + 1, 0.0}};
  return TimeRows{TimeRow{k, 1.0 - w}, TimeRow{k + 1, w}};
}

}  // namespace planetmodel

#endif  // PLANETMODEL_TIME_HPP
