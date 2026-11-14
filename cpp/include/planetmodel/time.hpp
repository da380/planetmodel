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
 * date, or holds its nearest value under `Extrapolation::Hold`.
 *
 * A date within a tolerance of a stored time is that time: the tolerance
 * is `time_rtol` (`TIME_RTOL` by default, zero for exact matching) times
 * the span of the axis, last time less first, or the magnitude of its
 * one time where it has one.  So a date a rounding error from a node
 * reads that node alone, under every rule (`none` and `instant` serve
 * it, `previous` and `interval` read from it), and a date that close
 * beyond an end of the span is that end.  The nearest stored time within
 * the tolerance is taken, the earlier on a tie.  The arithmetic is the
 * Python `time_rows`'s, so the two read slices identically.
 *
 * The supports and rules are the enumerations `TimeSupport` and
 * `TimeRule`; their names, as files state them, are `TIME_SUPPORTS` and
 * `TIME_METHODS` in the same order (`TimeSupportName`,
 * `TimeSupportNamed` and their `TimeRule` twins convert), and
 * `std::format` writes either as its name.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "planetmodel/error.hpp"

namespace planetmodel {

/** @brief How values sit in time; their leading dimension follows. */
enum class TimeSupport {
  Static,   /**< One value, at any date. */
  Node,     /**< A value a time node. */
  Interval, /**< A value an interval between nodes. */
  Instant   /**< A value an instant of a second axis. */
};

/** @brief The rule a `node` value is read by between nodes; every other
 * support states `None`. */
enum class TimeRule {
  Linear,   /**< Interpolated linearly between nodes. */
  Previous, /**< The latest node not after the date. */
  None      /**< Served at the stored nodes alone. */
};

/** @brief The names of the supports, as files state them, in the order of
 * `TimeSupport`. */
inline constexpr auto TIME_SUPPORTS =
    std::array<std::string_view, 4>{"static", "node", "interval", "instant"};

/** @brief The names of the time rules, as files state them, in the order
 * of `TimeRule`. */
inline constexpr auto TIME_METHODS = std::array<std::string_view, 3>{"linear", "previous", "none"};

/** @brief The tolerance of a date on a stored time by default, relative to
 * the span of the axis (file comment). */
inline constexpr double TIME_RTOL = 1e-12;

/**
 * @brief A support's name, as files state it.
 * @param support The support.
 */
[[nodiscard]] constexpr std::string_view TimeSupportName(TimeSupport support) noexcept {
  return TIME_SUPPORTS[static_cast<std::size_t>(support)];
}

/**
 * @brief A time rule's name, as files state it.
 * @param rule The rule.
 */
[[nodiscard]] constexpr std::string_view TimeRuleName(TimeRule rule) noexcept {
  return TIME_METHODS[static_cast<std::size_t>(rule)];
}

/**
 * @brief The support of a name, or empty where the name is not one of
 * `TIME_SUPPORTS`.
 * @param name The name.
 */
[[nodiscard]] constexpr std::optional<TimeSupport> TimeSupportNamed(
    std::string_view name) noexcept {
  for (auto k = std::size_t{0}; k < TIME_SUPPORTS.size(); ++k) {
    if (TIME_SUPPORTS[k] == name) return static_cast<TimeSupport>(k);
  }
  return std::nullopt;
}

/**
 * @brief The time rule of a name, or empty where the name is not one of
 * `TIME_METHODS`.
 * @param name The name.
 */
[[nodiscard]] constexpr std::optional<TimeRule> TimeRuleNamed(std::string_view name) noexcept {
  for (auto k = std::size_t{0}; k < TIME_METHODS.size(); ++k) {
    if (TIME_METHODS[k] == name) return static_cast<TimeRule>(k);
  }
  return std::nullopt;
}

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

namespace detail {

/// The stored time nearest a date within `tolerance` of it, the earlier
/// on a tie; empty where none is that close.
inline std::optional<std::size_t> StoredTimeAt(const std::vector<double>& axis, double at,
                                               double tolerance) {
  const auto after = static_cast<std::size_t>(std::ranges::upper_bound(axis, at) - axis.begin());
  auto best = std::optional<std::size_t>{};
  auto least = tolerance;
  for (const auto k : {after - 1, after}) {
    if (k >= axis.size()) continue;  // after - 1 wraps where after is 0
    const auto gap = std::abs(axis[k] - at);
    if (gap <= least && !(best && gap == least)) {
      best = k;
      least = gap;
    }
  }
  return best;
}

/// A date's tolerance on an axis: `rtol` times its span, or the magnitude
/// of its one time.
inline double TimeTolerance(const std::vector<double>& axis, double rtol) {
  const auto span = axis.back() - axis.front();
  return rtol * (span > 0.0 ? span : std::abs(axis.front()));
}

}  // namespace detail

/**
 * @brief The leading rows a date reads, with their weights, by a
 * support's time rule (file comment); a date on a stored node, or within
 * the tolerance of one, reads that node alone, its weight one and the
 * other row's zero.
 * @param support The support.
 * @param rule The rule, `TimeRule::None` for every support but `node`.
 * @param date The date, in the values' units; a `static` value takes
 *        none, and one given is not read.
 * @param time The axis of `node` and `interval` values.
 * @param instant The axis of `instant` values.
 * @param extrapolation What to do outside the span.
 * @param source What the error names as the source.
 * @param name What the error names as the place: the value read.
 * @param time_rtol The tolerance of a date on a stored time, relative to
 *        the span of the axis (file comment); zero matches exactly.
 * @return The rows, or empty for a `static` value, whose values are the
 *         answer whole.
 * @throws planetmodel::Error where a dated value is given no date, the
 *         date is refused, or the tolerance is negative or not finite.
 */
[[nodiscard]] inline std::optional<TimeRows> TimeRowsAt(
    TimeSupport support, TimeRule rule, std::optional<double> date,
    const std::optional<std::vector<double>>& time,
    const std::optional<std::vector<double>>& instant, Extrapolation extrapolation,
    std::string_view source, std::string_view name, double time_rtol = TIME_RTOL) {
  auto fail = [&source, &name](std::string_view message) { throw Error(source, name, message); };
  if (!(std::isfinite(time_rtol) && time_rtol >= 0.0)) {
    fail(std::format("time_rtol must be a finite number, not negative, got {}", time_rtol));
  }
  if (support == TimeSupport::Static) return std::nullopt;
  if (!date) fail(std::format("is a {} field and needs a date", TimeSupportName(support)));
  auto exact = [&fail, time_rtol](const std::vector<double>& axis, double at,
                                  std::string_view what) {
    const auto k = detail::StoredTimeAt(axis, at, detail::TimeTolerance(axis, time_rtol));
    if (!k) fail(what);
    return TimeRows{TimeRow{*k, 1.0}, TimeRow{*k, 0.0}};
  };
  if (support == TimeSupport::Instant) {
    if (!instant) fail("is held at instants, and no instant is given");
    return exact(*instant, *date, "is held at its stored instants alone, and the date is not one");
  }
  if (!time) fail("is on the time axis, and no time is given");
  const auto& axis = *time;
  if (support == TimeSupport::Node && rule == TimeRule::None) {
    return exact(axis, *date, "states no time rule and is held at its stored nodes alone");
  }
  auto at = *date;
  if (const auto k = detail::StoredTimeAt(axis, at, detail::TimeTolerance(axis, time_rtol))) {
    at = axis[*k];
  }
  if (at < axis.front() || at > axis.back()) {
    if (extrapolation != Extrapolation::Hold) {
      fail(std::format("is not sampled at {}, outside its span [{}, {}]", at, axis.front(),
                       axis.back()));
    }
    at = std::clamp(at, axis.front(), axis.back());
  }
  const auto after = static_cast<std::size_t>(std::ranges::upper_bound(axis, at) - axis.begin());
  if (support == TimeSupport::Interval || rule == TimeRule::Previous) {
    const auto last = support == TimeSupport::Interval ? axis.size() - 2 : axis.size() - 1;
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

/**
 * @brief The leading rows a date reads, the support and rule given by
 * name, as a file states them (the overload above).
 * @param support One of `TIME_SUPPORTS`.
 * @param rule One of `TIME_METHODS`, `none` for every support but `node`.
 * @param date The date, in the values' units; a `static` value takes none.
 * @param time The axis of `node` and `interval` values.
 * @param instant The axis of `instant` values.
 * @param extrapolation What to do outside the span.
 * @param source What the error names as the source.
 * @param name What the error names as the place: the value read.
 * @param time_rtol The tolerance of a date on a stored time, relative to
 *        the span of the axis; zero matches exactly.
 * @return The rows, or empty for a `static` value.
 * @throws planetmodel::Error for a support or rule that is not one, and as
 *         the overload above.
 */
[[nodiscard]] inline std::optional<TimeRows> TimeRowsAt(
    std::string_view support, std::string_view rule, std::optional<double> date,
    const std::optional<std::vector<double>>& time,
    const std::optional<std::vector<double>>& instant, Extrapolation extrapolation,
    std::string_view source, std::string_view name, double time_rtol = TIME_RTOL) {
  const auto s = TimeSupportNamed(support);
  const auto r = TimeRuleNamed(rule);
  if (!s) throw Error(source, name, std::format("time support {} is not one", support));
  if (!r) throw Error(source, name, std::format("time rule {} is not one", rule));
  return TimeRowsAt(*s, *r, date, time, instant, extrapolation, source, name, time_rtol);
}

}  // namespace planetmodel

/** @brief A time support formats as its name (`TimeSupportName`). */
template <>
struct std::formatter<planetmodel::TimeSupport> : std::formatter<std::string_view> {
  /**
   * @brief Format a support as its name.
   * @param support The support.
   * @param context The format context.
   */
  auto format(planetmodel::TimeSupport support, std::format_context& context) const {
    return std::formatter<std::string_view>::format(planetmodel::TimeSupportName(support),
                                                    context);
  }
};

/** @brief A time rule formats as its name (`TimeRuleName`). */
template <>
struct std::formatter<planetmodel::TimeRule> : std::formatter<std::string_view> {
  /**
   * @brief Format a rule as its name.
   * @param rule The rule.
   * @param context The format context.
   */
  auto format(planetmodel::TimeRule rule, std::format_context& context) const {
    return std::formatter<std::string_view>::format(planetmodel::TimeRuleName(rule), context);
  }
};

#endif  // PLANETMODEL_TIME_HPP
