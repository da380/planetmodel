// Slices in time: the supports and rules by name and back, and a date
// within the tolerance of a stored time read as that time, the cases of
// the Python's own test of `time_rows`.

#include <gtest/gtest.h>

#include <cmath>
#include <format>
#include <limits>
#include <optional>
#include <vector>

#include "planetmodel/time.hpp"

namespace {

namespace pm = planetmodel;
using pm::Extrapolation;
using pm::TimeRows;
using pm::TimeRule;
using pm::TimeSupport;

const auto TIME = std::optional<std::vector<double>>{std::vector{0.0, 1.0, 3.0}};
const auto INSTANT = std::optional<std::vector<double>>{std::vector{2.0, 5.0}};

TimeRows Rows(TimeSupport support, TimeRule rule, double date, double rtol = pm::TIME_RTOL,
              Extrapolation extrapolation = Extrapolation::Refuse) {
  return *pm::TimeRowsAt(support, rule, date, TIME, INSTANT, extrapolation, "test", "slip", rtol);
}

TimeRows Node(std::size_t k) { return {{k, 1.0}, {k, 0.0}}; }

}  // namespace

TEST(Time, TheSupportsAndRulesAreTheirNames) {
  for (const auto support :
       {TimeSupport::Static, TimeSupport::Node, TimeSupport::Interval, TimeSupport::Instant}) {
    EXPECT_EQ(pm::TimeSupportNamed(pm::TimeSupportName(support)), support);
  }
  for (const auto rule : {TimeRule::Linear, TimeRule::Previous, TimeRule::None}) {
    EXPECT_EQ(pm::TimeRuleNamed(pm::TimeRuleName(rule)), rule);
  }
  EXPECT_EQ(pm::TimeSupportName(TimeSupport::Node), "node");
  EXPECT_EQ(pm::TimeRuleName(TimeRule::Previous), "previous");
  EXPECT_EQ(std::format("{} {}", TimeSupport::Interval, TimeRule::Linear), "interval linear");
  EXPECT_FALSE(pm::TimeSupportNamed("nodes"));
  EXPECT_FALSE(pm::TimeRuleNamed("cubic"));
  static_assert(pm::TimeSupportNamed("instant") == TimeSupport::Instant);
  // by name, as a file states them, the same rows
  EXPECT_EQ(
      pm::TimeRowsAt("node", "linear", 2.0, TIME, INSTANT, Extrapolation::Refuse, "test", "slip"),
      Rows(TimeSupport::Node, TimeRule::Linear, 2.0));
  EXPECT_THROW(static_cast<void>(pm::TimeRowsAt("knot", "linear", 2.0, TIME, INSTANT,
                                                Extrapolation::Refuse, "test", "slip")),
               pm::Error);
}

TEST(Time, ADateWithinTheToleranceOfAStoredTimeIsThatTime) {
  EXPECT_EQ(pm::TIME_RTOL, 1e-12);
  const auto near = 1e-12, far = 1e-11;
  // served at the stored times alone
  EXPECT_EQ(Rows(TimeSupport::Node, TimeRule::None, 1.0 + near), Node(1));
  EXPECT_EQ(Rows(TimeSupport::Node, TimeRule::None, 1.0 - near), Node(1));
  EXPECT_THROW(static_cast<void>(Rows(TimeSupport::Node, TimeRule::None, 1.0 + far)), pm::Error);
  EXPECT_THROW(static_cast<void>(Rows(TimeSupport::Node, TimeRule::None, 1.0 + near, 0.0)),
               pm::Error);
  EXPECT_EQ(Rows(TimeSupport::Instant, TimeRule::None, 5.0 + near), Node(1));
  EXPECT_THROW(static_cast<void>(Rows(TimeSupport::Instant, TimeRule::None, 5.0 + near, 0.0)),
               pm::Error);
  // read from a node a rounding error below it
  EXPECT_EQ(Rows(TimeSupport::Node, TimeRule::Previous, 1.0 - near), Node(1));
  EXPECT_EQ(Rows(TimeSupport::Node, TimeRule::Previous, 1.0 - near, 0.0), Node(0));
  EXPECT_EQ(Rows(TimeSupport::Interval, TimeRule::None, 1.0 - near), Node(1));
  EXPECT_EQ(Rows(TimeSupport::Interval, TimeRule::None, 1.0 - near, 0.0), Node(0));
  // a node's own values alone, weights exact
  EXPECT_EQ(Rows(TimeSupport::Node, TimeRule::Linear, 1.0 + near), Node(1));
  const auto [first, second] = Rows(TimeSupport::Node, TimeRule::Linear, 1.0 + near, 0.0);
  EXPECT_EQ(first.first, 1u);
  EXPECT_EQ(second.first, 2u);
  EXPECT_GT(second.second, 0.0);
  EXPECT_LT(second.second, 1e-11);
  // the ends of the span
  EXPECT_EQ(Rows(TimeSupport::Node, TimeRule::Linear, 3.0 + near), Node(2));
  EXPECT_EQ(Rows(TimeSupport::Node, TimeRule::Linear, -near), Node(0));
  EXPECT_THROW(static_cast<void>(Rows(TimeSupport::Node, TimeRule::Linear, 3.0 + far)), pm::Error);
  EXPECT_THROW(static_cast<void>(Rows(TimeSupport::Node, TimeRule::Linear, 3.0 + near, 0.0)),
               pm::Error);
  EXPECT_EQ(
      Rows(TimeSupport::Node, TimeRule::Linear, 3.0 + far, pm::TIME_RTOL, Extrapolation::Hold),
      Node(2));
  // a lone time: its magnitude is the scale
  const auto lone = std::optional<std::vector<double>>{std::vector{4.0}};
  auto at = [&lone](double date) {
    return pm::TimeRowsAt(TimeSupport::Node, TimeRule::None, date, lone, std::nullopt,
                          Extrapolation::Refuse, "test", "slip");
  };
  EXPECT_EQ(at(4.0 + 3e-12), Node(0));
  EXPECT_THROW(static_cast<void>(at(4.0 + 5e-12)), pm::Error);
  // the nearest within the tolerance, the earlier on a tie
  const auto pair = std::optional<std::vector<double>>{std::vector{0.0, 1.0}};
  auto wide = [&pair](double date) {
    return pm::TimeRowsAt(TimeSupport::Node, TimeRule::None, date, pair, std::nullopt,
                          Extrapolation::Refuse, "test", "slip", 0.5);
  };
  EXPECT_EQ(wide(0.5), Node(0));
  EXPECT_EQ(wide(0.6), Node(1));
  for (const auto bad : {-1e-12, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(static_cast<void>(Rows(TimeSupport::Node, TimeRule::Linear, 1.0, bad)), pm::Error);
  }
}
