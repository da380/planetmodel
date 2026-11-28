#ifndef PLANETMODEL_RULES_HPP
#define PLANETMODEL_RULES_HPP

/**
 * @file
 * @brief The rules by which a model file's values are read between its
 * nodes, as `planetmodel.lateral` and `RadialMesh.to_ppoly` define them.
 *
 * **Radially**, a variable is held at the Gauss-Lobatto-Legendre nodes of
 * elements, the radii stored in the file, each element's nodes its own so
 * that an element boundary and every interface are two nodes at one
 * radius.  Within an element the rule `polynomial` reads the Lagrange
 * polynomial through the element's nodes and `linear` the straight line
 * between consecutive nodes; at a shared radius the element above
 * answers, and the outermost element answers at the outer end.  The rules
 * in the logarithm, `log-polynomial` and `log-linear`, read positive
 * values: on element e the value is y_e exp(P(r)), y_e the element's first
 * value and P the polynomial or the lines through log(y / y_e) at its
 * nodes, so that an element of one value reads it to the bit.
 *
 * **Laterally**, values are held on an angular grid.  Both local rules
 * read a padded grid: the wrap column, a copy of the first, one period
 * after it, and a row at each pole the grid lacks, constant along it at
 * the pole's value, which the file stores.  `linear` is the bilinear
 * interpolant of the padded grid; `cubic` the tensor-product cubic spline
 * periodic in longitude and, in colatitude, clamped at each pole to the
 * slopes a cos(phi) + b sin(phi), one per column, with (a, b) fitted by
 * least squares to the slopes the not-a-knot spline through each column
 * has there, so that the interpolant is continuously differentiable
 * everywhere, the poles and the seam included.  A query's longitude is
 * reduced from the grid's first by `Wrapped`.  `harmonic`, which reads a
 * spherical-harmonic expansion of the grid, is the Python's alone and is
 * refused by name.
 *
 * The interpolants are built once and evaluate with `const` methods that
 * share nothing mutable, so they are safe to use from many threads.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <numbers>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <Interpolation/BicubicSpline.hpp>
#include <Interpolation/Bilinear.hpp>
#include <Interpolation/CubicSpline.hpp>
#include <Interpolation/Lagrange.hpp>

#include "planetmodel/error.hpp"
#include "planetmodel/frames.hpp"
#include "planetmodel/grid.hpp"

/** @brief Defined where this header provides the model file's radial rules
 * in the logarithm, `RadialRule::LogPolynomial` and `RadialRule::LogLinear`
 * (`IsLogRule`, `BaseRule`). */
#define PLANETMODEL_HAS_LOG_RULES 1

namespace planetmodel {

/** @brief How a variable is read between the radial nodes of an element. */
enum class RadialRule {
  Polynomial,    /**< The Lagrange polynomial through the element's nodes. */
  Linear,        /**< The straight line between consecutive nodes. */
  LogPolynomial, /**< The exponential of the Lagrange polynomial through the logarithms. */
  LogLinear      /**< The exponential of the straight lines through the logarithms. */
};

/** @brief How a variable is read between the nodes of an angular grid. */
enum class LateralRule {
  Linear, /**< The bilinear interpolant of the padded grid. */
  Cubic   /**< The bicubic spline, clamped at the poles, periodic in longitude. */
};

/** @brief The radial rules a model file may state, in the order of `RadialRule`. */
inline constexpr auto RADIAL_RULES = std::array<std::string_view, 4>{
    "polynomial", "linear", "log-polynomial", "log-linear"};

/**
 * @brief The lateral rules a model file may state; `harmonic` is the
 * Python's alone, and `LateralRuleNamed` refuses it.
 */
inline constexpr auto LATERAL_RULES =
    std::array<std::string_view, 3>{"linear", "cubic", "harmonic"};

/**
 * @brief The name a file states a radial rule by.
 * @param rule The rule.
 */
[[nodiscard]] constexpr std::string_view RuleName(RadialRule rule) noexcept {
  return RADIAL_RULES[static_cast<std::size_t>(rule)];
}

/**
 * @brief Whether a radial rule reads the logarithms of the values.
 * @param rule The rule.
 */
[[nodiscard]] constexpr bool IsLogRule(RadialRule rule) noexcept {
  return rule == RadialRule::LogPolynomial || rule == RadialRule::LogLinear;
}

/**
 * @brief The rule in the values a radial rule reads by: itself, or for a
 * rule in the logarithm the rule it applies to the logarithms.
 * @param rule The rule.
 */
[[nodiscard]] constexpr RadialRule BaseRule(RadialRule rule) noexcept {
  if (rule == RadialRule::LogPolynomial) return RadialRule::Polynomial;
  if (rule == RadialRule::LogLinear) return RadialRule::Linear;
  return rule;
}

/**
 * @brief The name a file states a lateral rule by.
 * @param rule The rule.
 */
[[nodiscard]] constexpr std::string_view RuleName(LateralRule rule) noexcept {
  return rule == LateralRule::Linear ? LATERAL_RULES[0] : LATERAL_RULES[1];
}

/**
 * @brief The radial rule a file names.
 * @param name One of `RADIAL_RULES`.
 * @param source The file, as an error names it.
 * @param where The variable, as an error names it.
 * @throws planetmodel::Error if the name is not a radial rule.
 */
[[nodiscard]] inline RadialRule RadialRuleNamed(std::string_view name,
                                                const std::string& source = "",
                                                std::string_view where = "") {
  for (auto k = std::size_t{0}; k < RADIAL_RULES.size(); ++k) {
    if (name == RADIAL_RULES[k]) return static_cast<RadialRule>(k);
  }
  throw Error(source, where,
              std::format("{} is not a radial rule; they are polynomial, linear, "
                          "log-polynomial and log-linear",
                          name));
}

/**
 * @brief The lateral rule a file names.
 * @param name `linear` or `cubic`; `harmonic` is refused.
 * @param source The file, as an error names it.
 * @param where The variable, as an error names it.
 * @throws planetmodel::Error if the name is `harmonic`, saying how the file
 *         may be written or read otherwise, or is not a lateral rule.
 */
[[nodiscard]] inline LateralRule LateralRuleNamed(std::string_view name,
                                                  const std::string& source = "",
                                                  std::string_view where = "") {
  if (name == "linear") return LateralRule::Linear;
  if (name == "cubic") return LateralRule::Cubic;
  if (name == "harmonic") {
    throw Error(source, where,
                "states the harmonic rule, which reads a spherical-harmonic expansion "
                "and is the Python's alone: write the file with interpolation={name: "
                "(\"polynomial\", \"cubic\")}, or read the stored values and analyse them");
  }
  throw Error(source, where,
              std::format("{} is not a lateral rule; they are linear and cubic", name));
}

namespace detail {

/// What keeps a grid from carrying the cubic rule, or nothing where it
/// carries it: fewer than four rows once the poles the grid lacks are
/// added, or fewer than three longitudes.
inline std::optional<std::string> CubicRefusal(const AngularGrid& grid) {
  const auto rows = grid.NTheta() + (grid.NorthPoleNode() ? 0 : 1) + (grid.SouthPoleNode() ? 0 : 1);
  if (rows >= 4 && grid.NPhi() >= 3) return std::nullopt;
  return std::format("the cubic rule needs at least 4 rows on the padded grid and 3 "
                     "longitudes; this grid gives {} rows and has {} longitudes",
                     rows, grid.NPhi());
}

}  // namespace detail

/**
 * @brief Refuse a lateral rule the grid cannot carry, as `check_rule`
 * does: the cubic rule on fewer than four rows once the poles the grid
 * lacks are added, or on fewer than three longitudes.
 * @param grid The grid.
 * @param rule The rule.
 * @param where What the values are of, as an error names it.
 * @throws planetmodel::Error if the grid cannot carry the rule.
 */
inline void CheckRule(const AngularGrid& grid, LateralRule rule, std::string_view where = "") {
  if (rule != LateralRule::Cubic) return;
  if (const auto why = detail::CubicRefusal(grid)) throw Error("LateralInterpolant", where, *why);
}

/**
 * @brief The values of a field at the two poles, by the rule of
 * `polar_values`.
 *
 * A pole that is a node gives its row, which must hold one value to `rtol`
 * of the values' largest; a pole the grid lacks gives its entry of
 * `poles`, which must be finite.  An entry of `poles` for a pole that is a
 * node must agree with its row or be NaN.
 *
 * @param grid The grid the values are on.
 * @param values The values, `NTheta()` rows of `NPhi()`, row-major.
 * @param poles The values at the north and south poles, or empty where both
 *        are nodes.
 * @param rtol The tolerance of a pole row, relative to the values' largest.
 * @param where What the values are of, as an error names it.
 * @return The values at the north and south poles.
 * @throws planetmodel::Error if a pole row varies, a needed pole value is
 *         missing or not finite, or a given one disagrees with its row.
 */
[[nodiscard]] inline std::array<double, 2> PolarValues(const AngularGrid& grid,
                                                       std::span<const double> values,
                                                       std::span<const double> poles,
                                                       double rtol = 1e-9,
                                                       std::string_view where = "") {
  if (values.size() != grid.NTheta() * grid.NPhi()) {
    throw Error("PolarValues", where,
                std::format("has {} values for a grid of {} by {}", values.size(),
                            grid.NTheta(), grid.NPhi()));
  }
  auto scale = 0.0;
  for (const auto v : values) scale = std::max(scale, std::abs(v));
  const auto tol = rtol * scale;
  const auto nodes = std::array<bool, 2>{grid.NorthPoleNode(), grid.SouthPoleNode()};
  const auto rows = std::array<std::size_t, 2>{0, grid.NTheta() - 1};
  constexpr auto names = std::array<std::string_view, 2>{"north", "south"};
  if (!poles.empty() && poles.size() != 2) {
    throw Error("PolarValues", where, "the poles are two values, north then south");
  }
  auto out = std::array<double, 2>{};
  for (auto k = 0; k < 2; ++k) {
    if (nodes[k]) {
      const auto row = values.subspan(rows[k] * grid.NPhi(), grid.NPhi());
      for (const auto v : row) {
        if (std::abs(v - row[0]) > tol) {
          throw Error("PolarValues", where,
                      std::format("the {} pole is a node of the grid, but its row varies "
                                  "along the longitudes by more than rtol = {} of the "
                                  "values' largest",
                                  names[k], rtol));
        }
      }
      out[k] = row[0];
      if (!poles.empty() && std::isfinite(poles[k]) && std::abs(poles[k] - out[k]) > tol) {
        throw Error("PolarValues", where,
                    std::format("the {} pole is a node of the grid, and the value given "
                                "for it disagrees with its row",
                                names[k]));
      }
    } else {
      if (poles.empty() || !std::isfinite(poles[k])) {
        throw Error("PolarValues", where,
                    std::format("the grid lacks the {} pole, so its value must be given "
                                "and finite",
                                names[k]));
      }
      out[k] = poles[k];
    }
  }
  return out;
}

/**
 * @brief The radial layout of a part of a model file: elements of `ngll`
 * nodes each, at the radii the file stores, and the weights by which a
 * rule reads values held at them.
 */
class RadialLayout {
 public:
  /**
   * @brief A layout of the given node radii.
   * @param node_radii The radii of every element's nodes, element by
   *        element, increasing within each element and each element
   *        starting where the one below ends.
   * @param ngll The nodes per element, at least two.
   * @param tolerance How far beyond its ends a radius is taken as on them.
   * @throws planetmodel::Error if the radii do not form such elements.
   */
  RadialLayout(std::vector<double> node_radii, std::size_t ngll, double tolerance)
      : radii_(std::move(node_radii)), ngll_(ngll), tolerance_(tolerance) {
    if (ngll_ < 2 || radii_.empty() || radii_.size() % ngll_ != 0) {
      throw Error("RadialLayout", "",
                  std::format("{} node radii are not elements of ngll = {}", radii_.size(),
                              ngll_));
    }
    for (auto e = std::size_t{0}; e < NElements(); ++e) {
      const auto nodes = Nodes(e);
      for (auto k = std::size_t{1}; k < ngll_; ++k) {
        if (!(nodes[k] > nodes[k - 1])) {
          throw Error("RadialLayout", std::format("element {}", e),
                      "its node radii must increase strictly");
        }
      }
      if (e > 0 && nodes.front() != Nodes(e - 1).back()) {
        throw Error("RadialLayout", std::format("element {}", e),
                    "does not start where the element below ends");
      }
    }
    auto bases = std::vector<Basis>{};
    bases.reserve(NElements());
    for (auto e = std::size_t{0}; e < NElements(); ++e) {
      const auto nodes = Nodes(e);
      bases.emplace_back(std::ranges::owning_view(std::vector<double>(nodes.begin(), nodes.end())));
    }
    bases_ = std::make_shared<const std::vector<Basis>>(std::move(bases));
  }

  /** @brief The number of elements. */
  [[nodiscard]] std::size_t NElements() const noexcept { return radii_.size() / ngll_; }
  /** @brief The nodes per element. */
  [[nodiscard]] std::size_t Ngll() const noexcept { return ngll_; }
  /** @brief Every node's radius, element by element. */
  [[nodiscard]] const std::vector<double>& NodeRadii() const noexcept { return radii_; }

  /**
   * @brief The node radii of one element.
   * @param e The element, from the centre.
   */
  [[nodiscard]] std::span<const double> Nodes(std::size_t e) const {
    return std::span<const double>(radii_).subspan(e * ngll_, ngll_);
  }

  /** @brief The radii of the layout's ends. */
  [[nodiscard]] std::pair<double, double> Interval() const noexcept {
    return {radii_.front(), radii_.back()};
  }

  /**
   * @brief The element a radius is read in: the one above at a shared
   * radius, the outermost at the outer end.
   * @param r The radius.
   * @throws planetmodel::Error if `r` lies beyond the layout's ends by more
   *         than its tolerance, or is not a number.
   */
  [[nodiscard]] std::size_t ElementOf(double r) const {
    const auto [lo, hi] = Interval();
    if (!(r >= lo - tolerance_ && r <= hi + tolerance_)) {
      throw Error("RadialLayout", "",
                  std::format("r = {} lies outside the layout's [{}, {}]", r, lo, hi));
    }
    auto below = std::size_t{0};
    auto above = NElements();
    while (above - below > 1) {
      const auto mid = (below + above) / 2;
      if (Nodes(mid).front() <= r) {
        below = mid;
      } else {
        above = mid;
      }
    }
    return below;
  }

  /**
   * @brief The weights of element `e`'s nodes at `r` under `rule`, and
   * their radial derivatives: a value held at the nodes reads as the sum of
   * the values times `w`, its radial derivative as the sum times `dw`.
   * Under a rule in the logarithm they are its `BaseRule`'s, which read the
   * logarithms.
   * @param e The element, as `ElementOf` gives it.
   * @param r The radius, taken as the nearer end of the element where
   *        outside it.
   * @param rule The radial rule.
   * @param w The weights, `Ngll()` of them, or none where they are not
   *        wanted.
   * @param dw Their derivatives in r, `Ngll()` of them, or none where they
   *        are not wanted.
   */
  void Weights(std::size_t e, double r, RadialRule rule, std::span<double> w,
               std::span<double> dw) const {
    const auto nodes = Nodes(e);
    r = std::clamp(r, nodes.front(), nodes.back());
    if (BaseRule(rule) == RadialRule::Polynomial) {
      // the element's cardinal polynomials and their slopes
      const auto& basis = (*bases_)[e];
      for (auto k = std::size_t{0}; k < w.size(); ++k) w[k] = basis.Evaluate<0>(k, r);
      for (auto k = std::size_t{0}; k < dw.size(); ++k) dw[k] = basis.Evaluate<1>(k, r);
      return;
    }
    std::ranges::fill(w, 0.0);
    std::ranges::fill(dw, 0.0);
    auto k = std::size_t{0};
    while (k + 2 < ngll_ && nodes[k + 1] <= r) ++k;
    const auto h = nodes[k + 1] - nodes[k];
    if (!w.empty()) {
      w[k] = (nodes[k + 1] - r) / h;
      w[k + 1] = (r - nodes[k]) / h;
    }
    if (!dw.empty()) {
      dw[k] = -1.0 / h;
      dw[k + 1] = 1.0 / h;
    }
  }

 private:
  /// The cardinal basis on one element's nodes, which it owns.
  using Basis = Interpolation::LagrangeBasis<std::ranges::owning_view<std::vector<double>>>;

  std::vector<double> radii_;
  std::size_t ngll_;
  double tolerance_;
  /// Each element's basis, shared by the copies of the layout.
  std::shared_ptr<const std::vector<Basis>> bases_;
};

/**
 * @brief A function of the radius held at the nodes of a radial layout,
 * read by a radial rule.
 */
class RadialInterpolant {
 public:
  /**
   * @brief The interpolant of nodal values.
   * @param layout The radial layout.
   * @param values One value per node, element by element; positive and
   *        finite under a rule in the logarithm.
   * @param rule The radial rule.
   * @throws planetmodel::Error if there is not one value per node, or a
   *         rule in the logarithm meets a value that is not positive and
   *         finite.
   */
  RadialInterpolant(RadialLayout layout, std::vector<double> values, RadialRule rule)
      : layout_(std::move(layout)), values_(std::move(values)), rule_(rule) {
    if (values_.size() != layout_.NodeRadii().size()) {
      throw Error("RadialInterpolant", "",
                  std::format("{} values for {} nodes", values_.size(),
                              layout_.NodeRadii().size()));
    }
    if (!IsLogRule(rule_)) return;
    // each element's first value, and the logarithms over it
    const auto n = layout_.Ngll();
    for (auto e = std::size_t{0}; e < layout_.NElements(); ++e) {
      const auto first = values_[e * n];
      for (auto k = std::size_t{0}; k < n; ++k) {
        const auto v = values_[e * n + k];
        if (!(std::isfinite(v) && v > 0.0)) {
          throw Error("RadialInterpolant", "",
                      std::format("is read by {}, its logarithm interpolated, and its "
                                  "value at node {} is {}, not positive and finite",
                                  RuleName(rule_), e * n + k, v));
        }
      }
      scale_.push_back(first);
      for (auto k = std::size_t{0}; k < n; ++k) {
        values_[e * n + k] = std::log(values_[e * n + k] / first);
      }
    }
  }

  /**
   * @brief The value at a radius.
   * @param r The radius.
   * @throws planetmodel::Error if `r` lies outside the layout.
   */
  [[nodiscard]] double Value(double r) const { return Read(r, 0); }

  /**
   * @brief The radial derivative at a radius.
   * @param r The radius.
   * @throws planetmodel::Error if `r` lies outside the layout.
   */
  [[nodiscard]] double Derivative(double r) const { return Read(r, 1); }

 private:
  /// The value (`derivative` 0) or the radial derivative (1) at `r`.
  double Read(double r, int derivative) const {
    const auto e = layout_.ElementOf(r);
    const auto n = layout_.Ngll();
    auto w = std::vector<double>(n);
    auto dw = std::vector<double>(derivative == 0 ? 0 : n);
    layout_.Weights(e, r, rule_, w, dw);
    auto out = 0.0;
    auto slope = 0.0;
    for (auto k = std::size_t{0}; k < n; ++k) {
      out += w[k] * values_[e * n + k];
      if (derivative != 0) slope += dw[k] * values_[e * n + k];
    }
    if (!IsLogRule(rule_)) return derivative == 0 ? out : slope;
    // the values held are the logarithms over the element's first value
    const auto value = scale_[e] * std::exp(out);
    return derivative == 0 ? value : value * slope;
  }

  RadialLayout layout_;
  /// The values, or under a rule in the logarithm log(y / y_e).
  std::vector<double> values_;
  RadialRule rule_;
  /// Under a rule in the logarithm, each element's first value y_e.
  std::vector<double> scale_{};
};

/**
 * @brief Values on an angular grid read anywhere by a lateral rule, with
 * the two angular derivatives.
 */
class LateralInterpolant {
 public:
  /**
   * @brief The interpolant of values on a grid.
   * @param grid The angular grid.
   * @param values The values, `NTheta()` rows of `NPhi()`, row-major.
   * @param poles The values at the north and south poles, used where the
   *        grid lacks them, checked where they are nodes (see
   *        `PolarValues`); empty where both are nodes.
   * @param rule The lateral rule.
   * @param where What the values are of, as an error names it.
   * @throws planetmodel::Error if the values do not fit the grid, the poles
   *         are not as `PolarValues` asks, or the cubic rule has fewer than
   *         four rows on the padded grid or three longitudes.
   */
  LateralInterpolant(const AngularGrid& grid, std::span<const double> values,
                     std::span<const double> poles, LateralRule rule,
                     std::string_view where = "")
      : phi0_(grid.Longitudes().front()) {
    const auto at_poles = PolarValues(grid, values, poles, 1e-9, where);
    const auto nphi = grid.NPhi();
    auto theta = std::vector<double>{};
    auto rows = std::vector<std::span<const double>>{};
    const auto north = std::vector<double>(nphi, at_poles[0]);
    const auto south = std::vector<double>(nphi, at_poles[1]);
    if (!grid.NorthPoleNode()) {
      theta.push_back(0.0);
      rows.emplace_back(north);
    }
    for (auto i = std::size_t{0}; i < grid.NTheta(); ++i) {
      theta.push_back(grid.Colatitudes()[i]);
      rows.push_back(values.subspan(i * nphi, nphi));
    }
    if (!grid.SouthPoleNode()) {
      theta.push_back(std::numbers::pi);
      rows.emplace_back(south);
    }
    auto phi = grid.Longitudes();
    phi.push_back(phi0_ + 2.0 * std::numbers::pi);
    const auto ncols = nphi + 1;
    auto padded = std::vector<double>{};
    padded.reserve(rows.size() * ncols);
    for (const auto& row : rows) {
      padded.insert(padded.end(), row.begin(), row.end());
      padded.push_back(row[0]);
    }
    if (rule == LateralRule::Linear) {
      spline_ = std::make_shared<const Spline>(
          std::in_place_type<Linear>, std::move(theta), std::move(phi), std::move(padded));
      return;
    }
    CheckRule(grid, rule, where);
    const auto [north_slopes, south_slopes] = PoleSlopes(theta, rows, grid.Longitudes());
    spline_ = std::make_shared<const Spline>(
        std::in_place_type<Cubic>, std::move(theta), std::move(phi), std::move(padded),
        Interpolation::AxisConditions{Interpolation::BoundaryCondition::Clamped,
                                      Interpolation::BoundaryCondition::Clamped},
        Interpolation::AxisConditions{Interpolation::BoundaryCondition::Periodic,
                                      Interpolation::BoundaryCondition::Periodic},
        Interpolation::EdgeSlopes<double>{.xLow = north_slopes, .xHigh = south_slopes});
  }

  /**
   * @brief The value at a point.
   * @param theta The colatitude, in [0, pi].
   * @param phi The longitude, of any range.
   * @throws planetmodel::Error if `theta` lies outside [0, pi] or a
   *         coordinate is not finite.
   */
  [[nodiscard]] double Value(double theta, double phi) const {
    return Read<0, 0>(theta, phi);
  }

  /**
   * @brief The derivatives in colatitude and in longitude at a point.
   * @param theta The colatitude, in [0, pi].
   * @param phi The longitude, of any range.
   * @return (d/dtheta, d/dphi).
   * @throws planetmodel::Error if `theta` lies outside [0, pi] or a
   *         coordinate is not finite.
   */
  [[nodiscard]] std::array<double, 2> Gradient(double theta, double phi) const {
    return {Read<1, 0>(theta, phi), Read<0, 1>(theta, phi)};
  }

 private:
  using Linear = decltype(Interpolation::Bilinear(std::vector<double>{}, std::vector<double>{},
                                                  std::vector<double>{}));
  using Cubic = decltype(Interpolation::BicubicSpline(
      std::vector<double>{}, std::vector<double>{}, std::vector<double>{},
      Interpolation::AxisConditions{}, Interpolation::AxisConditions{},
      Interpolation::EdgeSlopes<double>{}));

  /// The clamped slopes at the poles, one per padded column, the wrap
  /// column's a copy of the first's: a cos(phi) + b sin(phi) fitted by
  /// least squares to the slopes of the not-a-knot spline through each
  /// column.
  static std::pair<std::vector<double>, std::vector<double>> PoleSlopes(
      const std::vector<double>& theta, const std::vector<std::span<const double>>& rows,
      const std::vector<double>& phi) {
    const auto nphi = phi.size();
    auto at = std::array<std::vector<double>, 2>{std::vector<double>(nphi),
                                                 std::vector<double>(nphi)};
    for (auto j = std::size_t{0}; j < nphi; ++j) {
      auto column = std::vector<double>{};
      for (const auto& row : rows) column.push_back(row[j]);
      const auto spline = Interpolation::CubicSpline(
          theta, std::move(column), Interpolation::BoundaryCondition::NotAKnot, 0.0, 0.0);
      at[0][j] = spline.Evaluate<1>(0.0);
      at[1][j] = spline.Evaluate<1>(std::numbers::pi);
    }
    // the normal equations of the least-squares fit, 2 by 2
    auto cc = 0.0, cs = 0.0, ss = 0.0;
    for (const auto p : phi) {
      cc += std::cos(p) * std::cos(p);
      cs += std::cos(p) * std::sin(p);
      ss += std::sin(p) * std::sin(p);
    }
    const auto det = cc * ss - cs * cs;
    auto out = std::array<std::vector<double>, 2>{};
    for (auto k = 0; k < 2; ++k) {
      auto cy = 0.0, sy = 0.0;
      for (auto j = std::size_t{0}; j < nphi; ++j) {
        cy += std::cos(phi[j]) * at[k][j];
        sy += std::sin(phi[j]) * at[k][j];
      }
      const auto a = (ss * cy - cs * sy) / det;
      const auto b = (cc * sy - cs * cy) / det;
      for (const auto p : phi) out[k].push_back(a * std::cos(p) + b * std::sin(p));
      out[k].push_back(out[k].front());
    }
    return {std::move(out[0]), std::move(out[1])};
  }

  template <std::size_t Nx, std::size_t Ny>
  double Read(double theta, double phi) const {
    if (!(theta >= 0.0 && theta <= std::numbers::pi) || !std::isfinite(phi)) {
      throw Error("LateralInterpolant", "",
                  std::format("({}, {}) is not a point: the colatitude lies in [0, pi] "
                              "and both are finite",
                              theta, phi));
    }
    const auto p = Wrapped(phi, phi0_);
    return std::visit(
        [&](const auto& s) -> double { return s.template Evaluate<Nx, Ny>(theta, p); },
        *spline_);
  }

  using Spline = std::variant<Linear, Cubic>;

  double phi0_;
  std::shared_ptr<const Spline> spline_;
};

}  // namespace planetmodel

#endif  // PLANETMODEL_RULES_HPP
