#ifndef PLANETMODEL_RADIAL_HPP
#define PLANETMODEL_RADIAL_HPP

/**
 * @file
 * @brief Functions of the radius on one interval, with their first
 * derivative.
 *
 * A `RadialFunction` evaluates at any radius of its interval, and at one
 * within its tolerance outside, which is taken as the nearer end, as the
 * Python's fields take a radius within their rtol of an interval as on
 * it; beyond that it throws.  It is cheap to copy, the interpolant behind
 * it being shared, and its evaluation is `const` and thread-safe.  The
 * constructors here are the functions through knots that a deck's columns
 * are joined by: the piecewise polynomials, the not-a-knot and the natural
 * cubic splines, the parabola, and the polyline; and each of them through
 * the logarithms of positive values, the value its exponential, as the
 * Python's `ExponentialLayer` holds it: y_0 exp(L(r)), L through
 * log(y_i / y_0), its derivative y_0 exp(L) L'.
 */

#include <cmath>
#include <format>
#include <functional>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <Interpolation/CubicSpline.hpp>
#include <Interpolation/Lagrange.hpp>
#include <Interpolation/Linear.hpp>

#include "planetmodel/error.hpp"

/** @brief Defined where this header provides the kinds through the
 * logarithms of the values, `KnotKind::LogCubic`, `KnotKind::LogNatural`
 * and `KnotKind::LogLinear` (`log-cubic`, `log-natural`, `log-linear`). */
#define PLANETMODEL_HAS_LOG_KINDS 1

namespace planetmodel {

/** @brief A function of the radius on [lo, hi] and its first derivative. */
class RadialFunction {
 public:
  /**
   * @brief What a function is built from: its value (`derivative` 0) or
   * first derivative (1) at a radius already within the interval.
   */
  using Evaluator = std::function<double(double r, int derivative)>;

  /**
   * @brief A function on an interval.
   * @param lo The inner end.
   * @param hi The outer end.
   * @param tolerance How far beyond an end a radius is taken as on it.
   * @param evaluate The function and its derivative within [lo, hi].
   */
  RadialFunction(double lo, double hi, double tolerance, Evaluator evaluate)
      : lo_(lo), hi_(hi), tolerance_(tolerance),
        evaluate_(std::make_shared<const Evaluator>(std::move(evaluate))) {}

  /**
   * @brief The value at a radius.
   * @param r The radius.
   * @throws planetmodel::Error if `r` lies beyond the interval by more
   *         than the tolerance.
   */
  [[nodiscard]] double Value(double r) const { return (*evaluate_)(Within(r), 0); }

  /**
   * @brief The first derivative at a radius.
   * @param r The radius.
   * @throws planetmodel::Error if `r` lies beyond the interval by more
   *         than the tolerance.
   */
  [[nodiscard]] double Derivative(double r) const { return (*evaluate_)(Within(r), 1); }

  /** @brief The interval the function is defined on. */
  [[nodiscard]] std::pair<double, double> Interval() const noexcept { return {lo_, hi_}; }

 private:
  double Within(double r) const {
    if (!(r >= lo_ - tolerance_ && r <= hi_ + tolerance_)) {
      throw Error("RadialFunction", "",
                  std::format("r = {} lies outside [{}, {}]", r, lo_, hi_));
    }
    return r < lo_ ? lo_ : (r > hi_ ? hi_ : r);
  }

  double lo_;
  double hi_;
  double tolerance_;
  std::shared_ptr<const Evaluator> evaluate_;
};

/**
 * @brief How a deck's knots are joined within a layer, the kinds both the
 * Python and the C++ read; two knots are a line under every kind, and a
 * line through the logarithms under every logarithmic one.
 */
enum class KnotKind {
  Cubic,      /**< scipy's not-a-knot `CubicSpline`, the parabola through three knots. */
  Natural,    /**< The natural cubic spline: no curvature at the ends. */
  Linear,     /**< The polyline. */
  LogCubic,   /**< `Cubic` through the logarithms of the values, `log-cubic`. */
  LogNatural, /**< `Natural` through the logarithms of the values, `log-natural`. */
  LogLinear   /**< `Linear` through the logarithms, geometric between knots, `log-linear`. */
};

/**
 * @brief Whether a kind joins the logarithms of the values.
 * @param kind The kind.
 */
[[nodiscard]] constexpr bool IsLogarithmic(KnotKind kind) noexcept {
  return kind == KnotKind::LogCubic || kind == KnotKind::LogNatural ||
         kind == KnotKind::LogLinear;
}

/**
 * @brief The function through knots, on the interval they span: the
 * piecewise polynomial of `kind`, or for a logarithmic kind y_0 exp(L(r)),
 * L the piecewise polynomial of its kind through log(y_i / y_0), as the
 * Python's deck holds it, so that values constant through the knots are
 * that constant exactly.
 * @param r The knots' radii, strictly increasing, at least two.
 * @param y The values at the knots, positive and finite for a logarithmic
 *        kind, their ratios within the range of a double.
 * @param kind How the knots are joined.
 * @param tolerance How far beyond an end a radius is taken as on it.
 * @throws planetmodel::Error if the knots are fewer than two, the values
 *         do not match them, the radii do not increase strictly, or a
 *         logarithmic kind is given a value that is not positive and finite
 *         or two whose ratio is not finite.
 */
[[nodiscard]] inline RadialFunction ThroughKnots(std::vector<double> r, std::vector<double> y,
                                                 KnotKind kind, double tolerance) {
  if (r.size() != y.size() || r.size() < 2) {
    throw Error("ThroughKnots", "", "needs at least two knots and one value at each");
  }
  for (auto i = std::size_t{1}; i < r.size(); ++i) {
    if (!(r[i] > r[i - 1])) throw Error("ThroughKnots", "", "the knots must increase strictly");
  }
  if (IsLogarithmic(kind)) {
    for (const auto v : y) {
      if (!(std::isfinite(v) && v > 0.0)) {
        throw Error("ThroughKnots", "",
                    std::format("{} is not positive and finite; the logarithm of every value "
                                "is joined",
                                v));
      }
    }
    const auto first = y.front();
    for (auto& v : y) {
      v = std::log(v / first);
      if (!std::isfinite(v)) {
        throw Error("ThroughKnots", "",
                    "the values span more than the range of a double: the ratio of two of "
                    "them is not finite");
      }
    }
    const auto polynomial = kind == KnotKind::LogCubic     ? KnotKind::Cubic
                            : kind == KnotKind::LogNatural ? KnotKind::Natural
                                                           : KnotKind::Linear;
    const auto log = ThroughKnots(std::move(r), std::move(y), polynomial, tolerance);
    const auto [lo, hi] = log.Interval();
    return RadialFunction(lo, hi, tolerance, [first, log](double x, int derivative) {
      const auto value = first * std::exp(log.Value(x));
      return derivative == 0 ? value : value * log.Derivative(x);
    });
  }
  const auto lo = r.front();
  const auto hi = r.back();
  // Each interpolant owns the knots it was given, shared by the copies of
  // the function.
  auto share = [](auto&& interpolant) {
    auto f = std::make_shared<const std::decay_t<decltype(interpolant)>>(
        std::forward<decltype(interpolant)>(interpolant));
    return RadialFunction::Evaluator([f](double x, int derivative) {
      return derivative == 0 ? f->template Evaluate<0>(x) : f->template Evaluate<1>(x);
    });
  };
  auto evaluator = RadialFunction::Evaluator{};
  if (r.size() == 2 || kind == KnotKind::Linear) {
    evaluator = share(Interpolation::Linear(std::move(r), std::move(y)));
  } else if (r.size() == 3 && kind == KnotKind::Cubic) {
    evaluator = share(Interpolation::Lagrange(std::move(r), std::move(y)));
  } else {
    const auto condition = kind == KnotKind::Cubic ? Interpolation::BoundaryCondition::NotAKnot
                                                   : Interpolation::BoundaryCondition::Natural;
    evaluator = share(Interpolation::CubicSpline(std::move(r), std::move(y), condition, 0.0, 0.0));
  }
  return RadialFunction(lo, hi, tolerance, std::move(evaluator));
}

}  // namespace planetmodel

#endif  // PLANETMODEL_RADIAL_HPP
