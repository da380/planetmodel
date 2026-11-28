#ifndef PLANETMODEL_PRONY_HPP
#define PLANETMODEL_PRONY_HPP

/**
 * @file
 * @brief A Burgers body and its two Prony branches, each from the other,
 * as `planetmodel.conversion.prony` forms them (`burgers_terms`,
 * `burgers_parameters`).
 *
 * A Burgers body in shear is a Maxwell element, of the shear modulus mu
 * and the viscosity eta, in series with a Kelvin element of the viscosity
 * eta_K and the shear modulus f mu, f the Kelvin modulus fraction: the
 * fields `viscosity`, `kelvin_viscosity` and `kelvin_modulus_fraction` of
 * the vocabulary.  It is exactly two Prony branches without a relaxed
 * part, the strengths w_k and times tau_k of a generalised Maxwell body
 * (`relaxation_strength_k`, `relaxation_time_k`), the faster branch
 * first.  With a = mu / eta,
 * b = mu / eta_K and c = f mu / eta_K, the rates 1 / tau_k are the roots
 * of x^2 - (a + b + c) x + a c, and w_1 = (1/tau_1 - c) / g,
 * w_2 = (c - 1/tau_2) / g, g the gap between the roots: both strengths
 * are positive and sum to one, and w_1 tau_1 + w_2 tau_2 = eta / mu.  The
 * terms are formed without cancellation.  The inverse takes any two
 * branches whose strengths sum to one and whose times differ: with
 * s = w_1 tau_1 + w_2 tau_2 and d = w_1 w_2 (tau_2 - tau_1)^2,
 * eta = mu s, eta_K = mu s tau_1 tau_2 / d and f = s^2 / d.  The
 * arithmetic is the Python's, operation for operation, so the two agree
 * to rounding; any consistent units.
 */

#include <array>
#include <cmath>
#include <format>
#include <limits>
#include <string_view>
#include <utility>

#include "planetmodel/error.hpp"

/** @brief Defined where this header provides `BurgersTerms` and
 * `BurgersParameters`. */
#define PLANETMODEL_HAS_BURGERS_TERMS 1

namespace planetmodel {

/**
 * @brief A Burgers body's Maxwell viscosity and its Kelvin element, the
 * shear modulus aside, each by the name the vocabulary gives its field.
 */
struct BurgersBody {
  double viscosity = std::numeric_limits<double>::quiet_NaN(); /**< The Maxwell element's, eta. */
  double kelvin_viscosity =
      std::numeric_limits<double>::quiet_NaN(); /**< The Kelvin element's, eta_K. */
  double kelvin_modulus_fraction =
      std::numeric_limits<double>::quiet_NaN(); /**< The Kelvin element's modulus over mu, f. */
};

/** @brief Two Prony branches, the faster first: their strengths w_k and
 * their times tau_k. */
using BurgersBranches = std::pair<std::array<double, 2>, std::array<double, 2>>;

/**
 * @brief A Burgers body's two Prony branches, (strengths, times), the
 * faster branch first, as the Python's `burgers_terms` forms them (file
 * comment): a Maxwell element of the shear modulus `modulus` and the
 * viscosity `body.viscosity` in series with a Kelvin element of the
 * viscosity `body.kelvin_viscosity` and the shear modulus
 * `body.kelvin_modulus_fraction` times `modulus`.
 * @param modulus The shear modulus mu.
 * @param body The viscosity and the Kelvin element, by name:
 *        `{.viscosity = ..., .kelvin_viscosity = ...,
 *        .kelvin_modulus_fraction = ...}`.
 * @throws planetmodel::Error where any of the four is not positive and
 *         finite.
 */
[[nodiscard]] inline BurgersBranches BurgersTerms(double modulus, const BurgersBody& body) {
  const auto given = std::array<std::pair<double, std::string_view>, 4>{
      {{modulus, "modulus"},
       {body.viscosity, "viscosity"},
       {body.kelvin_viscosity, "kelvin_viscosity"},
       {body.kelvin_modulus_fraction, "kelvin_modulus_fraction"}}};
  for (const auto& [x, what] : given) {
    if (!(std::isfinite(x) && x > 0.0)) {
      throw Error("BurgersTerms", "",
                  std::format("a Burgers body's {} is positive and finite", what));
    }
  }
  const auto mu = modulus;
  const auto a = mu / body.viscosity;
  const auto b = mu / body.kelvin_viscosity;
  const auto c = body.kelvin_modulus_fraction * b;
  const auto gap = std::sqrt((a - c) * (a - c) + b * b + 2.0 * b * (a + c));
  const auto fast = 0.5 * (a + b + c + gap);
  const auto slow = a * c / fast;  // the product of the roots is a c
  // gap + s, formed without cancellation where s < 0 (gap^2 - s^2 is 4 product)
  const auto plus = [gap](double s, double product) {
    return s >= 0.0 ? gap + s : 4.0 * product / (gap - s);
  };
  const auto above_slow = c * 0.5 * plus(b + c - a, a * b) / fast;  // c - slow
  const auto below_fast = 0.5 * plus(a + b - c, b * c);             // fast - c
  return {{below_fast / gap, above_slow / gap}, {1.0 / fast, 1.0 / slow}};
}

/**
 * @brief The Burgers body of two Prony branches, the inverse of
 * `BurgersTerms`, as the Python's `burgers_parameters` forms it (file
 * comment): the viscosity and the Kelvin element of the shear modulus
 * `modulus` and the branches' `strengths` and `times`, in either order.
 * The strengths are taken to sum to one, as a Burgers body's do.
 * @param modulus The shear modulus mu.
 * @param strengths The two strengths w_k.
 * @param times The two times tau_k.
 * @throws planetmodel::Error where any value is not positive and finite,
 *         or the two times are the same.
 */
[[nodiscard]] inline BurgersBody BurgersParameters(double modulus,
                                                   const std::array<double, 2>& strengths,
                                                   const std::array<double, 2>& times) {
  const auto positive = [](double x) { return std::isfinite(x) && x > 0.0; };
  const auto fail = [](std::string_view what) {
    throw Error("BurgersParameters", "",
                std::format("a Burgers body's {} are positive and finite", what));
  };
  if (!positive(modulus)) fail("modulus");
  if (!(positive(strengths[0]) && positive(strengths[1]))) fail("strengths");
  if (!(positive(times[0]) && positive(times[1]))) fail("times");
  if (times[0] == times[1]) {
    throw Error("BurgersParameters", "",
                "the two branches have the same time at a point; a Burgers body's two times "
                "differ");
  }
  const auto& w = strengths;
  const auto& tau = times;
  const auto steady = w[0] * tau[0] + w[1] * tau[1];
  const auto d = w[0] * w[1] * ((tau[1] - tau[0]) * (tau[1] - tau[0]));
  return {.viscosity = modulus * steady,
          .kelvin_viscosity = modulus * steady * tau[0] * tau[1] / d,
          .kelvin_modulus_fraction = steady * steady / d};
}

}  // namespace planetmodel

#endif  // PLANETMODEL_PRONY_HPP
