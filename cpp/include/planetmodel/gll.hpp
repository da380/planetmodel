#ifndef PLANETMODEL_GLL_HPP
#define PLANETMODEL_GLL_HPP

/**
 * @file
 * @brief Gauss-Lobatto-Legendre points, as `planetmodel.mesh1d.gll`
 * defines them, and the nodes of an element.
 *
 * The `ngll` points on [-1, 1] are the ends and the `ngll - 2` zeros of
 * P'_N, N = ngll - 1; an element [left, right] holds them at
 * left + (xi + 1) (right - left) / 2, the radii a model file stores.
 */

#include <cmath>
#include <cstddef>
#include <format>
#include <numbers>
#include <vector>

#include "planetmodel/error.hpp"

namespace planetmodel {

/**
 * @brief The Gauss-Lobatto-Legendre points on [-1, 1], increasing.
 *
 * The interior points are found by Newton's method on P'_N from the
 * Chebyshev-Gauss-Lobatto points, P''_N from Legendre's equation, and
 * made symmetric about zero.
 *
 * @param ngll The number of points, at least two.
 * @throws planetmodel::Error if `ngll` is less than two.
 */
[[nodiscard]] inline std::vector<double> GllPoints(std::size_t ngll) {
  if (ngll < 2) throw Error("GllPoints", "", "needs at least two points");
  const auto N = static_cast<int>(ngll) - 1;
  auto x = std::vector<double>(ngll);
  x.front() = -1.0;
  x.back() = 1.0;
  for (auto j = 1; j < N; ++j) {
    auto t = -std::cos(std::numbers::pi * j / N);
    for (auto iteration = 0; iteration < 100; ++iteration) {
      // P_N and P'_N by the three-term recurrence, P''_N from
      // (1 - t^2) P'' = 2 t P' - N (N + 1) P
      auto p0 = 1.0;
      auto p1 = t;
      for (auto k = 2; k <= N; ++k) {
        const auto p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k;
        p0 = p1;
        p1 = p2;
      }
      const auto dp = N * (t * p1 - p0) / (t * t - 1.0);
      const auto ddp = (2.0 * t * dp - N * (N + 1) * p1) / (1.0 - t * t);
      const auto step = dp / ddp;
      t -= step;
      if (std::abs(step) < 1e-16) break;
    }
    x[static_cast<std::size_t>(j)] = t;
  }
  for (auto j = std::size_t{0}; j < ngll / 2; ++j) {
    const auto mean = 0.5 * (x[ngll - 1 - j] - x[j]);
    x[j] = -mean;
    x[ngll - 1 - j] = mean;
  }
  if (ngll % 2 == 1) x[ngll / 2] = 0.0;
  return x;
}

/**
 * @brief The Gauss-Lobatto-Legendre nodes of an element, as a model file
 * holds them: the first and the last are the element's ends to the bit, so
 * that two elements sharing an end hold the one radius there.
 * @param left The element's inner radius.
 * @param right The element's outer radius.
 * @param ngll The number of nodes, at least two.
 * @throws planetmodel::Error if `ngll` is less than two.
 */
[[nodiscard]] inline std::vector<double> GllNodes(double left, double right, std::size_t ngll) {
  const auto jac = 0.5 * (right - left);
  auto out = GllPoints(ngll);
  for (auto& x : out) x = left + (x + 1.0) * jac;
  out.front() = left;
  out.back() = right;
  return out;
}

}  // namespace planetmodel

#endif  // PLANETMODEL_GLL_HPP
