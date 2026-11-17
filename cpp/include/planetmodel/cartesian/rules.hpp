#ifndef PLANETMODEL_CARTESIAN_RULES_HPP
#define PLANETMODEL_CARTESIAN_RULES_HPP

/**
 * @file
 * @brief The rules by which values on a horizontal grid are read between
 * its nodes, as `planetmodel.cartesian.lateral` defines them: the doubly
 * periodic counterparts of the sphere's lateral rules.
 *
 * A grid covers one period of its domain along each axis, its last node
 * one step short of the far end, so each axis is closed by the first node
 * again one period on.  `linear` is the tensor product of the linear
 * splines through the nodes and that wrap node, bilinear over a box and
 * linear along the one axis of a section; `cubic` the tensor product of
 * the periodic cubic splines through the nodes, continuous with its first
 * two derivatives across the seam.  A point is read at its coordinates
 * reduced into the period from the grid's first node (`Wrapped`), so both
 * rules are periodic; there are no poles, and a section's y is never
 * read.  The components of a vector or a tensor are in the local frame
 * (e_up, e_x, e_y), the same everywhere, and are read as they stand.
 * `fourier`, which reads a Fourier series of the grid, is the Python's
 * alone and is refused by name, as `harmonic` is on the sphere.
 *
 * The radial rules, which read a variable between its nodes in depth, are
 * the sphere's (`RadialRule`, `RadialLayout`, rules.hpp).  The
 * interpolants are built once and evaluate with `const` methods that share
 * nothing mutable, so they are safe to use from many threads.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <Interpolation/BicubicSpline.hpp>
#include <Interpolation/Bilinear.hpp>
#include <Interpolation/CubicSpline.hpp>
#include <Interpolation/Linear.hpp>

#include "planetmodel/cartesian/grid.hpp"
#include "planetmodel/error.hpp"
#include "planetmodel/frames.hpp"
#include "planetmodel/rules.hpp"

namespace planetmodel::cartesian {

/**
 * @brief The lateral rules a flat file may state; `fourier` is the Python's
 * alone, and `LateralRuleNamed` refuses it.  The two this library reads
 * are `planetmodel::LateralRule::Linear` and `Cubic`.
 */
inline constexpr auto LATERAL_RULES =
    std::array<std::string_view, 3>{"linear", "cubic", "fourier"};

/**
 * @brief The lateral rule a flat file names.
 * @param name `linear` or `cubic`; `fourier` is refused.
 * @param source The file, as an error names it.
 * @param where The variable, as an error names it.
 * @throws planetmodel::Error if the name is `fourier`, saying how the file
 *         may be written or read otherwise, or is not a flat lateral rule.
 */
[[nodiscard]] inline LateralRule LateralRuleNamed(std::string_view name,
                                                  const std::string& source = "",
                                                  std::string_view where = "") {
  if (name == "linear") return LateralRule::Linear;
  if (name == "cubic") return LateralRule::Cubic;
  if (name == "fourier") {
    throw Error(source, where,
                "states the fourier rule, which reads a Fourier series of the grid and is "
                "the Python's alone: write the file with interpolation={name: "
                "(\"polynomial\", \"cubic\")}, or read the stored values and analyse them");
  }
  throw Error(source, where,
              std::format("{} is not a flat lateral rule; they are linear and cubic", name));
}

/**
 * @brief Refuse a lateral rule the grid cannot carry, as the Python's
 * `cartesian.lateral.check_rule` does: the cubic rule on fewer than three
 * nodes along an axis, the fewest a periodic cubic spline is determined
 * by.
 * @param grid The grid.
 * @param rule The rule.
 * @param where What the values are of, as an error names it.
 * @throws planetmodel::Error if the grid cannot carry the rule.
 */
inline void CheckRule(const Grid& grid, LateralRule rule, std::string_view where = "") {
  if (rule != LateralRule::Cubic) return;
  for (const auto n : grid.Shape()) {
    if (n < 3) {
      throw Error("cartesian::LateralInterpolant", where,
                  std::format("the cubic rule needs at least 3 nodes along every axis; this "
                              "grid has {} by {}",
                              grid.NX(), grid.NY()));
    }
  }
}

/**
 * @brief Values on a horizontal grid read anywhere by a flat lateral rule,
 * with the two lateral derivatives.
 */
class LateralInterpolant {
 public:
  /**
   * @brief The interpolant of values on a grid.
   * @param grid The horizontal grid.
   * @param values The values, x first: `NX()` rows of `NY()`, row-major;
   *        `NX()` values over a section.  Every one finite.
   * @param rule The lateral rule.
   * @param where What the values are of, as an error names it.
   * @throws planetmodel::Error if the values do not fit the grid or are not
   *         finite, or the cubic rule has fewer than three nodes along an
   *         axis.
   */
  LateralInterpolant(const Grid& grid, std::span<const double> values, LateralRule rule,
                     std::string_view where = "")
      : section_(grid.Dimension() == 1),
        x0_(grid.X().front()),
        y0_(section_ ? 0.0 : grid.Y().front()),
        lx_(grid.GetDomain().Length(0)),
        ly_(section_ ? 1.0 : grid.GetDomain().Length(1)) {
    if (values.size() != grid.Size()) {
      throw Error("cartesian::LateralInterpolant", where,
                  std::format("has {} values for a grid of {} by {}", values.size(), grid.NX(),
                              grid.NY()));
    }
    auto sum = 0.0;
    for (const auto v : values) {
      if (!std::isfinite(v)) {
        throw Error("cartesian::LateralInterpolant", where,
                    "the lateral rules take finite values");
      }
      sum += v;
    }
    mean_ = sum / static_cast<double>(values.size());
    CheckRule(grid, rule, where);
    // each axis closed by its first node, one period on
    const auto nx = grid.NX();
    const auto ny = grid.NY();
    auto x = grid.X();
    x.push_back(x0_ + lx_);
    if (section_) {
      auto closed = std::vector<double>(values.begin(), values.end());
      closed.push_back(values[0]);
      if (rule == LateralRule::Linear) {
        spline_ = std::make_shared<const Spline>(std::in_place_type<Linear1>, std::move(x),
                                                 std::move(closed));
      } else {
        spline_ = std::make_shared<const Spline>(
            std::in_place_type<Cubic1>, std::move(x), std::move(closed),
            Interpolation::BoundaryCondition::Periodic, 0.0,
            Interpolation::BoundaryCondition::Periodic, 0.0);
      }
      return;
    }
    auto y = grid.Y();
    y.push_back(y0_ + ly_);
    auto closed = std::vector<double>{};
    closed.reserve((nx + 1) * (ny + 1));
    for (auto i = std::size_t{0}; i <= nx; ++i) {
      const auto row = values.subspan((i % nx) * ny, ny);
      closed.insert(closed.end(), row.begin(), row.end());
      closed.push_back(row[0]);
    }
    if (rule == LateralRule::Linear) {
      spline_ = std::make_shared<const Spline>(std::in_place_type<Linear2>, std::move(x),
                                               std::move(y), std::move(closed));
    } else {
      spline_ = std::make_shared<const Spline>(std::in_place_type<Cubic2>, std::move(x),
                                               std::move(y), std::move(closed),
                                               Interpolation::BoundaryCondition::Periodic);
    }
  }

  /**
   * @brief The value at a point.
   * @param x The first lateral coordinate, of any range.
   * @param y The second lateral coordinate, of any range; not read over a
   *        section.
   * @throws planetmodel::Error if a coordinate that is read is not finite.
   */
  [[nodiscard]] double Value(double x, double y) const { return Read<0, 0>(x, y); }

  /**
   * @brief The derivatives in x and in y at a point; d/dy is zero over a
   * section.
   * @param x The first lateral coordinate, of any range.
   * @param y The second lateral coordinate, of any range; not read over a
   *        section.
   * @return (d/dx, d/dy).
   * @throws planetmodel::Error if a coordinate that is read is not finite.
   */
  [[nodiscard]] std::array<double, 2> Gradient(double x, double y) const {
    return {Read<1, 0>(x, y), section_ ? 0.0 : Read<0, 1>(x, y)};
  }

  /**
   * @brief The mean over the period of the interpolant itself: the mean of
   * the values, which on equally spaced periodic nodes either rule's
   * interpolant has.
   */
  [[nodiscard]] double Mean() const noexcept { return mean_; }

 private:
  using Axis = std::vector<double>;
  using Linear1 = decltype(Interpolation::Linear(Axis{}, Axis{}));
  using Cubic1 = decltype(Interpolation::CubicSpline(
      Axis{}, Axis{}, Interpolation::BoundaryCondition::Periodic, 0.0,
      Interpolation::BoundaryCondition::Periodic, 0.0));
  using Linear2 = decltype(Interpolation::Bilinear(Axis{}, Axis{}, Axis{}));
  using Cubic2 = decltype(Interpolation::BicubicSpline(
      Axis{}, Axis{}, Axis{}, Interpolation::BoundaryCondition::Periodic));
  using Spline = std::variant<Linear1, Cubic1, Linear2, Cubic2>;

  template <std::size_t Nx, std::size_t Ny>
  double Read(double x, double y) const {
    if (!std::isfinite(x) || (!section_ && !std::isfinite(y))) {
      throw Error("cartesian::LateralInterpolant", "",
                  std::format("({}, {}) is not a point: the coordinates are finite", x, y));
    }
    const auto a = Wrapped(x, x0_, lx_);
    if (section_) {
      return std::visit(
          [&](const auto& s) -> double {
            if constexpr (requires { s.template Evaluate<Nx>(a); }) {
              return s.template Evaluate<Nx>(a);
            } else {
              return 0.0;
            }
          },
          *spline_);
    }
    const auto b = Wrapped(y, y0_, ly_);
    return std::visit(
        [&](const auto& s) -> double {
          if constexpr (requires { s.template Evaluate<Nx, Ny>(a, b); }) {
            return s.template Evaluate<Nx, Ny>(a, b);
          } else {
            return 0.0;
          }
        },
        *spline_);
  }

  bool section_;
  double x0_;
  double y0_;
  double lx_;
  double ly_;
  double mean_ = 0.0;
  std::shared_ptr<const Spline> spline_{};
};

}  // namespace planetmodel::cartesian

#endif  // PLANETMODEL_CARTESIAN_RULES_HPP
