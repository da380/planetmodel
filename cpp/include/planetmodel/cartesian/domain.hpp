#ifndef PLANETMODEL_CARTESIAN_DOMAIN_HPP
#define PLANETMODEL_CARTESIAN_DOMAIN_HPP

/**
 * @file
 * @brief Lateral domains of a flat body, as `planetmodel.cartesian.domains`
 * defines them: the box's (x, y) and the rectangle's x.
 *
 * A finite lateral domain stands for an unbounded one.  Its `bounds` are
 * the domain proper and its `padding` a periodic margin added on each side
 * of each axis, zero by default; the domain with its padding, its extent,
 * is the period of the lateral directions.  A box bounds x and y, the
 * lateral domain of a three-dimensional body; a rectangle bounds x alone,
 * the lateral domain of the cross-section of a body invariant along y,
 * every y inside it.  The files state a box as `box` and a rectangle as
 * `rectangle`.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <string_view>
#include <vector>

#include "planetmodel/error.hpp"

namespace planetmodel::cartesian {

/**
 * @brief The kinds of lateral domain a flat file states by `domain`: a box,
 * and a rectangle, the cross-section of a body invariant along y.
 */
inline constexpr auto DOMAIN_KINDS = std::array<std::string_view, 2>{"box", "rectangle"};

/** @brief The lateral domain of a flat body: a box or a rectangle. */
class Domain {
 public:
  /**
   * @brief A box: x in `x` and y in `y`, with a padding on each side.
   * @param x The bounds of x, finite and increasing.
   * @param y The bounds of y, finite and increasing.
   * @param padding The periodic margin on each side of x and of y, finite
   *        and not negative.
   * @throws planetmodel::Error if the bounds or the padding are not so.
   */
  [[nodiscard]] static Domain Box(std::array<double, 2> x, std::array<double, 2> y,
                                  std::array<double, 2> padding = {0.0, 0.0}) {
    return Domain({x, y}, padding, 2);
  }

  /**
   * @brief A rectangle: x in `x`, with a padding on each side, every y
   * inside.
   * @param x The bounds of x, finite and increasing.
   * @param padding The periodic margin on each side of x, finite and not
   *        negative.
   * @throws planetmodel::Error if the bounds or the padding are not so.
   */
  [[nodiscard]] static Domain Rectangle(std::array<double, 2> x, double padding = 0.0) {
    return Domain({x, std::array<double, 2>{0.0, 0.0}}, {padding, 0.0}, 1);
  }

  /** @brief Whether the domain bounds x and y: a box, not a rectangle. */
  [[nodiscard]] bool IsBox() const noexcept { return dimension_ == 2; }
  /** @brief The number of lateral axes it bounds: two for a box, one for a rectangle. */
  [[nodiscard]] std::size_t Dimension() const noexcept { return dimension_; }
  /** @brief What a file states it as: `box` or `rectangle`. */
  [[nodiscard]] std::string_view Kind() const noexcept {
    return DOMAIN_KINDS[IsBox() ? 0 : 1];
  }

  /**
   * @brief The bounds of an axis: the domain proper.
   * @param axis 0 for x, 1 for y.
   * @throws planetmodel::Error if the domain does not bound the axis.
   */
  [[nodiscard]] std::array<double, 2> Bounds(std::size_t axis) const {
    return bounds_[Checked(axis)];
  }

  /**
   * @brief The periodic margin on each side of an axis.
   * @param axis 0 for x, 1 for y.
   * @throws planetmodel::Error if the domain does not bound the axis.
   */
  [[nodiscard]] double Padding(std::size_t axis) const { return padding_[Checked(axis)]; }

  /**
   * @brief The bounds of an axis widened by its padding on each side: one
   * period.
   * @param axis 0 for x, 1 for y.
   * @throws planetmodel::Error if the domain does not bound the axis.
   */
  [[nodiscard]] std::array<double, 2> Extent(std::size_t axis) const {
    const auto [lo, hi] = bounds_[Checked(axis)];
    return {lo - padding_[axis], hi + padding_[axis]};
  }

  /**
   * @brief The period of an axis: its extent's width.
   * @param axis 0 for x, 1 for y.
   * @throws planetmodel::Error if the domain does not bound the axis.
   */
  [[nodiscard]] double Length(std::size_t axis) const {
    const auto [lo, hi] = Extent(axis);
    return hi - lo;
  }

  /**
   * @brief The bounds as a file states them in `domain_bounds`: (x0, x1,
   * y0, y1) of a box and (x0, x1) of a rectangle.
   */
  [[nodiscard]] std::vector<double> BoundsList() const {
    auto out = std::vector<double>{};
    for (auto axis = std::size_t{0}; axis < dimension_; ++axis) {
      out.push_back(bounds_[axis][0]);
      out.push_back(bounds_[axis][1]);
    }
    return out;
  }

  /**
   * @brief The padding as a file states it in `domain_padding`: (px, py) of
   * a box and (px) of a rectangle.
   */
  [[nodiscard]] std::vector<double> PaddingList() const {
    return {padding_.begin(), padding_.begin() + static_cast<std::ptrdiff_t>(dimension_)};
  }

  /**
   * @brief Whether a point lies in the bounds, to `rtol` of their widths;
   * the padding is outside, and every y is inside a rectangle.
   * @param x The point's x.
   * @param y The point's y.
   * @param rtol The tolerance, relative to each axis's width.
   */
  [[nodiscard]] bool Contains(double x, double y, double rtol = 1e-9) const noexcept {
    const auto at = std::array<double, 2>{x, y};
    for (auto axis = std::size_t{0}; axis < dimension_; ++axis) {
      const auto [lo, hi] = bounds_[axis];
      const auto tolerance = rtol * (hi - lo);
      if (!(at[axis] >= lo - tolerance && at[axis] <= hi + tolerance)) return false;
    }
    return true;
  }

  /** @brief Domains are equal when their kind, bounds and padding are. */
  friend bool operator==(const Domain&, const Domain&) = default;

 private:
  Domain(std::array<std::array<double, 2>, 2> bounds, std::array<double, 2> padding,
         std::size_t dimension)
      : bounds_(bounds), padding_(padding), dimension_(dimension) {
    constexpr auto names = std::array<std::string_view, 2>{"x", "y"};
    for (auto axis = std::size_t{0}; axis < dimension_; ++axis) {
      const auto [lo, hi] = bounds_[axis];
      if (!(std::isfinite(lo) && std::isfinite(hi) && hi > lo)) {
        throw Error("Domain", "",
                    std::format("the bounds of {} must be finite and increase, got ({}, {})",
                                names[axis], lo, hi));
      }
      if (!(std::isfinite(padding_[axis]) && padding_[axis] >= 0.0)) {
        throw Error("Domain", "",
                    std::format("the padding of {} must be finite and non-negative, got {}",
                                names[axis], padding_[axis]));
      }
    }
  }

  std::size_t Checked(std::size_t axis) const {
    if (axis >= dimension_) {
      throw Error("Domain", "",
                  std::format("a {} bounds {} lateral ax{}, so not axis {}", Kind(), dimension_,
                              dimension_ == 1 ? "is" : "es", axis));
    }
    return axis;
  }

  std::array<std::array<double, 2>, 2> bounds_;
  std::array<double, 2> padding_;
  std::size_t dimension_;
};

}  // namespace planetmodel::cartesian

#endif  // PLANETMODEL_CARTESIAN_DOMAIN_HPP
