#ifndef PLANETMODEL_CARTESIAN_GRID_HPP
#define PLANETMODEL_CARTESIAN_GRID_HPP

/**
 * @file
 * @brief The horizontal grid of a box or a rectangle, as
 * `planetmodel.cartesian.grid` defines it: the flat counterpart of the
 * sphere's angular grid.
 *
 * A grid is the product of equally spaced nodes over the period of a
 * lateral domain: over a box of extent [x0, x0 + Lx] by [y0, y0 + Ly] and
 * shape (nx, ny) the nodes x_i = x0 + i Lx / nx and y_j = y0 + j Ly / ny,
 * and over a rectangle of extent [x0, x0 + Lx] and shape (nx) the x_i
 * alone, the rectangle's one lateral coordinate.  The extent is the domain's
 * bounds widened by its padding; the last node of an axis is one step short
 * of the far end, the period closing it on the first.  A grid has no poles,
 * no kind and no weights: the mean over the period of values at the nodes
 * is their arithmetic mean.  Values on a grid are held x first, (nx, ny) in
 * C order.
 */

#include <array>
#include <cstddef>
#include <format>
#include <span>
#include <utility>
#include <vector>

#include "planetmodel/cartesian/domain.hpp"
#include "planetmodel/error.hpp"

namespace planetmodel::cartesian {

/** @brief The nodes of a lateral domain's period. */
class Grid {
 public:
  /**
   * @brief A grid over a box.
   * @param domain The box.
   * @param nx The number of nodes along x, at least one.
   * @param ny The number of nodes along y, at least one.
   * @throws planetmodel::Error if the domain is a rectangle, or a size is
   *         zero.
   */
  Grid(Domain domain, std::size_t nx, std::size_t ny)
      : Grid(std::move(domain), std::vector<std::size_t>{nx, ny}) {}

  /**
   * @brief A grid over a rectangle.
   * @param domain The rectangle.
   * @param nx The number of nodes along x, at least one.
   * @throws planetmodel::Error if the domain is a box, or the size is zero.
   */
  Grid(Domain domain, std::size_t nx) : Grid(std::move(domain), std::vector<std::size_t>{nx}) {}

  /**
   * @brief A grid of a shape over a domain.
   * @param domain The box or the rectangle.
   * @param shape The number of nodes along each of its axes: (nx, ny) over
   *        a box, (nx) over a rectangle.
   * @throws planetmodel::Error if the shape has not one size per axis, or a
   *         size is zero.
   */
  Grid(Domain domain, std::vector<std::size_t> shape)
      : domain_(std::move(domain)), shape_(std::move(shape)) {
    if (shape_.size() != domain_.Dimension()) {
      throw Error("Grid", "",
                  std::format("a grid over a {} has {} size{}, one per axis; got {}",
                              domain_.Kind(), domain_.Dimension(),
                              domain_.Dimension() > 1 ? "s" : "", shape_.size()));
    }
    for (auto axis = std::size_t{0}; axis < shape_.size(); ++axis) {
      const auto n = shape_[axis];
      if (n < 1) throw Error("Grid", "", "a grid has at least one node along each axis");
      const auto lo = domain_.Extent(axis)[0];
      const auto length = domain_.Length(axis);
      auto& nodes = axes_[axis];
      nodes.resize(n);
      for (auto i = std::size_t{0}; i < n; ++i) {
        nodes[i] = lo + static_cast<double>(i) * length / static_cast<double>(n);
      }
    }
  }

  /** @brief The lateral domain whose period the nodes cover. */
  [[nodiscard]] const Domain& GetDomain() const noexcept { return domain_; }
  /** @brief The nodes along each axis: (nx, ny) over a box, (nx) over a rectangle. */
  [[nodiscard]] const std::vector<std::size_t>& Shape() const noexcept { return shape_; }
  /** @brief The number of lateral axes: two over a box, one over a rectangle. */
  [[nodiscard]] std::size_t Dimension() const noexcept { return shape_.size(); }
  /** @brief The number of nodes along x. */
  [[nodiscard]] std::size_t NX() const noexcept { return shape_[0]; }
  /** @brief The number of nodes along y; one over a rectangle, whose body
   * is the same at every y. */
  [[nodiscard]] std::size_t NY() const noexcept { return shape_.size() > 1 ? shape_[1] : 1; }
  /** @brief The number of nodes: the size of values on the grid. */
  [[nodiscard]] std::size_t Size() const noexcept { return NX() * NY(); }
  /** @brief The nodes along x. */
  [[nodiscard]] const std::vector<double>& X() const noexcept { return axes_[0]; }
  /** @brief The nodes along y; empty over a rectangle. */
  [[nodiscard]] const std::vector<double>& Y() const noexcept { return axes_[1]; }

  /**
   * @brief The step between neighbouring nodes along an axis, its period
   * over its size.
   * @param axis 0 for x, 1 for y.
   * @throws planetmodel::Error if the grid has no such axis.
   */
  [[nodiscard]] double Spacing(std::size_t axis) const {
    return domain_.Length(axis) / static_cast<double>(shape_[axis]);
  }

  /** @brief The measure of one period: the area of a box's, the length of a rectangle's. */
  [[nodiscard]] double Area() const {
    return domain_.Length(0) * (Dimension() > 1 ? domain_.Length(1) : 1.0);
  }

  /**
   * @brief The lateral coordinates (x, y) of a node by its place in values
   * on the grid, x first; y is zero over a rectangle.
   * @param index The place, below `Size()`.
   */
  [[nodiscard]] std::array<double, 2> Node(std::size_t index) const {
    const auto ny = NY();
    return {axes_[0][index / ny], Dimension() > 1 ? axes_[1][index % ny] : 0.0};
  }

  /**
   * @brief The mean over the period of values at the nodes: their
   * arithmetic mean, the periodic trapezoid.
   * @param values One value per node.
   * @throws planetmodel::Error if there is not one value per node.
   */
  [[nodiscard]] double Mean(std::span<const double> values) const {
    if (values.size() != Size()) {
      throw Error("Grid", "",
                  std::format("{} values for a grid of {} nodes", values.size(), Size()));
    }
    auto sum = 0.0;
    for (const auto v : values) sum += v;
    return sum / static_cast<double>(values.size());
  }

  /** @brief Grids are equal when their domains and shapes are. */
  friend bool operator==(const Grid& a, const Grid& b) {
    return a.domain_ == b.domain_ && a.shape_ == b.shape_;
  }

 private:
  Domain domain_;
  std::vector<std::size_t> shape_;
  std::array<std::vector<double>, 2> axes_{};
};

}  // namespace planetmodel::cartesian

#endif  // PLANETMODEL_CARTESIAN_GRID_HPP
