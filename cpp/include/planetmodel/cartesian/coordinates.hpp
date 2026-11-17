#ifndef PLANETMODEL_CARTESIAN_COORDINATES_HPP
#define PLANETMODEL_CARTESIAN_COORDINATES_HPP

/**
 * @file
 * @brief The flat coordinate system, as `planetmodel.cartesian.coordinates`
 * defines it: (depth, x, y), with the local frame (e_up, e_x, e_y).
 *
 * Points are right-handed Cartesian (x, y, z), z up and z = 0 the free
 * surface of the reference body; the layered coordinate is the depth
 * d = -z, positive downward, and (x, y) are the lateral ones.  The local
 * frame is vertical first, (e_up, e_x, e_y), the same everywhere: a
 * vector's first local component is its upward one, and its Cartesian
 * components are (x, y, z) = (local 1, local 2, local 0).  A derivative
 * along the layered coordinate is d/d depth = -d/dz, downward.
 */

#include <cstddef>
#include <span>
#include <vector>

#include "planetmodel/frames.hpp"

namespace planetmodel::cartesian {

/** @brief The frame components are given in. */
enum class Frame {
  Local,    /**< The local frame (e_up, e_x, e_y), the same at every point. */
  Cartesian /**< The Cartesian frame (e_x, e_y, e_z) of the points, z up. */
};

/**
 * @brief The local frame (e_up, e_x, e_y) as the columns of a matrix in
 * Cartesian components: Cartesian = `LOCAL_FRAME` local.
 */
inline constexpr auto LOCAL_FRAME =
    Matrix3{{{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}}};

/** @brief The transpose of `LOCAL_FRAME`: local = `LOCAL_FRAME_TRANSPOSE` Cartesian. */
inline constexpr auto LOCAL_FRAME_TRANSPOSE =
    Matrix3{{{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}};

/** @brief A point by its flat coordinates: depth, x, y. */
struct Point {
  double depth = 0.0; /**< The depth below the reference free surface, positive down. */
  double x = 0.0;     /**< The first lateral coordinate. */
  double y = 0.0;     /**< The second lateral coordinate; not read over a rectangle. */
};

/**
 * @brief The Cartesian point (x, y, z), z up, of flat coordinates.
 * @param point The depth and the lateral coordinates.
 * @return (x, y, -depth).
 */
[[nodiscard]] constexpr Vector3 CartesianPoint(const Point& point) noexcept {
  return {point.x, point.y, -point.depth};
}

/**
 * @brief The flat coordinates of a Cartesian point.
 * @param X The point (x, y, z), z up.
 * @return The depth -z and the lateral coordinates.
 */
[[nodiscard]] constexpr Point FlatPoint(const Vector3& X) noexcept {
  return {.depth = -X[2], .x = X[0], .y = X[1]};
}

/**
 * @brief Stored components in the local frame as Cartesian ones.
 * @param stored The stored components, `rank` and `voigt` saying their
 *        layout (`RotateStored`).
 * @param rank The rank.
 * @param voigt Whether a rank-2 or rank-4 value is in Voigt form.
 * @return The Cartesian components, in the same layout.
 */
[[nodiscard]] inline std::vector<double> ToCartesian(std::span<const double> stored, int rank,
                                                     bool voigt) {
  if (rank == 0) return {stored.begin(), stored.end()};
  return RotateStored(stored, rank, voigt,
                      std::vector<Matrix3>(static_cast<std::size_t>(rank), LOCAL_FRAME));
}

/**
 * @brief Stored components in the Cartesian frame as local ones.
 * @param stored The stored components, `rank` and `voigt` saying their
 *        layout (`RotateStored`).
 * @param rank The rank.
 * @param voigt Whether a rank-2 or rank-4 value is in Voigt form.
 * @return The local components, in the same layout.
 */
[[nodiscard]] inline std::vector<double> ToLocal(std::span<const double> stored, int rank,
                                                 bool voigt) {
  if (rank == 0) return {stored.begin(), stored.end()};
  return RotateStored(
      stored, rank, voigt,
      std::vector<Matrix3>(static_cast<std::size_t>(rank), LOCAL_FRAME_TRANSPOSE));
}

}  // namespace planetmodel::cartesian

#endif  // PLANETMODEL_CARTESIAN_COORDINATES_HPP
