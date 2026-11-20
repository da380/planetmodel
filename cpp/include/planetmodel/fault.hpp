#ifndef PLANETMODEL_FAULT_HPP
#define PLANETMODEL_FAULT_HPP

/**
 * @file
 * @brief Faults: patches of surface inside a body and the slip on them in
 * time, evaluated as `planetmodel.fault` evaluates them.
 *
 * A `Fault` is patches in one `FaultFrame`, their slip on one time axis.
 * Every patch maps the parameter square (u, v), each in [-1/2, 1/2], u
 * along strike and v down dip, into space: a plane by its centroid,
 * strike, dip, length and width (a flat rectangle in space), or a
 * structured patch through control positions on a uniform grid, read as
 * Cartesian points by the tensor-product B-spline of degree one or three
 * (not-a-knot), the same knots and coefficients as scipy's
 * `make_interp_spline`.  With x_u and x_v the tangents, the unit normal
 * is x_v x x_u over its length (from the footwall into the hanging
 * wall), the area element the length of x_u x x_v, and the fault basis
 * (s, a, n): s = x_u / |x_u|, a = n x s.  The slip is held in fault
 * components (strike-slip, dip-slip positive up dip, opening) on a grid
 * of its own, read by its rule (constant a cell, or the splines), and the
 * vector at a point is the components on the fault basis there.  Slices
 * in time are read by `TimeRowsAt`, a date as `DateOptions` say.
 *
 * What a consumer does with a fault is here: positions, normals, the
 * fault basis and the area element at any (u, v); the slip at any
 * (u, v) and date in fault, ambient or Cartesian components; a
 * quadrature of n_u by n_v cells with Gauss-Legendre points, with the
 * slip at a date (`Fault::Quadrature`) or the geometry once and the slip
 * of every slice (`Fault::SliceQuadrature`, whose slices a date's
 * `Fault::Rows` combine); and the location of physical points near the
 * fault (`Locate`, `SlipAt`), by the Python's search: closed form on a
 * plane, projected Newton steps with second derivatives elsewhere, every
 * cell of a bilinear patch near the point searched.  Lengths are in the
 * fault's length unit, angles in degrees.  A fault is read from a file
 * by `ReadFaultFile` and written to one by `WriteFaultFile`
 * (`faultfile.hpp`).
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "planetmodel/error.hpp"
#include "planetmodel/frames.hpp"
#include "planetmodel/time.hpp"
#include "planetmodel/units.hpp"

/** @brief Defined where this header provides `Fault::SliceQuadrature` and
 * `FaultSliceQuadrature::SlipOf`, a fault's date read by `DateOptions`
 * (`Fault::Rows`, `Fault::SlipVector`, `Fault::Quadrature`), and
 * `FaultSlip::FromRake`. */
#define PLANETMODEL_HAS_FAULT_SLICE_QUADRATURE 1

namespace planetmodel {

/** @brief The interval u and v each run over: the parameter square is
 * FAULT_SQUARE x FAULT_SQUARE, (0, 0) at its centre. */
inline constexpr auto FAULT_SQUARE = std::array<double, 2>{-0.5, 0.5};

/** @brief The rules between a structured patch's control positions. */
inline constexpr auto FAULT_INTERPOLATIONS = std::array<std::string_view, 2>{"linear", "cubic"};

/** @brief The rules of a slip grid: constant a cell, or a spline between
 * nodes. */
inline constexpr auto SLIP_RULES = std::array<std::string_view, 3>{"cell", "linear", "cubic"};

/** @brief The time supports a fault holds, of `TIME_SUPPORTS`: one slice
 * at any date, or a slice a time node, the slip at a node cumulative. */
inline constexpr auto FAULT_SUPPORTS = std::array<std::string_view, 2>{"static", "node"};

/** @brief The names of the slip's three components, in order. */
inline constexpr auto SLIP_COMPONENTS =
    std::array<std::string_view, 3>{"strike_slip", "dip_slip", "opening"};

/**
 * @brief The fewest nodes (cells, for `cell`) each way a rule needs.
 * @param rule One of `SLIP_RULES`.
 * @throws planetmodel::Error for a rule that is not one.
 */
[[nodiscard]] inline std::size_t MinNodes(std::string_view rule) {
  if (rule == "cell") return 1;
  if (rule == "linear") return 2;
  if (rule == "cubic") return 4;
  throw Error("fault", "rule", std::format("{} is not one of cell, linear, cubic", rule));
}

/** @brief The basis a slip vector is given in. */
enum class SlipBasis {
  Fault,    /**< (strike-slip, dip-slip, opening) in the fault basis at the point. */
  Ambient,  /**< The frame's basis at the point: (east, north, up), or (x, y, z). */
  Cartesian /**< Cartesian components. */
};

namespace detail {

inline double Dot(const Vector3& a, const Vector3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vector3 Cross(const Vector3& a, const Vector3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

inline double Length(const Vector3& a) { return std::sqrt(Dot(a, a)); }

inline Vector3 Unit(const Vector3& a) {
  const auto n = Length(a);
  return {a[0] / n, a[1] / n, a[2] / n};
}

inline Vector3 Minus(const Vector3& a, const Vector3& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

/// The fault basis as columns (s, a, n) from the tangents: s = x_u / |x_u|,
/// n = x_v x x_u over its length, a = n x s.
inline Matrix3 FaultBasisOf(const Vector3& xu, const Vector3& xv) {
  const auto s = Unit(xu);
  const auto n = Unit(Cross(xv, xu));
  const auto a = Cross(n, s);
  auto out = Matrix3{};
  for (auto i = 0; i < 3; ++i) out[i] = {s[i], a[i], n[i]};
  return out;
}

/// Fault components on a fault basis F, in Cartesian components: F c.
inline Vector3 OnBasis(const Matrix3& F, const Vector3& c) {
  auto out = Vector3{};
  for (auto i = 0; i < 3; ++i) out[i] = F[i][0] * c[0] + F[i][1] * c[1] + F[i][2] * c[2];
  return out;
}

/// numpy's linspace(start, stop, num): start + i step, the last exactly stop.
inline std::vector<double> Linspace(double start, double stop, std::size_t num) {
  auto out = std::vector<double>(num);
  if (num == 0) return out;
  if (num == 1) {
    out[0] = start;
    return out;
  }
  const auto step = (stop - start) / static_cast<double>(num - 1);
  for (auto i = std::size_t{0}; i < num; ++i) out[i] = static_cast<double>(i) * step + start;
  out.back() = stop;
  return out;
}

/// n equally spaced nodes over the square, both ends included.
inline std::vector<double> SquareNodes(std::size_t n) {
  return Linspace(FAULT_SQUARE[0], FAULT_SQUARE[1], n);
}

/// scipy's interval search: l with t[l] <= x < t[l + 1], k <= l < n,
/// the last interval closed at its end and continued beyond.
inline std::size_t FindInterval(const std::vector<double>& t, std::size_t k, double x) {
  const auto n = t.size() - k - 1;
  auto l = k;
  while (x < t[l] && l != k) --l;
  ++l;
  while (x >= t[l] && l != n) ++l;
  return l - 1;
}

/// scipy's _deBoor_D: the k + 1 B-splines non-zero on [t[ell], t[ell+1])
/// at x, or their m-th derivatives, B_{ell-k} first.
inline std::array<double, 8> DeBoorD(const std::vector<double>& t, double x, std::size_t k,
                                     std::size_t ell, std::size_t m) {
  auto result = std::array<double, 8>{};
  if (m > k) return result;
  auto hh = std::array<double, 4>{};
  auto& h = result;
  h[0] = 1.0;
  for (auto j = std::size_t{1}; j <= k - m; ++j) {
    std::copy_n(h.begin(), j, hh.begin());
    h[0] = 0.0;
    for (auto n = std::size_t{1}; n <= j; ++n) {
      const auto xb = t[ell + n];
      const auto xa = t[ell + n - j];
      if (xb == xa) {
        h[n] = 0.0;
        continue;
      }
      const auto w = hh[n - 1] / (xb - xa);
      h[n - 1] += w * (xb - x);
      h[n] = w * (x - xa);
    }
  }
  for (auto j = k - m + 1; j <= k; ++j) {
    std::copy_n(h.begin(), j, hh.begin());
    h[0] = 0.0;
    for (auto n = std::size_t{1}; n <= j; ++n) {
      const auto xb = t[ell + n];
      const auto xa = t[ell + n - j];
      if (xb == xa) {
        h[m] = 0.0;
        continue;
      }
      const auto w = static_cast<double>(j) * hh[n - 1] / (xb - xa);
      h[n - 1] -= w;
      h[n] = w;
    }
  }
  return result;
}

/// The knots scipy's make_interp_spline gives uniform nodes over the
/// square: degree one, each end once more; degree three, not-a-knot.
inline std::vector<double> InterpolationKnots(std::size_t n, std::size_t k) {
  const auto x = SquareNodes(n);
  auto t = std::vector<double>{};
  if (k == 1) {
    t.push_back(x.front());
    t.insert(t.end(), x.begin(), x.end());
    t.push_back(x.back());
    return t;
  }
  t.assign(4, x.front());
  for (auto i = std::size_t{2}; i + 2 < n; ++i) t.push_back(x[i]);
  t.insert(t.end(), 4, x.back());
  return t;
}

/// The coefficients of the cubic spline interpolating at the nodes of
/// the square, from values `columns` at a stride: the collocation
/// matrix B_j(x_i) factored by LU with partial pivoting over its band
/// (kl = ku = 3, as LAPACK's gbsv), each column solved in place.
inline void InterpolateCubic(std::size_t n, const std::vector<double>& t, std::vector<double>& data,
                             std::size_t count, std::size_t stride) {
  // data holds n rows of `stride` numbers; columns 0 .. count - 1 are solved
  constexpr auto kl = std::size_t{3};
  constexpr auto ku = std::size_t{3};
  constexpr auto kv = kl + ku;
  constexpr auto ldab = 2 * kl + ku + 1;
  auto ab = std::vector<double>(ldab * n, 0.0);
  // A(r, c) at ab[(kv + r - c) + c ldab], kept within the band
  auto at = [&ab](std::size_t r, std::size_t c) -> double& { return ab[(kv + r - c) + c * ldab]; };
  const auto x = SquareNodes(n);
  auto left = std::size_t{3};
  for (auto i = std::size_t{0}; i < n; ++i) {
    left = FindInterval(t, 3, x[i]);
    const auto b = DeBoorD(t, x[i], 3, left, 0);
    for (auto a = std::size_t{0}; a <= 3; ++a) at(i, left - 3 + a) = b[a];
  }
  auto pivot = std::vector<std::size_t>(n);
  auto ju = std::size_t{0};
  for (auto j = std::size_t{0}; j < n; ++j) {
    const auto km = std::min(kl, n - 1 - j);
    auto jp = std::size_t{0};
    for (auto i = std::size_t{1}; i <= km; ++i) {
      if (std::abs(at(j + i, j)) > std::abs(at(j + jp, j))) jp = i;
    }
    pivot[j] = j + jp;
    if (at(j + jp, j) == 0.0) throw Error("fault", "spline", "the collocation matrix is singular");
    ju = std::max(ju, std::min(j + ku + jp, n - 1));
    if (jp != 0) {
      for (auto c = j; c <= ju; ++c) std::swap(at(j + jp, c), at(j, c));
    }
    const auto inverse = 1.0 / at(j, j);
    for (auto i = std::size_t{1}; i <= km; ++i) at(j + i, j) *= inverse;
    for (auto c = j + 1; c <= ju; ++c) {
      const auto u = at(j, c);
      if (u == 0.0) continue;
      for (auto i = std::size_t{1}; i <= km; ++i) at(j + i, c) -= at(j + i, j) * u;
    }
  }
  for (auto col = std::size_t{0}; col < count; ++col) {
    auto value = [&data, col, stride](std::size_t i) -> double& { return data[i * stride + col]; };
    for (auto j = std::size_t{0}; j < n; ++j) {
      if (pivot[j] != j) std::swap(value(j), value(pivot[j]));
      const auto km = std::min(kl, n - 1 - j);
      for (auto i = std::size_t{1}; i <= km; ++i) value(j + i) -= at(j + i, j) * value(j);
    }
    for (auto j = n; j-- > 0;) {
      value(j) /= at(j, j);
      const auto top = j >= kv ? j - kv : std::size_t{0};
      for (auto r = top; r < j; ++r) value(r) -= at(r, j) * value(j);
    }
  }
}

/// Values on a uniform grid over the square read by one rule: constant a
/// cell, or the tensor-product spline (degree one, or three not-a-knot)
/// through the nodes, as `planetmodel.fault`'s rule reads them; values
/// (n_u, n_v, m) in C order, m numbers a node.
class GridRule {
 public:
  GridRule() = default;

  GridRule(std::vector<double> values, std::size_t n_u, std::size_t n_v, std::size_t m,
           std::string_view rule)
      : n_u_(n_u), n_v_(n_v), m_(m), cell_(rule == "cell") {
    if (cell_) {
      edges_u_ = Linspace(FAULT_SQUARE[0], FAULT_SQUARE[1], n_u + 1);
      edges_v_ = Linspace(FAULT_SQUARE[0], FAULT_SQUARE[1], n_v + 1);
      c_ = std::move(values);
      return;
    }
    k_ = rule == "linear" ? 1 : 3;
    tu_ = InterpolationKnots(n_u, k_);
    tv_ = InterpolationKnots(n_v, k_);
    c_ = std::move(values);
    if (k_ == 3) {
      // along u for every (j, component), then along v for every (i, component)
      InterpolateCubic(n_u, tu_, c_, n_v * m, n_v * m);
      for (auto i = std::size_t{0}; i < n_u; ++i) {
        auto row = std::vector<double>(c_.begin() + static_cast<std::ptrdiff_t>(i * n_v * m),
                                       c_.begin() + static_cast<std::ptrdiff_t>((i + 1) * n_v * m));
        InterpolateCubic(n_v, tv_, row, m, m);
        std::ranges::copy(row, c_.begin() + static_cast<std::ptrdiff_t>(i * n_v * m));
      }
    }
  }

  /// Components [first, first + count) at (u, v), or their derivative of
  /// orders (du, dv), into out.
  void Evaluate(double u, double v, std::size_t du, std::size_t dv, std::size_t first,
                std::size_t count, std::span<double> out) const {
    if (cell_) {
      auto index = [](const std::vector<double>& edges, double x) {
        const auto above =
            static_cast<std::size_t>(std::ranges::upper_bound(edges, x) - edges.begin());
        return std::clamp(above, std::size_t{1}, edges.size() - 1) - 1;
      };
      const auto i = index(edges_u_, u);
      const auto j = index(edges_v_, v);
      for (auto c = std::size_t{0}; c < count; ++c) {
        out[c] = du + dv == 0 ? c_[(i * n_v_ + j) * m_ + first + c] : 0.0;
      }
      return;
    }
    const auto iu = FindInterval(tu_, k_, u);
    const auto iv = FindInterval(tv_, k_, v);
    const auto bu = DeBoorD(tu_, u, k_, iu, du);
    const auto bv = DeBoorD(tv_, v, k_, iv, dv);
    for (auto c = std::size_t{0}; c < count; ++c) out[c] = 0.0;
    for (auto a = std::size_t{0}; a <= k_; ++a) {
      for (auto b = std::size_t{0}; b <= k_; ++b) {
        const auto factor = 1.0 * bu[a] * bv[b];
        const auto base = ((iu - k_ + a) * n_v_ + (iv - k_ + b)) * m_ + first;
        for (auto c = std::size_t{0}; c < count; ++c) out[c] += c_[base + c] * factor;
      }
    }
  }

 private:
  std::size_t n_u_ = 0, n_v_ = 0, m_ = 0, k_ = 0;
  bool cell_ = true;
  std::vector<double> edges_u_{}, edges_v_{}, tu_{}, tv_{}, c_{};
};

/// The refusal of a parameter outside the square, as the Python's.
inline void CheckParameters(double u, double v) {
  if (!(std::isfinite(u) && std::isfinite(v))) {
    throw Error("fault patch", "", "(u, v) must be finite");
  }
  if (u < FAULT_SQUARE[0] || u > FAULT_SQUARE[1] || v < FAULT_SQUARE[0] || v > FAULT_SQUARE[1]) {
    throw Error("fault patch", "",
                "(u, v) must lie in the parameter square [-0.5, 0.5] x [-0.5, 0.5]");
  }
}

/// numpy's deg2rad and rad2deg.
inline double Radians(double degrees) { return degrees * (std::numbers::pi / 180.0); }
inline double Degrees(double radians) { return radians * (180.0 / std::numbers::pi); }

}  // namespace detail

/**
 * @brief Where a fault's positions are and what basis its vectors use:
 * geographic, about a reference sphere, or local Cartesian.
 *
 * A geographic position is (latitude, longitude, radius), the angles in
 * degrees, the Cartesian point r (cos lat cos lon, cos lat sin lon,
 * sin lat), the ambient basis (east, north, up); a Cartesian position is
 * (x, y, z), x east, y north and z up, the ambient basis (x, y, z).  The
 * reference surface (the sphere, or the plane z = surface) is what a
 * depth is measured from.
 */
class FaultFrame {
 public:
  /** @brief The two frames. */
  enum class Kind {
    Geographic, /**< On a reference sphere. */
    Cartesian   /**< Local, flat. */
  };

  /**
   * @brief A geographic frame.
   * @param radius The radius of the reference sphere, positive.
   * @throws planetmodel::Error if it is not.
   */
  [[nodiscard]] static FaultFrame Geographic(double radius) {
    if (!(std::isfinite(radius) && radius > 0.0)) {
      throw Error("fault frame", "radius", "the reference radius must be positive");
    }
    return FaultFrame(Kind::Geographic, radius);
  }

  /**
   * @brief A local Cartesian frame.
   * @param surface The height z of the reference plane.
   * @throws planetmodel::Error if it is not finite.
   */
  [[nodiscard]] static FaultFrame Cartesian(double surface = 0.0) {
    if (!std::isfinite(surface)) {
      throw Error("fault frame", "surface", "the reference surface must be finite");
    }
    return FaultFrame(Kind::Cartesian, surface);
  }

  /** @brief Which frame it is. */
  [[nodiscard]] Kind GetKind() const noexcept { return kind_; }

  /** @brief `geographic` or `cartesian`, as a fault file states it. */
  [[nodiscard]] std::string_view Name() const noexcept {
    return kind_ == Kind::Geographic ? "geographic" : "cartesian";
  }

  /** @brief The reference sphere's radius, or the reference plane's z. */
  [[nodiscard]] double Reference() const noexcept { return reference_; }

  /**
   * @brief The Cartesian point of a position in the frame's coordinates.
   * @param c The position: (latitude, longitude, radius), or (x, y, z).
   */
  [[nodiscard]] Vector3 CartesianPoint(const Vector3& c) const {
    if (kind_ == Kind::Cartesian) return c;
    const auto [theta, phi] = Angles(c);
    return planetmodel::CartesianPoint(c[2], theta, phi);
  }

  /**
   * @brief The frame's coordinates of a Cartesian point: the longitude in
   * [0, 360).
   * @param x The point.
   */
  [[nodiscard]] Vector3 Coordinates(const Vector3& x) const {
    if (kind_ == Kind::Cartesian) return x;
    const auto r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    const auto theta = r > 0.0 ? std::atan2(std::hypot(x[0], x[1]), x[2]) : 0.5 * std::numbers::pi;
    const auto phi = std::atan2(x[1], x[0]);
    return {90.0 - detail::Degrees(theta), Wrapped(detail::Degrees(phi), 0.0, 360.0), r};
  }

  /**
   * @brief The ambient basis at a position: the columns (east, north, up),
   * or the identity, in Cartesian components.
   * @param c The position in the frame's coordinates.
   */
  [[nodiscard]] Matrix3 Basis(const Vector3& c) const {
    if (kind_ == Kind::Cartesian) return {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    const auto [theta, phi] = Angles(c);
    const auto R = SphericalFrame(theta, phi);
    auto out = Matrix3{};
    for (auto i = 0; i < 3; ++i) out[i] = {R[i][2], -R[i][1], R[i][0]};
    return out;
  }

  /** @brief Whether two frames are the same. */
  [[nodiscard]] bool operator==(const FaultFrame&) const = default;

 private:
  FaultFrame(Kind kind, double reference) : kind_(kind), reference_(reference) {}

  static std::pair<double, double> Angles(const Vector3& c) {
    if (!(std::abs(c[0]) <= 90.0)) {
      throw Error("fault frame", "latitude", "latitudes must lie in [-90, 90]");
    }
    const auto theta = std::clamp(detail::Radians(90.0 - c[0]), 0.0, std::numbers::pi);
    return {theta, Wrapped(detail::Radians(c[1]))};
  }

  Kind kind_ = Kind::Geographic;
  double reference_ = 1.0;
};

/**
 * @brief What `FaultSlip::FromRake` takes besides the rake and the
 * magnitude: the Python's keywords.
 */
struct RakeOptions {
  std::vector<double> opening{0.0}; /**< Along the normal: one value, one a point, or one a slice
                                         and point. */
  std::string rule = "cell";        /**< One of `SLIP_RULES`. */
};

/**
 * @brief The slip on one patch: its fault components (strike-slip,
 * dip-slip, opening) on a grid over the square, and the rule between
 * them.
 *
 * The values are (nslices, n_u, n_v, 3) in C order: at the centres of
 * n_u by n_v equal cells for the `cell` rule, otherwise at the nodes of a
 * uniform grid, corners included.
 */
class FaultSlip {
 public:
  /**
   * @brief Slip from its values.
   * @param values The components, (nslices, n_u, n_v, 3) in C order.
   * @param nslices The slices in time.
   * @param n_u The cells or nodes along u.
   * @param n_v The cells or nodes along v.
   * @param rule One of `SLIP_RULES`.
   * @throws planetmodel::Error for a rule unknown, a grid too small for
   *         it, values of the wrong number or not finite.
   */
  FaultSlip(std::vector<double> values, std::size_t nslices, std::size_t n_u, std::size_t n_v,
            std::string rule = "cell")
      : values_(std::move(values)),
        nslices_(nslices),
        n_u_(n_u),
        n_v_(n_v),
        rule_(std::move(rule)) {
    if (!std::ranges::contains(SLIP_RULES, std::string_view{rule_})) {
      throw Error("fault slip", "rule", std::format("{} is not one of cell, linear, cubic", rule_));
    }
    const auto least = MinNodes(rule_);
    if (nslices_ < 1 || std::min(n_u_, n_v_) < least) {
      throw Error("fault slip", "",
                  std::format("the {} rule needs at least {} {} each way, got {} by {}", rule_,
                              least, rule_ == "cell" ? "cells" : "nodes", n_u_, n_v_));
    }
    if (values_.size() != nslices_ * n_u_ * n_v_ * 3) {
      throw Error(
          "fault slip", "",
          std::format("{} values, not {} x {} x {} x 3", values_.size(), nslices_, n_u_, n_v_));
    }
    if (!std::ranges::all_of(values_, [](double x) { return std::isfinite(x); })) {
      throw Error("fault slip", "", "slip values must be finite");
    }
    // the rule reads (n_u, n_v, nslices, 3): every slice at once
    auto moved = std::vector<double>(values_.size());
    for (auto s = std::size_t{0}; s < nslices_; ++s) {
      for (auto ij = std::size_t{0}; ij < n_u_ * n_v_; ++ij) {
        for (auto c = 0; c < 3; ++c) {
          moved[(ij * nslices_ + s) * 3 + c] = values_[(s * n_u_ * n_v_ + ij) * 3 + c];
        }
      }
    }
    rule_reader_ = detail::GridRule(std::move(moved), n_u_, n_v_, nslices_ * 3, rule_);
  }

  /**
   * @brief Slip of a magnitude m in the fault plane at a rake, with an
   * opening along the normal: the components (m cos rake, m sin rake,
   * opening), as the Python's `Slip.from_rake` forms them.  Each of the
   * rake, the magnitude and the opening is one value, for every slice and
   * point of the grid; one a point, (n_u, n_v) in C order, the same in
   * every slice; or one a slice and point, (nslices, n_u, n_v) in C order.
   * @param rake The rake in degrees, from the strike towards up dip.
   * @param magnitude The magnitude in the fault plane.
   * @param nslices The slices in time.
   * @param n_u The cells or nodes along u.
   * @param n_v The cells or nodes along v.
   * @param options The opening (none by default) and the rule (`cell` by
   *        default).
   * @throws planetmodel::Error for values of none of those numbers, and as
   *         the constructor.
   */
  [[nodiscard]] static FaultSlip FromRake(const std::vector<double>& rake,
                                          const std::vector<double>& magnitude,
                                          std::size_t nslices = 1, std::size_t n_u = 1,
                                          std::size_t n_v = 1, const RakeOptions& options = {}) {
    const auto points = n_u * n_v;
    const auto total = nslices * points;
    const auto at = [points, total](const std::vector<double>& values, std::string_view what,
                                    std::size_t s, std::size_t ij) {
      if (values.size() == 1) return values[0];
      if (values.size() == points) return values[ij];
      if (values.size() == total) return values[s * points + ij];
      throw Error("fault slip", what,
                  std::format("{} values; one, one a point ({}), or one a slice and point ({})",
                              values.size(), points, total));
    };
    auto values = std::vector<double>{};
    values.reserve(3 * total);
    for (auto s = std::size_t{0}; s < nslices; ++s) {
      for (auto ij = std::size_t{0}; ij < points; ++ij) {
        const auto lambda = detail::Radians(at(rake, "rake", s, ij));
        const auto m = at(magnitude, "magnitude", s, ij);
        values.push_back(m * std::cos(lambda));
        values.push_back(m * std::sin(lambda));
        values.push_back(at(options.opening, "opening", s, ij));
      }
    }
    return FaultSlip(std::move(values), nslices, n_u, n_v, options.rule);
  }

  /** @brief The values, (nslices, n_u, n_v, 3) in C order. */
  [[nodiscard]] const std::vector<double>& Values() const noexcept { return values_; }
  /** @brief The slices in time. */
  [[nodiscard]] std::size_t NSlices() const noexcept { return nslices_; }
  /** @brief The cells or nodes along u. */
  [[nodiscard]] std::size_t NU() const noexcept { return n_u_; }
  /** @brief The cells or nodes along v. */
  [[nodiscard]] std::size_t NV() const noexcept { return n_v_; }
  /** @brief The rule, one of `SLIP_RULES`. */
  [[nodiscard]] const std::string& Rule() const noexcept { return rule_; }

  /**
   * @brief The fault components of a slice at (u, v), by the rule.
   * @param u The parameter along strike, in the square.
   * @param v The parameter down dip, in the square.
   * @param index The slice.
   * @throws planetmodel::Error for a slice the slip does not have.
   */
  [[nodiscard]] Vector3 Components(double u, double v, std::size_t index) const {
    if (index >= nslices_) {
      throw Error("fault slip", "", std::format("slice {} of a slip of {}", index, nslices_));
    }
    auto out = Vector3{};
    rule_reader_.Evaluate(u, v, 0, 0, 3 * index, 3, out);
    return out;
  }

 private:
  std::vector<double> values_;
  std::size_t nslices_ = 1, n_u_ = 1, n_v_ = 1;
  std::string rule_;
  detail::GridRule rule_reader_{};
};

/** @brief The parameters of a plane (`FaultPatch::Plane`). */
struct PlaneParameters {
  Vector3 centroid{};  /**< The centroid, in the frame's coordinates. */
  double strike = 0.0; /**< Degrees clockwise from north (from +y towards +x). */
  double dip = 0.0;    /**< Degrees down from the horizontal, in [0, 90]. */
  double length = 1.0; /**< The extent along strike, positive. */
  double width = 1.0;  /**< The extent down dip, positive. */
};

/** @brief The control grid of a structured patch (`FaultPatch::Structured`). */
struct ControlGrid {
  std::size_t n_u = 0; /**< Nodes along u. */
  std::size_t n_v = 0; /**< Nodes along v. */
  std::vector<Vector3>
      positions{}; /**< (n_u, n_v) positions in the frame's coordinates, C order. */
  std::string interpolation = "cubic"; /**< One of `FAULT_INTERPOLATIONS`. */
};

/**
 * @brief Where a point lies relative to a patch or a fault (`Locate`).
 *
 * `u` and `v` are the parameters of the point's nearest point x of the
 * patch over the closed square (of the nearest patch, `patch`, on a
 * fault); `distance` the length of p - x signed as (p - x).n, positive on
 * the hanging-wall side and zero counted positive; `inside` whether x is
 * not held on the square's boundary with p beyond it; `converged` whether
 * the search met its tolerance.
 */
struct FaultLocation {
  std::size_t patch = 0; /**< The patch, on a fault. */
  double u = 0.0;        /**< The nearest point's u. */
  double v = 0.0;        /**< The nearest point's v. */
  double distance = 0.0; /**< The signed distance. */
  bool inside = true;    /**< Whether the nearest point is not held on an edge with p beyond it. */
  bool converged = true; /**< Whether the search met its tolerance. */

  /** @brief Whether the nearest point lies on the edge u = -1/2. */
  [[nodiscard]] bool OnULow() const noexcept { return u == FAULT_SQUARE[0]; }
  /** @brief Whether the nearest point lies on the edge u = 1/2. */
  [[nodiscard]] bool OnUHigh() const noexcept { return u == FAULT_SQUARE[1]; }
  /** @brief Whether the nearest point lies on the edge v = -1/2, up dip. */
  [[nodiscard]] bool OnVLow() const noexcept { return v == FAULT_SQUARE[0]; }
  /** @brief Whether the nearest point lies on the edge v = 1/2, down dip. */
  [[nodiscard]] bool OnVHigh() const noexcept { return v == FAULT_SQUARE[1]; }
  /** @brief Whether the nearest point lies on any edge of the square. */
  [[nodiscard]] bool OnEdge() const noexcept {
    return OnULow() || OnUHigh() || OnVLow() || OnVHigh();
  }
};

/** @brief How `Locate` searches: the Python's arguments. */
struct LocateOptions {
  double rtol = 1e-10;         /**< A full step of at most this, in the square, has converged. */
  std::size_t iterations = 50; /**< The steps allowed. */
  bool strict = true;          /**< Refuse a search that has not converged. */
};

/**
 * @brief Points on one patch, each with what it stands for: Cartesian
 * points and the same in the frame's coordinates, unit normals, weights
 * (the areas they stand for), the Cartesian slip and the parameters of
 * each (`FaultPatch::Quadrature`).
 */
struct PatchQuadrature {
  std::vector<Vector3> points{};    /**< Cartesian. */
  std::vector<Vector3> positions{}; /**< In the frame's coordinates. */
  std::vector<Vector3> normals{};   /**< Unit, from the footwall into the hanging wall. */
  std::vector<double> weights{};    /**< Areas, in the length unit squared. */
  std::vector<Vector3> slip{};      /**< Cartesian. */
  std::vector<double> u{};          /**< The points' u. */
  std::vector<double> v{};          /**< The points' v. */

  /** @brief The number of points. */
  [[nodiscard]] std::size_t Size() const noexcept { return weights.size(); }
};

/**
 * @brief Points on a fault (`Fault::Quadrature`): a patch's quadrature
 * over every patch in order, with the patch of each point.
 */
struct FaultQuadrature : PatchQuadrature {
  std::vector<std::size_t> patch{}; /**< The points' patches. */
};

/**
 * @brief Points on a fault with the slip of every slice
 * (`Fault::SliceQuadrature`): the geometry once, as `FaultQuadrature`
 * holds it, and `slip[s][i]`, the Cartesian slip of slice s at point i.
 *
 * The slip at a date is the slices a date reads (`Fault::Rows`) under
 * their weights, `SlipOf`, as `Fault::Quadrature` gives it at that date.
 */
struct FaultSliceQuadrature {
  std::vector<Vector3> points{};            /**< Cartesian. */
  std::vector<Vector3> positions{};         /**< In the frame's coordinates. */
  std::vector<Vector3> normals{};           /**< Unit, from the footwall into the hanging wall. */
  std::vector<double> weights{};            /**< Areas, in the length unit squared. */
  std::vector<double> u{};                  /**< The points' u. */
  std::vector<double> v{};                  /**< The points' v. */
  std::vector<std::size_t> patch{};         /**< The points' patches. */
  std::vector<std::vector<Vector3>> slip{}; /**< Cartesian, a slice a row, a point a column. */

  /** @brief The number of points. */
  [[nodiscard]] std::size_t Size() const noexcept { return weights.size(); }
  /** @brief The number of slices. */
  [[nodiscard]] std::size_t NSlices() const noexcept { return slip.size(); }

  /**
   * @brief The Cartesian slip at every point under a date's rows: the
   * first row's weight times its slice, plus the second's where its
   * weight is not zero.
   * @param rows The rows, as `Fault::Rows` gives them.
   * @throws planetmodel::Error for a row naming a slice it lacks.
   */
  [[nodiscard]] std::vector<Vector3> SlipOf(const TimeRows& rows) const {
    const auto [first, second] = rows;
    if (first.first >= slip.size() || second.first >= slip.size()) {
      throw Error(
          "fault quadrature", "rows",
          std::format("name slices {} and {} of {}", first.first, second.first, slip.size()));
    }
    auto out = slip[first.first];
    for (auto& s : out) {
      for (auto& x : s) x *= first.second;
    }
    if (second.second != 0.0) {
      for (auto i = std::size_t{0}; i < out.size(); ++i) {
        for (auto d = 0; d < 3; ++d)
          out[i][d] = out[i][d] + second.second * slip[second.first][i][d];
      }
    }
    return out;
  }
};

/** @brief Points and weights over the square (`ParameterRule`). */
struct ParameterPoints {
  std::vector<double> u{}; /**< The points' u. */
  std::vector<double> v{}; /**< The points' v. */
  std::vector<double> w{}; /**< Their weights, summing to one. */
};

/**
 * @brief The Gauss-Legendre points and weights on [-1, 1], increasing,
 * found by Newton's method and made symmetric about zero.
 * @param n The number of points, at least one.
 * @throws planetmodel::Error if there are none.
 */
[[nodiscard]] inline std::pair<std::vector<double>, std::vector<double>> GaussLegendrePoints(
    std::size_t n) {
  if (n < 1) throw Error("GaussLegendrePoints", "", "needs at least one point");
  auto x = std::vector<double>(n);
  auto w = std::vector<double>(n);
  const auto N = static_cast<double>(n);
  for (auto i = std::size_t{0}; i < n; ++i) {
    // from the largest root down
    auto z = std::cos(std::numbers::pi * (static_cast<double>(i) + 0.75) / (N + 0.5));
    auto dp = 0.0;
    for (auto it = 0; it < 100; ++it) {
      auto p0 = 1.0, p1 = z;
      for (auto k = std::size_t{2}; k <= n; ++k) {
        const auto kk = static_cast<double>(k);
        const auto p2 = ((2.0 * kk - 1.0) * z * p1 - (kk - 1.0) * p0) / kk;
        p0 = p1;
        p1 = p2;
      }
      if (n == 1) p0 = 1.0;
      dp = N * (z * p1 - p0) / (z * z - 1.0);
      const auto step = p1 / dp;
      z -= step;
      if (std::abs(step) <= 1e-16) break;
    }
    x[n - 1 - i] = z;
    w[n - 1 - i] = 2.0 / ((1.0 - z * z) * dp * dp);
  }
  if (n == 1) {
    x[0] = 0.0;
    w[0] = 2.0;
  }
  auto xs = x, ws = w;
  for (auto i = std::size_t{0}; i < n; ++i) {
    xs[i] = (x[i] - x[n - 1 - i]) / 2.0;
    ws[i] = (w[i] + w[n - 1 - i]) / 2.0;
  }
  auto total = 0.0;
  for (const auto value : ws) total += value;
  for (auto& value : ws) value *= 2.0 / total;
  return {xs, ws};
}

/**
 * @brief `points` by `points` Gauss-Legendre points in each of n_u by n_v
 * equal cells of the square, with weights summing to one (the square's
 * area); u slowest, each direction over its cells in order and over the
 * points within each.
 * @param n_u The cells along u.
 * @param n_v The cells along v.
 * @param points The points each way in a cell; one is the midpoint rule.
 * @throws planetmodel::Error if any count is zero.
 */
[[nodiscard]] inline ParameterPoints ParameterRule(std::size_t n_u, std::size_t n_v,
                                                   std::size_t points = 1) {
  if (n_u < 1 || n_v < 1 || points < 1) {
    throw Error("ParameterRule", "", "n_u, n_v and points must be positive integers");
  }
  const auto [x, w] = GaussLegendrePoints(points);
  auto composite = [&x = x, &w = w](std::size_t n) {
    const auto e = detail::Linspace(FAULT_SQUARE[0], FAULT_SQUARE[1], n + 1);
    auto p = std::vector<double>{}, q = std::vector<double>{};
    for (auto c = std::size_t{0}; c < n; ++c) {
      const auto half = 0.5 * (e[c + 1] - e[c]);
      const auto mid = 0.5 * (e[c + 1] + e[c]);
      for (auto k = std::size_t{0}; k < x.size(); ++k) {
        p.push_back(mid + half * x[k]);
        q.push_back(half * w[k]);
      }
    }
    return std::pair{p, q};
  };
  const auto [pu, wu] = composite(n_u);
  const auto [pv, wv] = composite(n_v);
  auto out = ParameterPoints{};
  for (auto i = std::size_t{0}; i < pu.size(); ++i) {
    for (auto j = std::size_t{0}; j < pv.size(); ++j) {
      out.u.push_back(pu[i]);
      out.v.push_back(pv[j]);
      out.w.push_back(wu[i] * wv[j]);
    }
  }
  return out;
}

namespace detail {

/// A patch's quadrature points with what each stands for but its slip,
/// and the fault basis at each (`FaultPatch::Geometry`).
struct PatchGeometry {
  std::vector<Vector3> points{}, positions{}, normals{};
  std::vector<double> weights{}, u{}, v{};
  std::vector<Matrix3> basis{};
};

}  // namespace detail

/**
 * @brief One patch of a fault: a map of the parameter square, its frame,
 * and its slip (file comment).
 *
 * A plane in closed form, or a structured patch by its control
 * positions' rule; every method takes (u, v) in the square and refuses
 * one outside it.
 */
class FaultPatch {
 public:
  /** @brief The kinds of patch. */
  enum class Kind {
    Plane,     /**< A flat rectangle by its centroid, strike, dip, length and width. */
    Structured /**< Control positions on a uniform grid and a rule between them. */
  };

  /**
   * @brief A plane: x(u, v) = c + u L s + v W d, with s and d the strike
   * and down-dip directions from (east, north, up) at the centroid.
   * @param frame The frame.
   * @param plane Its parameters.
   * @param slip Its slip.
   * @throws planetmodel::Error for parameters out of range.
   */
  [[nodiscard]] static FaultPatch Plane(const FaultFrame& frame, const PlaneParameters& plane,
                                        FaultSlip slip) {
    if (!std::ranges::all_of(plane.centroid, [](double x) { return std::isfinite(x); }) ||
        !std::isfinite(plane.strike)) {
      throw Error("fault patch", "plane", "the centroid and the strike must be finite");
    }
    if (!(plane.dip >= 0.0 && plane.dip <= 90.0)) {
      throw Error("fault patch", "plane", "dip must lie in [0, 90] degrees");
    }
    if (!(plane.length > 0.0 && plane.width > 0.0 && std::isfinite(plane.length) &&
          std::isfinite(plane.width))) {
      throw Error("fault patch", "plane", "length and width must be positive");
    }
    auto out = FaultPatch(Kind::Plane, frame, std::move(slip));
    out.plane_ = plane;
    const auto B = frame.Basis(plane.centroid);
    const auto phi = detail::Radians(plane.strike);
    const auto delta = detail::Radians(plane.dip);
    for (auto i = 0; i < 3; ++i) {
      const auto east = B[i][0], north = B[i][1], up = B[i][2];
      const auto s = std::sin(phi) * east + std::cos(phi) * north;
      const auto r = std::cos(phi) * east - std::sin(phi) * north;
      out.xu_[i] = plane.length * s;
      out.xv_[i] = plane.width * (std::cos(delta) * r - std::sin(delta) * up);
    }
    out.origin_ = frame.CartesianPoint(plane.centroid);
    return out;
  }

  /**
   * @brief A structured patch through control positions, read as
   * Cartesian points by its rule.
   * @param frame The frame.
   * @param grid The control positions and the rule.
   * @param slip Its slip.
   * @throws planetmodel::Error for a rule unknown, too few nodes, positions
   *         not finite, or an area element not positive at a node.
   */
  [[nodiscard]] static FaultPatch Structured(const FaultFrame& frame, ControlGrid grid,
                                             FaultSlip slip) {
    if (!std::ranges::contains(FAULT_INTERPOLATIONS, std::string_view{grid.interpolation})) {
      throw Error("fault patch", "interpolation",
                  std::format("{} is not one of linear, cubic", grid.interpolation));
    }
    const auto least = MinNodes(grid.interpolation);
    if (std::min(grid.n_u, grid.n_v) < least || grid.positions.size() != grid.n_u * grid.n_v) {
      throw Error("fault patch", "",
                  std::format("the {} rule needs at least {} nodes each way, and {} by {} "
                              "positions",
                              grid.interpolation, least, grid.n_u, grid.n_v));
    }
    auto control = std::vector<double>{};
    control.reserve(3 * grid.positions.size());
    for (const auto& p : grid.positions) {
      if (!std::ranges::all_of(p, [](double x) { return std::isfinite(x); })) {
        throw Error("fault patch", "", "control positions must be finite");
      }
      const auto x = frame.CartesianPoint(p);
      control.insert(control.end(), x.begin(), x.end());
    }
    auto out = FaultPatch(Kind::Structured, frame, std::move(slip));
    out.rule_ = detail::GridRule(control, grid.n_u, grid.n_v, 3, grid.interpolation);
    out.grid_ = std::move(grid);
    const auto nu = detail::SquareNodes(out.grid_.n_u);
    const auto nv = detail::SquareNodes(out.grid_.n_v);
    for (const auto u : nu) {
      for (const auto v : nv) {
        if (!(out.AreaElement(u, v) > 0.0)) {
          throw Error("fault patch", "",
                      "the patch is degenerate: its tangents vanish or are parallel at a "
                      "control node");
        }
      }
    }
    out.Prepare(control);
    return out;
  }

  /** @brief Which kind of patch it is. */
  [[nodiscard]] Kind GetKind() const noexcept { return kind_; }
  /** @brief `plane` or `structured`, as a fault file states it. */
  [[nodiscard]] std::string_view KindName() const noexcept {
    return kind_ == Kind::Plane ? "plane" : "structured";
  }
  /** @brief The frame. */
  [[nodiscard]] const FaultFrame& Frame() const noexcept { return frame_; }
  /** @brief The slip. */
  [[nodiscard]] const FaultSlip& Slip() const noexcept { return slip_; }
  /** @brief A plane's parameters (meaningful for a plane). */
  [[nodiscard]] const PlaneParameters& PlaneOf() const noexcept { return plane_; }
  /** @brief A structured patch's control grid (empty for a plane). */
  [[nodiscard]] const ControlGrid& Grid() const noexcept { return grid_; }

  /**
   * @brief x(u, v), Cartesian.
   * @param u Along strike.
   * @param v Down dip.
   * @throws planetmodel::Error for (u, v) outside the square.
   */
  [[nodiscard]] Vector3 Cartesian(double u, double v) const {
    detail::CheckParameters(u, v);
    return Map(u, v, 0, 0);
  }

  /**
   * @brief x(u, v) in the frame's coordinates.
   * @param u Along strike.
   * @param v Down dip.
   * @throws planetmodel::Error for (u, v) outside the square.
   */
  [[nodiscard]] Vector3 Position(double u, double v) const {
    return frame_.Coordinates(Cartesian(u, v));
  }

  /**
   * @brief (x_u, x_v), Cartesian.
   * @param u Along strike.
   * @param v Down dip.
   * @throws planetmodel::Error for (u, v) outside the square.
   */
  [[nodiscard]] std::array<Vector3, 2> Tangents(double u, double v) const {
    detail::CheckParameters(u, v);
    return {Map(u, v, 1, 0), Map(u, v, 0, 1)};
  }

  /**
   * @brief (x_uu, x_uv, x_vv), Cartesian: zero on a plane, x_uu and x_vv
   * zero under the bilinear rule.
   * @param u Along strike.
   * @param v Down dip.
   * @throws planetmodel::Error for (u, v) outside the square.
   */
  [[nodiscard]] std::array<Vector3, 3> SecondDerivatives(double u, double v) const {
    detail::CheckParameters(u, v);
    return {Map(u, v, 2, 0), Map(u, v, 1, 1), Map(u, v, 0, 2)};
  }

  /**
   * @brief The unit normal x_v x x_u over its length, from the footwall
   * into the hanging wall.
   * @param u Along strike.
   * @param v Down dip.
   * @throws planetmodel::Error for (u, v) outside the square.
   */
  [[nodiscard]] Vector3 Normal(double u, double v) const {
    const auto [xu, xv] = Tangents(u, v);
    return detail::Unit(detail::Cross(xv, xu));
  }

  /**
   * @brief The length of x_u x x_v: the area per unit of u v.
   * @param u Along strike.
   * @param v Down dip.
   * @throws planetmodel::Error for (u, v) outside the square.
   */
  [[nodiscard]] double AreaElement(double u, double v) const {
    const auto [xu, xv] = Tangents(u, v);
    return detail::Length(detail::Cross(xu, xv));
  }

  /**
   * @brief The fault basis as columns (strike s, up dip a, normal n),
   * Cartesian: s = x_u / |x_u|, a = n x s.
   * @param u Along strike.
   * @param v Down dip.
   * @throws planetmodel::Error for (u, v) outside the square.
   */
  [[nodiscard]] Matrix3 FaultBasis(double u, double v) const {
    const auto [xu, xv] = Tangents(u, v);
    return detail::FaultBasisOf(xu, xv);
  }

  /**
   * @brief The slip of a slice at (u, v): its fault components read by
   * the slip's rule, given in a basis through the fault basis there.
   * @param u Along strike.
   * @param v Down dip.
   * @param index The slice.
   * @param basis The basis of the answer.
   * @throws planetmodel::Error for (u, v) outside the square or a slice
   *         the slip lacks.
   */
  [[nodiscard]] Vector3 SlipVector(double u, double v, std::size_t index = 0,
                                   SlipBasis basis = SlipBasis::Ambient) const {
    detail::CheckParameters(u, v);
    const auto comp = slip_.Components(u, v, index);
    if (basis == SlipBasis::Fault) return comp;
    const auto cart = detail::OnBasis(FaultBasis(u, v), comp);
    if (basis == SlipBasis::Cartesian) return cart;
    const auto B = frame_.Basis(Position(u, v));
    auto out = Vector3{};
    for (auto i = 0; i < 3; ++i) out[i] = B[0][i] * cart[0] + B[1][i] * cart[1] + B[2][i] * cart[2];
    return out;
  }

  /**
   * @brief The patch's quadrature (`ParameterRule`), with the slip of a
   * slice; a fault's quadratures say which patch each point is on
   * (`Fault::Quadrature`, `Fault::SliceQuadrature`), which a patch alone
   * does not know.
   * @param n_u The cells along u.
   * @param n_v The cells along v.
   * @param points The Gauss-Legendre points each way in a cell.
   * @param index The slice.
   * @throws planetmodel::Error for a count of zero or a slice the slip
   *         lacks.
   */
  [[nodiscard]] PatchQuadrature Quadrature(std::size_t n_u, std::size_t n_v, std::size_t points = 1,
                                           std::size_t index = 0) const {
    auto g = Geometry(n_u, n_v, points);
    auto q = PatchQuadrature{.points = std::move(g.points),
                             .positions = std::move(g.positions),
                             .normals = std::move(g.normals),
                             .weights = std::move(g.weights),
                             .slip = {},
                             .u = std::move(g.u),
                             .v = std::move(g.v)};
    for (auto p = std::size_t{0}; p < q.Size(); ++p) {
      q.slip.push_back(detail::OnBasis(g.basis[p], slip_.Components(q.u[p], q.v[p], index)));
    }
    return q;
  }

  /**
   * @brief A point's nearest point of the patch, its signed distance, and
   * whether it is inside: in closed form on a plane; elsewhere from the
   * nearest point of a coarse sample, by projected Newton steps with a
   * line search, every cell of a bilinear patch whose bounding sphere
   * comes within the nearest sample's distance searched.
   * @param point The point, Cartesian.
   * @param options The tolerance, the steps allowed, and whether a search
   *        that has not converged is refused.
   * @throws planetmodel::Error for a point not finite, or a search that has
   *         not converged where `options.strict`.
   */
  [[nodiscard]] FaultLocation Locate(const Vector3& point,
                                     const LocateOptions& options = {}) const {
    if (!std::ranges::all_of(point, [](double x) { return std::isfinite(x); })) {
      throw Error("fault patch", "locate", "coordinates must be finite");
    }
    if (!(options.rtol > 0.0) || options.iterations < 1) {
      throw Error("fault patch", "locate", "rtol must be positive and iterations at least one");
    }
    const auto lo = FAULT_SQUARE[0], hi = FAULT_SQUARE[1];
    if (kind_ == Kind::Plane) {
      const auto r = detail::Minus(point, origin_);
      auto w = std::array<double, 2>{detail::Dot(r, xu_) / (plane_.length * plane_.length),
                                     detail::Dot(r, xv_) / (plane_.width * plane_.width)};
      const auto inside = w[0] >= lo - options.rtol && w[0] <= hi + options.rtol &&
                          w[1] >= lo - options.rtol && w[1] <= hi + options.rtol;
      w = {std::clamp(w[0], lo, hi), std::clamp(w[1], lo, hi)};
      return Located(point, w, inside, true, options.strict);
    }
    // the nearest sample
    auto nearest = std::size_t{0};
    auto best = std::numeric_limits<double>::infinity();
    for (auto k = std::size_t{0}; k < samples_.size(); ++k) {
      const auto d = detail::Minus(point, samples_[k]);
      const auto d2 = detail::Dot(d, d);
      if (d2 < best) {
        best = d2;
        nearest = k;
      }
    }
    const auto bound = std::sqrt(best);
    const auto start = std::array<double, 2>{sample_u_[nearest / sample_v_.size()],
                                             sample_v_[nearest % sample_v_.size()]};
    auto w = start;
    auto converged = false;
    if (grid_.interpolation != "linear") {
      std::tie(w, converged) = Descend(point, start, {lo, lo}, {hi, hi}, options);
    } else {
      const auto reach = bound * (1.0 + 1e-12) + 1e-12 * centre_scale_;
      const auto nu = detail::SquareNodes(grid_.n_u);
      const auto nv = detail::SquareNodes(grid_.n_v);
      auto least = std::numeric_limits<double>::infinity();
      for (auto k = std::size_t{0}; k < centres_.size(); ++k) {
        if (!(detail::Length(detail::Minus(point, centres_[k])) - radii_[k] <= reach)) continue;
        const auto iu = k / (grid_.n_v - 1), iv = k % (grid_.n_v - 1);
        const auto box_lo = std::array<double, 2>{nu[iu], nv[iv]};
        const auto box_hi = std::array<double, 2>{nu[iu + 1], nv[iv + 1]};
        const auto from = std::array<double, 2>{std::clamp(start[0], box_lo[0], box_hi[0]),
                                                std::clamp(start[1], box_lo[1], box_hi[1])};
        const auto [found, ok] = Descend(point, from, box_lo, box_hi, options);
        const auto r = detail::Minus(point, Map(found[0], found[1], 0, 0));
        const auto d2 = detail::Dot(r, r);
        if (d2 < least) {
          least = d2;
          w = found;
          converged = ok;
        }
      }
    }
    // inside unless held on the square's edge with the unconstrained foot
    // of the perpendicular beyond it
    const auto order = SecondOrder(point, w, std::nullopt);
    const auto step = NewtonStep(order, {false, false});
    auto inside = true;
    for (auto k = 0; k < 2; ++k) {
      const auto free = w[k] + step[k];
      if ((w[k] <= lo && free < lo - options.rtol) || (w[k] >= hi && free > hi + options.rtol)) {
        inside = false;
      }
    }
    return Located(point, w, inside, converged, options.strict);
  }

 private:
  friend class Fault;

  FaultPatch(Kind kind, const FaultFrame& frame, FaultSlip slip)
      : kind_(kind), frame_(frame), slip_(std::move(slip)) {}

  /// The points of `ParameterRule(n_u, n_v, points)` on the patch with
  /// what each stands for but its slip, and the fault basis at each, which
  /// turns a slice's components there into its Cartesian slip: the
  /// tangents formed once a point, for this patch's quadrature and a
  /// fault's.
  [[nodiscard]] detail::PatchGeometry Geometry(std::size_t n_u, std::size_t n_v,
                                               std::size_t points) const {
    const auto rule = ParameterRule(n_u, n_v, points);
    auto g = detail::PatchGeometry{};
    for (auto p = std::size_t{0}; p < rule.u.size(); ++p) {
      const auto u = rule.u[p], v = rule.v[p];
      const auto x = Cartesian(u, v);
      const auto [xu, xv] = Tangents(u, v);
      const auto c = detail::Cross(xv, xu);
      const auto area = detail::Length(c);
      g.points.push_back(x);
      g.positions.push_back(frame_.Coordinates(x));
      g.normals.push_back({c[0] / area, c[1] / area, c[2] / area});
      g.weights.push_back(area * rule.w[p]);
      g.u.push_back(u);
      g.v.push_back(v);
      g.basis.push_back(detail::FaultBasisOf(xu, xv));
    }
    return g;
  }

  /// The map or a derivative of it, (u, v) unchecked.
  [[nodiscard]] Vector3 Map(double u, double v, std::size_t du, std::size_t dv) const {
    if (kind_ == Kind::Plane) {
      if (du + dv == 0) {
        return {origin_[0] + u * xu_[0] + v * xv_[0], origin_[1] + u * xu_[1] + v * xv_[1],
                origin_[2] + u * xu_[2] + v * xv_[2]};
      }
      if (du == 1 && dv == 0) return xu_;
      if (du == 0 && dv == 1) return xv_;
      return {0.0, 0.0, 0.0};
    }
    auto out = Vector3{};
    rule_.Evaluate(u, v, du, dv, 0, 3, out);
    return out;
  }

  /// The coarse sample `Locate` starts from and, under the bilinear rule,
  /// the sphere about each cell.
  void Prepare(const std::vector<double>& control) {
    const auto su = std::clamp<std::size_t>(2 * grid_.n_u - 1, 9, 129);
    const auto sv = std::clamp<std::size_t>(2 * grid_.n_v - 1, 9, 129);
    sample_u_ = detail::SquareNodes(su);
    sample_v_ = detail::SquareNodes(sv);
    for (const auto u : sample_u_) {
      for (const auto v : sample_v_) samples_.push_back(Map(u, v, 0, 0));
    }
    if (grid_.interpolation != "linear") return;
    const auto n_v = grid_.n_v;
    auto node = [&control, n_v](std::size_t i, std::size_t j) {
      const auto k = 3 * (i * n_v + j);
      return Vector3{control[k], control[k + 1], control[k + 2]};
    };
    centre_scale_ = 0.0;
    for (auto i = std::size_t{0}; i + 1 < grid_.n_u; ++i) {
      for (auto j = std::size_t{0}; j + 1 < n_v; ++j) {
        const auto corners =
            std::array<Vector3, 4>{node(i, j), node(i + 1, j), node(i, j + 1), node(i + 1, j + 1)};
        auto c = Vector3{};
        for (auto d = 0; d < 3; ++d) {
          c[d] = (((corners[0][d] + corners[1][d]) + corners[2][d]) + corners[3][d]) / 4.0;
          centre_scale_ = std::max(centre_scale_, std::abs(c[d]));
        }
        auto rho = 0.0;
        for (const auto& corner : corners)
          rho = std::max(rho, detail::Length(detail::Minus(corner, c)));
        centres_.push_back(c);
        radii_.push_back(rho);
      }
    }
  }

  /// At (u, v): the residual r = p - x, the gradient of |r|^2 / 2, its
  /// Hessian and its Gauss-Newton part; on the upper edge of a box inside
  /// the square the derivatives are the box's own, read a hair inside it.
  struct Order {
    Vector3 r{};
    std::array<double, 2> g{};
    std::array<std::array<double, 2>, 2> H{}, A{};
  };

  [[nodiscard]] Order SecondOrder(
      const Vector3& p, std::array<double, 2> w,
      std::optional<std::pair<std::array<double, 2>, std::array<double, 2>>> box) const {
    auto out = Order{};
    out.r = detail::Minus(p, Map(w[0], w[1], 0, 0));
    if (box) {
      const auto& [lo, hi] = *box;
      for (auto k = 0; k < 2; ++k) {
        if (w[k] >= hi[k] && hi[k] < FAULT_SQUARE[1]) w[k] = hi[k] - 1e-9 * (hi[k] - lo[k]);
      }
    }
    const auto xu = Map(w[0], w[1], 1, 0), xv = Map(w[0], w[1], 0, 1);
    const auto xuu = Map(w[0], w[1], 2, 0), xuv = Map(w[0], w[1], 1, 1),
               xvv = Map(w[0], w[1], 0, 2);
    out.g = {-detail::Dot(out.r, xu), -detail::Dot(out.r, xv)};
    out.A = {
        {{detail::Dot(xu, xu), detail::Dot(xu, xv)}, {detail::Dot(xu, xv), detail::Dot(xv, xv)}}};
    out.H = out.A;
    out.H[0][0] -= detail::Dot(out.r, xuu);
    out.H[1][1] -= detail::Dot(out.r, xvv);
    out.H[0][1] -= detail::Dot(out.r, xuv);
    out.H[1][0] = out.H[0][1];
    return out;
  }

  /// The projected Newton step: zero on the held coordinates, the Hessian
  /// where it is positive definite and the Gauss-Newton matrix otherwise.
  [[nodiscard]] static std::array<double, 2> NewtonStep(const Order& o, std::array<bool, 2> held) {
    auto out = std::array<double, 2>{0.0, 0.0};
    if (!held[0] && !held[1]) {
      const auto det = o.H[0][0] * o.H[1][1] - o.H[0][1] * o.H[0][1];
      const auto& M = o.H[0][0] > 0.0 && det > 0.0 ? o.H : o.A;
      // LAPACK's partial pivoting on the 2 by 2
      auto a = M[0][0], b = M[0][1], c = M[1][0], d = M[1][1];
      auto g0 = o.g[0], g1 = o.g[1];
      if (std::abs(c) > std::abs(a)) {
        std::swap(a, c);
        std::swap(b, d);
        std::swap(g0, g1);
      }
      const auto l = c * (1.0 / a);
      const auto u22 = d - l * b;
      const auto y1 = g1 - l * g0;
      const auto x1 = y1 / u22;
      const auto x0 = (g0 - x1 * b) / a;
      return {-x0, -x1};
    }
    for (auto k = 0; k < 2; ++k) {
      if (!held[k] && held[1 - k]) {
        const auto m = o.H[k][k] > 0.0 ? o.H[k][k] : o.A[k][k];
        out[k] = -o.g[k] / m;
      }
    }
    return out;
  }

  /// Projected Newton with an Armijo line search over a box, from a start:
  /// the parameters reached and whether a full step fell to rtol.
  [[nodiscard]] std::pair<std::array<double, 2>, bool> Descend(const Vector3& p,
                                                               std::array<double, 2> w,
                                                               std::array<double, 2> lo,
                                                               std::array<double, 2> hi,
                                                               const LocateOptions& options) const {
    constexpr auto armijo = 1e-4;
    constexpr auto halvings = 40;
    const auto eps = std::numeric_limits<double>::epsilon();
    for (auto it = std::size_t{0}; it < options.iterations; ++it) {
      const auto o = SecondOrder(p, w, std::pair{lo, hi});
      const auto held =
          std::array<bool, 2>{(w[0] <= lo[0] && o.g[0] > 0.0) || (w[0] >= hi[0] && o.g[0] < 0.0),
                              (w[1] <= lo[1] && o.g[1] > 0.0) || (w[1] >= hi[1] && o.g[1] < 0.0)};
      const auto step = NewtonStep(o, held);
      auto clip = [&lo, &hi, &w, &step](double alpha) {
        return std::array<double, 2>{std::clamp(w[0] + alpha * step[0], lo[0], hi[0]),
                                     std::clamp(w[1] + alpha * step[1], lo[1], hi[1])};
      };
      auto trial = clip(1.0);
      const auto done =
          std::max(std::abs(trial[0] - w[0]), std::abs(trial[1] - w[1])) <= options.rtol;
      const auto f0 = 0.5 * detail::Dot(o.r, o.r);
      const auto noise = 8.0 * eps * (detail::Dot(p, p) + f0);
      auto alpha = 1.0;
      for (auto h = 0; h < halvings && !done; ++h) {
        const auto rt = detail::Minus(p, Map(trial[0], trial[1], 0, 0));
        const auto descent = o.g[0] * (trial[0] - w[0]) + o.g[1] * (trial[1] - w[1]);
        if (!(0.5 * detail::Dot(rt, rt) > f0 + noise + armijo * descent)) break;
        alpha *= 0.5;
        trial = clip(alpha);
      }
      w = trial;
      if (done) return {w, true};
    }
    return {w, false};
  }

  /// The location of p at its nearest parameters.
  [[nodiscard]] FaultLocation Located(const Vector3& p, const std::array<double, 2>& w, bool inside,
                                      bool converged, bool strict) const {
    if (strict && !converged) {
      throw Error("fault patch", "locate",
                  "did not converge; raise iterations, or pass strict false to flag it in "
                  "converged");
    }
    const auto r = detail::Minus(p, Map(w[0], w[1], 0, 0));
    const auto sign = detail::Dot(r, Normal(w[0], w[1])) < 0.0 ? -1.0 : 1.0;
    return FaultLocation{.patch = 0,
                         .u = w[0],
                         .v = w[1],
                         .distance = sign * detail::Length(r),
                         .inside = inside,
                         .converged = converged};
  }

  Kind kind_;
  FaultFrame frame_;
  FaultSlip slip_;
  PlaneParameters plane_{};
  ControlGrid grid_{};
  Vector3 origin_{}, xu_{}, xv_{};
  detail::GridRule rule_{};
  std::vector<double> sample_u_{}, sample_v_{};
  std::vector<Vector3> samples_{};
  std::vector<Vector3> centres_{};
  std::vector<double> radii_{};
  double centre_scale_ = 0.0;
};

/** @brief What `Fault::SlipAt` reads besides the points. */
struct SlipAtOptions {
  std::optional<double> date{};                        /**< The date; a static slip takes none. */
  Extrapolation extrapolation = Extrapolation::Refuse; /**< Outside the time span. */
  SlipBasis basis = SlipBasis::Ambient;                /**< The basis of the answer. */
  double rtol = 1e-10;                                 /**< `Locate`'s tolerance. */
  double time_rtol = TIME_RTOL; /**< The date's tolerance on a time node (`DateOptions`). */
};

/**
 * @brief Patches in one frame, their slip on one time axis: what a fault
 * file holds (`ReadFaultFile`).
 *
 * The support is `TimeSupport::Static` (one slice, any date, and no date
 * given) or `TimeSupport::Node` (a slice a time node, read between them
 * by the time rule, `TimeRowsAt`), `FAULT_SUPPORTS` by name; the slip at
 * a node is the cumulative slip.  A date is read as `DateOptions` say: a
 * date within `time_rtol` of a node, relative to the scale of the axis
 * (`TIME_RTOL` by default), is that node, and outside the span a `linear`
 * or `previous` slip refuses the date or holds its nearest slice, a
 * `none` slip refusing any date not a node either way.  `Scales` says
 * what the numbers are in.
 */
class Fault {
 public:
  /**
   * @brief A fault from its patches.
   * @param patches At least one, all in one frame, each with the slices
   *        the support implies.
   * @param support `TimeSupport::Static` or `TimeSupport::Node`.
   * @param time_interpolation `TimeRule::None` for a static slip; any rule
   *        for a node slip.
   * @param time The node times, finite and strictly increasing, for a node
   *        slip.
   * @param scales What the numbers are in.
   * @throws planetmodel::Error for any of these not so.
   */
  explicit Fault(std::vector<FaultPatch> patches, TimeSupport support = TimeSupport::Static,
                 TimeRule time_interpolation = TimeRule::None,
                 std::optional<std::vector<double>> time = {}, Scales scales = {})
      : patches_(std::move(patches)),
        support_(support),
        rule_(time_interpolation),
        time_(std::move(time)),
        scales_(scales) {
    auto fail = [](std::string_view where, std::string_view message) {
      throw Error("fault", where, message);
    };
    if (patches_.empty()) fail("", "a fault has at least one patch");
    for (const auto& p : patches_) {
      if (!(p.Frame() == patches_.front().Frame())) {
        fail("", "every patch of a fault is in the same frame");
      }
    }
    if (!std::ranges::contains(FAULT_SUPPORTS, SupportName())) {
      fail("support", std::format("is {}, not one of a fault's: static, node", SupportName()));
    }
    if (support_ != TimeSupport::Node && rule_ != TimeRule::None) {
      fail("time_interpolation",
           std::format("{} is not a rule of a {} slip", TimeInterpolationName(), SupportName()));
    }
    const auto node = support_ == TimeSupport::Node;
    if (node != time_.has_value()) {
      fail("time", node ? "a node slip needs its time axis" : "a static slip uses no time axis");
    }
    if (time_) {
      auto last = -std::numeric_limits<double>::infinity();
      for (const auto t : *time_) {
        if (!(std::isfinite(t) && t > last)) fail("time", "must be finite and strictly increasing");
        last = t;
      }
      if (time_->empty()) fail("time", "must be finite and strictly increasing");
    }
    const auto nslices = time_ ? time_->size() : std::size_t{1};
    for (auto k = std::size_t{0}; k < patches_.size(); ++k) {
      if (patches_[k].Slip().NSlices() != nslices) {
        fail(std::format("patch_{}", k),
             std::format("has {} slices of slip; a {} fault has {}", patches_[k].Slip().NSlices(),
                         SupportName(), nslices));
      }
    }
  }

  /** @brief The patches, in order. */
  [[nodiscard]] const std::vector<FaultPatch>& Patches() const noexcept { return patches_; }
  /** @brief The frame every patch is in. */
  [[nodiscard]] const FaultFrame& Frame() const noexcept { return patches_.front().Frame(); }
  /** @brief How the slices sit in time: static or node. */
  [[nodiscard]] TimeSupport Support() const noexcept { return support_; }
  /** @brief The support's name, `static` or `node`, as a fault file states it. */
  [[nodiscard]] std::string_view SupportName() const noexcept { return TimeSupportName(support_); }
  /** @brief The rule between nodes. */
  [[nodiscard]] TimeRule TimeInterpolation() const noexcept { return rule_; }
  /** @brief The rule's name, one of `TIME_METHODS`, as a fault file states it. */
  [[nodiscard]] std::string_view TimeInterpolationName() const noexcept {
    return TimeRuleName(rule_);
  }
  /** @brief The node times of a node slip. */
  [[nodiscard]] const std::optional<std::vector<double>>& Time() const noexcept { return time_; }
  /** @brief The slices every patch's slip has. */
  [[nodiscard]] std::size_t NSlices() const noexcept { return time_ ? time_->size() : 1; }
  /** @brief What one unit of each number is, in SI. */
  [[nodiscard]] const Scales& GetScales() const noexcept { return scales_; }

  /**
   * @brief The slices a date reads and their weights; a static slip reads
   * its one slice, and takes no date.
   * @param date The date; a static slip takes none.
   * @param options What to do outside the span, and the date's tolerance
   *        on a node and on an end of the span, relative to the scale of
   *        the axis (`TimeRowsAt`).
   * @throws planetmodel::Error for a date refused, none where one is
   *         needed, or one given to a static slip.
   */
  [[nodiscard]] TimeRows Rows(std::optional<double> date, const DateOptions& options = {}) const {
    if (support_ == TimeSupport::Static && date) {
      throw Error("fault", "slip", "is static, the same at every date, and takes no date");
    }
    const auto rows = TimeRowsAt(support_, rule_, date, time_, std::nullopt, options.extrapolation,
                                 "fault", "slip", options.time_rtol);
    return rows ? *rows : TimeRows{TimeRow{0, 1.0}, TimeRow{0, 0.0}};
  }

  /**
   * @brief The slip of a patch at (u, v) and a date, in a basis.
   * @param patch The patch.
   * @param u Along strike.
   * @param v Down dip.
   * @param date The date; a static slip takes none.
   * @param basis The basis of the answer.
   * @param options How the date is read (`Rows`).
   * @throws planetmodel::Error for a patch the fault lacks, (u, v) outside
   *         the square, or a date refused.
   */
  [[nodiscard]] Vector3 SlipVector(std::size_t patch, double u, double v,
                                   std::optional<double> date = {},
                                   SlipBasis basis = SlipBasis::Ambient,
                                   const DateOptions& options = {}) const {
    const auto& p = PatchAt(patch);
    const auto [first, second] = Rows(date, options);
    auto out = p.SlipVector(u, v, first.first, basis);
    for (auto& x : out) x *= first.second;
    if (second.second != 0.0) {
      const auto more = p.SlipVector(u, v, second.first, basis);
      for (auto i = 0; i < 3; ++i) out[i] = out[i] + second.second * more[i];
    }
    return out;
  }

  /**
   * @brief Every patch's quadrature, with the slip at a date, one count of
   * cells each way a patch.
   * @param n_u The cells along u, one a patch.
   * @param n_v The cells along v, one a patch.
   * @param points The Gauss-Legendre points each way in a cell.
   * @param date The date; a static slip takes none.
   * @param options How the date is read (`Rows`).
   * @throws planetmodel::Error for counts not one a patch, or a date
   *         refused.
   */
  [[nodiscard]] FaultQuadrature Quadrature(std::span<const std::size_t> n_u,
                                           std::span<const std::size_t> n_v, std::size_t points = 1,
                                           std::optional<double> date = {},
                                           const DateOptions& options = {}) const {
    const auto [first, second] = Rows(date, options);
    CheckCounts(n_u, n_v);
    auto out = FaultQuadrature{};
    for (auto k = std::size_t{0}; k < patches_.size(); ++k) {
      const auto& p = patches_[k];
      const auto g = p.Geometry(n_u[k], n_v[k], points);
      Append(out, g, k);
      for (auto i = std::size_t{0}; i < g.weights.size(); ++i) {
        auto slip = detail::OnBasis(g.basis[i], p.Slip().Components(g.u[i], g.v[i], first.first));
        for (auto& x : slip) x *= first.second;
        if (second.second != 0.0) {
          const auto more =
              detail::OnBasis(g.basis[i], p.Slip().Components(g.u[i], g.v[i], second.first));
          for (auto d = 0; d < 3; ++d) slip[d] = slip[d] + second.second * more[d];
        }
        out.slip.push_back(slip);
      }
    }
    return out;
  }

  /**
   * @brief Every patch's quadrature with the same counts, with the slip at
   * a date.
   * @param n_u The cells along u on every patch.
   * @param n_v The cells along v on every patch.
   * @param points The Gauss-Legendre points each way in a cell.
   * @param date The date; a static slip takes none.
   * @param options How the date is read (`Rows`).
   * @throws planetmodel::Error for a count of zero, or a date refused.
   */
  [[nodiscard]] FaultQuadrature Quadrature(std::size_t n_u, std::size_t n_v, std::size_t points = 1,
                                           std::optional<double> date = {},
                                           const DateOptions& options = {}) const {
    const auto nu = std::vector<std::size_t>(patches_.size(), n_u);
    const auto nv = std::vector<std::size_t>(patches_.size(), n_v);
    return Quadrature(nu, nv, points, date, options);
  }

  /**
   * @brief Every patch's quadrature with the slip of every slice: the
   * points, normals, weights, parameters and patches once, as
   * `Quadrature` gives them, and each slice's Cartesian slip at them, the
   * geometry and the fault basis formed once a point.  The slip at a date
   * is `SlipOf(Rows(date))`, what `Quadrature` gives at that date.
   * @param n_u The cells along u, one a patch.
   * @param n_v The cells along v, one a patch.
   * @param points The Gauss-Legendre points each way in a cell.
   * @throws planetmodel::Error for counts not one a patch, or a count of
   *         zero.
   */
  [[nodiscard]] FaultSliceQuadrature SliceQuadrature(std::span<const std::size_t> n_u,
                                                     std::span<const std::size_t> n_v,
                                                     std::size_t points = 1) const {
    CheckCounts(n_u, n_v);
    auto out = FaultSliceQuadrature{};
    out.slip.resize(NSlices());
    for (auto k = std::size_t{0}; k < patches_.size(); ++k) {
      const auto& p = patches_[k];
      const auto g = p.Geometry(n_u[k], n_v[k], points);
      Append(out, g, k);
      for (auto i = std::size_t{0}; i < g.weights.size(); ++i) {
        for (auto s = std::size_t{0}; s < NSlices(); ++s) {
          out.slip[s].push_back(
              detail::OnBasis(g.basis[i], p.Slip().Components(g.u[i], g.v[i], s)));
        }
      }
    }
    return out;
  }

  /**
   * @brief Every patch's quadrature with the same counts, with the slip of
   * every slice (the overload above).
   * @param n_u The cells along u on every patch.
   * @param n_v The cells along v on every patch.
   * @param points The Gauss-Legendre points each way in a cell.
   * @throws planetmodel::Error for a count of zero.
   */
  [[nodiscard]] FaultSliceQuadrature SliceQuadrature(std::size_t n_u, std::size_t n_v,
                                                     std::size_t points = 1) const {
    const auto nu = std::vector<std::size_t>(patches_.size(), n_u);
    const auto nv = std::vector<std::size_t>(patches_.size(), n_v);
    return SliceQuadrature(nu, nv, points);
  }

  /**
   * @brief A point's nearest patch (the smallest distance, the first patch
   * on a tie) and its location on it (`FaultPatch::Locate`).
   * @param point The point, Cartesian.
   * @param options As `FaultPatch::Locate` takes them; a search that did
   *        not converge on the chosen patch is refused where strict.
   * @throws planetmodel::Error for a point not finite, or a search refused.
   */
  [[nodiscard]] FaultLocation Locate(const Vector3& point,
                                     const LocateOptions& options = {}) const {
    auto lenient = options;
    lenient.strict = false;
    auto best = FaultLocation{};
    for (auto k = std::size_t{0}; k < patches_.size(); ++k) {
      auto found = patches_[k].Locate(point, lenient);
      found.patch = k;
      if (k == 0 || std::abs(found.distance) < std::abs(best.distance)) best = found;
    }
    if (options.strict && !best.converged) {
      throw Error("fault", "locate",
                  "did not converge; raise iterations, or pass strict false to flag it");
    }
    return best;
  }

  /**
   * @brief The slip at points near the fault: each located (`Locate`) and
   * its nearest patch's slip read there.  A point farther than
   * `max_distance` is refused, all of them with it; a point within it but
   * beyond an edge reads the edge.
   * @param points The points, Cartesian.
   * @param max_distance The farthest a point may lie from the fault, in the
   *        length unit.
   * @param options The date, its extrapolation and tolerance, the basis
   *        and `Locate`'s tolerance.
   * @throws planetmodel::Error for a point beyond `max_distance`, a search
   *         that did not converge, or a date refused.
   */
  [[nodiscard]] std::vector<Vector3> SlipAt(std::span<const Vector3> points, double max_distance,
                                            const SlipAtOptions& options = {}) const {
    if (!(max_distance >= 0.0) || !std::isfinite(max_distance)) {
      throw Error("fault", "slip_at", "max_distance must be a finite number, not negative");
    }
    auto where = std::vector<FaultLocation>{};
    auto far = std::size_t{0};
    auto farthest = 0.0;
    for (const auto& p : points) {
      where.push_back(Locate(p, LocateOptions{.rtol = options.rtol}));
      const auto d = std::abs(where.back().distance);
      if (d > max_distance) ++far;
      farthest = std::max(farthest, d);
    }
    if (far > 0) {
      throw Error("fault", "slip_at",
                  std::format("{} of {} points lie farther than max_distance = {:g} from the "
                              "fault, the farthest at {:g}",
                              far, points.size(), max_distance, farthest));
    }
    auto out = std::vector<Vector3>{};
    const auto reading =
        DateOptions{.extrapolation = options.extrapolation, .time_rtol = options.time_rtol};
    for (const auto& w : where) {
      out.push_back(SlipVector(w.patch, w.u, w.v, options.date, options.basis, reading));
    }
    return out;
  }

 private:
  [[nodiscard]] const FaultPatch& PatchAt(std::size_t patch) const {
    if (patch >= patches_.size()) {
      throw Error("fault", "patch", std::format("{} of a fault of {}", patch, patches_.size()));
    }
    return patches_[patch];
  }

  /// A patch's points, with what each stands for but its slip, added to a
  /// fault's quadrature, each named as on patch `k`.
  template <class Q>
  static void Append(Q& out, const detail::PatchGeometry& g, std::size_t k) {
    out.points.insert(out.points.end(), g.points.begin(), g.points.end());
    out.positions.insert(out.positions.end(), g.positions.begin(), g.positions.end());
    out.normals.insert(out.normals.end(), g.normals.begin(), g.normals.end());
    out.weights.insert(out.weights.end(), g.weights.begin(), g.weights.end());
    out.u.insert(out.u.end(), g.u.begin(), g.u.end());
    out.v.insert(out.v.end(), g.v.begin(), g.v.end());
    out.patch.insert(out.patch.end(), g.weights.size(), k);
  }

  /// Refuse counts that are not one a patch.
  void CheckCounts(std::span<const std::size_t> n_u, std::span<const std::size_t> n_v) const {
    if (n_u.size() != patches_.size() || n_v.size() != patches_.size()) {
      throw Error("fault", "quadrature",
                  std::format("gives {} and {} counts for {} patches", n_u.size(), n_v.size(),
                              patches_.size()));
    }
  }

  std::vector<FaultPatch> patches_;
  TimeSupport support_;
  TimeRule rule_;
  std::optional<std::vector<double>> time_;
  Scales scales_;
};

}  // namespace planetmodel

#endif  // PLANETMODEL_FAULT_HPP
