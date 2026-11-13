// The fault file: each fixture read and held to the Python's answers at
// stated (u, v) and dates, its quadratures, and the location of physical
// points about it; the malformed copies refused at the Python's places;
// and the fault built in code evaluated as the file's is.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/fault.hpp"
#include "planetmodel/faultfile.hpp"

namespace {

namespace pm = planetmodel;

/// The numbers left on a line.
std::vector<double> Numbers(std::istringstream& words) {
  auto out = std::vector<double>{};
  for (auto x = 0.0; words >> x;) out.push_back(x);
  return out;
}

std::vector<double> Flat(const pm::Vector3& v) { return {v[0], v[1], v[2]}; }

std::vector<double> Flat(const pm::Matrix3& m) {
  return {m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]};
}

std::optional<double> DateOf(const std::string& tag) {
  return tag == "-" ? std::nullopt : std::optional<double>{std::stod(tag)};
}

pm::SlipBasis BasisOf(const std::string& name) {
  if (name == "fault") return pm::SlipBasis::Fault;
  if (name == "cartesian") return pm::SlipBasis::Cartesian;
  return pm::SlipBasis::Ambient;
}

pm::Extrapolation ExtrapolationOf(const std::string& name) {
  return name == "hold" ? pm::Extrapolation::Hold : pm::Extrapolation::Refuse;
}

/// The largest disagreement seen of each kind, relative to a scale.
struct Agreement {
  std::map<std::string, double> worst{};

  /// Hold `got` to `want` within `tol` of `scale`, and record it.
  void Check(const std::string& what, const std::vector<double>& got,
             const std::vector<double>& want, double scale, double tol, const std::string& line) {
    ASSERT_EQ(got.size(), want.size()) << line;
    auto err = 0.0;
    for (auto i = std::size_t{0}; i < got.size(); ++i)
      err = std::max(err, std::abs(got[i] - want[i]));
    err /= scale;
    worst[what] = std::max(worst[what], err);
    EXPECT_LE(err, tol) << what << ": " << line;
  }

  void Report(const std::string& name) const {
    std::cout << "[ agreement ] " << name << ":";
    for (const auto& [what, err] : worst) std::cout << " " << what << " " << err << ";";
    std::cout << "\n";
  }
};

double Scale(const std::vector<double>& v, double floor = 1.0) {
  auto out = floor;
  for (const auto x : v) out = std::max(out, std::abs(x));
  return out;
}

void HoldToExpectations(const std::string& name) {
  PLANETMODEL_FIXTURE(path, name + ".nc");
  PLANETMODEL_FIXTURE(expect, name + ".expect");
  const auto file = pm::ReadFaultFile(path);
  const auto& fault = file.fault;
  EXPECT_EQ(file.schema, pm::FAULT_SCHEMA);
  EXPECT_EQ(file.source, "planetmodel fixtures");
  EXPECT_FALSE(file.created.empty());
  auto in = std::ifstream(expect);
  ASSERT_TRUE(in) << "cannot open " << expect;
  auto agreement = Agreement{};
  // the length scale of each patch: its largest coordinate
  auto size = std::vector<double>{};
  for (const auto& p : fault.Patches()) {
    size.push_back(Scale(Flat(p.Cartesian(0.5, 0.5))));
  }
  auto slip_scale = 1.0;
  for (const auto& p : fault.Patches()) slip_scale = std::max(slip_scale, Scale(p.Slip().Values()));
  auto checked = 0;
  auto quadrature = std::optional<pm::FaultQuadrature>{};
  for (auto line = std::string{}; std::getline(in, line);) {
    auto words = std::istringstream(line);
    auto what = std::string{};
    words >> what;
    if (what == "fault") {
      auto frame = std::string{}, support = std::string{}, rule = std::string{};
      auto reference = 0.0;
      auto patches = std::size_t{0}, slices = std::size_t{0};
      words >> frame >> reference >> support >> rule >> patches >> slices;
      EXPECT_EQ(fault.Frame().Name(), frame) << line;
      EXPECT_EQ(fault.Frame().Reference(), reference) << line;
      EXPECT_EQ(fault.Support(), support) << line;
      EXPECT_EQ(fault.TimeInterpolation(), rule) << line;
      EXPECT_EQ(fault.Patches().size(), patches) << line;
      EXPECT_EQ(fault.NSlices(), slices) << line;
    } else if (what == "scales") {
      const auto s = Numbers(words);
      EXPECT_EQ(fault.GetScales().Length(), s[0]) << line;
      EXPECT_EQ(fault.GetScales().Mass(), s[1]) << line;
      EXPECT_EQ(fault.GetScales().Time(), s[2]) << line;
      EXPECT_EQ(fault.GetScales().Temperature(), s[3]) << line;
    } else if (what == "time") {
      auto k = std::size_t{0};
      auto t = 0.0;
      words >> k >> t;
      ASSERT_TRUE(fault.Time()) << line;
      EXPECT_EQ(fault.Time()->at(k), t) << line;
    } else if (what == "patch") {
      auto k = std::size_t{0};
      auto kind = std::string{}, tag = std::string{};
      words >> k >> kind >> tag;
      EXPECT_EQ(fault.Patches().at(k).KindName(), kind) << line;
      if (tag == "-") {
        EXPECT_FALSE(file.names.at(k)) << line;
      } else {
        auto hex = std::string{};
        for (const auto c : file.names.at(k).value_or("")) {
          hex += std::format("{:02x}", static_cast<unsigned char>(c));
        }
        EXPECT_EQ(hex, tag) << line;
      }
    } else if (what == "point" || what == "position" || what == "tangents" || what == "second" ||
               what == "normal" || what == "area" || what == "basis") {
      auto k = std::size_t{0};
      auto u = 0.0, v = 0.0;
      words >> k >> u >> v;
      const auto want = Numbers(words);
      const auto& p = fault.Patches().at(k);
      auto got = std::vector<double>{};
      auto scale = 1.0, tol = 1e-14;
      if (what == "point") {
        got = Flat(p.Cartesian(u, v));
        scale = size[k];
      } else if (what == "position") {
        got = Flat(p.Position(u, v));
        scale = Scale(want);
        // a longitude at the seam may read 0 or 360
        if (std::abs(got[1] - want[1]) > 180.0) got[1] += got[1] < want[1] ? 360.0 : -360.0;
      } else if (what == "tangents") {
        const auto [xu, xv] = p.Tangents(u, v);
        got = Flat(xu);
        const auto b = Flat(xv);
        got.insert(got.end(), b.begin(), b.end());
        scale = Scale(want, 0.0);
        tol = 1e-12;
      } else if (what == "second") {
        const auto [xuu, xuv, xvv] = p.SecondDerivatives(u, v);
        for (const auto& d : {xuu, xuv, xvv}) {
          const auto b = Flat(d);
          got.insert(got.end(), b.begin(), b.end());
        }
        scale = std::max(Scale(want, 0.0), 1e-300);
        tol = 1e-10;
      } else if (what == "normal") {
        got = Flat(p.Normal(u, v));
        tol = 1e-12;
      } else if (what == "area") {
        got = {p.AreaElement(u, v)};
        scale = want[0];
        tol = 1e-12;
      } else {
        got = Flat(p.FaultBasis(u, v));
        tol = 1e-12;
      }
      agreement.Check(what, got, want, scale, tol, line);
      ++checked;
    } else if (what == "slip") {
      auto k = std::size_t{0};
      auto u = 0.0, v = 0.0;
      auto date = std::string{}, extrapolation = std::string{}, basis = std::string{};
      words >> k >> u >> v >> date >> extrapolation >> basis;
      auto rest = std::string{};
      if (line.ends_with(" refused")) {
        EXPECT_THROW(static_cast<void>(fault.SlipVector(
                         k, u, v, DateOf(date), ExtrapolationOf(extrapolation), BasisOf(basis))),
                     pm::Error)
            << line;
        continue;
      }
      const auto want = Numbers(words);
      const auto got = Flat(
          fault.SlipVector(k, u, v, DateOf(date), ExtrapolationOf(extrapolation), BasisOf(basis)));
      agreement.Check("slip", got, want, slip_scale, 1e-12, line);
      ++checked;
    } else if (what == "quadrature") {
      auto points = std::size_t{0}, total = std::size_t{0};
      auto date = std::string{}, extrapolation = std::string{}, nu = std::string{},
           nv = std::string{};
      words >> points >> date >> extrapolation >> nu >> nv >> total;
      auto counts = [](const std::string& text) {
        auto out = std::vector<std::size_t>{};
        auto parts = std::istringstream(text);
        for (auto part = std::string{}; std::getline(parts, part, ',');) {
          out.push_back(std::stoul(part));
        }
        return out;
      };
      quadrature = fault.Quadrature(counts(nu), counts(nv), points, DateOf(date),
                                    ExtrapolationOf(extrapolation));
      EXPECT_EQ(quadrature->Size(), total) << line;
    } else if (what == "qpoint") {
      ASSERT_TRUE(quadrature) << line;
      auto i = std::size_t{0}, patch = std::size_t{0};
      words >> i >> patch;
      const auto want = Numbers(words);
      ASSERT_LT(i, quadrature->Size()) << line;
      ASSERT_EQ(want.size(), 15u) << line;
      const auto& q = *quadrature;
      EXPECT_EQ(q.patch[i], patch) << line;
      agreement.Check("quadrature u v", {q.u[i], q.v[i]}, {want[0], want[1]}, 1.0, 1e-15, line);
      agreement.Check("quadrature weight", {q.weights[i]}, {want[2]}, want[2], 1e-12, line);
      agreement.Check("quadrature point", Flat(q.points[i]), {want[3], want[4], want[5]},
                      size[patch], 1e-14, line);
      auto position = Flat(q.positions[i]);
      if (std::abs(position[1] - want[7]) > 180.0) {
        position[1] += position[1] < want[7] ? 360.0 : -360.0;
      }
      agreement.Check("quadrature position", position, {want[6], want[7], want[8]},
                      Scale({want[6], want[7], want[8]}), 1e-14, line);
      agreement.Check("quadrature normal", Flat(q.normals[i]), {want[9], want[10], want[11]}, 1.0,
                      1e-12, line);
      agreement.Check("quadrature slip", Flat(q.slip[i]), {want[12], want[13], want[14]},
                      slip_scale, 1e-12, line);
      ++checked;
    } else if (what == "locate") {
      const auto n = Numbers(words);
      ASSERT_EQ(n.size(), 9u) << line;
      const auto found = fault.Locate({n[0], n[1], n[2]});
      EXPECT_EQ(found.patch, static_cast<std::size_t>(n[3])) << line;
      agreement.Check("locate u v", {found.u, found.v}, {n[4], n[5]}, 1.0, 1e-9, line);
      agreement.Check("locate distance", {found.distance}, {n[6]}, size[found.patch], 1e-11, line);
      EXPECT_EQ(found.inside, n[7] != 0.0) << line;
      EXPECT_EQ(found.converged, n[8] != 0.0) << line;
      if (n[4] == -0.5 || n[4] == 0.5 || n[5] == -0.5 || n[5] == 0.5) {
        EXPECT_EQ(found.OnEdge(), true) << line;
        EXPECT_EQ(found.OnULow(), n[4] == -0.5) << line;
        EXPECT_EQ(found.OnUHigh(), n[4] == 0.5) << line;
        EXPECT_EQ(found.OnVLow(), n[5] == -0.5) << line;
        EXPECT_EQ(found.OnVHigh(), n[5] == 0.5) << line;
      }
      ++checked;
    } else if (what == "slip_at") {
      auto limit = 0.0;
      auto date = std::string{}, basis = std::string{};
      words >> limit >> date >> basis;
      const auto n = Numbers(words);
      const auto points = std::vector<pm::Vector3>{{n[0], n[1], n[2]}};
      const auto got = fault.SlipAt(
          points, limit, pm::SlipAtOptions{.date = DateOf(date), .basis = BasisOf(basis)});
      agreement.Check("slip_at", Flat(got[0]), {n[3], n[4], n[5]}, slip_scale, 1e-9, line);
      ++checked;
    } else if (what == "slip_at_refused") {
      auto limit = 0.0;
      words >> limit;
      const auto n = Numbers(words);
      const auto points = std::vector<pm::Vector3>{{n[0], n[1], n[2]}};
      try {
        static_cast<void>(fault.SlipAt(
            points, limit,
            pm::SlipAtOptions{.date = fault.Time() ? std::optional<double>{fault.Time()->front()}
                                                   : std::nullopt}));
        ADD_FAILURE() << line << ": accepted";
      } catch (const pm::Error& e) {
        EXPECT_NE(std::string{e.what()}.find("farther than max_distance"), std::string::npos)
            << e.what();
      }
    }
  }
  EXPECT_GT(checked, 100);
  agreement.Report(name);
}

TEST(FaultFile, TheStaticPlaneIsThePythons) { HoldToExpectations("fault_static"); }

TEST(FaultFile, TheAfterslipIsThePythons) { HoldToExpectations("fault_afterslip"); }

TEST(FaultFile, TheCartesianFaultIsThePythons) { HoldToExpectations("fault_cartesian"); }

TEST(FaultFile, RefusesWhatThePythonRefusesAtItsPlace) {
  PLANETMODEL_FIXTURE(index, "fault_cases/bad.txt");
  auto in = std::ifstream(index);
  auto cases = 0;
  for (auto line = std::string{}; std::getline(in, line);) {
    const auto tab1 = line.find('\t');
    const auto tab2 = line.find('\t', tab1 + 1);
    const auto label = line.substr(0, tab1);
    const auto where = line.substr(tab1 + 1, tab2 - tab1 - 1);
    const auto phrase = line.substr(tab2 + 1);
    const auto path = index.parent_path() / ("bad_" + label + ".nc");
    try {
      static_cast<void>(pm::ReadFaultFile(path));
      ADD_FAILURE() << label << " was accepted";
    } catch (const pm::Error& e) {
      EXPECT_EQ(e.Where(), where) << label << ": " << e.what();
      EXPECT_NE(std::string{e.what()}.find(phrase), std::string::npos) << label << ": " << e.what();
    }
    ++cases;
  }
  EXPECT_GT(cases, 30);
  EXPECT_THROW(static_cast<void>(pm::ReadFaultFile("/nonexistent.nc")), pm::Error);
}

TEST(Fault, APlaneBuiltInCodeIsItsClosedForm) {
  // a thrust on an east-striking plane dipping 30 degrees south, in a
  // Cartesian frame: the hanging wall moves north and up
  const auto frame = pm::FaultFrame::Cartesian();
  const auto slip = pm::FaultSlip({0.0, 2.0, 0.0}, 1, 1, 1);
  const auto plane = pm::FaultPatch::Plane(
      frame,
      {.centroid = {0.0, 0.0, -10.0}, .strike = 90.0, .dip = 30.0, .length = 8.0, .width = 4.0},
      slip);
  const auto h = std::sqrt(3.0) / 2.0;
  const auto s = plane.SlipVector(0.3, -0.2, 0, pm::SlipBasis::Cartesian);
  EXPECT_NEAR(s[0], 0.0, 1e-15);
  EXPECT_NEAR(s[1], 2.0 * h, 1e-15);
  EXPECT_NEAR(s[2], 1.0, 1e-15);
  const auto n = plane.Normal(0.0, 0.0);
  EXPECT_NEAR(n[1], -0.5, 1e-15);
  EXPECT_NEAR(n[2], h, 1e-15);
  EXPECT_NEAR(plane.AreaElement(0.1, 0.4), 32.0, 1e-13);
  const auto q = plane.Quadrature(3, 2, 2);
  auto area = 0.0;
  for (const auto w : q.weights) area += w;
  EXPECT_NEAR(area, 32.0, 1e-13);
  // a point beyond the corner u = v = 1/2 comes back to it, outside
  const auto x = plane.Cartesian(0.5, 0.5);
  const auto [xu, xv] = plane.Tangents(0.5, 0.5);
  const auto beyond =
      pm::Vector3{x[0] + 0.1 * xu[0] + 0.1 * xv[0], x[1] + 0.1 * xu[1] + 0.1 * xv[1],
                  x[2] + 0.1 * xu[2] + 0.1 * xv[2]};
  const auto found = plane.Locate(beyond);
  EXPECT_EQ(found.u, 0.5);
  EXPECT_EQ(found.v, 0.5);
  EXPECT_FALSE(found.inside);
  EXPECT_TRUE(found.OnUHigh() && found.OnVHigh());
  EXPECT_THROW(static_cast<void>(plane.Cartesian(0.6, 0.0)), pm::Error);
  EXPECT_THROW(static_cast<void>(plane.SlipVector(0.0, 0.0, 1)), pm::Error);
  EXPECT_THROW(static_cast<void>(pm::FaultSlip({0.0, 0.0}, 1, 1, 1)), pm::Error);
  EXPECT_THROW(static_cast<void>(pm::FaultSlip(std::vector<double>(9, 0.0), 1, 1, 3, "cubic")),
               pm::Error);
  EXPECT_THROW(static_cast<void>(pm::Fault({plane}, "node", "linear")), pm::Error);
  EXPECT_THROW(static_cast<void>(pm::Fault({plane}, "interval", "none", std::vector{0.0, 1.0})),
               pm::Error);
}

TEST(Fault, TheGaussLegendreRuleIntegratesItsPolynomials) {
  for (auto n = std::size_t{1}; n <= 8; ++n) {
    const auto [x, w] = pm::GaussLegendrePoints(n);
    for (auto degree = 0; degree < static_cast<int>(2 * n); ++degree) {
      auto sum = 0.0;
      for (auto i = std::size_t{0}; i < n; ++i) sum += w[i] * std::pow(x[i], degree);
      const auto exact = degree % 2 == 1 ? 0.0 : 2.0 / (degree + 1);
      EXPECT_NEAR(sum, exact, 2e-15) << n << " " << degree;
    }
  }
}

}  // namespace
