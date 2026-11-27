// The fault file: each fixture read and held to the Python's answers at
// stated (u, v) and dates, its quadratures, and the location of physical
// points about it; the malformed copies refused at the Python's places;
// and the fault built in code evaluated as the file's is.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "fixtures.hpp"
#include "planetmodel/fault.hpp"
#include "planetmodel/faultfile.hpp"
#include "planetmodel/version.hpp"

#if !defined(PLANETMODEL_HAS_FAULT_SLICE_QUADRATURE) || !defined(PLANETMODEL_HAS_FAULT_FILE_WRITER)
#error "fault.hpp and faultfile.hpp define the macros of their features"
#endif

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

/// Whether a quadrature says which patch each point is on.
template <class Q>
concept NamesPatches = requires(Q q) { q.patch; };

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
  auto slices = std::optional<pm::FaultSliceQuadrature>{};
  auto counts = [](const std::string& text) {
    auto out = std::vector<std::size_t>{};
    auto parts = std::istringstream(text);
    for (auto part = std::string{}; std::getline(parts, part, ',');) {
      out.push_back(std::stoul(part));
    }
    return out;
  };
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
      EXPECT_EQ(fault.SupportName(), support) << line;
      EXPECT_EQ(fault.TimeInterpolationName(), rule) << line;
      EXPECT_EQ(fault.Support(), pm::TimeSupportNamed(support)) << line;
      EXPECT_EQ(fault.TimeInterpolation(), pm::TimeRuleNamed(rule)) << line;
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
                         k, u, v, DateOf(date), BasisOf(basis),
                         {.extrapolation = ExtrapolationOf(extrapolation)})),
                     pm::Error)
            << line;
        continue;
      }
      const auto want = Numbers(words);
      const auto got = Flat(fault.SlipVector(k, u, v, DateOf(date), BasisOf(basis),
                                             {.extrapolation = ExtrapolationOf(extrapolation)}));
      agreement.Check("slip", got, want, slip_scale, 1e-12, line);
      ++checked;
    } else if (what == "quadrature") {
      auto points = std::size_t{0}, total = std::size_t{0};
      auto date = std::string{}, extrapolation = std::string{}, nu = std::string{},
           nv = std::string{};
      words >> points >> date >> extrapolation >> nu >> nv >> total;
      quadrature = fault.Quadrature(counts(nu), counts(nv), points, DateOf(date),
                                    {.extrapolation = ExtrapolationOf(extrapolation)});
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
    } else if (what == "slices") {
      auto points = std::size_t{0}, total = std::size_t{0}, nslices = std::size_t{0};
      auto nu = std::string{}, nv = std::string{};
      words >> points >> nu >> nv >> total >> nslices;
      slices = fault.SliceQuadrature(counts(nu), counts(nv), points);
      EXPECT_EQ(slices->Size(), total) << line;
      EXPECT_EQ(slices->NSlices(), nslices) << line;
      // the slices under a date's rows are the quadrature at that date: at
      // each node and, between them, at a third of the way across each
      // interval and beyond the span held
      auto dates = std::vector<std::optional<double>>{std::nullopt};
      auto hold = pm::Extrapolation::Refuse;
      if (fault.Time()) {
        const auto& t = *fault.Time();
        dates.assign(t.begin(), t.end());
        if (fault.TimeInterpolation() != pm::TimeRule::None) {
          for (auto k = std::size_t{0}; k + 1 < t.size(); ++k) {
            dates.emplace_back((2.0 * t[k] + t[k + 1]) / 3.0);
          }
          dates.emplace_back(t.back() + 1.0);
          hold = pm::Extrapolation::Hold;
        }
      }
      for (const auto& date : dates) {
        const auto at = fault.Quadrature(counts(nu), counts(nv), points, date,
                                         {.extrapolation = hold});
        const auto slip = slices->SlipOf(fault.Rows(date, {.extrapolation = hold}));
        ASSERT_EQ(at.Size(), slices->Size()) << line;
        for (auto i = std::size_t{0}; i < at.Size(); ++i) {
          EXPECT_EQ(slip[i], at.slip[i]) << line << " point " << i;
          EXPECT_EQ(slices->points[i], at.points[i]) << line << " point " << i;
          EXPECT_EQ(slices->weights[i], at.weights[i]) << line << " point " << i;
          EXPECT_EQ(slices->patch[i], at.patch[i]) << line << " point " << i;
        }
      }
    } else if (what == "spoint") {
      ASSERT_TRUE(slices) << line;
      auto i = std::size_t{0}, patch = std::size_t{0};
      words >> i >> patch;
      const auto want = Numbers(words);
      const auto& q = *slices;
      ASSERT_LT(i, q.Size()) << line;
      ASSERT_EQ(want.size(), 12 + 3 * q.NSlices()) << line;
      EXPECT_EQ(q.patch[i], patch) << line;
      agreement.Check("slices u v", {q.u[i], q.v[i]}, {want[0], want[1]}, 1.0, 1e-15, line);
      agreement.Check("slices weight", {q.weights[i]}, {want[2]}, want[2], 1e-12, line);
      agreement.Check("slices point", Flat(q.points[i]), {want[3], want[4], want[5]}, size[patch],
                      1e-14, line);
      agreement.Check("slices normal", Flat(q.normals[i]), {want[9], want[10], want[11]}, 1.0,
                      1e-12, line);
      for (auto s = std::size_t{0}; s < q.NSlices(); ++s) {
        agreement.Check("slices slip", Flat(q.slip[s][i]),
                        {want[12 + 3 * s], want[13 + 3 * s], want[14 + 3 * s]}, slip_scale, 1e-12,
                        line);
      }
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

TEST(FaultFile, TheQuadraturesAreThePatchsOwnEvaluationsPointByPoint) {
  // the geometry and the fault basis formed once a point give, to the bit,
  // what a patch's evaluations give point by point: the point, the normal
  // and the weight from the tangents there, and each slice's Cartesian
  // slip as SlipVector reads it, under the weights of a date's rows
  for (const auto* name : {"fault_static", "fault_afterslip", "fault_cartesian"}) {
    SCOPED_TRACE(name);
    PLANETMODEL_FIXTURE(path, std::string{name} + ".nc");
    const auto fault = pm::ReadFaultFile(path).fault;
    const auto n = fault.Patches().size();
    const auto nu = std::vector<std::size_t>(n, 3), nv = std::vector<std::size_t>(n, 2);
    const auto rule = pm::ParameterRule(3, 2, 2);
    const auto slices = fault.SliceQuadrature(nu, nv, 2);
    auto dates = std::vector<std::optional<double>>{std::nullopt};
    if (fault.Time()) {
      const auto t0 = fault.Time()->front(), t1 = fault.Time()->back();
      dates = {t0, (2.0 * t0 + t1) / 3.0, t1};
    }
    for (const auto& date : dates) {
      const auto [first, second] = fault.Rows(date);
      const auto q = fault.Quadrature(nu, nv, 2, date);
      ASSERT_EQ(q.Size(), slices.Size());
      ASSERT_EQ(q.Size(), n * rule.w.size());
      for (auto i = std::size_t{0}; i < q.Size(); ++i) {
        const auto& p = fault.Patches().at(q.patch[i]);
        const auto u = q.u[i], v = q.v[i];
        EXPECT_EQ(u, rule.u[i % rule.u.size()]);
        EXPECT_EQ(v, rule.v[i % rule.v.size()]);
        const auto [xu, xv] = p.Tangents(u, v);
        const auto c = pm::Vector3{xv[1] * xu[2] - xv[2] * xu[1], xv[2] * xu[0] - xv[0] * xu[2],
                                   xv[0] * xu[1] - xv[1] * xu[0]};
        const auto area = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
        EXPECT_EQ(q.points[i], p.Cartesian(u, v));
        EXPECT_EQ(q.positions[i], p.Position(u, v));
        EXPECT_EQ(q.normals[i], (pm::Vector3{c[0] / area, c[1] / area, c[2] / area}));
        EXPECT_EQ(q.weights[i], area * rule.w[i % rule.w.size()]);
        auto want = p.SlipVector(u, v, first.first, pm::SlipBasis::Cartesian);
        for (auto& x : want) x *= first.second;
        if (second.second != 0.0) {
          const auto more = p.SlipVector(u, v, second.first, pm::SlipBasis::Cartesian);
          for (auto d = 0; d < 3; ++d) want[d] = want[d] + second.second * more[d];
        }
        EXPECT_EQ(q.slip[i], want) << i;
        EXPECT_EQ(slices.points[i], q.points[i]);
        EXPECT_EQ(slices.positions[i], q.positions[i]);
        EXPECT_EQ(slices.normals[i], q.normals[i]);
        EXPECT_EQ(slices.weights[i], q.weights[i]);
        for (auto k = std::size_t{0}; k < fault.NSlices(); ++k) {
          EXPECT_EQ(slices.slip[k][i], p.SlipVector(u, v, k, pm::SlipBasis::Cartesian)) << i;
        }
      }
      EXPECT_EQ(slices.SlipOf({first, second}), q.slip);
      // a patch's own quadrature is the same points with a slice's slip
      const auto own = fault.Patches().front().Quadrature(3, 2, 2, fault.NSlices() - 1);
      for (auto i = std::size_t{0}; i < own.Size(); ++i) {
        EXPECT_EQ(own.points[i], q.points[i]);
        EXPECT_EQ(own.normals[i], q.normals[i]);
        EXPECT_EQ(own.weights[i], q.weights[i]);
        EXPECT_EQ(own.slip[i], slices.slip[fault.NSlices() - 1][i]);
      }
    }
  }
}

TEST(Fault, ASlipFromARakeIsThePythons) {
  PLANETMODEL_FIXTURE(path, "slip_rake.txt");
  auto in = std::ifstream(path);
  auto cases = 0;
  auto nslices = std::size_t{0}, n_u = std::size_t{0}, n_v = std::size_t{0};
  auto rule = std::string{};
  auto given = std::map<std::string, std::vector<double>>{};
  for (auto line = std::string{}; std::getline(in, line);) {
    auto words = std::istringstream(line);
    auto what = std::string{};
    words >> what;
    if (what == "case") {
      words >> nslices >> n_u >> n_v >> rule;
      given.clear();
      continue;
    }
    given[what] = Numbers(words);
    if (what != "values") continue;
    const auto slip = pm::FaultSlip::FromRake(given["rake"], given["magnitude"], nslices, n_u, n_v,
                                              {.opening = given["opening"], .rule = rule});
    EXPECT_EQ(slip.NSlices(), nslices) << line;
    EXPECT_EQ(slip.NU(), n_u) << line;
    EXPECT_EQ(slip.Rule(), rule) << line;
    const auto& want = given["values"];
    ASSERT_EQ(slip.Values().size(), want.size()) << line;
    auto scale = 1.0;
    for (const auto x : want) scale = std::max(scale, std::abs(x));
    for (auto i = std::size_t{0}; i < want.size(); ++i) {
      EXPECT_LE(std::abs(slip.Values()[i] - want[i]), 4.0 * 2.220446049250313e-16 * scale)
          << "case " << cases << " value " << i;
    }
    ++cases;
  }
  EXPECT_EQ(cases, 4);
  // in code: one cell by default, an opening and a rule by name
  const auto thrust = pm::FaultSlip::FromRake({90.0}, {2.0});
  EXPECT_NEAR(thrust.Values()[0], 0.0, 1e-15);
  EXPECT_EQ(thrust.Values()[1], 2.0);
  EXPECT_EQ(thrust.Values()[2], 0.0);
  EXPECT_EQ(pm::FaultSlip::FromRake({0.0}, {2.0}, 1, 1, 1, {.opening = {0.5}}).Values(),
            (std::vector<double>{2.0, 0.0, 0.5}));
  const auto grid = pm::FaultSlip::FromRake({0.0, 180.0, 0.0, 180.0}, {1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
                                                                     4.0, 4.0},
                                            2, 2, 2, {.rule = "linear"});
  EXPECT_EQ(grid.Values()[3 * 7], -4.0);  // slice 1, point (1, 1): the rake of point 3
  EXPECT_EQ(grid.Rule(), "linear");
  try {
    static_cast<void>(pm::FaultSlip::FromRake({0.0, 1.0}, {1.0}, 1, 2, 2));
    ADD_FAILURE() << "two rakes on four points were accepted";
  } catch (const pm::Error& e) {
    EXPECT_EQ(e.Where(), "rake") << e.what();
  }
  EXPECT_THROW(static_cast<void>(pm::FaultSlip::FromRake({0.0}, {1.0}, 1, 1, 1, {.rule = "cubic"})),
               pm::Error);
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
  EXPECT_THROW(static_cast<void>(pm::Fault({plane}, pm::TimeSupport::Node, pm::TimeRule::Linear)),
               pm::Error);
  EXPECT_THROW(static_cast<void>(pm::Fault({plane}, pm::TimeSupport::Interval, pm::TimeRule::None,
                                           std::vector{0.0, 1.0})),
               pm::Error);
}

TEST(Fault, ItsSupportAndRuleAreEnumerationsNamedAsTheFileNamesThem) {
  const auto frame = pm::FaultFrame::Cartesian();
  const auto plane = [&frame](std::size_t slices) {
    return pm::FaultPatch::Plane(
        frame,
        {.centroid = {0.0, 0.0, -10.0}, .strike = 90.0, .dip = 30.0, .length = 8.0, .width = 4.0},
        pm::FaultSlip(std::vector<double>(3 * slices, 1.0), slices, 1, 1));
  };
  const auto still = pm::Fault({plane(1)});
  EXPECT_EQ(still.Support(), pm::TimeSupport::Static);
  EXPECT_EQ(still.SupportName(), "static");
  EXPECT_EQ(still.TimeInterpolation(), pm::TimeRule::None);
  EXPECT_EQ(still.TimeInterpolationName(), "none");
  EXPECT_EQ(still.Rows(std::nullopt), (pm::TimeRows{{0, 1.0}, {0, 0.0}}));
  // a static fault takes no date
  try {
    static_cast<void>(still.Rows(1.0));
    ADD_FAILURE() << "a date was read by a static fault";
  } catch (const pm::Error& e) {
    EXPECT_NE(std::string{e.what()}.find("static, the same at every date"), std::string::npos)
        << e.what();
  }
  EXPECT_THROW(static_cast<void>(still.SlipVector(0, 0.0, 0.0, 1.0)), pm::Error);
  EXPECT_THROW(static_cast<void>(still.Quadrature(1, 1, 1, 1.0)), pm::Error);
  const auto timed =
      pm::Fault({plane(2)}, pm::TimeSupport::Node, pm::TimeRule::Previous, std::vector{0.0, 2.0});
  EXPECT_EQ(timed.SupportName(), "node");
  EXPECT_EQ(timed.TimeInterpolationName(), "previous");
  EXPECT_EQ(timed.Rows(2.0 - 1e-12), (pm::TimeRows{{1, 1.0}, {1, 0.0}}));
  EXPECT_EQ(timed.Rows(2.0 - 1e-12, {.time_rtol = 0.0}), (pm::TimeRows{{0, 1.0}, {0, 0.0}}));
  EXPECT_EQ(timed.Rows(2.5, {.extrapolation = pm::Extrapolation::Hold}),
            (pm::TimeRows{{1, 1.0}, {1, 0.0}}));
  EXPECT_THROW(static_cast<void>(timed.Rows(2.5)), pm::Error);
  for (const auto support : {pm::TimeSupport::Interval, pm::TimeSupport::Instant}) {
    EXPECT_THROW(static_cast<void>(pm::Fault({plane(1)}, support)), pm::Error);
  }
  EXPECT_THROW(
      static_cast<void>(pm::Fault({plane(1)}, pm::TimeSupport::Static, pm::TimeRule::Linear)),
      pm::Error);
  EXPECT_EQ(pm::FAULT_SUPPORTS, (std::array<std::string_view, 2>{"static", "node"}));
}

TEST(Fault, APatchsQuadratureNamesNoPatchAndAFaultsNamesEach) {
  static_assert(!NamesPatches<pm::PatchQuadrature>);
  static_assert(NamesPatches<pm::FaultQuadrature>);
  const auto frame = pm::FaultFrame::Cartesian();
  auto patches = std::vector<pm::FaultPatch>{};
  for (const auto x : {0.0, 20.0, 40.0}) {
    patches.push_back(pm::FaultPatch::Plane(
        frame,
        {.centroid = {x, 0.0, -10.0}, .strike = 0.0, .dip = 45.0, .length = 8.0, .width = 4.0},
        pm::FaultSlip({1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 4.0, 1.0, 0.0}, 3, 1, 1)));
  }
  const auto fault =
      pm::Fault(patches, pm::TimeSupport::Node, pm::TimeRule::Linear, std::vector{0.0, 1.0, 3.0});
  const auto q = fault.Quadrature(2, 1, 1, 2.0);
  const auto s = fault.SliceQuadrature(2, 1, 1);
  EXPECT_EQ(q.patch, (std::vector<std::size_t>{0, 0, 1, 1, 2, 2}));
  EXPECT_EQ(s.patch, q.patch);
  EXPECT_EQ(s.NSlices(), 3u);
  EXPECT_EQ(s.SlipOf(fault.Rows(2.0)), q.slip);
  EXPECT_THROW(static_cast<void>(s.SlipOf({{3, 1.0}, {0, 0.0}})), pm::Error);
  const auto spans = std::vector<std::size_t>{1, 2, 3};
  EXPECT_THROW(static_cast<void>(fault.SliceQuadrature(std::span{spans}.first(2), spans)),
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

namespace {

/// Two faults alike: frame, time, scales, every patch's map and slip as
/// stored, and their evaluations number for number.
void ExpectSameFault(const pm::Fault& a, const pm::Fault& b) {
  EXPECT_EQ(a.Frame(), b.Frame());
  EXPECT_EQ(a.Support(), b.Support());
  EXPECT_EQ(a.TimeInterpolation(), b.TimeInterpolation());
  EXPECT_EQ(a.Time(), b.Time());
  EXPECT_EQ(a.GetScales().Length(), b.GetScales().Length());
  EXPECT_EQ(a.GetScales().Mass(), b.GetScales().Mass());
  EXPECT_EQ(a.GetScales().Time(), b.GetScales().Time());
  EXPECT_EQ(a.GetScales().Temperature(), b.GetScales().Temperature());
  ASSERT_EQ(a.Patches().size(), b.Patches().size());
  for (auto k = std::size_t{0}; k < a.Patches().size(); ++k) {
    const auto& p = a.Patches()[k];
    const auto& q = b.Patches()[k];
    EXPECT_EQ(p.KindName(), q.KindName()) << k;
    if (p.GetKind() == pm::FaultPatch::Kind::Plane) {
      EXPECT_EQ(p.PlaneOf().centroid, q.PlaneOf().centroid) << k;
      EXPECT_EQ(p.PlaneOf().strike, q.PlaneOf().strike) << k;
      EXPECT_EQ(p.PlaneOf().dip, q.PlaneOf().dip) << k;
      EXPECT_EQ(p.PlaneOf().length, q.PlaneOf().length) << k;
      EXPECT_EQ(p.PlaneOf().width, q.PlaneOf().width) << k;
    } else {
      EXPECT_EQ(p.Grid().n_u, q.Grid().n_u) << k;
      EXPECT_EQ(p.Grid().n_v, q.Grid().n_v) << k;
      EXPECT_EQ(p.Grid().positions, q.Grid().positions) << k;
      EXPECT_EQ(p.Grid().interpolation, q.Grid().interpolation) << k;
    }
    EXPECT_EQ(p.Slip().Values(), q.Slip().Values()) << k;
    EXPECT_EQ(p.Slip().Rule(), q.Slip().Rule()) << k;
    EXPECT_EQ(p.Slip().NU(), q.Slip().NU()) << k;
  }
  auto dates = std::vector<std::optional<double>>{std::nullopt};
  if (a.Time()) {
    dates.clear();
    for (const auto t : *a.Time()) dates.emplace_back(t);
    if (a.TimeInterpolation() != pm::TimeRule::None) {
      dates.emplace_back(0.3 * a.Time()->front() + 0.7 * a.Time()->back());
    }
  }
  for (const auto& date : dates) {
    for (auto k = std::size_t{0}; k < a.Patches().size(); ++k) {
      for (const auto& [u, v] :
           {std::pair{0.0, 0.0}, std::pair{-0.5, 0.5}, std::pair{0.31, -0.17}}) {
        for (const auto basis :
             {pm::SlipBasis::Fault, pm::SlipBasis::Ambient, pm::SlipBasis::Cartesian}) {
          EXPECT_EQ(a.SlipVector(k, u, v, date, basis), b.SlipVector(k, u, v, date, basis));
        }
      }
    }
    const auto qa = a.Quadrature(2, 3, 2, date);
    const auto qb = b.Quadrature(2, 3, 2, date);
    EXPECT_EQ(qa.points, qb.points);
    EXPECT_EQ(qa.weights, qb.weights);
    EXPECT_EQ(qa.normals, qb.normals);
    EXPECT_EQ(qa.slip, qb.slip);
  }
}

/// A geographic fault on three nodes in kilometres and years, of a plane,
/// a bent bicubic patch and a bilinear patch across the date line, every
/// slip rule among them, read linearly between nodes.
pm::FaultFile BuiltNodeFault() {
  const auto km = pm::FaultFrame::Geographic(6371.0);
  auto values = [](std::size_t n, double seed) {
    auto out = std::vector<double>(n);
    for (auto i = std::size_t{0}; i < n; ++i) {
      out[i] = std::sin(seed * static_cast<double>(i + 1)) + 0.1 * static_cast<double>(i % 7);
    }
    return out;
  };
  const auto plane =
      pm::FaultPatch::Plane(km,
                            {.centroid = {35.0, 140.0, 6351.0},
                             .strike = 200.0,
                             .dip = 15.0,
                             .length = 100.0,
                             .width = 50.0},
                            pm::FaultSlip(values(3 * 2 * 3 * 3, 0.7), 3, 2, 3, "cell"));
  auto bent = pm::ControlGrid{.n_u = 5, .n_v = 4, .interpolation = "cubic"};
  for (auto i = std::size_t{0}; i < 5; ++i) {
    for (auto j = std::size_t{0}; j < 4; ++j) {
      const auto u = static_cast<double>(i) / 4.0 - 0.5, v = static_cast<double>(j) / 3.0 - 0.5;
      bent.positions.push_back({34.2 + 0.3 * u + 0.05 * v * v, 139.7 + 0.5 * u,
                                6341.0 - 30.0 * (v + 0.5) - 3.0 * u * u});
    }
  }
  const auto curved = pm::FaultPatch::Structured(
      km, bent, pm::FaultSlip(values(3 * 4 * 4 * 3, 1.3), 3, 4, 4, "cubic"));
  auto line = pm::ControlGrid{.n_u = 3, .n_v = 3, .interpolation = "linear"};
  for (auto i = std::size_t{0}; i < 3; ++i) {
    for (auto j = std::size_t{0}; j < 3; ++j) {
      const auto u = static_cast<double>(i) / 2.0 - 0.5, v = static_cast<double>(j) / 2.0 - 0.5;
      auto lon = 179.6 + 0.8 * (u + 0.5);
      if (lon > 180.0) lon -= 360.0;
      line.positions.push_back({-10.0 - 0.3 * (v + 0.5), lon, 6366.0 - 30.0 * (v + 0.5) - u * v});
    }
  }
  const auto crossing = pm::FaultPatch::Structured(
      km, line, pm::FaultSlip(values(3 * 2 * 2 * 3, 2.1), 3, 2, 2, "linear"));
  return pm::FaultFile{.fault = pm::Fault({plane, curved, crossing}, pm::TimeSupport::Node,
                                          pm::TimeRule::Linear, std::vector{0.0, 0.5, 2.0},
                                          pm::Scales({.length = 1e3, .time = 3.15576e7})),
                       .names = {"main", std::nullopt, "splay"},
                       .source = "planetmodel C++ tests",
                       .title = "a fault written by the C++",
                       .meta = {{"event", std::string{"test"}},
                                {"magnitude", std::vector<double>{7.5}},
                                {"count", std::vector<long long>{3}}},
                       .citation = {"A. Author (2026).  A fault.", "B. Author (2026).  Its slip."}};
}

/// A static fault in a Cartesian frame: a plane and a twisted bilinear
/// patch, its slip cells and nodes.
pm::FaultFile BuiltStaticFault() {
  const auto local = pm::FaultFrame::Cartesian(1.5);
  const auto plane = pm::FaultPatch::Plane(
      local,
      {.centroid = {1.0, 2.0, -3.0}, .strike = 250.0, .dip = 90.0, .length = 2.0, .width = 1.0},
      pm::FaultSlip({0.5, -0.25, 0.0, 1.0, 0.75, 0.125}, 1, 2, 1, "cell"));
  auto twisted = pm::ControlGrid{.n_u = 3, .n_v = 3, .interpolation = "linear"};
  for (auto i = std::size_t{0}; i < 3; ++i) {
    for (auto j = std::size_t{0}; j < 3; ++j) {
      const auto u = static_cast<double>(i) / 2.0 - 0.5, v = static_cast<double>(j) / 2.0 - 0.5;
      twisted.positions.push_back({3.0 * u, 2.0 * v, -4.0 + 1.6 * u * v - 0.4 * v});
    }
  }
  auto slip = std::vector<double>{};
  for (auto n = 0; n < 2 * 2 * 3; ++n) slip.push_back(0.25 * n - 1.0);
  const auto patch =
      pm::FaultPatch::Structured(local, twisted, pm::FaultSlip(slip, 1, 2, 2, "linear"));
  return pm::FaultFile{.fault = pm::Fault({plane, patch}), .source = "planetmodel C++ tests"};
}

/// The C++'s evaluations of a fault, for the Python to hold its reading
/// of the file to: `slip <k> <u> <v> <date|-> <basis> <x> <y> <z>` and
/// `qpoint <i> <patch> <u> <v> <weight> <point> <normal> <slip>` of the
/// quadrature at the last date.
void WriteEvaluations(const pm::Fault& fault, const std::filesystem::path& path) {
  auto out = std::ofstream(path);
  auto dates = std::vector<std::optional<double>>{std::nullopt};
  if (fault.Time()) dates = {fault.Time()->front(), 0.25, 1.2, fault.Time()->back()};
  const auto names = std::array<std::pair<pm::SlipBasis, const char*>, 3>{
      {{pm::SlipBasis::Fault, "fault"},
       {pm::SlipBasis::Ambient, "ambient"},
       {pm::SlipBasis::Cartesian, "cartesian"}}};
  for (const auto& date : dates) {
    for (auto k = std::size_t{0}; k < fault.Patches().size(); ++k) {
      for (const auto& [u, v] : {std::pair{0.0, 0.0}, std::pair{-0.5, 0.5}, std::pair{0.31, -0.17},
                                 std::pair{0.5, 0.2}}) {
        for (const auto& [basis, name] : names) {
          const auto s = fault.SlipVector(k, u, v, date, basis);
          out << std::format("slip {} {:.17g} {:.17g} {} {} {:.17g} {:.17g} {:.17g}\n", k, u, v,
                             date ? std::format("{:.17g}", *date) : std::string{"-"}, name, s[0],
                             s[1], s[2]);
        }
      }
    }
  }
  const auto q = fault.Quadrature(2, 3, 2, dates.back());
  out << std::format("quadrature {}\n",
                     dates.back() ? std::format("{:.17g}", *dates.back()) : std::string{"-"});
  for (auto i = std::size_t{0}; i < q.Size(); ++i) {
    out << std::format(
        "qpoint {} {} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} "
        "{:.17g} {:.17g} {:.17g}\n",
        i, q.patch[i], q.u[i], q.v[i], q.weights[i], q.points[i][0], q.points[i][1], q.points[i][2],
        q.normals[i][0], q.normals[i][1], q.normals[i][2], q.slip[i][0], q.slip[i][1],
        q.slip[i][2]);
  }
}

}  // namespace

TEST(FaultFile, WritesBackWhatItReadsForThePython) {
  for (const auto* name : {"fault_static", "fault_afterslip", "fault_cartesian"}) {
    SCOPED_TRACE(name);
    PLANETMODEL_FIXTURE(fixture, std::string{name} + ".nc");
    const auto file = pm::ReadFaultFile(fixture);
    const auto path = pm::testing::WrittenDir() / (std::string{"faultfile_back_"} + name + ".nc");
    pm::WriteFaultFile(path, file);
    const auto back = pm::ReadFaultFile(path);
    EXPECT_EQ(back.schema, pm::FAULT_SCHEMA);
    EXPECT_EQ(back.names, file.names);
    EXPECT_EQ(back.source, file.source);
    EXPECT_EQ(back.created, file.created);
    EXPECT_EQ(back.title, file.title);
    EXPECT_EQ(back.meta, file.meta);
    EXPECT_EQ(back.citation, file.citation);
    ExpectSameFault(back.fault, file.fault);
  }
}

TEST(FaultFile, WritesAFaultBuiltInCodeThatReadsBackToItself) {
  for (const auto& [name, file] : {std::pair{"faultfile_node", BuiltNodeFault()},
                                   std::pair{"faultfile_static", BuiltStaticFault()}}) {
    SCOPED_TRACE(name);
    const auto path = pm::testing::WrittenDir() / (std::string{name} + ".nc");
    pm::WriteFaultFile(path, file);
    const auto back = pm::ReadFaultFile(path);
    EXPECT_EQ(back.schema, pm::FAULT_SCHEMA);
    EXPECT_EQ(back.source, file.source);
    EXPECT_NE(back.created.find('T'), std::string::npos);
    EXPECT_EQ(back.title, file.title);
    EXPECT_EQ(back.meta, file.meta);
    EXPECT_EQ(back.citation, file.citation);
    auto names = file.names;
    names.resize(file.fault.Patches().size());
    EXPECT_EQ(back.names, names);
    ExpectSameFault(back.fault, file.fault);
    WriteEvaluations(file.fault, pm::testing::WrittenDir() / (std::string{name} + ".expect"));
  }
}

TEST(FaultFile, WritesACitationByTheRuleOfAModels) {
  // each work trimmed of white space at its ends, an empty one or a repeat
  // dropped
  auto file = BuiltStaticFault();
  file.citation = {" A. ", "", "B.", "A.", "\tC.\n"};
  const auto dir = pm::testing::WrittenDir() / "fault_cited";
  std::filesystem::create_directories(dir);
  pm::WriteFaultFile(dir / "fault.nc", file);
  EXPECT_EQ(pm::ReadFaultFile(dir / "fault.nc").citation,
            (std::vector<std::string>{"A.", "B.", "C."}));
  file.citation = {"  ", ""};
  pm::WriteFaultFile(dir / "fault.nc", file);
  EXPECT_TRUE(pm::ReadFaultFile(dir / "fault.nc").citation.empty());
}

TEST(FaultFile, StampsThisLibraryAsItsSourceUnlessGivenOne) {
  // as the model file's writer does
  const auto dir = pm::testing::WrittenDir() / "fault_stamped";
  std::filesystem::create_directories(dir);
  auto file = BuiltStaticFault();
  file.source = "";
  pm::WriteFaultFile(dir / "fault.nc", file);
  EXPECT_EQ(pm::ReadFaultFile(dir / "fault.nc").source,
            "planetmodel C++ " + std::string{pm::Version()});
}

TEST(FaultFile, RefusesWhatTheFormatCannotHoldBeforeWritingAnything) {
  const auto dir = pm::testing::WrittenDir() / "fault_refused";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  auto refused = [&dir](pm::FaultFile file, const std::string& where) {
    try {
      pm::WriteFaultFile(dir / "fault.nc", file);
      ADD_FAILURE() << where << ": written";
    } catch (const pm::Error& e) {
      EXPECT_EQ(e.Where(), where) << e.what();
    }
  };
  auto file = BuiltStaticFault();
  file.source = " ";
  refused(file, "source");
  file = BuiltStaticFault();
  file.schema = "planetmodel.fault/2";
  refused(file, "schema");
  file = BuiltStaticFault();
  file.names = {"one"};
  refused(file, "names");
  file = BuiltStaticFault();
  file.meta = {{"pair", std::vector<double>{1.0, 2.0}}};
  refused(file, "meta_pair");
  file = BuiltStaticFault();
  file.citation = {"A. Author (2026).\nA fault."};
  refused(file, "citation");
  EXPECT_TRUE(std::filesystem::is_empty(dir));
  // a directory that is not there; a file that stands until replaced
  EXPECT_THROW(pm::WriteFaultFile(dir / "missing" / "fault.nc", BuiltStaticFault()), pm::Error);
  const auto path = dir / "fault.nc";
  pm::WriteFaultFile(path, BuiltStaticFault());
  file = BuiltStaticFault();
  file.source = "\t ";
  EXPECT_THROW(pm::WriteFaultFile(path, file), pm::Error);
  EXPECT_EQ(pm::ReadFaultFile(path).source, "planetmodel C++ tests");
  auto left = 0;
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    left += 1;
    EXPECT_EQ(entry.path().filename(), "fault.nc");
  }
  EXPECT_EQ(left, 1);
}

#if defined(__unix__) || defined(__APPLE__)
TEST(FaultFile, GivesTheFileThePermissionsAPlainOpenWould) {
  const auto file = BuiltNodeFault();
  const auto dir = pm::testing::WrittenDir() / "permissions";
  std::filesystem::create_directories(dir);
  pm::testing::ExpectPlainPermissions(
      dir / "fault.nc", [&file](const std::filesystem::path& p) { pm::WriteFaultFile(p, file); });
}
#endif
