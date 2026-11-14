// A Burgers body and its two Prony branches: each from the other held to
// the Python's `burgers_terms` and `burgers_parameters` on the bodies the
// fixture lists, to rounding; the two inverse to each other; the branches
// what the body's compliance says; and the refusals in the Python's words.
#include "planetmodel/prony.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures.hpp"

#ifndef PLANETMODEL_HAS_BURGERS_TERMS
#error "prony.hpp defines the macro of its Burgers terms"
#endif

namespace {

namespace pm = planetmodel;

constexpr auto EPS = std::numeric_limits<double>::epsilon();

std::vector<double> Numbers(std::istringstream& words) {
  auto out = std::vector<double>{};
  for (auto x = 0.0; words >> x;) out.push_back(x);
  return out;
}

/// |got - want| over |want|.
double Relative(double got, double want) { return std::abs(got - want) / std::abs(want); }

/// Whether the message of what `call` throws ends with `says`.
template <class Call>
::testing::AssertionResult RefusedSaying(Call call, const std::string& says) {
  try {
    call();
  } catch (const pm::Error& e) {
    if (std::string{e.what()}.ends_with(says)) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "said " << e.what();
  }
  return ::testing::AssertionFailure() << "accepted";
}

}  // namespace

TEST(Prony, TheBurgersTermsAndParametersAreThePythons) {
  PLANETMODEL_FIXTURE(path, "burgers.txt");
  auto in = std::ifstream(path);
  auto terms = 0, parameters = 0;
  auto worst = 0.0;
  for (auto line = std::string{}; std::getline(in, line);) {
    auto words = std::istringstream(line);
    auto what = std::string{};
    words >> what;
    const auto n = Numbers(words);
    if (what == "terms") {
      ASSERT_EQ(n.size(), 8u) << line;
      const auto body =
          pm::BurgersBody{.viscosity = n[1], .kelvin_viscosity = n[2], .kelvin_modulus_fraction = n[3]};
      const auto [w, tau] = pm::BurgersTerms(n[0], body);
      for (auto k = 0; k < 2; ++k) {
        worst = std::max({worst, Relative(w[k], n[4 + k]), Relative(tau[k], n[6 + k])});
        EXPECT_LE(Relative(w[k], n[4 + k]), 8.0 * EPS) << line;
        EXPECT_LE(Relative(tau[k], n[6 + k]), 8.0 * EPS) << line;
      }
      // the faster branch first; strengths positive, summing to one to
      // the rounding of the gap between two nearly equal roots
      EXPECT_LT(tau[0], tau[1]) << line;
      EXPECT_GT(w[0], 0.0) << line;
      EXPECT_GT(w[1], 0.0) << line;
      EXPECT_NEAR(w[0] + w[1], 1.0, 1e-13) << line;
      ++terms;
    } else if (what == "parameters") {
      ASSERT_EQ(n.size(), 8u) << line;
      const auto body = pm::BurgersParameters(n[0], {n[1], n[2]}, {n[3], n[4]});
      const auto got = std::array{body.viscosity, body.kelvin_viscosity, body.kelvin_modulus_fraction};
      for (auto k = 0; k < 3; ++k) {
        worst = std::max(worst, Relative(got[k], n[5 + k]));
        EXPECT_LE(Relative(got[k], n[5 + k]), 8.0 * EPS) << line;
      }
      ++parameters;
    }
  }
  EXPECT_GE(terms, 20);
  EXPECT_EQ(parameters, 2 * terms);
  std::cout << "[ agreement ] Burgers terms and parameters: " << worst / EPS << " eps\n";
}

TEST(Prony, TheBurgersTermsAndParametersAreInverses) {
  // a body to its branches and back, and the branches of the body the
  // branches give, to rounding, from the near-equal rates to the far apart
  for (const auto mu : {3e10, 7e10}) {
    for (const auto eta : {1e18, 1e21, 1e23}) {
      for (const auto eta_k : {1e16, 1e20, 1e24}) {
        for (const auto f : {1e-3, 0.5, 1.0, 30.0}) {
          const auto body =
              pm::BurgersBody{.viscosity = eta, .kelvin_viscosity = eta_k, .kelvin_modulus_fraction = f};
          const auto [w, tau] = pm::BurgersTerms(mu, body);
          const auto back = pm::BurgersParameters(mu, w, tau);
          EXPECT_LE(Relative(back.viscosity, eta), 64.0 * EPS) << eta << " " << eta_k << " " << f;
          EXPECT_LE(Relative(back.kelvin_viscosity, eta_k), 1e-11) << eta << " " << eta_k << " " << f;
          EXPECT_LE(Relative(back.kelvin_modulus_fraction, f), 1e-11)
              << eta << " " << eta_k << " " << f;
          const auto [w2, tau2] = pm::BurgersTerms(mu, back);
          for (auto k = 0; k < 2; ++k) {
            EXPECT_LE(Relative(w2[k], w[k]), 1e-11);
            EXPECT_LE(Relative(tau2[k], tau[k]), 1e-11);
          }
          // in either order of the branches
          const auto swapped = pm::BurgersParameters(mu, {w[1], w[0]}, {tau[1], tau[0]});
          EXPECT_EQ(swapped.viscosity, back.viscosity);
          // the steady viscosity is mu sum w_k tau_k
          EXPECT_LE(Relative(mu * (w[0] * tau[0] + w[1] * tau[1]), eta), 8.0 * EPS);
        }
      }
    }
  }
}

TEST(Prony, TheBranchesAreTheRootsOfTheBodysCompliance) {
  // the rates 1 / tau_k are the roots of x^2 - (a + b + c) x + a c, and
  // w_1 x_2 + w_2 x_1 = c: with mu = 2, eta = 2, eta_K = 1 and f = 1.5,
  // a = 1, b = 2 and c = 3, the roots 3 + sqrt(6) and 3 - sqrt(6)
  const auto [w, tau] = pm::BurgersTerms(
      2.0, {.viscosity = 2.0, .kelvin_viscosity = 1.0, .kelvin_modulus_fraction = 1.5});
  const auto x1 = 1.0 / tau[0], x2 = 1.0 / tau[1];
  EXPECT_NEAR(x1, 3.0 + std::sqrt(6.0), 1e-14);
  EXPECT_NEAR(x2, 3.0 - std::sqrt(6.0), 1e-14);
  EXPECT_NEAR(w[0] * x2 + w[1] * x1, 3.0, 1e-14);
  EXPECT_NEAR(w[0] + w[1], 1.0, 1e-15);
}

TEST(Prony, RefusesWhatIsNoBurgersBodyInThePythonsWords) {
  const auto good =
      pm::BurgersBody{.viscosity = 1e21, .kelvin_viscosity = 1e20, .kelvin_modulus_fraction = 0.5};
  EXPECT_TRUE(RefusedSaying([&] { static_cast<void>(pm::BurgersTerms(0.0, good)); },
                            "a Burgers body's modulus is positive and finite"));
  auto bad = good;
  bad.kelvin_viscosity = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(RefusedSaying([&] { static_cast<void>(pm::BurgersTerms(7e10, bad)); },
                            "a Burgers body's kelvin_viscosity is positive and finite"));
  // a member left out of the designated initialiser is refused, not read
  EXPECT_TRUE(RefusedSaying(
      [] {
        static_cast<void>(pm::BurgersTerms(7e10, {.viscosity = 1e21, .kelvin_viscosity = 1e20}));
      },
      "a Burgers body's kelvin_modulus_fraction is positive and finite"));
  EXPECT_TRUE(RefusedSaying(
      [] { static_cast<void>(pm::BurgersParameters(7e10, {0.5, 0.0}, {1.0, 2.0})); },
      "a Burgers body's strengths are positive and finite"));
  EXPECT_TRUE(RefusedSaying(
      [] { static_cast<void>(pm::BurgersParameters(7e10, {0.5, 0.5}, {1.0, -2.0})); },
      "a Burgers body's times are positive and finite"));
  EXPECT_TRUE(RefusedSaying(
      [] { static_cast<void>(pm::BurgersParameters(7e10, {0.5, 0.5}, {2.0, 2.0})); },
      "the two branches have the same time at a point; a Burgers body's two times differ"));
}
