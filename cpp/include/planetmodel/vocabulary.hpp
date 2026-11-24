#ifndef PLANETMODEL_VOCABULARY_HPP
#define PLANETMODEL_VOCABULARY_HPP

/**
 * @file
 * @brief The shipped field names, as `planetmodel.vocabulary` defines
 * them.
 *
 * A name of the vocabulary means one thing in every model: its character,
 * its dimensions and a gloss.  The Python's `read_model` refuses a file
 * that states a vocabulary name otherwise, so the writer takes a name's
 * spec from here and refuses another.  Restated from the Python, which a
 * fixture holds it to.
 */

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "planetmodel/character.hpp"
#include "planetmodel/units.hpp"

namespace planetmodel {

/** @brief What a field name means: its character, dimensions and a gloss. */
struct FieldSpec {
  Character character = SCALAR;           /**< Its law under a mapping. */
  std::optional<Dimensions> dimensions{}; /**< Its dimensions, where they are declared. */
  std::string meaning{};                  /**< What it is. */
};

/** @brief The shipped names, each with its spec. */
[[nodiscard]] inline const std::map<std::string, FieldSpec, std::less<>>& Vocabulary() {
  namespace u = units;
  static const auto vocabulary = std::map<std::string, FieldSpec, std::less<>>{
      {"rho", {DENSITY, u::DENSITY, "mass density"}},
      {"vp", {SCALAR, u::VELOCITY, "P-wave speed of an isotropic medium"}},
      {"vs", {SCALAR, u::VELOCITY, "S-wave speed of an isotropic medium"}},
      {"vpv", {SCALAR, u::VELOCITY, "P-wave speed along the symmetry axis (vertical)"}},
      {"vsv",
       {SCALAR, u::VELOCITY, "S-wave speed polarised along the symmetry axis (vertical)"}},
      {"vph", {SCALAR, u::VELOCITY, "P-wave speed normal to the symmetry axis (horizontal)"}},
      {"vsh",
       {SCALAR, u::VELOCITY,
        "S-wave speed polarised normal to the symmetry axis (horizontal)"}},
      {"eta", {SCALAR, u::DIMENSIONLESS, "the anisotropy parameter F / (A - 2L)"}},
      {"qkappa", {SCALAR, u::DIMENSIONLESS, "bulk quality factor; zero attenuates nothing"}},
      {"qmu",
       {SCALAR, u::DIMENSIONLESS, "shear quality factor; zero attenuates nothing (a fluid)"}},
      {"qp", {SCALAR, u::DIMENSIONLESS, "quality factor of P waves; zero attenuates nothing"}},
      {"kappa", {DENSITY, u::MODULUS, "bulk modulus of an isotropic medium"}},
      {"mu", {DENSITY, u::MODULUS, "shear modulus of an isotropic medium"}},
      {"A", {DENSITY, u::MODULUS, "Love modulus A = rho vph^2"}},
      {"C", {DENSITY, u::MODULUS, "Love modulus C = rho vpv^2"}},
      {"F", {DENSITY, u::MODULUS, "Love modulus F = eta (A - 2L)"}},
      {"L", {DENSITY, u::MODULUS, "Love modulus L = rho vsv^2"}},
      {"N", {DENSITY, u::MODULUS, "Love modulus N = rho vsh^2"}},
      {"elastic_moduli",
       {ELASTIC, u::MODULUS,
        "the second elasticity tensor, static, Voigt-reduced in the spherical frame"}},
      {"viscosity", {SCALAR, u::VISCOSITY, "dynamic viscosity"}},
      {"kelvin_viscosity",
       {SCALAR, u::VISCOSITY,
        "viscosity of the Kelvin element of a Burgers body, in series with its Maxwell "
        "element"}},
      {"kelvin_modulus_fraction",
       {SCALAR, u::DIMENSIONLESS,
        "shear modulus of the Kelvin element of a Burgers body, as a fraction of the unrelaxed "
        "shear modulus"}},
      {"temperature", {SCALAR, u::TEMPERATURE, "absolute temperature"}},
      {"stress_exponent",
       {SCALAR, u::DIMENSIONLESS,
        "stress exponent n of power-law creep, its strain rate growing as the stress to the n"}},
      {"transition_stress",
       {DENSITY, u::MODULUS,
        "deviatoric stress at which power-law creep strains as fast as linear creep"}},
      {"pressure",
       {DENSITY, u::MODULUS, "hydrostatic pressure of the spherically symmetric reference"}},
      {"g",
       {SCALAR, u::GRAVITY,
        "gravitational acceleration of the spherically symmetric reference, radially "
        "inward, by magnitude"}},
      {"drho_dphi0",
       {DENSITY, Dimensions{.mass = 1, .length = -5, .time = 2},
        "stratification of a fluid layer: d rho / d Phi_0 = rho'(r) / g(r)"}},
      {"bullen_excess",
       {SCALAR, u::DIMENSIONLESS,
        "excess of the Bullen parameter of a layer over one, -kappa rho'(r) / (rho^2 g) - 1: "
        "zero where the layer is neutrally stratified, positive where it is stably "
        "stratified"}},
  };
  return vocabulary;
}

/**
 * @brief The families of names: each stem with the spec every member has, a member being the
 * stem, an underscore and a positive index written without leading zeros (`relaxation_time_3`):
 * the Prony branches of a generalised Maxwell body in shear.
 */
[[nodiscard]] inline const std::map<std::string, FieldSpec, std::less<>>& Families() {
  namespace u = units;
  static const auto families = std::map<std::string, FieldSpec, std::less<>>{
      {"relaxation_strength",
       {SCALAR, u::DIMENSIONLESS,
        "fraction of the unrelaxed shear modulus that relaxes in Prony branch k"}},
      {"relaxation_time", {SCALAR, u::TIME, "relaxation time of Prony branch k"}},
  };
  return families;
}

/**
 * @brief The family and index of a member of one.
 * @param name The name.
 * @return Its stem and index, or empty where it is a member of no family.
 */
[[nodiscard]] inline std::optional<std::pair<std::string, int>> Member(std::string_view name) {
  const auto cut = name.rfind('_');
  if (cut == std::string_view::npos || cut == 0 || cut + 1 == name.size()) return std::nullopt;
  const auto stem = name.substr(0, cut);
  const auto digits = name.substr(cut + 1);
  if (digits.front() == '0' || digits.size() > 9) return std::nullopt;
  auto index = 0;
  for (const auto c : digits) {
    if (c < '0' || c > '9') return std::nullopt;
    index = 10 * index + (c - '0');
  }
  if (!Families().contains(stem)) return std::nullopt;
  return std::pair{std::string{stem}, index};
}

/**
 * @brief The spec of a name the library speaks, of the vocabulary or a family.
 * @param name The name.
 * @return Its spec, or empty where the library has no such name.
 */
[[nodiscard]] inline std::optional<FieldSpec> VocabularySpec(std::string_view name) {
  const auto& vocabulary = Vocabulary();
  if (const auto it = vocabulary.find(name); it != vocabulary.end()) return it->second;
  if (const auto found = Member(name)) return Families().find(found->first)->second;
  return std::nullopt;
}

}  // namespace planetmodel

#endif  // PLANETMODEL_VOCABULARY_HPP
