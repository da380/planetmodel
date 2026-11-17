// Decks of depths: an elastic deck (transversely isotropic and isotropic),
// a Maxwell, a prony and a burgers deck read as the Python reads them, the
// layers and interfaces named from the surface down, each kind of
// interpolant the Python's; a deck of radii refused by its column line and
// the other way round; the refusals naming a knot's depth; and every
// fixture written back, and decks built here, left for the Python to read.
#include "planetmodel/cartesian/deck.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures.hpp"

namespace {

namespace flat = planetmodel::cartesian;
using planetmodel::Deck;
using planetmodel::DeckFormat;
using planetmodel::Error;

double ToDouble(const std::string& text) {
  auto value = 0.0;
  std::from_chars(text.data(), text.data() + text.size(), value);
  return value;
}

/// The columns of a deck positive and finite wherever they are not NaN, and
/// not NaN throughout, as the fixtures' `_positive_columns` finds them.
std::vector<std::string> PositiveColumns(const Deck& deck) {
  auto out = std::vector<std::string>{};
  for (const auto& name : deck.Names()) {
    const auto& v = deck.Column(name);
    const auto held = std::ranges::any_of(v, [](double x) { return !std::isnan(x); });
    const auto positive = std::ranges::all_of(
        v, [](double x) { return std::isnan(x) || (std::isfinite(x) && x > 0.0); });
    if (held && positive) out.push_back(name);
  }
  return out;
}

/// The fixtures, each with its format and the kinds its format joins by.
struct Case {
  std::string label;
  const DeckFormat& format;
  std::string kinds;
};

std::vector<Case> Cases() {
  return {{"elastic", flat::ELASTIC(), ""},
          {"isotropic", flat::ELASTIC(), ""},
          {"maxwell", flat::MAXWELL(), "maxwell"},
          {"prony", flat::PRONY(), "prony"},
          {"burgers", flat::BURGERS(), "burgers"}};
}

/// The deck against the Python's expectations: the layering, the names its
/// header gives, and every kind's values and derivatives.
void ExpectAsThePython(const Deck& deck, const std::filesystem::path& expect,
                       const std::string& own, double& worst) {
  auto in = std::ifstream(expect);
  auto values = 0;
  auto functions = std::map<std::string, planetmodel::DeckFunctions>{};
  for (const auto* kind : {"cubic", "natural", "linear"}) {
    functions.emplace(kind, flat::DeckLayers(deck, kind));
  }
  const auto positive = PositiveColumns(deck);
  for (const auto* kind : {"log-cubic", "log-natural", "log-linear"}) {
    auto kinds = std::map<std::string, std::string, std::less<>>{};
    for (const auto& name : positive) kinds.emplace(name, kind);
    functions.emplace(kind, flat::DeckLayers(deck, "cubic", kinds));
  }
  if (own == "maxwell") {
    functions.emplace(own, flat::DeckLayers(deck, "cubic", planetmodel::MAXWELL_KINDS));
  } else if (own == "prony") {
    functions.emplace(own, flat::DeckLayers(deck, "cubic", planetmodel::PRONY_KINDS));
  } else if (own == "burgers") {
    functions.emplace(own, flat::DeckLayers(deck, "cubic", planetmodel::BURGERS_KINDS));
  }
  const auto names = flat::DepthNames(deck);
  for (auto line = std::string{}; std::getline(in, line);) {
    auto words = std::istringstream{line};
    auto key = std::string{};
    words >> key;
    if (key == "knots") {
      auto n = std::size_t{0};
      words >> n;
      EXPECT_EQ(deck.NKnots(), n);
    } else if (key == "boundaries") {
      auto want = std::vector<double>{};
      for (auto w = std::string{}; words >> w;) want.push_back(ToDouble(w));
      EXPECT_EQ(deck.Boundaries(), want);
      EXPECT_EQ(functions.at("cubic").skeleton.Boundaries(), want);
    } else if (key == "layer_name" || key == "interface_name") {
      auto i = std::size_t{0};
      auto name = std::string{};
      words >> i >> name;
      const auto& list = key == "layer_name" ? names.layers : names.interfaces;
      ASSERT_LT(i, list.size());
      EXPECT_EQ(list[i].value_or("-"), name) << key << " " << i;
    } else if (key == "absent") {
      auto i = std::size_t{0};
      auto column = std::string{};
      words >> i >> column;
      EXPECT_FALSE(functions.at("cubic").layers.at(i).contains(column)) << line;
    } else if (key == "value") {
      auto kind = std::string{};
      auto i = std::size_t{0};
      auto column = std::string{};
      auto depth = std::string{};
      auto value = std::string{};
      auto derivative = std::string{};
      words >> kind >> i >> column >> depth >> value >> derivative;
      const auto& f = functions.at(kind).layers.at(i).at(column);
      const auto& [lo, hi] = f.Interval();
      const auto scale = std::abs(ToDouble(value)) + 1.0;
      const auto got = f.Value(ToDouble(depth));
      EXPECT_NEAR(got, ToDouble(value), 1e-12 * scale) << line;
      EXPECT_NEAR(f.Derivative(ToDouble(depth)), ToDouble(derivative), 1e-10 * scale / (hi - lo))
          << line;
      worst = std::max(worst, std::abs(got - ToDouble(value)) / scale);
      ++values;
    }
  }
  EXPECT_GT(values, 20);
}

/// What refusing `call` says.
template <class Call>
std::string Refusal(Call call) {
  try {
    call();
  } catch (const Error& e) {
    return e.what();
  }
  ADD_FAILURE() << "nothing was refused";
  return {};
}

}  // namespace

TEST(CartesianDeck, ReadsEveryDeckOfDepthsAsThePythonDoes) {
  auto worst = 0.0;
  for (const auto& [label, format, kinds] : Cases()) {
    SCOPED_TRACE(label);
    PLANETMODEL_FIXTURE(path, "flat_deck_" + label + ".txt");
    const auto deck = planetmodel::ReadDeck(path, format);
    ExpectAsThePython(deck, path.parent_path() / ("flat_deck_" + label + ".expect"), kinds,
                      worst);
  }
  RecordProperty("worst_relative_value", std::format("{:.3e}", worst));
}

TEST(CartesianDeck, NamesItsLayersAndInterfacesFromTheSurfaceDown) {
  PLANETMODEL_FIXTURE(path, "flat_deck_maxwell.txt");
  const auto deck = planetmodel::ReadDeck(path, flat::MAXWELL());
  // the knots are depths, increasing down the file from the free surface
  EXPECT_EQ(deck.Radius().front(), 0.0);
  EXPECT_EQ(deck.Boundaries(), (std::vector{0.0, 30e3, 100e3, 200e3, 400e3}));
  const auto names = flat::DepthNames(deck);
  EXPECT_EQ(names.layers.front(), "crust");
  EXPECT_FALSE(names.layers[2]);
  // one interface per layer, its top; the base is none
  ASSERT_EQ(names.interfaces.size(), 4u);
  EXPECT_EQ(names.interfaces.front(), "surface");
  EXPECT_FALSE(names.interfaces.back());
  // the crust is elastic: its viscosity absent
  const auto layers = flat::DeckLayers(deck, "cubic", planetmodel::MAXWELL_KINDS);
  EXPECT_FALSE(layers.layers[0].contains("viscosity"));
  // geometric in depth across the lithosphere, two decades down
  const auto& eta = layers.layers[1].at("viscosity");
  EXPECT_NEAR(eta.Value(65e3), 1e22, 1e-7 * 1e22);  // written to eight figures
  // an unnamed deck: the top is the surface where it starts at depth 0, and
  // unnamed where it starts below it
  const auto plain = Deck({0.0, 1.0, 1.0, 2.0}, {{"rho", {1.0, 1.0, 2.0, 2.0}}});
  EXPECT_EQ(flat::DepthNames(plain).interfaces,
            (std::vector<std::optional<std::string>>{"surface", std::nullopt}));
  PLANETMODEL_FIXTURE(buried, "flat_deck_isotropic.txt");
  const auto below = flat::DepthNames(planetmodel::ReadDeck(buried, flat::ELASTIC()));
  EXPECT_FALSE(below.interfaces.front());
  // a count other than the deck's is refused
  const auto wrong = Deck({0.0, 1.0, 1.0, 2.0}, {{"rho", {1.0, 1.0, 2.0, 2.0}}},
                          {"a deck", "depth rho", "# interfaces surface moho base"});
  EXPECT_NE(Refusal([&] { static_cast<void>(flat::DepthNames(wrong)); })
                .find("names 3 interfaces; its table has 2"),
            std::string::npos);
}

TEST(CartesianDeck, ADeckOfRadiiIsNotADeckOfDepthsNorTheOtherWayRound) {
  PLANETMODEL_FIXTURE(depths, "flat_deck_maxwell.txt");
  PLANETMODEL_FIXTURE(radii, "deck_maxwell.txt");
  auto said = Refusal([&] { static_cast<void>(planetmodel::ReadDeck(radii, flat::MAXWELL())); });
  EXPECT_NE(said.find("a Maxwell deck's second line names its columns, depth rho vpv"),
            std::string::npos)
      << said;
  said = Refusal([&] { static_cast<void>(planetmodel::ReadDeck(depths, planetmodel::MAXWELL())); });
  EXPECT_NE(said.find("names its columns, r rho vpv"), std::string::npos) << said;
  PLANETMODEL_FIXTURE(prony, "flat_deck_prony.txt");
  said = Refusal([&] { static_cast<void>(planetmodel::ReadDeck(prony, planetmodel::PRONY())); });
  EXPECT_NE(said.find("r rho vpv vsv vph vsh eta then relaxation_strength_k"),
            std::string::npos)
      << said;
  PLANETMODEL_FIXTURE(sphere, "deck_prony.txt");
  said = Refusal([&] { static_cast<void>(planetmodel::ReadDeck(sphere, flat::PRONY())); });
  EXPECT_NE(said.find("depth rho vpv vsv vph vsh eta then relaxation_strength_k"),
            std::string::npos)
      << said;
  PLANETMODEL_FIXTURE(elastic, "flat_deck_elastic.txt");
  said = Refusal([&] { static_cast<void>(planetmodel::ReadDeck(elastic, flat::MAXWELL())); });
  EXPECT_NE(said.find("a Maxwell deck's second line"), std::string::npos) << said;
}

TEST(CartesianDeck, RefusesWhatBreaksADeckNamingTheDepth) {
  const auto text = [](std::string_view columns, std::string_view rows) {
    return std::format("a deck\n{}\n{}", columns, rows);
  };
  // an elastic deck's column line is one of its two, and its rows agree
  auto said = Refusal([&] {
    static_cast<void>(planetmodel::ReadDeckText(
        text("depth rho vp", "0 1 2\n1 1 2\n"), flat::ELASTIC()));
  });
  EXPECT_NE(said.find("an elastic deck's second line names its columns, depth rho vpv vsv "
                      "qkappa qmu vph vsh eta or depth rho vp vs qkappa qmu"),
            std::string::npos)
      << said;
  said = Refusal([&] {
    static_cast<void>(planetmodel::ReadDeckText(
        text("depth rho vp vs qkappa qmu", "0 1 2 3 4 5 6 7 8\n1 1 2 3 4 5 6 7 8\n"),
        flat::ELASTIC()));
  });
  EXPECT_NE(said.find("the column line names 5 columns after the depth; the rows have 8"),
            std::string::npos)
      << said;
  said = Refusal([&] {
    static_cast<void>(planetmodel::ReadDeckText(
        text("depth rho vp vs qkappa qmu", "0 1 2 3\n1 1 2 3\n"), flat::ELASTIC()));
  });
  EXPECT_NE(said.find("columns after the depth, not 3"), std::string::npos) << said;
  // a prony deck's strengths and times, at a depth
  const auto prony = "depth rho vpv vsv vph vsh eta relaxation_strength_1 relaxation_time_1";
  said = Refusal([&] {
    static_cast<void>(planetmodel::ReadDeckText(
        text(prony, "0 1 2 3 4 5 6 1.5 1e9\n10 1 2 3 4 5 6 0.5 1e9\n"), flat::PRONY()));
  });
  EXPECT_NE(said.find("relaxation_strength_1 is 1.5 at depth = 0; a strength lies in [0, 1]"),
            std::string::npos)
      << said;
  said = Refusal([&] {
    static_cast<void>(planetmodel::ReadDeckText(
        text(prony, "0 1 2 3 4 5 6 0.5 1e9\n10 1 2 3 4 5 6 0.5 -1\n"), flat::PRONY()));
  });
  EXPECT_NE(said.find("relaxation_time_1 is -1 at depth = 10; a time is positive and finite"),
            std::string::npos)
      << said;
  // a burgers deck's viscosities, at a depth
  const auto burgers =
      "depth rho vpv vsv vph vsh eta viscosity kelvin_viscosity kelvin_modulus_fraction";
  said = Refusal([&] {
    static_cast<void>(planetmodel::ReadDeckText(
        text(burgers, "0 1 2 3 4 5 6 1e21 nan nan\n20 1 2 3 4 5 6 0 nan nan\n"),
        flat::BURGERS()));
  });
  EXPECT_NE(said.find("viscosity is 0 at depth = 20; it is positive and finite"),
            std::string::npos)
      << said;
  // a column joined in its logarithm, at a depth
  const auto deck = Deck({0.0, 5.0, 10.0}, {{"viscosity", {1e21, 0.0, 1e20}}});
  said = Refusal([&] {
    static_cast<void>(flat::DeckLayers(deck, "cubic", {{"viscosity", "log-linear"}}));
  });
  EXPECT_NE(said.find("is 0 at depth = 5 on layer 0"), std::string::npos) << said;
  // the spherical formats still name the radius
  said = Refusal([&] {
    static_cast<void>(planetmodel::DeckLayers(deck, "cubic", {{"viscosity", "log-linear"}}));
  });
  EXPECT_NE(said.find("is 0 at r = 5 on layer 0"), std::string::npos) << said;
}

TEST(CartesianDeck, WritesEveryFixtureBackForThePython) {
  // tests/test_cpp_written_cartesian.py holds each to the Python's own writing
  for (const auto& [label, format, kinds] : Cases()) {
    SCOPED_TRACE(label);
    PLANETMODEL_FIXTURE(path, "flat_deck_" + label + ".txt");
    const auto deck = planetmodel::ReadDeck(path, format);
    const auto out = planetmodel::testing::WrittenDir() / ("flat_deck_" + label + ".txt");
    planetmodel::WriteDeck(out, deck, format);
    const auto back = planetmodel::ReadDeck(out, format);
    EXPECT_EQ(back.Radius(), deck.Radius());
    EXPECT_EQ(back.Names(), deck.Names());
    // the fixture is the Python's writing: the C++'s is the same text
    auto text = [](const std::filesystem::path& p) {
      auto in = std::ifstream(p, std::ios::binary);
      return std::string(std::istreambuf_iterator<char>(in), {});
    };
    EXPECT_EQ(text(out), text(path));
  }
}

TEST(CartesianDeck, WritesDecksBuiltHereForThePython) {
  // a crust over a viscous mantle, as the Python's test rebuilds them
  const auto nan = std::nan("");
  const auto depth = std::vector{0.0, 15e3, 30e3, 30e3, 100e3, 200e3};
  const auto elastic = std::vector<std::pair<std::string, std::vector<double>>>{
      {"rho", {2.7e3, 2.8e3, 2.9e3, 3.3e3, 3.35e3, 3.4e3}},
      {"vpv", {6e3, 6.2e3, 6.4e3, 8e3, 8.1e3, 8.2e3}},
      {"vsv", {3.5e3, 3.6e3, 3.7e3, 4.5e3, 4.55e3, 4.6e3}},
      {"vph", {6.1e3, 6.3e3, 6.5e3, 8.1e3, 8.2e3, 8.3e3}},
      {"vsh", {3.6e3, 3.7e3, 3.8e3, 4.6e3, 4.65e3, 4.7e3}},
      {"eta", {1.0, 1.0, 1.0, 0.95, 0.97, 1.0}}};
  const auto with = [&](std::vector<std::pair<std::string, std::vector<double>>> more) {
    auto out = elastic;
    out.insert(out.end(), more.begin(), more.end());
    return out;
  };
  const auto dir = planetmodel::testing::WrittenDir();
  const auto layers = std::vector<std::optional<std::string>>{"crust", std::nullopt};
  const auto faces = std::vector<std::optional<std::string>>{"surface", "moho"};

  const auto maxwell = Deck(depth, with({{"viscosity", {nan, nan, nan, 1e23, 1e22, 1e21}}}),
                            flat::MaxwellHeader("a box built in C++", layers, faces));
  planetmodel::WriteDeck(dir / "flat_deck_maxwell_built.txt", maxwell, flat::MAXWELL());
  const auto back = planetmodel::ReadDeck(dir / "flat_deck_maxwell_built.txt", flat::MAXWELL());
  EXPECT_EQ(back.Header()[1], "depth rho vpv vsv vph vsh eta viscosity");
  EXPECT_EQ(flat::DepthNames(back).layers, layers);
  EXPECT_EQ(flat::DepthNames(back).interfaces, faces);

  // a layer of one branch in a deck of two: completed as it is written
  const auto prony = Deck(depth,
                          with({{"relaxation_strength_1", {nan, nan, nan, 1.0, 1.0, 1.0}},
                                {"relaxation_time_1", {nan, nan, nan, 3e11, 2e11, 1e11}},
                                {"relaxation_strength_2", {nan, nan, nan, nan, nan, nan}},
                                {"relaxation_time_2", {nan, nan, nan, nan, nan, nan}}}),
                          flat::PronyHeader("a box built in C++", 2, layers, faces));
  planetmodel::WriteDeck(dir / "flat_deck_prony_built.txt", prony, flat::PRONY());
  const auto completed = planetmodel::ReadDeck(dir / "flat_deck_prony_built.txt", flat::PRONY());
  EXPECT_EQ(planetmodel::PronyBranches(completed), 2u);
  EXPECT_EQ(completed.Column("relaxation_strength_2")[4], 0.0);
  EXPECT_EQ(completed.Column("relaxation_time_2")[4], 2e11);

  const auto burgers =
      Deck(depth,
           with({{"viscosity", {nan, nan, nan, 1e22, 3e21, 1e21}},
                 {"kelvin_viscosity", {nan, nan, nan, 1e20, 5e19, 2e19}},
                 {"kelvin_modulus_fraction", {nan, nan, nan, 0.5, 1.0, 1.5}}}),
           flat::BurgersHeader("a Burgers box built in C++", layers));
  planetmodel::WriteDeck(dir / "flat_deck_burgers_built.txt", burgers, flat::BURGERS());

  // an elastic deck is written with the widest set of columns it holds
  const auto isotropic = Deck(
      depth,
      {{"rho", elastic[0].second}, {"vp", elastic[1].second}, {"vs", elastic[2].second},
       {"qkappa", {nan, nan, nan, 5e4, 5e4, 5e4}}, {"qmu", {600.0, 600.0, 600.0, 300.0, 250.0, 200.0}}},
      flat::ElasticHeader("an elastic box built in C++", true, layers, faces));
  planetmodel::WriteDeck(dir / "flat_deck_elastic_built.txt", isotropic, flat::ELASTIC());
  const auto read = planetmodel::ReadDeck(dir / "flat_deck_elastic_built.txt", flat::ELASTIC());
  EXPECT_EQ(read.Header()[1], "depth rho vp vs qkappa qmu");
  EXPECT_EQ(read.Names(), (std::vector<std::string>{"rho", "vp", "vs", "qkappa", "qmu"}));
  EXPECT_TRUE(std::isnan(read.Column("qkappa").front()));
  EXPECT_EQ(flat::ElasticHeader("t")[1], "depth rho vpv vsv qkappa qmu vph vsh eta");
  // a deck lacking the columns of either set is refused by name
  EXPECT_NE(Refusal([&] {
              static_cast<void>(planetmodel::WriteDeckText(maxwell, flat::ELASTIC()));
            }).find("elastic deck needs columns"),
            std::string::npos);
}
