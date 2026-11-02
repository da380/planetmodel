// Decks: the shipped mineos decks, a Maxwell deck, TauP's two tables and
// one of short layers read as the Python reads them, each kind of interpolant the Python's, a
// malformed deck refused by name; and every fixture written back, and a
// Maxwell deck built here, left for the Python to read.
#include "planetmodel/deck.hpp"

#include <gtest/gtest.h>

#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <limits>
#include <optional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "fixtures.hpp"

namespace {

using planetmodel::Deck;
using planetmodel::Error;

double ToDouble(const std::string& text) {
  auto value = 0.0;
  std::from_chars(text.data(), text.data() + text.size(), value);
  return value;
}

std::string Unhex(const std::string& hex) {
  auto out = std::string{};
  for (auto i = std::size_t{0}; i + 1 < hex.size(); i += 2) {
    out += static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16));
  }
  return out;
}

/// What names a deck's layers: no header, a mineos or Maxwell header, or
/// a TauP table's labels.
enum class Named { None, Mineos, Maxwell, Nd };

/// The deck against the Python's expectations: the layering, the header
/// and its names, and every kind's values and derivatives.
void ExpectAsThePython(const Deck& deck, const std::filesystem::path& expect, Named named) {
  auto in = std::ifstream(expect);
  auto values = 0;
  auto functions = std::map<std::string, planetmodel::DeckFunctions>{};
  for (const auto* kind : {"cubic", "natural", "linear"}) {
    functions.emplace(kind, planetmodel::DeckLayers(deck, kind));
  }
  if (named == Named::Maxwell) {
    functions.emplace("maxwell",
                      planetmodel::DeckLayers(deck, "cubic", planetmodel::MAXWELL_KINDS));
  }
  const auto mineos = named == Named::Mineos;
  const auto header = mineos ? std::optional{planetmodel::ReadMineosHeader(deck.Header(),
                                                                           deck.Source())}
                             : std::nullopt;
  const auto names = mineos                       ? std::optional{planetmodel::MineosNames(deck)}
                     : named == Named::Maxwell ? std::optional{planetmodel::MaxwellNames(deck)}
                     : named == Named::Nd      ? std::optional{planetmodel::NdNames(deck)}
                                               : std::nullopt;
  auto label = std::size_t{0};
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
    } else if (key == "header") {
      auto field = std::string{};
      auto value = std::string{};
      words >> field >> value;
      if (field == "name") {
        EXPECT_EQ(header->name, Unhex(value));
      } else if (field == "tref") {
        EXPECT_EQ(header->tref, ToDouble(value));
      } else {
        const auto counts = std::map<std::string, int>{{"ifanis", header->ifanis},
                                                       {"ifdeck", header->ifdeck},
                                                       {"nknot", header->nknot},
                                                       {"nic", header->nic},
                                                       {"noc", header->noc}};
        EXPECT_EQ(counts.at(field), std::stoi(value)) << field;
      }
    } else if (key == "label") {
      auto r = std::string{};
      auto text = std::string{};
      words >> r >> text;
      ASSERT_LT(label, deck.Labels().size());
      EXPECT_EQ(deck.Labels()[label].first, ToDouble(r));
      EXPECT_EQ(deck.Labels()[label].second, text);
      ++label;
    } else if (key == "extras2") {
      auto want = std::vector<std::string>{};
      for (auto w = std::string{}; words >> w;) want.push_back(w);
      EXPECT_EQ(header->extras2, want);
    } else if (key == "layer_name" || key == "interface_name") {
      auto i = std::size_t{0};
      auto name = std::string{};
      words >> i >> name;
      const auto& list = key == "layer_name" ? names->layers : names->interfaces;
      ASSERT_LT(i, list.size());
      EXPECT_EQ(list[i].value_or("-"), name) << key << " " << i;
    } else if (key == "ignored") {
      auto n = std::size_t{0};
      words >> n;
      EXPECT_EQ(names->ignored.size(), n);
    } else if (key == "absent") {
      auto i = std::size_t{0};
      auto column = std::string{};
      words >> i >> column;
      EXPECT_FALSE(functions.at("cubic").layers.at(i).contains(column)) << line;
    } else if (key == "value") {
      auto kind = std::string{};
      auto i = std::size_t{0};
      auto column = std::string{};
      auto r = std::string{};
      auto value = std::string{};
      auto derivative = std::string{};
      words >> kind >> i >> column >> r >> value >> derivative;
      const auto& f = functions.at(kind).layers.at(i).at(column);
      const auto& [lo, hi] = f.Interval();
      const auto scale = std::abs(ToDouble(value)) + 1.0;
      EXPECT_NEAR(f.Value(ToDouble(r)), ToDouble(value), 1e-12 * scale) << line;
      EXPECT_NEAR(f.Derivative(ToDouble(r)), ToDouble(derivative), 1e-10 * scale / (hi - lo))
          << line;
      ++values;
    }
  }
  EXPECT_GT(values, 20);
}

}  // namespace

TEST(Deck, ReadsTheShippedMineosDecksAsThePythonDoes) {
  for (const auto* label : {"prem200", "nocrust"}) {
    SCOPED_TRACE(label);
    PLANETMODEL_FIXTURE(path, std::string{"deck_"} + label + ".txt");
    const auto deck = planetmodel::ReadDeck(path, planetmodel::MINEOS());
    ExpectAsThePython(deck, path.parent_path() / (std::string{"deck_"} + label + ".expect"),
                      Named::Mineos);
  }
}

TEST(Deck, ReadsAMaxwellDeckWithItsNamesAndAViscosityAbsentFromTheCore) {
  PLANETMODEL_FIXTURE(path, "deck_maxwell.txt");
  const auto deck = planetmodel::ReadDeck(path, planetmodel::MAXWELL());
  ExpectAsThePython(deck, path.parent_path() / "deck_maxwell.expect", Named::Maxwell);
  const auto layers = planetmodel::DeckLayers(deck, "cubic", planetmodel::MAXWELL_KINDS);
  EXPECT_FALSE(layers.layers[0].contains("viscosity"));
  EXPECT_TRUE(layers.layers[2].contains("viscosity"));
  EXPECT_EQ(deck.Names(), planetmodel::MAXWELL_COLUMNS);
  EXPECT_EQ(deck.Header().size(), 4u);
  const auto names = planetmodel::MaxwellNames(deck);
  EXPECT_EQ(names.layers.at(2), "lowermost_mantle");
  EXPECT_FALSE(names.layers.at(3).has_value());
  EXPECT_EQ(names.interfaces.back(), "surface");
}

TEST(Deck, ReadsTauPTablesAsThePythonDoes) {
  for (const auto& [label, format] :
       std::vector<std::pair<std::string, planetmodel::DeckFormat>>{
           {"nd", planetmodel::ND()}, {"tvel", planetmodel::TVEL()}}) {
    SCOPED_TRACE(label);
    PLANETMODEL_FIXTURE(path, "deck_" + label + ".txt");
    const auto deck = planetmodel::ReadDeck(path, format);
    ExpectAsThePython(deck, path.parent_path() / ("deck_" + label + ".expect"), Named::Nd);
  }
  PLANETMODEL_FIXTURE(path, "deck_nd.txt");
  const auto deck = planetmodel::ReadDeck(path, planetmodel::ND());
  EXPECT_EQ(deck.Names(), (std::vector<std::string>{"vp", "vs", "rho", "qp", "qmu"}));
  EXPECT_EQ(deck.Radius().front(), 0.0);
  EXPECT_EQ(deck.Radius().back(), 100e3);
  EXPECT_EQ(deck.Column("vp").back(), 5.8e3);
  const auto names = planetmodel::NdNames(deck);
  EXPECT_EQ(names.layers.front(), "inner_core");
  EXPECT_EQ(names.interfaces,
            (std::vector<std::optional<std::string>>{"icb", "cmb", std::nullopt, std::nullopt,
                                                     "moho", "surface"}));
  const auto tvel = planetmodel::ReadDeck(path.parent_path() / "deck_tvel.txt",
                                          planetmodel::TVEL());
  EXPECT_EQ(tvel.Header(), (std::vector<std::string>{"a P model", "an S model"}));
}

TEST(Deck, RefusesAMalformedTauPTableByName) {
  auto refusal = [](std::string_view text) {
    try {
      static_cast<void>(planetmodel::ReadDeckText(text, planetmodel::ND(), "t.nd"));
    } catch (const Error& e) {
      return std::string{e.what()};
    }
    return std::string{"accepted"};
  };
  const auto has = [](const std::string& got, std::string_view want) {
    return got.find(want) != std::string::npos;
  };
  EXPECT_PRED2(has, refusal("0 5.8 3.4\n10 5.8\n"), "the same on every row");
  EXPECT_PRED2(has, refusal("0 1 2 3 4 5 6\n10 1 2 3 4 5 6\n"), "three to six numbers");
  EXPECT_PRED2(has, refusal("0 5.8 3.4\nlabel\n10 5.8 3.4\n"), "sits at no boundary");
  EXPECT_PRED2(has, refusal("10 5.8 3.4\n0 5.8 3.4\n"), "must not decrease");
  EXPECT_PRED2(has, refusal("0 5.8 x\n10 5.8 3.4\n"), "a row is numbers");
  EXPECT_PRED2(has, refusal(""), "got widths none");
  const auto absent = Deck({0.0, 1.0}, {{"vp", {1.0, std::nan("")}}, {"vs", {1.0, 1.0}}});
  EXPECT_THROW(static_cast<void>(planetmodel::WriteDeckText(absent, planetmodel::ND())),
               Error);
}

TEST(Deck, NamesAMaxwellDeckFromItsHeaderOrNotAtAll) {
  const auto rows = std::string{
      "0 10 9 0 9 0 1 nan\n1 10 9 0 9 0 1 nan\n1 4 8 4 8 4 1 1e21\n2 3 7 3 7 3 1 1e22\n"};
  const auto read = [&](std::string_view naming) {
    return planetmodel::ReadDeckText(
        std::format("a body\nr rho vpv vsv vph vsh eta viscosity # (SI)\n{}{}", naming, rows),
        planetmodel::MAXWELL(), "m.txt");
  };
  const auto named = planetmodel::MaxwellNames(read("# layers core -\n# interfaces cmb top\n"));
  EXPECT_EQ(named.layers, (std::vector<std::optional<std::string>>{"core", std::nullopt}));
  EXPECT_EQ(named.interfaces, (std::vector<std::optional<std::string>>{"cmb", "top"}));
  const auto plain = planetmodel::MaxwellNames(read(""));
  EXPECT_EQ(plain.layers, (std::vector<std::optional<std::string>>(2)));
  EXPECT_EQ(plain.interfaces,
            (std::vector<std::optional<std::string>>{std::nullopt, "surface"}));
  // a comment of no keyword ends the header, and the naming line after it
  // is a comment like any other
  const auto noted = read("# a note\n# layers core mantle\n");
  EXPECT_EQ(noted.Header().size(), 2u);
  EXPECT_EQ(noted.NKnots(), 4u);
  EXPECT_THROW(static_cast<void>(planetmodel::MaxwellNames(read("# layers a b c\n"))), Error);
  EXPECT_THROW(static_cast<void>(planetmodel::MaxwellNames(read("# interfaces top\n"))),
               Error);
  EXPECT_THROW(static_cast<void>(read("# layers a b\n# layers c d\n")), Error);
}

TEST(Deck, JoinsAColumnByAKindOfItsOwn) {
  const auto deck = Deck({0.0, 0.3, 0.5, 1.0}, {{"rho", {5.0, 4.0, 4.5, 3.0}},
                                                {"viscosity", {1e20, 1e23, 1e21, 1e22}}});
  const auto cubic = planetmodel::DeckLayers(deck);
  const auto mixed = planetmodel::DeckLayers(deck, "cubic", planetmodel::MAXWELL_KINDS);
  auto lowest_cubic = 1e30;
  auto lowest_mixed = 1e30;
  for (auto i = 0; i <= 40; ++i) {
    const auto r = i / 40.0;
    EXPECT_EQ(mixed.layers[0].at("rho").Value(r), cubic.layers[0].at("rho").Value(r));
    lowest_cubic = std::min(lowest_cubic, cubic.layers[0].at("viscosity").Value(r));
    lowest_mixed = std::min(lowest_mixed, mixed.layers[0].at("viscosity").Value(r));
  }
  EXPECT_LT(lowest_cubic, 0.0);  // a cubic overshoots
  EXPECT_GE(lowest_mixed, 1e20);
  EXPECT_THROW(static_cast<void>(planetmodel::DeckLayers(deck, "cubic", {{"qmu", "linear"}})),
               Error);
  EXPECT_THROW(
      static_cast<void>(planetmodel::DeckLayers(deck, "cubic", {{"viscosity", "pchip"}})),
      Error);
}

TEST(Deck, WritesEveryFixtureBackForThePython) {
  // tests/test_cpp_written_decks.py holds each to the Python's own writing
  const auto cases = std::vector<std::pair<std::string, planetmodel::DeckFormat>>{
      {"prem200", planetmodel::MINEOS()},
      {"nocrust", planetmodel::MINEOS()},
      {"maxwell", planetmodel::MAXWELL()},
      {"nd", planetmodel::ND()},
      {"tvel", planetmodel::TVEL()},
      {"knots", planetmodel::ColumnsFormat({"rho", "vp"}, 1)}};
  for (const auto& [label, format] : cases) {
    SCOPED_TRACE(label);
    PLANETMODEL_FIXTURE(path, "deck_" + label + ".txt");
    const auto deck = planetmodel::ReadDeck(path, format);
    const auto out = planetmodel::testing::WrittenDir() / ("deck_" + label + ".txt");
    planetmodel::WriteDeck(out, deck, format);
    const auto back = planetmodel::ReadDeck(out, format);
    EXPECT_EQ(back.Radius(), deck.Radius());
    EXPECT_EQ(back.Names(), deck.Names());
    EXPECT_EQ(back.Labels(), deck.Labels());
  }
}

TEST(Deck, WritesAMaxwellDeckBuiltHereForThePython) {
  // two layers, a fluid core under a viscous mantle, as the Python's test
  // rebuilds it
  const auto nan = std::nan("");
  const auto deck = Deck(
      {0.0, 0.5e6, 1.0e6, 1.0e6, 1.5e6, 2.0e6},
      {{"rho", {1.1e4, 1.05e4, 1.0e4, 4.5e3, 4.0e3, 3.5e3}},
       {"vpv", {9e3, 8.9e3, 8.8e3, 8e3, 7.5e3, 7e3}},
       {"vsv", {0.0, 0.0, 0.0, 4.5e3, 4.2e3, 4e3}},
       {"vph", {9e3, 8.9e3, 8.8e3, 8.1e3, 7.6e3, 7.1e3}},
       {"vsh", {0.0, 0.0, 0.0, 4.6e3, 4.3e3, 4.1e3}},
       {"eta", {1.0, 1.0, 1.0, 0.95, 0.97, 1.0}},
       {"viscosity", {nan, nan, nan, 1e23, 1e22, 1e21}}},
      planetmodel::MaxwellHeader("a body built in C++", {"core", std::nullopt},
                                 {"cmb", "surface"}));
  const auto out = planetmodel::testing::WrittenDir() / "deck_maxwell_built.txt";
  planetmodel::WriteDeck(out, deck, planetmodel::MAXWELL());
  const auto names = planetmodel::MaxwellNames(planetmodel::ReadDeck(out, planetmodel::MAXWELL()));
  EXPECT_EQ(names.layers.at(0), "core");
  EXPECT_EQ(names.interfaces.at(0), "cmb");
}

TEST(Deck, WritesAsThePythonDoes) {
  const auto nan = std::nan("");
  const auto deck = Deck({0.0, 1.0, 1.0, 2.5},
                         {{"rho", {13088.5, 12000.25, 4000.0, -0.0}},
                          {"vpv", {1e21, 1.5e-5, 0.0001, std::numeric_limits<double>::infinity()}},
                          {"vsv", {nan, -nan, 3.0, 2.0}},
                          {"qkappa", {nan, nan, 1.0, 1.0}},
                          {"qmu", {nan, nan, 100.0, 100.0}},
                          {"vph", {1.0, 1.0, 1.0, 1.0}},
                          {"vsh", {1.0, 1.0, 1.0, 1.0}},
                          {"eta", {1.0, 1.0, 1.0, 1.0}},
                          {"vp", {1.0, 1.0, 1.0, 1.0}},
                          {"vs", {1.0, 1.0, 1.0, 1.0}}},
                         {"a title"});
  // the widest set of columns, the counts from the table, no loss as zero
  const auto text = planetmodel::WriteDeckText(deck, planetmodel::MINEOS());
  EXPECT_EQ(text,
            "a title\n1 1 1\n4 2 2\n"
            "         0.000         13088.5           1e+21             nan               0"
            "               0               1               1               1\n"
            "         1.000        12000.25         1.5e-05             nan               0"
            "               0               1               1               1\n"
            "         1.000            4000          0.0001               3               1"
            "             100               1               1               1\n"
            "         2.500              -0             inf               2               1"
            "             100               1               1               1\n");
}

TEST(Deck, JoinsShortLayersAsThePythonDoes) {
  PLANETMODEL_FIXTURE(path, "deck_knots.txt");
  const auto deck = planetmodel::ReadDeck(path, planetmodel::ColumnsFormat({"rho", "vp"}, 1));
  EXPECT_EQ(deck.NLayers(), 4u);
  ExpectAsThePython(deck, path.parent_path() / "deck_knots.expect", Named::None);
}

TEST(Deck, ReadsNumbersAsPythonDoesAndSkipsComments) {
  const auto text = std::string{
      "# a header line\n"
      "0.0  +1.5  nan\n"
      "\n"
      "1.   2e0   NaN  # a comment\n"
      "1.0  3.0   4.0\n"
      "2.0  4.0   inf\n"};
  const auto deck =
      planetmodel::ReadDeckText(text, planetmodel::ColumnsFormat({"a", "b"}, 1), "t.txt");
  EXPECT_EQ(deck.Radius(), (std::vector<double>{0.0, 1.0, 1.0, 2.0}));
  EXPECT_EQ(deck.Column("a"), (std::vector<double>{1.5, 2.0, 3.0, 4.0}));
  EXPECT_TRUE(std::isnan(deck.Column("b")[0]));
  EXPECT_TRUE(std::isinf(deck.Column("b")[3]));
  EXPECT_EQ(deck.Boundaries(), (std::vector<double>{0.0, 1.0, 2.0}));
  const auto layers = planetmodel::DeckLayers(deck, "linear");
  EXPECT_FALSE(layers.layers[0].contains("b"));
  EXPECT_DOUBLE_EQ(layers.layers[0].at("a").Value(0.5), 1.75);
}

TEST(Deck, RefusesWhatIsNotADeckByName) {
  auto refusal = [](std::string_view text, const planetmodel::DeckFormat& format) {
    try {
      static_cast<void>(planetmodel::ReadDeckText(text, format, "d.txt"));
    } catch (const Error& e) {
      return std::string{e.what()};
    }
    return std::string{"accepted"};
  };
  const auto two = planetmodel::ColumnsFormat({"a"});
  const auto has = [](const std::string& got, std::string_view want) {
    return got.find(want) != std::string::npos;
  };
  EXPECT_PRED2(has, refusal("", two), "no knots");
  EXPECT_PRED2(has, refusal("0 1\n1 2 3\n", two), "line 2: has 3 values");
  EXPECT_PRED2(has, refusal("0 1\n1 x\n", two), "line 2: x is not a number");
  EXPECT_PRED2(has, refusal("0 1 2\n1 2 3\n", two), "1 columns after the radius, not 2");
  EXPECT_PRED2(has, refusal("1 1\n0 2\n", two), "non-decreasing");
  EXPECT_PRED2(has, refusal("0 1\n1 1\n1 1\n1 1\n2 1\n", two), "repeats more than twice");
  EXPECT_PRED2(has, refusal("0 1\n1 1\n1 1\n", two), "has one knot");
  EXPECT_PRED2(has, refusal("t\n0 1\n", planetmodel::MINEOS()), "fewer than the 3 header");
  EXPECT_PRED2(has, refusal("t\n1 1.0\n2 0 0\n0 1 2 3 4 5\n1 1 2 3 4 5\n",
                            planetmodel::MINEOS()),
               "ifanis tref ifdeck");
  EXPECT_PRED2(has, refusal("t\nr rho\n0 1 2 3 4 5 6 7\n1 1 2 3 4 5 6 7\n",
                            planetmodel::MAXWELL()),
               "names its columns");
  const auto one = Deck({0.0, 1.0}, {{"a", {1.0, 2.0}}}, {"t", "# extra"});
  EXPECT_THROW(static_cast<void>(planetmodel::WriteDeckText(one, planetmodel::ColumnsFormat(
                                                                     {"a"}, 1))),
               Error);
  auto keyed = planetmodel::ColumnsFormat({"a"}, 1);
  keyed.header_comments = {"extra"};
  EXPECT_EQ(planetmodel::WriteDeckText(one, keyed),
            "t\n# extra\n         0.000               1\n         1.000               2\n");
  EXPECT_THROW(static_cast<void>(planetmodel::WriteDeckText(one, planetmodel::MINEOS())),
               Error);
  for (const auto* bad : {"two words", "-", "a#b", ""}) {
    EXPECT_THROW(static_cast<void>(planetmodel::MaxwellHeader("t", {"core", bad})), Error)
        << bad;
  }
  const auto partial = Deck({0.0, 0.5, 1.0}, {{"a", {1.0, std::nan(""), 2.0}}});
  EXPECT_THROW(static_cast<void>(planetmodel::DeckLayers(partial)), Error);
  const auto fine = Deck({0.0, 0.5, 1.0}, {{"a", {1.0, 1.5, 2.0}}});
  EXPECT_THROW(static_cast<void>(planetmodel::DeckLayers(fine, "pchip")), Error);
  EXPECT_THROW(static_cast<void>(planetmodel::DeckLayers(fine, "quintic")), Error);
  EXPECT_THROW(static_cast<void>(planetmodel::ReadDeck("/nonexistent", two)), Error);
}

TEST(Deck, AParabolaIsAccurateBesideItsKnots) {
  // a three-knot layer under the cubic kind, read a hair from a knot
  const auto layers = planetmodel::DeckLayers(
      Deck({0.0, 0.3, 1.0}, {{"a", {1.0, 2.0, 0.5}}}), "cubic");
  const auto& f = layers.layers[0].at("a");
  const auto x = std::nextafter(0.3, 1.0);
  // p(x) = 1 + (x - 0)(d01 + (x - 0.3) d012), d01 = 10/3, d012 = (-15/7 - 10/3) / 1
  const auto d01 = 1.0 / 0.3;
  const auto d012 = (-1.5 / 0.7 - d01) / 1.0;
  EXPECT_NEAR(f.Derivative(x), d01 + (x + (x - 0.3)) * d012, 1e-12);
  EXPECT_NEAR(f.Value(0.3), 2.0, 1e-15);
}

TEST(Deck, ARadialFunctionHoldsToItsLayer) {
  const auto layers = planetmodel::DeckLayers(Deck({0.0, 0.5, 1.0}, {{"a", {1.0, 1.5, 2.0}}}));
  const auto& f = layers.layers[0].at("a");
  EXPECT_DOUBLE_EQ(f.Value(1.0 + 1e-12), 2.0);
  EXPECT_THROW(static_cast<void>(f.Value(1.0 + 1e-6)), Error);
  EXPECT_THROW(static_cast<void>(f.Value(std::nan(""))), Error);
  const auto copy = f;
  EXPECT_DOUBLE_EQ(copy.Derivative(0.25), 1.0);
}
