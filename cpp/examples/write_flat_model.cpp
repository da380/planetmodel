// Write a flat model file of fields of one's own: a crust over a mantle in
// a box, a density of depth alone and a viscosity that varies laterally,
// and topography as a vertical stretch, the free surface raised to a ridge
// and the base held.
//
//   planetmodel_write_flat_model out.nc
#include <planetmodel/cartesian/modelfile.hpp>

#include <cmath>
#include <iostream>
#include <numbers>
#include <optional>

int main(int argc, char** argv) {
  namespace pm = planetmodel;
  namespace flat = planetmodel::cartesian;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " out.nc\n";
    return 2;
  }
  try {
    // depths from the free surface down, over a box that is its own period
    const auto domain = flat::Domain::Box({0.0, 400e3}, {0.0, 300e3});
    auto file = flat::ModelFile(pm::Skeleton({0.0, 30e3, 200e3}), domain);
    file.SetName("example box");
    file.SetLayerName(0, "crust");
    file.SetLayerName(1, "mantle");
    file.SetInterfaceName(0, "surface");
    file.SetInterfaceName(1, "moho");

    // one part, both layers on one grid, elements of 25 km or less
    const auto& skeleton = file.GetSkeleton();
    file.AddPart(0, 1, pm::ElementEdges(skeleton, 0, 1, 25e3), 5, flat::Grid(domain, 32, 24));

    // the density, a vocabulary name, of depth alone
    file.AddField("rho", {{0, flat::LayerSource::Depth([](double) { return 2800.0; })},
                          {1, flat::LayerSource::Depth([](double depth) {
                             return 3300.0 + 0.0005 * (depth - 30e3);
                           })}});

    // the viscosity of the mantle, weaker under a strip of the box, read
    // between the grid's nodes by the bicubic rule; the crust is elastic
    constexpr auto two_pi = 2.0 * std::numbers::pi;
    file.AddField("viscosity",
                  {{1, flat::LayerSource::Lateral([=](double depth, double x, double y) {
                      const auto strip = std::cos(two_pi * x / 400e3) * std::cos(two_pi * y / 300e3);
                      return 1e21 * std::exp(depth / 100e3 - 1.5 * strip);
                    })}},
                  {.lateral = pm::LateralRule::Cubic});

    // topography: heights, positive up, of the free surface and of the
    // Moho, its root; the base stays where it is
    auto ridge = [=](double, double x, double) { return 2e3 * std::cos(two_pi * x / 400e3); };
    auto root = [=](double depth, double x, double y) { return -5.0 * ridge(depth, x, y); };
    file.SetStack({0.0, 30e3, 200e3}, flat::Grid(domain, 64, 48),
                  {flat::LayerSource::Lateral(ridge), flat::LayerSource::Lateral(root),
                   std::nullopt});

    flat::WriteModelFile(argv[1], file);
    std::cout << "wrote " << argv[1] << "\n";
  } catch (const pm::Error& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
