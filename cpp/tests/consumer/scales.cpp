// Uses the parts of the library that need no netCDF-C.
#include <iostream>

#include "planetmodel/units.hpp"
#include "planetmodel/version.hpp"

// the version's integers come with the target, fetched or installed
#if !PLANETMODEL_VERSION_AT_LEAST(1, 0, 0)
#error "planetmodel's version is not 1.0.0 or later, or its integers are missing"
#endif

int main() {
  const auto scales = planetmodel::Scales::Geophysical(6371e3);
  std::cout << "planetmodel " << planetmodel::Version() << " (" << PLANETMODEL_VERSION_MAJOR << "."
            << PLANETMODEL_VERSION_MINOR << "." << PLANETMODEL_VERSION_PATCH << "): G = "
            << planetmodel::G_SI / scales.Factor(planetmodel::units::GRAVITATIONAL_CONSTANT)
            << '\n';
  return 0;
}
