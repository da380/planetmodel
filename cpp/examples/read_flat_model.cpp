// Read a flat model file: its domain, what it holds on each layer, every
// scalar field's value and gradient at a point of its layer, the height
// the vertical stretch raises that point by, and every field on an
// interface at a point of it.
//
//   planetmodel_read_flat_model model.nc
#include <planetmodel/cartesian/modelfile.hpp>

#include <format>
#include <iostream>

int main(int argc, char** argv) {
  namespace pm = planetmodel;
  namespace flat = planetmodel::cartesian;
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " model.nc\n";
    return 2;
  }
  try {
    const auto file = flat::ReadModelFile(argv[1]);
    const auto& domain = file.GetDomain();
    std::cout << std::format("{}: {}, schema {}, a {} of period {} m along x\n", file.Source(),
                             file.Name().value_or("unnamed"), file.Schema(), domain.Kind(),
                             domain.Length(0) * file.GetScales().Length());
    const auto mapping = file.Mapping();
    const auto& skeleton = file.GetSkeleton();
    // a lateral place inside the domain; y is not read in a section
    const auto x = 0.5 * (domain.Bounds(0)[0] + domain.Bounds(0)[1]) + 0.1 * domain.Length(0);
    const auto y = domain.IsBox() ? domain.Bounds(1)[0] + 0.3 * domain.Length(1) : 0.0;
    for (auto layer = 0; layer < static_cast<int>(skeleton.NLayers()); ++layer) {
      const auto [top, bottom] = skeleton.Interval(static_cast<std::size_t>(layer));
      const auto name = file.LayerNames()[static_cast<std::size_t>(layer)].value_or("-");
      std::cout << std::format("layer {} ({}), depth in [{}, {}]:\n", layer, name, top, bottom);
      const auto depth = 0.5 * (top + bottom);
      for (const auto& field_name : file.Names(layer)) {
        const auto field = file.FieldOn(layer, field_name);
        if (field.GetCharacter().Rank() != 0) {
          std::cout << std::format("  {}: rank {}, {} components\n", field_name,
                                   field.GetCharacter().Rank(), field.Size());
          continue;
        }
        const auto scalar = field.AsScalar();
        const auto [ddepth, dx, dy] = scalar.Gradient(depth, x, y);
        std::cout << std::format("  {}{}: {:.6g}, gradient ({:.4g}, {:.4g}, {:.4g})\n",
                                 field_name, field.IsRadial() ? " (of depth)" : "",
                                 scalar.Value(depth, x, y), ddepth, dx, dy);
      }
      // the point as (x, y, z), z up, and where the mapping puts it
      const auto moved = mapping({x, y, -depth});
      std::cout << std::format("  the point at depth {} rises by {:.4g} to z = {:.6g}\n", depth,
                               mapping.Height(depth, x, y), moved[2]);
    }
    for (auto k = 0; k < static_cast<int>(file.InterfaceDepths().size()); ++k) {
      const auto name = file.InterfaceNames()[static_cast<std::size_t>(k)].value_or("-");
      for (const auto& field_name : file.SurfaceNames(k)) {
        const auto f = file.SurfaceFieldOn(k, field_name);
        const auto [dx, dy] = f.Gradient(x, y);
        std::cout << std::format("interface {} ({}), {}: {:.6g}, gradient ({:.4g}, {:.4g})\n",
                                 k, name, field_name, f.Value(x, y), dx, dy);
      }
    }
  } catch (const pm::Error& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
