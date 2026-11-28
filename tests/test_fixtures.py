"""The fixtures the C++ readers are held to are written and validate."""
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

SCRIPT = Path(__file__).resolve().parents[1] / "cpp" / "tests" / "make_fixtures.py"


def test_the_fixtures_are_written_and_validate(tmp_path):
    spec = importlib.util.spec_from_file_location("make_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = module.write_fixtures(tmp_path, meshes=False)
    names = {p.name for p in written}
    assert {"model_prem.nc", "model_parts.nc", "model_stacked.nc",
            "model_flattened.nc", "model_radial.nc", "case_radial_polynomial.nc",
            "case_lateral_driscoll_healy_cubic.nc"} <= names
    from planetmodel.modelfile import read_model, read_model_file
    assert read_model(tmp_path / "model_parts.nc").nlayers == 13
    radial = read_model_file(tmp_path / "model_radial.nc")
    assert radial.behaviours == ("Elastic", "Tidal", "SelfGravitating")
    assert radial.parts[0].grid.ntheta == radial.parts[0].grid.nphi == 1
    with netCDF4.Dataset(tmp_path / "case_lateral_custom_offset_cubic.nc") as ds:
        assert ds.method == "cubic"
        assert ds.variables["expected"].shape == ds.variables["query_colatitude"].shape
    assert {"constants.txt", "json_corners.json", "json_corners.json.walk"} <= names
    assert {"fault_static.nc", "fault_afterslip.nc", "fault_cartesian.nc",
            "fault_static.expect", "bad.txt"} <= names
    assert {"surface_basic.nc", "surface_basic.expect", "surface_poles.nc",
            "surface_poles.expect"} <= names
    bad = (tmp_path / "fault_cases" / "bad.txt").read_text().splitlines()
    assert len(bad) > 30 and all(line.count("\t") == 2 for line in bad)
    assert not (tmp_path / "manifests").exists()
    assert not (tmp_path / "fault_meshes").exists()
    # the flat fixtures, of planetmodel.cartesian
    assert {"flat_model_layered.nc", "flat_model_layered_rectangle.nc",
            "flat_model_lateral.nc", "flat_model_rectangle.nc", "flat_model_fourier.nc",
            "flat_model_stacked.nc", "flat_model_stretched.nc",
            "flat_model_surfaces.nc", "flat_model_held.nc", "flat_model_log.nc",
            "flat_model_depth.nc",
            "flat_model_lateral.expect", "flat_case_lateral_box_cubic.nc",
            "flat_case_lateral_rectangle_linear.nc", "flat_surface_box.nc",
            "flat_surface_rectangle.expect"} <= names
    assert "flat_model_fourier.expect" not in names
    from planetmodel import cartesian
    flat = read_model(tmp_path / "flat_model_held.nc")
    assert isinstance(flat, cartesian.Model) and flat.nlayers == 5
    assert (tmp_path / "flat_manifest_cases" / "base.json").exists()
    assert not (tmp_path / "flat_manifests").exists()
    assert not (tmp_path / "flat_fault_meshes").exists()


@pytest.mark.gmsh
@pytest.mark.mfem
def test_the_manifests_are_written_with_their_walks(tmp_path):
    """The manifests the C++ reader is held to, one per way the Python
    writes one, each valid and walked."""
    pytest.importorskip("gmsh")
    pytest.importorskip("mfem.ser")
    from planetmodel.mesh3d import manifest
    spec = importlib.util.spec_from_file_location("make_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = module.manifests(tmp_path)
    manifests = [p for p in written if p.suffix == ".json"]
    assert {p.parent.name for p in manifests} == {
        "mesher", "mesher_refined", "mesh_only", "hollow_shell", "referential",
        "physical", "nondimensional"}
    for path in manifests:
        manifest.read(path)          # which validates
        assert path.with_name(path.name + ".walk").stat().st_size > 0


@pytest.mark.mfem
def test_the_manifests_of_cubed_meshes_are_written_with_their_walks(tmp_path):
    """The manifests of cubed meshes the C++ reader is held to, a cubed
    sphere's and a squared circle's, each valid and walked; gmsh is not
    needed."""
    pytest.importorskip("mfem.ser")
    from planetmodel.mesh3d import manifest
    spec = importlib.util.spec_from_file_location("make_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = module.cubed_manifests(tmp_path)
    manifests = [p for p in written if p.suffix == ".json"]
    assert {p.parent.name for p in manifests} == {"cubed_sphere", "squared_circle"}
    for path in manifests:
        manifest.read(path)          # which validates
        assert path.with_name(path.name + ".walk").stat().st_size > 0


@pytest.mark.gmsh
def test_the_fault_meshes_are_written_with_their_answers(tmp_path):
    """The meshes with faults the C++ rules are held to: each manifest
    valid, a face of each fault with its two cells and its plus one, the
    curves the fault's record names, the faults it ends on, and its walls
    (its hanging wall the cells of its collar attributes)."""
    pytest.importorskip("gmsh")
    from planetmodel.mesh3d import manifest
    spec = importlib.util.spec_from_file_location("make_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    written = module.fault_meshes(tmp_path)
    assert {p.name for p in written} == {
        f"{case}{suffix}" for case in ("surface", "splay", "crossing")
        for suffix in (".json", ".mesh.txt", ".expect")}
    for case, kinds in (("surface", {"tip", "free_edge"}),
                        ("splay", {"tip", "junction", "junction_end"}),
                        ("crossing", {"tip", "junction"})):
        root = tmp_path / "fault_meshes" / case
        card = manifest.read(root / f"{case}.json")
        lines = (root / f"{case}.expect").read_text().splitlines()
        faces = {int(line.split()[1]) for line in lines if line.startswith("pair ")}
        boundary = [line.split() for line in (root / f"{case}.mesh.txt").read_text()
                    .splitlines() if line.startswith("boundary ")]
        on_faults = {i for i, b in enumerate(boundary)
                     if int(b[4]) in {f["attribute"] for f in card.faults}}
        assert faces == on_faults and faces
        edges = {(line.split()[1], line.split()[2]) for line in lines
                 if line.startswith("edge ")}
        assert {kind for _, kind in edges} == kinds
        for record in card.faults:
            for kind in ("tip", "free_edge", "junction"):
                assert ((record["name"], kind) in edges) == (record[kind] is not None)
        ends = {tuple(line.split()[1:]) for line in lines if line.startswith("ends ")}
        assert ends == {(r["name"], o) for r in card.faults for o in r["ends_on"]}
        assert ends == ({("splay", "main")} if case == "splay" else set())
        cells = [line.split() for line in (root / f"{case}.mesh.txt").read_text()
                 .splitlines() if line.startswith("cell ")]
        for record in card.faults:
            hanging = {int(line.split()[2]) for line in lines
                       if line.startswith(f"wall {record['name']} ")
                       and line.split()[3] == "1"}
            assert hanging == {i for i, c in enumerate(cells)
                               if int(c[5]) in record["hanging_wall"]} and hanging


@pytest.mark.gmsh
def test_the_python_refuses_the_manifests_the_cpp_refuses(tmp_path):
    """Both readers agree on what is malformed: the Python reads the base
    and the accepted variants and refuses every bad one."""
    pytest.importorskip("gmsh")
    from planetmodel.mesh3d import manifest
    spec = importlib.util.spec_from_file_location("make_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.manifest_cases(tmp_path)
    root = tmp_path / "manifest_cases"
    for path in [root / "base.json", *sorted(root.glob("accept_*.json"))]:
        manifest.read(path)
    bad = [line.split("\t")[0] for line in
           (root / "bad.txt").read_text().splitlines()]
    assert len(bad) == len(list(root.glob("bad_*.json"))) > 30
    for label in bad:
        with pytest.raises(ValueError):
            manifest.read(root / f"bad_{label}.json")


def test_the_python_refuses_the_model_files_the_cpp_refuses(tmp_path):
    """Both readers agree on what is malformed: every bad copy is refused
    by the Python's read_model_file, which validates the structure and
    checks the nodes, and every expectation file is written."""
    from planetmodel.modelfile import read_model_file, validate_structure
    spec = importlib.util.spec_from_file_location("make_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write_fixtures(tmp_path, meshes=False)
    cases = [line.split("\t") for line in
             (tmp_path / "model_cases" / "bad.txt").read_text().splitlines()]
    bad = [label for label, _, _ in cases]
    assert len(bad) > 15
    for label in bad:
        with pytest.raises(ValueError):
            read_model_file(tmp_path / "model_cases" / f"bad_{label}.nc")
    # the polar values a grid lacking a pole asks for, the surfaces of a
    # stack and the layer bounds: refused at the C++'s place, in its words
    held = [case for case in cases if "poles" in case[0]
            or case[0].startswith(("stack_radius", "stack_one", "layer_bounds_",
                                   "no_layers"))]
    assert len(held) >= 16
    for label, where, phrase in held:
        with pytest.raises(ValueError) as refusal:
            validate_structure(tmp_path / "model_cases" / f"bad_{label}.nc")
        assert f": {where} " in str(refusal.value), label
        assert phrase in str(refusal.value), label
    # a latitude beyond a pole, in the grid of a part, of the stack of
    # surfaces and of the fields on interfaces
    beyond = [case for case in cases if case[0].endswith("latitude_range")]
    assert {where for _, where, _ in beyond} == {"part_0", "geometry", "surfaces"}
    for label, where, phrase in beyond:
        with pytest.raises(ValueError) as refusal:
            read_model_file(tmp_path / "model_cases" / f"bad_{label}.nc")
        assert str(refusal.value).endswith(f": {where}: {phrase}"), label
        assert phrase.endswith("latitudes must lie in [-90, 90]")
    validate_structure(tmp_path / "model_harmonic.nc")
    expected = {p.stem for p in tmp_path.glob("model_*.expect")}
    assert {"model_kinds", "model_tensors", "model_stacked",
            "model_flattened"} <= expected
    assert "model_harmonic" not in expected


def _held_to_the_words(root, read, *, also=frozenset()):
    """A stamp that is a number, dimensions that are not four whole numbers
    and, of `also`, a latitude beyond a pole: each refused by `read` at the
    place and in the words the index of `root` gives, the C++'s."""
    cases = {label: (where, phrase) for label, where, phrase in (
        line.split("\t") for line in (root / "bad.txt").read_text().splitlines())}
    held = {"source_number", "created_number", "dimensions_three",
            "dimensions_text", "dimensions_fraction"} | set(also)
    assert held <= set(cases)
    for label in sorted(held):
        where, phrase = cases[label]
        with pytest.raises(ValueError) as refusal:
            read(root / f"bad_{label}.nc")
        path = root / f"bad_{label}.nc"
        assert str(refusal.value).endswith(
            f"{path}: {where} {phrase}" if where else f"{path}: {phrase}"), label


def test_the_python_refuses_the_surface_files_the_cpp_refuses(tmp_path):
    """Both readers agree on what is malformed in a surface file: every bad
    copy is refused by the Python's read_surface_file, which reads the files
    they are copies of, one on a grid lacking its poles and one on a grid
    holding both; and each file's expectations hold its fields as functions."""
    from planetmodel.surfacefile import read_surface_file
    spec = importlib.util.spec_from_file_location("make_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.surface_files(tmp_path)
    assert not all(read_surface_file(tmp_path / "surface_basic.nc").grid.pole_nodes)
    assert all(read_surface_file(tmp_path / "surface_poles.nc").grid.pole_nodes)
    for name, least in (("surface_basic", 30), ("surface_poles", 20)):
        lines = (tmp_path / f"{name}.expect").read_text().splitlines()
        assert sum(line.startswith("function ") for line in lines) >= least
    root = tmp_path / "surface_cases"
    bad = [line.split("\t")[0] for line in (root / "bad.txt").read_text().splitlines()]
    assert len(bad) == len(list(root.glob("bad_*.nc"))) > 2
    assert {"poles_on_both", "pole_length"} <= set(bad)
    for label in bad:
        with pytest.raises(ValueError):
            read_surface_file(root / f"bad_{label}.nc")
    _held_to_the_words(root, read_surface_file, also={"latitude_range"})


def test_the_python_refuses_the_flat_files_the_cpp_refuses(tmp_path):
    """Both readers agree on what is malformed in a flat file: every bad
    copy of a flat model file and of a flat surface file is refused by the
    Python's flat readers, and every bad flat manifest by `manifest.read`,
    which reads the base and the accepted variants."""
    from planetmodel.cartesian.modelfile import read_model_file
    from planetmodel.cartesian.surfacefile import read_surface_file
    from planetmodel.mesh3d import manifest
    spec = importlib.util.spec_from_file_location("make_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.flat_model_files(tmp_path)
    module.flat_model_cases(tmp_path)
    module.flat_surface_files(tmp_path)
    module.flat_manifest_cases(tmp_path)
    for folder, read, least in (("flat_model_cases", read_model_file, 30),
                                ("flat_surface_cases", read_surface_file, 10)):
        bad = [line.split("\t")[0] for line in
               (tmp_path / folder / "bad.txt").read_text().splitlines()]
        assert len(bad) > least
        for label in bad:
            with pytest.raises(ValueError):
                read(tmp_path / folder / f"bad_{label}.nc")
    _held_to_the_words(tmp_path / "flat_surface_cases", read_surface_file)
    root = tmp_path / "flat_manifest_cases"
    for path in [root / "base.json", *sorted(root.glob("accept_*.json"))]:
        assert manifest.read(path).is_flat
    bad = [line.split("\t")[0] for line in (root / "bad.txt").read_text().splitlines()]
    assert len(bad) == len(list(root.glob("bad_*.json"))) > 30
    for label in bad:
        with pytest.raises(ValueError):
            manifest.read(root / f"bad_{label}.json")


@pytest.mark.gmsh
@pytest.mark.mfem
def test_the_flat_manifests_and_meshes_are_written_with_their_answers(tmp_path):
    """The flat manifests the C++ reader is held to, each valid, flat and
    walked, and the flat meshes with faults, each fault's curves those its
    record names, its face edge among them."""
    pytest.importorskip("gmsh")
    pytest.importorskip("mfem.ser")
    from planetmodel.mesh3d import manifest
    spec = importlib.util.spec_from_file_location("make_fixtures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifests = [p for p in module.flat_manifests(tmp_path) if p.suffix == ".json"]
    assert {p.parent.name for p in manifests} == {
        "box", "rectangle", "faulted", "referential", "physical_rectangle"}
    for path in manifests:
        card = manifest.read(path)
        assert card.is_flat
        assert path.with_name(path.name + ".walk").stat().st_size > 0
        if path.parent.name in ("referential", "physical_rectangle"):
            assert card.model["schema"] == "planetmodel.model/3"
            assert (path.parent / card.model["file"]).exists()
    written = module.flat_fault_meshes(tmp_path)
    assert {p.name for p in written} == {
        f"{case}{suffix}" for case in ("side", "several")
        for suffix in (".json", ".mesh.txt", ".expect")}
    for case, kinds in (("side", {"tip", "free_edge", "face_edge"}),
                        ("several", {"tip", "free_edge", "face_edge", "junction",
                                     "junction_end"})):
        root = tmp_path / "flat_fault_meshes" / case
        card = manifest.read(root / f"{case}.json")
        lines = (root / f"{case}.expect").read_text().splitlines()
        edges = {(line.split()[1], line.split()[2]) for line in lines
                 if line.startswith("edge ")}
        assert {kind for _, kind in edges} == kinds
        for record in card.faults:
            for kind in manifest.FLAT_FAULT_CURVES:
                assert ((record["name"], kind) in edges) == (record[kind] is not None)
