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
            "model_flattened.nc", "case_radial_polynomial.nc",
            "case_lateral_driscoll_healy_cubic.nc"} <= names
    from planetmodel.modelfile import read_model
    assert read_model(tmp_path / "model_parts.nc").nlayers == 13
    with netCDF4.Dataset(tmp_path / "case_lateral_custom_offset_cubic.nc") as ds:
        assert ds.method == "cubic"
        assert ds.variables["expected"].shape == ds.variables["query_colatitude"].shape
    assert {"constants.txt", "json_corners.json", "json_corners.json.walk"} <= names
    assert {"fault_static.nc", "fault_afterslip.nc", "fault_cartesian.nc",
            "fault_static.expect", "bad.txt"} <= names
    bad = (tmp_path / "fault_cases" / "bad.txt").read_text().splitlines()
    assert len(bad) > 30 and all(line.count("\t") == 2 for line in bad)
    assert not (tmp_path / "manifests").exists()
    assert not (tmp_path / "fault_meshes").exists()


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
    bad = [line.split("\t")[0] for line in
           (tmp_path / "model_cases" / "bad.txt").read_text().splitlines()]
    assert len(bad) > 15
    for label in bad:
        with pytest.raises(ValueError):
            read_model_file(tmp_path / "model_cases" / f"bad_{label}.nc")
    validate_structure(tmp_path / "model_harmonic.nc")
    expected = {p.stem for p in tmp_path.glob("model_*.expect")}
    assert {"model_kinds", "model_tensors", "model_stacked",
            "model_flattened"} <= expected
    assert "model_harmonic" not in expected
