import json

import numpy as np
import pytest
import tifffile

from scripts.flatten_sheet import flatten
from src.flatten.anchor import align_origins
from src.trace.slices import resolve_slice, select_slices


def test_click_selects_physical_z_and_keeps_original_indices(tmp_path):
    for z in (6100, 6101, 6103, 6104):
        (tmp_path / f"{z:05}.tif").touch()
    refs = select_slices(tmp_path, 6101, 2, z0=6100)
    assert [(r.index, r.source_z) for r in refs] == [(1, 6101), (2, 6103)]
    assert refs[0].metadata() == {"z": 1, "source_z": 6101,
                                  "source_file": "06101.tif"}
    with pytest.raises(ValueError, match="not present"):
        select_slices(tmp_path, 6102, 2)


def test_nonnumeric_names_use_explicit_origin(tmp_path):
    for name in ("a.tif", "b.tif", "c.tif"):
        (tmp_path / name).touch()
    refs = select_slices(tmp_path, 101, 10, z0=100)
    assert [(r.index, r.source_z) for r in refs] == [(1, 101), (2, 102)]


def test_flatten_samples_correct_tiffs_after_missing_trace(tmp_path):
    slices, curves = tmp_path / "slices", tmp_path / "curves"
    slices.mkdir()
    curves.mkdir()
    for i in range(4):
        tifffile.imwrite(slices / f"{6100+i:05}.tif",
                         np.full((24, 32), 100 + i * 100, np.uint16))
        rec = {"z": i, "source_z": 6100 + i,
               "source_file": f"{6100+i:05}.tif",
               "points": [] if i == 1 else [[5, 12], [25, 12]]}
        (curves / f"{i:05}.jsonl").write_text(json.dumps(rec))
    volume = flatten(curves, slices, tmp_path / "out")
    assert volume.shape == (3, 512, 9)
    np.testing.assert_array_equal(volume[:, 0, 0], [100, 300, 400])
    meta = json.loads((tmp_path / "out/sheet.json").read_text())
    assert [r["z"] for r in meta["shifts"]] == [0, 2, 3]
    assert meta["missing_z"] == [1]


def test_legacy_local_indices_and_mismatched_provenance(tmp_path):
    paths = [tmp_path / "06100.tif", tmp_path / "06101.tif"]
    assert resolve_slice({"z": 1}, paths) == paths[1]
    for bad in ({"z": 2}, {"z": -1},
                {"z": 1, "source_file": "06100.tif"},
                {"z": 1, "source_z": 6100}):
        with pytest.raises(ValueError):
            resolve_slice(bad, paths)


def test_alignment_corrects_reversed_traversal_without_wrapping():
    x = np.linspace(5, 100, 512)
    first = np.column_stack((x, np.zeros(512)))
    second = first[::-1] + [0, 1]
    aligned, shifts = align_origins([first, second])
    assert shifts == [0, 0]
    np.testing.assert_allclose(aligned[1], first + [0, 1])


def test_tracker_writes_source_identity_when_click_is_not_first_slice(tmp_path, monkeypatch):
    from src.trace import ridge_tracker

    slices = tmp_path / "slices"
    slices.mkdir()
    for i in range(4):
        tifffile.imwrite(slices / f"{6100+i:05}.tif",
                         np.full((20, 20), i, np.uint16))
    seen = []

    def trace(image, *args):
        seen.append(int(image[0, 0]))
        return np.array([[5., 5.], [6., 5.], [7., 5.]])

    monkeypatch.setattr(ridge_tracker, "trace_slice", trace)
    monkeypatch.setattr(ridge_tracker, "ROOT", tmp_path)
    out = tmp_path / "curves"
    assert ridge_tracker.main(["--slices", str(slices), "--click", "5,5,6102",
                               "--window", "2", "--out", str(out)]) == 0
    assert seen == [2, 3]
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["00002.jsonl", "00003.jsonl"]
    rec = json.loads((out / "00002.jsonl").read_text())
    assert rec["z"] == 2 and rec["source_z"] == 6102
    assert rec["source_file"] == "06102.tif"
