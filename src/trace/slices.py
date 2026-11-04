"""Explicit mapping between video frames, source TIFF indices and physical z.

Numeric TIFF stems are physical z IDs. For other names, --z0 is the physical
z of the first sorted TIFF. Curve `z` remains the index in that complete list
for compatibility with older curve files; source_file/source_z remove ambiguity.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SliceRef:
    index: int
    source_z: int
    path: Path

    def metadata(self):
        return {"z": self.index, "source_z": self.source_z,
                "source_file": self.path.name}


def select_slices(directory, click_z, window, z0=0):
    paths = sorted(Path(directory).glob("*.tif*"))
    if not paths or window <= 0:
        raise ValueError("need TIFF slices and a positive window")
    numeric = all(p.stem.isdecimal() for p in paths)
    refs = [SliceRef(i, int(p.stem) if numeric else z0 + i, p)
            for i, p in enumerate(paths)]
    if len({r.source_z for r in refs}) != len(refs):
        raise ValueError("duplicate source slice IDs")
    starts = [i for i, r in enumerate(refs) if r.source_z == click_z]
    if not starts:
        raise ValueError(f"click z={click_z} is not present in {directory}")
    return refs[starts[0]:starts[0] + window]


def resolve_slice(record, paths):
    """Resolve a curve to its source, including legacy z-only records."""
    index = record["z"]
    if not isinstance(index, int) or not 0 <= index < len(paths):
        raise ValueError(f"curve slice index out of range: {index}")
    path = paths[index]
    if record.get("source_file", path.name) != path.name:
        raise ValueError(f"source_file disagrees with slice index {index}")
    if path.stem.isdecimal() and "source_z" in record:
        if int(path.stem) != record["source_z"]:
            raise ValueError(f"source_z disagrees with {path.name}")
    return path
