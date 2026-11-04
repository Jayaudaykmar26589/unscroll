"""Parse segment .obj vertex bounds -> pick scroll slice window."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "cache" / "scroll1_segments"


def bounds(path):
    mins = [1e18] * 3
    maxs = [-1e18] * 3
    n = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                x, y, z = (float(v) for v in line.split()[1:4])
                mins[0] = min(mins[0], x)
                mins[1] = min(mins[1], y)
                mins[2] = min(mins[2], z)
                maxs[0] = max(maxs[0], x)
                maxs[1] = max(maxs[1], y)
                maxs[2] = max(maxs[2], z)
                n += 1
    return n, mins, maxs


def main():
    for obj in sorted(D.glob("*.obj")):
        n, lo, hi = bounds(obj)
        print(f"{obj.name}: verts={n}")
        print(f"  x [{lo[0]:.0f}, {hi[0]:.0f}] "
              f"y [{lo[1]:.0f}, {hi[1]:.0f}] z [{lo[2]:.0f}, {hi[2]:.0f}]")


if __name__ == "__main__":
    main()
