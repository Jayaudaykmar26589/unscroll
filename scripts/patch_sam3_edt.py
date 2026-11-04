"""One-shot: add cv2 EDT fallback to vendored sam3 (Windows, no triton)."""
from pathlib import Path

P = Path("external/sam3/sam3/model/edt.py")


def main():
    t = P.read_text()
    old_stub = (
        "except ImportError:\n"
        "    triton, tl = None, None\n"
        "    _HAS_TRITON = False  # Windows: CPU fallback (VENDORED PATCH)\n"
    )
    new_stub = (
        "except ImportError:\n"
        "    _HAS_TRITON = False  # Windows: CPU fallback (VENDORED PATCH)\n"
        "\n"
        "    class _TritonStub:\n"
        "        def __getattr__(self, name):\n"
        "            if name == 'jit':\n"
        "                return lambda fn: fn\n"
        "            raise RuntimeError('triton unavailable on Windows')\n"
        "\n"
        "    class _TlStub:\n"
        "        def __getattr__(self, name):\n"
        "            return object\n"
        "\n"
        "    triton, tl = _TritonStub(), _TlStub()\n"
    )
    assert old_stub in t, "stub anchor missing"
    t = t.replace(old_stub, new_stub)
    old_fn = (
        "    assert data.dim() == 3\n"
        "    assert data.is_cuda\n"
        "    B, H, W = data.shape\n"
    )
    new_fn = (
        "    assert data.dim() == 3\n"
        "    if not _HAS_TRITON:\n"
        "        return _edt_cv2(data)\n"
        "    assert data.is_cuda\n"
        "    B, H, W = data.shape\n"
    )
    assert old_fn in t, "fn anchor missing"
    t = t.replace(old_fn, new_fn)
    helper = (
        "\n\ndef _edt_cv2(data):\n"
        '    """Exact CPU fallback (VENDORED PATCH)."""\n'
        "    import cv2\n"
        "    import numpy as np\n"
        "    outs = []\n"
        "    for b in range(data.shape[0]):\n"
        "        src = (data[b].detach().cpu().numpy() > 0).astype(np.uint8)\n"
        "        outs.append(cv2.distanceTransform(src, cv2.DIST_L2, 0))\n"
        "    import torch\n"
        "    return torch.from_numpy(np.stack(outs)).to(\n"
        "        data.device, torch.float32)\n"
    )
    P.write_text(t.rstrip() + "\n" + helper)
    print("edt fallback patch ok")


if __name__ == "__main__":
    main()
