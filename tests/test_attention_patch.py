"""The chunked attention shipped in patches/sam3-unscroll.patch must give the same output as full attention.

The function is taken from the patch text itself, so this tests the code we ship, not a copy.
"""
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as torchF  # noqa: E402

PATCH = Path(__file__).resolve().parents[1] / "patches" / "sam3-unscroll.patch"


def _load_chunked_sdpa(budget_bytes):
    text = PATCH.read_text(encoding="utf-8")
    m = re.search(r"^\+def _unscroll_sdpa\(.*?(?=^ )", text, re.S | re.M)
    assert m, "_unscroll_sdpa not found in the patch"
    src = "\n".join(line[1:] for line in m.group(0).splitlines())
    ns = {"torch": torch, "torchF": torchF, "_UNSCROLL_FLASH_OK": False, "_UNSCROLL_ATTN_BYTES": budget_bytes}
    exec(src, ns)
    return ns["_unscroll_sdpa"]


def test_chunked_attention_is_identical_to_full_attention_on_cpu():
    dtype = torch.float32
    g = torch.Generator().manual_seed(0)
    b, h, n, m, d = 1, 8, 1000, 3000, 32            # many queries over a long memory, as in SAM's memory attention
    q, k, v = (torch.randn(b, h, x, d, generator=g, dtype=dtype) for x in (n, m, m))
    budget = 2 * 2 ** 20                              # small budget so the call really splits into chunks
    sdpa = _load_chunked_sdpa(budget)
    rows = budget // (b * h * m * 4)
    assert rows < n                                   # chunking is exercised
    full = torchF.scaled_dot_product_attention(q, k, v)
    out = sdpa(q, k, v)
    assert out.shape == full.shape
    assert torch.equal(out, full)                     # bit-identical, not just close

