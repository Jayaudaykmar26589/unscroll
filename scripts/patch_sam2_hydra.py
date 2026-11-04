"""One-shot: lazy hydra in vendored sam2 (Windows, antlr4 4.9 unbuildable)."""
from pathlib import Path

P = Path("external/sam2/sam2/__init__.py")


def main():
    t = P.read_text()
    old = (
        "from hydra import initialize_config_module\n"
        "from hydra.core.global_hydra import GlobalHydra\n"
        "\n"
        "if not GlobalHydra.instance().is_initialized():\n"
        '    initialize_config_module("sam2", version_base="1.2")\n'
    )
    new = (
        "# VENDORED PATCH: hydra optional (from_pretrained path needs none)\n"
        "try:\n"
        "    from hydra import initialize_config_module\n"
        "    from hydra.core.global_hydra import GlobalHydra\n"
        "\n"
        "    if not GlobalHydra.instance().is_initialized():\n"
        '        initialize_config_module("sam2", version_base="1.2")\n'
        "except ImportError:\n"
        "    pass\n"
    )
    assert old in t, "anchor missing"
    P.write_text(t.replace(old, new))
    print("sam2 hydra patch ok")


if __name__ == "__main__":
    main()
