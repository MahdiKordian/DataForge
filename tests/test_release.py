# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Release hygiene: the version, license, notices and sample files must agree with each other."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(name):
    return (ROOT / name).read_text(encoding="utf-8")


def _version():
    return re.search(r'^VERSION = "v([^"]+)"', _text("core.py"), re.M).group(1)


def test_version_is_the_same_everywhere():
    v = _version()
    assert f"version-v{v}-" in _text("README.md")
    assert re.search(rf"^## \[{re.escape(v)}\]", _text("CHANGELOG.md"), re.M)


def test_license_files():
    assert _text("LICENSE").startswith("MIT License")
    assert "Permission is hereby granted" in _text("LICENSE")
    assert "Apache" not in _text("LICENSE")
    assert "MIT License" in _text("NOTICE") and "MIT License" in _text("README.md")


def test_no_trace_of_the_old_license_terms():
    needle = "DataForge " + "License"
    for p in ROOT.rglob("*"):
        if p.suffix in {".py", ".md", ".yml", ".toml"} or p.name in {"LICENSE", "NOTICE"}:
            if "__pycache__" in p.parts or p.name == "test_release.py":
                continue
            assert needle not in p.read_text(encoding="utf-8"), str(p.relative_to(ROOT))


def test_every_required_library_is_in_the_notice():
    notice = _text("NOTICE").lower()

    for filename in ("requirements.txt", "requirements-dev.txt"):
        for line in _text(filename).splitlines():
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            m = re.match(r"([A-Za-z0-9_.-]+)", line)

            if m:
                assert m.group(1).lower() in notice, (
                    f"{m.group(1)} is missing from NOTICE"
                )


def test_sample_files_named_in_core_exist():
    names = re.findall(r'"file": "([^"]+\.csv)"', _text("core.py"))
    assert len(names) == 3
    for n in names:
        assert (ROOT / "sample_data" / n).is_file(), n


def test_every_source_file_carries_the_copyright_line():
    missing = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py")
               if "__pycache__" not in p.parts and "Copyright (c) 2026 Mahdi Kordian" not in p.read_text(encoding="utf-8")[:300]]
    assert not missing, missing


def test_no_old_address_or_license_left_in_the_docs():
    for name in ("README.md", "CHANGELOG.md", "NOTICE"):
        assert "dataforge-prep" not in _text(name)
    assert "Apache License 2.0" not in _text("README.md").split("## License")[0]
