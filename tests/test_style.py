"""The no-em-dash rule, checked against the repository itself.

If this file is the only thing enforcing the rule for source and docs, it has
to actually run in CI, and it has to fail loudly. See tools/check_style.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LINTER = REPO_ROOT / "tools" / "check_style.py"

sys.path.insert(0, str(REPO_ROOT / "tools"))

from check_style import fix, scan  # noqa: E402

# Built with chr() rather than written as literals, so this test file passes
# the very check it exercises. No exemption list needed anywhere.
EM_DASH = chr(0x2014)
HORIZONTAL_BAR = chr(0x2015)
EN_DASH = chr(0x2013)


def test_repository_is_free_of_em_dashes() -> None:
    result = subprocess.run(
        [sys.executable, str(LINTER)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, (
        "House style check failed. Run 'python tools/check_style.py --fix'.\n"
        f"{result.stdout}\n{result.stderr}"
    )


class TestScan:
    def test_flags_an_em_dash(self, tmp_path: Path) -> None:
        offender = tmp_path / "sample.md"
        offender.write_text(f"This is a thought {EM_DASH} and here is the rest.", encoding="utf-8")

        violations = scan(offender)

        assert len(violations) == 1
        assert "em dash" in violations[0].description
        assert violations[0].line_number == 1

    def test_flags_a_horizontal_bar(self, tmp_path: Path) -> None:
        offender = tmp_path / "sample.md"
        offender.write_text(f"a {HORIZONTAL_BAR} b", encoding="utf-8")

        assert any("horizontal bar" in v.description for v in scan(offender))

    def test_flags_a_spaced_en_dash(self, tmp_path: Path) -> None:
        # A spaced en dash is an em dash in disguise.
        offender = tmp_path / "sample.md"
        offender.write_text(f"one thing {EN_DASH} then another", encoding="utf-8")

        assert any("en dash" in v.description for v in scan(offender))

    def test_allows_an_unspaced_en_dash(self, tmp_path: Path) -> None:
        # A numeric range is legitimate and must not be rewritten.
        clean = tmp_path / "sample.md"
        clean.write_text(f"The 2024{EN_DASH}2025 season.", encoding="utf-8")

        assert scan(clean) == []

    def test_allows_ordinary_hyphens_and_flags(self, tmp_path: Path) -> None:
        clean = tmp_path / "sample.md"
        clean.write_text("Run `ruff check --fix` on the well-formed input.", encoding="utf-8")

        assert scan(clean) == []

    def test_reports_every_offending_line(self, tmp_path: Path) -> None:
        offender = tmp_path / "sample.md"
        offender.write_text(
            f"first {EM_DASH} line\nclean line\nthird {EM_DASH} line", encoding="utf-8"
        )

        violations = scan(offender)

        assert [v.line_number for v in violations] == [1, 3]

    def test_ignores_binary_content(self, tmp_path: Path) -> None:
        binary = tmp_path / "blob.py"
        binary.write_bytes(b"\xff\xfe\x00\x01")

        assert scan(binary) == []


class TestFix:
    @pytest.mark.parametrize("bad", [EM_DASH, HORIZONTAL_BAR, f" {EN_DASH} "])
    def test_rewrites_offenders_to_a_comma(self, tmp_path: Path, bad: str) -> None:
        offender = tmp_path / "sample.md"
        offender.write_text(f"one{bad}two", encoding="utf-8")

        assert fix(offender) is True
        assert offender.read_text(encoding="utf-8") == "one, two"
        assert scan(offender) == []

    def test_leaves_clean_files_untouched(self, tmp_path: Path) -> None:
        clean = tmp_path / "sample.md"
        clean.write_text("nothing wrong here", encoding="utf-8")

        assert fix(clean) is False
        assert clean.read_text(encoding="utf-8") == "nothing wrong here"
