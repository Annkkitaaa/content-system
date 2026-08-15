#!/usr/bin/env python3
"""House style linter.

Right now it enforces one rule, hard: no em dashes anywhere in the repository.

That rule exists for a real reason rather than as taste. The em dash is one of
the strongest surface tells of machine-written text, and this project's whole
purpose is producing content that does not read as machine-written. Banning it
in the source and docs as well as in generated output keeps the rule honest:
the prompts that tell the model to avoid it are themselves checked.

Generated drafts get the same treatment at runtime through the sanitizer, so
a slip in model output never reaches the workbook. This script covers the
files a human or an agent writes.

Run it directly, or through CI:

    python tools/check_style.py
    python tools/check_style.py --fix
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The offending characters are written as escapes, not literals, so this file
# and its tests pass their own check. The rule then holds with no exemptions,
# which matters: a linter that has to whitelist itself invites the next
# whitelist.
EM_DASH = chr(0x2014)
HORIZONTAL_BAR = chr(0x2015)
EN_DASH = chr(0x2013)

#: Characters that are never acceptable, with the replacement used by --fix.
BANNED: dict[str, tuple[str, str]] = {
    EM_DASH: ("em dash", ", "),
    HORIZONTAL_BAR: ("horizontal bar", ", "),
}

#: A spaced en dash is an em dash wearing a disguise. An unspaced one is a
#: legitimate range ("2024-2025"), so only the spaced form is rejected.
SPACED_EN_DASH = f" {EN_DASH} "

TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".cfg",
        ".ini",
        ".sql",
        ".sh",
        ".ps1",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".css",
        ".html",
        ".env",
        ".example",
    }
)

SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "dist",
        "build",
        ".eggs",
        "htmlcov",
        "data",
        "exports",
    }
)


@dataclass(frozen=True)
class Violation:
    path: Path
    line_number: int
    column: int
    description: str
    line: str

    def render(self, root: Path) -> str:
        try:
            shown = self.path.relative_to(root)
        except ValueError:
            shown = self.path
        return (
            f"{shown}:{self.line_number}:{self.column}: {self.description}\n    {self.line.strip()}"
        )


def iter_text_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore", "Dockerfile"}:
            yield path


def scan(path: Path) -> list[Violation]:
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []  # not text we can meaningfully lint

    violations: list[Violation] = []
    for number, line in enumerate(content.splitlines(), start=1):
        for character, (label, _) in BANNED.items():
            column = line.find(character)
            if column >= 0:
                violations.append(
                    Violation(path, number, column + 1, f"{label} is banned in this repo", line)
                )
        column = line.find(SPACED_EN_DASH)
        if column >= 0:
            violations.append(
                Violation(
                    path,
                    number,
                    column + 2,
                    "spaced en dash reads as an em dash, use a comma or a full stop",
                    line,
                )
            )
    return violations


def fix(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = original
    for character, (_, replacement) in BANNED.items():
        updated = updated.replace(character, replacement)
    updated = updated.replace(SPACED_EN_DASH, ", ")
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fix", action="store_true", help="rewrite offending characters in place")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="directory to scan")
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    all_violations: list[Violation] = []
    fixed: list[Path] = []

    for path in iter_text_files(root):
        violations = scan(path)
        if not violations:
            continue
        if args.fix and fix(path):
            fixed.append(path)
        else:
            all_violations.extend(violations)

    if fixed:
        print(f"Fixed {len(fixed)} file(s):")
        for path in fixed:
            print(f"  {path.relative_to(root)}")

    if all_violations:
        print(f"\nFound {len(all_violations)} style violation(s):\n", file=sys.stderr)
        for violation in all_violations:
            print(violation.render(root), file=sys.stderr)
        print("\nRun 'python tools/check_style.py --fix' to rewrite them.", file=sys.stderr)
        return 1

    if not fixed:
        print("Style check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
