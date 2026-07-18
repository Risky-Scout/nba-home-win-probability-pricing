"""Documentation and navigation integrity tests."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KEY_NAVIGATION_FILES = {
    "README.md",
    "SUMMARY.md",
    "docs/REVIEWER_GUIDE.md",
    "docs/ENSEMBLE_METHOD.md",
    "docs/MODEL_EVOLUTION.md",
    "docs/MODEL_CARD.md",
    "docs/LIMITATIONS_AND_ROADMAP.md",
    "docs/REPRODUCIBILITY.md",
    "docs/ARTIFACT_MANIFEST.md",
}


def test_key_navigation_files_exist() -> None:
    """Every public review path must resolve to a committed file."""

    missing = sorted(
        relative
        for relative in KEY_NAVIGATION_FILES
        if not (ROOT / relative).is_file()
    )
    assert missing == []


def test_markdown_has_no_control_characters_or_tabs() -> None:
    """Rendered documentation must not contain damaged LaTeX escapes."""

    failures: list[str] = []

    for path in sorted(ROOT.rglob("*.md")):
        if ".git" in path.parts:
            continue

        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            bad = [
                character
                for character in line
                if ord(character) < 32
            ]
            if bad:
                codes = ", ".join(
                    f"U+{ord(character):04X}"
                    for character in bad
                )
                failures.append(
                    f"{path.relative_to(ROOT)}:{line_number}: {codes}"
                )

    assert failures == []


def test_readme_relative_links_resolve() -> None:
    """Local file links on the main landing page must exist."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
    missing: list[str] = []

    for target in targets:
        if target.startswith(("http://", "https://", "#")):
            continue

        path_text = target.split("#", 1)[0]
        if not path_text:
            continue

        if not (ROOT / path_text).exists():
            missing.append(target)

    assert missing == []
