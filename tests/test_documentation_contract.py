"""Documentation and navigation integrity tests."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

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
    "docs/CONTRIBUTING.md",
}


def markdown_files() -> list[Path]:
    """Return every public Markdown file outside Git internals."""

    return [
        path
        for path in sorted(ROOT.rglob("*.md"))
        if ".git" not in path.parts
    ]


def test_key_navigation_files_exist() -> None:
    """Every public review path must resolve to a committed file."""

    missing = sorted(
        relative
        for relative in KEY_NAVIGATION_FILES
        if not (ROOT / relative).is_file()
    )
    assert missing == []


def test_markdown_has_no_control_characters_or_tabs() -> None:
    """Rendered documentation must not contain damaged escape characters."""

    failures: list[str] = []

    for path in markdown_files():
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


def test_all_relative_markdown_links_resolve() -> None:
    """Every local Markdown link must resolve from its source document."""

    missing: list[str] = []
    pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")

    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue

            path_text = unquote(target.split("#", 1)[0])
            if not path_text:
                continue

            resolved = (path.parent / path_text).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                missing.append(
                    f"{path.relative_to(ROOT)} -> {target} (outside root)"
                )
                continue

            if not resolved.exists():
                missing.append(
                    f"{path.relative_to(ROOT)} -> {target} (missing)"
                )

    assert missing == []


def test_reviewer_code_search_anchors_exist() -> None:
    """The public screen-share route must point to real code anchors."""

    guide = (ROOT / "docs/REVIEWER_GUIDE.md").read_text(
        encoding="utf-8"
    )
    model = (ROOT / "nba_win_probability.py").read_text(
        encoding="utf-8"
    )
    anchors = {
        "build_sequential_features",
        "feature_values",
        "make_model",
        "component_predictions",
        "ensemble_probability",
    }

    for anchor in anchors:
        assert anchor in guide
        assert f"def {anchor}(" in model
