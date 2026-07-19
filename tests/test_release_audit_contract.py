"""Final GitHub rendering, dependency-lock, and research-label contracts."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def markdown_files() -> list[Path]:
    """Return public Markdown files outside Git internals."""

    return [
        path
        for path in sorted(ROOT.rglob("*.md"))
        if ".git" not in path.parts
    ]


def test_markdown_uses_github_supported_math_delimiters() -> None:
    """Legacy LaTeX delimiters render as broken text on GitHub."""

    failures: list[str] = []
    legacy = (r"\[", r"\]", r"\(", r"\)")

    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        for delimiter in legacy:
            if delimiter in text:
                failures.append(
                    f"{path.relative_to(ROOT)} contains {delimiter}"
                )

    assert failures == []


def test_dependabot_version_updates_are_locked() -> None:
    """Routine dependency PRs must not reopen during the frozen submission."""

    config = yaml.safe_load(
        (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
    )
    updates = config["updates"]

    assert {
        entry["package-ecosystem"]
        for entry in updates
    } == {"github-actions", "pip"}

    for entry in updates:
        assert entry["open-pull-requests-limit"] == 0


def test_research_archive_is_unambiguously_historical() -> None:
    """Research files must not contradict the current official ensemble."""

    index = (ROOT / "research/README.md").read_text(encoding="utf-8")
    venue = (
        ROOT / "research/TEAM_SPECIFIC_HOME_EFFECTS.md"
    ).read_text(encoding="utf-8")
    historical_code = (
        ROOT / "research/single_model_governance.py"
    ).read_text(encoding="utf-8")

    combined = "\n".join([index, venue, historical_code]).lower()

    assert "current official model" in combined
    assert "40-component ensemble" in combined
    assert "historical" in combined
    assert "production champion remains" not in combined
    assert "official champion's previously reported" not in combined


def test_public_readme_badge_targets_main() -> None:
    """The release badge must report the branch being submitted."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "badge.svg?branch=main" in readme
