"""Validate that the public Git repository contains no private or transient files."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PATH_PATTERNS = (
    re.compile(
        r"(PRIVATE|PREP|PLAYBOOK|REHEARSAL|INTERVIEW|JESSICA)",
        re.IGNORECASE,
    ),
    re.compile(r"nba-win-probability-data\.csv$", re.IGNORECASE),
)

FORBIDDEN_PARTS = {
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}

FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
SECRET_PATTERNS = (
    re.compile(r"\bgho_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def tracked_files() -> list[Path]:
    """Return tracked paths when Git is available, otherwise scan the tree."""

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return [
            Path(item.decode("utf-8"))
            for item in result.stdout.split(b"\0")
            if item
        ]

    return [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ]


def main() -> None:
    """Fail on private materials, caches, compiled files, or obvious secrets."""

    paths = tracked_files()
    violations: list[str] = []

    for relative in paths:
        text_path = relative.as_posix()

        if any(pattern.search(text_path) for pattern in FORBIDDEN_PATH_PATTERNS):
            violations.append(f"forbidden public path: {text_path}")

        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            violations.append(f"transient cache/environment tracked: {text_path}")

        if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"compiled Python file tracked: {text_path}")

        absolute = ROOT / relative
        if not absolute.is_file() or absolute.stat().st_size > 2_000_000:
            continue

        try:
            content = absolute.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                violations.append(f"possible credential in: {text_path}")

    if violations:
        details = "\n".join(f"- {item}" for item in sorted(set(violations)))
        raise SystemExit(f"Repository policy failed:\n{details}")

    print(
        "PASS: no private materials, source CSV, caches, compiled files, "
        "or obvious credentials are tracked."
    )


if __name__ == "__main__":
    main()
