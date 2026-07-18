"""Regenerate and validate the core recruiter submission."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def execute(command: list[str], cwd: Path) -> None:
    """Run one command and stop on failure."""

    print("\n" + "=" * 88)
    print(" ".join(command))
    result = subprocess.run(command, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def run(root: Path, data_path: Path) -> None:
    """Regenerate official prices, governance artifacts and validation."""

    output_dir = root / "outputs"
    figure_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    execute(
        [
            sys.executable,
            "nba_win_probability.py",
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
        ],
        root,
    )
    execute(
        [
            sys.executable,
            "enhanced_governance.py",
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--figure-dir",
            str(figure_dir),
        ],
        root,
    )
    execute(
        [
            sys.executable,
            "validate_submission.py",
            "--root",
            str(root),
            "--data",
            str(data_path),
        ],
        root,
    )


def parse_args() -> argparse.Namespace:
    """Define repository and data paths."""

    parser = argparse.ArgumentParser(
        description="Regenerate and validate the core NBA submission."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.root.resolve(), arguments.data.resolve())
