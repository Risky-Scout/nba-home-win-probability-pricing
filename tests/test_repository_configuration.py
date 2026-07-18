"""Configuration tests for packaging and GitHub Actions."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, object]:
    """Parse YAML syntax while preserving words such as `on` as strings."""

    parsed = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(parsed, dict)
    return parsed


def test_pyproject_declares_supported_python_and_exact_dependencies() -> None:
    """Package metadata must reject Python 3.14 and retain locked direct deps."""

    metadata = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = metadata["project"]

    assert project["requires-python"] == ">=3.11,<3.14"
    assert set(project["dependencies"]) == {
        "numpy==2.3.5",
        "pandas==2.2.3",
        "scikit-learn==1.8.0",
        "matplotlib==3.10.8",
    }


def test_github_actions_yaml_is_valid_and_complete() -> None:
    """CI must cover every supported Python version without fail-fast."""

    workflow = load_yaml(ROOT / ".github/workflows/tests.yml")
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"][
        "python-version"
    ]

    assert matrix == ["3.11", "3.12", "3.13"]
    assert workflow["jobs"]["test"]["strategy"]["fail-fast"] == "false"

    workflow_text = (
        ROOT / ".github/workflows/tests.yml"
    ).read_text(encoding="utf-8")
    assert "actions/checkout@v7.0.0" in workflow_text
    assert "actions/setup-python@v6.3.0" in workflow_text
    assert "python -m pytest -q" in workflow_text


def test_dependabot_yaml_is_valid() -> None:
    """Dependency maintenance configuration must parse and cover both ecosystems."""

    config = load_yaml(ROOT / ".github/dependabot.yml")
    ecosystems = {
        update["package-ecosystem"]
        for update in config["updates"]
    }

    assert ecosystems == {"github-actions", "pip"}
