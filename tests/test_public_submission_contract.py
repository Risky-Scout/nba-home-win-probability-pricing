"""Final recruiter-facing narrative and metric consistency tests."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = [
    ROOT / "README.md",
    ROOT / "SUMMARY.md",
    ROOT / "docs/REVIEWER_GUIDE.md",
    ROOT / "docs/ENSEMBLE_METHOD.md",
    ROOT / "docs/MODEL_CARD.md",
    ROOT / "docs/MODEL_EVOLUTION.md",
    ROOT / "docs/LIMITATIONS_AND_ROADMAP.md",
    ROOT / "docs/REPRODUCIBILITY.md",
    ROOT / "docs/ARTIFACT_MANIFEST.md",
]


def public_text() -> str:
    """Combine the recruiter-facing documentation for consistency checks."""

    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in PUBLIC_DOCS
    )


def test_public_version_and_official_model_are_consistent() -> None:
    """Package metadata and public narrative must agree on v1.4."""

    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    selected = json.loads(
        (ROOT / "outputs/selected_model.json").read_text(
            encoding="utf-8"
        )
    )
    narrative = public_text().lower()

    assert project["version"] == "1.4.0"
    assert (
        selected["model"]
        == "uniform_40_component_logistic_ensemble"
    )
    assert selected["component_count"] == 40
    assert selected["component_weight"] == 0.025
    assert "uniform 40-component" in narrative
    assert "v1.4" in narrative


def test_public_narrative_contains_no_stale_official_model_claim() -> None:
    """The prior single champion must be labelled only as a benchmark."""

    narrative = public_text().lower()
    stale_claims = {
        "the selected model remains a strongly regularized three-signal",
        "the single model remains selected",
        "official three-signal champion",
        "enhanced_governance.py",
    }

    for claim in stale_claims:
        assert claim not in narrative

    assert "single benchmark" in narrative
    assert "not statistically decisive" in narrative


def test_readme_and_summary_metrics_match_committed_artifacts() -> None:
    """Published headline metrics must be generated from committed tables."""

    validation = pd.read_csv(
        ROOT / "outputs/ensemble_validation_metrics.csv"
    )
    march = pd.read_csv(
        ROOT / "outputs/march_temporal_check_metrics.csv"
    )
    april = pd.read_csv(
        ROOT / "outputs/april_descriptive_metrics.csv"
    )
    benchmark = pd.read_csv(
        ROOT / "outputs/single_model_benchmark_april_metrics.csv"
    )

    ensemble_name = "uniform_40_component_logistic_ensemble"
    single_name = "validation_best_single_component"

    values = {
        validation.loc[
            validation["model"] == ensemble_name,
            "log_loss",
        ].item(),
        validation.loc[
            validation["model"] == single_name,
            "log_loss",
        ].item(),
        march.loc[
            march["model"] == ensemble_name,
            "log_loss",
        ].item(),
        march.loc[
            march["model"] == single_name,
            "log_loss",
        ].item(),
        april["log_loss"].item(),
        april["brier_score"].item(),
        benchmark["log_loss"].item(),
        benchmark["brier_score"].item(),
    }

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    summary = (ROOT / "SUMMARY.md").read_text(encoding="utf-8")

    for value in values:
        formatted = f"{value:.6f}"
        assert formatted in readme
        assert formatted in summary


def test_public_documents_have_no_placeholders_or_personal_paths() -> None:
    """Recruiter-facing files must not expose local paths or draft markers."""

    forbidden = [
        re.compile(r"/Users/", re.IGNORECASE),
        re.compile(r"Submission July 18", re.IGNORECASE),
        re.compile(r"\[(?:INSERT|YOUR NAME|REPO URL|URL)\]", re.IGNORECASE),
        re.compile(r"\b(?:TODO|FIXME|CHANGEME|PLACEHOLDER)\b", re.IGNORECASE),
        re.compile(r"JESSICA", re.IGNORECASE),
        re.compile(r"PRIVATE_PREP", re.IGNORECASE),
    ]

    failures: list[str] = []
    for path in PUBLIC_DOCS:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern.search(text):
                failures.append(
                    f"{path.relative_to(ROOT)}: {pattern.pattern}"
                )

    assert failures == []


def test_public_commands_reference_existing_entry_points() -> None:
    """Every documented public Python entry point must exist."""

    narrative = public_text()
    documented = set(
        re.findall(
            r"python ([A-Za-z0-9_./-]+\.py)\b",
            narrative,
        )
    )

    missing = sorted(
        path
        for path in documented
        if not (ROOT / path).is_file()
    )
    assert missing == []
