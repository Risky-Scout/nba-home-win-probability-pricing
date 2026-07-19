"""Validate the final recruiter-facing ensemble submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

from project_runtime import require_supported_python

require_supported_python()

import numpy as np
import pandas as pd


REQUIRED_FILES = {
    "README.md",
    "SUMMARY.md",
    "pyproject.toml",
    ".python-version",
    ".gitignore",
    "nba_win_probability.py",
    "model_governance.py",
    "challenger_analysis.py",
    "ablation_and_timing.py",
    "run_submission.py",
    "validate_submission.py",
    "project_runtime.py",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-challengers.txt",
    "docs/REVIEWER_GUIDE.md",
    "docs/ENSEMBLE_METHOD.md",
    "docs/MODEL_CARD.md",
    "docs/MODEL_EVOLUTION.md",
    "docs/LIMITATIONS_AND_ROADMAP.md",
    "docs/REPRODUCIBILITY.md",
    "docs/ARTIFACT_MANIFEST.md",
    "docs/CONTRIBUTING.md",
    "scripts/check_python.py",
    "scripts/check_repository_policy.py",
    "scripts/validate_committed_artifacts.py",
    "scripts/run_quality_checks.sh",
    "scripts/bootstrap_macos.sh",
    "tests/test_model_contracts.py",
    "tests/test_runtime_contract.py",
    "tests/test_repository_configuration.py",
    "tests/test_documentation_contract.py",
    "tests/test_ensemble_contract.py",
    "tests/test_public_submission_contract.py",
    "tests/test_release_audit_contract.py",
    ".github/workflows/tests.yml",
    ".github/dependabot.yml",
    "outputs/data_fingerprint.json",
    "outputs/april_predictions.csv",
    "outputs/april_descriptive_metrics.csv",
    "outputs/april_component_dispersion.csv",
    "outputs/april_repricing_backtest.csv",
    "outputs/single_model_benchmark_april_predictions.csv",
    "outputs/single_model_benchmark_april_metrics.csv",
    "outputs/validation_grid.csv",
    "outputs/ensemble_validation_metrics.csv",
    "outputs/march_temporal_check_metrics.csv",
    "outputs/march_calibration_bins.csv",
    "outputs/ensemble_component_summary.csv",
    "outputs/selected_model.json",
    "outputs/model_summary.json",
    "outputs/governance_model_comparison.csv",
    "outputs/governance_candidate_grid.csv",
    "outputs/governance_bootstrap_differences.csv",
    "outputs/governance_monthly_backtest.csv",
    "outputs/governance_march_calibration_summary.csv",
    "outputs/governance_march_calibration_bins.csv",
    "outputs/governance_feature_correlation.csv",
    "outputs/governance_feature_vif.csv",
    "outputs/governance_runtime.csv",
    "outputs/governance_selection_decision.json",
    "figures/validation_model_comparison.png",
    "figures/march_model_comparison.png",
    "figures/march_calibration.png",
    "figures/ensemble_mean_coefficients.png",
    "figures/april_component_dispersion.png",
    "figures/governance_model_comparison.png",
    "figures/monthly_model_stability.png",
    "figures/march_calibration_reliability.png",
}


def fail(message: str) -> None:
    """Raise one concise validation failure."""

    raise AssertionError(message)


def validate_files(root: Path) -> None:
    """Confirm every promised public artifact exists."""

    missing = sorted(
        relative
        for relative in REQUIRED_FILES
        if not (root / relative).is_file()
    )
    if missing:
        fail(f"Missing required files: {missing}")


def validate_package(root: Path) -> None:
    """Confirm Python support and exact direct dependencies."""

    metadata = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = metadata["project"]

    if project["version"] != "1.4.1":
        fail("pyproject version must be 1.4.1.")
    if project["requires-python"] != ">=3.11,<3.14":
        fail("Supported Python range changed.")

    expected = {
        "numpy==2.3.5",
        "pandas==2.2.3",
        "scikit-learn==1.8.0",
        "matplotlib==3.10.8",
    }
    if set(project["dependencies"]) != expected:
        fail("Core direct dependencies changed unexpectedly.")

    if (
        root.joinpath(".python-version")
        .read_text(encoding="utf-8")
        .strip()
        != "3.12.13"
    ):
        fail("Recommended Python must remain 3.12.13.")


def validate_workflow(root: Path) -> None:
    """Confirm CI covers supported runtimes and core quality gates."""

    workflow = (
        root / ".github/workflows/tests.yml"
    ).read_text(encoding="utf-8")
    required = {
        "actions/checkout@v7.0.0",
        "actions/setup-python@v6.3.0",
        '- "3.11"',
        '- "3.12"',
        '- "3.13"',
        "python -m pytest -q",
        "python scripts/check_repository_policy.py",
        "python scripts/validate_committed_artifacts.py",
        "model_governance.py",
        "persist-credentials: false",
    }
    missing = sorted(item for item in required if item not in workflow)
    if missing:
        fail(f"CI workflow is missing controls: {missing}")


def load_source(source_csv: Path) -> pd.DataFrame:
    """Load the source with identifiers preserved."""

    source = pd.read_csv(
        source_csv,
        dtype={"game_id": "string"},
    )
    source["game_date"] = pd.to_datetime(
        source["game_date"],
        errors="raise",
    )
    return source


def validate_source_fingerprint(
    root: Path,
    source_csv: Path,
    source: pd.DataFrame,
) -> None:
    """Confirm committed outputs came from the exact supplied source."""

    fingerprint = json.loads(
        (root / "outputs/data_fingerprint.json").read_text(
            encoding="utf-8"
        )
    )
    digest = hashlib.sha256(source_csv.read_bytes()).hexdigest()

    if fingerprint["sha256"] != digest:
        fail("Data fingerprint does not match the supplied CSV.")
    if fingerprint["rows"] != len(source):
        fail("Fingerprint row count is inconsistent.")
    if len(source) != 1230:
        fail("Expected 1,230 source games.")


def validate_april_predictions(
    root: Path,
    source: pd.DataFrame,
) -> None:
    """Validate official ensemble, benchmark, and fair-odds contracts."""

    expected = source.loc[
        source["game_date"].dt.month == 4,
        ["game_id", "game_date", "away", "home"],
    ].copy()
    expected["game_date"] = expected["game_date"].dt.strftime(
        "%Y-%m-%d"
    )
    expected = expected.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)

    official = pd.read_csv(
        root / "outputs/april_predictions.csv",
        dtype={"game_id": "string"},
    )
    benchmark = pd.read_csv(
        root / "outputs/single_model_benchmark_april_predictions.csv",
        dtype={"game_id": "string"},
    )
    dispersion = pd.read_csv(
        root / "outputs/april_component_dispersion.csv",
        dtype={"game_id": "string"},
    )

    for name, table in (
        ("official", official),
        ("benchmark", benchmark),
        ("dispersion", dispersion),
    ):
        if len(table) != 96:
            fail(f"Expected 96 {name} April rows.")
        if not table["game_id"].is_unique:
            fail(f"{name} April IDs are not unique.")
        if not table["game_id"].str.fullmatch(r"\d{10}").all():
            fail(f"{name} game IDs did not preserve ten characters.")
        if set(table["game_id"]) != set(expected["game_id"]):
            fail(f"{name} game IDs do not match the source.")

    official_key = official[
        ["game_id", "game_date", "away", "home"]
    ].copy()
    official_key["game_date"] = pd.to_datetime(
        official_key["game_date"]
    ).dt.strftime("%Y-%m-%d")
    official_key = official_key.sort_values(
        ["game_date", "game_id"]
    ).reset_index(drop=True)
    if not official_key.equals(expected):
        fail("Official April schedule fields do not match the source.")

    probability = official["home_win_probability"].to_numpy()
    if not np.all((probability > 0.0) & (probability < 1.0)):
        fail("An official probability is outside (0, 1).")

    np.testing.assert_allclose(
        official["fair_home_decimal_odds"],
        1.0 / probability,
        atol=1e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        official["fair_away_decimal_odds"],
        1.0 / (1.0 - probability),
        atol=1e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        dispersion["home_win_probability"],
        probability,
        atol=1e-10,
        rtol=0.0,
    )


def validate_model_selection(root: Path) -> None:
    """Confirm the official ensemble and benchmark metadata agree."""

    selected = json.loads(
        (root / "outputs/selected_model.json").read_text(
            encoding="utf-8"
        )
    )
    governance = json.loads(
        (
            root / "outputs/governance_selection_decision.json"
        ).read_text(encoding="utf-8")
    )

    if (
        selected["model"]
        != "uniform_40_component_logistic_ensemble"
    ):
        fail("Official model is not the 40-component ensemble.")
    if selected["component_count"] != 40:
        fail("Official component count changed.")
    if not np.isclose(selected["component_weight"], 0.025):
        fail("Official equal weight changed.")
    if selected["weights_tuned"]:
        fail("Official weights must remain fixed.")
    if selected["april_outcomes_used_for_component_or_weight_tuning"]:
        fail("April outcomes were used to tune components or weights.")
    if not selected["april_outcomes_viewed_descriptively"]:
        fail("April descriptive review is not disclosed.")
    if selected["validation_best_single_component"] != {
        "half_life": 12.0,
        "C": 0.0075,
    }:
        fail("Single benchmark hyperparameters changed.")

    if (
        governance["selected_model"]
        != "uniform_40_component_logistic_ensemble"
    ):
        fail("Governance selection disagrees with official metadata.")
    if not governance["march_used_for_promotion_governance"]:
        fail("March governance role is not disclosed.")
    if governance["march_used_for_component_or_weight_tuning"]:
        fail("March improperly tuned components or weights.")
    if governance["april_outcomes_used_for_component_or_weight_tuning"]:
        fail("April improperly tuned components or weights.")
    if not governance["april_outcomes_viewed_descriptively"]:
        fail("April descriptive review is not disclosed in governance.")


def validate_performance_contract(root: Path) -> None:
    """Confirm committed proper-score results remain reproducible."""

    validation = pd.read_csv(
        root / "outputs/ensemble_validation_metrics.csv"
    ).set_index("model")
    march_table = pd.read_csv(
        root / "outputs/march_temporal_check_metrics.csv"
    )
    if set(march_table["split"]) != {"March governance check"}:
        fail("March is not labeled consistently as a governance check.")
    march = march_table.set_index("model")
    april = pd.read_csv(
        root / "outputs/april_descriptive_metrics.csv"
    ).iloc[0]

    ensemble = "uniform_40_component_logistic_ensemble"
    single = "validation_best_single_component"

    if not (
        validation.loc[ensemble, "log_loss"]
        < validation.loc[single, "log_loss"]
    ):
        fail("Ensemble no longer improves Jan-Feb validation.")
    if not (
        march.loc[ensemble, "log_loss"]
        < march.loc[single, "log_loss"]
    ):
        fail("Ensemble no longer improves March governance.")
    if not np.isclose(
        april["log_loss"],
        0.4676068796067532,
        atol=1e-12,
    ):
        fail("Official April descriptive log loss changed.")


def validate_component_summary(root: Path) -> None:
    """Confirm all 40 components and equal weights are exported."""

    components = pd.read_csv(
        root / "outputs/ensemble_component_summary.csv"
    )
    if len(components) != 40:
        fail("Expected 40 component-summary rows.")
    if not np.allclose(
        components["weight"],
        0.025,
        atol=0.0,
        rtol=0.0,
    ):
        fail("Component weights are not uniform.")
    if set(components["half_life"]) != {
        5.0,
        8.0,
        12.0,
        16.0,
        24.0,
    }:
        fail("Half-life grid changed.")


def run(root: Path, source_csv: Path) -> None:
    """Run all final submission checks."""

    validate_files(root)
    validate_package(root)
    validate_workflow(root)
    source = load_source(source_csv)
    validate_source_fingerprint(root, source_csv, source)
    validate_april_predictions(root, source)
    validate_model_selection(root)
    validate_performance_contract(root)
    validate_component_summary(root)

    print(
        "PASS: files, package metadata, CI, source fingerprint, "
        "96 April IDs, official ensemble probabilities, fair odds, "
        "single benchmark, component summary, governance selection, "
        "and performance artifacts are internally consistent."
    )


def parse_args() -> argparse.Namespace:
    """Define repository and source-data paths."""

    parser = argparse.ArgumentParser(
        description="Validate the final ensemble submission."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.root.resolve(), arguments.data.resolve())
