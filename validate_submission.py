"""Validate the final recruiter-facing NBA submission."""

from __future__ import annotations

import argparse
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
    "MODEL_CARD.md",
    "LIMITATIONS_AND_ROADMAP.md",
    "ARTIFACT_MANIFEST.md",
    "CONTRIBUTING.md",
    "nba_win_probability.py",
    "enhanced_governance.py",
    "challenger_analysis.py",
    "ablation_and_timing.py",
    "run_submission.py",
    "validate_submission.py",
    "requirements.txt",
    "requirements-challengers.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    ".python-version",
    ".gitignore",
    "project_runtime.py",
    "scripts/check_python.py",
    "scripts/__init__.py",
    "scripts/check_repository_policy.py",
    "scripts/validate_committed_artifacts.py",
    "scripts/run_quality_checks.sh",
    "scripts/bootstrap_macos.sh",
    "tests/test_runtime_contract.py",
    "tests/test_repository_configuration.py",
    ".github/workflows/tests.yml",
    ".github/dependabot.yml",
    "MODEL_EVOLUTION.md",
    "REPRODUCIBILITY.md",
    "research/TEAM_SPECIFIC_HOME_EFFECTS.md",
    "research/__init__.py",
    "research/team_specific_home_effects.py",
    "research/outputs/team_specific_home_effect_grid.csv",
    "research/outputs/team_specific_home_effect_summary.csv",
    "research/outputs/team_specific_home_effect_bootstrap.csv",
    "research/outputs/team_specific_home_effect_april_descriptive.csv",
    "research/outputs/team_specific_home_effect_decision.json",
    "research/figures/team_specific_home_effect_comparison.png",
    "tests/test_model_contracts.py",
    ".github/workflows/tests.yml",
    "outputs/april_predictions.csv",
    "outputs/april_model_uncertainty.csv",
    "outputs/validation_grid.csv",
    "outputs/march_temporal_check_metrics.csv",
    "outputs/final_model_coefficients.csv",
    "outputs/model_summary.json",
    "outputs/selected_hyperparameters.json",
    "outputs/enhanced_model_comparison.csv",
    "outputs/enhanced_candidate_grid.csv",
    "outputs/enhanced_bootstrap_model_differences.csv",
    "outputs/enhanced_monthly_backtest.csv",
    "outputs/enhanced_march_calibration_summary.csv",
    "outputs/enhanced_march_reliability_bins.csv",
    "outputs/enhanced_march_calibration_bootstrap.csv",
    "outputs/enhanced_feature_correlation.csv",
    "outputs/enhanced_feature_vif.csv",
    "outputs/enhanced_coefficient_stability.csv",
    "outputs/enhanced_selection_decision.json",
    "outputs/challenger_benchmark.csv",
    "outputs/ensemble_benchmark.csv",
    "outputs/calibration_benchmark.csv",
    "outputs/feature_ablation.csv",
    "figures/march_model_comparison.png",
    "figures/march_calibration.png",
    "figures/final_model_coefficients.png",
    "figures/enhanced_candidate_comparison.png",
    "figures/monthly_model_stability.png",
    "figures/coefficient_stability.png",
    "figures/march_calibration_uncertainty.png",
}


def fail(message: str) -> None:
    """Raise one concise validation error."""

    raise AssertionError(message)


def validate_files(root: Path) -> None:
    """Confirm that every promised source, document and artifact exists."""

    missing = sorted(
        relative
        for relative in REQUIRED_FILES
        if not (root / relative).exists()
    )
    if missing:
        fail(f"Missing required files: {missing}")


def validate_requirements(root: Path) -> None:
    """Confirm package metadata, locks, and supported Python are synchronized."""

    metadata = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = metadata["project"]

    if project["requires-python"] != ">=3.11,<3.14":
        fail("pyproject.toml does not enforce Python 3.11-3.13.")

    expected_core = {
        "numpy==2.3.5",
        "pandas==2.2.3",
        "scikit-learn==1.8.0",
        "matplotlib==3.10.8",
    }
    if set(project["dependencies"]) != expected_core:
        fail("Core dependencies in pyproject.toml changed unexpectedly.")

    expected_dev = {
        "pytest==9.0.2",
        "PyYAML==6.0.3",
    }
    if set(project["optional-dependencies"]["dev"]) != expected_dev:
        fail("Development dependencies changed unexpectedly.")

    expected_challengers = {
        "xgboost==3.1.3",
        "catboost==1.2.8",
        "shap==0.50.0",
        "scipy==1.17.0",
    }
    if (
        set(project["optional-dependencies"]["challengers"])
        != expected_challengers
    ):
        fail("Challenger dependencies changed unexpectedly.")

    requirement_expectations = {
        "requirements.txt": "-e .",
        "requirements-dev.txt": "-e .[dev]",
        "requirements-challengers.txt": "-e .[challengers]",
    }
    for filename, expected in requirement_expectations.items():
        lines = [
            line.strip()
            for line in (root / filename).read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if lines != [expected]:
            fail(f"{filename} is not synchronized with pyproject.toml.")

    if (
        root.joinpath(".python-version").read_text(
            encoding="utf-8"
        ).strip()
        != "3.12.13"
    ):
        fail(".python-version must identify the recommended Python 3.12.13.")


def validate_workflow(root: Path) -> None:
    """Confirm CI uses supported runtimes and warning-clean commands."""

    workflow = (
        root / ".github/workflows/tests.yml"
    ).read_text(encoding="utf-8")

    required_fragments = {
        "actions/checkout@v7.0.0",
        "actions/setup-python@v6.3.0",
        'PIP_DISABLE_PIP_VERSION_CHECK: "1"',
        "fail-fast: false",
        '- "3.11"',
        '- "3.12"',
        '- "3.13"',
        "python -m pytest -q",
        "python scripts/check_repository_policy.py",
        "python scripts/validate_committed_artifacts.py",
        "persist-credentials: false",
    }
    missing = sorted(
        fragment
        for fragment in required_fragments
        if fragment not in workflow
    )
    if missing:
        fail(f"CI workflow is missing required controls: {missing}")

    if "pytest -q" in workflow.replace("python -m pytest -q", ""):
        fail("CI contains a fragile bare pytest invocation.")


def validate_predictions(root: Path, source_csv: Path) -> None:
    """Validate official IDs, probabilities, fair odds and uncertainty rows."""

    source = pd.read_csv(source_csv, dtype={"game_id": "string"})
    source_dates = pd.to_datetime(source["game_date"], errors="raise")
    expected_ids = set(
        source.loc[source_dates.dt.month == 4, "game_id"]
    )

    predictions = pd.read_csv(
        root / "outputs/april_predictions.csv",
        dtype={"game_id": "string"},
    )
    uncertainty = pd.read_csv(
        root / "outputs/april_model_uncertainty.csv",
        dtype={"game_id": "string"},
    )

    if len(predictions) != 96 or not predictions["game_id"].is_unique:
        fail("Expected exactly 96 unique official April predictions.")
    if len(uncertainty) != 96 or not uncertainty["game_id"].is_unique:
        fail("Expected exactly 96 unique April uncertainty rows.")
    if set(predictions["game_id"]) != expected_ids:
        fail("Official April identifiers do not match the source.")
    if set(uncertainty["game_id"]) != expected_ids:
        fail("Uncertainty identifiers do not match the source.")
    if not predictions["game_id"].str.fullmatch(r"\d{10}").all():
        fail("Official game IDs did not retain ten characters.")
    if not uncertainty["game_id"].str.fullmatch(r"\d{10}").all():
        fail("Uncertainty game IDs did not retain ten characters.")

    for column in ("home_win_probability", "daily_repricing_probability"):
        if not predictions[column].between(
            0.0,
            1.0,
            inclusive="neither",
        ).all():
            fail(f"{column} contains an invalid probability.")

    if not np.allclose(
        predictions["fair_home_decimal_odds"],
        1.0 / predictions["home_win_probability"],
        atol=1e-9,
    ):
        fail("Official home fair odds are inconsistent.")
    if not np.allclose(
        predictions["fair_away_decimal_odds"],
        1.0 / (1.0 - predictions["home_win_probability"]),
        atol=1e-9,
    ):
        fail("Official away fair odds are inconsistent.")

    merged = predictions.merge(
        uncertainty[
            ["game_id", "official_home_win_probability"]
        ],
        on="game_id",
        validate="one_to_one",
    )
    if not np.allclose(
        merged["home_win_probability"],
        merged["official_home_win_probability"],
        atol=1e-6,
    ):
        fail("Official and uncertainty-file probabilities disagree.")

    interval_columns = [
        "model_uncertainty_2_5_percent",
        "model_uncertainty_5_percent",
        "bootstrap_median",
        "model_uncertainty_95_percent",
        "model_uncertainty_97_5_percent",
    ]
    if not uncertainty[interval_columns].apply(
        lambda column: column.between(0.0, 1.0)
    ).all().all():
        fail("A model-uncertainty quantile falls outside [0, 1].")
    if not (
        (
            uncertainty["model_uncertainty_2_5_percent"]
            <= uncertainty["model_uncertainty_5_percent"]
        )
        & (
            uncertainty["model_uncertainty_5_percent"]
            <= uncertainty["bootstrap_median"]
        )
        & (
            uncertainty["bootstrap_median"]
            <= uncertainty["model_uncertainty_95_percent"]
        )
        & (
            uncertainty["model_uncertainty_95_percent"]
            <= uncertainty["model_uncertainty_97_5_percent"]
        )
    ).all():
        fail("April model-uncertainty quantiles are not ordered.")


def validate_home_advantage(root: Path) -> None:
    """Confirm coefficient and JSON home-baseline artifacts agree."""

    coefficients = pd.read_csv(
        root / "outputs/final_model_coefficients.csv"
    )
    summary = json.loads(
        (root / "outputs/model_summary.json").read_text(
            encoding="utf-8"
        )
    )
    home_row = coefficients.loc[
        coefficients["term"] == "equal_strength_home_advantage"
    ]
    if len(home_row) != 1:
        fail("Expected exactly one equal-strength home-advantage row.")

    probability = summary["home_advantage"][
        "equal_strength_home_win_probability"
    ]
    log_odds = summary["home_advantage"][
        "equal_strength_home_log_odds"
    ]
    if not np.isclose(
        home_row.iloc[0]["reference_home_win_probability"],
        probability,
    ):
        fail("Home probability artifacts disagree.")
    if not np.isclose(
        home_row.iloc[0]["coefficient_standardized"],
        log_odds,
    ):
        fail("Home log-odds artifacts disagree.")
    if not np.isclose(
        probability,
        1.0 / (1.0 + np.exp(-log_odds)),
    ):
        fail("Home probability is not the sigmoid of home log odds.")


def validate_selection(root: Path) -> None:
    """Confirm enhanced governance retains the official champion honestly."""

    selection = json.loads(
        (root / "outputs/enhanced_selection_decision.json").read_text(
            encoding="utf-8"
        )
    )
    if selection["selected_model"] != "three_signal_champion":
        fail("Enhanced governance no longer selects the official champion.")
    if not selection[
        "march_is_a_governance_check_not_an_untouched_test"
    ]:
        fail("March is not labelled as a governance check.")
    if selection["april_outcomes_used_for_model_selection"]:
        fail("April outcomes were incorrectly marked as model-selection data.")

    parameters = json.loads(
        (root / "outputs/selected_hyperparameters.json").read_text(
            encoding="utf-8"
        )
    )
    if not np.isclose(parameters["half_life"], 12.0):
        fail("Official half-life changed.")
    if not np.isclose(parameters["C"], 0.0075):
        fail("Official C changed.")

    comparison = pd.read_csv(
        root / "outputs/enhanced_model_comparison.csv"
    )
    champion = comparison.loc[
        comparison["family"] == "three_signal_champion"
    ]
    if len(champion) != 1:
        fail("Enhanced comparison is missing the champion.")
    if not np.isclose(
        champion.iloc[0]["validation_log_loss"],
        selection["validation_log_loss"],
    ):
        fail("Selection and comparison validation scores disagree.")
    if not np.isclose(
        champion.iloc[0]["march_log_loss"],
        selection["march_governance_log_loss"],
    ):
        fail("Selection and comparison March scores disagree.")


def validate_calibration(root: Path) -> None:
    """Confirm calibration diagnostics and intervals are internally valid."""

    summary = pd.read_csv(
        root / "outputs/enhanced_march_calibration_summary.csv"
    )
    intervals = pd.read_csv(
        root / "outputs/enhanced_march_calibration_bootstrap.csv"
    )
    if len(summary) != 1:
        fail("Expected one March calibration summary row.")

    gap = (
        summary.iloc[0]["actual_home_win_rate"]
        - summary.iloc[0]["mean_probability"]
    )
    if not np.isclose(
        gap,
        summary.iloc[0][
            "calibration_in_the_large_gap_actual_minus_predicted"
        ],
    ):
        fail("March calibration gap is inconsistent.")

    expected_metrics = {
        "calibration_gap",
        "calibration_intercept",
        "calibration_slope",
        "log_loss",
        "brier_score",
    }
    if set(intervals["metric"]) != expected_metrics:
        fail("Calibration bootstrap metrics are incomplete.")
    if not (
        intervals["bootstrap_2_5_percent"]
        <= intervals["bootstrap_median"]
    ).all():
        fail("Calibration bootstrap lower quantile exceeds median.")
    if not (
        intervals["bootstrap_median"]
        <= intervals["bootstrap_97_5_percent"]
    ).all():
        fail("Calibration bootstrap median exceeds upper quantile.")


def validate_collinearity(root: Path) -> None:
    """Confirm all three official features have reported VIF diagnostics."""

    vif = pd.read_csv(root / "outputs/enhanced_feature_vif.csv")
    expected = {
        "net_wins_diff",
        "cumulative_margin_diff",
        "recent_margin_evidence_diff",
    }
    if set(vif["feature"]) != expected:
        fail("Feature VIF artifact does not contain the official features.")
    if not (vif["variance_inflation_factor"] >= 1.0).all():
        fail("A variance-inflation factor is below one.")



def validate_team_specific_home_effects(root: Path) -> None:
    """Confirm the team-specific venue research remains a rejected challenger."""

    summary = pd.read_csv(
        root
        / "research/outputs/team_specific_home_effect_summary.csv"
    )
    bootstrap = pd.read_csv(
        root
        / "research/outputs/team_specific_home_effect_bootstrap.csv"
    )
    decision = json.loads(
        (
            root
            / "research/outputs/team_specific_home_effect_decision.json"
        ).read_text(encoding="utf-8")
    )

    expected_models = {
        "three_signal_champion",
        "team_specific_venue_challenger",
    }
    if set(summary["model"]) != expected_models:
        fail("Team-specific home-effect summary has unexpected models.")

    champion = summary.loc[
        summary["model"] == "three_signal_champion"
    ].iloc[0]
    challenger = summary.loc[
        summary["model"] == "team_specific_venue_challenger"
    ].iloc[0]

    if not (
        challenger["validation_log_loss"]
        < champion["validation_log_loss"]
    ):
        fail(
            "The documented venue challenger no longer improves "
            "validation log loss."
        )

    if decision["promote_to_champion"]:
        fail(
            "Team-specific home effects were incorrectly promoted "
            "despite inconclusive evidence."
        )
    if decision["april_outcomes_used_for_selection"]:
        fail("April outcomes were incorrectly used for venue-model selection.")
    if not decision["march_is_governance_not_pristine_test"]:
        fail("March venue analysis is not labelled as governance.")

    if set(bootstrap["period"]) != {
        "January-February validation",
        "March governance",
    }:
        fail("Team-specific bootstrap periods are incomplete.")

    if not (
        (
            bootstrap["bootstrap_2_5_percent"]
            <= bootstrap[
                "observed_challenger_minus_champion_log_loss"
            ]
        )
        & (
            bootstrap[
                "observed_challenger_minus_champion_log_loss"
            ]
            <= bootstrap["bootstrap_97_5_percent"]
        )
    ).all():
        fail("Observed venue-model differences fall outside bootstrap bounds.")

    march_row = bootstrap.loc[
        bootstrap["period"] == "March governance"
    ].iloc[0]
    if not (
        march_row["bootstrap_2_5_percent"]
        <= 0.0
        <= march_row["bootstrap_97_5_percent"]
    ):
        fail(
            "March team-specific comparison no longer reflects "
            "an inconclusive interval."
        )

def run(root: Path, source_csv: Path) -> None:
    """Run all final package checks."""

    validate_files(root)
    validate_requirements(root)
    validate_workflow(root)
    validate_predictions(root, source_csv)
    validate_home_advantage(root)
    validate_selection(root)
    validate_calibration(root)
    validate_collinearity(root)
    validate_team_specific_home_effects(root)
    print(
        "PASS: files, package metadata, CI workflow, 96 April IDs, "
        "probabilities, odds, "
        "uncertainty intervals, home advantage, governance selection, "
        "calibration, collinearity and team-specific venue artifacts are "
        "internally consistent."
    )


def parse_args() -> argparse.Namespace:
    """Define repository and source-data paths."""

    parser = argparse.ArgumentParser(
        description="Validate the final NBA probability submission."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.root, arguments.data)
