"""Validate committed outputs without requiring the private source CSV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_runtime import require_supported_python  # noqa: E402

require_supported_python()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def require(condition: bool, message: str) -> None:
    """Raise a useful assertion instead of optimized-away assertions."""

    if not condition:
        raise AssertionError(message)


def validate_predictions() -> None:
    """Check official and benchmark April prediction contracts."""

    official = pd.read_csv(
        ROOT / "outputs/april_predictions.csv",
        dtype={"game_id": "string"},
    )
    benchmark = pd.read_csv(
        ROOT / "outputs/single_model_benchmark_april_predictions.csv",
        dtype={"game_id": "string"},
    )
    dispersion = pd.read_csv(
        ROOT / "outputs/april_component_dispersion.csv",
        dtype={"game_id": "string"},
    )

    for name, table in (
        ("official", official),
        ("benchmark", benchmark),
        ("dispersion", dispersion),
    ):
        require(len(table) == 96, f"Expected 96 {name} April rows.")
        require(
            table["game_id"].is_unique,
            f"{name} April game IDs are not unique.",
        )
        require(
            table["game_id"].str.fullmatch(r"\d{10}").all(),
            f"{name} April IDs must remain ten-character digit strings.",
        )

    require(
        official["game_id"].tolist() == benchmark["game_id"].tolist(),
        "Official and benchmark game order differs.",
    )
    require(
        official["game_id"].tolist() == dispersion["game_id"].tolist(),
        "Official and dispersion game order differs.",
    )

    probability = official["home_win_probability"].to_numpy()
    require(
        np.all((probability > 0.0) & (probability < 1.0)),
        "An official probability is outside (0, 1).",
    )

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

    require(
        np.all(
            dispersion["component_minimum"]
            <= dispersion["component_median"]
        ),
        "Component minimum exceeds median.",
    )
    require(
        np.all(
            dispersion["component_median"]
            <= dispersion["component_maximum"]
        ),
        "Component median exceeds maximum.",
    )


def validate_json_and_tables() -> None:
    """Parse every committed table and verify model-governance decisions."""

    for directory in (
        ROOT / "outputs",
        ROOT / "research/outputs",
        ROOT / "research/single_model_outputs",
    ):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            pd.read_csv(path)
        for path in sorted(directory.glob("*.json")):
            json.loads(path.read_text(encoding="utf-8"))

    selected = json.loads(
        (ROOT / "outputs/selected_model.json").read_text(
            encoding="utf-8"
        )
    )
    require(
        selected["model"]
        == "uniform_40_component_logistic_ensemble",
        "Official selected-model metadata changed.",
    )
    require(selected["component_count"] == 40, "Expected 40 components.")
    require(
        selected["component_weight"] == 0.025,
        "Official component weight changed.",
    )
    require(
        selected["weights_tuned"] is False,
        "Ensemble weights must remain fixed and untuned.",
    )
    require(
        selected["april_outcomes_used_for_component_or_weight_tuning"]
        is False,
        "April outcomes must not tune components or weights.",
    )
    require(
        selected["april_outcomes_viewed_descriptively"] is True,
        "April descriptive review must be disclosed.",
    )

    governance = json.loads(
        (
            ROOT / "outputs/governance_selection_decision.json"
        ).read_text(encoding="utf-8")
    )
    require(
        governance["selected_model"]
        == "uniform_40_component_logistic_ensemble",
        "Governance no longer selects the official ensemble.",
    )
    require(
        governance["march_used_for_promotion_governance"] is True,
        "March governance role is not disclosed.",
    )
    require(
        governance["march_used_for_component_or_weight_tuning"] is False,
        "March must not tune component definitions or weights.",
    )
    require(
        governance["april_outcomes_used_for_component_or_weight_tuning"]
        is False,
        "April outcomes must not tune components or weights.",
    )
    require(
        governance["april_outcomes_viewed_descriptively"] is True,
        "April descriptive review must be disclosed in governance.",
    )

    venue = json.loads(
        (
            ROOT
            / "research/outputs/team_specific_home_effect_decision.json"
        ).read_text(encoding="utf-8")
    )
    require(
        venue["promote_to_champion"] is False,
        "Team-specific venue challenger was unexpectedly promoted.",
    )


def validate_figures() -> None:
    """Confirm every committed PNG is nonempty and has a valid signature."""

    signature = b"\x89PNG\r\n\x1a\n"
    paths = [
        *sorted((ROOT / "figures").glob("*.png")),
        *sorted((ROOT / "research/figures").glob("*.png")),
        *sorted((ROOT / "research/single_model_figures").glob("*.png")),
    ]
    require(bool(paths), "No figures were found.")

    for path in paths:
        data = path.read_bytes()
        require(
            len(data) > len(signature),
            f"Figure is empty: {path.relative_to(ROOT)}",
        )
        require(
            data.startswith(signature),
            f"Figure is not a valid PNG: {path.relative_to(ROOT)}",
        )


def main() -> None:
    """Run all data-free committed-artifact checks."""

    validate_predictions()
    validate_json_and_tables()
    validate_figures()
    print(
        "PASS: official ensemble predictions, benchmark, dispersion, "
        "governance tables, JSON, and figures are internally valid."
    )


if __name__ == "__main__":
    main()
