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
    """Raise a useful assertion instead of relying on optimized-away asserts."""

    if not condition:
        raise AssertionError(message)


def validate_predictions() -> None:
    """Check April identifiers, probabilities, odds, and uncertainty rows."""

    predictions = pd.read_csv(
        ROOT / "outputs/april_predictions.csv",
        dtype={"game_id": "string"},
    )
    uncertainty = pd.read_csv(
        ROOT / "outputs/april_model_uncertainty.csv",
        dtype={"game_id": "string"},
    )

    require(len(predictions) == 96, "Expected 96 April predictions.")
    require(predictions["game_id"].is_unique, "April game IDs are not unique.")
    require(
        predictions["game_id"].str.fullmatch(r"\d{10}").all(),
        "April game IDs must remain ten-character digit strings.",
    )

    probability = predictions["home_win_probability"]
    require(
        probability.between(0.0, 1.0, inclusive="neither").all(),
        "An official probability is outside (0, 1).",
    )

    np.testing.assert_allclose(
        predictions["fair_home_decimal_odds"],
        1.0 / probability,
        atol=1e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        predictions["fair_away_decimal_odds"],
        1.0 / (1.0 - probability),
        atol=1e-9,
        rtol=0.0,
    )

    require(len(uncertainty) == 96, "Expected 96 uncertainty rows.")
    require(
        uncertainty["game_id"].is_unique,
        "Uncertainty game IDs are not unique.",
    )
    require(
        set(uncertainty["game_id"]) == set(predictions["game_id"]),
        "Uncertainty and official prediction IDs differ.",
    )

    merged = predictions.merge(
        uncertainty[
            ["game_id", "official_home_win_probability"]
        ],
        on="game_id",
        validate="one_to_one",
    )
    np.testing.assert_allclose(
        merged["home_win_probability"],
        merged["official_home_win_probability"],
        atol=1e-6,
        rtol=0.0,
    )


def validate_json_and_tables() -> None:
    """Parse every committed CSV/JSON and verify governance decisions."""

    for path in sorted((ROOT / "outputs").glob("*.csv")):
        pd.read_csv(path)

    for path in sorted((ROOT / "research/outputs").glob("*.csv")):
        pd.read_csv(path)

    for path in sorted((ROOT / "outputs").glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))

    for path in sorted((ROOT / "research/outputs").glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))

    selected = json.loads(
        (
            ROOT / "outputs/enhanced_selection_decision.json"
        ).read_text(encoding="utf-8")
    )
    require(
        selected["selected_model"] == "three_signal_champion",
        "Enhanced governance no longer selects the official champion.",
    )
    require(
        selected["march_is_a_governance_check_not_an_untouched_test"] is True,
        "March must be labelled as a governance check.",
    )
    require(
        selected["april_outcomes_used_for_model_selection"] is False,
        "April outcomes must not be used for selection.",
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
    require(
        venue["april_outcomes_used_for_selection"] is False,
        "April outcomes were used in venue-model selection.",
    )
    require(
        venue["march_is_governance_not_pristine_test"] is True,
        "March venue results must be labelled as governance.",
    )


def validate_figures() -> None:
    """Confirm every committed PNG is nonempty and has a valid signature."""

    png_signature = b"\x89PNG\r\n\x1a\n"
    figure_paths = [
        *sorted((ROOT / "figures").glob("*.png")),
        *sorted((ROOT / "research/figures").glob("*.png")),
    ]
    require(bool(figure_paths), "No figures were found.")

    for path in figure_paths:
        data = path.read_bytes()
        require(
            len(data) > len(png_signature),
            f"Figure is empty: {path.relative_to(ROOT)}",
        )
        require(
            data.startswith(png_signature),
            f"Figure is not a valid PNG: {path.relative_to(ROOT)}",
        )


def main() -> None:
    """Run all data-free committed-artifact checks."""

    validate_predictions()
    validate_json_and_tables()
    validate_figures()
    print(
        "PASS: committed predictions, odds, uncertainty, tables, JSON "
        "governance artifacts, and figures are internally valid."
    )


if __name__ == "__main__":
    main()
