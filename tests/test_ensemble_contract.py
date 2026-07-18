"""Tests for the official late-season ensemble contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import nba_win_probability as model

ROOT = Path(__file__).resolve().parents[1]


def test_official_component_grid_is_fixed() -> None:
    """The published ensemble must contain exactly 40 predeclared components."""

    assert model.HALF_LIFE_GRID == (5.0, 8.0, 12.0, 16.0, 24.0)
    assert model.C_GRID == (
        0.003,
        0.005,
        0.0075,
        0.010,
        0.015,
        0.020,
        0.030,
        0.050,
    )
    assert len(model.HALF_LIFE_GRID) * len(model.C_GRID) == 40


def test_ensemble_probability_is_uniform_arithmetic_mean() -> None:
    """Official aggregation must remain an untuned equal-weight mean."""

    matrix = np.tile(np.linspace(0.2, 0.8, 40), (5, 1))
    probability = model.ensemble_probability(matrix)

    np.testing.assert_allclose(
        probability,
        matrix.mean(axis=1),
        atol=0.0,
        rtol=0.0,
    )


def test_ensemble_rejects_wrong_component_count() -> None:
    """A malformed component matrix must fail rather than silently reweight."""

    malformed = np.full((2, 39), 0.5)

    try:
        model.ensemble_probability(malformed)
    except ValueError as error:
        assert "Unexpected number" in str(error)
    else:
        raise AssertionError("Malformed component matrix was accepted.")


def test_selected_model_metadata_matches_official_contract() -> None:
    """Committed metadata must describe the exact public price engine."""

    selected = json.loads(
        (ROOT / "outputs/selected_model.json").read_text(
            encoding="utf-8"
        )
    )

    assert selected["model"] == (
        "uniform_40_component_logistic_ensemble"
    )
    assert selected["component_count"] == 40
    assert selected["component_weight"] == 0.025
    assert selected["weights_tuned"] is False
    assert (
        selected["april_outcomes_used_for_component_or_weight_tuning"]
        is False
    )
    assert selected["april_outcomes_viewed_descriptively"] is True
    assert selected["validation_best_single_component"] == {
        "half_life": 12.0,
        "C": 0.0075,
    }


def test_official_and_benchmark_april_files_are_distinct() -> None:
    """Official ensemble and single benchmark must be clearly separated."""

    ensemble = pd.read_csv(
        ROOT / "outputs/april_predictions.csv",
        dtype={"game_id": "string"},
    )
    single = pd.read_csv(
        ROOT / "outputs/single_model_benchmark_april_predictions.csv",
        dtype={"game_id": "string"},
    )

    assert len(ensemble) == 96
    assert len(single) == 96
    assert ensemble["game_id"].tolist() == single["game_id"].tolist()
    assert not np.allclose(
        ensemble["home_win_probability"],
        single["home_win_probability"],
        atol=0.0,
        rtol=0.0,
    )



def test_march_is_labeled_as_governance_not_pristine_test() -> None:
    """Public artifacts must describe March consistently as governance."""

    selected = json.loads(
        (ROOT / "outputs/selected_model.json").read_text(
            encoding="utf-8"
        )
    )
    assert "governance" in selected["march_role"].lower()

    metrics = pd.read_csv(
        ROOT / "outputs/march_temporal_check_metrics.csv"
    )
    assert set(metrics["split"]) == {"March governance check"}

def test_official_fair_odds_recompute_exactly_from_serialized_price() -> None:
    """Displayed fair odds must derive from the exact stored probability."""

    predictions = pd.read_csv(
        ROOT / "outputs/april_predictions.csv",
        dtype={"game_id": "string"},
    )
    probability = predictions["home_win_probability"].to_numpy()

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
