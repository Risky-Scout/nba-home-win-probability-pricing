"""Focused unit tests for the leakage and timing contracts."""

from __future__ import annotations

from pathlib import Path

from project_runtime import require_supported_python

require_supported_python()

import numpy as np
import pandas as pd
import pytest

import nba_win_probability as model


def synthetic_games() -> pd.DataFrame:
    """Create three internally consistent games for two synthetic teams."""

    rows = [
        {
            "game_id": "0000000001",
            "game_date": "2025-10-01",
            "away": "BBB",
            "away_wins": 0,
            "away_losses": 0,
            "away_points": 90,
            "away_turnovers": 10,
            "away_fouls": 20,
            "away_rebounds": 40,
            "home": "AAA",
            "home_wins": 0,
            "home_losses": 0,
            "home_points": 100,
            "home_turnovers": 11,
            "home_fouls": 19,
            "home_rebounds": 45,
        },
        {
            "game_id": "0000000002",
            "game_date": "2025-10-02",
            "away": "AAA",
            "away_wins": 1,
            "away_losses": 0,
            "away_points": 92,
            "away_turnovers": 12,
            "away_fouls": 21,
            "away_rebounds": 42,
            "home": "BBB",
            "home_wins": 0,
            "home_losses": 1,
            "home_points": 95,
            "home_turnovers": 9,
            "home_fouls": 18,
            "home_rebounds": 43,
        },
        {
            "game_id": "0000000003",
            "game_date": "2025-10-03",
            "away": "BBB",
            "away_wins": 1,
            "away_losses": 1,
            "away_points": 99,
            "away_turnovers": 13,
            "away_fouls": 22,
            "away_rebounds": 41,
            "home": "AAA",
            "home_wins": 1,
            "home_losses": 1,
            "home_points": 101,
            "home_turnovers": 10,
            "home_fouls": 20,
            "home_rebounds": 44,
        },
    ]
    table = pd.DataFrame(rows)
    table["game_date"] = pd.to_datetime(table["game_date"])
    table["game_id"] = table["game_id"].astype("string")
    return table


def test_feature_snapshot_precedes_current_result() -> None:
    """The first result may affect game two, never game one."""

    features = model.build_sequential_features(
        synthetic_games(),
        half_life=12.0,
    )

    first = features.iloc[0]
    assert first["net_wins_diff"] == 0.0
    assert first["cumulative_margin_diff"] == 0.0
    assert first["recent_margin_evidence_diff"] == 0.0

    second = features.iloc[1]
    assert second["home"] == "BBB"
    assert second["away"] == "AAA"
    assert second["net_wins_diff"] == -2.0
    assert second["cumulative_margin_diff"] == -20.0
    assert second["recent_margin_evidence_diff"] == -20.0


def test_frozen_features_ignore_post_cutoff_results() -> None:
    """All post-cutoff games must use one unchanged state snapshot."""

    frozen = model.build_frozen_features(
        synthetic_games(),
        half_life=12.0,
        cutoff=pd.Timestamp("2025-10-01"),
    )

    game_two = frozen.loc[
        frozen["game_id"] == "0000000002"
    ].iloc[0]
    game_three = frozen.loc[
        frozen["game_id"] == "0000000003"
    ].iloc[0]

    assert game_two["net_wins_diff"] == -2.0
    assert game_two["cumulative_margin_diff"] == -20.0

    # Game three is priced from the game-one snapshot, not after game two.
    assert game_three["net_wins_diff"] == 2.0
    assert game_three["cumulative_margin_diff"] == 20.0


def test_game_id_is_preserved_as_text(tmp_path: Path) -> None:
    """Loading must preserve leading zeroes in the source identifier."""

    path = tmp_path / "games.csv"
    synthetic_games().to_csv(path, index=False)

    loaded = model.load_and_validate(path)

    assert str(loaded.iloc[0]["game_id"]) == "0000000001"
    assert loaded["game_id"].dtype.name.startswith("string")


def test_fair_odds_identity() -> None:
    """Home and away fair implied probabilities must sum to one."""

    probability = np.array([0.25, 0.50, 0.80])
    home_odds = 1.0 / probability
    away_odds = 1.0 / (1.0 - probability)

    implied_sum = 1.0 / home_odds + 1.0 / away_odds
    assert np.allclose(implied_sum, 1.0)


def test_duplicate_game_id_is_rejected(tmp_path: Path) -> None:
    """Duplicate identifiers must fail before modeling."""

    games = synthetic_games()
    games.loc[1, "game_id"] = games.loc[0, "game_id"]
    path = tmp_path / "duplicate.csv"
    games.to_csv(path, index=False)

    with pytest.raises(ValueError, match="game_id is not unique"):
        model.load_and_validate(path)


def test_inconsistent_pregame_record_is_rejected(tmp_path: Path) -> None:
    """A stale or shifted pregame record must fail at the exact row."""

    games = synthetic_games()
    games.loc[1, "away_wins"] = 0
    path = tmp_path / "record_mismatch.csv"
    games.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Pregame record mismatch"):
        model.load_and_validate(path)


def test_model_probabilities_are_strictly_bounded() -> None:
    """The fitted logistic link must return finite probabilities in (0, 1)."""

    features = pd.DataFrame(
        {
            "net_wins_diff": [-5.0, -2.0, 2.0, 5.0],
            "cumulative_margin_diff": [-50.0, -20.0, 20.0, 50.0],
            "recent_margin_evidence_diff": [-80.0, -25.0, 25.0, 80.0],
        }
    )
    outcome = pd.Series([0, 0, 1, 1])

    fitted = model.make_model(0.0075)
    fitted.fit(features, outcome)
    probability = fitted.predict_proba(features)[:, 1]

    assert np.isfinite(probability).all()
    assert ((probability > 0.0) & (probability < 1.0)).all()
