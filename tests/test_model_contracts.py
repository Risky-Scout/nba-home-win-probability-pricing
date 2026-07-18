"""Focused unit tests for the leakage and timing contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
