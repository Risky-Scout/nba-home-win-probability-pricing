"""Evaluate team-specific home-court information as a challenger.

Important distinction
---------------------
The official model already produces a matchup-specific probability for every
home team. This script tests a different question: whether the model should
also estimate team-specific venue effects rather than one global home-court
baseline.

The challenger uses only information available before each game:

1. A strongly shrunk league-wide home-win baseline.
2. A team-specific venue deviation:
   - the home team's shrunk home performance relative to its overall strength;
   - minus the away team's shrunk road performance relative to its overall
     strength.

The challenger is selected using January-February validation log loss. March
is reported only as a later governance check. April outcomes are descriptive
and never used for selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_runtime import require_supported_python  # noqa: E402

require_supported_python()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nba_win_probability as champion  # noqa: E402


CORE_FEATURES = list(champion.FEATURE_COLUMNS)
C_GRID = (0.005, 0.0075, 0.010, 0.015, 0.020)
TEAM_PRIOR_GRID = (5.0, 10.0, 20.0, 40.0, 80.0)
GLOBAL_PRIOR_GRID = (100.0, 200.0, 500.0)
FEATURE_SETS = {
    "global_home_trend": ["global_home_logit"],
    "team_specific_venue_deviation": ["venue_deviation_diff"],
    "global_and_team_specific": [
        "global_home_logit",
        "venue_deviation_diff",
    ],
}
RANDOM_SEED = 365
BOOTSTRAP_REPLICATES = 20_000

TRAIN_END = pd.Timestamp("2025-12-31")
VALIDATION_START = pd.Timestamp("2026-01-01")
VALIDATION_END = pd.Timestamp("2026-02-28")
MARCH_START = pd.Timestamp("2026-03-01")
MARCH_END = pd.Timestamp("2026-03-31")
APRIL_START = pd.Timestamp("2026-04-01")


def safe_logit(probability: float | np.ndarray) -> float | np.ndarray:
    """Convert a probability to finite log odds."""

    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def make_model(c_value: float) -> Pipeline:
    """Construct the same scaled L2 logistic family as the champion."""

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def probability_metrics(
    outcome: pd.Series | np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    """Return proper probability scores and secondary diagnostics."""

    return {
        "log_loss": float(log_loss(outcome, probability)),
        "brier_score": float(brier_score_loss(outcome, probability)),
        "roc_auc": float(roc_auc_score(outcome, probability)),
        "accuracy_0_5": float(accuracy_score(outcome, probability >= 0.5)),
        "mean_probability": float(np.mean(probability)),
        "actual_home_win_rate": float(np.mean(outcome)),
    }


def initialize_venue_states(
    data: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """Create one neutral venue record for every team."""

    teams = sorted(set(data["home"]).union(data["away"]))
    return {
        team: {
            "wins": 0,
            "losses": 0,
            "home_wins": 0,
            "home_losses": 0,
            "away_wins": 0,
            "away_losses": 0,
        }
        for team in teams
    }


def venue_feature_values(
    home_state: dict[str, int],
    away_state: dict[str, int],
    global_home_wins: int,
    global_games: int,
    team_prior: float,
    global_prior: float,
) -> dict[str, float]:
    """Construct shrunk global and team-specific venue signals."""

    global_home_rate = (
        global_home_wins + 0.5 * global_prior
    ) / (global_games + global_prior)

    home_home_rate = (
        home_state["home_wins"]
        + team_prior * global_home_rate
    ) / (
        home_state["home_wins"]
        + home_state["home_losses"]
        + team_prior
    )
    away_away_rate = (
        away_state["away_wins"]
        + team_prior * (1.0 - global_home_rate)
    ) / (
        away_state["away_wins"]
        + away_state["away_losses"]
        + team_prior
    )

    home_overall_rate = (
        home_state["wins"] + 0.5 * team_prior
    ) / (
        home_state["wins"]
        + home_state["losses"]
        + team_prior
    )
    away_overall_rate = (
        away_state["wins"] + 0.5 * team_prior
    ) / (
        away_state["wins"]
        + away_state["losses"]
        + team_prior
    )

    home_venue_deviation = (
        safe_logit(home_home_rate)
        - safe_logit(home_overall_rate)
    )
    away_venue_deviation = (
        safe_logit(away_away_rate)
        - safe_logit(away_overall_rate)
    )

    return {
        "global_home_logit": float(
            safe_logit(global_home_rate)
        ),
        "venue_deviation_diff": float(
            home_venue_deviation - away_venue_deviation
        ),
    }


def update_venue_states(
    home_state: dict[str, int],
    away_state: dict[str, int],
    home_win: int,
) -> None:
    """Apply one completed result after the current feature snapshot."""

    home_state["wins"] += home_win
    home_state["losses"] += 1 - home_win
    home_state["home_wins"] += home_win
    home_state["home_losses"] += 1 - home_win

    away_state["wins"] += 1 - home_win
    away_state["losses"] += home_win
    away_state["away_wins"] += 1 - home_win
    away_state["away_losses"] += home_win


def build_sequential_venue_features(
    data: pd.DataFrame,
    team_prior: float,
    global_prior: float,
) -> pd.DataFrame:
    """Build as-of-game-time venue features for all games."""

    states = initialize_venue_states(data)
    global_home_wins = 0
    global_games = 0
    rows: list[dict[str, object]] = []

    for row in data.itertuples(index=False):
        features = venue_feature_values(
            states[row.home],
            states[row.away],
            global_home_wins,
            global_games,
            team_prior,
            global_prior,
        )
        rows.append({"game_id": row.game_id, **features})

        home_win = int(row.home_points > row.away_points)
        update_venue_states(
            states[row.home],
            states[row.away],
            home_win,
        )
        global_home_wins += home_win
        global_games += 1

    return pd.DataFrame(rows)


def build_frozen_april_venue_features(
    data: pd.DataFrame,
    team_prior: float,
    global_prior: float,
    cutoff: pd.Timestamp = MARCH_END,
) -> pd.DataFrame:
    """Build all April venue features from one March 31 snapshot."""

    states = initialize_venue_states(data)
    global_home_wins = 0
    global_games = 0

    for row in data.loc[data["game_date"] <= cutoff].itertuples(
        index=False
    ):
        home_win = int(row.home_points > row.away_points)
        update_venue_states(
            states[row.home],
            states[row.away],
            home_win,
        )
        global_home_wins += home_win
        global_games += 1

    rows: list[dict[str, object]] = []
    for row in data.loc[data["game_date"] > cutoff].itertuples(
        index=False
    ):
        features = venue_feature_values(
            states[row.home],
            states[row.away],
            global_home_wins,
            global_games,
            team_prior,
            global_prior,
        )
        rows.append({"game_id": row.game_id, **features})

    return pd.DataFrame(rows)


def fit_predict(
    table: pd.DataFrame,
    columns: list[str],
    c_value: float,
    training_mask: pd.Series,
    prediction_mask: pd.Series,
) -> tuple[Pipeline, np.ndarray]:
    """Fit one candidate and return its probabilities."""

    model = make_model(c_value)
    model.fit(
        table.loc[training_mask, columns],
        table.loc[training_mask, "home_win"],
    )
    probability = model.predict_proba(
        table.loc[prediction_mask, columns]
    )[:, 1]
    return model, probability


def evaluate_grid(
    data: pd.DataFrame,
    core_table: pd.DataFrame,
) -> pd.DataFrame:
    """Tune every team-specific challenger on validation log loss."""

    training = core_table["game_date"] <= TRAIN_END
    validation = core_table["game_date"].between(
        VALIDATION_START,
        VALIDATION_END,
    )
    train_validation = core_table["game_date"] <= VALIDATION_END
    march = core_table["game_date"].between(
        MARCH_START,
        MARCH_END,
    )
    rows: list[dict[str, object]] = []

    for team_prior in TEAM_PRIOR_GRID:
        for global_prior in GLOBAL_PRIOR_GRID:
            venue = build_sequential_venue_features(
                data,
                team_prior,
                global_prior,
            )
            table = core_table.merge(
                venue,
                on="game_id",
                how="left",
                validate="one_to_one",
            )

            for feature_name, extra_features in FEATURE_SETS.items():
                columns = [*CORE_FEATURES, *extra_features]

                for c_value in C_GRID:
                    _, validation_probability = fit_predict(
                        table,
                        columns,
                        c_value,
                        training,
                        validation,
                    )
                    validation_metrics = probability_metrics(
                        table.loc[validation, "home_win"],
                        validation_probability,
                    )
                    rows.append(
                        {
                            "feature_family": feature_name,
                            "team_prior": team_prior,
                            "global_prior": global_prior,
                            "C": c_value,
                            "features": " + ".join(columns),
                            **{
                                f"validation_{key}": value
                                for key, value
                                in validation_metrics.items()
                            },
                        }
                    )

    grid = pd.DataFrame(rows).sort_values(
        [
            "validation_log_loss",
            "validation_brier_score",
            "C",
            "team_prior",
            "global_prior",
        ]
    ).reset_index(drop=True)

    # Score only the validation-selected candidate in March.
    best = grid.iloc[0]
    venue = build_sequential_venue_features(
        data,
        float(best["team_prior"]),
        float(best["global_prior"]),
    )
    table = core_table.merge(
        venue,
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    selected_columns = str(best["features"]).split(" + ")
    _, march_probability = fit_predict(
        table,
        selected_columns,
        float(best["C"]),
        train_validation,
        march,
    )
    march_metrics = probability_metrics(
        table.loc[march, "home_win"],
        march_probability,
    )

    for key, value in march_metrics.items():
        grid.loc[0, f"march_{key}"] = value

    return grid


def champion_probabilities(
    core_table: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Reproduce champion validation and March probabilities."""

    training = core_table["game_date"] <= TRAIN_END
    validation = core_table["game_date"].between(
        VALIDATION_START,
        VALIDATION_END,
    )
    train_validation = core_table["game_date"] <= VALIDATION_END
    march = core_table["game_date"].between(
        MARCH_START,
        MARCH_END,
    )

    _, validation_probability = fit_predict(
        core_table,
        CORE_FEATURES,
        0.0075,
        training,
        validation,
    )
    _, march_probability = fit_predict(
        core_table,
        CORE_FEATURES,
        0.0075,
        train_validation,
        march,
    )
    return {
        "validation": validation_probability,
        "march": march_probability,
    }


def selected_challenger_probabilities(
    data: pd.DataFrame,
    core_table: pd.DataFrame,
    best: pd.Series,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Recreate the validation-selected venue challenger."""

    venue = build_sequential_venue_features(
        data,
        float(best["team_prior"]),
        float(best["global_prior"]),
    )
    table = core_table.merge(
        venue,
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    columns = str(best["features"]).split(" + ")

    training = table["game_date"] <= TRAIN_END
    validation = table["game_date"].between(
        VALIDATION_START,
        VALIDATION_END,
    )
    train_validation = table["game_date"] <= VALIDATION_END
    march = table["game_date"].between(
        MARCH_START,
        MARCH_END,
    )

    _, validation_probability = fit_predict(
        table,
        columns,
        float(best["C"]),
        training,
        validation,
    )
    _, march_probability = fit_predict(
        table,
        columns,
        float(best["C"]),
        train_validation,
        march,
    )

    return table, {
        "validation": validation_probability,
        "march": march_probability,
    }


def per_game_log_loss(
    outcome: np.ndarray,
    probability: np.ndarray,
) -> np.ndarray:
    """Return one log-loss contribution per game."""

    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return -(
        outcome * np.log(clipped)
        + (1 - outcome) * np.log(1.0 - clipped)
    )


def date_block_bootstrap(
    dates: pd.Series,
    outcome: pd.Series,
    champion_probability: np.ndarray,
    challenger_probability: np.ndarray,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    """Estimate uncertainty in paired log-loss differences by date."""

    date_array = pd.to_datetime(dates).to_numpy()
    difference = (
        per_game_log_loss(
            outcome.to_numpy(),
            challenger_probability,
        )
        - per_game_log_loss(
            outcome.to_numpy(),
            champion_probability,
        )
    )
    unique_dates = np.array(sorted(np.unique(date_array)))
    date_sums = np.array(
        [
            difference[date_array == date].sum()
            for date in unique_dates
        ]
    )
    date_counts = np.array(
        [
            np.sum(date_array == date)
            for date in unique_dates
        ]
    )

    random = np.random.default_rng(RANDOM_SEED)
    draws = random.integers(
        0,
        len(unique_dates),
        size=(replicates, len(unique_dates)),
    )
    bootstrap_difference = (
        date_sums[draws].sum(axis=1)
        / date_counts[draws].sum(axis=1)
    )
    lower, median, upper = np.quantile(
        bootstrap_difference,
        (0.025, 0.5, 0.975),
    )

    return {
        "observed_challenger_minus_champion_log_loss": float(
            difference.mean()
        ),
        "bootstrap_2_5_percent": float(lower),
        "bootstrap_median": float(median),
        "bootstrap_97_5_percent": float(upper),
        "probability_challenger_is_better": float(
            np.mean(bootstrap_difference < 0.0)
        ),
        "replicates": replicates,
    }


def april_descriptive_audit(
    data: pd.DataFrame,
    sequential_table: pd.DataFrame,
    best: pd.Series,
) -> dict[str, float]:
    """Score strict March 31 challenger prices descriptively."""

    frozen_core = champion.build_frozen_features(
        data,
        half_life=12.0,
        cutoff=MARCH_END,
    )
    frozen_april = frozen_core.loc[
        frozen_core["game_date"] >= APRIL_START
    ].reset_index(drop=True)
    frozen_venue = build_frozen_april_venue_features(
        data,
        float(best["team_prior"]),
        float(best["global_prior"]),
        cutoff=MARCH_END,
    )
    frozen_table = frozen_april.merge(
        frozen_venue,
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    sequential_venue = build_sequential_venue_features(
        data,
        float(best["team_prior"]),
        float(best["global_prior"]),
    )
    training_table = sequential_table.merge(
        sequential_venue,
        on="game_id",
        how="left",
        validate="one_to_one",
    )
    training = training_table["game_date"] <= MARCH_END
    columns = str(best["features"]).split(" + ")

    model = make_model(float(best["C"]))
    model.fit(
        training_table.loc[training, columns],
        training_table.loc[training, "home_win"],
    )
    probability = model.predict_proba(
        frozen_table[columns]
    )[:, 1]

    return probability_metrics(
        frozen_table["home_win"],
        probability,
    )


def save_figure(
    summary: pd.DataFrame,
    figure_path: Path,
) -> None:
    """Create a concise champion-versus-challenger chart."""

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)
    positions = np.arange(len(summary))
    width = 0.38
    axis.bar(
        positions - width / 2,
        summary["validation_log_loss"],
        width=width,
        label="January-February validation",
    )
    axis.bar(
        positions + width / 2,
        summary["march_log_loss"],
        width=width,
        label="March governance",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(summary["model"])
    axis.set_ylabel("Log loss — lower is better")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)


def run(
    data_path: Path,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    """Run the complete team-specific venue-effect analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = champion.load_and_validate(data_path)
    core_table = champion.build_sequential_features(
        data,
        half_life=12.0,
    )

    grid = evaluate_grid(data, core_table)
    best = grid.iloc[0]
    challenger_table, challenger_probability = (
        selected_challenger_probabilities(
            data,
            core_table,
            best,
        )
    )
    champion_probability = champion_probabilities(core_table)

    validation = core_table["game_date"].between(
        VALIDATION_START,
        VALIDATION_END,
    )
    march = core_table["game_date"].between(
        MARCH_START,
        MARCH_END,
    )

    champion_validation = probability_metrics(
        core_table.loc[validation, "home_win"],
        champion_probability["validation"],
    )
    challenger_validation = probability_metrics(
        challenger_table.loc[validation, "home_win"],
        challenger_probability["validation"],
    )
    champion_march = probability_metrics(
        core_table.loc[march, "home_win"],
        champion_probability["march"],
    )
    challenger_march = probability_metrics(
        challenger_table.loc[march, "home_win"],
        challenger_probability["march"],
    )

    summary = pd.DataFrame(
        [
            {
                "model": "three_signal_champion",
                **{
                    f"validation_{key}": value
                    for key, value in champion_validation.items()
                },
                **{
                    f"march_{key}": value
                    for key, value in champion_march.items()
                },
            },
            {
                "model": "team_specific_venue_challenger",
                "team_prior": float(best["team_prior"]),
                "global_prior": float(best["global_prior"]),
                "C": float(best["C"]),
                "feature_family": str(best["feature_family"]),
                **{
                    f"validation_{key}": value
                    for key, value in challenger_validation.items()
                },
                **{
                    f"march_{key}": value
                    for key, value in challenger_march.items()
                },
            },
        ]
    )

    bootstrap = pd.DataFrame(
        [
            {
                "period": "January-February validation",
                **date_block_bootstrap(
                    core_table.loc[validation, "game_date"],
                    core_table.loc[validation, "home_win"],
                    champion_probability["validation"],
                    challenger_probability["validation"],
                ),
            },
            {
                "period": "March governance",
                **date_block_bootstrap(
                    core_table.loc[march, "game_date"],
                    core_table.loc[march, "home_win"],
                    champion_probability["march"],
                    challenger_probability["march"],
                ),
            },
        ]
    )

    april_metrics = april_descriptive_audit(
        data,
        core_table,
        best,
    )
    april_table = pd.DataFrame(
        [
            {
                "model": "team_specific_venue_challenger",
                **april_metrics,
            }
        ]
    )

    decision = {
        "promote_to_champion": False,
        "selected_challenger": {
            "feature_family": str(best["feature_family"]),
            "team_prior": float(best["team_prior"]),
            "global_prior": float(best["global_prior"]),
            "C": float(best["C"]),
        },
        "interpretation": (
            "The official model already produces team-specific matchup "
            "probabilities. This challenger tests team-specific venue "
            "deviations. It improves validation log loss and is nearly tied "
            "in March, but the paired bootstrap intervals include zero, "
            "March Brier and accuracy do not improve, and the strict April "
            "descriptive score is worse. The evidence is insufficient to "
            "replace the simpler champion."
        ),
        "april_outcomes_used_for_selection": False,
        "march_is_governance_not_pristine_test": True,
    }

    grid.to_csv(
        output_dir / "team_specific_home_effect_grid.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "team_specific_home_effect_summary.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_dir / "team_specific_home_effect_bootstrap.csv",
        index=False,
    )
    april_table.to_csv(
        output_dir / "team_specific_home_effect_april_descriptive.csv",
        index=False,
    )
    with (
        output_dir / "team_specific_home_effect_decision.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(decision, file, indent=2)

    save_figure(
        summary,
        figure_dir / "team_specific_home_effect_comparison.png",
    )

    print(summary.to_string(index=False))
    print()
    print(bootstrap.to_string(index=False))
    print()
    print(json.dumps(decision, indent=2))


def parse_args() -> argparse.Namespace:
    """Define data, output, and figure paths."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate shrunk team-specific home-court effects as a "
            "challenger to the official NBA probability model."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to nba-win-probability-data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/outputs"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("research/figures"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.data,
        arguments.output_dir,
        arguments.figure_dir,
    )
