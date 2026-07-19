"""Leakage-safe governance analysis for the optimized NBA ensemble."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from project_runtime import require_supported_python

require_supported_python()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nba_win_probability as champion


RANDOM_SEED = 365
BOOTSTRAP_REPLICATES = 500
RICH_C_GRID = champion.C_GRID
BT_C_GRID = (0.01, 0.03, 0.10, 0.30, 1.00)
ELO_K_GRID = (5.0, 8.0, 12.0, 20.0)
ELO_HOME_GRID = (0.0, 20.0, 40.0, 60.0)
ELO_MARGIN_GRID = (0.0, 0.5, 1.0)


@dataclass
class RichState:
    """Pregame state containing only completed-game information."""

    games: int = 0
    cumulative_margin: float = 0.0
    ewma_margin: float = 0.0
    cumulative_turnover_advantage: float = 0.0
    ewma_turnover_advantage: float = 0.0
    cumulative_rebound_advantage: float = 0.0
    ewma_rebound_advantage: float = 0.0
    cumulative_foul_advantage: float = 0.0
    ewma_foul_advantage: float = 0.0
    last_game_date: pd.Timestamp | None = None


def update_ewma(previous: float, value: float, games: int, alpha: float) -> float:
    """Apply a postgame EWMA update."""

    return value if games == 0 else alpha * value + (1.0 - alpha) * previous


def build_rich_features(
    data: pd.DataFrame,
    half_life: float = 12.0,
) -> pd.DataFrame:
    """Construct lagged box-score and schedule features without leakage."""

    teams = sorted(set(data["home"]).union(data["away"]))
    states = {team: RichState() for team in teams}
    alpha = 1.0 - math.exp(math.log(0.5) / half_life)
    rows: list[dict[str, object]] = []

    for row in data.itertuples(index=False):
        home_state = states[row.home]
        away_state = states[row.away]
        home_rest = (
            7
            if home_state.last_game_date is None
            else min((row.game_date - home_state.last_game_date).days, 7)
        )
        away_rest = (
            7
            if away_state.last_game_date is None
            else min((row.game_date - away_state.last_game_date).days, 7)
        )
        rows.append(
            {
                "game_id": row.game_id,
                "game_date": row.game_date,
                "home": row.home,
                "away": row.away,
                "home_win": int(row.home_points > row.away_points),
                "net_wins_diff": (
                    row.home_wins
                    - row.home_losses
                    - row.away_wins
                    + row.away_losses
                ),
                "cumulative_margin_diff": (
                    home_state.cumulative_margin
                    - away_state.cumulative_margin
                ),
                "recent_margin_evidence_diff": (
                    home_state.games * home_state.ewma_margin
                    - away_state.games * away_state.ewma_margin
                ),
                "cumulative_turnover_adv_diff": (
                    home_state.cumulative_turnover_advantage
                    - away_state.cumulative_turnover_advantage
                ),
                "recent_turnover_adv_evidence_diff": (
                    home_state.games * home_state.ewma_turnover_advantage
                    - away_state.games * away_state.ewma_turnover_advantage
                ),
                "cumulative_rebound_adv_diff": (
                    home_state.cumulative_rebound_advantage
                    - away_state.cumulative_rebound_advantage
                ),
                "recent_rebound_adv_evidence_diff": (
                    home_state.games * home_state.ewma_rebound_advantage
                    - away_state.games * away_state.ewma_rebound_advantage
                ),
                "cumulative_foul_adv_diff": (
                    home_state.cumulative_foul_advantage
                    - away_state.cumulative_foul_advantage
                ),
                "recent_foul_adv_evidence_diff": (
                    home_state.games * home_state.ewma_foul_advantage
                    - away_state.games * away_state.ewma_foul_advantage
                ),
                "rest_days_diff": home_rest - away_rest,
                "home_back_to_back": int(home_rest == 1),
                "away_back_to_back": int(away_rest == 1),
                "games_played_diff": home_state.games - away_state.games,
            }
        )

        margin = float(row.home_points - row.away_points)
        turnover_advantage = float(row.away_turnovers - row.home_turnovers)
        rebound_advantage = float(row.home_rebounds - row.away_rebounds)
        foul_advantage = float(row.away_fouls - row.home_fouls)

        for state, margin_value, turnover_value, rebound_value, foul_value in (
            (
                home_state,
                margin,
                turnover_advantage,
                rebound_advantage,
                foul_advantage,
            ),
            (
                away_state,
                -margin,
                -turnover_advantage,
                -rebound_advantage,
                -foul_advantage,
            ),
        ):
            state.ewma_margin = update_ewma(
                state.ewma_margin,
                margin_value,
                state.games,
                alpha,
            )
            state.ewma_turnover_advantage = update_ewma(
                state.ewma_turnover_advantage,
                turnover_value,
                state.games,
                alpha,
            )
            state.ewma_rebound_advantage = update_ewma(
                state.ewma_rebound_advantage,
                rebound_value,
                state.games,
                alpha,
            )
            state.ewma_foul_advantage = update_ewma(
                state.ewma_foul_advantage,
                foul_value,
                state.games,
                alpha,
            )
            state.games += 1
            state.cumulative_margin += margin_value
            state.cumulative_turnover_advantage += turnover_value
            state.cumulative_rebound_advantage += rebound_value
            state.cumulative_foul_advantage += foul_value
            state.last_game_date = row.game_date

    return pd.DataFrame(rows)


def metric_values(
    outcome: pd.Series | np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    """Return directly comparable probability diagnostics."""

    return {
        "log_loss": float(log_loss(outcome, probability)),
        "brier_score": float(brier_score_loss(outcome, probability)),
        "roc_auc": float(roc_auc_score(outcome, probability)),
        "mean_probability": float(np.mean(probability)),
        "actual_home_win_rate": float(np.mean(outcome)),
    }


def fit_tuned_logistic(
    table: pd.DataFrame,
    columns: list[str],
    family: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Tune one regularized logistic feature family on January-February."""

    train = table["game_date"] <= champion.TRAIN_END
    validation = table["game_date"].between(
        champion.VALIDATION_START,
        champion.VALIDATION_END,
    )
    train_validation = table["game_date"] <= champion.VALIDATION_END
    march = table["game_date"].between(
        champion.MARCH_START,
        champion.MARCH_END,
    )
    rows: list[dict[str, object]] = []
    validation_predictions: dict[float, np.ndarray] = {}

    for c_value in RICH_C_GRID:
        model = champion.make_model(c_value)
        model.fit(
            table.loc[train, columns],
            table.loc[train, "home_win"],
        )
        probability = model.predict_proba(
            table.loc[validation, columns]
        )[:, 1]
        validation_predictions[c_value] = probability
        rows.append(
            {
                "family": family,
                "C": c_value,
                "features": " + ".join(columns),
                **{
                    f"validation_{key}": value
                    for key, value in metric_values(
                        table.loc[validation, "home_win"],
                        probability,
                    ).items()
                },
            }
        )

    selected = min(
        rows,
        key=lambda row: (
            row["validation_log_loss"],
            row["validation_brier_score"],
            row["C"],
        ),
    )
    selected_c = float(selected["C"])
    march_model = champion.make_model(selected_c)
    march_model.fit(
        table.loc[train_validation, columns],
        table.loc[train_validation, "home_win"],
    )
    march_probability = march_model.predict_proba(
        table.loc[march, columns]
    )[:, 1]
    result = {
        **selected,
        **{
            f"march_{key}": value
            for key, value in metric_values(
                table.loc[march, "home_win"],
                march_probability,
            ).items()
        },
        "validation_probability": validation_predictions[selected_c],
        "march_probability": march_probability,
    }
    return result, rows


def team_difference_matrix(
    table: pd.DataFrame,
    teams: list[str],
) -> np.ndarray:
    """Encode each matchup as +1 home team and -1 away team."""

    index = {team: position for position, team in enumerate(teams)}
    matrix = np.zeros((len(table), len(teams)), dtype=float)
    for row_index, row in enumerate(table.itertuples(index=False)):
        matrix[row_index, index[row.home]] = 1.0
        matrix[row_index, index[row.away]] = -1.0
    return matrix


def tune_bradley_terry(
    table: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Tune a direct regularized team-strength Bradley-Terry model."""

    teams = sorted(set(table["home"]).union(table["away"]))
    matrix = team_difference_matrix(table, teams)
    train = table["game_date"] <= champion.TRAIN_END
    validation = table["game_date"].between(
        champion.VALIDATION_START,
        champion.VALIDATION_END,
    )
    train_validation = table["game_date"] <= champion.VALIDATION_END
    march = table["game_date"].between(
        champion.MARCH_START,
        champion.MARCH_END,
    )

    rows = []
    validation_predictions = {}
    for c_value in BT_C_GRID:
        model = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=5000,
            random_state=RANDOM_SEED,
        )
        model.fit(matrix[train], table.loc[train, "home_win"])
        probability = model.predict_proba(matrix[validation])[:, 1]
        validation_predictions[c_value] = probability
        rows.append(
            {
                "family": "bradley_terry_team_effects",
                "C": c_value,
                "features": "regularized team pair effects",
                **{
                    f"validation_{key}": value
                    for key, value in metric_values(
                        table.loc[validation, "home_win"],
                        probability,
                    ).items()
                },
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["validation_log_loss"],
            row["validation_brier_score"],
            row["C"],
        ),
    )
    selected_c = float(selected["C"])
    model = LogisticRegression(
        C=selected_c,
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    model.fit(
        matrix[train_validation],
        table.loc[train_validation, "home_win"],
    )
    march_probability = model.predict_proba(matrix[march])[:, 1]
    result = {
        **selected,
        **{
            f"march_{key}": value
            for key, value in metric_values(
                table.loc[march, "home_win"],
                march_probability,
            ).items()
        },
        "validation_probability": validation_predictions[selected_c],
        "march_probability": march_probability,
    }
    return result, rows


def elo_probabilities(
    data: pd.DataFrame,
    k_factor: float,
    home_rating_points: float,
    margin_weight: float,
) -> np.ndarray:
    """Replay a deterministic margin-sensitive Elo rating."""

    teams = sorted(set(data["home"]).union(data["away"]))
    ratings = {team: 1500.0 for team in teams}
    probabilities = np.empty(len(data), dtype=float)

    for index, row in enumerate(data.itertuples(index=False)):
        rating_difference = ratings[row.home] - ratings[row.away]
        probability = 1.0 / (
            1.0
            + 10.0
            ** (
                -(
                    rating_difference + home_rating_points
                )
                / 400.0
            )
        )
        probabilities[index] = probability
        outcome = float(row.home_points > row.away_points)
        margin = abs(float(row.home_points - row.away_points))
        multiplier = 1.0 + margin_weight * math.log1p(margin)
        update = k_factor * multiplier * (outcome - probability)
        ratings[row.home] += update
        ratings[row.away] -= update

    return probabilities


def tune_elo(
    data: pd.DataFrame,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Tune a compact leakage-safe Elo challenger on January-February."""

    validation = data["game_date"].between(
        champion.VALIDATION_START,
        champion.VALIDATION_END,
    )
    march = data["game_date"].between(
        champion.MARCH_START,
        champion.MARCH_END,
    )
    outcome = (data["home_points"] > data["away_points"]).astype(int)
    rows = []
    prediction_cache = {}

    for k_factor in ELO_K_GRID:
        for home_points in ELO_HOME_GRID:
            for margin_weight in ELO_MARGIN_GRID:
                probability = elo_probabilities(
                    data,
                    k_factor,
                    home_points,
                    margin_weight,
                )
                key = (k_factor, home_points, margin_weight)
                prediction_cache[key] = probability
                rows.append(
                    {
                        "family": "margin_sensitive_elo",
                        "k_factor": k_factor,
                        "home_rating_points": home_points,
                        "margin_weight": margin_weight,
                        "features": "sequential outcomes and score margin",
                        **{
                            f"validation_{name}": value
                            for name, value in metric_values(
                                outcome.loc[validation],
                                probability[validation],
                            ).items()
                        },
                    }
                )
    selected = min(
        rows,
        key=lambda row: (
            row["validation_log_loss"],
            row["validation_brier_score"],
            row["k_factor"],
        ),
    )
    key = (
        float(selected["k_factor"]),
        float(selected["home_rating_points"]),
        float(selected["margin_weight"]),
    )
    probability = prediction_cache[key]
    result = {
        **selected,
        **{
            f"march_{name}": value
            for name, value in metric_values(
                outcome.loc[march],
                probability[march],
            ).items()
        },
        "validation_probability": probability[validation],
        "march_probability": probability[march],
    }
    return result, rows


def champion_predictions(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recreate ensemble and validation-best single forward probabilities."""

    feature_tables = champion.build_component_feature_tables(data)
    validation_matrix, validation_components, _ = champion.component_predictions(
        feature_tables,
        champion.TRAIN_END,
        champion.VALIDATION_START,
        champion.VALIDATION_END,
    )
    validation_grid = validation_components.sort_values(
        ["log_loss", "brier_score", "C", "half_life"]
    ).reset_index(drop=True)
    best = validation_grid.iloc[0]
    best_index = (
        list(champion.HALF_LIFE_GRID).index(float(best["half_life"]))
        * len(champion.C_GRID)
        + list(champion.C_GRID).index(float(best["C"]))
    )
    validation_ensemble = champion.ensemble_probability(validation_matrix)
    validation_single = validation_matrix[:, best_index]

    march_matrix, _, _ = champion.component_predictions(
        feature_tables,
        champion.VALIDATION_END,
        champion.MARCH_START,
        champion.MARCH_END,
    )
    march_ensemble = champion.ensemble_probability(march_matrix)
    march_single = march_matrix[:, best_index]
    return (
        validation_grid,
        validation_ensemble,
        validation_single,
        march_ensemble,
        march_single,
    )


def date_block_bootstrap(
    dates: pd.Series,
    outcome: pd.Series,
    champion_probability: np.ndarray,
    challenger_probability: np.ndarray,
) -> dict[str, float]:
    """Bootstrap paired log-loss differences by game date."""

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).to_numpy(),
            "outcome": np.asarray(outcome),
            "champion": champion_probability,
            "challenger": challenger_probability,
        }
    )
    unique_dates = np.array(sorted(frame["date"].unique()))
    random = np.random.default_rng(RANDOM_SEED)
    differences = np.empty(BOOTSTRAP_REPLICATES)

    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_dates = random.choice(
            unique_dates,
            size=len(unique_dates),
            replace=True,
        )
        indices = np.concatenate(
            [
                frame.index[frame["date"] == date].to_numpy()
                for date in sampled_dates
            ]
        )
        sample = frame.loc[indices]
        differences[replicate] = (
            log_loss(sample["outcome"], sample["challenger"])
            - log_loss(sample["outcome"], sample["champion"])
        )

    observed = (
        log_loss(frame["outcome"], frame["challenger"])
        - log_loss(frame["outcome"], frame["champion"])
    )
    lower, median, upper = np.quantile(
        differences,
        [0.025, 0.5, 0.975],
    )
    return {
        "observed_log_loss_difference_challenger_minus_champion": float(
            observed
        ),
        "bootstrap_2_5_percent": float(lower),
        "bootstrap_median": float(median),
        "bootstrap_97_5_percent": float(upper),
        "probability_challenger_is_worse": float(
            np.mean(differences > 0.0)
        ),
        "replicates": BOOTSTRAP_REPLICATES,
    }


def calibration_diagnostics(
    outcome: pd.Series,
    probability: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report calibration-in-the-large, slope, and reliability bins."""

    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    log_odds = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibration_model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    calibration_model.fit(log_odds, outcome)
    summary = pd.DataFrame(
        [
            {
                "n": len(outcome),
                **metric_values(outcome, probability),
                "calibration_gap_actual_minus_predicted": float(
                    np.mean(outcome) - np.mean(probability)
                ),
                "calibration_intercept": float(
                    calibration_model.intercept_[0]
                ),
                "calibration_slope": float(
                    calibration_model.coef_[0, 0]
                ),
            }
        ]
    )
    bins = champion.calibration_bins(outcome, probability, bins=5)
    return summary, bins


def feature_collinearity(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate correlation and variance-inflation diagnostics."""

    development = table.loc[
        table["game_date"] <= champion.MARCH_END,
        champion.FEATURE_COLUMNS,
    ].astype(float)
    correlation = development.corr()
    rows = []
    for feature in champion.FEATURE_COLUMNS:
        target = development[feature].to_numpy()
        others = [
            column
            for column in champion.FEATURE_COLUMNS
            if column != feature
        ]
        design = np.column_stack(
            [
                np.ones(len(development)),
                development[others].to_numpy(),
            ]
        )
        coefficients, *_ = np.linalg.lstsq(
            design,
            target,
            rcond=None,
        )
        residual = target - design @ coefficients
        total = np.sum((target - target.mean()) ** 2)
        residual_sum = np.sum(residual**2)
        r_squared = 1.0 - residual_sum / total
        rows.append(
            {
                "feature": feature,
                "r_squared_against_other_features": float(r_squared),
                "variance_inflation_factor": float(
                    1.0 / (1.0 - r_squared)
                ),
            }
        )
    return correlation, pd.DataFrame(rows)


def monthly_backtest(
    data: pd.DataFrame,
    best_half_life: float,
    best_c: float,
) -> pd.DataFrame:
    """Compare the ensemble and best single component by month."""

    feature_tables = champion.build_component_feature_tables(data)
    rows = []
    periods = (
        pd.Period("2025-12"),
        pd.Period("2026-01"),
        pd.Period("2026-02"),
        pd.Period("2026-03"),
    )

    for period in periods:
        matrix, _, _ = champion.component_predictions(
            feature_tables,
            period.start_time - pd.Timedelta(days=1),
            period.start_time,
            period.end_time,
        )
        ensemble = champion.ensemble_probability(matrix)
        reference = feature_tables[12.0]
        evaluation = reference["game_date"].between(
            period.start_time,
            period.end_time,
        )
        outcome = reference.loc[evaluation, "home_win"]

        best_table = feature_tables[best_half_life]
        training = best_table["game_date"] < period.start_time
        best_evaluation = best_table["game_date"].between(
            period.start_time,
            period.end_time,
        )
        model = champion.make_model(best_c)
        model.fit(
            best_table.loc[training, champion.FEATURE_COLUMNS],
            best_table.loc[training, "home_win"],
        )
        single = model.predict_proba(
            best_table.loc[
                best_evaluation,
                champion.FEATURE_COLUMNS,
            ]
        )[:, 1]

        for name, probability in (
            ("uniform_40_component_logistic_ensemble", ensemble),
            ("validation_best_single_component", single),
        ):
            rows.append(
                {
                    "model": name,
                    "month": str(period),
                    "training_games": int(training.sum()),
                    "evaluation_games": int(evaluation.sum()),
                    **metric_values(outcome, probability),
                }
            )
    return pd.DataFrame(rows)


def save_figures(
    comparison: pd.DataFrame,
    monthly: pd.DataFrame,
    calibration_bins: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Create focused governance figures."""

    figure_dir.mkdir(parents=True, exist_ok=True)

    plot_table = comparison.sort_values("march_log_loss")
    figure = plt.figure(figsize=(10, 5))
    axis = figure.add_subplot(111)
    positions = np.arange(len(plot_table))
    width = 0.38
    axis.bar(
        positions - width / 2,
        plot_table["validation_log_loss"],
        width=width,
        label="January-February validation",
    )
    axis.bar(
        positions + width / 2,
        plot_table["march_log_loss"],
        width=width,
        label="March governance check",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        plot_table["family"],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Log loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figure_dir / "governance_model_comparison.png",
        dpi=180,
    )
    plt.close(figure)

    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    for model_name, group in monthly.groupby("model"):
        axis.plot(
            group["month"],
            group["log_loss"],
            marker="o",
            label=model_name,
        )
    axis.set_ylabel("Expanding-window monthly log loss")
    axis.set_xlabel("Evaluation month")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figure_dir / "monthly_model_stability.png",
        dpi=180,
    )
    plt.close(figure)

    figure = plt.figure(figsize=(6, 6))
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.plot(
        calibration_bins["mean_probability"],
        calibration_bins["actual_home_win_rate"],
        marker="o",
    )
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed home-win rate")
    figure.tight_layout()
    figure.savefig(
        figure_dir / "march_calibration_reliability.png",
        dpi=180,
    )
    plt.close(figure)


def run(
    data_path: Path,
    output_dir: Path,
    figure_dir: Path,
) -> None:
    """Execute the complete core-dependency governance analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    data = champion.load_and_validate(data_path)
    reference = champion.build_sequential_features(data, 12.0)
    rich = build_rich_features(data, 12.0)

    (
        validation_grid,
        validation_ensemble,
        validation_single,
        march_ensemble,
        march_single,
    ) = champion_predictions(data)
    best = validation_grid.iloc[0]
    best_half_life = float(best["half_life"])
    best_c = float(best["C"])

    validation = reference["game_date"].between(
        champion.VALIDATION_START,
        champion.VALIDATION_END,
    )
    march = reference["game_date"].between(
        champion.MARCH_START,
        champion.MARCH_END,
    )
    validation_outcome = reference.loc[validation, "home_win"]
    march_outcome = reference.loc[march, "home_win"]

    core_columns = champion.FEATURE_COLUMNS
    box_columns = [
        "cumulative_turnover_adv_diff",
        "recent_turnover_adv_evidence_diff",
        "cumulative_rebound_adv_diff",
        "recent_rebound_adv_evidence_diff",
        "cumulative_foul_adv_diff",
        "recent_foul_adv_evidence_diff",
    ]
    schedule_columns = [
        "rest_days_diff",
        "home_back_to_back",
        "away_back_to_back",
        "games_played_diff",
    ]

    candidates: dict[str, dict[str, object]] = {
        "uniform_40_component_logistic_ensemble": {
            "family": "uniform_40_component_logistic_ensemble",
            "features": "three core signals across 40 fixed components",
            "validation_log_loss": log_loss(
                validation_outcome,
                validation_ensemble,
            ),
            "validation_brier_score": brier_score_loss(
                validation_outcome,
                validation_ensemble,
            ),
            "validation_roc_auc": roc_auc_score(
                validation_outcome,
                validation_ensemble,
            ),
            "march_log_loss": log_loss(
                march_outcome,
                march_ensemble,
            ),
            "march_brier_score": brier_score_loss(
                march_outcome,
                march_ensemble,
            ),
            "march_roc_auc": roc_auc_score(
                march_outcome,
                march_ensemble,
            ),
            "validation_probability": validation_ensemble,
            "march_probability": march_ensemble,
        },
        "validation_best_single_component": {
            "family": "validation_best_single_component",
            "features": "three core signals",
            "C": best_c,
            "half_life": best_half_life,
            "validation_log_loss": log_loss(
                validation_outcome,
                validation_single,
            ),
            "validation_brier_score": brier_score_loss(
                validation_outcome,
                validation_single,
            ),
            "validation_roc_auc": roc_auc_score(
                validation_outcome,
                validation_single,
            ),
            "march_log_loss": log_loss(
                march_outcome,
                march_single,
            ),
            "march_brier_score": brier_score_loss(
                march_outcome,
                march_single,
            ),
            "march_roc_auc": roc_auc_score(
                march_outcome,
                march_single,
            ),
            "validation_probability": validation_single,
            "march_probability": march_single,
        },
    }

    grid_rows: list[dict[str, object]] = []
    for family, columns in (
        ("core_plus_box_scores", [*core_columns, *box_columns]),
        ("core_plus_schedule", [*core_columns, *schedule_columns]),
        (
            "core_plus_box_scores_and_schedule",
            [*core_columns, *box_columns, *schedule_columns],
        ),
    ):
        result, rows = fit_tuned_logistic(rich, columns, family)
        candidates[family] = result
        grid_rows.extend(rows)

    bt_result, bt_rows = tune_bradley_terry(reference)
    candidates["bradley_terry_team_effects"] = bt_result
    grid_rows.extend(bt_rows)

    elo_result, elo_rows = tune_elo(data)
    candidates["margin_sensitive_elo"] = elo_result
    grid_rows.extend(elo_rows)

    comparison_columns = [
        "family",
        "features",
        "validation_log_loss",
        "validation_brier_score",
        "validation_roc_auc",
        "march_log_loss",
        "march_brier_score",
        "march_roc_auc",
    ]
    comparison = pd.DataFrame(
        [
            {
                key: value
                for key, value in candidate.items()
                if key in comparison_columns
            }
            for candidate in candidates.values()
        ]
    ).sort_values("validation_log_loss").reset_index(drop=True)

    bootstrap_rows = []
    for split_name, dates, outcome, champion_probability, probability_key in (
        (
            "January-February validation",
            reference.loc[validation, "game_date"],
            validation_outcome,
            validation_ensemble,
            "validation_probability",
        ),
        (
            "March governance check",
            reference.loc[march, "game_date"],
            march_outcome,
            march_ensemble,
            "march_probability",
        ),
    ):
        for family, candidate in candidates.items():
            if family == "uniform_40_component_logistic_ensemble":
                continue
            bootstrap_rows.append(
                {
                    "split": split_name,
                    "challenger": family,
                    **date_block_bootstrap(
                        dates,
                        outcome,
                        champion_probability,
                        np.asarray(candidate[probability_key]),
                    ),
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)

    monthly = monthly_backtest(data, best_half_life, best_c)
    calibration_summary, calibration_bins = calibration_diagnostics(
        march_outcome,
        march_ensemble,
    )
    correlation, vif = feature_collinearity(reference)

    elapsed = time.perf_counter() - start_time
    runtime = pd.DataFrame(
        [
            {
                "analysis": "core_governance_workflow",
                "games": len(data),
                "ensemble_components": (
                    len(champion.HALF_LIFE_GRID)
                    * len(champion.C_GRID)
                ),
                "elapsed_seconds": elapsed,
            }
        ]
    )

    comparison.to_csv(
        output_dir / "governance_model_comparison.csv",
        index=False,
    )
    pd.DataFrame(grid_rows).to_csv(
        output_dir / "governance_candidate_grid.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_dir / "governance_bootstrap_differences.csv",
        index=False,
    )
    monthly.to_csv(
        output_dir / "governance_monthly_backtest.csv",
        index=False,
    )
    calibration_summary.to_csv(
        output_dir / "governance_march_calibration_summary.csv",
        index=False,
    )
    calibration_bins.to_csv(
        output_dir / "governance_march_calibration_bins.csv",
        index=False,
    )
    correlation.to_csv(
        output_dir / "governance_feature_correlation.csv",
    )
    vif.to_csv(
        output_dir / "governance_feature_vif.csv",
        index=False,
    )
    runtime.to_csv(
        output_dir / "governance_runtime.csv",
        index=False,
    )

    selection = {
        "selected_model": "uniform_40_component_logistic_ensemble",
        "selection_basis": (
            "Fixed equal-weight probability averaging across the complete "
            "fixed candidate half-life and L2 grid."
        ),
        "validation_log_loss": float(
            candidates[
                "uniform_40_component_logistic_ensemble"
            ]["validation_log_loss"]
        ),
        "march_governance_check_log_loss": float(
            candidates[
                "uniform_40_component_logistic_ensemble"
            ]["march_log_loss"]
        ),
        "validation_best_single_component": {
            "half_life": best_half_life,
            "C": best_c,
            "validation_log_loss": float(
                candidates[
                    "validation_best_single_component"
                ]["validation_log_loss"]
            ),
            "march_log_loss": float(
                candidates[
                    "validation_best_single_component"
                ]["march_log_loss"]
            ),
        },
        "ensemble_weights_tuned": False,
        "march_used_for_promotion_governance": True,
        "march_used_for_component_or_weight_tuning": False,
        "april_outcomes_used_for_component_or_weight_tuning": False,
        "april_outcomes_viewed_descriptively": True,
        "decision": (
            "Promote the ensemble for the April target because it improves "
            "January-February validation log loss before April, preserves "
            "the direction in the March governance period, reduces "
            "single-grid-point selection risk, and retains the same "
            "interpretable feature family at negligible runtime cost."
        ),
    }
    with (output_dir / "governance_selection_decision.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(selection, file, indent=2)

    save_figures(
        comparison,
        monthly,
        calibration_bins,
        figure_dir,
    )
    print(json.dumps(selection, indent=2))


def parse_args() -> argparse.Namespace:
    """Define command-line paths."""

    parser = argparse.ArgumentParser(
        description="Run leakage-safe model governance for the NBA submission."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.data, arguments.output_dir, arguments.figure_dir)
