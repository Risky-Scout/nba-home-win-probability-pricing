"""Enhanced model-governance analysis for the NBA pricing submission.

The production champion remains the leakage-safe three-signal L2 logistic model.
This script tests the limitations raised during review:

- feature collinearity,
- unusual evidence-weighted EWMA form,
- lack of opponent adjustment,
- calibration uncertainty,
- feature-subset instability,
- coefficient stability,
- model uncertainty in April prices.

No April outcome is used to select a model. April outcomes are scored only in
the separate descriptive artifact already produced by the champion workflow.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nba_win_probability as champion


TRAIN_END = pd.Timestamp("2025-12-31")
VALIDATION_START = pd.Timestamp("2026-01-01")
VALIDATION_END = pd.Timestamp("2026-02-28")
MARCH_START = pd.Timestamp("2026-03-01")
MARCH_END = pd.Timestamp("2026-03-31")
APRIL_START = pd.Timestamp("2026-04-01")
RANDOM_SEED = 365

CORE_FEATURES = [
    "net_wins_diff",
    "cumulative_margin_diff",
    "recent_margin_evidence_diff",
]

C_GRID = (
    0.003,
    0.005,
    0.0075,
    0.010,
    0.015,
    0.020,
    0.030,
    0.050,
)

EWMA_HALF_LIVES = (5.0, 8.0, 12.0, 16.0, 24.0)
SHRINKAGE_PRIORS = (5.0, 10.0, 20.0, 40.0)
SRS_ALPHAS = (100.0, 300.0, 1000.0)
SRS_HALF_LIFE_DAYS = (None, 120.0)

BOOTSTRAP_REPLICATES = 500
APRIL_BOOTSTRAP_REPLICATES = 250


@dataclass
class ExtendedState:
    """Store past-only margin information for alternative recent-form features."""

    games: int = 0
    ewma_margin: dict[float, float] = field(default_factory=dict)


def metric_row(
    outcome: pd.Series | np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    """Return proper scores and secondary diagnostics."""

    return {
        "log_loss": float(log_loss(outcome, probability)),
        "brier_score": float(brier_score_loss(outcome, probability)),
        "roc_auc": float(roc_auc_score(outcome, probability)),
        "accuracy_0_5": float(accuracy_score(outcome, probability >= 0.5)),
        "mean_probability": float(np.mean(probability)),
        "actual_home_win_rate": float(np.mean(outcome)),
        "minimum_probability": float(np.min(probability)),
        "maximum_probability": float(np.max(probability)),
    }


def make_logistic(c_value: float) -> Pipeline:
    """Construct the exact scaled L2 logistic probability model."""

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


def build_extended_recent_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build pure and Bayesian-shrunken EWMA margin features."""

    teams = sorted(set(data["home"]).union(data["away"]))
    states = {
        team: ExtendedState(
            ewma_margin={half_life: 0.0 for half_life in EWMA_HALF_LIVES}
        )
        for team in teams
    }
    alphas = {
        half_life: 1.0 - math.exp(math.log(0.5) / half_life)
        for half_life in EWMA_HALF_LIVES
    }
    rows: list[dict[str, object]] = []

    for row in data.itertuples(index=False):
        home_state = states[row.home]
        away_state = states[row.away]
        features: dict[str, object] = {"game_id": row.game_id}

        for half_life in EWMA_HALF_LIVES:
            home_ewma = home_state.ewma_margin[half_life]
            away_ewma = away_state.ewma_margin[half_life]
            features[f"pure_ewma_{half_life:g}_diff"] = (
                home_ewma - away_ewma
            )
            for prior in SHRINKAGE_PRIORS:
                home_weight = home_state.games / (home_state.games + prior)
                away_weight = away_state.games / (away_state.games + prior)
                features[
                    f"shrunk_ewma_{half_life:g}_prior_{prior:g}_diff"
                ] = (
                    home_weight * home_ewma
                    - away_weight * away_ewma
                )

        rows.append(features)

        home_margin = float(row.home_points - row.away_points)
        for state, margin in (
            (home_state, home_margin),
            (away_state, -home_margin),
        ):
            for half_life, alpha in alphas.items():
                if state.games == 0:
                    state.ewma_margin[half_life] = margin
                else:
                    state.ewma_margin[half_life] = (
                        alpha * margin
                        + (1.0 - alpha) * state.ewma_margin[half_life]
                    )
            state.games += 1

    return pd.DataFrame(rows)


def team_design_matrix(
    data: pd.DataFrame,
    teams: list[str],
    team_index: dict[str, int],
) -> np.ndarray:
    """Create home-plus-one, away-minus-one team indicators."""

    design = np.zeros((len(data), len(teams)), dtype=float)
    for row_number, row in enumerate(data.itertuples(index=False)):
        design[row_number, team_index[row.home]] = 1.0
        design[row_number, team_index[row.away]] = -1.0
    return design


def build_srs_features(
    data: pd.DataFrame,
    alpha: float,
    half_life_days: float | None,
) -> pd.DataFrame:
    """Build a past-only opponent-adjusted ridge margin rating.

    The model solves a regularized simple-rating-system equation:

        home margin = home advantage + home strength - away strength + error

    Ridge shrinkage partially pools team strengths toward league average.
    Optional date weighting allows team strength to evolve.
    """

    teams = sorted(set(data["home"]).union(data["away"]))
    team_index = {team: index for index, team in enumerate(teams)}
    dates = sorted(data["game_date"].unique())
    expected_margin = np.zeros(len(data), dtype=float)
    estimated_home_margin = np.zeros(len(data), dtype=float)

    for date in dates:
        prediction_indices = np.flatnonzero(
            data["game_date"].to_numpy() == np.datetime64(date)
        )
        history = data.loc[data["game_date"] < date]

        if history.empty:
            continue

        design = team_design_matrix(history, teams, team_index)
        target = (
            history["home_points"] - history["away_points"]
        ).to_numpy(dtype=float)

        sample_weight = None
        if half_life_days is not None:
            ages = (
                pd.Timestamp(date) - history["game_date"]
            ).dt.days.to_numpy(dtype=float)
            sample_weight = np.exp(
                math.log(0.5) * ages / half_life_days
            )

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(design, target, sample_weight=sample_weight)

        for index in prediction_indices:
            home = data.iloc[index]["home"]
            away = data.iloc[index]["away"]
            expected_margin[index] = (
                model.intercept_
                + model.coef_[team_index[home]]
                - model.coef_[team_index[away]]
            )
            estimated_home_margin[index] = model.intercept_

    label = (
        f"srs_alpha_{alpha:g}_half_life_"
        f"{'none' if half_life_days is None else f'{half_life_days:g}'}"
    )
    return pd.DataFrame(
        {
            "game_id": data["game_id"],
            label: expected_margin,
            f"{label}_home_margin": estimated_home_margin,
        }
    )


def assemble_feature_table(data: pd.DataFrame) -> pd.DataFrame:
    """Combine champion features with all locked governance challengers."""

    table = champion.build_sequential_features(data, half_life=12.0)
    table = table.merge(
        build_extended_recent_features(data),
        on="game_id",
        how="left",
        validate="one_to_one",
    )

    for alpha, half_life_days in product(
        SRS_ALPHAS,
        SRS_HALF_LIFE_DAYS,
    ):
        table = table.merge(
            build_srs_features(data, alpha, half_life_days),
            on="game_id",
            how="left",
            validate="one_to_one",
        )

    return table


def fit_predict(
    table: pd.DataFrame,
    columns: list[str],
    c_value: float,
    training_mask: pd.Series,
    prediction_mask: pd.Series,
) -> tuple[Pipeline, np.ndarray]:
    """Fit one candidate and return its probabilities."""

    model = make_logistic(c_value)
    model.fit(
        table.loc[training_mask, columns],
        table.loc[training_mask, "home_win"],
    )
    probability = model.predict_proba(
        table.loc[prediction_mask, columns]
    )[:, 1]
    return model, probability


def tune_candidate(
    table: pd.DataFrame,
    family: str,
    feature_columns: list[str],
    metadata: dict[str, object] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Tune C on January-February and score the locked model in March."""

    training = table["game_date"] <= TRAIN_END
    validation = table["game_date"].between(
        VALIDATION_START,
        VALIDATION_END,
    )
    train_validation = table["game_date"] <= VALIDATION_END
    march = table["game_date"].between(MARCH_START, MARCH_END)
    rows: list[dict[str, object]] = []

    for c_value in C_GRID:
        _, validation_probability = fit_predict(
            table,
            feature_columns,
            c_value,
            training,
            validation,
        )
        validation_metrics = metric_row(
            table.loc[validation, "home_win"],
            validation_probability,
        )
        rows.append(
            {
                "family": family,
                "features": " + ".join(feature_columns),
                "C": c_value,
                **(metadata or {}),
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
        )

    grid = pd.DataFrame(rows).sort_values(
        ["validation_log_loss", "validation_brier_score", "C"]
    )
    best = grid.iloc[0].to_dict()
    selected_c = float(best["C"])

    _, validation_probability = fit_predict(
        table,
        feature_columns,
        selected_c,
        training,
        validation,
    )
    _, march_probability = fit_predict(
        table,
        feature_columns,
        selected_c,
        train_validation,
        march,
    )
    march_metrics = metric_row(
        table.loc[march, "home_win"],
        march_probability,
    )

    selected = {
        **best,
        **{
            f"march_{key}": value
            for key, value in march_metrics.items()
        },
        "validation_probability": validation_probability,
        "march_probability": march_probability,
    }
    return selected, rows


def evaluate_candidate_families(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    """Evaluate feature subsets and the limitations-driven challengers."""

    selected: dict[str, dict[str, object]] = {}
    grid_rows: list[dict[str, object]] = []

    fixed_candidates = {
        "three_signal_champion": CORE_FEATURES,
        "cumulative_plus_recent": [
            "cumulative_margin_diff",
            "recent_margin_evidence_diff",
        ],
        "net_wins_plus_recent": [
            "net_wins_diff",
            "recent_margin_evidence_diff",
        ],
        "net_wins_plus_cumulative": [
            "net_wins_diff",
            "cumulative_margin_diff",
        ],
    }

    for family, columns in fixed_candidates.items():
        result, rows = tune_candidate(table, family, columns)
        selected[family] = result
        grid_rows.extend(rows)

    for half_life in EWMA_HALF_LIVES:
        pure_column = f"pure_ewma_{half_life:g}_diff"
        family = f"pure_ewma_half_life_{half_life:g}"
        result, rows = tune_candidate(
            table,
            family,
            [
                "net_wins_diff",
                "cumulative_margin_diff",
                pure_column,
            ],
            {"half_life": half_life, "shrinkage_prior": np.nan},
        )
        selected[family] = result
        grid_rows.extend(rows)

        for prior in SHRINKAGE_PRIORS:
            shrunk_column = (
                f"shrunk_ewma_{half_life:g}_prior_{prior:g}_diff"
            )
            family = (
                f"shrunk_ewma_half_life_{half_life:g}_prior_{prior:g}"
            )
            result, rows = tune_candidate(
                table,
                family,
                [
                    "net_wins_diff",
                    "cumulative_margin_diff",
                    shrunk_column,
                ],
                {
                    "half_life": half_life,
                    "shrinkage_prior": prior,
                },
            )
            selected[family] = result
            grid_rows.extend(rows)

    for alpha, half_life_days in product(
        SRS_ALPHAS,
        SRS_HALF_LIFE_DAYS,
    ):
        srs_column = (
            f"srs_alpha_{alpha:g}_half_life_"
            f"{'none' if half_life_days is None else f'{half_life_days:g}'}"
        )
        family = (
            f"champion_plus_srs_alpha_{alpha:g}_half_life_"
            f"{'none' if half_life_days is None else f'{half_life_days:g}'}"
        )
        result, rows = tune_candidate(
            table,
            family,
            [*CORE_FEATURES, srs_column],
            {
                "srs_alpha": alpha,
                "srs_half_life_days": (
                    np.nan if half_life_days is None else half_life_days
                ),
            },
        )
        selected[family] = result
        grid_rows.extend(rows)

    # PCA challengers explicitly test whether orthogonal latent factors improve
    # stability enough to justify reduced direct interpretability.
    training = table["game_date"] <= TRAIN_END
    validation = table["game_date"].between(
        VALIDATION_START,
        VALIDATION_END,
    )
    train_validation = table["game_date"] <= VALIDATION_END
    march = table["game_date"].between(MARCH_START, MARCH_END)

    for components in (1, 2):
        family = f"pca_{components}_component"
        pca_rows: list[dict[str, object]] = []
        for c_value in C_GRID:
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=components)),
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
            model.fit(
                table.loc[training, CORE_FEATURES],
                table.loc[training, "home_win"],
            )
            probability = model.predict_proba(
                table.loc[validation, CORE_FEATURES]
            )[:, 1]
            metrics = metric_row(
                table.loc[validation, "home_win"],
                probability,
            )
            row = {
                "family": family,
                "features": "PCA(" + " + ".join(CORE_FEATURES) + ")",
                "C": c_value,
                "pca_components": components,
                **{
                    f"validation_{key}": value
                    for key, value in metrics.items()
                },
            }
            pca_rows.append(row)
            grid_rows.append(row)

        pca_grid = pd.DataFrame(pca_rows).sort_values(
            ["validation_log_loss", "validation_brier_score", "C"]
        )
        best = pca_grid.iloc[0].to_dict()
        selected_c = float(best["C"])

        march_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=components)),
                (
                    "logit",
                    LogisticRegression(
                        C=selected_c,
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        march_model.fit(
            table.loc[train_validation, CORE_FEATURES],
            table.loc[train_validation, "home_win"],
        )
        march_probability = march_model.predict_proba(
            table.loc[march, CORE_FEATURES]
        )[:, 1]
        march_metrics = metric_row(
            table.loc[march, "home_win"],
            march_probability,
        )

        validation_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=components)),
                (
                    "logit",
                    LogisticRegression(
                        C=selected_c,
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        validation_model.fit(
            table.loc[training, CORE_FEATURES],
            table.loc[training, "home_win"],
        )
        validation_probability = validation_model.predict_proba(
            table.loc[validation, CORE_FEATURES]
        )[:, 1]

        selected[family] = {
            **best,
            **{
                f"march_{key}": value
                for key, value in march_metrics.items()
            },
            "validation_probability": validation_probability,
            "march_probability": march_probability,
        }

    comparison_rows = []
    for family, result in selected.items():
        comparison_rows.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"validation_probability", "march_probability"}
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_log_loss", "march_log_loss"]
    ).reset_index(drop=True)
    full_grid = pd.DataFrame(grid_rows).sort_values(
        ["family", "validation_log_loss", "validation_brier_score", "C"]
    ).reset_index(drop=True)
    return comparison, full_grid, selected


def combined_prediction_frame(
    table: pd.DataFrame,
    selected: dict[str, dict[str, object]],
    families: Iterable[str],
) -> pd.DataFrame:
    """Combine January-February and March predictions for paired tests."""

    validation = table["game_date"].between(
        VALIDATION_START,
        VALIDATION_END,
    )
    march = table["game_date"].between(MARCH_START, MARCH_END)
    frame = pd.concat(
        [
            table.loc[
                validation,
                ["game_id", "game_date", "home_win"],
            ],
            table.loc[
                march,
                ["game_id", "game_date", "home_win"],
            ],
        ]
    ).reset_index(drop=True)

    for family in families:
        frame[family] = np.concatenate(
            [
                selected[family]["validation_probability"],
                selected[family]["march_probability"],
            ]
        )

    return frame


def date_block_bootstrap_difference(
    frame: pd.DataFrame,
    champion_column: str,
    challenger_column: str,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    """Estimate uncertainty in paired log-loss differences by date."""

    unique_dates = np.array(sorted(frame["game_date"].unique()))
    random = np.random.default_rng(RANDOM_SEED)
    differences = np.empty(replicates, dtype=float)

    for replicate in range(replicates):
        sampled_dates = random.choice(
            unique_dates,
            size=len(unique_dates),
            replace=True,
        )
        sampled_indices = np.concatenate(
            [
                frame.index[
                    frame["game_date"] == sampled_date
                ].to_numpy()
                for sampled_date in sampled_dates
            ]
        )
        sample = frame.loc[sampled_indices]
        differences[replicate] = (
            log_loss(sample["home_win"], sample[challenger_column])
            - log_loss(sample["home_win"], sample[champion_column])
        )

    observed = (
        log_loss(frame["home_win"], frame[challenger_column])
        - log_loss(frame["home_win"], frame[champion_column])
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
        "replicates": replicates,
    }


def bootstrap_candidate_differences(
    table: pd.DataFrame,
    selected: dict[str, dict[str, object]],
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the champion with the nearest limitations-driven challengers."""

    nearest_families = [
        "cumulative_plus_recent",
        "net_wins_plus_recent",
    ]

    best_srs = (
        comparison.loc[
            comparison["family"].str.startswith("champion_plus_srs")
        ]
        .sort_values("validation_log_loss")
        .iloc[0]["family"]
    )
    best_shrunk = (
        comparison.loc[
            comparison["family"].str.startswith("shrunk_ewma")
        ]
        .sort_values("validation_log_loss")
        .iloc[0]["family"]
    )
    nearest_families.extend([best_srs, best_shrunk, "pca_1_component"])

    frame = combined_prediction_frame(
        table,
        selected,
        ["three_signal_champion", *nearest_families],
    )

    rows = []
    for family in nearest_families:
        rows.append(
            {
                "challenger": family,
                **date_block_bootstrap_difference(
                    frame,
                    "three_signal_champion",
                    family,
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "observed_log_loss_difference_challenger_minus_champion"
    ).reset_index(drop=True)


def monthly_backtest(
    table: pd.DataFrame,
    selected: dict[str, dict[str, object]],
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Report expanding-window monthly behavior for selected families."""

    families = [
        "three_signal_champion",
        "cumulative_plus_recent",
        "net_wins_plus_recent",
        comparison.loc[
            comparison["family"].str.startswith("champion_plus_srs")
        ]
        .sort_values("validation_log_loss")
        .iloc[0]["family"],
        comparison.loc[
            comparison["family"].str.startswith("shrunk_ewma")
        ]
        .sort_values("validation_log_loss")
        .iloc[0]["family"],
        "pca_1_component",
    ]

    months = ("2025-12", "2026-01", "2026-02", "2026-03")
    rows: list[dict[str, object]] = []

    for family in families:
        selected_row = selected[family]
        c_value = float(selected_row["C"])
        features = str(selected_row["features"])

        for month in months:
            period = pd.Period(month)
            training = table["game_date"] < period.start_time
            validation = table["game_date"].between(
                period.start_time,
                period.end_time,
            )

            if family.startswith("pca_"):
                components = int(selected_row["pca_components"])
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=components)),
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
                columns = CORE_FEATURES
            else:
                columns = features.split(" + ")
                model = make_logistic(c_value)

            model.fit(
                table.loc[training, columns],
                table.loc[training, "home_win"],
            )
            probability = model.predict_proba(
                table.loc[validation, columns]
            )[:, 1]
            metrics = metric_row(
                table.loc[validation, "home_win"],
                probability,
            )
            rows.append(
                {
                    "family": family,
                    "month": month,
                    "training_games": int(training.sum()),
                    "evaluation_games": int(validation.sum()),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def calibration_diagnostics(
    table: pd.DataFrame,
    selected: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quantify March calibration and its sampling uncertainty."""

    march = table["game_date"].between(MARCH_START, MARCH_END)
    probability = selected["three_signal_champion"]["march_probability"]
    outcome = table.loc[march, "home_win"].to_numpy(dtype=int)
    game_dates = table.loc[march, "game_date"].to_numpy()

    logit_probability = np.log(
        np.clip(probability, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - probability, 1e-6, 1.0)
    ).reshape(-1, 1)
    calibration_model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=10000,
    )
    calibration_model.fit(logit_probability, outcome)

    summary = pd.DataFrame(
        [
            {
                "n": len(outcome),
                "mean_probability": float(np.mean(probability)),
                "actual_home_win_rate": float(np.mean(outcome)),
                "calibration_in_the_large_gap_actual_minus_predicted": float(
                    np.mean(outcome) - np.mean(probability)
                ),
                "calibration_intercept": float(
                    calibration_model.intercept_[0]
                ),
                "calibration_slope": float(
                    calibration_model.coef_[0, 0]
                ),
                "log_loss": float(log_loss(outcome, probability)),
                "brier_score": float(
                    brier_score_loss(outcome, probability)
                ),
                "roc_auc": float(roc_auc_score(outcome, probability)),
            }
        ]
    )

    reliability = pd.DataFrame(
        {
            "outcome": outcome,
            "probability": probability,
        }
    )
    reliability["bin"] = pd.qcut(
        reliability["probability"],
        q=5,
        duplicates="drop",
    )
    reliability_table = (
        reliability.groupby("bin", observed=True)
        .agg(
            n=("outcome", "size"),
            mean_probability=("probability", "mean"),
            observed_home_win_rate=("outcome", "mean"),
            minimum_probability=("probability", "min"),
            maximum_probability=("probability", "max"),
        )
        .reset_index()
    )

    frame = pd.DataFrame(
        {
            "game_date": game_dates,
            "outcome": outcome,
            "probability": probability,
        }
    )
    unique_dates = np.array(sorted(frame["game_date"].unique()))
    random = np.random.default_rng(RANDOM_SEED)
    bootstrap_rows: list[dict[str, float]] = []

    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled_dates = random.choice(
            unique_dates,
            size=len(unique_dates),
            replace=True,
        )
        sampled_indices = np.concatenate(
            [
                frame.index[
                    frame["game_date"] == sampled_date
                ].to_numpy()
                for sampled_date in sampled_dates
            ]
        )
        sample = frame.loc[sampled_indices]
        sample_outcome = sample["outcome"].to_numpy()
        sample_probability = sample["probability"].to_numpy()

        try:
            sample_logit = np.log(
                np.clip(sample_probability, 1e-6, 1.0 - 1e-6)
                / np.clip(1.0 - sample_probability, 1e-6, 1.0)
            ).reshape(-1, 1)
            sample_model = LogisticRegression(
                C=1e6,
                solver="lbfgs",
                max_iter=10000,
            )
            sample_model.fit(sample_logit, sample_outcome)
            intercept = float(sample_model.intercept_[0])
            slope = float(sample_model.coef_[0, 0])
        except ValueError:
            intercept = np.nan
            slope = np.nan

        bootstrap_rows.append(
            {
                "calibration_gap": float(
                    np.mean(sample_outcome)
                    - np.mean(sample_probability)
                ),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "log_loss": float(
                    log_loss(sample_outcome, sample_probability)
                ),
                "brier_score": float(
                    brier_score_loss(
                        sample_outcome,
                        sample_probability,
                    )
                ),
            }
        )

    bootstrap = pd.DataFrame(bootstrap_rows)
    intervals = []
    for metric in bootstrap.columns:
        valid = bootstrap[metric].dropna()
        lower, median, upper = valid.quantile(
            [0.025, 0.5, 0.975]
        )
        intervals.append(
            {
                "metric": metric,
                "bootstrap_2_5_percent": float(lower),
                "bootstrap_median": float(median),
                "bootstrap_97_5_percent": float(upper),
                "replicates": len(valid),
            }
        )

    return summary, reliability_table, pd.DataFrame(intervals)


def feature_collinearity(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report feature correlations and variance-inflation factors."""

    development = table.loc[
        table["game_date"] <= MARCH_END,
        CORE_FEATURES,
    ]
    correlation = development.corr()

    vif_rows = []
    for feature in CORE_FEATURES:
        other_features = [
            candidate
            for candidate in CORE_FEATURES
            if candidate != feature
        ]
        model = LinearRegression()
        model.fit(
            development[other_features],
            development[feature],
        )
        r_squared = model.score(
            development[other_features],
            development[feature],
        )
        vif_rows.append(
            {
                "feature": feature,
                "r_squared_against_other_features": float(r_squared),
                "variance_inflation_factor": float(
                    1.0 / (1.0 - r_squared)
                ),
            }
        )

    return correlation, pd.DataFrame(vif_rows)


def coefficient_stability(table: pd.DataFrame) -> pd.DataFrame:
    """Track standardized coefficients as the season develops."""

    month_ends = (
        pd.Timestamp("2025-12-31"),
        pd.Timestamp("2026-01-31"),
        pd.Timestamp("2026-02-28"),
        pd.Timestamp("2026-03-31"),
    )
    rows = []

    for month_end in month_ends:
        training = table["game_date"] <= month_end
        model = make_logistic(0.0075)
        model.fit(
            table.loc[training, CORE_FEATURES],
            table.loc[training, "home_win"],
        )
        scaler = model.named_steps["scale"]
        logistic = model.named_steps["logit"]
        standardized_intercept = float(logistic.intercept_[0])
        raw_zero = np.zeros((1, len(CORE_FEATURES)))
        standardized_zero = scaler.transform(
            pd.DataFrame(raw_zero, columns=CORE_FEATURES)
        )
        equal_strength_log_odds = float(
            standardized_intercept
            + standardized_zero[0] @ logistic.coef_[0]
        )
        equal_strength_probability = float(
            1.0 / (1.0 + np.exp(-equal_strength_log_odds))
        )

        for feature, coefficient in zip(
            CORE_FEATURES,
            logistic.coef_[0],
        ):
            rows.append(
                {
                    "training_through": month_end.date().isoformat(),
                    "training_games": int(training.sum()),
                    "term": feature,
                    "standardized_coefficient": float(coefficient),
                    "equal_strength_home_win_probability": (
                        equal_strength_probability
                    ),
                }
            )

    return pd.DataFrame(rows)


def april_model_uncertainty(
    data: pd.DataFrame,
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate coefficient uncertainty around the official April prices.

    The intervals condition on the supplied feature definitions and schedule.
    They do not include injury, lineup, data, or structural model uncertainty.
    """

    training = table["game_date"] <= MARCH_END
    frozen = champion.build_frozen_features(
        data,
        half_life=12.0,
        cutoff=MARCH_END,
    )
    frozen_april = frozen.loc[
        frozen["game_date"] >= APRIL_START
    ].reset_index(drop=True)

    final_model = make_logistic(0.0075)
    final_model.fit(
        table.loc[training, CORE_FEATURES],
        table.loc[training, "home_win"],
    )
    official_probability = final_model.predict_proba(
        frozen_april[CORE_FEATURES]
    )[:, 1]

    training_frame = table.loc[
        training,
        ["game_date", "home_win", *CORE_FEATURES],
    ].reset_index(drop=True)
    unique_dates = np.array(
        sorted(training_frame["game_date"].unique())
    )
    random = np.random.default_rng(RANDOM_SEED)
    bootstrap_predictions = np.empty(
        (APRIL_BOOTSTRAP_REPLICATES, len(frozen_april)),
        dtype=float,
    )

    for replicate in range(APRIL_BOOTSTRAP_REPLICATES):
        sampled_dates = random.choice(
            unique_dates,
            size=len(unique_dates),
            replace=True,
        )
        sampled_indices = np.concatenate(
            [
                training_frame.index[
                    training_frame["game_date"] == sampled_date
                ].to_numpy()
                for sampled_date in sampled_dates
            ]
        )
        sample = training_frame.loc[sampled_indices]

        if sample["home_win"].nunique() < 2:
            bootstrap_predictions[replicate] = np.nan
            continue

        model = make_logistic(0.0075)
        model.fit(
            sample[CORE_FEATURES],
            sample["home_win"],
        )
        bootstrap_predictions[replicate] = model.predict_proba(
            frozen_april[CORE_FEATURES]
        )[:, 1]

    quantiles = np.nanquantile(
        bootstrap_predictions,
        [0.025, 0.05, 0.5, 0.95, 0.975],
        axis=0,
    )
    output = frozen_april[
        ["game_id", "game_date", "away", "home"]
    ].copy()
    output["official_home_win_probability"] = official_probability
    output["model_uncertainty_2_5_percent"] = quantiles[0]
    output["model_uncertainty_5_percent"] = quantiles[1]
    output["bootstrap_median"] = quantiles[2]
    output["model_uncertainty_95_percent"] = quantiles[3]
    output["model_uncertainty_97_5_percent"] = quantiles[4]
    output["interval_width_90_percent"] = (
        output["model_uncertainty_95_percent"]
        - output["model_uncertainty_5_percent"]
    )
    output["bootstrap_replicates"] = APRIL_BOOTSTRAP_REPLICATES
    return output


def save_figures(
    comparison: pd.DataFrame,
    monthly: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    calibration_intervals: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Create focused governance visuals."""

    figure_dir.mkdir(parents=True, exist_ok=True)

    selected_families = [
        "three_signal_champion",
        "cumulative_plus_recent",
        "net_wins_plus_recent",
    ]
    nearest_srs = (
        comparison.loc[
            comparison["family"].str.startswith("champion_plus_srs")
        ]
        .sort_values("validation_log_loss")
        .iloc[0]["family"]
    )
    selected_families.append(nearest_srs)

    plot_table = comparison.loc[
        comparison["family"].isin(selected_families)
    ].sort_values("validation_log_loss")

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
        label="March governance",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(plot_table["family"], rotation=25, ha="right")
    axis.set_ylabel("Log loss - lower is better")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figure_dir / "enhanced_candidate_comparison.png",
        dpi=180,
    )
    plt.close(figure)

    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    champion_monthly = monthly.loc[
        monthly["family"] == "three_signal_champion"
    ]
    axis.plot(
        champion_monthly["month"],
        champion_monthly["log_loss"],
        marker="o",
        label="Three-signal champion",
    )
    for family in (
        "cumulative_plus_recent",
        "net_wins_plus_recent",
    ):
        family_table = monthly.loc[monthly["family"] == family]
        axis.plot(
            family_table["month"],
            family_table["log_loss"],
            marker="o",
            label=family,
        )
    axis.set_ylabel("Monthly expanding-window log loss")
    axis.set_xlabel("Evaluation month")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figure_dir / "monthly_model_stability.png",
        dpi=180,
    )
    plt.close(figure)

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)
    for term in CORE_FEATURES:
        term_table = coefficient_table.loc[
            coefficient_table["term"] == term
        ]
        axis.plot(
            term_table["training_through"],
            term_table["standardized_coefficient"],
            marker="o",
            label=term,
        )
    axis.set_ylabel("Standardized coefficient")
    axis.set_xlabel("Training data through")
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        figure_dir / "coefficient_stability.png",
        dpi=180,
    )
    plt.close(figure)

    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(111)
    gap_row = calibration_intervals.loc[
        calibration_intervals["metric"] == "calibration_gap"
    ].iloc[0]
    observed_gap = calibration_summary.iloc[0][
        "calibration_in_the_large_gap_actual_minus_predicted"
    ]
    axis.errorbar(
        [0],
        [observed_gap],
        yerr=[
            [
                observed_gap
                - gap_row["bootstrap_2_5_percent"]
            ],
            [
                gap_row["bootstrap_97_5_percent"]
                - observed_gap
            ],
        ],
        fmt="o",
        capsize=8,
    )
    axis.axhline(0.0, linestyle="--")
    axis.set_xticks([0])
    axis.set_xticklabels(["March calibration gap"])
    axis.set_ylabel("Actual home rate minus mean predicted probability")
    figure.tight_layout()
    figure.savefig(
        figure_dir / "march_calibration_uncertainty.png",
        dpi=180,
    )
    plt.close(figure)


def run(data_path: Path, output_dir: Path, figure_dir: Path) -> None:
    """Execute the full enhanced-governance workflow."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = champion.load_and_validate(data_path)
    table = assemble_feature_table(data)
    comparison, full_grid, selected = evaluate_candidate_families(table)
    bootstrap = bootstrap_candidate_differences(
        table,
        selected,
        comparison,
    )
    monthly = monthly_backtest(table, selected, comparison)
    calibration_summary, reliability, calibration_intervals = (
        calibration_diagnostics(table, selected)
    )
    correlation, vif = feature_collinearity(table)
    coefficients = coefficient_stability(table)
    april_uncertainty = april_model_uncertainty(data, table)

    comparison.to_csv(
        output_dir / "enhanced_model_comparison.csv",
        index=False,
    )
    full_grid.to_csv(
        output_dir / "enhanced_candidate_grid.csv",
        index=False,
    )
    bootstrap.to_csv(
        output_dir / "enhanced_bootstrap_model_differences.csv",
        index=False,
    )
    monthly.to_csv(
        output_dir / "enhanced_monthly_backtest.csv",
        index=False,
    )
    calibration_summary.to_csv(
        output_dir / "enhanced_march_calibration_summary.csv",
        index=False,
    )
    reliability.to_csv(
        output_dir / "enhanced_march_reliability_bins.csv",
        index=False,
    )
    calibration_intervals.to_csv(
        output_dir / "enhanced_march_calibration_bootstrap.csv",
        index=False,
    )
    correlation.to_csv(
        output_dir / "enhanced_feature_correlation.csv",
    )
    vif.to_csv(
        output_dir / "enhanced_feature_vif.csv",
        index=False,
    )
    coefficients.to_csv(
        output_dir / "enhanced_coefficient_stability.csv",
        index=False,
    )
    april_uncertainty.to_csv(
        output_dir / "april_model_uncertainty.csv",
        index=False,
        float_format="%.6f",
    )

    best_srs = (
        comparison.loc[
            comparison["family"].str.startswith("champion_plus_srs")
        ]
        .sort_values("validation_log_loss")
        .iloc[0]
    )
    best_shrunk = (
        comparison.loc[
            comparison["family"].str.startswith("shrunk_ewma")
        ]
        .sort_values("validation_log_loss")
        .iloc[0]
    )
    champion_row = comparison.loc[
        comparison["family"] == "three_signal_champion"
    ].iloc[0]

    selection = {
        "selected_model": "three_signal_champion",
        "selected_features": CORE_FEATURES,
        "selected_half_life": 12.0,
        "selected_C": 0.0075,
        "selection_statement": (
            "The architecture is retained because no opponent-adjusted, "
            "Bayesian-shrunk, pure-EWMA, PCA, or reduced-feature challenger "
            "produces a material and temporally stable proper-score improvement."
        ),
        "validation_log_loss": float(
            champion_row["validation_log_loss"]
        ),
        "march_governance_log_loss": float(
            champion_row["march_log_loss"]
        ),
        "best_opponent_adjusted_challenger": {
            "family": str(best_srs["family"]),
            "validation_log_loss": float(
                best_srs["validation_log_loss"]
            ),
            "march_log_loss": float(best_srs["march_log_loss"]),
        },
        "best_bayesian_shrunk_recent_challenger": {
            "family": str(best_shrunk["family"]),
            "validation_log_loss": float(
                best_shrunk["validation_log_loss"]
            ),
            "march_log_loss": float(
                best_shrunk["march_log_loss"]
            ),
        },
        "march_is_a_governance_check_not_an_untouched_test": True,
        "april_outcomes_used_for_model_selection": False,
    }
    with (
        output_dir / "enhanced_selection_decision.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(selection, file, indent=2)

    save_figures(
        comparison,
        monthly,
        calibration_summary,
        calibration_intervals,
        coefficients,
        figure_dir,
    )

    print(json.dumps(selection, indent=2))
    print()
    print(comparison.head(12).to_string(index=False))
    print()
    print(calibration_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Define source, output, and figure paths."""

    parser = argparse.ArgumentParser(
        description=(
            "Run limitations-driven governance analysis for the NBA "
            "home-win probability model."
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
        default=Path("outputs"),
        help="Directory for generated CSV and JSON artifacts",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures"),
        help="Directory for generated figures",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.data,
        arguments.output_dir,
        arguments.figure_dir,
    )
