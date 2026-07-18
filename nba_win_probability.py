"""Leakage-safe NBA home-win probability model for the bet365 technical task."""

from __future__ import annotations  # Keeps type annotations forward-compatible.

import argparse  # Provides a reproducible command-line interface.
import hashlib  # Fingerprints the exact source data used for the run.
import json  # Writes selected parameters in a machine-readable format.
import math  # Converts an intuitive half-life into an EWMA update rate.
from dataclasses import dataclass  # Stores each team's pregame state explicitly.
from pathlib import Path  # Handles file paths portably.

from project_runtime import require_supported_python

require_supported_python()


import matplotlib.pyplot as plt  # Creates compact interview-ready diagnostics.
import numpy as np  # Supports vectorized probability calculations.
import pandas as pd  # Loads and processes the game-level data.
from sklearn.linear_model import LogisticRegression  # Fits regularized home-win probabilities.
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score  # Evaluates probability and ranking quality.
from sklearn.pipeline import Pipeline  # Keeps scaling inside the fitted model.
from sklearn.preprocessing import StandardScaler  # Makes L2 regularization comparable across feature units.

EXPECTED_COLUMNS = {  # Defines the required raw-data contract.
    "game_id", "game_date", "away", "away_wins", "away_losses", "away_points",
    "away_turnovers", "away_fouls", "away_rebounds", "home", "home_wins",
    "home_losses", "home_points", "home_turnovers", "home_fouls", "home_rebounds",
}

FEATURE_COLUMNS = [  # Restricts the final model to three distinct pregame strength signals.
    "net_wins_diff",
    "cumulative_margin_diff",
    "recent_margin_evidence_diff",
]

TRAIN_END = pd.Timestamp("2025-12-31")  # Ends coefficient estimation before model selection.
VALIDATION_START = pd.Timestamp("2026-01-01")  # Starts the forward validation block.
VALIDATION_END = pd.Timestamp("2026-02-28")  # Leaves March for a later temporal check.
MARCH_START = pd.Timestamp("2026-03-01")  # Starts the final pre-April robustness period.
MARCH_END = pd.Timestamp("2026-03-31")  # Defines the information cutoff for official April prices.
APRIL_START = pd.Timestamp("2026-04-01")  # Starts the requested prediction period.
HALF_LIFE_GRID = (5.0, 8.0, 12.0, 16.0, 24.0)  # Tests plausible recent-form memory lengths.
C_GRID = (0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.050)  # Tests strong-to-moderate L2 regularization.
RANDOM_SEED = 365  # Makes any solver-level randomness reproducible.


@dataclass
class TeamState:  # Contains only information available before a team's next game.
    games: int = 0  # Counts completed games represented in the state.
    wins: int = 0  # Counts completed wins for frozen-cutoff forecasting.
    losses: int = 0  # Counts completed losses for frozen-cutoff forecasting.
    cumulative_margin: float = 0.0  # Accumulates point differential from completed games.
    ewma_margin: float = 0.0  # Stores exponentially weighted recent point differential.


def load_and_validate(path: Path) -> pd.DataFrame:  # Loads the CSV and fails fast on silent data errors.
    data = pd.read_csv(path, dtype={"game_id": "string"})  # Preserves the identifier exactly, including leading zeros.
    missing_columns = EXPECTED_COLUMNS.difference(data.columns)  # Finds required fields absent from the file.
    if missing_columns:  # Stops because the model is undefined without the required schema.
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")  # Reports the exact schema failure.
    data = data.loc[:, [column for column in data.columns if column in EXPECTED_COLUMNS]].copy()  # Keeps only assignment fields in source order.
    data["game_date"] = pd.to_datetime(data["game_date"], errors="raise")  # Converts and validates every date.
    data = data.sort_values(["game_date", "game_id"]).reset_index(drop=True)  # Enforces deterministic chronological processing.
    if data.isna().any().any():  # Rejects hidden imputation decisions.
        raise ValueError("The dataset contains missing values.")  # Makes the data issue explicit.
    if data["game_id"].duplicated().any():  # Verifies the assignment's unique-game identifier.
        raise ValueError("game_id is not unique.")  # Prevents accidental double-counting.
    if data["game_id"].str.fullmatch(r"\d+").ne(True).any():  # Ensures identifiers remain digit strings rather than parsed numbers.
        raise ValueError("At least one game_id is not a nonempty digit string.")  # Rejects malformed identifiers before joins or exports.
    if (data["home"] == data["away"]).any():  # Checks for an impossible self-match.
        raise ValueError("At least one game has identical home and away teams.")  # Stops on invalid scheduling data.
    if (data["home_points"] == data["away_points"]).any():  # NBA games cannot finish tied.
        raise ValueError("At least one game has a tied final score.")  # Protects the binary target definition.
    count_columns = [column for column in data.columns if column.endswith(("wins", "losses", "points", "turnovers", "fouls", "rebounds"))]  # Identifies nonnegative count fields.
    if (data[count_columns] < 0).any().any():  # Rejects impossible negative counts.
        raise ValueError("At least one count field is negative.")  # Reports the violated invariant.
    team_dates = pd.concat([  # Creates one row for every team-date appearance.
        data[["game_date", "home"]].rename(columns={"home": "team"}),  # Adds home-team appearances.
        data[["game_date", "away"]].rename(columns={"away": "team"}),  # Adds away-team appearances.
    ])
    if team_dates.duplicated().any():  # Same-day doubleheaders would require batched state updates.
        raise ValueError("A team appears more than once on the same date.")  # Stops rather than imposing an arbitrary within-day order.
    audit_pregame_records(data)  # Reconciles every supplied record against earlier results.
    return data  # Returns a clean chronological table.


def audit_pregame_records(data: pd.DataFrame) -> None:  # Verifies that wins and losses are truly pregame fields.
    teams = sorted(set(data["home"]).union(data["away"]))  # Builds the complete team universe.
    records = {team: [0, 0] for team in teams}  # Initializes every team with zero wins and losses.
    for row in data.itertuples(index=False):  # Replays the season in chronological order.
        if records[row.home] != [row.home_wins, row.home_losses]:  # Checks the home team's supplied record.
            raise ValueError(f"Pregame record mismatch for {row.home} in game {row.game_id}.")  # Identifies the exact bad row.
        if records[row.away] != [row.away_wins, row.away_losses]:  # Checks the away team's supplied record.
            raise ValueError(f"Pregame record mismatch for {row.away} in game {row.game_id}.")  # Identifies the exact bad row.
        if row.home_points > row.away_points:  # Routes the completed result to the correct counters.
            records[row.home][0] += 1  # Adds one home win.
            records[row.away][1] += 1  # Adds one away loss.
        else:  # Handles an away victory.
            records[row.away][0] += 1  # Adds one away win.
            records[row.home][1] += 1  # Adds one home loss.


def initialize_states(data: pd.DataFrame) -> dict[str, TeamState]:  # Creates one independent neutral state per team.
    teams = sorted(set(data["home"]).union(data["away"]))  # Uses both team columns to avoid omissions.
    return {team: TeamState() for team in teams}  # Returns mutable states keyed by team identifier.


def feature_values(home_state: TeamState, away_state: TeamState, home_record: tuple[int, int], away_record: tuple[int, int]) -> dict[str, float]:  # Constructs one pregame matchup vector.
    home_net_wins = home_record[0] - home_record[1]  # Measures the home team's accumulated result evidence.
    away_net_wins = away_record[0] - away_record[1]  # Measures the away team's accumulated result evidence.
    return {  # Expresses every strength signal as home minus away.
        "net_wins_diff": float(home_net_wins - away_net_wins),  # Retains outcome information while naturally giving larger samples more weight.
        "cumulative_margin_diff": home_state.cumulative_margin - away_state.cumulative_margin,  # Combines scoring dominance with the amount of evidence behind it.
        "recent_margin_evidence_diff": home_state.games * home_state.ewma_margin - away_state.games * away_state.ewma_margin,  # Shrinks early recent form through limited game evidence.
    }


def update_state(state: TeamState, margin: float, win: int, alpha: float) -> None:  # Applies one completed result to one team's state.
    state.ewma_margin = margin if state.games == 0 else alpha * margin + (1.0 - alpha) * state.ewma_margin  # Updates form only after the prediction point.
    state.games += 1  # Increments the amount of evidence.
    state.wins += win  # Updates completed wins for future frozen forecasts.
    state.losses += 1 - win  # Updates completed losses consistently.
    state.cumulative_margin += margin  # Adds the team-perspective point differential.


def build_sequential_features(data: pd.DataFrame, half_life: float) -> pd.DataFrame:  # Builds as-of-game-time features for backtesting and daily repricing.
    states = initialize_states(data)  # Starts every team from an identical neutral state.
    alpha = 1.0 - math.exp(math.log(0.5) / half_life)  # Converts half-life into the EWMA's per-game update weight.
    rows: list[dict[str, object]] = []  # Collects one feature record per game.
    for row in data.itertuples(index=False):  # Walks forward exactly as information arrived.
        home_state = states[row.home]  # Retrieves the home state before the current result.
        away_state = states[row.away]  # Retrieves the away state before the current result.
        features = feature_values(home_state, away_state, (row.home_wins, row.home_losses), (row.away_wins, row.away_losses))  # Uses only pregame information.
        home_win = int(row.home_points > row.away_points)  # Creates the target after the feature snapshot.
        rows.append({  # Stores the prediction row before any current-game update.
            "game_id": row.game_id,
            "game_date": row.game_date,
            "home": row.home,
            "away": row.away,
            "home_win": home_win,
            **features,
        })
        margin = float(row.home_points - row.away_points)  # Converts the final score into home point differential.
        update_state(home_state, margin, home_win, alpha)  # Makes the completed game available only to later home-team predictions.
        update_state(away_state, -margin, 1 - home_win, alpha)  # Applies the symmetric away-team update.
    return pd.DataFrame(rows)  # Returns the chronological model table.


def build_frozen_features(data: pd.DataFrame, half_life: float, cutoff: pd.Timestamp) -> pd.DataFrame:  # Prices all future games from one fixed information snapshot.
    states = initialize_states(data)  # Starts with the same neutral states used in backtesting.
    alpha = 1.0 - math.exp(math.log(0.5) / half_life)  # Uses the identical recent-form definition.
    for row in data.loc[data["game_date"] <= cutoff].itertuples(index=False):  # Replays only observable games through the cutoff.
        margin = float(row.home_points - row.away_points)  # Derives the completed home margin.
        home_win = int(margin > 0.0)  # Converts the completed result to a binary outcome.
        update_state(states[row.home], margin, home_win, alpha)  # Updates the home state through the cutoff.
        update_state(states[row.away], -margin, 1 - home_win, alpha)  # Updates the away state through the cutoff.
    rows: list[dict[str, object]] = []  # Collects fixed-snapshot future matchups.
    for row in data.loc[data["game_date"] > cutoff].itertuples(index=False):  # Prices every post-cutoff game without updating on future results.
        home_state = states[row.home]  # Reads the frozen home state.
        away_state = states[row.away]  # Reads the frozen away state.
        features = feature_values(home_state, away_state, (home_state.wins, home_state.losses), (away_state.wins, away_state.losses))  # Uses cutoff records rather than later CSV records.
        rows.append({  # Stores both forecast inputs and the realized result for descriptive evaluation.
            "game_id": row.game_id,
            "game_date": row.game_date,
            "home": row.home,
            "away": row.away,
            "home_win": int(row.home_points > row.away_points),
            **features,
        })
    return pd.DataFrame(rows)  # Returns strict cutoff-based future features.


def make_model(c_value: float) -> Pipeline:  # Constructs the exact final probability model.
    return Pipeline([  # Fits transformations and coefficients as one leakage-safe object.
        ("scale", StandardScaler()),  # Estimates means and scales on training data only.
        ("logit", LogisticRegression(C=c_value, solver="lbfgs", max_iter=2000, random_state=RANDOM_SEED)),  # Fits L2-regularized log odds.
    ])


def probability_metrics(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> dict[str, float]:  # Calculates sportsbook-relevant diagnostics.
    return {  # Returns proper scores first and classification summaries second.
        "log_loss": float(log_loss(y_true, probability)),  # Primary score because confident pricing errors are penalized strongly.
        "brier_score": float(brier_score_loss(y_true, probability)),  # Measures mean squared probability error.
        "roc_auc": float(roc_auc_score(y_true, probability)),  # Measures ranking ability without claiming calibration.
        "accuracy_0_5": float(accuracy_score(y_true, probability >= 0.5)),  # Provides an intuitive secondary summary.
        "mean_probability": float(np.mean(probability)),  # Helps diagnose systematic over- or under-pricing.
        "actual_home_win_rate": float(np.mean(y_true)),  # Provides the observed comparison for mean calibration.
    }


def tune_reduced_baseline(features: pd.DataFrame, columns: list[str]) -> float:  # Tunes a reduced challenger fairly on the same validation period.
    train_mask = features["game_date"] <= TRAIN_END  # Matches the champion's training window.
    validation_mask = features["game_date"].between(VALIDATION_START, VALIDATION_END)  # Matches the champion's validation window.
    results: list[tuple[float, float]] = []  # Stores validation loss and C.
    for c_value in C_GRID:  # Uses the same regularization search as the champion.
        model = make_model(c_value)  # Builds a fresh reduced model.
        model.fit(features.loc[train_mask, columns], features.loc[train_mask, "home_win"])  # Fits only October-December.
        probability = model.predict_proba(features.loc[validation_mask, columns])[:, 1]  # Prices January-February.
        results.append((float(log_loss(features.loc[validation_mask, "home_win"], probability)), c_value))  # Records the proper score.
    return min(results)[1]  # Locks the best C before March.


def calibration_bins(y_true: pd.Series | np.ndarray, probability: np.ndarray, bins: int = 5) -> pd.DataFrame:  # Summarizes reliability with stable sample sizes.
    table = pd.DataFrame({"actual": np.asarray(y_true), "probability": probability})  # Aligns outcomes and forecasts.
    table["bin"] = pd.qcut(table["probability"], q=bins, duplicates="drop")  # Uses approximately equal observations per bin.
    return table.groupby("bin", observed=True).agg(  # Reports observed and forecast rates by price band.
        n=("actual", "size"),
        mean_probability=("probability", "mean"),
        actual_home_win_rate=("actual", "mean"),
        minimum_probability=("probability", "min"),
        maximum_probability=("probability", "max"),
    ).reset_index()


def coefficient_table(model: Pipeline) -> pd.DataFrame:  # Converts final parameters into presentation-friendly effects.
    scaler = model.named_steps["scale"]  # Retrieves training means and standard deviations.
    logit_model = model.named_steps["logit"]  # Retrieves regularized coefficients.
    coefficients = logit_model.coef_[0]  # Extracts the binary-class slope vector.
    standardized_intercept = float(logit_model.intercept_[0])  # Captures the intercept when standardized predictors equal zero.
    equal_strength_raw = pd.DataFrame([[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)  # Defines an evenly matched home-away matchup.
    equal_strength_standardized = scaler.transform(equal_strength_raw)  # Places that raw matchup on the model's standardized scale.
    equal_strength_log_odds = float(standardized_intercept + equal_strength_standardized[0] @ coefficients)  # Computes structural home advantage at raw feature differences of zero.
    equal_strength_probability = float(1.0 / (1.0 + np.exp(-equal_strength_log_odds)))  # Converts equal-strength home log odds into probability.
    rows = [{  # Reports the fitted standardized intercept explicitly.
        "term": "standardized_intercept",
        "term_type": "intercept",
        "training_mean": np.nan,
        "training_standard_deviation": np.nan,
        "coefficient_standardized": standardized_intercept,
        "coefficient_original_units": np.nan,
        "odds_multiplier": float(np.exp(standardized_intercept)),
        "reference_home_win_probability": float(1.0 / (1.0 + np.exp(-standardized_intercept))),
        "interpretation": "Home log odds when all standardized predictors equal zero.",
    }]
    rows.append({  # Reports the sportsbook-relevant equal-team home-court baseline.
        "term": "equal_strength_home_advantage",
        "term_type": "derived_home_advantage",
        "training_mean": 0.0,
        "training_standard_deviation": np.nan,
        "coefficient_standardized": equal_strength_log_odds,
        "coefficient_original_units": equal_strength_log_odds,
        "odds_multiplier": float(np.exp(equal_strength_log_odds)),
        "reference_home_win_probability": equal_strength_probability,
        "interpretation": "Home log odds and probability when every raw home-away feature difference equals zero.",
    })
    for feature, mean, scale, coefficient in zip(FEATURE_COLUMNS, scaler.mean_, scaler.scale_, coefficients):  # Adds one interpretable row per team-strength signal.
        rows.append({
            "term": feature,
            "term_type": "feature",
            "training_mean": float(mean),
            "training_standard_deviation": float(scale),
            "coefficient_standardized": float(coefficient),
            "coefficient_original_units": float(coefficient / scale),
            "odds_multiplier": float(np.exp(coefficient)),
            "reference_home_win_probability": np.nan,
            "interpretation": "Odds multiplier is for a one-standard-deviation increase, holding the other standardized features fixed.",
        })
    return pd.DataFrame(rows)  # Returns one auditable coefficient and home-advantage artifact.



def build_component_feature_tables(
    data: pd.DataFrame,
) -> dict[float, pd.DataFrame]:
    """Build one leakage-safe sequential table for every EWMA half-life."""

    tables = {
        half_life: build_sequential_features(data, half_life)
        for half_life in HALF_LIFE_GRID
    }
    reference_ids = tables[HALF_LIFE_GRID[0]]["game_id"].tolist()
    for half_life, table in tables.items():
        if table["game_id"].tolist() != reference_ids:
            raise ValueError(
                f"Feature rows are misaligned for half-life {half_life}."
            )
    return tables


def component_predictions(
    feature_tables: dict[float, pd.DataFrame],
    train_end: pd.Timestamp,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    frozen_tables: dict[float, pd.DataFrame] | None = None,
    return_models: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, list[Pipeline]]:
    """Fit every predeclared component and return its probability matrix."""

    prediction_columns: list[np.ndarray] = []
    component_rows: list[dict[str, float]] = []
    fitted_models: list[Pipeline] = []

    for half_life in HALF_LIFE_GRID:
        training_table = feature_tables[half_life]
        training_mask = training_table["game_date"] <= train_end

        if frozen_tables is None:
            scoring_table = training_table
            scoring_mask = scoring_table["game_date"].between(
                evaluation_start,
                evaluation_end,
            )
            scoring_features = scoring_table.loc[
                scoring_mask,
                FEATURE_COLUMNS,
            ]
            outcomes = scoring_table.loc[scoring_mask, "home_win"]
        else:
            scoring_table = frozen_tables[half_life]
            scoring_mask = scoring_table["game_date"].between(
                evaluation_start,
                evaluation_end,
            )
            scoring_features = scoring_table.loc[
                scoring_mask,
                FEATURE_COLUMNS,
            ]
            outcomes = scoring_table.loc[scoring_mask, "home_win"]

        for c_value in C_GRID:
            model = make_model(c_value)
            model.fit(
                training_table.loc[training_mask, FEATURE_COLUMNS],
                training_table.loc[training_mask, "home_win"],
            )
            probability = model.predict_proba(scoring_features)[:, 1]
            prediction_columns.append(probability)
            metrics = probability_metrics(outcomes, probability)
            component_rows.append(
                {
                    "half_life": float(half_life),
                    "C": float(c_value),
                    "n": int(len(probability)),
                    **metrics,
                }
            )
            if return_models:
                fitted_models.append(model)

    matrix = np.column_stack(prediction_columns)
    return matrix, pd.DataFrame(component_rows), fitted_models


def ensemble_probability(component_matrix: np.ndarray) -> np.ndarray:
    """Average component probabilities using fixed, untuned equal weights."""

    if component_matrix.ndim != 2:
        raise ValueError("Component predictions must be a two-dimensional matrix.")
    if component_matrix.shape[1] != len(HALF_LIFE_GRID) * len(C_GRID):
        raise ValueError("Unexpected number of ensemble components.")
    probability = component_matrix.mean(axis=1)
    if not np.all((probability > 0.0) & (probability < 1.0)):
        raise ValueError("The ensemble produced an invalid probability.")
    return probability


def validation_audit(
    feature_tables: dict[float, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, float]]:
    """Evaluate every component and the fixed ensemble on January-February."""

    matrix, grid, _ = component_predictions(
        feature_tables,
        train_end=TRAIN_END,
        evaluation_start=VALIDATION_START,
        evaluation_end=VALIDATION_END,
    )
    ensemble = ensemble_probability(matrix)
    reference = feature_tables[HALF_LIFE_GRID[0]]
    validation_mask = reference["game_date"].between(
        VALIDATION_START,
        VALIDATION_END,
    )
    outcomes = reference.loc[validation_mask, "home_win"]
    train_mask = reference["game_date"] <= TRAIN_END
    constant = np.full(
        len(outcomes),
        reference.loc[train_mask, "home_win"].mean(),
    )

    grid = grid.sort_values(
        ["log_loss", "brier_score", "C", "half_life"]
    ).reset_index(drop=True)
    best = {
        "half_life": float(grid.iloc[0]["half_life"]),
        "C": float(grid.iloc[0]["C"]),
    }
    best_index = (
        list(HALF_LIFE_GRID).index(best["half_life"]) * len(C_GRID)
        + list(C_GRID).index(best["C"])
    )
    best_probability = matrix[:, best_index]

    comparison = pd.DataFrame(
        [
            {
                "model": "constant_home_rate",
                "split": "January-February validation",
                "n": len(outcomes),
                **probability_metrics(outcomes, constant),
            },
            {
                "model": "validation_best_single_component",
                "split": "January-February validation",
                "n": len(outcomes),
                **probability_metrics(outcomes, best_probability),
            },
            {
                "model": "uniform_40_component_logistic_ensemble",
                "split": "January-February validation",
                "n": len(outcomes),
                **probability_metrics(outcomes, ensemble),
            },
        ]
    )
    return grid, comparison, ensemble, best


def march_governance_check(
    feature_tables: dict[float, pd.DataFrame],
    best_single: dict[str, float],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Run the March governance check after the ensemble rule is fixed."""

    matrix, component_metrics, _ = component_predictions(
        feature_tables,
        train_end=VALIDATION_END,
        evaluation_start=MARCH_START,
        evaluation_end=MARCH_END,
    )
    ensemble = ensemble_probability(matrix)
    reference = feature_tables[12.0]
    train_mask = reference["game_date"] <= VALIDATION_END
    march_mask = reference["game_date"].between(MARCH_START, MARCH_END)
    outcomes = reference.loc[march_mask, "home_win"]

    constant = np.full(
        len(outcomes),
        reference.loc[train_mask, "home_win"].mean(),
    )
    best_table = feature_tables[best_single["half_life"]]
    best_model = make_model(best_single["C"])
    best_model.fit(
        best_table.loc[
            best_table["game_date"] <= VALIDATION_END,
            FEATURE_COLUMNS,
        ],
        best_table.loc[
            best_table["game_date"] <= VALIDATION_END,
            "home_win",
        ],
    )
    best_probability = best_model.predict_proba(
        best_table.loc[
            best_table["game_date"].between(MARCH_START, MARCH_END),
            FEATURE_COLUMNS,
        ]
    )[:, 1]

    net_c = tune_reduced_baseline(reference, ["net_wins_diff"])
    net_model = make_model(net_c)
    net_model.fit(
        reference.loc[train_mask, ["net_wins_diff"]],
        reference.loc[train_mask, "home_win"],
    )
    net_probability = net_model.predict_proba(
        reference.loc[march_mask, ["net_wins_diff"]]
    )[:, 1]

    margin_c = tune_reduced_baseline(reference, ["cumulative_margin_diff"])
    margin_model = make_model(margin_c)
    margin_model.fit(
        reference.loc[train_mask, ["cumulative_margin_diff"]],
        reference.loc[train_mask, "home_win"],
    )
    margin_probability = margin_model.predict_proba(
        reference.loc[march_mask, ["cumulative_margin_diff"]]
    )[:, 1]

    comparison = pd.DataFrame(
        [
            {
                "model": "constant_home_rate",
                "split": "March governance check",
                "n": len(outcomes),
                **probability_metrics(outcomes, constant),
            },
            {
                "model": "net_wins_only",
                "split": "March governance check",
                "n": len(outcomes),
                **probability_metrics(outcomes, net_probability),
            },
            {
                "model": "cumulative_margin_only",
                "split": "March governance check",
                "n": len(outcomes),
                **probability_metrics(outcomes, margin_probability),
            },
            {
                "model": "validation_best_single_component",
                "split": "March governance check",
                "n": len(outcomes),
                **probability_metrics(outcomes, best_probability),
            },
            {
                "model": "uniform_40_component_logistic_ensemble",
                "split": "March governance check",
                "n": len(outcomes),
                **probability_metrics(outcomes, ensemble),
            },
        ]
    )
    return comparison, ensemble, component_metrics


def final_component_summary(
    models: list[Pipeline],
    validation_grid: pd.DataFrame,
    march_components: pd.DataFrame,
) -> pd.DataFrame:
    """Record every final component's fit and equal-strength home baseline."""

    rows: list[dict[str, float]] = []
    component_index = 0
    for half_life in HALF_LIFE_GRID:
        for c_value in C_GRID:
            model = models[component_index]
            coefficients = coefficient_table(model)
            home_row = coefficients.loc[
                coefficients["term"] == "equal_strength_home_advantage"
            ].iloc[0]
            feature_rows = coefficients.loc[
                coefficients["term_type"] == "feature"
            ].set_index("term")
            validation_row = validation_grid.loc[
                (validation_grid["half_life"] == half_life)
                & np.isclose(validation_grid["C"], c_value)
            ].iloc[0]
            march_row = march_components.loc[
                (march_components["half_life"] == half_life)
                & np.isclose(march_components["C"], c_value)
            ].iloc[0]
            rows.append(
                {
                    "component": component_index + 1,
                    "weight": 1.0 / (len(HALF_LIFE_GRID) * len(C_GRID)),
                    "half_life": float(half_life),
                    "C": float(c_value),
                    "validation_log_loss": float(validation_row["log_loss"]),
                    "march_log_loss": float(march_row["log_loss"]),
                    "equal_strength_home_win_probability": float(
                        home_row["reference_home_win_probability"]
                    ),
                    "net_wins_coefficient_per_sd": float(
                        feature_rows.loc[
                            "net_wins_diff",
                            "coefficient_standardized",
                        ]
                    ),
                    "cumulative_margin_coefficient_per_sd": float(
                        feature_rows.loc[
                            "cumulative_margin_diff",
                            "coefficient_standardized",
                        ]
                    ),
                    "recent_margin_coefficient_per_sd": float(
                        feature_rows.loc[
                            "recent_margin_evidence_diff",
                            "coefficient_standardized",
                        ]
                    ),
                }
            )
            component_index += 1
    return pd.DataFrame(rows)


def save_figures(
    validation_comparison: pd.DataFrame,
    march_metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    component_summary: pd.DataFrame,
    dispersion: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Create concise interview-ready model diagnostics."""

    figure_dir.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    comparison = validation_comparison.loc[
        validation_comparison["model"] != "constant_home_rate"
    ]
    axis.bar(comparison["model"], comparison["log_loss"])
    axis.set_ylabel("January-February log loss")
    axis.tick_params(axis="x", rotation=18)
    figure.tight_layout()
    figure.savefig(figure_dir / "validation_model_comparison.png", dpi=180)
    plt.close(figure)

    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    axis.bar(march_metrics["model"], march_metrics["log_loss"])
    axis.set_ylabel("March log loss")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(figure_dir / "march_model_comparison.png", dpi=180)
    plt.close(figure)

    figure = plt.figure(figsize=(6, 6))
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    axis.plot(
        calibration["mean_probability"],
        calibration["actual_home_win_rate"],
        marker="o",
        label="ensemble",
    )
    axis.set_xlabel("Mean predicted home-win probability")
    axis.set_ylabel("Observed home-win rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "march_calibration.png", dpi=180)
    plt.close(figure)

    coefficient_columns = [
        "net_wins_coefficient_per_sd",
        "cumulative_margin_coefficient_per_sd",
        "recent_margin_coefficient_per_sd",
    ]
    coefficient_means = component_summary[coefficient_columns].mean()
    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    axis.bar(coefficient_means.index, coefficient_means.values)
    axis.set_ylabel("Mean standardized coefficient across components")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(figure_dir / "ensemble_mean_coefficients.png", dpi=180)
    plt.close(figure)

    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    axis.hist(dispersion["component_standard_deviation"], bins=12)
    axis.set_xlabel("Across-component probability standard deviation")
    axis.set_ylabel("April games")
    figure.tight_layout()
    figure.savefig(figure_dir / "april_component_dispersion.png", dpi=180)
    plt.close(figure)


def run(data_path: Path, output_dir: Path) -> None:
    """Execute the complete robust-ensemble workflow."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir.parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_validate(data_path)
    source_bytes = data_path.read_bytes()
    data_fingerprint = {
        "source_file_name": data_path.name,
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "bytes": len(source_bytes),
        "rows": len(data),
        "minimum_game_date": str(data["game_date"].min().date()),
        "maximum_game_date": str(data["game_date"].max().date()),
    }
    with (output_dir / "data_fingerprint.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(data_fingerprint, file, indent=2)
    feature_tables = build_component_feature_tables(data)

    validation_grid, validation_comparison, _, best_single = validation_audit(
        feature_tables,
    )
    march_metrics, march_probability, march_components = march_governance_check(
        feature_tables,
        best_single,
    )
    reference = feature_tables[12.0]
    march_mask = reference["game_date"].between(MARCH_START, MARCH_END)
    march_calibration = calibration_bins(
        reference.loc[march_mask, "home_win"],
        march_probability,
    )

    frozen_tables = {
        half_life: build_frozen_features(data, half_life, MARCH_END)
        for half_life in HALF_LIFE_GRID
    }
    frozen_matrix, _, final_models = component_predictions(
        feature_tables,
        train_end=MARCH_END,
        evaluation_start=APRIL_START,
        evaluation_end=data["game_date"].max(),
        frozen_tables=frozen_tables,
        return_models=True,
    )
    rolling_matrix, _, _ = component_predictions(
        feature_tables,
        train_end=MARCH_END,
        evaluation_start=APRIL_START,
        evaluation_end=data["game_date"].max(),
    )
    frozen_probability = ensemble_probability(frozen_matrix)
    rolling_probability = ensemble_probability(rolling_matrix)

    frozen_reference = frozen_tables[12.0].loc[
        frozen_tables[12.0]["game_date"] >= APRIL_START
    ].reset_index(drop=True)
    sequential_reference = reference.loc[
        reference["game_date"] >= APRIL_START
    ].reset_index(drop=True)
    if frozen_reference["game_id"].tolist() != sequential_reference["game_id"].tolist():
        raise ValueError("Frozen and rolling April rows are not aligned.")

    predictions = frozen_reference[
        ["game_id", "game_date", "away", "home"]
    ].copy()
    submitted_probability = np.round(frozen_probability, 10)
    predictions["home_win_probability"] = submitted_probability
    predictions["fair_home_decimal_odds"] = 1.0 / submitted_probability
    predictions["fair_away_decimal_odds"] = 1.0 / (
        1.0 - submitted_probability
    )

    best_single_index = (
        list(HALF_LIFE_GRID).index(best_single["half_life"]) * len(C_GRID)
        + list(C_GRID).index(best_single["C"])
    )
    single_probability = np.round(
        frozen_matrix[:, best_single_index],
        10,
    )
    single_predictions = predictions[
        ["game_id", "game_date", "away", "home"]
    ].copy()
    single_predictions["home_win_probability"] = single_probability
    single_predictions["fair_home_decimal_odds"] = (
        1.0 / single_probability
    )
    single_predictions["fair_away_decimal_odds"] = (
        1.0 / (1.0 - single_probability)
    )

    repricing = predictions[
        ["game_id", "game_date", "away", "home"]
    ].copy()
    repricing["frozen_march_31_probability"] = submitted_probability
    repricing["rolling_as_of_game_probability"] = rolling_probability
    repricing["absolute_probability_change"] = np.abs(
        rolling_probability - frozen_probability
    )

    dispersion = predictions[
        ["game_id", "game_date", "away", "home", "home_win_probability"]
    ].copy()
    dispersion["component_standard_deviation"] = frozen_matrix.std(axis=1)
    dispersion["component_minimum"] = frozen_matrix.min(axis=1)
    dispersion["component_5_percent"] = np.quantile(
        frozen_matrix,
        0.05,
        axis=1,
    )
    dispersion["component_median"] = np.quantile(
        frozen_matrix,
        0.50,
        axis=1,
    )
    dispersion["component_95_percent"] = np.quantile(
        frozen_matrix,
        0.95,
        axis=1,
    )
    dispersion["component_maximum"] = frozen_matrix.max(axis=1)

    component_summary = final_component_summary(
        final_models,
        validation_grid,
        march_components,
    )
    ensemble_home_probability = float(
        component_summary["equal_strength_home_win_probability"].mean()
    )
    april_metrics = pd.DataFrame(
        [
            {
                "model": "uniform_40_component_logistic_ensemble",
                "split": "April post-lock descriptive audit",
                "n": len(frozen_reference),
                **probability_metrics(
                    frozen_reference["home_win"],
                    submitted_probability,
                ),
            }
        ]
    )

    single_april_metrics = pd.DataFrame(
        [
            {
                "model": "validation_best_single_component",
                "split": "April post-lock descriptive benchmark",
                "n": len(frozen_reference),
                **probability_metrics(
                    frozen_reference["home_win"],
                    single_probability,
                ),
            }
        ]
    )

    validation_grid.to_csv(
        output_dir / "validation_grid.csv",
        index=False,
    )
    validation_comparison.to_csv(
        output_dir / "ensemble_validation_metrics.csv",
        index=False,
    )
    march_metrics.to_csv(
        output_dir / "march_temporal_check_metrics.csv",
        index=False,
    )
    march_calibration.to_csv(
        output_dir / "march_calibration_bins.csv",
        index=False,
    )
    component_summary.to_csv(
        output_dir / "ensemble_component_summary.csv",
        index=False,
    )
    predictions.to_csv(
        output_dir / "april_predictions.csv",
        index=False,
        float_format="%.10f",
    )
    single_predictions.to_csv(
        output_dir / "single_model_benchmark_april_predictions.csv",
        index=False,
        float_format="%.10f",
    )
    repricing.to_csv(
        output_dir / "april_repricing_backtest.csv",
        index=False,
        float_format="%.10f",
    )
    dispersion.to_csv(
        output_dir / "april_component_dispersion.csv",
        index=False,
        float_format="%.10f",
    )
    april_metrics.to_csv(
        output_dir / "april_descriptive_metrics.csv",
        index=False,
    )
    single_april_metrics.to_csv(
        output_dir / "single_model_benchmark_april_metrics.csv",
        index=False,
    )

    selected_model = {
        "model": "uniform_40_component_logistic_ensemble",
        "features": FEATURE_COLUMNS,
        "half_lives": list(HALF_LIFE_GRID),
        "C_values": list(C_GRID),
        "component_count": len(HALF_LIFE_GRID) * len(C_GRID),
        "component_weight": 1.0 / (
            len(HALF_LIFE_GRID) * len(C_GRID)
        ),
        "aggregation": "arithmetic mean of component probabilities",
        "weights_tuned": False,
        "validation_best_single_component": best_single,
        "april_outcomes_used_for_component_or_weight_tuning": False,
        "april_outcomes_viewed_descriptively": True,
        "march_role": (
            "Later governance check used to confirm that the ensemble "
            "validation advantage did not reverse before April."
        ),
        "selection_basis": (
            "The fixed equal-weight ensemble improved January-February "
            "validation log loss before April and was retained after the "
            "March governance check."
        ),
    }
    with (output_dir / "selected_model.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(selected_model, file, indent=2)

    summary = {
        "data_fingerprint": data_fingerprint,
        "selected_model": selected_model,
        "equal_strength_home_win_probability": ensemble_home_probability,
        "validation": validation_comparison.to_dict(orient="records"),
        "march_governance_check": march_metrics.to_dict(orient="records"),
        "april_prediction_count": len(predictions),
    }
    with (output_dir / "model_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    save_figures(
        validation_comparison,
        march_metrics,
        march_calibration,
        component_summary,
        dispersion,
        figure_dir,
    )

    print(json.dumps(summary, indent=2, default=str))


def parse_args() -> argparse.Namespace:  # Defines an interview-friendly command-line interface.
    parser = argparse.ArgumentParser(description="Build NBA home-win probabilities without target leakage.")  # Creates helpful usage text.
    parser.add_argument("--data", type=Path, required=True, help="Path to nba-win-probability-data.csv")  # Requires an explicit input path.
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for predictions and diagnostics")  # Allows clean output redirection.
    return parser.parse_args()  # Returns validated command-line arguments.


if __name__ == "__main__":  # Runs only when the file is executed directly.
    arguments = parse_args()  # Reads command-line paths.
    run(arguments.data, arguments.output_dir)  # Executes the complete model pipeline.
