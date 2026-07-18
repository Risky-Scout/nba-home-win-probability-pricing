"""Leakage-safe NBA home-win probability model for the bet365 technical task."""

from __future__ import annotations  # Keeps type annotations forward-compatible.

import argparse  # Provides a reproducible command-line interface.
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


def tune_model(data: pd.DataFrame) -> tuple[float, float, pd.DataFrame, pd.DataFrame]:  # Selects half-life and L2 strength using only January-February.
    candidates: list[dict[str, float]] = []  # Preserves every validation result for auditability.
    feature_tables: dict[float, pd.DataFrame] = {}  # Avoids rebuilding identical features for each C value.
    for half_life in HALF_LIFE_GRID:  # Tests the predeclared recent-form memory values.
        features = build_sequential_features(data, half_life)  # Builds leakage-safe features for this half-life.
        feature_tables[half_life] = features  # Caches the result for the selected model.
        train_mask = features["game_date"] <= TRAIN_END  # Uses October-December for coefficient estimation.
        validation_mask = features["game_date"].between(VALIDATION_START, VALIDATION_END)  # Uses January-February only for selection.
        for c_value in C_GRID:  # Tests progressively weaker L2 shrinkage.
            model = make_model(c_value)  # Creates a fresh pipeline for this candidate.
            model.fit(features.loc[train_mask, FEATURE_COLUMNS], features.loc[train_mask, "home_win"])  # Fits without validation leakage.
            probability = model.predict_proba(features.loc[validation_mask, FEATURE_COLUMNS])[:, 1]  # Produces forward validation prices.
            candidates.append({  # Records parameters and all validation diagnostics.
                "half_life": half_life,
                "C": c_value,
                "n": int(validation_mask.sum()),
                **probability_metrics(features.loc[validation_mask, "home_win"], probability),
            })
    grid = pd.DataFrame(candidates).sort_values(["log_loss", "brier_score", "C"]).reset_index(drop=True)  # Applies the predeclared metric hierarchy.
    best = grid.iloc[0]  # Locks the lowest-log-loss candidate before examining March.
    selected_half_life = float(best["half_life"])  # Extracts the chosen recent-form memory.
    selected_c = float(best["C"])  # Extracts the chosen L2 strength.
    return selected_half_life, selected_c, grid, feature_tables[selected_half_life]  # Returns the locked model specification and audit table.


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


def evaluate_march(features: pd.DataFrame, selected_c: float) -> tuple[pd.DataFrame, np.ndarray]:  # Runs the final pre-April temporal robustness check.
    train_validation_mask = features["game_date"] <= VALIDATION_END  # Uses all information available before March.
    march_mask = features["game_date"].between(MARCH_START, MARCH_END)  # Identifies March games.
    y_march = features.loc[march_mask, "home_win"]  # Extracts realized March outcomes.
    constant_probability = np.full(march_mask.sum(), features.loc[train_validation_mask, "home_win"].mean())  # Defines the historical home-rate baseline.
    net_wins_c = tune_reduced_baseline(features, ["net_wins_diff"])  # Tunes the result-only challenger without March.
    net_wins_model = make_model(net_wins_c)  # Recreates the locked result-only model.
    net_wins_model.fit(features.loc[train_validation_mask, ["net_wins_diff"]], features.loc[train_validation_mask, "home_win"])  # Refits through February.
    net_wins_probability = net_wins_model.predict_proba(features.loc[march_mask, ["net_wins_diff"]])[:, 1]  # Prices March from results alone.
    margin_c = tune_reduced_baseline(features, ["cumulative_margin_diff"])  # Tunes the margin-only challenger without March.
    margin_model = make_model(margin_c)  # Recreates the locked margin-only model.
    margin_model.fit(features.loc[train_validation_mask, ["cumulative_margin_diff"]], features.loc[train_validation_mask, "home_win"])  # Refits through February.
    margin_probability = margin_model.predict_proba(features.loc[march_mask, ["cumulative_margin_diff"]])[:, 1]  # Prices March from cumulative margin alone.
    selected_model = make_model(selected_c)  # Recreates the locked three-feature champion.
    selected_model.fit(features.loc[train_validation_mask, FEATURE_COLUMNS], features.loc[train_validation_mask, "home_win"])  # Refits through February.
    selected_probability = selected_model.predict_proba(features.loc[march_mask, FEATURE_COLUMNS])[:, 1]  # Prices March prospectively.
    metrics = pd.DataFrame([  # Places every March comparison in one auditable table.
        {"model": "constant_home_rate", "split": "March temporal check", "n": int(march_mask.sum()), "selected_C": np.nan, **probability_metrics(y_march, constant_probability)},
        {"model": "net_wins_only", "split": "March temporal check", "n": int(march_mask.sum()), "selected_C": net_wins_c, **probability_metrics(y_march, net_wins_probability)},
        {"model": "cumulative_margin_only", "split": "March temporal check", "n": int(march_mask.sum()), "selected_C": margin_c, **probability_metrics(y_march, margin_probability)},
        {"model": "selected_three_feature_logistic", "split": "March temporal check", "n": int(march_mask.sum()), "selected_C": selected_c, **probability_metrics(y_march, selected_probability)},
    ])
    return metrics, selected_probability  # Returns the benchmark table and game-level champion prices.


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


def save_figures(march_metrics: pd.DataFrame, calibration: pd.DataFrame, coefficients: pd.DataFrame, figure_dir: Path) -> None:  # Creates concise interview visuals.
    figure_dir.mkdir(parents=True, exist_ok=True)  # Ensures the destination exists.
    figure = plt.figure(figsize=(8, 5))  # Opens the model-comparison chart.
    axis = figure.add_subplot(111)  # Creates one plotting area.
    axis.bar(march_metrics["model"], march_metrics["log_loss"])  # Compares the primary temporal-check score.
    axis.set_ylabel("March log loss (lower is better)")  # Labels the decision metric.
    axis.tick_params(axis="x", rotation=20)  # Prevents long model names from overlapping.
    figure.tight_layout()  # Keeps all labels visible.
    figure.savefig(figure_dir / "march_model_comparison.png", dpi=180)  # Saves a screen-share-ready image.
    plt.close(figure)  # Releases plotting memory.
    figure = plt.figure(figsize=(6, 6))  # Opens the reliability diagram.
    axis = figure.add_subplot(111)  # Creates one calibration plotting area.
    axis.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")  # Adds the ideal reference.
    axis.plot(calibration["mean_probability"], calibration["actual_home_win_rate"], marker="o", label="selected model")  # Plots observed versus forecast rates.
    axis.set_xlabel("Mean predicted home-win probability")  # Labels forecast confidence.
    axis.set_ylabel("Observed home-win rate")  # Labels realized frequency.
    axis.legend()  # Identifies the lines.
    figure.tight_layout()  # Avoids clipped labels.
    figure.savefig(figure_dir / "march_calibration.png", dpi=180)  # Saves the reliability chart.
    plt.close(figure)  # Releases plotting memory.
    figure = plt.figure(figsize=(8, 5))  # Opens the standardized coefficient chart.
    axis = figure.add_subplot(111)  # Creates one coefficient plotting area.
    axis.bar(coefficients["feature"], coefficients["coefficient_per_one_sd"])  # Shows directly comparable effects.
    axis.set_ylabel("Log-odds coefficient per 1 SD")  # Explains the standardized units.
    axis.tick_params(axis="x", rotation=20)  # Keeps feature names readable.
    figure.tight_layout()  # Avoids clipped labels.
    figure.savefig(figure_dir / "final_model_coefficients.png", dpi=180)  # Saves the interpretability visual.
    plt.close(figure)  # Releases plotting memory.


def run(data_path: Path, output_dir: Path) -> None:  # Executes the complete auditable workflow.
    output_dir.mkdir(parents=True, exist_ok=True)  # Creates the artifact directory.
    data = load_and_validate(data_path)  # Loads and audits the supplied CSV.
    half_life, selected_c, validation_grid, features = tune_model(data)  # Locks the model using January-February only.
    march_metrics, march_probability = evaluate_march(features, selected_c)  # Runs the final pre-April temporal check.
    march_mask = features["game_date"].between(MARCH_START, MARCH_END)  # Recreates the March index for diagnostics.
    march_calibration = calibration_bins(features.loc[march_mask, "home_win"], march_probability)  # Builds the reliability table.
    final_train_mask = features["game_date"] <= MARCH_END  # Uses all permitted October-March development data.
    april_sequential_mask = features["game_date"] >= APRIL_START  # Identifies daily-as-of-game-time April rows.
    final_model = make_model(selected_c)  # Recreates the locked specification for final fitting.
    final_model.fit(features.loc[final_train_mask, FEATURE_COLUMNS], features.loc[final_train_mask, "home_win"])  # Refits on all permitted development games.
    daily_probability = final_model.predict_proba(features.loc[april_sequential_mask, FEATURE_COLUMNS])[:, 1]  # Produces operational daily-repricing probabilities.
    frozen_features = build_frozen_features(data, half_life, MARCH_END)  # Creates a strict March-31 information snapshot.
    frozen_april = frozen_features.loc[frozen_features["game_date"] >= APRIL_START].reset_index(drop=True)  # Keeps only requested April games.
    frozen_probability = final_model.predict_proba(frozen_april[FEATURE_COLUMNS])[:, 1]  # Prices every April game without any April update.
    sequential_april = features.loc[april_sequential_mask, ["game_id", "game_date", "away", "home"]].reset_index(drop=True)  # Builds the submission identifiers.
    if sequential_april["game_id"].tolist() != frozen_april["game_id"].tolist():  # Protects against silent probability-row misalignment.
        raise ValueError("Frozen and sequential April rows are not aligned.")  # Stops before writing incorrect prices.
    predictions = sequential_april.copy()  # Creates the final clean output table.
    submitted_probability = np.round(frozen_probability, 10)  # Defines one canonical submitted probability before deriving any displayed odds.
    submitted_daily_probability = np.round(daily_probability, 10)  # Applies the same serialization contract to the operational repricing illustration.
    predictions["home_win_probability"] = submitted_probability  # Uses the strict March-31 cutoff as the official answer.
    predictions["daily_repricing_probability"] = submitted_daily_probability  # Shows the operational sportsbook timing sensitivity.
    predictions["fair_home_decimal_odds"] = 1.0 / submitted_probability  # Derives fair home odds from the exact probability written to the output.
    predictions["fair_away_decimal_odds"] = 1.0 / (1.0 - submitted_probability)  # Derives fair away odds from the same canonical submitted price.
    coefficients = coefficient_table(final_model)  # Extracts final effect sizes and the equal-strength home-court baseline.
    standardized_intercept_row = coefficients.loc[coefficients["term"] == "standardized_intercept"].iloc[0]  # Retrieves the fitted centered-scale intercept.
    home_advantage_row = coefficients.loc[coefficients["term"] == "equal_strength_home_advantage"].iloc[0]  # Retrieves the sportsbook-relevant equal-team baseline.
    home_advantage_summary = {  # Stores both quantities so the centering distinction is explicit.
        "standardized_intercept_log_odds": float(standardized_intercept_row["coefficient_standardized"]),
        "standardized_intercept_probability": float(standardized_intercept_row["reference_home_win_probability"]),
        "equal_strength_home_log_odds": float(home_advantage_row["coefficient_standardized"]),
        "equal_strength_home_win_probability": float(home_advantage_row["reference_home_win_probability"]),
        "equal_strength_home_odds_multiplier": float(home_advantage_row["odds_multiplier"]),
    }
    april_metrics = pd.DataFrame([  # Descriptively scores the locked frozen prices against supplied April outcomes.
        {"model": "selected_three_feature_logistic", "split": "April descriptive audit", "n": len(frozen_april), **probability_metrics(frozen_april["home_win"], submitted_probability)}
    ])
    validation_grid.to_csv(output_dir / "validation_grid.csv", index=False)  # Preserves every tuning result.
    march_metrics.to_csv(output_dir / "march_temporal_check_metrics.csv", index=False)  # Saves the pre-April benchmark comparison.
    march_calibration.to_csv(output_dir / "march_calibration_bins.csv", index=False)  # Saves reliability diagnostics.
    coefficients.to_csv(output_dir / "final_model_coefficients.csv", index=False)  # Saves interpretable final effects.
    predictions.to_csv(output_dir / "april_predictions.csv", index=False, float_format="%.10f")  # Saves one internally consistent ten-decimal probability-and-odds contract.
    april_metrics.to_csv(output_dir / "april_descriptive_metrics.csv", index=False)  # Saves the post-forecast descriptive audit.
    with (output_dir / "selected_hyperparameters.json").open("w", encoding="utf-8") as file:  # Opens a reproducibility record.
        json.dump({"half_life": half_life, "C": selected_c, "features": FEATURE_COLUMNS}, file, indent=2)  # Writes every locked model choice.
    with (output_dir / "model_summary.json").open("w", encoding="utf-8") as file:  # Opens the concise fitted-model summary.
        json.dump({"selected_parameters": {"half_life": half_life, "C": selected_c, "features": FEATURE_COLUMNS}, "home_advantage": home_advantage_summary}, file, indent=2)  # Writes the centered intercept and equal-strength home advantage separately.
    save_figures(march_metrics, march_calibration, coefficients.loc[coefficients["term_type"] == "feature"].rename(columns={"term": "feature", "coefficient_standardized": "coefficient_per_one_sd"}), output_dir.parent / "figures")  # Creates feature-only presentation graphics.
    print(json.dumps({  # Prints a concise run summary for screen sharing.
        "selected_parameters": {"half_life": half_life, "C": selected_c, "features": FEATURE_COLUMNS},
        "home_advantage": home_advantage_summary,
        "march_temporal_check": march_metrics.to_dict(orient="records"),
        "april_descriptive_audit": april_metrics.to_dict(orient="records"),
        "april_predictions": len(predictions),
    }, indent=2, default=str))


def parse_args() -> argparse.Namespace:  # Defines an interview-friendly command-line interface.
    parser = argparse.ArgumentParser(description="Build NBA home-win probabilities without target leakage.")  # Creates helpful usage text.
    parser.add_argument("--data", type=Path, required=True, help="Path to nba-win-probability-data.csv")  # Requires an explicit input path.
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for predictions and diagnostics")  # Allows clean output redirection.
    return parser.parse_args()  # Returns validated command-line arguments.


if __name__ == "__main__":  # Runs only when the file is executed directly.
    arguments = parse_args()  # Reads command-line paths.
    run(arguments.data, arguments.output_dir)  # Executes the complete model pipeline.
