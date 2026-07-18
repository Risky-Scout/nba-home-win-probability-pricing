"""Reproduce locked champion-challenger, calibration, ensemble, and SHAP results.

The hyperparameter search spaces and selection rationale are documented in
MODEL_GOVERNANCE_APPENDIX.md. This script reruns the locked validation-selected
specifications so it remains suitable for a live interview screen share.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from project_runtime import require_supported_python

require_supported_python()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

import nba_win_probability as champion


CORE_FEATURES = champion.FEATURE_COLUMNS
TRAIN_END = pd.Timestamp("2025-12-31")
VALIDATION_START = pd.Timestamp("2026-01-01")
VALIDATION_END = pd.Timestamp("2026-02-28")
MARCH_START = pd.Timestamp("2026-03-01")
MARCH_END = pd.Timestamp("2026-03-31")
RANDOM_SEED = 365

LOCKED_PARAMETERS = {
    "extra_trees": {
        "n_estimators": 200,
        "max_depth": 2,
        "min_samples_leaf": 10,
        "max_features": 1.0,
    },
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 2,
        "min_samples_leaf": 50,
        "max_features": "sqrt",
    },
    "catboost": {
        "depth": 3,
        "learning_rate": 0.02,
        "iterations": 100,
        "l2_leaf_reg": 3,
    },
    "xgboost": {
        "max_depth": 1,
        "min_child_weight": 30,
        "learning_rate": 0.05,
        "n_estimators": 50,
        "reg_lambda": 1.0,
    },
    "xgboost_residual": {
        "max_depth": 1,
        "min_child_weight": 60,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "reg_lambda": 5.0,
    },
    "rich_xgboost": {
        "max_depth": 1,
        "min_child_weight": 30,
        "learning_rate": 0.10,
        "n_estimators": 50,
        "reg_lambda": 5.0,
    },
    "rich_extra_trees": {
        "n_estimators": 200,
        "max_depth": 3,
        "min_samples_leaf": 50,
        "max_features": 1.0,
    },
    "rich_random_forest": {
        "n_estimators": 200,
        "max_depth": 3,
        "min_samples_leaf": 50,
        "max_features": "sqrt",
    },
    "rich_catboost": {
        "depth": 2,
        "learning_rate": 0.05,
        "iterations": 50,
        "l2_leaf_reg": 10,
    },
}


@dataclass
class RichState:
    """Hold only information observable before a team's next game."""

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


def safe_logit(probability: np.ndarray | pd.Series) -> np.ndarray:
    """Convert probabilities to finite log odds."""

    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def common_masks(table: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return train, validation, train-plus-validation, and March masks."""

    return (
        table["game_date"] <= TRAIN_END,
        table["game_date"].between(VALIDATION_START, VALIDATION_END),
        table["game_date"] <= VALIDATION_END,
        table["game_date"].between(MARCH_START, MARCH_END),
    )


def metric_row(
    model_name: str,
    validation_outcome: pd.Series,
    validation_probability: np.ndarray,
    march_outcome: pd.Series,
    march_probability: np.ndarray,
) -> dict[str, float | str]:
    """Create one directly comparable model row."""

    return {
        "model": model_name,
        "validation_log_loss": float(
            log_loss(validation_outcome, validation_probability)
        ),
        "validation_brier": float(
            brier_score_loss(validation_outcome, validation_probability)
        ),
        "march_log_loss": float(log_loss(march_outcome, march_probability)),
        "march_brier": float(
            brier_score_loss(march_outcome, march_probability)
        ),
        "march_auc": float(roc_auc_score(march_outcome, march_probability)),
    }


def xgboost_model(parameters: dict[str, object]) -> XGBClassifier:
    """Construct a deterministic XGBoost probability model."""

    return XGBClassifier(
        **parameters,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        verbosity=0,
    )


def fit_two_periods(
    features: pd.DataFrame,
    make_model: Callable[[], object],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the same locked specification for validation and March."""

    train, validation, train_validation, march = common_masks(features)

    validation_model = make_model()
    validation_model.fit(
        features.loc[train, CORE_FEATURES],
        features.loc[train, "home_win"],
    )
    validation_probability = validation_model.predict_proba(
        features.loc[validation, CORE_FEATURES]
    )[:, 1]

    march_model = make_model()
    march_model.fit(
        features.loc[train_validation, CORE_FEATURES],
        features.loc[train_validation, "home_win"],
    )
    march_probability = march_model.predict_proba(
        features.loc[march, CORE_FEATURES]
    )[:, 1]

    return validation_probability, march_probability


def expanding_logistic_predictions(features: pd.DataFrame) -> pd.DataFrame:
    """Create expanding-window logistic scores for meta-models and calibration."""

    blocks: list[pd.DataFrame] = []

    for month in pd.period_range("2025-11", "2026-02", freq="M"):
        start = month.start_time
        end = month.end_time
        train = features["game_date"] < start
        test = features["game_date"].between(start, end)

        model = champion.make_model(0.0075)
        model.fit(
            features.loc[train, CORE_FEATURES],
            features.loc[train, "home_win"],
        )
        probability = model.predict_proba(
            features.loc[test, CORE_FEATURES]
        )[:, 1]

        block = features.loc[
            test,
            ["game_id", "game_date", "home_win", *CORE_FEATURES],
        ].copy()
        block["base_probability"] = probability
        block["base_logit"] = safe_logit(probability)
        blocks.append(block)

    return pd.concat(blocks).sort_values(["game_date", "game_id"]).reset_index(
        drop=True
    )


def residual_xgboost_probabilities(
    features: pd.DataFrame,
    validation_base_probability: np.ndarray,
    march_base_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Learn nonlinear corrections on top of cross-fitted logistic log odds."""

    _, validation, _, march = common_masks(features)
    oof = expanding_logistic_predictions(features)
    meta_train = oof[oof["game_date"] <= TRAIN_END]

    validation_model = xgboost_model(LOCKED_PARAMETERS["xgboost_residual"])
    validation_model.fit(
        meta_train[CORE_FEATURES],
        meta_train["home_win"],
        base_margin=meta_train["base_logit"].to_numpy(),
    )
    validation_probability = validation_model.predict_proba(
        features.loc[validation, CORE_FEATURES],
        base_margin=safe_logit(validation_base_probability),
    )[:, 1]

    march_model = xgboost_model(LOCKED_PARAMETERS["xgboost_residual"])
    march_model.fit(
        oof[CORE_FEATURES],
        oof["home_win"],
        base_margin=oof["base_logit"].to_numpy(),
    )
    march_probability = march_model.predict_proba(
        features.loc[march, CORE_FEATURES],
        base_margin=safe_logit(march_base_probability),
    )[:, 1]

    return validation_probability, march_probability


def fit_platt(
    probability: np.ndarray,
    outcome: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit an intercept and slope to existing log odds."""

    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    model.fit(safe_logit(probability).reshape(-1, 1), outcome)

    def predict(new_probability: np.ndarray) -> np.ndarray:
        return model.predict_proba(
            safe_logit(new_probability).reshape(-1, 1)
        )[:, 1]

    return predict


def fit_beta(
    probability: np.ndarray,
    outcome: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit beta calibration using log(p) and minus log(1-p)."""

    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    design = np.column_stack((np.log(clipped), -np.log(1.0 - clipped)))
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    model.fit(design, outcome)

    def predict(new_probability: np.ndarray) -> np.ndarray:
        new_clipped = np.clip(new_probability, 1e-6, 1.0 - 1e-6)
        new_design = np.column_stack(
            (np.log(new_clipped), -np.log(1.0 - new_clipped))
        )
        return model.predict_proba(new_design)[:, 1]

    return predict


def fit_isotonic(
    probability: np.ndarray,
    outcome: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a monotonic nonparametric calibration mapping."""

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(probability, outcome)

    def predict(new_probability: np.ndarray) -> np.ndarray:
        return np.clip(model.predict(new_probability), 1e-6, 1.0 - 1e-6)

    return predict


def calibration_table(
    oof: pd.DataFrame,
    march_probability: np.ndarray,
    march_outcome: pd.Series,
) -> pd.DataFrame:
    """Select calibration on February and report the subsequent March score."""

    january = oof[oof["game_date"].dt.month == 1]
    february = oof[oof["game_date"].dt.month == 2]
    january_february = oof[
        oof["game_date"].between(VALIDATION_START, VALIDATION_END)
    ]

    rows = [
        {
            "method": "Identity / no recalibration",
            "february_log_loss": log_loss(
                february["home_win"], february["base_probability"]
            ),
            "february_brier": brier_score_loss(
                february["home_win"], february["base_probability"]
            ),
            "march_log_loss": log_loss(march_outcome, march_probability),
            "march_brier": brier_score_loss(march_outcome, march_probability),
        }
    ]

    for name, fitter in (
        ("Beta calibration", fit_beta),
        ("Platt calibration", fit_platt),
        ("Isotonic calibration", fit_isotonic),
    ):
        february_mapping = fitter(
            january["base_probability"].to_numpy(),
            january["home_win"].to_numpy(),
        )
        calibrated_february = february_mapping(
            february["base_probability"].to_numpy()
        )

        march_mapping = fitter(
            january_february["base_probability"].to_numpy(),
            january_february["home_win"].to_numpy(),
        )
        calibrated_march = march_mapping(march_probability)

        rows.append(
            {
                "method": name,
                "february_log_loss": log_loss(
                    february["home_win"], calibrated_february
                ),
                "february_brier": brier_score_loss(
                    february["home_win"], calibrated_february
                ),
                "march_log_loss": log_loss(
                    march_outcome, calibrated_march
                ),
                "march_brier": brier_score_loss(
                    march_outcome, calibrated_march
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("february_log_loss")


def convex_blends(
    validation_outcome: pd.Series,
    march_outcome: pd.Series,
    logistic_validation: np.ndarray,
    logistic_march: np.ndarray,
    challengers: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Select each probability-blend weight using validation log loss."""

    rows: list[dict[str, float | str]] = []

    for name, (validation_probability, march_probability) in challengers.items():
        candidates: list[tuple[float, float, float]] = []

        for weight in np.linspace(0.0, 1.0, 101):
            blend = (
                (1.0 - weight) * logistic_validation
                + weight * validation_probability
            )
            candidates.append(
                (
                    log_loss(validation_outcome, blend),
                    brier_score_loss(validation_outcome, blend),
                    float(weight),
                )
            )

        validation_loss, validation_brier, weight = min(
            candidates, key=lambda result: (result[0], result[1])
        )
        march_blend = (
            (1.0 - weight) * logistic_march
            + weight * march_probability
        )

        rows.append(
            {
                "ensemble": f"Logistic + {name}",
                "challenger_weight": weight,
                "validation_log_loss": validation_loss,
                "validation_brier": validation_brier,
                "march_log_loss": log_loss(march_outcome, march_blend),
                "march_brier": brier_score_loss(march_outcome, march_blend),
            }
        )

    rows.append(
        {
            "ensemble": "Logistic alone",
            "challenger_weight": 0.0,
            "validation_log_loss": log_loss(
                validation_outcome, logistic_validation
            ),
            "validation_brier": brier_score_loss(
                validation_outcome, logistic_validation
            ),
            "march_log_loss": log_loss(march_outcome, logistic_march),
            "march_brier": brier_score_loss(march_outcome, logistic_march),
        }
    )

    return pd.DataFrame(rows).sort_values("validation_log_loss")


def update_ewma(previous: float, value: float, games: int, alpha: float) -> float:
    """Apply one postgame EWMA update."""

    return value if games == 0 else alpha * value + (1.0 - alpha) * previous


def build_rich_features(data: pd.DataFrame, half_life: float = 12.0) -> pd.DataFrame:
    """Build lagged box-score and schedule features for the SHAP challenger."""

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
                state.ewma_margin, margin_value, state.games, alpha
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


def date_block_bootstrap(
    dates: pd.Series,
    outcome: pd.Series,
    champion_probability: np.ndarray,
    challenger_probability: np.ndarray,
    samples: int = 2000,
) -> dict[str, float]:
    """Estimate paired log-loss uncertainty by resampling game dates."""

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
    differences: list[float] = []

    for _ in range(samples):
        sampled_dates = random.choice(
            unique_dates, size=len(unique_dates), replace=True
        )
        sampled_indices = np.concatenate(
            [
                frame.index[frame["date"] == sampled_date].to_numpy()
                for sampled_date in sampled_dates
            ]
        )
        sampled = frame.loc[sampled_indices]
        differences.append(
            log_loss(sampled["outcome"], sampled["challenger"])
            - log_loss(sampled["outcome"], sampled["champion"])
        )

    lower, median, upper = np.quantile(differences, (0.025, 0.5, 0.975))
    return {
        "observed_difference": (
            log_loss(outcome, challenger_probability)
            - log_loss(outcome, champion_probability)
        ),
        "date_block_bootstrap_2_5_percent": float(lower),
        "date_block_bootstrap_median": float(median),
        "date_block_bootstrap_97_5_percent": float(upper),
    }


def run(data_path: Path, output_dir: Path) -> None:
    """Run the locked reproducible model-governance analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir.parent / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = champion.load_and_validate(data_path)
    features = champion.build_sequential_features(data, half_life=12.0)
    train, validation, train_validation, march = common_masks(features)
    validation_outcome = features.loc[validation, "home_win"]
    march_outcome = features.loc[march, "home_win"]

    logistic_validation, logistic_march = fit_two_periods(
        features,
        lambda: champion.make_model(0.0075),
    )

    extra_validation, extra_march = fit_two_periods(
        features,
        lambda: ExtraTreesClassifier(
            **LOCKED_PARAMETERS["extra_trees"],
            criterion="log_loss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    )
    forest_validation, forest_march = fit_two_periods(
        features,
        lambda: RandomForestClassifier(
            **LOCKED_PARAMETERS["random_forest"],
            criterion="log_loss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    )
    cat_validation, cat_march = fit_two_periods(
        features,
        lambda: CatBoostClassifier(
            **LOCKED_PARAMETERS["catboost"],
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
        ),
    )
    xgb_validation, xgb_march = fit_two_periods(
        features,
        lambda: xgboost_model(LOCKED_PARAMETERS["xgboost"]),
    )
    residual_validation, residual_march = residual_xgboost_probabilities(
        features,
        logistic_validation,
        logistic_march,
    )

    model_probabilities = {
        "Logistic + XGBoost residual correction": (
            residual_validation,
            residual_march,
        ),
        "L2 logistic champion": (logistic_validation, logistic_march),
        "ExtraTrees": (extra_validation, extra_march),
        "CatBoost": (cat_validation, cat_march),
        "Random forest": (forest_validation, forest_march),
        "XGBoost": (xgb_validation, xgb_march),
    }

    challenger_rows = [
        metric_row(
            name,
            validation_outcome,
            validation_probability,
            march_outcome,
            march_probability,
        )
        for name, (validation_probability, march_probability)
        in model_probabilities.items()
    ]
    challenger_table = pd.DataFrame(challenger_rows).sort_values(
        "validation_log_loss"
    )
    challenger_table.to_csv(
        output_dir / "challenger_benchmark.csv", index=False
    )

    blend_table = convex_blends(
        validation_outcome,
        march_outcome,
        logistic_validation,
        logistic_march,
        {
            "ExtraTrees": (extra_validation, extra_march),
            "Random forest": (forest_validation, forest_march),
            "CatBoost": (cat_validation, cat_march),
            "XGBoost": (xgb_validation, xgb_march),
        },
    )
    blend_table.to_csv(output_dir / "ensemble_benchmark.csv", index=False)

    oof = expanding_logistic_predictions(features)
    calibration = calibration_table(oof, logistic_march, march_outcome)
    calibration.to_csv(
        output_dir / "calibration_benchmark.csv", index=False
    )

    bootstrap = pd.DataFrame(
        [
            {
                "comparison": "Residual XGBoost minus logistic - validation",
                **date_block_bootstrap(
                    features.loc[validation, "game_date"],
                    validation_outcome,
                    logistic_validation,
                    residual_validation,
                ),
            },
            {
                "comparison": "Residual XGBoost minus logistic - March",
                **date_block_bootstrap(
                    features.loc[march, "game_date"],
                    march_outcome,
                    logistic_march,
                    residual_march,
                ),
            },
        ]
    )
    bootstrap.to_csv(
        output_dir / "xgboost_residual_bootstrap.csv", index=False
    )

    rich_features = build_rich_features(data)
    rich_columns = [
        column
        for column in rich_features.columns
        if column
        not in {"game_id", "game_date", "home", "away", "home_win"}
    ]
    rich_train, rich_validation, rich_train_validation, rich_march = common_masks(
        rich_features
    )
    rich_validation_outcome = rich_features.loc[rich_validation, "home_win"]
    rich_march_outcome = rich_features.loc[rich_march, "home_win"]

    rich_extra_validation_model = ExtraTreesClassifier(
        **LOCKED_PARAMETERS["rich_extra_trees"],
        criterion="log_loss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rich_extra_validation_model.fit(
        rich_features.loc[rich_train, rich_columns],
        rich_features.loc[rich_train, "home_win"],
    )
    rich_extra_validation = rich_extra_validation_model.predict_proba(
        rich_features.loc[rich_validation, rich_columns]
    )[:, 1]
    rich_extra_march_model = ExtraTreesClassifier(
        **LOCKED_PARAMETERS["rich_extra_trees"],
        criterion="log_loss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rich_extra_march_model.fit(
        rich_features.loc[rich_train_validation, rich_columns],
        rich_features.loc[rich_train_validation, "home_win"],
    )
    rich_extra_march = rich_extra_march_model.predict_proba(
        rich_features.loc[rich_march, rich_columns]
    )[:, 1]

    rich_forest_validation_model = RandomForestClassifier(
        **LOCKED_PARAMETERS["rich_random_forest"],
        criterion="log_loss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rich_forest_validation_model.fit(
        rich_features.loc[rich_train, rich_columns],
        rich_features.loc[rich_train, "home_win"],
    )
    rich_forest_validation = rich_forest_validation_model.predict_proba(
        rich_features.loc[rich_validation, rich_columns]
    )[:, 1]
    rich_forest_march_model = RandomForestClassifier(
        **LOCKED_PARAMETERS["rich_random_forest"],
        criterion="log_loss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rich_forest_march_model.fit(
        rich_features.loc[rich_train_validation, rich_columns],
        rich_features.loc[rich_train_validation, "home_win"],
    )
    rich_forest_march = rich_forest_march_model.predict_proba(
        rich_features.loc[rich_march, rich_columns]
    )[:, 1]

    rich_xgb_validation_model = xgboost_model(
        LOCKED_PARAMETERS["rich_xgboost"]
    )
    rich_xgb_validation_model.set_params(colsample_bytree=0.8)
    rich_xgb_validation_model.fit(
        rich_features.loc[rich_train, rich_columns],
        rich_features.loc[rich_train, "home_win"],
    )
    rich_xgb_validation = rich_xgb_validation_model.predict_proba(
        rich_features.loc[rich_validation, rich_columns]
    )[:, 1]
    rich_xgb_march_model = xgboost_model(
        LOCKED_PARAMETERS["rich_xgboost"]
    )
    rich_xgb_march_model.set_params(colsample_bytree=0.8)
    rich_xgb_march_model.fit(
        rich_features.loc[rich_train_validation, rich_columns],
        rich_features.loc[rich_train_validation, "home_win"],
    )
    rich_xgb_march = rich_xgb_march_model.predict_proba(
        rich_features.loc[rich_march, rich_columns]
    )[:, 1]

    rich_cat_columns = ["home", "away", *rich_columns]
    rich_cat_validation_model = CatBoostClassifier(
        **LOCKED_PARAMETERS["rich_catboost"],
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_SEED,
        random_strength=0.0,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    rich_cat_validation_model.fit(
        rich_features.loc[rich_train, rich_cat_columns],
        rich_features.loc[rich_train, "home_win"],
        cat_features=[0, 1],
    )
    rich_cat_validation = rich_cat_validation_model.predict_proba(
        rich_features.loc[rich_validation, rich_cat_columns]
    )[:, 1]
    rich_cat_march_model = CatBoostClassifier(
        **LOCKED_PARAMETERS["rich_catboost"],
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_SEED,
        random_strength=0.0,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    rich_cat_march_model.fit(
        rich_features.loc[rich_train_validation, rich_cat_columns],
        rich_features.loc[rich_train_validation, "home_win"],
        cat_features=[0, 1],
    )
    rich_cat_march = rich_cat_march_model.predict_proba(
        rich_features.loc[rich_march, rich_cat_columns]
    )[:, 1]

    rich_table = pd.DataFrame(
        [
            {
                "model": "L2 logistic champion - core features",
                "validation_log_loss": log_loss(
                    validation_outcome, logistic_validation
                ),
                "march_log_loss": log_loss(
                    march_outcome, logistic_march
                ),
                "march_brier": brier_score_loss(
                    march_outcome, logistic_march
                ),
            },
            {
                "model": "ExtraTrees - richer lagged features",
                "validation_log_loss": log_loss(
                    rich_validation_outcome, rich_extra_validation
                ),
                "march_log_loss": log_loss(
                    rich_march_outcome, rich_extra_march
                ),
                "march_brier": brier_score_loss(
                    rich_march_outcome, rich_extra_march
                ),
            },
            {
                "model": "XGBoost - richer lagged features",
                "validation_log_loss": log_loss(
                    rich_validation_outcome, rich_xgb_validation
                ),
                "march_log_loss": log_loss(
                    rich_march_outcome, rich_xgb_march
                ),
                "march_brier": brier_score_loss(
                    rich_march_outcome, rich_xgb_march
                ),
            },
            {
                "model": "Random forest - richer lagged features",
                "validation_log_loss": log_loss(
                    rich_validation_outcome, rich_forest_validation
                ),
                "march_log_loss": log_loss(
                    rich_march_outcome, rich_forest_march
                ),
                "march_brier": brier_score_loss(
                    rich_march_outcome, rich_forest_march
                ),
            },
            {
                "model": "CatBoost - richer features and team identifiers",
                "validation_log_loss": log_loss(
                    rich_validation_outcome, rich_cat_validation
                ),
                "march_log_loss": log_loss(
                    rich_march_outcome, rich_cat_march
                ),
                "march_brier": brier_score_loss(
                    rich_march_outcome, rich_cat_march
                ),
            },
        ]
    ).sort_values("validation_log_loss")
    rich_table.to_csv(
        output_dir / "rich_feature_challenger_benchmark.csv",
        index=False,
    )

    explainer = shap.TreeExplainer(rich_xgb_march_model)
    shap_values = np.asarray(
        explainer.shap_values(
            rich_features.loc[rich_march, rich_columns]
        )
    )
    shap_table = pd.DataFrame(
        {
            "feature": rich_columns,
            "mean_absolute_shap_on_march": np.abs(shap_values).mean(axis=0),
            "mean_shap_on_march": shap_values.mean(axis=0),
        }
    ).sort_values("mean_absolute_shap_on_march", ascending=False)
    shap_table.to_csv(
        output_dir / "xgboost_shap_importance.csv", index=False
    )

    with (output_dir / "locked_challenger_hyperparameters.json").open(
        "w", encoding="utf-8"
    ) as file:
        import json

        json.dump(LOCKED_PARAMETERS, file, indent=2)

    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    ordered = challenger_table.sort_values("march_log_loss")
    axis.bar(ordered["model"], ordered["march_log_loss"])
    axis.set_ylabel("March log loss - lower is better")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(
        figure_dir / "challenger_march_log_loss.png", dpi=180
    )
    plt.close(figure)

    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    top_shap = shap_table.head(10).sort_values(
        "mean_absolute_shap_on_march"
    )
    axis.barh(
        top_shap["feature"],
        top_shap["mean_absolute_shap_on_march"],
    )
    axis.set_xlabel("Mean absolute SHAP value")
    figure.tight_layout()
    figure.savefig(
        figure_dir / "xgboost_shap_importance.png", dpi=180
    )
    plt.close(figure)

    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(111)
    ordered_calibration = calibration.sort_values("march_log_loss")
    axis.bar(
        ordered_calibration["method"],
        ordered_calibration["march_log_loss"],
    )
    axis.set_ylabel("March log loss - lower is better")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(
        figure_dir / "calibration_comparison.png", dpi=180
    )
    plt.close(figure)

    print(challenger_table.to_string(index=False))
    print()
    print(blend_table.to_string(index=False))
    print()
    print(calibration.to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Define paths for the locked governance analysis."""

    parser = argparse.ArgumentParser(
        description="Reproduce locked challengers, ensembles, calibration, and SHAP."
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
        help="Directory for governance outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.data, arguments.output_dir)
