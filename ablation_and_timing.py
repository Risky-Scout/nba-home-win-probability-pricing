"""Reproduce feature ablations and hardware-specific timing benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import nba_win_probability as champion


FEATURE_SETS = [
    ["cumulative_margin_diff", "recent_margin_evidence_diff"],
    ["cumulative_margin_diff"],
    ["net_wins_diff", "cumulative_margin_diff", "recent_margin_evidence_diff"],
    ["net_wins_diff", "cumulative_margin_diff"],
    ["net_wins_diff", "recent_margin_evidence_diff"],
    ["recent_margin_evidence_diff"],
    ["net_wins_diff"],
]

LOCKED_TIMING_MODELS: dict[str, Callable[[], object]] = {
    "L2 logistic": lambda: champion.make_model(0.0075),
    "XGBoost": lambda: XGBClassifier(
        max_depth=1,
        min_child_weight=30,
        learning_rate=0.05,
        n_estimators=50,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=champion.RANDOM_SEED,
        n_jobs=1,
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        verbosity=0,
    ),
    "CatBoost": lambda: CatBoostClassifier(
        depth=3,
        learning_rate=0.02,
        iterations=100,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=champion.RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    ),
    "ExtraTrees": lambda: ExtraTreesClassifier(
        n_estimators=200,
        max_depth=2,
        min_samples_leaf=10,
        max_features=1.0,
        criterion="log_loss",
        random_state=champion.RANDOM_SEED,
        n_jobs=-1,
    ),
}


def score(outcome: pd.Series, probability: np.ndarray) -> dict[str, float]:
    """Return directly comparable probability and classification diagnostics."""

    return {
        "log_loss": float(log_loss(outcome, probability)),
        "brier_score": float(brier_score_loss(outcome, probability)),
        "roc_auc": float(roc_auc_score(outcome, probability)),
        "accuracy_0_5": float(accuracy_score(outcome, probability >= 0.5)),
    }


def reproduce_ablation(features: pd.DataFrame) -> pd.DataFrame:
    """Tune each feature subset on January-February and evaluate it in March."""

    train_mask = features["game_date"] <= champion.TRAIN_END
    validation_mask = features["game_date"].between(
        champion.VALIDATION_START,
        champion.VALIDATION_END,
    )
    train_validation_mask = features["game_date"] <= champion.VALIDATION_END
    march_mask = features["game_date"].between(
        champion.MARCH_START,
        champion.MARCH_END,
    )
    rows: list[dict[str, object]] = []

    for columns in FEATURE_SETS:
        validation_candidates: list[tuple[float, float, float]] = []

        for c_value in champion.C_GRID:
            model = champion.make_model(c_value)
            model.fit(
                features.loc[train_mask, columns],
                features.loc[train_mask, "home_win"],
            )
            probability = model.predict_proba(
                features.loc[validation_mask, columns]
            )[:, 1]
            validation_candidates.append(
                (
                    float(log_loss(features.loc[validation_mask, "home_win"], probability)),
                    float(brier_score_loss(features.loc[validation_mask, "home_win"], probability)),
                    float(c_value),
                )
            )

        validation_log_loss, validation_brier, selected_c = min(
            validation_candidates,
            key=lambda result: (result[0], result[1], result[2]),
        )

        final_model = champion.make_model(selected_c)
        final_model.fit(
            features.loc[train_validation_mask, columns],
            features.loc[train_validation_mask, "home_win"],
        )
        march_probability = final_model.predict_proba(
            features.loc[march_mask, columns]
        )[:, 1]
        march_metrics = score(
            features.loc[march_mask, "home_win"],
            march_probability,
        )

        rows.append(
            {
                "features": " + ".join(columns),
                "number_of_features": len(columns),
                "selected_C": selected_c,
                "validation_log_loss": validation_log_loss,
                "validation_brier": validation_brier,
                "march_log_loss": march_metrics["log_loss"],
                "march_brier": march_metrics["brier_score"],
                "march_auc": march_metrics["roc_auc"],
                "march_accuracy_0_5": march_metrics["accuracy_0_5"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["validation_log_loss", "march_log_loss"]
    ).reset_index(drop=True)


def benchmark_model(
    constructor: Callable[[], object],
    train_features: pd.DataFrame,
    train_outcome: pd.Series,
    prediction_features: pd.DataFrame,
    repeats: int,
) -> dict[str, float]:
    """Measure model construction, fitting, and batch probability prediction."""

    fit_seconds: list[float] = []
    prediction_seconds: list[float] = []

    warmup = constructor()
    warmup.fit(train_features, train_outcome)
    warmup.predict_proba(prediction_features)

    for _ in range(repeats):
        model = constructor()
        fit_start = time.perf_counter()
        model.fit(train_features, train_outcome)
        fit_seconds.append(time.perf_counter() - fit_start)

        prediction_start = time.perf_counter()
        probability = model.predict_proba(prediction_features)[:, 1]
        prediction_seconds.append(time.perf_counter() - prediction_start)

        if not np.isfinite(probability).all():
            raise ValueError("A timing model produced a non-finite probability.")

    return {
        "median_fit_seconds": float(statistics.median(fit_seconds)),
        "minimum_fit_seconds": float(min(fit_seconds)),
        "maximum_fit_seconds": float(max(fit_seconds)),
        "median_prediction_seconds": float(statistics.median(prediction_seconds)),
        "minimum_prediction_seconds": float(min(prediction_seconds)),
        "maximum_prediction_seconds": float(max(prediction_seconds)),
    }


def reproduce_timing(features: pd.DataFrame, repeats: int) -> pd.DataFrame:
    """Benchmark locked model specifications on the current machine."""

    train_mask = features["game_date"] <= champion.TRAIN_END
    validation_mask = features["game_date"].between(
        champion.VALIDATION_START,
        champion.VALIDATION_END,
    )
    train_features = features.loc[train_mask, champion.FEATURE_COLUMNS]
    train_outcome = features.loc[train_mask, "home_win"]
    prediction_features = features.loc[validation_mask, champion.FEATURE_COLUMNS]
    rows: list[dict[str, object]] = []

    for name, constructor in LOCKED_TIMING_MODELS.items():
        timing = benchmark_model(
            constructor,
            train_features,
            train_outcome,
            prediction_features,
            repeats,
        )
        rows.append(
            {
                "model": name,
                "training_rows": len(train_features),
                "prediction_rows": len(prediction_features),
                "repeats": repeats,
                **timing,
            }
        )

    return pd.DataFrame(rows).sort_values("median_fit_seconds").reset_index(drop=True)


def environment_summary() -> dict[str, object]:
    """Record enough machine information to interpret hardware-dependent timings."""

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "timing_note": (
            "Timing is hardware- and load-dependent. Re-run this script on the "
            "machine used for the interview before committing or presenting timings."
        ),
    }


def run(data_path: Path, output_dir: Path, repeats: int, skip_timing: bool) -> None:
    """Generate all ablation and timing artifacts cited by the documentation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    data = champion.load_and_validate(data_path)
    features = champion.build_sequential_features(data, half_life=12.0)

    ablation = reproduce_ablation(features)
    ablation.to_csv(output_dir / "feature_ablation.csv", index=False)

    if not skip_timing:
        timing = reproduce_timing(features, repeats)
        timing.to_csv(output_dir / "model_timing_benchmark.csv", index=False)
        with (output_dir / "timing_environment.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(environment_summary(), file, indent=2)

    print(ablation.to_string(index=False))
    if not skip_timing:
        print()
        print(timing.to_string(index=False))
        print()
        print("Timing is machine-specific; re-run on the interview laptop.")


def parse_args() -> argparse.Namespace:
    """Define command-line paths and the timing repeat count."""

    parser = argparse.ArgumentParser(
        description="Reproduce feature ablations and hardware timing benchmarks."
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
        "--repeats",
        type=int,
        default=20,
        help="Number of fit/predict repetitions for each timing model",
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Generate the deterministic ablation table without hardware timings",
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be at least 1")
    return arguments


if __name__ == "__main__":
    args = parse_args()
    run(args.data, args.output_dir, args.repeats, args.skip_timing)
