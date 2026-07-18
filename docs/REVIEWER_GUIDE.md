# Reviewer Guide

This page provides the shortest defensible route through the repository.

## Ten-minute review

### 1. Confirm the output

Open:

- [`outputs/april_predictions.csv`](../outputs/april_predictions.csv)

Check:

- 96 April rows.
- Ten-character `game_id` values.
- `home_win_probability`.
- Zero-margin home and away decimal odds.
- A separate daily-repricing sensitivity column.

### 2. Understand the information timestamp

Open [`nba_win_probability.py`](../nba_win_probability.py) and search for:

```text
build_sequential_features
```

The sequence is:

1. Read the teams' states before the game.
2. Construct and store the feature row.
3. Observe the result.
4. Update both teams for later games.

This prevents current-game points, turnovers, fouls, and rebounds from
predicting their own outcome.

### 3. Inspect the three signals

Search for:

```text
feature_values
```

The model uses:

- `net_wins_diff`
- `cumulative_margin_diff`
- `recent_margin_evidence_diff`

All are home minus away.

### 4. Inspect the probability model

Search for:

```text
make_model
```

The pipeline is:

- `StandardScaler`
- L2-regularized `LogisticRegression`

Selected `C = 0.0075`.

### 5. Inspect chronological selection

Search for:

```text
tune_model
```

The development design is:

- October-December training.
- January-February validation.
- March governance.
- Final refit through March.
- Frozen April forecast.

### 6. Review proper-score evidence

Open:

- [`outputs/march_temporal_check_metrics.csv`](../outputs/march_temporal_check_metrics.csv)
- [`outputs/enhanced_model_comparison.csv`](../outputs/enhanced_model_comparison.csv)
- [`outputs/challenger_benchmark.csv`](../outputs/challenger_benchmark.csv)
- [`outputs/calibration_benchmark.csv`](../outputs/calibration_benchmark.csv)

The selected March log loss is **0.509645**.

### 7. Review the decision logic

Read:

- [Model Evolution](MODEL_EVOLUTION.md)
- [Model Card](MODEL_CARD.md)
- [Limitations and Roadmap](LIMITATIONS_AND_ROADMAP.md)

The deployment rule is:

> Added complexity must improve forward proper scores materially and stably
> while preserving the pregame information timestamp, reproducibility, and
> interpretability.

## Thirty-minute technical review

1. Run `bash scripts/run_quality_checks.sh`.
2. Run the official model using the supplied CSV.
3. Inspect `outputs/april_predictions.csv`.
4. Walk through `build_sequential_features`.
5. Walk through `tune_model`.
6. Compare champion and challenger artifacts.
7. Inspect calibration and uncertainty.
8. Review limitations and production extensions.

## What is official versus research

### Official price

- `nba_win_probability.py`
- `outputs/april_predictions.csv`
- Three-signal L2 logistic model

### Governance analysis

- `enhanced_governance.py`
- `challenger_analysis.py`
- `ablation_and_timing.py`

### Shadow research

- `research/team_specific_home_effects.py`

Research artifacts do not silently replace the official price.
