# Reviewer Guide

## Two-minute route

1. Read [`SUMMARY.md`](../SUMMARY.md).
2. Open [`outputs/april_predictions.csv`](../outputs/april_predictions.csv).
3. Confirm the official model in [`outputs/selected_model.json`](../outputs/selected_model.json).

## Ten-minute route

### 1. Business target

The model prices the probability that the listed home team wins each April
game from a March 31 information snapshot.

### 2. Leakage control

Open [`nba_win_probability.py`](../nba_win_probability.py) and search:

```text
build_sequential_features
```

Features are captured before the current result updates team state.

### 3. Sports features

Search:

```text
feature_values
```

Every component uses the same three home-minus-away signals.

### 4. Ensemble construction

Search:

```text
component_predictions
ensemble_probability
```

The official price is the arithmetic mean of 40 fixed component probabilities.

### 5. Promotion evidence

Open:

- [`outputs/ensemble_validation_metrics.csv`](../outputs/ensemble_validation_metrics.csv)
- [`outputs/march_temporal_check_metrics.csv`](../outputs/march_temporal_check_metrics.csv)
- [`outputs/governance_monthly_backtest.csv`](../outputs/governance_monthly_backtest.csv)

The ensemble improves January-February validation and March governance log
loss before the April audit.

### 6. Interpretability

Open:

- [`outputs/ensemble_component_summary.csv`](../outputs/ensemble_component_summary.csv)
- [`outputs/single_model_benchmark_april_predictions.csv`](../outputs/single_model_benchmark_april_predictions.csv)
- [Ensemble Method](ENSEMBLE_METHOD.md)

### 7. Limitations and production translation

Read:

- [Model Card](MODEL_CARD.md)
- [Limitations and Roadmap](LIMITATIONS_AND_ROADMAP.md)

## Thirty-minute technical route

1. Run `bash scripts/run_quality_checks.sh`.
2. Run the official model using the supplied CSV.
3. Inspect the strict April output.
4. Walk through sequential state construction.
5. Walk through component fitting and probability averaging.
6. Compare ensemble and single benchmark.
7. Review rich challengers and bootstrap uncertainty.
8. Discuss late-season target relevance and production extensions.

## Official versus benchmark versus research

### Official

- `nba_win_probability.py`
- Uniform 40-component ensemble
- `outputs/april_predictions.csv`

### Benchmark

- Best validation single component
- Half-life 12, `C = 0.0075`
- Separate April benchmark artifacts

### Governance

- `model_governance.py`
- Rich box-score, schedule, Bradley-Terry, Elo, bootstrap, and monthly analysis

### Research

- Historical single-model governance
- Team-specific home effects
- Optional ML challengers
