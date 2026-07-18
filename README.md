# NBA Home-Win Probability Pricing

[![CI](https://github.com/Risky-Scout/nba-home-win-probability-pricing/actions/workflows/tests.yml/badge.svg)](https://github.com/Risky-Scout/nba-home-win-probability-pricing/actions/workflows/tests.yml)

A leakage-safe, chronologically validated late-season probability-pricing
model for the bet365 NBA technical task.

## Start here

| Time available | Review path |
|---|---|
| 2 minutes | Read [SUMMARY.md](SUMMARY.md) and open [April predictions](outputs/april_predictions.csv) |
| 10 minutes | Follow the [Reviewer Guide](docs/REVIEWER_GUIDE.md) |
| 20 minutes | Read [Ensemble Method](docs/ENSEMBLE_METHOD.md) and inspect [the official model](nba_win_probability.py) |
| Full audit | Read [Model Evolution](docs/MODEL_EVOLUTION.md), [Model Card](docs/MODEL_CARD.md), and [Limitations](docs/LIMITATIONS_AND_ROADMAP.md) |
| Reproduce | Follow [Reproducibility](docs/REPRODUCIBILITY.md) |

## Business objective

Estimate the probability that the team listed as `home` wins each April game
using only information available through March.

The official answer is:

- [`outputs/april_predictions.csv`](outputs/april_predictions.csv)
- Column: `home_win_probability`

All 96 official April prices are frozen at the March 31 information cutoff.

## Official model

The final model is a **uniform 40-component ensemble of strongly regularized
logistic paired-comparison models**.

Every component uses the same three interpretable home-minus-away signals:

1. Net-wins differential.
2. Cumulative point-margin differential.
3. Evidence-weighted recent point-margin differential.

The components differ only in:

- EWMA half-life: 5, 8, 12, 16, or 24 games.
- Logistic `C`: 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, or 0.05.

The official price is:

\[
P_{ensemble} =
\frac{1}{40}
\sum_{k=1}^{40} P_k
\]

Equal weights are fixed. They are not fitted to April outcomes.

## Why ensemble averaging

The validation surface is flat: several half-life and regularization choices
perform almost identically. Selecting one grid minimum creates avoidable
hyperparameter-selection risk.

Averaging across the complete predeclared grid:

- Preserves the same interpretable feature family.
- Reduces dependence on one narrowly selected grid point.
- Improves January-February validation log loss.
- Preserves the direction in March governance.
- Adds negligible computational cost.

The best single model remains the coefficient-level benchmark.

## Pre-April selection evidence

| Model | Jan-Feb validation log loss | March governance log loss |
|---|---:|---:|
| Best single component | 0.627529 | 0.509645 |
| **Uniform 40-component ensemble** | **0.627259** | **0.508638** |

March was used as a governance check, not to tune component weights.

## April descriptive audit

April outcomes were not used to select component weights or aggregation.

| Model | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| Best single benchmark | 0.468596 | 0.150628 | **0.868196** | **81.250%** |
| **Official ensemble** | **0.467607** | **0.150287** | 0.865497 | 80.208% |

April is reported descriptively. The promotion case rests on pre-April
validation and March governance.

## Information timing

The current game's points, turnovers, fouls, and rebounds are postgame
information. They cannot predict the same game.

Each game is processed in this order:

1. Read both teams' pregame states.
2. Construct and store the feature row.
3. Observe the result.
4. Update both teams for later games.

For the official April file, every team state is frozen on March 31.

## Interpretability

The ensemble does not introduce new black-box features. Every component uses
the same three sports signals and the same logistic probability link.

For an equal-strength matchup, the average home-win baseline is approximately
**55.568%**.

Interpretation artifacts:

- [`outputs/ensemble_component_summary.csv`](outputs/ensemble_component_summary.csv)
- [`outputs/april_component_dispersion.csv`](outputs/april_component_dispersion.csv)
- [`outputs/single_model_benchmark_april_predictions.csv`](outputs/single_model_benchmark_april_predictions.csv)

## Repository map

```text
.
├── README.md
├── SUMMARY.md
├── nba_win_probability.py        # Official ensemble and benchmark
├── model_governance.py           # Promotion evidence and rich challengers
├── challenger_analysis.py        # Optional ML challengers
├── ablation_and_timing.py        # Feature subsets and timing
├── outputs/                      # Official prices and governance artifacts
├── figures/                      # Reviewer-facing diagnostics
├── docs/                         # Model, governance, and reproduction docs
├── research/                     # Non-promoted historical challengers
├── scripts/                      # Setup, privacy, and integrity utilities
└── tests/                        # Leakage, runtime, artifact, and docs tests
```

## Run locally

Supported Python: 3.11-3.13. Python 3.12 is recommended.

```bash
bash scripts/bootstrap_macos.sh
```

Run the official model:

```bash
python nba_win_probability.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs
```

Run the core submission workflow:

```bash
python run_submission.py \
  --root . \
  --data /path/to/nba-win-probability-data.csv
```

Run the data-free quality gate:

```bash
bash scripts/run_quality_checks.sh
```

## Limitations

This remains a one-season, team-level technical-task model. A production NBA
price should add injuries, expected lineups, player minutes, possession-based
strength, travel context, market information, overround, liability, and
multiple seasons.

See [`docs/LIMITATIONS_AND_ROADMAP.md`](docs/LIMITATIONS_AND_ROADMAP.md).
