# NBA Home-Win Probability Pricing — v1.4.1 Ensemble Champion

[![CI](https://github.com/Risky-Scout/nba-home-win-probability-pricing/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/Risky-Scout/nba-home-win-probability-pricing/actions/workflows/tests.yml)

A leakage-safe, chronologically validated probability-pricing submission for
the bet365 NBA technical task.

## Start here

| Review time | Route |
|---|---|
| 2 minutes | Read [SUMMARY.md](SUMMARY.md), then open the [96 April prices](outputs/april_predictions.csv) |
| 10 minutes | Follow the [Reviewer Guide](docs/REVIEWER_GUIDE.md) |
| 20 minutes | Read the [Ensemble Method](docs/ENSEMBLE_METHOD.md), inspect [the official code](nba_win_probability.py), and review [Model Evolution](docs/MODEL_EVOLUTION.md) |
| Reproduce | Follow [Reproducibility](docs/REPRODUCIBILITY.md) |
| Audit | Review the [Model Card](docs/MODEL_CARD.md), [Limitations](docs/LIMITATIONS_AND_ROADMAP.md), and [Artifact Manifest](docs/ARTIFACT_MANIFEST.md) |

## Decision at a glance

| Item | Final decision |
|---|---|
| Release | `v1.4.1-final-audit` |
| Deployment target | April home-win probabilities |
| Information cutoff | Strict March 31 snapshot |
| Official model | Uniform mean of 40 L2-logistic paired-comparison models |
| Shared sports signals | Net wins, cumulative point margin, recent point-margin evidence |
| Component variation | Five EWMA half-lives × eight regularization values |
| Primary benchmark | Validation-best single model: half-life 12, `C = 0.0075` |
| Primary metric | Forward log loss |
| April output | [`outputs/april_predictions.csv`](outputs/april_predictions.csv) |
| Interpretation | Feature-level ensemble plus all 40 component coefficient vectors |
| Claim strength | Modest target-relevant improvement; not statistically decisive |

## Business objective

Estimate the probability that the team listed as `home` wins each April game
using only information available through March.

The official deliverable is:

- [`outputs/april_predictions.csv`](outputs/april_predictions.csv)
- Probability column: `home_win_probability`
- 96 games
- Ten-character `game_id` values preserved
- Zero-margin home and away decimal odds derived from the submitted probability

No April result updates another official April price.

## Official model

The final model is a fixed equal-weight ensemble of 40 strongly regularized
logistic paired-comparison models.

Every component uses the same three home-minus-away signals:

1. `net_wins_diff`
2. `cumulative_margin_diff`
3. `recent_margin_evidence_diff`

The components differ only in:

- EWMA half-life: 5, 8, 12, 16, or 24 games.
- Logistic `C`: 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, or 0.05.

For component probability $p_k$:

$$
p_{\mathrm{official}}
=
\frac{1}{40}
\sum_{k=1}^{40} p_k
$$

Each weight is fixed at $1/40 = 0.025$. No stacking or calibration weights
are estimated.

## Why an ensemble earned promotion

The validation surface was flat: several plausible half-life and
regularization combinations produced nearly identical scores. Selecting only
one grid minimum would overstate certainty about a noisy hyperparameter
choice.

Uniform averaging:

- Reduces dependence on one narrowly selected grid point.
- Preserves the same sports features and logistic probability link.
- Avoids estimating 40 unstable stacking weights from 399 validation games.
- Keeps every component inspectable.
- Adds negligible runtime cost.
- Improves the target-relevant pre-April proper score.

The gain is intentionally described as **modest**, not statistically
decisive.

## Development and governance timeline

| Period | Role |
|---|---|
| October–December | Fit candidate component coefficients |
| January–February | Compare the fixed ensemble with the best single component |
| March | Later governance check; verify that the validation direction does not reverse |
| October–March | Refit every locked component |
| April | Requested forecast period and descriptive audit |

March was not used to fit component weights. April has been reviewed
descriptively and is not represented as a pristine untouched test.

## Pre-April selection evidence

| Model | January–February log loss | March governance log loss |
|---|---:|---:|
| Validation-best single component | 0.627529 | 0.509645 |
| **Official 40-component ensemble** | **0.627259** | **0.508638** |

The promotion case is based on the pre-April validation direction, its
persistence in March, target relevance, and reduced single-grid-point risk.
Paired bootstrap intervals include zero, so the gain is not presented as
statistically conclusive.

## April descriptive audit

| Model | Log loss | Brier | ROC AUC | Accuracy |
|---|---:|---:|---:|---:|
| Validation-best single benchmark | 0.468596 | 0.150628 | **0.868196** | **81.250%** |
| **Official ensemble** | **0.467607** | **0.150287** | 0.865497 | 80.208% |

Log loss and Brier score are primary because the required product is a
probability price. AUC and accuracy are secondary diagnostics.

## Information timing and leakage control

The current game's points, turnovers, fouls, and rebounds are postgame
information. They cannot predict that same game.

Each row is created in this order:

1. Read both teams' states before the game.
2. Construct and store the matchup features.
3. Observe the result.
4. Update both team states for later games.

For the official April batch, all team-performance states are frozen on
March 31.

## Interpretation

The ensemble remains interpretable at two levels:

1. **Sports-feature level:** every component uses the same three team-strength
   signals.
2. **Component level:** all 40 standardized coefficient vectors and home
   baselines are exported.

The validation-best single component remains the simplest coefficient-level
explanation and fallback benchmark.

Key interpretation artifacts:

- [`outputs/ensemble_component_summary.csv`](outputs/ensemble_component_summary.csv)
- [`outputs/april_component_dispersion.csv`](outputs/april_component_dispersion.csv)
- [`outputs/single_model_benchmark_april_predictions.csv`](outputs/single_model_benchmark_april_predictions.csv)
- [`figures/ensemble_mean_coefficients.png`](figures/ensemble_mean_coefficients.png)

For an equal-strength raw matchup, the mean component home-win baseline is
approximately **55.568%**.

## Model progression

| Stage | Question | Decision |
|---|---|---|
| Leakage-safe baseline | Can same-game information be excluded correctly? | Sequential feature-before-update state |
| Three-signal single model | What low-variance model prices the target well? | Strongly regularized logistic benchmark |
| ML and calibration challengers | Do nonlinear learners or mappings improve forward proper scores? | No stable promotion |
| Limitation-driven governance | Do opponent adjustment, shrinkage, PCA, or subsets earn inclusion? | Retain core signal family |
| Team-specific venue research | Does each team need a separate home effect? | Shadow challenger only |
| Reliability hardening | Can the project install, test, and validate consistently? | Python 3.11–3.13, CI, artifact checks |
| Reviewer navigation | Can a reviewer understand the decision quickly? | Guided 2-, 10-, and 20-minute routes |
| Ensemble promotion | Can specification risk be reduced for the late-season target? | Fixed 40-component official price |
| Final release audit | Does GitHub render cleanly and stay internally consistent? | Math, metadata, research labels, and dependency lock verified |

Full history: [docs/MODEL_EVOLUTION.md](docs/MODEL_EVOLUTION.md).

## Official, benchmark, governance, and research layers

| Layer | Main files |
|---|---|
| Official price | `nba_win_probability.py`, `outputs/april_predictions.csv` |
| Single benchmark | `outputs/single_model_benchmark_april_predictions.csv` |
| Promotion governance | `model_governance.py`, `outputs/governance_*` |
| Optional ML challengers | `challenger_analysis.py` |
| Historical/shadow research | [`research/README.md`](research/README.md) |

No research artifact silently replaces the official April price.

## Repository map

```text
.
├── README.md
├── SUMMARY.md
├── nba_win_probability.py        # Official ensemble and single benchmark
├── model_governance.py           # Promotion evidence and rich challengers
├── challenger_analysis.py        # Optional ML challengers
├── ablation_and_timing.py        # Feature subsets and timing
├── outputs/                      # Official prices and quantitative artifacts
├── figures/                      # Reviewer-facing diagnostics
├── docs/                         # Method, governance, limitations, reproduction
├── research/                     # Historical and non-promoted challengers
├── scripts/                      # Setup, privacy, and integrity utilities
├── tests/                        # Model, runtime, documentation, artifact contracts
└── validate_submission.py        # Full source-data submission validator
```

## Reproduce

Supported Python: **3.11–3.13**. Python 3.12 is recommended.

One-command macOS setup:

```bash
bash scripts/bootstrap_macos.sh
```

Run the official model:

```bash
python nba_win_probability.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs
```

Run promotion governance:

```bash
python model_governance.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs \
  --figure-dir figures
```

Run the complete core workflow:

```bash
python run_submission.py \
  --root . \
  --data /path/to/nba-win-probability-data.csv
```

Run the data-free quality gate:

```bash
bash scripts/run_quality_checks.sh
```

Run the full validator:

```bash
python validate_submission.py \
  --root . \
  --data /path/to/nba-win-probability-data.csv
```

## Scope and limitations

This is a strong one-season, team-level technical-task model—not a complete
production NBA sportsbook price.

Important missing information includes:

- Injuries and expected lineups.
- Player minutes and player impact.
- Possession-based offense, defense, and pace.
- Travel and schedule context.
- Market-implied probability.
- Multiple seasons.
- Trading margin, liability, limits, and manual overrides.

The ensemble advantage is small, and the hyperparameter grid acts like a
discrete prior over specifications. These limitations are documented rather
than hidden.

See [docs/LIMITATIONS_AND_ROADMAP.md](docs/LIMITATIONS_AND_ROADMAP.md).
