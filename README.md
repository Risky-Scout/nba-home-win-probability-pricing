# NBA Home-Win Probability — Final Sportsbook Submission

## Objective

Estimate the home team's probability of winning each April NBA game using
October-through-March information.

The official answer is:

- `outputs/april_predictions.csv`
- Column: `home_win_probability`

Every official April probability is frozen at the March 31 information cutoff.
No April result enters another official April price.

## Executive decision

The selected model remains a **strongly regularized three-signal
paired-comparison logistic model**.

Features:

1. Home-minus-away net wins.
2. Home-minus-away cumulative point margin.
3. Home-minus-away evidence-weighted recent point margin.

Selected parameters:

- Recent-margin half-life: **12 games**
- Logistic `C`: **0.0075**

The model remains selected after explicit tests of the limitations raised
during review:

- Opponent-adjusted ridge team ratings.
- Pure EWMA recent form.
- Bayesian-shrunken EWMA recent form.
- One- and two-component PCA latent-strength models.
- Every two-feature subset.
- Rich lagged box-score and schedule features.
- Elo, XGBoost, CatBoost, random forest and ExtraTrees.
- Residual boosting and convex probability ensembles.
- Platt, beta and isotonic calibration.

No addition produced a material and temporally stable improvement in proper
probability scores.

## What “optimal” means here

This repository does **not** claim that the model is the universally optimal
NBA pricing system.

It is the strongest defensible submission conditional on:

- The supplied one-season dataset.
- The pregame information timestamp.
- The absence of injuries, lineups, player minutes and market odds.
- Proper probability scoring.
- Reproducibility and computational-efficiency requirements.
- The challenger families tested.
- A governance rule that complexity must earn deployment through a stable
  forward improvement.


## What “home-team win probability” means

The assignment asks for a probability for the team listed as `home` in each
April matchup.

The official model already produces a **team- and matchup-specific**
probability because the inputs depend on the specific home and away teams'
pregame states.

That should not be confused with estimating a different home-court advantage
for every team.

A separate hierarchically shrunk team-specific venue challenger is included in
`research/team_specific_home_effects.py`. It improves validation log loss
slightly, but the bootstrap interval includes zero, March is effectively tied,
and April descriptive performance is worse. It is therefore retained as a
shadow challenger rather than promoted to the official price.

See `research/TEAM_SPECIFIC_HOME_EFFECTS.md`.

## Information timing

The current game's points, turnovers, fouls and rebounds are postgame
information. They cannot predict the same game.

Each row is constructed in this order:

1. Read both teams' pregame states.
2. Construct home-minus-away features.
3. Store the prediction row.
4. Construct the outcome.
5. Update both team states for later games.

That ordering is the central leakage control.

## Data audit

The source contains:

- 1,230 unique games.
- 30 teams.
- 82 games per team.
- 1,134 October-March development games.
- 96 April games.
- 16 actual columns, despite the brief saying 14.
- No missing values, duplicate game IDs, tied final scores or record
  inconsistencies.

`game_id` is read as text so leading zeros survive downstream joins.

## Model equation

For matchup features \(x_1,x_2,x_3\):

\[
P(	ext{home win})=
\sigma\left(
eta_0+eta_1z_1+eta_2z_2+eta_3z_3

ight)
\]

where each \(z_j\) is standardized using training-only means and standard
deviations.

L2 regularization is essential because the features are highly related.

Observed variance-inflation factors:

| Feature | VIF |
|---|---:|
| `net_wins_diff` | 11.40 |
| `cumulative_margin_diff` | 18.01 |
| `recent_margin_evidence_diff` | 6.26 |

The correlations are a known limitation, not hidden evidence of independent
signals. L2 shrinkage stabilizes the joint price.

## Home-court advantage

Because the model standardizes its inputs, the centered fitted intercept and
the equal-strength home baseline are not identical.

- Standardized intercept log odds:
  **0.228745**
- Equal-strength home log odds:
  **0.221501**
- Equal-strength home-win probability:
  **55.515%**
- Equal-strength home odds multiplier:
  **1.2479x**

## Chronological development design

- October-December: coefficient training.
- January-February: hyperparameter and architecture validation.
- March: later **governance/robustness check**.
- October-March: final refit.
- April: requested forecast period.

March is deliberately not described as a pristine untouched test. It has been
used during model-governance review.

April outcomes are not used to select the submitted architecture.

## Primary measure

The primary metric is forward log loss.

Log loss is appropriate because the business output is a probability price.
It rewards informative probabilities and penalizes confidently wrong prices.

Brier score is secondary. AUC and 0.50-threshold accuracy are diagnostics, not
the pricing objective.

## March governance result

| Model | Log loss | Brier | AUC | Accuracy |
|---|---:|---:|---:|---:|
| Constant home-rate baseline | 0.680580 | 0.243727 | 0.500000 | 60.251% |
| Net-wins-only logistic | 0.520933 | 0.171274 | 0.823684 | 74.059% |
| Cumulative-margin-only logistic | 0.522780 | 0.172852 | 0.816118 | 75.732% |
| **Three-signal champion** | **0.509645** | **0.167618** | **0.823538** | **75.732%** |

## Limitations-driven model tests

| Candidate | Validation log loss | March log loss | Decision |
|---|---:|---:|---|
| Cumulative margin + recent margin | 0.626556 | 0.513661 | Tiny validation gain; later deterioration |
| Net wins + recent margin | 0.629909 | 0.505835 | Later gain; validation deterioration |
| Champion + opponent-adjusted SRS | 0.627278 | 0.511107 | Validation gain too small; March worse |
| Bayesian-shrunken recent margin | 0.628354 | 0.511124 | Worse in both periods |
| One-component latent PCA | 0.627744 | 0.509144 | Similar, less directly interpretable |

The nearest opponent-adjusted challenger improves validation log loss by only
**0.000251**
relative to the champion, then worsens March by
**0.001462**.

The paired date-block bootstrap interval for its combined
January-March log-loss difference includes zero. The apparent improvement is
not stable enough to earn deployment.

## Why the full three-feature model remains selected

The feature-ablation result is nuanced:

- Cumulative margin plus recent margin is fractionally better in
  January-February validation.
- Net wins plus recent margin is fractionally better in March.
- The full model is best when the January-March observations are pooled.
- All differences are small and bootstrap intervals overlap zero.

The selected model is therefore described as a stable, interpretable
specification—not as mathematically dominant on every slice.

## March calibration

March was not perfectly calibrated:

- Mean predicted home probability:
  **54.743%**
- Observed home-win rate:
  **60.251%**
- Difference:
  **5.508%**
- Calibration slope:
  **1.341**

The date-block bootstrap interval for the mean calibration gap is:

**[0.886%,
10.192%]**

A slope above one indicates that March outcomes were more separated than the
model's prices—the model was conservative during that period.

No recalibration layer is forced. Prospective Platt, beta and isotonic
mappings all failed to improve next-period scores. A one-month calibration
deviation is monitored rather than overfit.

## April model uncertainty

`outputs/april_model_uncertainty.csv` contains date-block bootstrap intervals
around the fitted coefficient model.

Across the 96 games, the median 90% interval width is
**5.650%**.

These intervals represent parameter-estimation uncertainty conditional on:

- The supplied feature definitions.
- The schedule.
- The model family.

They do not include injury, lineup, market, data-source or structural model
uncertainty.

## Fair odds

The official file includes zero-margin decimal odds:

\[
	ext{fair home odds}=1/P(	ext{home win})
\]

\[
	ext{fair away odds}=1/(1-P(	ext{home win}))
\]

These are fundamental fair prices, not offered customer prices.

A production trading layer would add:

- Player availability and expected minutes.
- Lineup and news adjustments.
- Market-price blending.
- Overround.
- Liability and limit controls.
- Trader overrides.

## Main limitations

1. One season only; no external-season validation.
2. No injuries, player-level impact or expected lineups.
3. Cumulative margin is not pace-adjusted.
4. The three signals are highly correlated.
5. March showed conservative home pricing.
6. Evidence-weighted EWMA is unusual, although explicit challengers did not
   improve it reliably.
7. Opponent-adjusted SRS did not earn inclusion, but remains a production
   research path.
8. April has already been examined descriptively and is not represented as a
   pristine untouched test.

## Run the champion

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python nba_win_probability.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs
```

## Run enhanced governance

```bash
python enhanced_governance.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs \
  --figure-dir figures
```

## Run optional machine-learning challengers

```bash
python -m pip install -r requirements-challengers.txt

python challenger_analysis.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir outputs
```

## Validate the package

```bash
python validate_submission.py \
  --root . \
  --data /path/to/nba-win-probability-data.csv
```

## Repository contents

- `nba_win_probability.py`: official champion and April prices.
- `enhanced_governance.py`: limitations-driven SRS, shrinkage, PCA,
  calibration, stability and uncertainty analysis.
- `challenger_analysis.py`: nonlinear models, ensembles, calibration and SHAP.
- `ablation_and_timing.py`: feature ablation and local timing.
- `MODEL_CARD.md`: intended use, evidence and restrictions.
- `LIMITATIONS_AND_ROADMAP.md`: explicit weaknesses and production research.
- `SUMMARY.md`: recruiter-ready brief.
- `outputs/`: predictions and audit artifacts.
- `figures/`: focused diagnostics.


## Repository research history

The commit history and tags document the model's progression:

- Leakage-safe three-signal champion.
- Machine-learning and calibration challengers.
- Opponent-adjusted, shrinkage and latent-factor governance.
- Team-specific home-court-effect research.
- Final recruiter-facing documentation and validation.

See `MODEL_EVOLUTION.md`.

## Repository layout

```text
.
├── nba_win_probability.py
├── enhanced_governance.py
├── challenger_analysis.py
├── ablation_and_timing.py
├── validate_submission.py
├── research/
│   ├── team_specific_home_effects.py
│   ├── TEAM_SPECIFIC_HOME_EFFECTS.md
│   ├── outputs/
│   └── figures/
├── tests/
├── outputs/
├── figures/
├── MODEL_CARD.md
├── LIMITATIONS_AND_ROADMAP.md
├── MODEL_EVOLUTION.md
└── REPRODUCIBILITY.md
```
