# Model Evolution

This repository preserves the submission's actual research path through
versioned commits and tags.

## Stage 1 — Leakage-safe probability baseline

The project first established the correct information timestamp:

- Same-game box scores cannot predict their own result.
- Team states update only after the current prediction row.
- Dates are split chronologically.
- The target is probability quality, not winner accuracy.

## Stage 2 — Three-signal regularized champion

The core model was reduced to three interpretable home-minus-away signals:

- Net wins.
- Cumulative point margin.
- Evidence-weighted recent point margin.

Strong L2 regularization controls their correlation.

## Stage 3 — Machine-learning and calibration challengers

The model was compared with:

- XGBoost.
- Residual XGBoost.
- CatBoost.
- Random forest.
- ExtraTrees.
- Probability blends.
- Platt, beta and isotonic calibration.
- Rich lagged box-score and schedule features.

Added complexity did not produce a stable forward proper-score gain.

## Stage 4 — Limitations-driven governance

The review was extended to:

- Opponent-adjusted ridge SRS.
- Pure EWMA.
- Bayesian-shrunken EWMA.
- PCA latent factors.
- Feature-subset ablations.
- Calibration slope and uncertainty.
- Coefficient stability.
- April parameter-uncertainty intervals.

The champion remained selected.

## Stage 5 — Team-specific home-court research

The assignment's “home-team win probability” wording was examined carefully.

The official output is already team- and matchup-specific. A separate question
is whether each team needs its own home-court deviation.

A hierarchically shrunk venue challenger was implemented. It improved
validation log loss slightly, but the bootstrap interval included zero,
March was effectively tied, and April descriptive performance was worse.

It remains a shadow challenger rather than the production champion.

## Final decision rule

A change enters the official price only when it produces:

1. A proper-score improvement.
2. Material magnitude.
3. Temporal stability.
4. Reproducibility.
5. Explainability.
6. A clearly defined pregame timestamp.

No challenger has cleared all six requirements.
