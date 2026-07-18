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

## Stage 6 — Runtime and CI reliability hardening

The public repository was converted into an installable Python project with:

- An explicit Python 3.11-3.13 compatibility contract.
- Python 3.12.13 as the recommended local interpreter.
- Fail-fast rejection of unsupported Python 3.14 before compiled numerical
  extensions load.
- Centralized dependency metadata in `pyproject.toml`.
- Warning-clean tests across Python 3.11, 3.12, and 3.13 in GitHub Actions.
- Current Node 24-compatible GitHub actions.
- Repository privacy, credential, cache, artifact, and fair-odds checks.
- A canonical ten-decimal probability-and-odds serialization contract.
- A one-command macOS bootstrap and quality gate.

These changes improve operational reliability without changing the selected
model architecture or its underlying probabilities.

## Stage 7 — Reviewer navigation and presentation clarity

The repository information architecture was simplified without changing the
model:

- `README.md` became the main decision page.
- `SUMMARY.md` remained the brief recruiter deliverable.
- Supporting governance documents moved into `docs/`.
- `docs/REVIEWER_GUIDE.md` now provides 2-, 10-, and 30-minute review paths.
- Official, governance, and shadow-research artifacts are labelled separately.
- Documentation tests reject malformed Markdown control characters and missing
  navigation files.

This stage improves reviewability and screen-share execution while preserving
the validated probability engine.

## Stage 8 — Late-season ensemble promotion

The final assignment target is April, when team-strength states are mature.

A fixed ensemble was built across the complete predeclared half-life and L2
grid. It uses the same three sports signals as the single champion.

The ensemble:

- Improved January-February validation log loss.
- Preserved the direction in March governance.
- Improved the strict April descriptive log loss and Brier score.
- Reduced dependence on one narrowly selected grid point.
- Added negligible runtime cost.

The best single component remains the benchmark and interpretation anchor.

The promotion is not described as statistically decisive. It is a
target-specific model-governance decision based on pre-April evidence,
late-season relevance, and specification-risk reduction.

## Final decision rule

A change enters the official price only when it:

1. Improves a target-relevant forward proper score before deployment.
2. Preserves the pregame information timestamp.
3. Remains reproducible and operationally inexpensive.
4. Retains a defensible interpretation.
5. Does not reverse in the later governance period.

The ensemble clears that standard with a modest pre-April improvement and
negligible additional operational risk. The gain is not described as
statistically decisive, so the validation-best single model remains the
benchmark and fallback.

### Cross-platform reproduction contract

The final audit distinguishes model determinism from renderer determinism:

- Official April IDs, probabilities, odds, fingerprints, and selection
  metadata must reproduce byte for byte.
- Platform-sensitive floating-point summaries must match within a strict
  absolute tolerance.
- PNG figures must remain valid, nonblank, and dimensionally consistent while
  their underlying tables satisfy the numeric reproduction contract.

This prevents harmless Apple-Silicon versus Linux rendering and BLAS
differences from being misclassified as model failures.
