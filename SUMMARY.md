# Brief Submission Summary

## Objective

Estimate each April game's home-team win probability using only information
available through March.

## Selected model

A strongly regularized logistic paired-comparison model using:

- Net-wins differential.
- Cumulative point-margin differential.
- Evidence-weighted recent point-margin differential.

Selected hyperparameters:

- 12-game recent-form half-life.
- Logistic `C = 0.0075`.

## Validation and governance

- October-December: coefficient training.
- January-February: model and hyperparameter validation.
- March: later governance/robustness check.
- October-March: final refit.
- April: strict March 31 frozen probabilities.

March is not represented as a pristine untouched test because it has informed
later governance discussions.

## March performance

- Log loss: **0.509645**
- Brier score: **0.167618**
- ROC AUC: **0.823538**
- Accuracy: **75.732%**

Constant home-rate baseline log loss: **0.680580**.

## Enhanced limitation testing

The final review explicitly tested:

- Opponent-adjusted ridge team strength.
- Pure and Bayesian-shrunken EWMA recent form.
- PCA latent-strength models.
- Every two-feature subset.
- Rich box-score and schedule features.
- Tree models, residual boosting and probability ensembles.
- Platt, beta and isotonic calibration.

No challenger produced a material and temporally stable proper-score gain.

## Calibration

March mean probability was **54.743%**,
versus an observed home-win rate of
**60.251%**.

This limitation is disclosed. A post-hoc mapping is not added because
prospective calibration tests worsened later scores.

## Sportsbook interpretation

The output is a fundamental zero-margin fair price. A production price would
also incorporate injuries, expected lineups, player-level information, market
prices, overround, liability, limits and trader judgment.

## Final claim

This is not claimed to be the universally optimal NBA model. It is the most
defensible submission for the supplied data and business constraints after
the serious challenger families tested failed to earn additional complexity.


## Team-specific interpretation

The official probability is already specific to each home-away matchup.

A separate shrunk team-specific home-court challenger was also tested. It
improved validation log loss slightly, but its bootstrap interval included
zero, March was effectively tied, and April descriptive performance was
worse. The global home baseline therefore remains the official specification.
