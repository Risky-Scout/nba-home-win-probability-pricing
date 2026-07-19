# Team-Specific Home-Court Effects

## Status

This is a **historical shadow-challenger study**. It was developed when the
validation-best single logistic model was the active benchmark. The current
official price is the fixed 40-component ensemble in the repository root.

The study remains useful because it answers a separate modeling question:
should each NBA team receive its own venue-effect parameter?

## The interpretation question

The phrase **home-team win probability** in the assignment means:

> For each April matchup, estimate the probability that the team listed as
> `home` wins that game.

The current official ensemble already does this. Its probability changes with
the specific home team, away team, and their pregame strength states.

That is different from estimating a separate **home-court advantage parameter
for each NBA team**.

The official ensemble's components share one structural home baseline:

- Mean equal-strength home probability: approximately 55.6%.

This research asks whether team-specific venue effects should replace or
supplement that global baseline.

## Challenger construction

For every game, using only earlier results:

1. Estimate the league home-win rate with a strong 0.500 prior.
2. Estimate the home team's home record with shrinkage toward the league home
   rate.
3. Estimate the away team's road record with shrinkage toward the league road
   rate.
4. Compare each venue-specific rate with that team's overall rate.
5. Form:

$$
\text{venue deviation}
=
(\text{home team's home deviation})
-
(\text{away team's road deviation})
$$

The validation-selected challenger uses:

- Team prior: **10 pseudo-games**
- Global home-rate prior: **500 pseudo-games**
- Logistic `C`: **0.010**
- Features: single-benchmark core + global home trend + team-specific venue
  deviation

The large global prior deliberately prevents a temporary league home streak
from creating a large baseline shift.

## Historical benchmark results

The comparison below uses the validation-best single component that was active
when this study was conducted.

| Model | Validation log loss | Validation Brier | March log loss | March Brier |
|---|---:|---:|---:|---:|
| Validation-best single benchmark | 0.627529 | 0.217481 | 0.509645 | **0.167618** |
| Team-specific venue challenger | **0.624285** | **0.216346** | **0.509434** | 0.167916 |

The challenger improves validation log loss by:

$$
0.003243
$$

Its March log-loss improvement is only:

$$
0.000211
$$

March accuracy falls from **75.73%** to **74.48%**.

## Bootstrap uncertainty

| Period | Observed challenger − benchmark log loss | 95% interval | Probability challenger is better |
|---|---:|---:|---:|
| January–February validation | -0.003243 | [-0.010854, 0.004250] | 80.0% |
| March governance | -0.000211 | [-0.005831, 0.005714] | 52.9% |

Both intervals include zero. The March comparison is effectively a tie.

## April descriptive audit

The team-specific challenger uses a strict March 31 information cutoff.

Its April descriptive results are:

- Log loss: **0.471213**
- Brier score: **0.151636**
- ROC AUC: **0.868196**
- Accuracy: **79.17%**

For context:

- Current official ensemble April log loss: **0.467607**
- Validation-best single benchmark April log loss: **0.468596**

The team-specific challenger is worse descriptively than both. April outcomes
are not used to select its hyperparameters.

## Decision

**Do not promote the team-specific venue challenger.**

Reasons:

- The official ensemble already produces matchup-specific probabilities.
- The venue extension's validation gain is small and statistically uncertain.
- March log loss is effectively tied.
- March Brier score and accuracy do not improve.
- April descriptive performance is worse than the official ensemble.
- Each team has only about 41 home games in one season, so permanent
  team-specific venue deviations are difficult to separate from noise.

The correct sportsbook architecture is:

- Retain the global home baseline in the official ensemble.
- Keep team-specific venue effects as a shadow challenger.
- Revisit them with multiple seasons, arena and travel context, injuries, and
  player-level availability.

## Reproduce

```bash
python research/team_specific_home_effects.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir research/outputs \
  --figure-dir research/figures
```
