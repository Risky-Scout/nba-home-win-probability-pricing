# Research Archive

This directory contains **historical and shadow challengers**. Nothing here
silently replaces the official April price.

## Current official model

The official model lives at the repository root:

- `../nba_win_probability.py`
- `../outputs/april_predictions.csv`
- Fixed equal-weight 40-component logistic ensemble

## Historical single-model governance

- `single_model_governance.py`
- `single_model_outputs/`
- `single_model_figures/`

These files preserve the governance work conducted before the ensemble was
promoted. References to a “champion” inside historical artifact names refer to
the then-current single benchmark, not the current official model.

## Team-specific venue challenger

- `team_specific_home_effects.py`
- `TEAM_SPECIFIC_HOME_EFFECTS.md`
- `outputs/`
- `figures/`

This study tests whether permanent team-specific venue deviations improve the
global home baseline. It remains a non-promoted shadow challenger.

## Reproduction

The source CSV is intentionally excluded from version control.

```bash
python research/team_specific_home_effects.py \
  --data /path/to/nba-win-probability-data.csv \
  --output-dir research/outputs \
  --figure-dir research/figures
```
