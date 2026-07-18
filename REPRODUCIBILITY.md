# Reproducibility

## Source data

The supplied CSV is intentionally excluded from version control.

Place it outside or inside the repository locally, then pass its path through
`--data`.

## Core environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Official model

```bash
python nba_win_probability.py   --data /path/to/nba-win-probability-data.csv   --output-dir outputs
```

## Enhanced governance

```bash
python enhanced_governance.py   --data /path/to/nba-win-probability-data.csv   --output-dir outputs   --figure-dir figures
```

## Team-specific venue research

```bash
python research/team_specific_home_effects.py   --data /path/to/nba-win-probability-data.csv   --output-dir research/outputs   --figure-dir research/figures
```

## Optional machine-learning challengers

```bash
python -m pip install -r requirements-challengers.txt

python challenger_analysis.py   --data /path/to/nba-win-probability-data.csv   --output-dir outputs
```

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Submission validator

```bash
python validate_submission.py   --root .   --data /path/to/nba-win-probability-data.csv
```

The validator checks identifiers, probability bounds, fair-odds identities,
home advantage, model selection, uncertainty outputs, calibration diagnostics
and collinearity artifacts.
