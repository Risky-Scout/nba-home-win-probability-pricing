# Reproducibility

## Source data

The supplied CSV is intentionally excluded from version control.

Place it outside or inside the repository locally, then pass its path through
`--data`.

## Core environment

Supported interpreters are Python 3.11-3.13. Python 3.12.13 is the
recommended local interpreter.

```bash
brew install python@3.12
rm -rf .venv
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate

python scripts/check_python.py
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip check
```

On macOS, the equivalent one-command setup is:

```bash
bash scripts/bootstrap_macos.sh
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
python -m pytest -q
```

## Submission validator

```bash
python validate_submission.py   --root .   --data /path/to/nba-win-probability-data.csv
```

The validator checks identifiers, probability bounds, fair-odds identities,
home advantage, model selection, uncertainty outputs, calibration diagnostics
and collinearity artifacts.


## Complete data-free quality gate

```bash
bash scripts/run_quality_checks.sh
```

This command verifies:

- Supported Python.
- Installed dependency consistency.
- Python compilation.
- Warning-clean unit tests.
- Public-repository privacy policy.
- Committed CSV, JSON, PNG, probability, odds, and governance artifacts.
- Git whitespace integrity.
