# Contributing and Local Quality Checks

## Supported Python

Use Python 3.11, 3.12, or 3.13. Python 3.12.13 is recommended.

Python 3.14 is not supported by this locked numerical environment. The
repository fails before importing compiled numerical extensions when an
unsupported interpreter is detected.

## macOS setup

```bash
bash scripts/bootstrap_macos.sh
```

## Manual setup

```bash
brew install python@3.12
rm -rf .venv
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## Required checks before committing

```bash
bash scripts/run_quality_checks.sh
```

The command runs compilation, warning-clean tests, repository-policy checks,
committed-artifact validation, dependency consistency, and whitespace checks.

## Source data

Do not commit `nba-win-probability-data.csv`. Pass its local path to model and
validator commands through `--data`.

## Model-governance rule

A challenger enters the official price only when pre-deployment evidence supports an improvement relevant to the target period, the information timestamp is preserved, runtime remains acceptable, and the uncertainty and trade-offs are disclosed. Small gains must not be described as statistically conclusive.
