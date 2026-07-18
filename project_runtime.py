"""Runtime compatibility contract for every executable module."""

from __future__ import annotations

import sys
from collections.abc import Sequence

MINIMUM_PYTHON = (3, 11)
MAXIMUM_PYTHON_EXCLUSIVE = (3, 14)
RECOMMENDED_PYTHON = "3.12"


class UnsupportedPythonError(RuntimeError):
    """Raised before compiled numerical libraries load on unsupported Python."""


def require_supported_python(
    version_info: Sequence[int] | None = None,
) -> None:
    """Fail clearly unless Python 3.11, 3.12, or 3.13 is running."""

    current = tuple(
        (version_info if version_info is not None else sys.version_info)[:2]
    )
    if not (
        MINIMUM_PYTHON
        <= current
        < MAXIMUM_PYTHON_EXCLUSIVE
    ):
        detected = ".".join(str(part) for part in current)
        raise UnsupportedPythonError(
            "Unsupported Python "
            f"{detected}. This repository supports Python 3.11-3.13; "
            f"Python {RECOMMENDED_PYTHON} is recommended. "
            "Delete the existing .venv and recreate it with a supported "
            "interpreter before installing dependencies."
        )
