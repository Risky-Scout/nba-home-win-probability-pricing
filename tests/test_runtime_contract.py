"""Tests for the explicit Python compatibility contract."""

from __future__ import annotations

import pytest

from project_runtime import (
    UnsupportedPythonError,
    require_supported_python,
)


@pytest.mark.parametrize(
    "version",
    [
        (3, 11),
        (3, 12),
        (3, 13),
    ],
)
def test_supported_python_versions_are_accepted(
    version: tuple[int, int],
) -> None:
    """The documented Python matrix must remain accepted."""

    require_supported_python(version)


@pytest.mark.parametrize(
    "version",
    [
        (3, 10),
        (3, 14),
        (4, 0),
    ],
)
def test_unsupported_python_versions_fail_cleanly(
    version: tuple[int, int],
) -> None:
    """Unsupported interpreters must fail before numerical imports."""

    with pytest.raises(
        UnsupportedPythonError,
        match="supports Python 3.11-3.13",
    ):
        require_supported_python(version)
