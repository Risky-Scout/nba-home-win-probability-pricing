"""Exit clearly before unsupported Python loads compiled dependencies."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_runtime import require_supported_python  # noqa: E402


if __name__ == "__main__":
    require_supported_python()
    print("PASS: supported Python interpreter.")
