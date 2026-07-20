"""JSON-safety helpers for report export (dependency-free, no argus imports → no cycle).

The report types carry computed ``@property`` numbers (recall, fire-on-clean, Wilson CIs,
balanced-acc, macro-F1) that ``dataclasses.asdict`` would drop, and those numbers are ``nan``
when a denominator is zero. Bare ``nan`` is not valid JSON (``json.dumps`` emits the token
``NaN``, which strict parsers reject), so ``to_dict`` maps it to ``None`` via ``nan_to_none``.
"""

from __future__ import annotations

from typing import Any


def nan_to_none(x: Any) -> Any:
    """Map a float NaN to None (valid JSON null); pass everything else through unchanged."""
    return None if isinstance(x, float) and x != x else x


def ci_or_none(ci: tuple[float, float], n: int) -> list[float] | None:
    """A [lo, hi] CI as a JSON list, or None when n == 0 (the CI is undefined, not (0,0))."""
    return None if n == 0 else [ci[0], ci[1]]
