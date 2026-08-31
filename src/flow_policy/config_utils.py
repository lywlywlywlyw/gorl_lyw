"""Validation helpers for explicitly populated model configurations."""

from dataclasses import fields, replace
from typing import Any, Collection


def require_config_values(
    config: Any,
    *,
    allow_none: Collection[str] = (),
) -> None:
    """Reject model configs that still contain an unspecified value."""
    allowed = frozenset(allow_none)
    missing = [
        field.name
        for field in fields(config)
        if field.name not in allowed and getattr(config, field.name) is None
    ]
    if missing:
        raise ValueError(
            f"{type(config).__name__} has unspecified parameters: "
            + ", ".join(missing)
        )


def fill_unspecified_config_values(config: Any, **values: Any) -> Any:
    """Fill legacy checkpoint fields from the active project configuration."""
    updates = {
        name: value
        for name, value in values.items()
        if getattr(config, name, None) is None
    }
    return replace(config, **updates) if updates else config
