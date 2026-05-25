"""
Trinity Lint - import-time validation for Trinity classes.
"""
from __future__ import annotations

import warnings
from trinity.decorators.ops import decompose, validate_steps


def lint(cls: type) -> list[str]:
    """
    Validate a single Trinity class against composition rules.

    Args:
        cls: A Trinity class to validate.

    Returns:
        List of error strings (empty if valid).
    """
    steps = decompose(cls)
    result = validate_steps(steps)
    return result.get("errors", [])


_original_engine_new = None
_lint_enabled = False


def install_lint_hook() -> None:
    """
    Install an import-time lint hook on EngineMeta.__new__.

    After calling this, every new Trinity class created will be
    automatically validated. Validation errors are emitted as warnings.
    """
    global _original_engine_new, _lint_enabled

    if _lint_enabled:
        return  # Already installed

    from trinity.metaclasses.engine_meta import EngineMeta

    _original_engine_new = EngineMeta.__new__

    def _lint_new(mcs, name, bases, namespace, **kwargs):
        cls = _original_engine_new(mcs, name, bases, namespace, **kwargs)
        errors = lint(cls)
        if errors:
            for error in errors:
                warnings.warn(
                    f"Trinity lint [{name}]: {error}",
                    UserWarning,
                    stacklevel=2,
                )
        return cls

    EngineMeta.__new__ = _lint_new
    _lint_enabled = True


def uninstall_lint_hook() -> None:
    """Remove the lint hook from EngineMeta.__new__."""
    global _original_engine_new, _lint_enabled

    if not _lint_enabled:
        return

    from trinity.metaclasses.engine_meta import EngineMeta

    if _original_engine_new is not None:
        EngineMeta.__new__ = _original_engine_new
        _original_engine_new = None
    _lint_enabled = False
