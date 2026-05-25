"""
Local conftest for physics simulation tests.

Overrides the global autouse metaclass-clearing fixture to prevent
import errors, since physics tests do not use Trinity metaclasses.
"""
import pytest


@pytest.fixture(autouse=True)
def clear_all_registries():
    """No-op: physics tests do not use Trinity metaclass registries."""
    yield
