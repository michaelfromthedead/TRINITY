"""
Local conftest for postprocess tests.

Overrides the project-wide conftest's autouse fixture that imports trinity
(which has a pre-existing import error unrelated to postprocess/bloom).
"""
import pytest


@pytest.fixture(autouse=True)
def clear_all_registries():
    """No-op override: postprocess tests don't touch trinity metaclasses."""
    pass
