"""
Pytest configuration and fixtures for Trinity Pattern tests.

Key concerns:
1. Registry isolation - metaclasses maintain global registries that persist
   between tests. We must clear them before/after each test.
2. Dynamic class creation - classes defined at module level trigger metaclass
   code at import time. Tests should create classes dynamically.
"""

import pytest


@pytest.fixture(autouse=True)
def clear_all_registries():
    """
    Clear all metaclass registries before and after each test.

    This is critical because:
    - ComponentMeta, SystemMeta, etc. maintain class-level registries
    - IDs are assigned incrementally and persist across tests
    - Without clearing, test order affects test results
    - Foundation registry also tracks types and must be cleared
    """
    # Import all metaclasses
    from trinity.metaclasses import (
        AssetMeta,
        ComponentMeta,
        EngineMeta,
        EventMeta,
        ProtocolMeta,
        ResourceMeta,
        StateMeta,
        SystemMeta,
    )
    # Import Foundation registry
    from foundation import registry

    # Clear before test
    ComponentMeta.clear_registry()
    SystemMeta.clear_registry()
    ResourceMeta.clear_registry()
    EventMeta.clear_registry()
    AssetMeta.clear_registry()
    ProtocolMeta.clear_registry()
    StateMeta.clear_registry()
    EngineMeta.clear_registry()
    registry.clear()  # Clear Foundation registry

    yield  # Run the test

    # Clear after test (defensive)
    ComponentMeta.clear_registry()
    SystemMeta.clear_registry()
    ResourceMeta.clear_registry()
    EventMeta.clear_registry()
    AssetMeta.clear_registry()
    ProtocolMeta.clear_registry()
    StateMeta.clear_registry()
    EngineMeta.clear_registry()
    registry.clear()  # Clear Foundation registry


@pytest.fixture
def component_meta():
    """Provide ComponentMeta for tests that need direct access."""
    from trinity.metaclasses import ComponentMeta

    return ComponentMeta


@pytest.fixture
def system_meta():
    """Provide SystemMeta for tests that need direct access."""
    from trinity.metaclasses import SystemMeta

    return SystemMeta


@pytest.fixture
def resource_meta():
    """Provide ResourceMeta for tests that need direct access."""
    from trinity.metaclasses import ResourceMeta

    return ResourceMeta
