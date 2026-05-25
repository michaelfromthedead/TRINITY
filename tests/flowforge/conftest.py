"""
Conftest for flowforge test suite.

The root conftest clears all metaclass and foundation registries before each
test. This conftest re-registers the Trinity graph node types (defined at
import time via _GraphNodeMeta) so that foundation.registry and
EngineMeta registry tests pass.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def register_trinity_graph_types():
    """Re-register Trinity graph types after root conftest clears registries."""
    from trinity.metaclasses.engine_meta import EngineMeta

    from flowforge_backend.ast_parser.trinity_nodes import (
        TrinityComponentData,
        TrinityEventData,
        TrinityFieldData,
        TrinityGraphEdge,
        TrinityGraphNode,
        TrinityMethodData,
        TrinityNodeGraph,
        TrinityNodePosition,
        TrinityParameterData,
        TrinityResourceData,
        TrinitySourceLocation,
        TrinitySystemData,
        register_all_trinity_graph_types,
    )

    # Re-register with foundation.registry
    register_all_trinity_graph_types()

    # Re-register with EngineMeta's type registry
    types = [
        TrinityNodePosition,
        TrinitySourceLocation,
        TrinityFieldData,
        TrinityParameterData,
        TrinityMethodData,
        TrinityComponentData,
        TrinitySystemData,
        TrinityResourceData,
        TrinityEventData,
        TrinityGraphNode,
        TrinityGraphEdge,
        TrinityNodeGraph,
    ]
    with EngineMeta._lock:
        for cls in types:
            qualified_name = f"{cls.__module__}.{cls.__name__}"
            if qualified_name not in EngineMeta._all_engine_types:
                EngineMeta._all_engine_types[qualified_name] = cls

    yield  # Test runs here
