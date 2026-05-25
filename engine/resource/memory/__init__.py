"""Resource memory subsystem: pools, budgets, eviction, and residency."""

from engine.resource.constants import (
    DEFAULT_AUDIO_BUDGET,
    DEFAULT_MESH_BUDGET,
    DEFAULT_POOL_CAPACITY,
    DEFAULT_TEXTURE_BUDGET,
)
from engine.resource.memory.asset_pool import (
    AssetPool,
    PoolSlot,
)
from engine.resource.memory.budget_manager import (
    AssetCategory,
    BudgetEntry,
    BudgetManager,
)
from engine.resource.memory.eviction import (
    EvictionCandidate,
    EvictionManager,
    EvictionPolicy,
    LFUEviction,
    LRUEviction,
    PriorityEviction,
    SizeEviction,
)
from engine.resource.memory.residency_manager import (
    ResidencyInfo,
    ResidencyManager,
    ResidencyState,
)

__all__ = [
    "AssetPool",
    "DEFAULT_POOL_CAPACITY",
    "PoolSlot",
    "AssetCategory",
    "BudgetEntry",
    "BudgetManager",
    "DEFAULT_AUDIO_BUDGET",
    "DEFAULT_MESH_BUDGET",
    "DEFAULT_TEXTURE_BUDGET",
    "EvictionCandidate",
    "EvictionManager",
    "EvictionPolicy",
    "LFUEviction",
    "LRUEviction",
    "PriorityEviction",
    "SizeEviction",
    "ResidencyInfo",
    "ResidencyManager",
    "ResidencyState",
]
