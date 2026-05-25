"""
Lag compensation module for server-side hit detection.

This module provides systems for:
- Server-side world state rewind
- Historical hitbox tracking
- Client view time calculation
"""

from engine.networking.lag_compensation.rewind_manager import (
    HistoryFrame,
    RewindManager,
    WorldState,
)
from engine.networking.lag_compensation.hitbox_history import (
    HitboxSnapshot,
    HitboxHistory,
    Bounds,
)
from engine.networking.lag_compensation.view_time import (
    ViewTimeCalculator,
    calculate_client_view_time,
)

__all__ = [
    # Rewind manager
    "HistoryFrame",
    "RewindManager",
    "WorldState",
    # Hitbox history
    "HitboxSnapshot",
    "HitboxHistory",
    "Bounds",
    # View time
    "ViewTimeCalculator",
    "calculate_client_view_time",
]
