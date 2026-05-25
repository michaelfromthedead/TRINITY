"""
Display widgets for the UI system.

This module provides display-only widgets for presenting information:
    - Label: Single-line text display with icon support
    - ProgressBar: Value visualization with multiple styles
    - Icon: Icon/sprite display with tinting and animation
    - Badge: Notification indicators and counts

All widgets follow the Trinity Pattern with TrackedDescriptor fields
for automatic change detection and re-rendering.
"""

from __future__ import annotations

from engine.ui.widgets.display.label import Label
from engine.ui.widgets.display.progress_bar import (
    ProgressBar,
    ProgressBarMode,
    ProgressBarStyle,
)
from engine.ui.widgets.display.icon import (
    Icon,
    IconSize,
    IconAnimation,
    IconFlip,
    IconAtlasEntry,
    IconAtlasManager,
)
from engine.ui.widgets.display.badge import (
    Badge,
    BadgeMode,
    BadgePosition,
    BadgeVariant,
    BadgeStyle,
)

__all__ = [
    # Label
    "Label",
    # Progress Bar
    "ProgressBar",
    "ProgressBarMode",
    "ProgressBarStyle",
    # Icon
    "Icon",
    "IconSize",
    "IconAnimation",
    "IconFlip",
    "IconAtlasEntry",
    "IconAtlasManager",
    # Badge
    "Badge",
    "BadgeMode",
    "BadgePosition",
    "BadgeVariant",
    "BadgeStyle",
]
