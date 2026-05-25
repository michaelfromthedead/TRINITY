"""
UI Configuration constants for the AI Game Engine.

Centralizes magic numbers, default values, and configurable settings
for the UI framework to improve maintainability and allow customization.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewportDefaults:
    """Default viewport configuration."""

    WIDTH: int = 1920
    HEIGHT: int = 1080
    DPI_SCALE: float = 1.0


@dataclass(frozen=True)
class FocusConfig:
    """Focus management configuration."""

    MAX_HISTORY_SIZE: int = 10
    """Maximum number of focus history entries to retain."""


@dataclass(frozen=True)
class LayoutDefaults:
    """Default layout configuration values."""

    DEFAULT_SPACING: float = 0.0
    MIN_SIZE: float = 0.0
    MAX_SIZE: float = float("inf")


# Singleton instances for easy access
VIEWPORT = ViewportDefaults()
FOCUS = FocusConfig()
LAYOUT = LayoutDefaults()


__all__ = [
    "ViewportDefaults",
    "FocusConfig",
    "LayoutDefaults",
    "VIEWPORT",
    "FOCUS",
    "LAYOUT",
]
