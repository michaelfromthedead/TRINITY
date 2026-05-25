"""
Spacer Widget - Layout spacing utility.

Provides a flexible spacer widget for creating gaps and
distributing space in layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class SpacerMode(Enum):
    """Spacer sizing modes."""
    FIXED = auto()      # Fixed size in pixels
    FLEXIBLE = auto()   # Flexible size (fills available space)
    MINIMUM = auto()    # Minimum size with potential to grow


class Spacer:
    """Spacer widget for layout spacing.

    A non-visible widget that creates space in layouts.
    Can be fixed size or flexible to fill available space.

    Attributes:
        width: Spacer width
        height: Spacer height
        mode: Sizing mode (fixed, flexible, minimum)
        flex: Flex factor for flexible mode
    """

    __slots__ = (
        '_width', '_height', '_mode', '_flex',
        '_min_width', '_min_height',
        '_max_width', '_max_height',
    )

    def __init__(
        self,
        width: float = 0.0,
        height: float = 0.0,
        mode: SpacerMode = SpacerMode.FIXED,
        flex: float = 1.0,
        min_width: Optional[float] = None,
        min_height: Optional[float] = None,
        max_width: Optional[float] = None,
        max_height: Optional[float] = None,
    ) -> None:
        """Initialize the spacer.

        Args:
            width: Initial/preferred width
            height: Initial/preferred height
            mode: Sizing mode
            flex: Flex factor for flexible mode
            min_width: Minimum width constraint
            min_height: Minimum height constraint
            max_width: Maximum width constraint
            max_height: Maximum height constraint
        """
        self._width = max(0.0, width)
        self._height = max(0.0, height)
        self._mode = mode
        self._flex = max(0.0, flex)
        self._min_width = min_width
        self._min_height = min_height
        self._max_width = max_width
        self._max_height = max_height

    @classmethod
    def fixed(cls, width: float = 0.0, height: float = 0.0) -> "Spacer":
        """Create a fixed-size spacer.

        Args:
            width: Fixed width
            height: Fixed height

        Returns:
            Fixed spacer instance
        """
        return cls(width=width, height=height, mode=SpacerMode.FIXED)

    @classmethod
    def horizontal(cls, width: float) -> "Spacer":
        """Create a horizontal spacer (fixed width, zero height).

        Args:
            width: Spacer width

        Returns:
            Horizontal spacer instance
        """
        return cls(width=width, height=0.0, mode=SpacerMode.FIXED)

    @classmethod
    def vertical(cls, height: float) -> "Spacer":
        """Create a vertical spacer (zero width, fixed height).

        Args:
            height: Spacer height

        Returns:
            Vertical spacer instance
        """
        return cls(width=0.0, height=height, mode=SpacerMode.FIXED)

    @classmethod
    def flexible(cls, flex: float = 1.0) -> "Spacer":
        """Create a flexible spacer that fills available space.

        Args:
            flex: Flex factor (higher = more space)

        Returns:
            Flexible spacer instance
        """
        return cls(mode=SpacerMode.FLEXIBLE, flex=flex)

    @classmethod
    def expand(cls) -> "Spacer":
        """Create an expanding spacer (shorthand for flexible(1.0))."""
        return cls.flexible(1.0)

    @property
    def width(self) -> float:
        """Get spacer width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set spacer width."""
        self._width = max(0.0, value)
        if self._max_width is not None:
            self._width = min(self._width, self._max_width)
        if self._min_width is not None:
            self._width = max(self._width, self._min_width)

    @property
    def height(self) -> float:
        """Get spacer height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set spacer height."""
        self._height = max(0.0, value)
        if self._max_height is not None:
            self._height = min(self._height, self._max_height)
        if self._min_height is not None:
            self._height = max(self._height, self._min_height)

    @property
    def mode(self) -> SpacerMode:
        """Get sizing mode."""
        return self._mode

    @mode.setter
    def mode(self, value: SpacerMode) -> None:
        """Set sizing mode."""
        self._mode = value

    @property
    def flex(self) -> float:
        """Get flex factor."""
        return self._flex

    @flex.setter
    def flex(self, value: float) -> None:
        """Set flex factor."""
        self._flex = max(0.0, value)

    @property
    def min_width(self) -> Optional[float]:
        """Get minimum width."""
        return self._min_width

    @property
    def min_height(self) -> Optional[float]:
        """Get minimum height."""
        return self._min_height

    @property
    def max_width(self) -> Optional[float]:
        """Get maximum width."""
        return self._max_width

    @property
    def max_height(self) -> Optional[float]:
        """Get maximum height."""
        return self._max_height

    @property
    def is_flexible(self) -> bool:
        """Check if spacer is flexible."""
        return self._mode == SpacerMode.FLEXIBLE

    @property
    def is_fixed(self) -> bool:
        """Check if spacer is fixed size."""
        return self._mode == SpacerMode.FIXED

    @property
    def bounds(self) -> Tuple[float, float]:
        """Get spacer bounds (width, height)."""
        return (self._width, self._height)

    def set_size(self, width: float, height: float) -> None:
        """Set spacer size.

        Args:
            width: New width
            height: New height
        """
        self.width = width
        self.height = height

    def set_constraints(
        self,
        min_width: Optional[float] = None,
        min_height: Optional[float] = None,
        max_width: Optional[float] = None,
        max_height: Optional[float] = None,
    ) -> None:
        """Set size constraints.

        Args:
            min_width: Minimum width
            min_height: Minimum height
            max_width: Maximum width
            max_height: Maximum height
        """
        self._min_width = min_width
        self._min_height = min_height
        self._max_width = max_width
        self._max_height = max_height
        # Re-apply constraints to current size
        self.width = self._width
        self.height = self._height

    def __repr__(self) -> str:
        return (
            f"Spacer(width={self._width}, height={self._height}, "
            f"mode={self._mode.name}, flex={self._flex})"
        )


__all__ = [
    "Spacer",
    "SpacerMode",
]
