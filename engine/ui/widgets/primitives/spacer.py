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
    FILL = FLEXIBLE     # Alias for FLEXIBLE
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
        '_min_size', '_max_size',
        '_dirty', '_size', '_horizontal',
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
        size: Optional[float] = None,
        min_size: Optional[float] = None,
        max_size: Optional[float] = None,
        horizontal: Optional[bool] = None,
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
            size: Unified size (sets both width and height)
            min_size: Unified minimum size constraint
            max_size: Unified maximum size constraint
            horizontal: If True, size applies to width; if False, to height
        """
        if size is not None:
            if size < 0:
                raise ValueError("size must be >= 0")
            # If horizontal is specified, only set one dimension
            if horizontal is True:
                width = size
                height = 0.0
            elif horizontal is False:
                width = 0.0
                height = size
            else:
                width = size
                height = size
        if width < 0:
            raise ValueError("width must be >= 0")
        if height < 0:
            raise ValueError("height must be >= 0")
        if flex <= 0:
            raise ValueError("flex must be > 0")
        if min_size is not None and max_size is not None and min_size > max_size:
            raise ValueError("min_size cannot be greater than max_size")
        self._width = width
        self._height = height
        self._size = size
        self._mode = mode
        self._flex = flex
        self._min_width = min_width
        self._min_height = min_height
        self._max_width = max_width
        self._max_height = max_height
        self._min_size = min_size
        self._max_size = max_size
        self._horizontal = horizontal
        self._dirty = True

    @classmethod
    def fixed(cls, size: float = 0.0, width: Optional[float] = None, height: Optional[float] = None) -> "Spacer":
        """Create a fixed-size spacer.

        Args:
            size: Uniform size (sets both width and height)
            width: Fixed width (overrides size if provided)
            height: Fixed height (overrides size if provided)

        Returns:
            Fixed spacer instance
        """
        w = width if width is not None else size
        h = height if height is not None else size
        return cls(width=w, height=h, size=size, mode=SpacerMode.FIXED)

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

    @classmethod
    def fill(cls, flex: float = 1.0) -> "Spacer":
        """Create a fill spacer (alias for flexible with FILL mode).

        Args:
            flex: Flex factor (higher = more space)

        Returns:
            Fill spacer instance
        """
        return cls(mode=SpacerMode.FILL, flex=flex)

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
        self._dirty = True

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
    def min_size(self) -> Optional[float]:
        """Get minimum size constraint."""
        return self._min_size

    @property
    def max_size(self) -> Optional[float]:
        """Get maximum size constraint."""
        return self._max_size

    @property
    def computed_size(self) -> float:
        """Get computed size after constraints."""
        size = self._size if self._size is not None else self._width
        if self._min_size is not None:
            size = max(size, self._min_size)
        if self._max_size is not None:
            size = min(size, self._max_size)
        return size

    def clamp_size(self, value: float) -> float:
        """Clamp a size value to constraints."""
        if self._min_size is not None:
            value = max(value, self._min_size)
        if self._max_size is not None:
            value = min(value, self._max_size)
        return value

    @property
    def horizontal(self) -> Optional[bool]:
        """Get horizontal orientation."""
        return self._horizontal

    def compute_size(self, available_space: float, total_flex: float = 1.0) -> float:
        """Compute actual size based on mode and available space.

        Args:
            available_space: Total available space for layout
            total_flex: Total flex of all flexible items in container

        Returns:
            Computed size for this spacer
        """
        if self._mode == SpacerMode.FIXED:
            size = self._size if self._size is not None else self._width
        elif self._mode == SpacerMode.FLEXIBLE:
            size = available_space * (self._flex / total_flex)
        else:  # FILL or MINIMUM
            size = available_space
        return self.clamp_size(size)

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

    @property
    def is_dirty(self) -> bool:
        """Check if spacer needs re-rendering."""
        return self._dirty

    @property
    def size(self) -> float:
        """Get unified size (returns average of width/height if not explicitly set)."""
        if self._size is not None:
            return self._size
        return (self._width + self._height) / 2.0 if self._width == self._height else 0.0

    @size.setter
    def size(self, value: float) -> None:
        """Set unified size (sets both width and height)."""
        self._size = value
        self._width = max(0.0, value)
        self._height = max(0.0, value)
        self._dirty = True

    def mark_clean(self) -> None:
        """Mark spacer as rendered (clears dirty flag)."""
        self._dirty = False

    def to_dict(self) -> dict:
        """Serialize spacer to dictionary.

        Returns:
            Dictionary representation of spacer
        """
        data = {
            "mode": self._mode.name,
            "flex": self._flex,
        }
        if self._size is not None:
            data["size"] = self._size
        else:
            data["width"] = self._width
            data["height"] = self._height
        if self._min_width is not None:
            data["min_width"] = self._min_width
        if self._min_height is not None:
            data["min_height"] = self._min_height
        if self._max_width is not None:
            data["max_width"] = self._max_width
        if self._max_height is not None:
            data["max_height"] = self._max_height
        if self._min_size is not None:
            data["min_size"] = self._min_size
        if self._max_size is not None:
            data["max_size"] = self._max_size
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Spacer":
        """Deserialize spacer from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Spacer instance
        """
        mode = SpacerMode[data.get("mode", "FIXED")]
        return cls(
            width=data.get("width", 0.0),
            height=data.get("height", 0.0),
            mode=mode,
            flex=data.get("flex", 1.0),
            min_width=data.get("min_width"),
            min_height=data.get("min_height"),
            max_width=data.get("max_width"),
            max_height=data.get("max_height"),
            size=data.get("size"),
            min_size=data.get("min_size"),
            max_size=data.get("max_size"),
        )


__all__ = [
    "Spacer",
    "SpacerMode",
]
