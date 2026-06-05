"""
Spacer Widget - Layout spacing utility.

Provides a flexible spacer widget for creating gaps and
distributing space in layouts.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Dict, Optional


class SpacerMode(Enum):
    """Spacer sizing modes."""
    FIXED = auto()      # Fixed size in pixels
    FLEXIBLE = auto()   # Flexible size based on flex factor
    FILL = auto()       # Fill all available space
    MINIMUM = auto()    # Minimum size with potential to grow


class Spacer:
    """Spacer widget for layout spacing.

    A non-visible widget that creates space in layouts.
    Can be fixed size, flexible, or fill available space.

    Attributes:
        mode: Sizing mode (fixed, flexible, fill)
        size: Size value (width for horizontal, height for vertical)
        flex: Flex factor for flexible mode
        horizontal: Whether spacer is horizontal (default True)
        min_size: Minimum size constraint
        max_size: Maximum size constraint
        is_dirty: Whether the spacer needs re-layout
    """

    __slots__ = (
        '_mode', '_size', '_flex', '_horizontal',
        '_min_size', '_max_size', '_is_dirty',
    )

    def __init__(
        self,
        mode: SpacerMode = SpacerMode.FIXED,
        size: float = 0.0,
        flex: float = 1.0,
        horizontal: bool = True,
        min_size: Optional[float] = None,
        max_size: Optional[float] = None,
    ) -> None:
        """Initialize the spacer.

        Args:
            mode: Sizing mode
            size: Initial size value
            flex: Flex factor for flexible mode
            horizontal: Whether spacer expands horizontally
            min_size: Minimum size constraint
            max_size: Maximum size constraint

        Raises:
            ValueError: If size < 0, flex <= 0, or min_size > max_size
        """
        if size < 0:
            raise ValueError(f"size must be >= 0, got {size}")
        if flex <= 0:
            raise ValueError(f"flex must be > 0, got {flex}")
        if min_size is not None and max_size is not None:
            if min_size > max_size:
                raise ValueError(
                    f"min_size cannot be greater than max_size "
                    f"({min_size} > {max_size})"
                )

        self._mode = mode
        self._size = size
        self._flex = flex
        self._horizontal = horizontal
        self._min_size = min_size
        self._max_size = max_size
        self._is_dirty = True

    @classmethod
    def fixed(cls, size: float) -> "Spacer":
        """Create a fixed-size spacer.

        Args:
            size: Fixed size value

        Returns:
            Fixed spacer instance
        """
        return cls(mode=SpacerMode.FIXED, size=size)

    @classmethod
    def flexible(cls, flex: float = 1.0) -> "Spacer":
        """Create a flexible spacer that distributes space.

        Args:
            flex: Flex factor (higher = more space)

        Returns:
            Flexible spacer instance
        """
        return cls(mode=SpacerMode.FLEXIBLE, flex=flex)

    @classmethod
    def fill(cls) -> "Spacer":
        """Create a fill spacer that takes all available space.

        Returns:
            Fill spacer instance
        """
        return cls(mode=SpacerMode.FILL)

    @classmethod
    def horizontal_spacer(cls, size: float) -> "Spacer":
        """Create a horizontal spacer (fixed width, zero height).

        Args:
            size: Spacer width

        Returns:
            Horizontal spacer instance
        """
        return cls(mode=SpacerMode.FIXED, size=size, horizontal=True)

    @classmethod
    def vertical_spacer(cls, size: float) -> "Spacer":
        """Create a vertical spacer (zero width, fixed height).

        Args:
            size: Spacer height

        Returns:
            Vertical spacer instance
        """
        return cls(mode=SpacerMode.FIXED, size=size, horizontal=False)

    @classmethod
    def expand(cls) -> "Spacer":
        """Create an expanding spacer (shorthand for flexible(1.0))."""
        return cls.flexible(1.0)

    @property
    def mode(self) -> SpacerMode:
        """Get sizing mode."""
        return self._mode

    @mode.setter
    def mode(self, value: SpacerMode) -> None:
        """Set sizing mode."""
        if value != self._mode:
            self._mode = value
            self._is_dirty = True

    @property
    def size(self) -> float:
        """Get size value."""
        return self._size

    @size.setter
    def size(self, value: float) -> None:
        """Set size value."""
        if value < 0:
            raise ValueError(f"size must be >= 0, got {value}")
        if value != self._size:
            self._size = value
            self._is_dirty = True

    @property
    def flex(self) -> float:
        """Get flex factor."""
        return self._flex

    @flex.setter
    def flex(self, value: float) -> None:
        """Set flex factor."""
        if value <= 0:
            raise ValueError(f"flex must be > 0, got {value}")
        if value != self._flex:
            self._flex = value
            self._is_dirty = True

    @property
    def horizontal(self) -> bool:
        """Get orientation."""
        return self._horizontal

    @horizontal.setter
    def horizontal(self, value: bool) -> None:
        """Set orientation."""
        if value != self._horizontal:
            self._horizontal = value
            self._is_dirty = True

    @property
    def min_size(self) -> Optional[float]:
        """Get minimum size constraint."""
        return self._min_size

    @min_size.setter
    def min_size(self, value: Optional[float]) -> None:
        """Set minimum size constraint."""
        if value is not None and self._max_size is not None:
            if value > self._max_size:
                raise ValueError(
                    f"min_size cannot be greater than max_size "
                    f"({value} > {self._max_size})"
                )
        self._min_size = value
        self._is_dirty = True

    @property
    def max_size(self) -> Optional[float]:
        """Get maximum size constraint."""
        return self._max_size

    @max_size.setter
    def max_size(self, value: Optional[float]) -> None:
        """Set maximum size constraint."""
        if value is not None and self._min_size is not None:
            if self._min_size > value:
                raise ValueError(
                    f"min_size cannot be greater than max_size "
                    f"({self._min_size} > {value})"
                )
        self._max_size = value
        self._is_dirty = True

    @property
    def is_dirty(self) -> bool:
        """Check if spacer needs re-layout."""
        return self._is_dirty

    @property
    def width(self) -> float:
        """Get width based on orientation."""
        return self._size if self._horizontal else 0.0

    @property
    def height(self) -> float:
        """Get height based on orientation."""
        return 0.0 if self._horizontal else self._size

    @property
    def is_flexible(self) -> bool:
        """Check if spacer is flexible."""
        return self._mode == SpacerMode.FLEXIBLE

    @property
    def is_fixed(self) -> bool:
        """Check if spacer is fixed size."""
        return self._mode == SpacerMode.FIXED

    def mark_clean(self) -> None:
        """Mark spacer as clean (layout up to date)."""
        self._is_dirty = False

    def clamp_size(self, value: float) -> float:
        """Clamp a size value to constraints.

        Args:
            value: Size value to clamp

        Returns:
            Clamped size value
        """
        result = value
        if self._min_size is not None:
            result = max(result, self._min_size)
        if self._max_size is not None:
            result = min(result, self._max_size)
        return result

    def compute_size(
        self,
        available_space: float = 0.0,
        total_flex: float = 1.0,
    ) -> float:
        """Compute the actual size based on mode and available space.

        Args:
            available_space: Total available space
            total_flex: Total flex factor of all flex items

        Returns:
            Computed size
        """
        if self._mode == SpacerMode.FIXED:
            result = self._size
        elif self._mode == SpacerMode.FLEXIBLE:
            # Distribute space proportionally based on flex factor
            if total_flex > 0:
                result = available_space * (self._flex / total_flex)
            else:
                result = 0.0
        elif self._mode == SpacerMode.FILL:
            result = available_space
        else:  # MINIMUM
            result = max(self._size, 0.0)

        return self.clamp_size(result)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize spacer to dictionary.

        Returns:
            Dictionary representation
        """
        data: Dict[str, Any] = {
            "mode": self._mode.name,
        }

        if self._mode == SpacerMode.FIXED or self._mode == SpacerMode.MINIMUM:
            data["size"] = self._size
        if self._mode == SpacerMode.FLEXIBLE:
            data["flex"] = self._flex

        if self._min_size is not None:
            data["min_size"] = self._min_size
        if self._max_size is not None:
            data["max_size"] = self._max_size

        data["horizontal"] = self._horizontal

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Spacer":
        """Deserialize spacer from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Spacer instance
        """
        mode_name = data.get("mode", "FIXED")
        mode = SpacerMode[mode_name]

        return cls(
            mode=mode,
            size=data.get("size", 0.0),
            flex=data.get("flex", 1.0),
            horizontal=data.get("horizontal", True),
            min_size=data.get("min_size"),
            max_size=data.get("max_size"),
        )

    def __repr__(self) -> str:
        return (
            f"Spacer(mode={self._mode.name}, size={self._size}, "
            f"flex={self._flex}, horizontal={self._horizontal})"
        )


__all__ = [
    "Spacer",
    "SpacerMode",
]
