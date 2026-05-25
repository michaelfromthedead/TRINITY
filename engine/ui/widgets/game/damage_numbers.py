"""
Floating damage numbers for combat feedback.

Provides animated floating text for:
- Damage display with color coding
- Healing numbers
- Critical hit emphasis
- Number stacking/combining
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional


class DamageType(Enum):
    """Types of damage for color coding."""
    PHYSICAL = auto()
    MAGIC = auto()
    FIRE = auto()
    ICE = auto()
    LIGHTNING = auto()
    POISON = auto()
    HEAL = auto()
    SHIELD = auto()
    EXPERIENCE = auto()
    MISS = auto()
    BLOCKED = auto()
    ABSORBED = auto()
    CUSTOM = auto()


@dataclass(slots=True)
class DamageNumberConfig:
    """Configuration for damage number appearance and behavior."""
    # Colors by type
    physical_color: str = "#ffffff"
    magic_color: str = "#a855f7"
    fire_color: str = "#ef4444"
    ice_color: str = "#06b6d4"
    lightning_color: str = "#facc15"
    poison_color: str = "#22c55e"
    heal_color: str = "#10b981"
    shield_color: str = "#3b82f6"
    experience_color: str = "#8b5cf6"
    miss_color: str = "#6b7280"
    blocked_color: str = "#9ca3af"
    absorbed_color: str = "#60a5fa"

    # Critical hit styling
    crit_color: str = "#fbbf24"
    crit_scale: float = 1.5
    crit_prefix: str = ""
    crit_suffix: str = "!"

    # Animation
    duration: float = 1.5  # Total animation time
    rise_distance: float = 60.0  # How far to float up
    rise_speed: float = 80.0  # Initial rise speed
    fade_start: float = 0.7  # When to start fading (0-1)
    spread_range: float = 20.0  # Horizontal spread range

    # Text
    font_size: float = 16.0
    crit_font_size: float = 24.0
    font_weight: str = "bold"
    outline_color: str = "#000000"
    outline_width: float = 2.0

    # Stacking
    stack_threshold: float = 0.2  # Time window for stacking
    stack_distance: float = 30.0  # Distance for combining
    max_stack_count: int = 5  # Max numbers to combine

    # Misc
    random_offset: bool = True
    random_rotation: bool = False
    max_rotation: float = 15.0


@dataclass(slots=True)
class DamageNumber:
    """A single floating damage number."""
    id: int
    value: float
    damage_type: DamageType
    world_x: float
    world_y: float
    is_critical: bool = False
    is_stacked: bool = False
    stack_count: int = 1

    # Animation state
    elapsed: float = 0.0
    screen_x: float = 0.0
    screen_y: float = 0.0
    opacity: float = 1.0
    scale: float = 1.0
    rotation: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    # Styling
    color: str = "#ffffff"
    font_size: float = 16.0

    # Flags
    is_active: bool = True
    is_visible: bool = True

    # Custom
    custom_text: Optional[str] = None
    custom_data: dict = field(default_factory=dict)

    def get_display_text(self) -> str:
        """Get the text to display."""
        if self.custom_text:
            return self.custom_text

        if self.damage_type == DamageType.MISS:
            return "MISS"
        elif self.damage_type == DamageType.BLOCKED:
            return "BLOCKED"
        elif self.damage_type == DamageType.ABSORBED:
            return "ABSORBED"

        # Format number
        if self.value >= 1000000:
            text = f"{self.value / 1000000:.1f}M"
        elif self.value >= 1000:
            text = f"{self.value / 1000:.1f}K"
        else:
            text = str(int(self.value))

        # Add prefix for heals
        if self.damage_type == DamageType.HEAL:
            text = f"+{text}"
        elif self.damage_type == DamageType.EXPERIENCE:
            text = f"+{text} XP"

        return text


class DamageNumberManager:
    """Manager for floating damage numbers.

    Features:
    - Spawn damage numbers at world positions
    - Automatic animation and lifecycle
    - Number stacking/combining
    - Color coding by damage type
    - Critical hit emphasis
    - Screen space conversion
    """

    __slots__ = (
        '_numbers', '_next_id',
        '_config',
        '_pending_stacks',
        '_world_to_screen',
        '_max_active',
        '_pool',
    )

    def __init__(
        self,
        config: Optional[DamageNumberConfig] = None,
        max_active: int = 100,
    ):
        """Initialize the damage number manager.

        Args:
            config: Display configuration
            max_active: Maximum concurrent numbers
        """
        self._numbers: list[DamageNumber] = []
        self._next_id = 0
        self._config = config or DamageNumberConfig()
        self._pending_stacks: dict[tuple[float, float], list[DamageNumber]] = {}
        self._world_to_screen: Optional[Callable[[float, float], tuple[float, float]]] = None
        self._max_active = max_active
        self._pool: list[DamageNumber] = []

    @property
    def config(self) -> DamageNumberConfig:
        """Get configuration."""
        return self._config

    @config.setter
    def config(self, value: DamageNumberConfig) -> None:
        """Set configuration."""
        self._config = value

    @property
    def active_count(self) -> int:
        """Get number of active damage numbers."""
        return len(self._numbers)

    @property
    def active_numbers(self) -> list[DamageNumber]:
        """Get list of active damage numbers."""
        return self._numbers.copy()

    def set_world_to_screen_converter(
        self,
        converter: Callable[[float, float], tuple[float, float]]
    ) -> None:
        """Set the world-to-screen coordinate converter.

        Args:
            converter: Function(world_x, world_y) -> (screen_x, screen_y)
        """
        self._world_to_screen = converter

    def spawn(
        self,
        value: float,
        world_x: float,
        world_y: float,
        damage_type: DamageType = DamageType.PHYSICAL,
        is_critical: bool = False,
        custom_text: Optional[str] = None,
        custom_color: Optional[str] = None,
    ) -> int:
        """Spawn a new damage number.

        Args:
            value: Damage/heal value
            world_x: World X position
            world_y: World Y position
            damage_type: Type of damage
            is_critical: Whether it's a critical hit
            custom_text: Override display text
            custom_color: Override color

        Returns:
            Damage number ID
        """
        # Remove oldest if at capacity
        if len(self._numbers) >= self._max_active:
            self._numbers.pop(0)

        # Check for stacking
        if self._config.stack_threshold > 0:
            stacked = self._try_stack(value, world_x, world_y, damage_type)
            if stacked is not None:
                return stacked.id

        # Create new number
        number_id = self._next_id
        self._next_id += 1

        # Get color
        if custom_color:
            color = custom_color
        elif is_critical:
            color = self._config.crit_color
        else:
            color = self._get_color_for_type(damage_type)

        # Get font size
        font_size = self._config.crit_font_size if is_critical else self._config.font_size

        # Calculate initial velocity with spread
        velocity_y = -self._config.rise_speed
        velocity_x = 0.0

        if self._config.random_offset:
            velocity_x = random.uniform(
                -self._config.spread_range,
                self._config.spread_range
            )

        # Calculate initial screen position
        screen_x, screen_y = world_x, world_y
        if self._world_to_screen:
            screen_x, screen_y = self._world_to_screen(world_x, world_y)

        # Calculate rotation if enabled
        rotation = 0.0
        if self._config.random_rotation:
            rotation = random.uniform(
                -self._config.max_rotation,
                self._config.max_rotation
            )

        number = DamageNumber(
            id=number_id,
            value=value,
            damage_type=damage_type,
            world_x=world_x,
            world_y=world_y,
            is_critical=is_critical,
            screen_x=screen_x,
            screen_y=screen_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            color=color,
            font_size=font_size,
            scale=self._config.crit_scale if is_critical else 1.0,
            rotation=rotation,
            custom_text=custom_text,
        )

        self._numbers.append(number)

        # Track for potential stacking
        key = self._get_stack_key(world_x, world_y)
        if key not in self._pending_stacks:
            self._pending_stacks[key] = []
        self._pending_stacks[key].append(number)

        return number_id

    def spawn_heal(
        self,
        value: float,
        world_x: float,
        world_y: float,
        is_critical: bool = False,
    ) -> int:
        """Spawn a healing number.

        Args:
            value: Heal value
            world_x: World X position
            world_y: World Y position
            is_critical: Whether it's a critical heal

        Returns:
            Damage number ID
        """
        return self.spawn(
            value=value,
            world_x=world_x,
            world_y=world_y,
            damage_type=DamageType.HEAL,
            is_critical=is_critical,
        )

    def spawn_miss(self, world_x: float, world_y: float) -> int:
        """Spawn a miss indicator.

        Args:
            world_x: World X position
            world_y: World Y position

        Returns:
            Damage number ID
        """
        return self.spawn(
            value=0,
            world_x=world_x,
            world_y=world_y,
            damage_type=DamageType.MISS,
        )

    def spawn_blocked(self, world_x: float, world_y: float) -> int:
        """Spawn a blocked indicator.

        Args:
            world_x: World X position
            world_y: World Y position

        Returns:
            Damage number ID
        """
        return self.spawn(
            value=0,
            world_x=world_x,
            world_y=world_y,
            damage_type=DamageType.BLOCKED,
        )

    def spawn_experience(
        self,
        value: float,
        world_x: float,
        world_y: float,
    ) -> int:
        """Spawn an experience gain indicator.

        Args:
            value: XP amount
            world_x: World X position
            world_y: World Y position

        Returns:
            Damage number ID
        """
        return self.spawn(
            value=value,
            world_x=world_x,
            world_y=world_y,
            damage_type=DamageType.EXPERIENCE,
        )

    def update(self, delta_time: float) -> None:
        """Update all damage numbers.

        Args:
            delta_time: Time since last update
        """
        # Update pending stack windows
        self._update_pending_stacks(delta_time)

        # Update each number
        active_numbers = []

        for number in self._numbers:
            if not number.is_active:
                continue

            number.elapsed += delta_time

            # Check if expired
            if number.elapsed >= self._config.duration:
                number.is_active = False
                continue

            # Update position
            number.screen_x += number.velocity_x * delta_time
            number.screen_y += number.velocity_y * delta_time

            # Apply deceleration to vertical velocity
            decel_factor = 0.95
            number.velocity_y *= decel_factor
            number.velocity_x *= decel_factor

            # Update opacity (fade out near end)
            progress = number.elapsed / self._config.duration
            if progress > self._config.fade_start:
                fade_progress = (progress - self._config.fade_start) / (1.0 - self._config.fade_start)
                number.opacity = 1.0 - fade_progress

            # Scale animation for criticals
            if number.is_critical:
                # Pop-in effect
                if number.elapsed < 0.1:
                    t = number.elapsed / 0.1
                    number.scale = self._config.crit_scale * self._ease_out_back(t)

            # Update world-to-screen if converter set
            if self._world_to_screen and number.elapsed < 0.01:
                number.screen_x, number.screen_y = self._world_to_screen(
                    number.world_x,
                    number.world_y
                )

            active_numbers.append(number)

        self._numbers = active_numbers

    def clear(self) -> None:
        """Clear all damage numbers."""
        self._numbers.clear()
        self._pending_stacks.clear()

    def get_number(self, number_id: int) -> Optional[DamageNumber]:
        """Get a damage number by ID.

        Args:
            number_id: Number ID

        Returns:
            DamageNumber if found
        """
        for number in self._numbers:
            if number.id == number_id:
                return number
        return None

    def remove(self, number_id: int) -> bool:
        """Remove a damage number.

        Args:
            number_id: Number ID

        Returns:
            True if removed
        """
        for i, number in enumerate(self._numbers):
            if number.id == number_id:
                self._numbers.pop(i)
                return True
        return False

    # Rendering helpers
    def get_visible_numbers(self) -> list[DamageNumber]:
        """Get all visible damage numbers for rendering.

        Returns:
            List of visible numbers sorted by screen Y
        """
        visible = [n for n in self._numbers if n.is_visible and n.opacity > 0]
        # Sort by Y for proper layering
        visible.sort(key=lambda n: n.screen_y)
        return visible

    def get_render_data(self, number: DamageNumber) -> dict:
        """Get render data for a damage number.

        Args:
            number: The damage number

        Returns:
            Dict with rendering properties
        """
        text = number.get_display_text()

        if number.is_critical and not number.custom_text:
            text = f"{self._config.crit_prefix}{text}{self._config.crit_suffix}"

        return {
            "text": text,
            "x": number.screen_x,
            "y": number.screen_y,
            "color": number.color,
            "font_size": number.font_size * number.scale,
            "opacity": number.opacity,
            "rotation": number.rotation,
            "outline_color": self._config.outline_color,
            "outline_width": self._config.outline_width,
            "font_weight": self._config.font_weight,
        }

    # Private methods
    def _get_color_for_type(self, damage_type: DamageType) -> str:
        """Get color for damage type."""
        colors = {
            DamageType.PHYSICAL: self._config.physical_color,
            DamageType.MAGIC: self._config.magic_color,
            DamageType.FIRE: self._config.fire_color,
            DamageType.ICE: self._config.ice_color,
            DamageType.LIGHTNING: self._config.lightning_color,
            DamageType.POISON: self._config.poison_color,
            DamageType.HEAL: self._config.heal_color,
            DamageType.SHIELD: self._config.shield_color,
            DamageType.EXPERIENCE: self._config.experience_color,
            DamageType.MISS: self._config.miss_color,
            DamageType.BLOCKED: self._config.blocked_color,
            DamageType.ABSORBED: self._config.absorbed_color,
            DamageType.CUSTOM: "#ffffff",
        }
        return colors.get(damage_type, "#ffffff")

    def _get_stack_key(self, world_x: float, world_y: float) -> tuple[int, int]:
        """Get grid key for stacking."""
        grid_size = self._config.stack_distance
        return (int(world_x / grid_size), int(world_y / grid_size))

    def _try_stack(
        self,
        value: float,
        world_x: float,
        world_y: float,
        damage_type: DamageType,
    ) -> Optional[DamageNumber]:
        """Try to stack with existing number.

        Returns:
            Stacked number if successful, None otherwise
        """
        key = self._get_stack_key(world_x, world_y)

        if key not in self._pending_stacks:
            return None

        candidates = self._pending_stacks[key]

        for number in candidates:
            if not number.is_active:
                continue

            if number.damage_type != damage_type:
                continue

            if number.stack_count >= self._config.max_stack_count:
                continue

            # Check distance
            dx = world_x - number.world_x
            dy = world_y - number.world_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= self._config.stack_distance:
                # Stack with this number
                number.value += value
                number.stack_count += 1
                number.is_stacked = True

                # Reset animation slightly
                number.elapsed = max(0, number.elapsed - 0.1)
                number.opacity = 1.0

                # Scale up slightly
                number.scale = min(2.0, number.scale + 0.1)

                return number

        return None

    def _update_pending_stacks(self, delta_time: float) -> None:
        """Clean up old pending stack entries."""
        threshold = self._config.stack_threshold

        keys_to_remove = []

        for key, numbers in self._pending_stacks.items():
            # Remove expired numbers from tracking
            numbers[:] = [
                n for n in numbers
                if n.is_active and n.elapsed < threshold
            ]

            if not numbers:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._pending_stacks[key]

    @staticmethod
    def _ease_out_back(t: float) -> float:
        """Ease-out-back interpolation for pop effect."""
        c1 = 1.70158
        c3 = c1 + 1
        return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)

    def __repr__(self) -> str:
        return f"DamageNumberManager(active={len(self._numbers)})"
