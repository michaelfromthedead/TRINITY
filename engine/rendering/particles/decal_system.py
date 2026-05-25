"""
Projected Decal System.

Provides screen-space and deferred decals that project textures onto scene surfaces.
Used for bullet holes, blood splatter, graffiti, tire marks, and other surface details.

Architecture:
    Decal - Single projected decal instance
    DecalVolume - Box projection volume for decal bounds
    DeferredDecal - Modifies G-Buffer channels (albedo, normal, roughness)
    DecalAtlas - Texture packing for multiple decal textures
    DecalConfig - Configuration from @decal decorator

Features:
    - Lifetime and fade-out support
    - Multiple G-Buffer channel modification
    - Priority-based sorting
    - Depth-based sorting for correct layering
    - Atlas packing for efficient batching
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from engine.rendering.particles.particle_system import Vec3, Vec4


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class DecalChannel(Enum):
    """G-Buffer channels that decals can modify."""

    ALBEDO = auto()  # Base color/diffuse
    NORMAL = auto()  # Surface normal
    ROUGHNESS = auto()  # PBR roughness
    METALLIC = auto()  # PBR metallic
    AO = auto()  # Ambient occlusion
    EMISSIVE = auto()  # Emissive/glow


class DecalBlendMode(Enum):
    """How decal blends with surface."""

    REPLACE = auto()  # Replace surface values
    MULTIPLY = auto()  # Multiply with surface
    ADD = auto()  # Add to surface
    ALPHA_BLEND = auto()  # Standard alpha blend
    OVERLAY = auto()  # Overlay blend mode


class DecalProjection(Enum):
    """Decal projection type."""

    BOX = auto()  # Standard box projection
    SPHERE = auto()  # Spherical projection
    CYLINDER = auto()  # Cylindrical projection


class DecalSortMode(Enum):
    """How decals are sorted for rendering."""

    PRIORITY = auto()  # Sort by user-defined priority
    DEPTH = auto()  # Sort by depth from camera
    CREATION_TIME = auto()  # Sort by creation order


# Import centralized constants
from engine.rendering.particles.constants import (
    PARTICLE_CONSTANTS,
    DEFAULT_DECAL_LIFETIME,
    DEFAULT_DECAL_FADE_TIME,
    DEFAULT_DECAL_CHANNEL,
    DEFAULT_DECAL_PRIORITY,
)

# Legacy names for backwards compatibility
DEFAULT_FADE_TIME = DEFAULT_DECAL_FADE_TIME
DEFAULT_CHANNEL = DEFAULT_DECAL_CHANNEL
DEFAULT_PRIORITY = DEFAULT_DECAL_PRIORITY
DEFAULT_DECAL_SIZE = Vec3(*PARTICLE_CONSTANTS.DECAL_DEFAULT_SIZE)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DecalConfig:
    """
    Configuration for decal from @decal decorator.

    Attributes:
        lifetime: Time before decal starts fading (None = infinite)
        fade_time: Duration of fade-out
        channel: Decal channel/layer index
        priority: Rendering priority (higher = rendered later/on top)
    """

    lifetime: Optional[float] = DEFAULT_DECAL_LIFETIME
    fade_time: float = DEFAULT_FADE_TIME
    channel: int = DEFAULT_CHANNEL
    priority: int = DEFAULT_PRIORITY
    blend_mode: DecalBlendMode = DecalBlendMode.ALPHA_BLEND

    @classmethod
    def from_decorator_params(
        cls,
        lifetime: Optional[float] = DEFAULT_DECAL_LIFETIME,
        fade_time: float = DEFAULT_FADE_TIME,
        channel: int = DEFAULT_CHANNEL,
        **kwargs: Any,
    ) -> "DecalConfig":
        """Create config from @decal decorator parameters."""
        return cls(
            lifetime=lifetime,
            fade_time=fade_time,
            channel=channel,
            **kwargs,
        )


# =============================================================================
# DECAL VOLUME
# =============================================================================


@dataclass
class DecalVolume:
    """
    Box projection volume for decal bounds.

    Defines the 3D region where the decal projects onto surfaces.
    """

    position: Vec3 = field(default_factory=Vec3)
    rotation: Vec3 = field(default_factory=Vec3)  # Euler angles in radians
    size: Vec3 = field(default_factory=lambda: Vec3(1, 1, 1))

    # Computed transformation matrices (cached)
    _world_to_decal: Optional[list[float]] = field(default=None, repr=False)
    _decal_to_world: Optional[list[float]] = field(default=None, repr=False)

    def get_corners(self) -> list[Vec3]:
        """Get the 8 corners of the box volume in world space."""
        half = Vec3(self.size.x / 2, self.size.y / 2, self.size.z / 2)

        # Local corners
        local_corners = [
            Vec3(-half.x, -half.y, -half.z),
            Vec3(half.x, -half.y, -half.z),
            Vec3(-half.x, half.y, -half.z),
            Vec3(half.x, half.y, -half.z),
            Vec3(-half.x, -half.y, half.z),
            Vec3(half.x, -half.y, half.z),
            Vec3(-half.x, half.y, half.z),
            Vec3(half.x, half.y, half.z),
        ]

        # Transform to world space (simplified - no rotation)
        return [self.position + c for c in local_corners]

    def contains_point(self, point: Vec3) -> bool:
        """Check if a point is inside the volume."""
        # Transform point to local space (simplified)
        local = point - self.position

        half = Vec3(self.size.x / 2, self.size.y / 2, self.size.z / 2)

        return (
            abs(local.x) <= half.x
            and abs(local.y) <= half.y
            and abs(local.z) <= half.z
        )

    def get_uv_for_point(self, point: Vec3) -> Tuple[float, float]:
        """Get UV coordinates for a point on the decal surface."""
        # Transform to local normalized coordinates
        local = point - self.position
        half = Vec3(self.size.x / 2, self.size.y / 2, self.size.z / 2)

        u = (local.x / half.x + 1.0) * 0.5 if half.x > 0 else 0.5
        v = (local.z / half.z + 1.0) * 0.5 if half.z > 0 else 0.5

        return (max(0, min(1, u)), max(0, min(1, v)))


# =============================================================================
# DECAL
# =============================================================================


class Decal:
    """
    Single projected decal instance.

    Projects a texture onto surfaces within its projection volume.
    """

    def __init__(
        self,
        config: Optional[DecalConfig] = None,
        texture_id: Optional[str] = None,
    ) -> None:
        self._id = str(uuid.uuid4())[:8]
        self._config = config or DecalConfig()
        self._texture_id = texture_id

        # Transform
        self._volume = DecalVolume()

        # State
        self._age = 0.0
        self._alpha = 1.0
        self._is_alive = True
        self._is_fading = False

        # Rendering
        self._priority = self._config.priority
        self._depth = 0.0  # Distance from camera (for sorting)
        self._visible = True

        # G-Buffer channels to modify
        self._channels: set[DecalChannel] = {DecalChannel.ALBEDO}
        self._blend_mode = self._config.blend_mode

        # Color/tint
        self._color = Vec4(1, 1, 1, 1)

        # Atlas region (if using atlas)
        self._atlas_region: Optional[Tuple[float, float, float, float]] = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def config(self) -> DecalConfig:
        return self._config

    @property
    def volume(self) -> DecalVolume:
        return self._volume

    @property
    def position(self) -> Vec3:
        return self._volume.position

    @position.setter
    def position(self, value: Vec3) -> None:
        self._volume.position = value

    @property
    def size(self) -> Vec3:
        return self._volume.size

    @size.setter
    def size(self, value: Vec3) -> None:
        self._volume.size = value

    @property
    def rotation(self) -> Vec3:
        return self._volume.rotation

    @rotation.setter
    def rotation(self, value: Vec3) -> None:
        self._volume.rotation = value

    @property
    def age(self) -> float:
        return self._age

    @property
    def alpha(self) -> float:
        return self._alpha * self._color.w

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    @property
    def is_fading(self) -> bool:
        return self._is_fading

    @property
    def priority(self) -> int:
        return self._priority

    @priority.setter
    def priority(self, value: int) -> None:
        self._priority = value

    @property
    def depth(self) -> float:
        return self._depth

    @property
    def visible(self) -> bool:
        return self._visible and self._is_alive

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = value

    @property
    def color(self) -> Vec4:
        return self._color

    @color.setter
    def color(self, value: Vec4) -> None:
        self._color = value

    @property
    def texture_id(self) -> Optional[str]:
        return self._texture_id

    @texture_id.setter
    def texture_id(self, value: str) -> None:
        self._texture_id = value

    @property
    def channels(self) -> set[DecalChannel]:
        return self._channels

    def set_channels(self, *channels: DecalChannel) -> None:
        """Set which G-Buffer channels this decal modifies."""
        self._channels = set(channels)

    def set_atlas_region(
        self,
        u_min: float,
        v_min: float,
        u_max: float,
        v_max: float,
    ) -> None:
        """Set UV region within atlas."""
        self._atlas_region = (u_min, v_min, u_max, v_max)

    def update(self, dt: float) -> None:
        """Update decal state (aging and fading)."""
        if not self._is_alive:
            return

        self._age += dt

        # Check lifetime
        if self._config.lifetime is not None:
            if self._age >= self._config.lifetime:
                self._is_fading = True

        # Handle fading
        if self._is_fading:
            fade_progress = 0.0
            if self._config.lifetime is not None:
                time_since_fade_start = self._age - self._config.lifetime
                if self._config.fade_time > 0:
                    fade_progress = time_since_fade_start / self._config.fade_time
                else:
                    fade_progress = 1.0
            else:
                fade_progress = 1.0

            self._alpha = max(0.0, 1.0 - fade_progress)

            if self._alpha <= 0.0:
                self._is_alive = False

    def update_depth(self, camera_position: Vec3) -> None:
        """Update depth for sorting."""
        delta = self._volume.position - camera_position
        self._depth = math.sqrt(delta.x ** 2 + delta.y ** 2 + delta.z ** 2)

    def kill(self, immediate: bool = False) -> None:
        """Kill the decal."""
        if immediate:
            self._is_alive = False
            self._alpha = 0.0
        else:
            self._is_fading = True
            # Set lifetime to current age to start fading now
            if self._config.lifetime is None:
                # Need to create new config with lifetime set
                pass


# =============================================================================
# DEFERRED DECAL
# =============================================================================


class DeferredDecal(Decal):
    """
    Deferred decal that modifies G-Buffer channels.

    Projects onto the G-Buffer during the deferred lighting pass,
    allowing modification of albedo, normal, roughness, and other
    material properties.
    """

    def __init__(
        self,
        config: Optional[DecalConfig] = None,
        albedo_texture: Optional[str] = None,
        normal_texture: Optional[str] = None,
        roughness_texture: Optional[str] = None,
    ) -> None:
        super().__init__(config, albedo_texture)

        # Per-channel textures
        self._albedo_texture = albedo_texture
        self._normal_texture = normal_texture
        self._roughness_texture = roughness_texture

        # Channel-specific blend weights
        self._channel_weights: dict[DecalChannel, float] = {
            DecalChannel.ALBEDO: 1.0,
            DecalChannel.NORMAL: 1.0,
            DecalChannel.ROUGHNESS: 1.0,
            DecalChannel.METALLIC: 1.0,
            DecalChannel.AO: 1.0,
            DecalChannel.EMISSIVE: 1.0,
        }

        # Normal blending parameters
        self._normal_strength = 1.0
        self._normal_blend_mode = "reoriented"  # "reoriented", "rnm", "linear"

        # Set channels based on available textures
        self._update_channels()

    def _update_channels(self) -> None:
        """Update active channels based on available textures."""
        self._channels.clear()
        if self._albedo_texture:
            self._channels.add(DecalChannel.ALBEDO)
        if self._normal_texture:
            self._channels.add(DecalChannel.NORMAL)
        if self._roughness_texture:
            self._channels.add(DecalChannel.ROUGHNESS)

    @property
    def albedo_texture(self) -> Optional[str]:
        return self._albedo_texture

    @albedo_texture.setter
    def albedo_texture(self, value: str) -> None:
        self._albedo_texture = value
        self._update_channels()

    @property
    def normal_texture(self) -> Optional[str]:
        return self._normal_texture

    @normal_texture.setter
    def normal_texture(self, value: str) -> None:
        self._normal_texture = value
        self._update_channels()

    @property
    def roughness_texture(self) -> Optional[str]:
        return self._roughness_texture

    @roughness_texture.setter
    def roughness_texture(self, value: str) -> None:
        self._roughness_texture = value
        self._update_channels()

    @property
    def normal_strength(self) -> float:
        return self._normal_strength

    @normal_strength.setter
    def normal_strength(self, value: float) -> None:
        self._normal_strength = max(0.0, value)

    def set_channel_weight(self, channel: DecalChannel, weight: float) -> None:
        """Set blend weight for a specific channel."""
        self._channel_weights[channel] = max(0.0, min(1.0, weight))

    def get_channel_weight(self, channel: DecalChannel) -> float:
        """Get blend weight for a channel."""
        return self._channel_weights.get(channel, 1.0)


# =============================================================================
# DECAL ATLAS
# =============================================================================


@dataclass
class AtlasRegion:
    """Region within a texture atlas."""

    x: int
    y: int
    width: int
    height: int
    texture_id: str

    @property
    def u_min(self) -> float:
        return 0.0  # Calculated during packing

    @property
    def v_min(self) -> float:
        return 0.0

    @property
    def u_max(self) -> float:
        return 1.0

    @property
    def v_max(self) -> float:
        return 1.0


class DecalAtlas:
    """
    Texture atlas for packing multiple decal textures.

    Combines multiple decal textures into a single atlas texture
    for efficient batched rendering.
    """

    def __init__(
        self,
        width: int = PARTICLE_CONSTANTS.DECAL_ATLAS_DEFAULT_WIDTH,
        height: int = PARTICLE_CONSTANTS.DECAL_ATLAS_DEFAULT_HEIGHT,
        padding: int = PARTICLE_CONSTANTS.DECAL_ATLAS_DEFAULT_PADDING,
    ) -> None:
        self._width = width
        self._height = height
        self._padding = padding

        # Packed regions
        self._regions: dict[str, AtlasRegion] = {}

        # Packing state (simple shelf algorithm)
        self._shelves: list[Tuple[int, int, int]] = []  # (y, height, x_offset)
        self._current_shelf_y = 0
        self._current_shelf_height = 0
        self._current_shelf_x = 0

        # Atlas texture data (in real implementation, would be GPU texture)
        self._texture_handle: Optional[Any] = None
        self._dirty = False

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def add_texture(
        self,
        texture_id: str,
        texture_width: int,
        texture_height: int,
    ) -> Optional[AtlasRegion]:
        """
        Add a texture to the atlas.

        Returns:
            AtlasRegion if successful, None if no space available
        """
        if texture_id in self._regions:
            return self._regions[texture_id]

        # Add padding
        padded_width = texture_width + self._padding * 2
        padded_height = texture_height + self._padding * 2

        # Try to fit on current shelf
        if self._current_shelf_x + padded_width <= self._width:
            if self._current_shelf_y + padded_height <= self._height:
                # Fits on current shelf
                x = self._current_shelf_x + self._padding
                y = self._current_shelf_y + self._padding

                region = AtlasRegion(
                    x=x,
                    y=y,
                    width=texture_width,
                    height=texture_height,
                    texture_id=texture_id,
                )

                self._regions[texture_id] = region
                self._current_shelf_x += padded_width
                self._current_shelf_height = max(
                    self._current_shelf_height, padded_height
                )
                self._dirty = True

                return region

        # Start new shelf
        new_shelf_y = self._current_shelf_y + self._current_shelf_height
        if new_shelf_y + padded_height <= self._height:
            self._current_shelf_y = new_shelf_y
            self._current_shelf_height = padded_height
            self._current_shelf_x = padded_width

            x = self._padding
            y = new_shelf_y + self._padding

            region = AtlasRegion(
                x=x,
                y=y,
                width=texture_width,
                height=texture_height,
                texture_id=texture_id,
            )

            self._regions[texture_id] = region
            self._dirty = True

            return region

        # No space available
        return None

    def get_region(self, texture_id: str) -> Optional[AtlasRegion]:
        """Get atlas region for a texture."""
        return self._regions.get(texture_id)

    def get_uv_rect(self, texture_id: str) -> Optional[Tuple[float, float, float, float]]:
        """Get UV coordinates (u_min, v_min, u_max, v_max) for a texture."""
        region = self._regions.get(texture_id)
        if not region:
            return None

        return (
            region.x / self._width,
            region.y / self._height,
            (region.x + region.width) / self._width,
            (region.y + region.height) / self._height,
        )

    def clear(self) -> None:
        """Clear all regions."""
        self._regions.clear()
        self._current_shelf_y = 0
        self._current_shelf_height = 0
        self._current_shelf_x = 0
        self._dirty = True

    def get_occupancy(self) -> float:
        """Get atlas occupancy ratio (0-1)."""
        total_area = self._width * self._height
        used_area = sum(
            (r.width + self._padding * 2) * (r.height + self._padding * 2)
            for r in self._regions.values()
        )
        return used_area / total_area if total_area > 0 else 0.0


# =============================================================================
# DECAL SORTING
# =============================================================================


class DecalSorting:
    """Utilities for sorting decals for correct rendering order."""

    @staticmethod
    def sort_by_priority(decals: list[Decal]) -> list[Decal]:
        """Sort decals by priority (lower priority renders first)."""
        return sorted(decals, key=lambda d: d.priority)

    @staticmethod
    def sort_by_depth(decals: list[Decal], camera_position: Vec3) -> list[Decal]:
        """Sort decals by depth (back to front for alpha blending)."""
        for decal in decals:
            decal.update_depth(camera_position)
        return sorted(decals, key=lambda d: -d.depth)  # Far to near

    @staticmethod
    def sort_by_priority_then_depth(
        decals: list[Decal],
        camera_position: Vec3,
    ) -> list[Decal]:
        """Sort by priority, then by depth within same priority."""
        for decal in decals:
            decal.update_depth(camera_position)
        return sorted(decals, key=lambda d: (d.priority, -d.depth))


# =============================================================================
# DECAL SYSTEM
# =============================================================================


class DecalSystem:
    """
    Manager for all decals in the scene.

    Handles decal creation, updates, sorting, and rendering preparation.
    """

    def __init__(
        self,
        max_decals: int = PARTICLE_CONSTANTS.DECAL_SYSTEM_MAX_DECALS,
        default_config: Optional[DecalConfig] = None,
    ) -> None:
        self._max_decals = max_decals
        self._default_config = default_config or DecalConfig()

        # Decal storage
        self._decals: dict[str, Decal] = {}
        self._decal_order: list[str] = []  # Ordered list for iteration

        # Atlas
        self._atlas = DecalAtlas()

        # Sorting
        self._sort_mode = DecalSortMode.PRIORITY
        self._needs_sort = True

        # Camera (for depth sorting)
        self._camera_position = Vec3()

        # Statistics
        self._visible_count = 0
        self._spawned_count = 0
        self._killed_count = 0

    @property
    def decal_count(self) -> int:
        return len(self._decals)

    @property
    def visible_count(self) -> int:
        return self._visible_count

    @property
    def atlas(self) -> DecalAtlas:
        return self._atlas

    def set_sort_mode(self, mode: DecalSortMode) -> None:
        """Set decal sorting mode."""
        self._sort_mode = mode
        self._needs_sort = True

    def set_camera(self, position: Vec3) -> None:
        """Set camera position for depth sorting."""
        self._camera_position = position
        if self._sort_mode == DecalSortMode.DEPTH:
            self._needs_sort = True

    def spawn(
        self,
        position: Vec3,
        rotation: Vec3 = None,
        size: Vec3 = None,
        config: Optional[DecalConfig] = None,
        texture_id: Optional[str] = None,
    ) -> Optional[Decal]:
        """
        Spawn a new decal.

        Returns:
            Created decal, or None if at max capacity
        """
        # Check capacity
        if len(self._decals) >= self._max_decals:
            # Try to remove oldest dead decal
            self._cleanup_dead()
            if len(self._decals) >= self._max_decals:
                return None

        decal = Decal(
            config=config or self._default_config,
            texture_id=texture_id,
        )
        decal.position = position
        decal.rotation = rotation or Vec3()
        decal.size = size or DEFAULT_DECAL_SIZE

        self._decals[decal.id] = decal
        self._decal_order.append(decal.id)
        self._needs_sort = True
        self._spawned_count += 1

        return decal

    def spawn_deferred(
        self,
        position: Vec3,
        rotation: Vec3 = None,
        size: Vec3 = None,
        config: Optional[DecalConfig] = None,
        albedo_texture: Optional[str] = None,
        normal_texture: Optional[str] = None,
        roughness_texture: Optional[str] = None,
    ) -> Optional[DeferredDecal]:
        """Spawn a deferred decal with multiple G-Buffer textures."""
        if len(self._decals) >= self._max_decals:
            self._cleanup_dead()
            if len(self._decals) >= self._max_decals:
                return None

        decal = DeferredDecal(
            config=config or self._default_config,
            albedo_texture=albedo_texture,
            normal_texture=normal_texture,
            roughness_texture=roughness_texture,
        )
        decal.position = position
        decal.rotation = rotation or Vec3()
        decal.size = size or DEFAULT_DECAL_SIZE

        self._decals[decal.id] = decal
        self._decal_order.append(decal.id)
        self._needs_sort = True
        self._spawned_count += 1

        return decal

    def get_decal(self, decal_id: str) -> Optional[Decal]:
        """Get decal by ID."""
        return self._decals.get(decal_id)

    def remove_decal(self, decal_id: str) -> None:
        """Remove a decal immediately."""
        if decal_id in self._decals:
            del self._decals[decal_id]
            if decal_id in self._decal_order:
                self._decal_order.remove(decal_id)
            self._killed_count += 1

    def update(self, dt: float) -> None:
        """Update all decals."""
        dead_ids = []

        for decal_id, decal in self._decals.items():
            decal.update(dt)
            if not decal.is_alive:
                dead_ids.append(decal_id)

        # Remove dead decals
        for decal_id in dead_ids:
            del self._decals[decal_id]
            if decal_id in self._decal_order:
                self._decal_order.remove(decal_id)
            self._killed_count += 1

        # Update visibility count
        self._visible_count = sum(1 for d in self._decals.values() if d.visible)

        # Sort if needed
        if self._needs_sort:
            self._sort_decals()
            self._needs_sort = False

    def _cleanup_dead(self) -> None:
        """Remove all dead decals."""
        dead_ids = [
            decal_id
            for decal_id, decal in self._decals.items()
            if not decal.is_alive
        ]
        for decal_id in dead_ids:
            del self._decals[decal_id]
            if decal_id in self._decal_order:
                self._decal_order.remove(decal_id)

    def _sort_decals(self) -> None:
        """Sort decals based on current sort mode."""
        decal_list = list(self._decals.values())

        if self._sort_mode == DecalSortMode.PRIORITY:
            sorted_decals = DecalSorting.sort_by_priority(decal_list)
        elif self._sort_mode == DecalSortMode.DEPTH:
            sorted_decals = DecalSorting.sort_by_depth(decal_list, self._camera_position)
        else:
            sorted_decals = decal_list  # Creation order (already in order)

        self._decal_order = [d.id for d in sorted_decals]

    def iter_visible(self) -> Iterator[Decal]:
        """Iterate over visible decals in render order."""
        for decal_id in self._decal_order:
            decal = self._decals.get(decal_id)
            if decal and decal.visible:
                yield decal

    def iter_by_channel(self, channel: DecalChannel) -> Iterator[Decal]:
        """Iterate over decals that modify a specific channel."""
        for decal_id in self._decal_order:
            decal = self._decals.get(decal_id)
            if decal and decal.visible and channel in decal.channels:
                yield decal

    def get_stats(self) -> dict[str, Any]:
        """Get system statistics."""
        return {
            "total_decals": len(self._decals),
            "visible_decals": self._visible_count,
            "max_decals": self._max_decals,
            "spawned_total": self._spawned_count,
            "killed_total": self._killed_count,
            "atlas_occupancy": self._atlas.get_occupancy(),
            "sort_mode": self._sort_mode.name,
        }


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Enums
    "DecalChannel",
    "DecalBlendMode",
    "DecalProjection",
    "DecalSortMode",
    # Configuration
    "DecalConfig",
    # Data structures
    "DecalVolume",
    "AtlasRegion",
    # Decal types
    "Decal",
    "DeferredDecal",
    # Atlas
    "DecalAtlas",
    # Sorting
    "DecalSorting",
    # System
    "DecalSystem",
]
