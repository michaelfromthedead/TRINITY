"""
Foliage placement tools for terrain in the AI Game Engine.

Provides foliage instance management, density painting,
and LOD configuration for vegetation and decoration systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, Callable
import math
import random


class FoliageType(Enum):
    """Types of foliage objects."""
    GRASS = auto()
    TREE = auto()
    BUSH = auto()
    ROCK = auto()
    FLOWER = auto()
    DEBRIS = auto()
    CUSTOM = auto()


class FoliageLODLevel(Enum):
    """Foliage LOD levels."""
    LOD0 = 0  # Full detail
    LOD1 = 1  # Reduced detail
    LOD2 = 2  # Low detail
    LOD3 = 3  # Billboard/Impostor
    CULLED = 4  # Not rendered


@dataclass(slots=True)
class FoliageTransform:
    """Transform data for a foliage instance."""
    position_x: float = 0.0
    position_y: float = 0.0
    position_z: float = 0.0
    rotation_y: float = 0.0  # Yaw rotation in radians
    scale: float = 1.0


@dataclass(slots=True)
class FoliageInstance:
    """
    A single foliage instance in the world.

    Contains transform and instance-specific properties.
    """
    id: int
    layer_id: int
    transform: FoliageTransform = field(default_factory=FoliageTransform)
    health: float = 1.0  # For destructible foliage
    color_variation: float = 0.0  # Per-instance color offset
    current_lod: FoliageLODLevel = FoliageLODLevel.LOD0

    def distance_to(self, x: float, y: float, z: float) -> float:
        """Calculate distance to a point."""
        dx = self.transform.position_x - x
        dy = self.transform.position_y - y
        dz = self.transform.position_z - z
        return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(slots=True)
class FoliageLODSettings:
    """LOD settings for a foliage type."""
    lod0_distance: float = 50.0
    lod1_distance: float = 100.0
    lod2_distance: float = 200.0
    lod3_distance: float = 400.0
    cull_distance: float = 800.0
    crossfade_range: float = 10.0
    shadow_distance: float = 100.0

    def get_lod_for_distance(self, distance: float) -> FoliageLODLevel:
        """Determine LOD level based on distance."""
        if distance >= self.cull_distance:
            return FoliageLODLevel.CULLED
        elif distance >= self.lod3_distance:
            return FoliageLODLevel.LOD3
        elif distance >= self.lod2_distance:
            return FoliageLODLevel.LOD2
        elif distance >= self.lod1_distance:
            return FoliageLODLevel.LOD1
        else:
            return FoliageLODLevel.LOD0


@dataclass(slots=True)
class FoliageLayerSettings:
    """Settings for a foliage layer."""
    mesh_id: str = ""
    foliage_type: FoliageType = FoliageType.GRASS
    density: float = 1.0  # Instances per square unit
    min_scale: float = 0.8
    max_scale: float = 1.2
    random_rotation: bool = True
    align_to_terrain: bool = True
    align_to_terrain_strength: float = 0.5
    color_variation: float = 0.1
    min_slope: float = 0.0  # Minimum slope in degrees
    max_slope: float = 45.0  # Maximum slope in degrees
    min_height: float = -1000.0
    max_height: float = 1000.0
    collision_radius: float = 0.5
    receives_decals: bool = False
    cast_shadow: bool = True


@dataclass(slots=True)
class FoliageLayer:
    """
    A layer containing foliage instances of one type.

    Manages all instances of a specific foliage mesh with
    unified settings and LOD configuration.
    """
    id: int
    name: str
    settings: FoliageLayerSettings = field(default_factory=FoliageLayerSettings)
    lod_settings: FoliageLODSettings = field(default_factory=FoliageLODSettings)
    instances: list[FoliageInstance] = field(default_factory=list)
    _next_instance_id: int = 0
    density_map: Optional[list[list[float]]] = None

    def add_instance(self, transform: FoliageTransform) -> FoliageInstance:
        """Add a new foliage instance."""
        instance = FoliageInstance(
            id=self._next_instance_id,
            layer_id=self.id,
            transform=transform,
        )
        self._next_instance_id += 1
        self.instances.append(instance)
        return instance

    def remove_instance(self, instance_id: int) -> bool:
        """Remove an instance by ID."""
        for i, inst in enumerate(self.instances):
            if inst.id == instance_id:
                self.instances.pop(i)
                return True
        return False

    def get_instance(self, instance_id: int) -> Optional[FoliageInstance]:
        """Get an instance by ID."""
        for inst in self.instances:
            if inst.id == instance_id:
                return inst
        return None

    def get_instances_in_radius(
        self,
        x: float,
        y: float,
        z: float,
        radius: float
    ) -> list[FoliageInstance]:
        """Get all instances within a radius."""
        radius_sq = radius * radius
        result = []
        for inst in self.instances:
            dx = inst.transform.position_x - x
            dy = inst.transform.position_y - y
            dz = inst.transform.position_z - z
            if dx * dx + dy * dy + dz * dz <= radius_sq:
                result.append(inst)
        return result

    def clear_instances(self) -> None:
        """Remove all instances."""
        self.instances.clear()

    @property
    def instance_count(self) -> int:
        """Get number of instances."""
        return len(self.instances)


@dataclass(slots=True)
class DensityBrushSettings:
    """Settings for the density painting brush."""
    size: float = 10.0
    strength: float = 0.5
    falloff: float = 0.5
    spacing: float = 0.25


@dataclass(slots=True)
class FoliageDensityBrush:
    """
    Brush for painting foliage density.

    Controls where foliage instances are spawned based on
    painted density values.
    """
    settings: DensityBrushSettings = field(default_factory=DensityBrushSettings)

    def get_falloff(self, distance: float, max_distance: float) -> float:
        """Calculate brush falloff."""
        if max_distance <= 0:
            return 0.0

        normalized = min(1.0, distance / max_distance)
        falloff_start = 1.0 - self.settings.falloff

        if normalized < falloff_start:
            return 1.0

        t = (normalized - falloff_start) / self.settings.falloff if self.settings.falloff > 0 else 1.0
        return 1.0 - (t * t * (3.0 - 2.0 * t))

    def get_influence(self, x: float, y: float, center_x: float, center_y: float) -> float:
        """Get brush influence at a position."""
        dx = x - center_x
        dy = y - center_y
        distance = math.sqrt(dx * dx + dy * dy)
        radius = self.settings.size / 2.0

        if distance > radius:
            return 0.0

        return self.get_falloff(distance, radius) * self.settings.strength


class FoliagePlacementTool:
    """
    Main foliage placement and management tool.

    Handles foliage layer management, density painting,
    procedural placement, and LOD updates.
    """
    __slots__ = (
        "_terrain_width",
        "_terrain_height",
        "_terrain_heights",
        "_layers",
        "_brush",
        "_current_layer_id",
        "_random",
        "_seed",
    )

    def __init__(
        self,
        terrain_width: int,
        terrain_height: int,
        terrain_heights: Optional[list[list[float]]] = None
    ):
        """
        Initialize foliage tool.

        Args:
            terrain_width: Terrain width
            terrain_height: Terrain height
            terrain_heights: Optional terrain height data for placement
        """
        self._terrain_width = terrain_width
        self._terrain_height = terrain_height
        self._terrain_heights = terrain_heights
        self._layers: dict[int, FoliageLayer] = {}
        self._brush = FoliageDensityBrush()
        self._current_layer_id: Optional[int] = None
        self._random = random.Random()
        self._seed = 42

    @property
    def brush(self) -> FoliageDensityBrush:
        """Get density brush."""
        return self._brush

    @property
    def current_layer_id(self) -> Optional[int]:
        """Get current layer ID."""
        return self._current_layer_id

    @current_layer_id.setter
    def current_layer_id(self, value: Optional[int]) -> None:
        """Set current layer ID."""
        self._current_layer_id = value

    def set_brush(self, brush: FoliageDensityBrush) -> None:
        """Set the density brush."""
        self._brush = brush

    def set_seed(self, seed: int) -> None:
        """Set random seed for procedural placement."""
        self._seed = seed
        self._random.seed(seed)

    def set_terrain_heights(self, heights: list[list[float]]) -> None:
        """Set terrain height data."""
        self._terrain_heights = heights

    def add_layer(
        self,
        name: str,
        mesh_id: str,
        foliage_type: FoliageType = FoliageType.GRASS
    ) -> FoliageLayer:
        """
        Add a new foliage layer.

        Args:
            name: Layer name
            mesh_id: ID of the mesh to use
            foliage_type: Type of foliage

        Returns:
            The created layer
        """
        layer_id = len(self._layers)
        layer = FoliageLayer(
            id=layer_id,
            name=name,
            settings=FoliageLayerSettings(
                mesh_id=mesh_id,
                foliage_type=foliage_type,
            ),
        )
        self._layers[layer_id] = layer

        if self._current_layer_id is None:
            self._current_layer_id = layer_id

        return layer

    def remove_layer(self, layer_id: int) -> bool:
        """Remove a foliage layer."""
        if layer_id in self._layers:
            del self._layers[layer_id]
            if self._current_layer_id == layer_id:
                self._current_layer_id = next(iter(self._layers.keys()), None)
            return True
        return False

    def get_layer(self, layer_id: int) -> Optional[FoliageLayer]:
        """Get a layer by ID."""
        return self._layers.get(layer_id)

    def get_all_layers(self) -> list[FoliageLayer]:
        """Get all foliage layers."""
        return list(self._layers.values())

    def _get_terrain_height(self, x: float, z: float) -> float:
        """Get interpolated terrain height at a position."""
        if self._terrain_heights is None:
            return 0.0

        ix = int(x)
        iz = int(z)
        fx = x - ix
        fz = z - iz

        def sample(sx: int, sz: int) -> float:
            sx = max(0, min(self._terrain_width - 1, sx))
            sz = max(0, min(self._terrain_height - 1, sz))
            return self._terrain_heights[sz][sx]

        h00 = sample(ix, iz)
        h10 = sample(ix + 1, iz)
        h01 = sample(ix, iz + 1)
        h11 = sample(ix + 1, iz + 1)

        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx

        return h0 * (1 - fz) + h1 * fz

    def _get_terrain_slope(self, x: float, z: float) -> float:
        """Get terrain slope in degrees at a position."""
        if self._terrain_heights is None:
            return 0.0

        h = self._get_terrain_height(x, z)
        h_right = self._get_terrain_height(x + 1, z)
        h_up = self._get_terrain_height(x, z + 1)

        dx = h_right - h
        dz = h_up - h
        slope = math.sqrt(dx * dx + dz * dz)

        return math.degrees(math.atan(slope))

    def paint_density(
        self,
        center_x: float,
        center_z: float,
        add: bool = True
    ) -> int:
        """
        Paint foliage density at a position.

        Args:
            center_x, center_z: World position
            add: True to add density, False to remove

        Returns:
            Number of instances affected
        """
        if self._current_layer_id is None or self._current_layer_id not in self._layers:
            return 0

        layer = self._layers[self._current_layer_id]
        affected = 0

        if add:
            # Add new instances within brush area
            radius = self._brush.settings.size / 2.0
            density = layer.settings.density * self._brush.settings.strength

            # Calculate number of instances to add
            area = math.pi * radius * radius
            target_count = int(area * density)

            for _ in range(target_count):
                # Random position within brush
                angle = self._random.random() * 2 * math.pi
                dist = self._random.random() * radius

                x = center_x + math.cos(angle) * dist
                z = center_z + math.sin(angle) * dist

                # Check bounds
                if x < 0 or x >= self._terrain_width or z < 0 or z >= self._terrain_height:
                    continue

                # Check slope and height constraints
                slope = self._get_terrain_slope(x, z)
                if slope < layer.settings.min_slope or slope > layer.settings.max_slope:
                    continue

                height = self._get_terrain_height(x, z)
                if height < layer.settings.min_height or height > layer.settings.max_height:
                    continue

                # Check brush influence
                influence = self._brush.get_influence(x, z, center_x, center_z)
                if self._random.random() > influence:
                    continue

                # Create transform
                transform = FoliageTransform(
                    position_x=x,
                    position_y=height,
                    position_z=z,
                    rotation_y=self._random.random() * 2 * math.pi if layer.settings.random_rotation else 0,
                    scale=self._random.uniform(layer.settings.min_scale, layer.settings.max_scale),
                )

                layer.add_instance(transform)
                affected += 1
        else:
            # Remove instances within brush area
            instances_to_remove = []
            radius = self._brush.settings.size / 2.0

            for inst in layer.instances:
                influence = self._brush.get_influence(
                    inst.transform.position_x,
                    inst.transform.position_z,
                    center_x,
                    center_z
                )
                if influence > 0 and self._random.random() < influence:
                    instances_to_remove.append(inst.id)

            for inst_id in instances_to_remove:
                layer.remove_instance(inst_id)
                affected += 1

        return affected

    def place_instance(
        self,
        layer_id: int,
        x: float,
        z: float,
        rotation: float = 0.0,
        scale: float = 1.0
    ) -> Optional[FoliageInstance]:
        """
        Place a single foliage instance.

        Args:
            layer_id: Target layer
            x, z: World position
            rotation: Y rotation in radians
            scale: Instance scale

        Returns:
            The created instance, or None if layer not found
        """
        layer = self._layers.get(layer_id)
        if layer is None:
            return None

        height = self._get_terrain_height(x, z)

        transform = FoliageTransform(
            position_x=x,
            position_y=height,
            position_z=z,
            rotation_y=rotation,
            scale=scale,
        )

        return layer.add_instance(transform)

    def remove_instances_in_radius(
        self,
        x: float,
        y: float,
        z: float,
        radius: float,
        layer_id: Optional[int] = None
    ) -> int:
        """
        Remove all instances within a radius.

        Args:
            x, y, z: Center position
            radius: Removal radius
            layer_id: Optional specific layer, None for all layers

        Returns:
            Number of instances removed
        """
        removed = 0

        layers = [self._layers[layer_id]] if layer_id is not None else self._layers.values()

        for layer in layers:
            instances_to_remove = [
                inst.id for inst in layer.get_instances_in_radius(x, y, z, radius)
            ]
            for inst_id in instances_to_remove:
                if layer.remove_instance(inst_id):
                    removed += 1

        return removed

    def fill_area(
        self,
        layer_id: int,
        min_x: float,
        min_z: float,
        max_x: float,
        max_z: float,
        density_override: Optional[float] = None
    ) -> int:
        """
        Fill an area with foliage.

        Args:
            layer_id: Target layer
            min_x, min_z: Minimum corner
            max_x, max_z: Maximum corner
            density_override: Optional density override

        Returns:
            Number of instances placed
        """
        layer = self._layers.get(layer_id)
        if layer is None:
            return 0

        density = density_override or layer.settings.density
        width = max_x - min_x
        height = max_z - min_z
        area = width * height
        target_count = int(area * density)

        placed = 0

        for _ in range(target_count):
            x = self._random.uniform(min_x, max_x)
            z = self._random.uniform(min_z, max_z)

            # Check bounds
            if x < 0 or x >= self._terrain_width or z < 0 or z >= self._terrain_height:
                continue

            # Check constraints
            slope = self._get_terrain_slope(x, z)
            if slope < layer.settings.min_slope or slope > layer.settings.max_slope:
                continue

            height = self._get_terrain_height(x, z)
            if height < layer.settings.min_height or height > layer.settings.max_height:
                continue

            transform = FoliageTransform(
                position_x=x,
                position_y=height,
                position_z=z,
                rotation_y=self._random.random() * 2 * math.pi if layer.settings.random_rotation else 0,
                scale=self._random.uniform(layer.settings.min_scale, layer.settings.max_scale),
            )

            layer.add_instance(transform)
            placed += 1

        return placed

    def update_lod(
        self,
        camera_x: float,
        camera_y: float,
        camera_z: float
    ) -> dict[FoliageLODLevel, int]:
        """
        Update LOD levels for all instances based on camera position.

        Args:
            camera_x, camera_y, camera_z: Camera position

        Returns:
            Dictionary of LOD level counts
        """
        lod_counts: dict[FoliageLODLevel, int] = {level: 0 for level in FoliageLODLevel}

        for layer in self._layers.values():
            for inst in layer.instances:
                distance = inst.distance_to(camera_x, camera_y, camera_z)
                inst.current_lod = layer.lod_settings.get_lod_for_distance(distance)
                lod_counts[inst.current_lod] += 1

        return lod_counts

    def get_visible_instances(
        self,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        max_distance: Optional[float] = None
    ) -> list[FoliageInstance]:
        """
        Get all visible (non-culled) instances.

        Args:
            camera_x, camera_y, camera_z: Camera position
            max_distance: Optional maximum distance

        Returns:
            List of visible instances
        """
        visible = []

        for layer in self._layers.values():
            cull_distance = max_distance or layer.lod_settings.cull_distance

            for inst in layer.instances:
                distance = inst.distance_to(camera_x, camera_y, camera_z)
                if distance < cull_distance:
                    visible.append(inst)

        return visible

    def get_total_instance_count(self) -> int:
        """Get total number of instances across all layers."""
        return sum(layer.instance_count for layer in self._layers.values())

    def clear_all(self) -> None:
        """Clear all foliage instances from all layers."""
        for layer in self._layers.values():
            layer.clear_instances()
