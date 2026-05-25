"""
Grass-specific foliage system for the game engine World Layer.

Provides specialized grass rendering with:
- Procedural blade generation
- Chunk-based streaming
- Distance-based density scaling
- Wind animation support
- Compute-shader-like generation pattern
"""

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from .placement import Bounds, TerrainInterface
from .types import FoliageCategory, GrassType


@dataclass
class GrassSettings:
    """
    Grass rendering settings.

    Global settings that control grass appearance and performance.

    Attributes:
        density_scale: Global density multiplier
        distance_scale: Distance at which density starts reducing
        wind_sway_amount: Amplitude of wind sway
        wind_sway_speed: Speed of wind animation
        alpha_cutoff: Alpha test threshold for transparency
        cull_distance: Maximum render distance
        fade_distance: Distance over which grass fades out
    """

    density_scale: float = 1.0
    distance_scale: float = 1.0
    wind_sway_amount: float = 1.0
    wind_sway_speed: float = 1.0
    alpha_cutoff: float = 0.5
    cull_distance: float = 100.0
    fade_distance: float = 20.0

    def __post_init__(self) -> None:
        """Validate grass settings."""
        if self.density_scale < 0:
            raise ValueError("density_scale must be >= 0")
        if self.distance_scale <= 0:
            raise ValueError("distance_scale must be > 0")
        if self.wind_sway_amount < 0:
            raise ValueError("wind_sway_amount must be >= 0")
        if self.wind_sway_speed < 0:
            raise ValueError("wind_sway_speed must be >= 0")
        if not 0.0 <= self.alpha_cutoff <= 1.0:
            raise ValueError("alpha_cutoff must be between 0 and 1")
        if self.cull_distance <= 0:
            raise ValueError("cull_distance must be > 0")
        if self.fade_distance < 0:
            raise ValueError("fade_distance must be >= 0")


@dataclass
class GrassInstance:
    """
    Single grass blade instance.

    Represents one grass blade with position, orientation,
    and visual properties.

    Attributes:
        position: World position (x, y, z)
        rotation: Y-axis rotation in radians
        height: Blade height
        width: Blade width
        bend: Amount of blade bend
        color_blend: Blend factor between base and tip color
    """

    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: float = 0.0
    height: float = 0.3
    width: float = 0.05
    bend: float = 0.5
    color_blend: float = 0.5


@dataclass
class GrassChunk:
    """
    Chunk of grass instances for streaming.

    Groups grass instances by region for efficient loading/unloading.

    Attributes:
        bounds: Spatial bounds of this chunk
        chunk_x: X index in chunk grid
        chunk_z: Z index in chunk grid
        instance_count: Number of instances in this chunk
        instance_buffer: List of grass instances
        is_generated: Whether chunk has been generated
        is_visible: Whether chunk is currently visible
    """

    bounds: Bounds = field(default_factory=Bounds)
    chunk_x: int = 0
    chunk_z: int = 0
    instance_count: int = 0
    instance_buffer: List[GrassInstance] = field(default_factory=list)
    is_generated: bool = False
    is_visible: bool = True

    def get_center(self) -> Tuple[float, float]:
        """Get chunk center position."""
        return self.bounds.center

    def get_distance_to(self, x: float, z: float) -> float:
        """
        Get distance from point to chunk center.

        Args:
            x: X coordinate
            z: Z coordinate

        Returns:
            Distance to chunk center
        """
        cx, cz = self.bounds.center
        dx = x - cx
        dz = z - cz
        return math.sqrt(dx * dx + dz * dz)

    def clear(self) -> None:
        """Clear all instances."""
        self.instance_buffer.clear()
        self.instance_count = 0
        self.is_generated = False


class ProceduralGrass:
    """
    Procedural grass generation system.

    Generates grass instances based on terrain data and grass type
    settings. Uses compute-shader-like pattern for GPU generation.
    """

    __slots__ = ("_settings", "_terrain_weights", "_seed", "_noise_cache")

    def __init__(
        self,
        settings: GrassSettings,
        terrain_weights: Optional[List[int]] = None,
        seed: int = 0,
    ) -> None:
        """
        Initialize procedural grass generator.

        Args:
            settings: Grass rendering settings
            terrain_weights: Terrain layer indices that grow grass
            seed: Random seed for deterministic generation
        """
        self._settings = settings
        self._terrain_weights = terrain_weights if terrain_weights is not None else []
        self._seed = seed
        self._noise_cache: Dict[Tuple[int, int], float] = {}

    @property
    def settings(self) -> GrassSettings:
        """Get grass settings."""
        return self._settings

    @property
    def terrain_weights(self) -> List[int]:
        """Get terrain weight layers."""
        return self._terrain_weights

    def set_terrain_weights(self, weights: List[int]) -> None:
        """
        Set terrain layer weights.

        Args:
            weights: Layer indices that grow grass
        """
        self._terrain_weights = weights

    def _hash_position(self, x: float, z: float, channel: int = 0) -> float:
        """Hash position for deterministic randomness."""
        # Quantize to reduce hash calls
        qx = round(x * 100)
        qz = round(z * 100)
        key = (qx, qz, channel)

        data = f"{self._seed}:{qx}:{qz}:{channel}".encode()
        h = hashlib.md5(data).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def _sample_noise(self, x: float, z: float, scale: float = 5.0) -> float:
        """Sample noise value at position."""
        qx = int(x / scale)
        qz = int(z / scale)
        key = (qx, qz)

        if key not in self._noise_cache:
            self._noise_cache[key] = self._hash_position(qx * scale, qz * scale, 99)

        return self._noise_cache[key]

    def should_grow_grass(
        self, terrain: TerrainInterface, x: float, z: float
    ) -> bool:
        """
        Check if grass should grow at position.

        Args:
            terrain: Terrain data interface
            x: X coordinate
            z: Z coordinate

        Returns:
            True if grass should grow
        """
        # Check terrain layer
        if self._terrain_weights:
            layer = terrain.get_layer_at(x, z)
            if layer not in self._terrain_weights:
                return False

        # Check water
        if terrain.is_water_at(x, z):
            return False

        # Check slope (grass doesn't grow on steep slopes)
        normal = terrain.get_normal_at(x, z)
        if normal[1] < 0.7:  # ~45 degrees
            return False

        # Noise-based rejection for natural look
        noise = self._sample_noise(x, z)
        if noise < 0.3:
            return False

        return True

    def generate_for_chunk(
        self,
        terrain: TerrainInterface,
        chunk_bounds: Bounds,
        grass_type: GrassType,
    ) -> List[GrassInstance]:
        """
        Generate grass instances for a chunk.

        Uses compute-shader-like pattern for efficient parallel generation.

        Args:
            terrain: Terrain data interface
            chunk_bounds: Bounds of chunk to generate
            grass_type: Grass type settings

        Returns:
            List of grass instances
        """
        instances = []

        # Calculate effective density
        density = grass_type.density * self._settings.density_scale
        if density <= 0:
            return instances

        # Calculate spacing from density
        spacing = 1.0 / math.sqrt(density)

        # Generate grid of candidates
        x = chunk_bounds.min_x
        while x <= chunk_bounds.max_x:
            z = chunk_bounds.min_z
            while z <= chunk_bounds.max_z:
                # Add jitter
                jitter_x = (self._hash_position(x, z, 0) - 0.5) * spacing
                jitter_z = (self._hash_position(x, z, 1) - 0.5) * spacing
                px = x + jitter_x
                pz = z + jitter_z

                # Clamp to bounds
                px = max(chunk_bounds.min_x, min(chunk_bounds.max_x, px))
                pz = max(chunk_bounds.min_z, min(chunk_bounds.max_z, pz))

                # Check if grass should grow
                if self.should_grow_grass(terrain, px, pz):
                    # Get height
                    py = terrain.get_height_at(px, pz)

                    # Generate variation
                    rotation = self._hash_position(px, pz, 2) * math.pi * 2
                    height_var = 0.7 + self._hash_position(px, pz, 3) * 0.6
                    width_var = 0.7 + self._hash_position(px, pz, 4) * 0.6
                    bend_var = 0.3 + self._hash_position(px, pz, 5) * 0.7
                    color_var = self._hash_position(px, pz, 6)

                    instances.append(
                        GrassInstance(
                            position=(px, py, pz),
                            rotation=rotation,
                            height=grass_type.blade_height * height_var,
                            width=grass_type.blade_width * width_var,
                            bend=grass_type.blade_bend * bend_var,
                            color_blend=color_var,
                        )
                    )

                z += spacing
            x += spacing

        return instances

    def generate_instance_buffer(
        self, instances: List[GrassInstance]
    ) -> List[Dict]:
        """
        Convert instances to GPU buffer format.

        Args:
            instances: List of grass instances

        Returns:
            List of buffer entries for GPU
        """
        buffer = []
        for inst in instances:
            buffer.append(
                {
                    "position": inst.position,
                    "rotation": inst.rotation,
                    "height": inst.height,
                    "width": inst.width,
                    "bend": inst.bend,
                    "color_blend": inst.color_blend,
                }
            )
        return buffer


class LandscapeGrass:
    """
    Landscape grass system with chunk-based streaming.

    Manages grass generation, streaming, and rendering across
    large landscapes.
    """

    __slots__ = (
        "_grass_types",
        "_chunks",
        "_chunk_size",
        "_view_distance",
        "_generator",
        "_active_chunks",
        "_terrain",
    )

    def __init__(
        self,
        settings: GrassSettings,
        chunk_size: float = 32.0,
        view_distance: float = 100.0,
        seed: int = 0,
    ) -> None:
        """
        Initialize landscape grass system.

        Args:
            settings: Grass rendering settings
            chunk_size: Size of grass chunks
            view_distance: Maximum grass view distance
            seed: Random seed for generation
        """
        self._grass_types: List[GrassType] = []
        self._chunks: Dict[Tuple[int, int], GrassChunk] = {}
        self._chunk_size = chunk_size
        self._view_distance = min(view_distance, settings.cull_distance)
        self._generator = ProceduralGrass(settings, seed=seed)
        self._active_chunks: set = set()
        self._terrain: Optional[TerrainInterface] = None

    @property
    def chunk_size(self) -> float:
        """Get chunk size."""
        return self._chunk_size

    @property
    def view_distance(self) -> float:
        """Get view distance."""
        return self._view_distance

    @property
    def active_chunk_count(self) -> int:
        """Get number of active chunks."""
        return len(self._active_chunks)

    @property
    def total_chunk_count(self) -> int:
        """Get total generated chunk count."""
        return len(self._chunks)

    def set_terrain(self, terrain: TerrainInterface) -> None:
        """
        Set terrain interface for grass generation.

        Args:
            terrain: Terrain data interface
        """
        self._terrain = terrain

    def add_grass_type(self, grass_type: GrassType) -> None:
        """
        Add a grass type to render.

        Args:
            grass_type: Grass type to add
        """
        self._grass_types.append(grass_type)

    def remove_grass_type(self, type_id: str) -> bool:
        """
        Remove a grass type.

        Args:
            type_id: Type ID to remove

        Returns:
            True if type was removed
        """
        for i, gt in enumerate(self._grass_types):
            if gt.type_id == type_id:
                del self._grass_types[i]
                return True
        return False

    def set_terrain_weights(self, weights: List[int]) -> None:
        """
        Set terrain layers that grow grass.

        Args:
            weights: Terrain layer indices
        """
        self._generator.set_terrain_weights(weights)

    def _get_chunk_key(self, x: float, z: float) -> Tuple[int, int]:
        """Get chunk key for position."""
        return (
            int(math.floor(x / self._chunk_size)),
            int(math.floor(z / self._chunk_size)),
        )

    def generate_chunk(
        self,
        chunk_x: int,
        chunk_z: int,
        terrain: Optional[TerrainInterface] = None,
    ) -> GrassChunk:
        """
        Generate or get a grass chunk.

        Args:
            chunk_x: Chunk X index
            chunk_z: Chunk Z index
            terrain: Optional terrain interface override

        Returns:
            Generated or existing chunk
        """
        key = (chunk_x, chunk_z)

        # Return existing chunk if already generated
        if key in self._chunks and self._chunks[key].is_generated:
            return self._chunks[key]

        # Create chunk bounds
        bounds = Bounds(
            min_x=chunk_x * self._chunk_size,
            min_z=chunk_z * self._chunk_size,
            max_x=(chunk_x + 1) * self._chunk_size,
            max_z=(chunk_z + 1) * self._chunk_size,
        )

        # Create chunk
        chunk = GrassChunk(
            bounds=bounds,
            chunk_x=chunk_x,
            chunk_z=chunk_z,
        )

        # Use provided terrain or stored terrain
        t = terrain if terrain is not None else self._terrain
        if t is None:
            # No terrain, create empty chunk
            chunk.is_generated = True
            self._chunks[key] = chunk
            return chunk

        # Generate grass for each type
        all_instances = []
        for grass_type in self._grass_types:
            instances = self._generator.generate_for_chunk(t, bounds, grass_type)
            all_instances.extend(instances)

        chunk.instance_buffer = all_instances
        chunk.instance_count = len(all_instances)
        chunk.is_generated = True

        self._chunks[key] = chunk
        return chunk

    def update(
        self,
        camera_position: Tuple[float, float, float],
        terrain: Optional[TerrainInterface] = None,
    ) -> None:
        """
        Update grass streaming based on camera position.

        Generates nearby chunks and culls distant ones.

        Args:
            camera_position: Camera world position
            terrain: Optional terrain interface override
        """
        cam_x, _, cam_z = camera_position
        t = terrain if terrain is not None else self._terrain

        # Calculate chunk range
        chunk_radius = int(math.ceil(self._view_distance / self._chunk_size))
        center_key = self._get_chunk_key(cam_x, cam_z)

        # Track which chunks should be active
        new_active: set = set()

        # Generate/activate nearby chunks
        for dx in range(-chunk_radius, chunk_radius + 1):
            for dz in range(-chunk_radius, chunk_radius + 1):
                cx = center_key[0] + dx
                cz = center_key[1] + dz
                key = (cx, cz)

                # Check if chunk is within view distance
                chunk_center_x = (cx + 0.5) * self._chunk_size
                chunk_center_z = (cz + 0.5) * self._chunk_size
                dist = math.sqrt(
                    (chunk_center_x - cam_x) ** 2 + (chunk_center_z - cam_z) ** 2
                )

                if dist <= self._view_distance + self._chunk_size:
                    # Generate if needed
                    if key not in self._chunks or not self._chunks[key].is_generated:
                        self.generate_chunk(cx, cz, t)

                    if key in self._chunks:
                        self._chunks[key].is_visible = True
                        new_active.add(key)

        # Mark out-of-range chunks as not visible
        for key in self._active_chunks - new_active:
            if key in self._chunks:
                self._chunks[key].is_visible = False

        self._active_chunks = new_active

    def get_render_chunks(self, camera_position: Tuple[float, float, float]) -> List[GrassChunk]:
        """
        Get visible chunks sorted by distance.

        Args:
            camera_position: Camera position for distance sorting

        Returns:
            List of visible chunks
        """
        cam_x, _, cam_z = camera_position
        visible = []

        for key in self._active_chunks:
            if key in self._chunks and self._chunks[key].is_visible:
                visible.append(self._chunks[key])

        # Sort by distance (nearest first)
        visible.sort(key=lambda c: c.get_distance_to(cam_x, cam_z))

        return visible

    def get_instance_buffer(
        self, camera_position: Tuple[float, float, float]
    ) -> List[Dict]:
        """
        Get combined instance buffer for rendering.

        Args:
            camera_position: Camera position

        Returns:
            Combined instance buffer from all visible chunks
        """
        chunks = self.get_render_chunks(camera_position)
        buffer = []

        for chunk in chunks:
            buffer.extend(self._generator.generate_instance_buffer(chunk.instance_buffer))

        return buffer

    def get_total_instances(self) -> int:
        """Get total instance count across all chunks."""
        return sum(c.instance_count for c in self._chunks.values())

    def get_visible_instances(self) -> int:
        """Get visible instance count."""
        total = 0
        for key in self._active_chunks:
            if key in self._chunks:
                total += self._chunks[key].instance_count
        return total

    def clear_chunk(self, chunk_x: int, chunk_z: int) -> bool:
        """
        Clear a specific chunk.

        Args:
            chunk_x: Chunk X index
            chunk_z: Chunk Z index

        Returns:
            True if chunk was cleared
        """
        key = (chunk_x, chunk_z)
        if key in self._chunks:
            self._chunks[key].clear()
            self._active_chunks.discard(key)
            return True
        return False

    def clear_all(self) -> None:
        """Clear all chunks."""
        self._chunks.clear()
        self._active_chunks.clear()

    def unload_distant_chunks(
        self, camera_position: Tuple[float, float, float], unload_distance: float
    ) -> int:
        """
        Unload chunks beyond distance to free memory.

        Args:
            camera_position: Camera position
            unload_distance: Distance beyond which to unload

        Returns:
            Number of chunks unloaded
        """
        cam_x, _, cam_z = camera_position
        to_unload = []

        for key, chunk in self._chunks.items():
            dist = chunk.get_distance_to(cam_x, cam_z)
            if dist > unload_distance:
                to_unload.append(key)

        for key in to_unload:
            del self._chunks[key]
            self._active_chunks.discard(key)

        return len(to_unload)


class GrassRenderer:
    """
    Grass rendering interface.

    Provides render data preparation and shader parameters
    for grass rendering.
    """

    __slots__ = ("_landscape_grass", "_settings", "_wind_time")

    def __init__(
        self, landscape_grass: LandscapeGrass, settings: GrassSettings
    ) -> None:
        """
        Initialize grass renderer.

        Args:
            landscape_grass: Landscape grass system
            settings: Rendering settings
        """
        self._landscape_grass = landscape_grass
        self._settings = settings
        self._wind_time = 0.0

    def update(self, delta_time: float) -> None:
        """
        Update renderer state.

        Args:
            delta_time: Time since last update
        """
        self._wind_time += delta_time * self._settings.wind_sway_speed

    def get_shader_params(self) -> Dict:
        """
        Get shader parameters for grass rendering.

        Returns:
            Dictionary of shader parameters
        """
        return {
            "wind_time": self._wind_time,
            "wind_sway_amount": self._settings.wind_sway_amount,
            "alpha_cutoff": self._settings.alpha_cutoff,
            "fade_start": self._settings.cull_distance - self._settings.fade_distance,
            "fade_end": self._settings.cull_distance,
        }

    def get_render_data(
        self, camera_position: Tuple[float, float, float]
    ) -> Tuple[List[Dict], Dict]:
        """
        Get complete render data.

        Args:
            camera_position: Camera position

        Returns:
            Tuple of (instance buffer, shader params)
        """
        instances = self._landscape_grass.get_instance_buffer(camera_position)
        params = self.get_shader_params()
        return instances, params
