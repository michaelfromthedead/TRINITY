"""
Visibility Buffer Pipeline for GPU-driven rendering.

Implements a Nanite-style visibility buffer rendering pipeline that
decouples geometry from shading for optimal GPU utilization.

Pipeline:
1. Visibility Buffer Pass - Write triangle ID + instance ID + depth per pixel
2. Material Sorting Pass - Group pixels by material for coherent shading
3. Deferred Texturing Pass - Fetch vertex data and shade pixels

References:
- RENDERING_CONTEXT.md Section 6.2 Visibility Buffer Pipeline
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Optional, Sequence

from engine.rendering.gpu_driven.culling import Vec3, Vec4


# =============================================================================
# VISIBILITY BUFFER CONSTANTS
# =============================================================================


class VisibilityBufferConstants:
    """Constants for visibility buffer pipeline."""
    # Default tile size for material classification
    DEFAULT_TILE_SIZE: int = 8

    # Default screen dimensions
    DEFAULT_SCREEN_WIDTH: int = 1920
    DEFAULT_SCREEN_HEIGHT: int = 1080


# =============================================================================
# VISIBILITY BUFFER DATA FORMATS
# =============================================================================


class VisibilityBufferFormat(IntEnum):
    """Formats for visibility buffer storage."""
    # 32-bit format: 12-bit triangle ID + 20-bit instance ID
    R32_UINT = auto()

    # 64-bit format: 24-bit triangle ID + 24-bit instance ID + 16-bit flags
    RG32_UINT = auto()

    # High precision: Full 32-bit triangle ID + 32-bit instance ID
    RG32_UINT_FULL = auto()


@dataclass(slots=True)
class VisibilityData:
    """
    Per-pixel visibility data.

    Encodes which triangle of which instance is visible at each pixel.
    """
    triangle_id: int = 0  # Triangle index within the mesh
    instance_id: int = 0  # Instance index in the scene
    meshlet_id: int = 0  # Meshlet index (for meshlet rendering)
    depth: float = 1.0  # Normalized depth value
    valid: bool = False  # Whether this pixel has valid geometry

    def pack_32bit(self) -> int:
        """Pack to 32-bit format (12-bit tri + 20-bit instance)."""
        return ((self.triangle_id & 0xFFF) << 20) | (self.instance_id & 0xFFFFF)

    @classmethod
    def unpack_32bit(cls, packed: int) -> "VisibilityData":
        """Unpack from 32-bit format."""
        return cls(
            triangle_id=(packed >> 20) & 0xFFF,
            instance_id=packed & 0xFFFFF,
            valid=packed != 0,
        )

    def pack_64bit(self) -> tuple[int, int]:
        """Pack to 64-bit format (24-bit tri + 24-bit instance + 16-bit flags)."""
        rg0 = ((self.triangle_id & 0xFFFFFF) << 8) | (self.instance_id >> 16)
        rg1 = ((self.instance_id & 0xFFFF) << 16) | (self.meshlet_id & 0xFFFF)
        return rg0, rg1

    @classmethod
    def unpack_64bit(cls, rg0: int, rg1: int) -> "VisibilityData":
        """Unpack from 64-bit format."""
        tri_id = (rg0 >> 8) & 0xFFFFFF
        inst_id = ((rg0 & 0xFF) << 16) | ((rg1 >> 16) & 0xFFFF)
        meshlet_id = rg1 & 0xFFFF
        return cls(
            triangle_id=tri_id,
            instance_id=inst_id,
            meshlet_id=meshlet_id,
            valid=True,
        )


# =============================================================================
# VISIBILITY BUFFER
# =============================================================================


@dataclass
class VisibilityBufferConfig:
    """Configuration for visibility buffer."""
    width: int = VisibilityBufferConstants.DEFAULT_SCREEN_WIDTH
    height: int = VisibilityBufferConstants.DEFAULT_SCREEN_HEIGHT
    format: VisibilityBufferFormat = VisibilityBufferFormat.R32_UINT
    enable_depth: bool = True
    enable_motion_vectors: bool = False


class VisibilityBuffer:
    """
    GPU visibility buffer storing triangle/instance IDs per pixel.

    The visibility buffer replaces the traditional G-Buffer approach
    by storing minimal data during rasterization and deferring
    material evaluation to a compute pass.

    Benefits:
    - Decouple geometry complexity from shading complexity
    - Better GPU occupancy for material shading
    - Efficient handling of many small triangles (meshlets)
    - Natural support for virtual geometry (Nanite-style)
    """

    def __init__(self, config: Optional[VisibilityBufferConfig] = None) -> None:
        self._config = config or VisibilityBufferConfig()

        # Visibility data (CPU simulation of GPU buffer)
        self._visibility_data: list[list[int]] = []
        self._depth_buffer: list[list[float]] = []

        # Motion vectors (optional)
        self._motion_vectors: list[list[tuple[float, float]]] = []

        self._initialize_buffers()

    def _initialize_buffers(self) -> None:
        """Initialize buffer storage."""
        w, h = self._config.width, self._config.height

        # Visibility buffer (packed triangle + instance IDs)
        self._visibility_data = [[0 for _ in range(w)] for _ in range(h)]

        # Depth buffer
        if self._config.enable_depth:
            self._depth_buffer = [[1.0 for _ in range(w)] for _ in range(h)]

        # Motion vectors
        if self._config.enable_motion_vectors:
            self._motion_vectors = [[(0.0, 0.0) for _ in range(w)] for _ in range(h)]

    @property
    def config(self) -> VisibilityBufferConfig:
        return self._config

    @property
    def width(self) -> int:
        return self._config.width

    @property
    def height(self) -> int:
        return self._config.height

    def clear(self) -> None:
        """Clear the visibility buffer for a new frame."""
        w, h = self._config.width, self._config.height
        for y in range(h):
            for x in range(w):
                self._visibility_data[y][x] = 0
                if self._config.enable_depth:
                    self._depth_buffer[y][x] = 1.0
                if self._config.enable_motion_vectors:
                    self._motion_vectors[y][x] = (0.0, 0.0)

    def write_pixel(
        self,
        x: int,
        y: int,
        triangle_id: int,
        instance_id: int,
        depth: float,
        motion_x: float = 0.0,
        motion_y: float = 0.0,
    ) -> bool:
        """
        Write visibility data to a pixel (with depth test).

        Returns True if the pixel was written (passed depth test).
        """
        if x < 0 or x >= self._config.width or y < 0 or y >= self._config.height:
            return False

        # Depth test
        if self._config.enable_depth:
            if depth >= self._depth_buffer[y][x]:
                return False
            self._depth_buffer[y][x] = depth

        # Pack and write visibility data
        vis = VisibilityData(triangle_id=triangle_id, instance_id=instance_id)
        self._visibility_data[y][x] = vis.pack_32bit()

        # Write motion vectors
        if self._config.enable_motion_vectors:
            self._motion_vectors[y][x] = (motion_x, motion_y)

        return True

    def read_pixel(self, x: int, y: int) -> VisibilityData:
        """Read visibility data from a pixel."""
        if x < 0 or x >= self._config.width or y < 0 or y >= self._config.height:
            return VisibilityData(valid=False)

        packed = self._visibility_data[y][x]
        if packed == 0:
            return VisibilityData(valid=False)

        vis = VisibilityData.unpack_32bit(packed)
        if self._config.enable_depth:
            vis.depth = self._depth_buffer[y][x]
        return vis

    def read_depth(self, x: int, y: int) -> float:
        """Read depth value from a pixel."""
        if not self._config.enable_depth:
            return 1.0
        if x < 0 or x >= self._config.width or y < 0 or y >= self._config.height:
            return 1.0
        return self._depth_buffer[y][x]

    def get_depth_buffer(self) -> list[list[float]]:
        """Get the depth buffer (for HZB building)."""
        return self._depth_buffer

    def get_raw_buffer(self) -> list[list[int]]:
        """Get the raw visibility buffer data."""
        return self._visibility_data


# =============================================================================
# MATERIAL TILE CLASSIFICATION
# =============================================================================


@dataclass(slots=True)
class MaterialTile:
    """A tile of pixels sharing material properties."""
    tile_x: int  # Tile X coordinate
    tile_y: int  # Tile Y coordinate
    material_id: int  # Material for this tile
    pixel_count: int  # Number of pixels using this material in tile
    pixel_mask: int  # Bitmask of pixels using this material (for 8x8 tiles)


class MaterialTileClassifier:
    """
    Classifies visibility buffer pixels by material for efficient shading.

    Tiles the screen and identifies which materials are present in each tile.
    This enables coherent material shading with minimal divergence.
    """

    def __init__(
        self,
        tile_size: int = VisibilityBufferConstants.DEFAULT_TILE_SIZE,
        screen_width: int = VisibilityBufferConstants.DEFAULT_SCREEN_WIDTH,
        screen_height: int = VisibilityBufferConstants.DEFAULT_SCREEN_HEIGHT,
    ) -> None:
        self._tile_size = tile_size
        self._screen_width = screen_width
        self._screen_height = screen_height

        # Number of tiles
        self._tiles_x = (screen_width + tile_size - 1) // tile_size
        self._tiles_y = (screen_height + tile_size - 1) // tile_size

        # Material lookup table (instance_id -> material_id)
        self._material_lut: dict[int, int] = {}

        # Classified tiles by material
        self._material_tiles: dict[int, list[MaterialTile]] = {}

    @property
    def tile_size(self) -> int:
        return self._tile_size

    @property
    def tiles_x(self) -> int:
        return self._tiles_x

    @property
    def tiles_y(self) -> int:
        return self._tiles_y

    def set_material_lut(self, lut: dict[int, int]) -> None:
        """Set the instance-to-material lookup table."""
        self._material_lut = lut

    def classify(self, visibility_buffer: VisibilityBuffer) -> dict[int, list[MaterialTile]]:
        """
        Classify visibility buffer pixels by material.

        Returns:
            Dictionary mapping material_id to list of tiles using that material
        """
        self._material_tiles.clear()

        for ty in range(self._tiles_y):
            for tx in range(self._tiles_x):
                # Analyze pixels in this tile
                tile_materials = self._analyze_tile(visibility_buffer, tx, ty)

                # Add tiles to material groups
                for mat_id, (pixel_count, pixel_mask) in tile_materials.items():
                    if mat_id not in self._material_tiles:
                        self._material_tiles[mat_id] = []

                    self._material_tiles[mat_id].append(MaterialTile(
                        tile_x=tx,
                        tile_y=ty,
                        material_id=mat_id,
                        pixel_count=pixel_count,
                        pixel_mask=pixel_mask,
                    ))

        return self._material_tiles

    def _analyze_tile(
        self,
        visibility_buffer: VisibilityBuffer,
        tile_x: int,
        tile_y: int,
    ) -> dict[int, tuple[int, int]]:
        """
        Analyze a tile and return materials present.

        Returns:
            Dictionary of material_id -> (pixel_count, pixel_mask)
        """
        materials: dict[int, tuple[int, int]] = {}

        base_x = tile_x * self._tile_size
        base_y = tile_y * self._tile_size

        for local_y in range(self._tile_size):
            for local_x in range(self._tile_size):
                px = base_x + local_x
                py = base_y + local_y

                vis = visibility_buffer.read_pixel(px, py)
                if not vis.valid:
                    continue

                # Look up material for this instance
                mat_id = self._material_lut.get(vis.instance_id, 0)

                # Update material stats
                pixel_idx = local_y * self._tile_size + local_x
                pixel_bit = 1 << (pixel_idx % 64)  # For 8x8 tile = 64 pixels

                if mat_id in materials:
                    count, mask = materials[mat_id]
                    materials[mat_id] = (count + 1, mask | pixel_bit)
                else:
                    materials[mat_id] = (1, pixel_bit)

        return materials


# =============================================================================
# VISIBILITY BUFFER PASS
# =============================================================================


class VisibilityBufferPass:
    """
    Render pass that writes to the visibility buffer.

    This pass performs minimal work during rasterization:
    - Compute barycentric coordinates for interpolation
    - Write triangle ID and instance ID
    - Write depth for depth testing
    """

    def __init__(self, visibility_buffer: VisibilityBuffer) -> None:
        self._visibility_buffer = visibility_buffer
        self._current_instance_id: int = 0
        self._current_meshlet_id: int = 0

    @property
    def visibility_buffer(self) -> VisibilityBuffer:
        return self._visibility_buffer

    def begin(self) -> None:
        """Begin visibility buffer pass."""
        self._visibility_buffer.clear()

    def set_instance(self, instance_id: int, meshlet_id: int = 0) -> None:
        """Set the current instance being rendered."""
        self._current_instance_id = instance_id
        self._current_meshlet_id = meshlet_id

    def rasterize_triangle(
        self,
        triangle_id: int,
        v0: tuple[float, float, float],  # Screen-space position (x, y, depth)
        v1: tuple[float, float, float],
        v2: tuple[float, float, float],
    ) -> int:
        """
        Rasterize a triangle to the visibility buffer.

        Args:
            triangle_id: ID of the triangle within the mesh
            v0, v1, v2: Screen-space positions (x, y, depth) of triangle vertices

        Returns:
            Number of pixels written
        """
        # Compute bounding box
        min_x = max(0, int(min(v0[0], v1[0], v2[0])))
        max_x = min(self._visibility_buffer.width - 1, int(max(v0[0], v1[0], v2[0])))
        min_y = max(0, int(min(v0[1], v1[1], v2[1])))
        max_y = min(self._visibility_buffer.height - 1, int(max(v0[1], v1[1], v2[1])))

        if min_x > max_x or min_y > max_y:
            return 0

        # Edge equations for rasterization
        def edge_function(a: tuple[float, ...], b: tuple[float, ...], p: tuple[float, float]) -> float:
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        pixels_written = 0

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = (x + 0.5, y + 0.5)

                # Barycentric coordinates
                w0 = edge_function(v1, v2, p)
                w1 = edge_function(v2, v0, p)
                w2 = edge_function(v0, v1, p)

                # Check if point is inside triangle
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Normalize barycentric coordinates
                    area = w0 + w1 + w2
                    if area > 0:
                        w0 /= area
                        w1 /= area
                        w2 /= area

                        # Interpolate depth
                        depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]

                        # Write to visibility buffer
                        if self._visibility_buffer.write_pixel(
                            x, y,
                            triangle_id,
                            self._current_instance_id,
                            depth,
                        ):
                            pixels_written += 1

        return pixels_written

    def end(self) -> None:
        """
        End visibility buffer pass.

        This method serves as a synchronization point in the rendering pipeline.
        In a GPU implementation, this would typically:
        - Flush any pending visibility writes
        - Signal completion of the visibility pass
        - Prepare the buffer for subsequent material sorting/shading passes

        Currently a no-op as the CPU simulation writes are synchronous.
        """
        # No explicit action needed for CPU simulation
        # GPU implementation would handle buffer synchronization here


# =============================================================================
# DEFERRED TEXTURING PASS
# =============================================================================


@dataclass
class ShadingInput:
    """Input data for shading a pixel."""
    # Position
    world_position: Vec3 = field(default_factory=Vec3)
    screen_position: tuple[int, int] = (0, 0)

    # Geometry
    normal: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))
    tangent: Vec3 = field(default_factory=lambda: Vec3(1.0, 0.0, 0.0))

    # Material
    material_id: int = 0
    uv: tuple[float, float] = (0.0, 0.0)

    # IDs
    triangle_id: int = 0
    instance_id: int = 0


@dataclass
class ShadingOutput:
    """Output from shading a pixel."""
    color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    emission: tuple[float, float, float] = (0.0, 0.0, 0.0)
    motion_vector: tuple[float, float] = (0.0, 0.0)


# Type alias for material shading function
MaterialShader = Callable[[ShadingInput], ShadingOutput]


class DeferredTexturingPass:
    """
    Deferred texturing pass that shades visibility buffer pixels.

    This pass:
    1. Reads triangle/instance IDs from visibility buffer
    2. Fetches vertex data and computes interpolated attributes
    3. Calls material shader for each pixel
    4. Writes final color to render target
    """

    def __init__(
        self,
        visibility_buffer: VisibilityBuffer,
        tile_classifier: Optional[MaterialTileClassifier] = None,
    ) -> None:
        self._visibility_buffer = visibility_buffer
        self._tile_classifier = tile_classifier or MaterialTileClassifier(
            screen_width=visibility_buffer.width,
            screen_height=visibility_buffer.height,
        )

        # Registered material shaders
        self._material_shaders: dict[int, MaterialShader] = {}

        # Geometry fetch callbacks
        self._vertex_fetch: Optional[Callable[[int, int, int], ShadingInput]] = None

        # Output buffer
        self._output_buffer: list[list[ShadingOutput]] = []

    def set_vertex_fetch_callback(
        self,
        callback: Callable[[int, int, int], ShadingInput],
    ) -> None:
        """
        Set callback for fetching vertex data.

        Callback signature: (instance_id, triangle_id, barycentric_coords) -> ShadingInput
        """
        self._vertex_fetch = callback

    def register_material_shader(
        self,
        material_id: int,
        shader: MaterialShader,
    ) -> None:
        """Register a shader function for a material."""
        self._material_shaders[material_id] = shader

    def set_material_lut(self, lut: dict[int, int]) -> None:
        """Set instance-to-material lookup table."""
        self._tile_classifier.set_material_lut(lut)

    def execute(self) -> list[list[ShadingOutput]]:
        """
        Execute deferred texturing pass.

        Returns:
            2D array of shading outputs
        """
        # Initialize output buffer
        w = self._visibility_buffer.width
        h = self._visibility_buffer.height
        self._output_buffer = [
            [ShadingOutput() for _ in range(w)]
            for _ in range(h)
        ]

        # Classify tiles by material
        material_tiles = self._tile_classifier.classify(self._visibility_buffer)

        # Shade each material group
        for material_id, tiles in material_tiles.items():
            shader = self._material_shaders.get(material_id)
            if shader is None:
                continue

            self._shade_material_tiles(material_id, tiles, shader)

        return self._output_buffer

    def _shade_material_tiles(
        self,
        material_id: int,
        tiles: list[MaterialTile],
        shader: MaterialShader,
    ) -> None:
        """Shade all tiles using a specific material."""
        tile_size = self._tile_classifier.tile_size

        for tile in tiles:
            base_x = tile.tile_x * tile_size
            base_y = tile.tile_y * tile_size

            for local_y in range(tile_size):
                for local_x in range(tile_size):
                    px = base_x + local_x
                    py = base_y + local_y

                    if px >= self._visibility_buffer.width:
                        continue
                    if py >= self._visibility_buffer.height:
                        continue

                    vis = self._visibility_buffer.read_pixel(px, py)
                    if not vis.valid:
                        continue

                    # Fetch shading input
                    shading_input = self._fetch_shading_input(
                        px, py,
                        vis.instance_id,
                        vis.triangle_id,
                        material_id,
                    )

                    # Execute shader
                    output = shader(shading_input)

                    # Write output
                    self._output_buffer[py][px] = output

    def _fetch_shading_input(
        self,
        px: int,
        py: int,
        instance_id: int,
        triangle_id: int,
        material_id: int,
    ) -> ShadingInput:
        """Fetch shading input for a pixel."""
        if self._vertex_fetch is not None:
            shading_input = self._vertex_fetch(instance_id, triangle_id, 0)
        else:
            shading_input = ShadingInput()

        shading_input.screen_position = (px, py)
        shading_input.triangle_id = triangle_id
        shading_input.instance_id = instance_id
        shading_input.material_id = material_id

        return shading_input


# =============================================================================
# MATERIAL SORTING PASS
# =============================================================================


class MaterialSortingPass:
    """
    Sorts visibility buffer pixels by material for coherent shading.

    Creates a sorted list of pixel coordinates grouped by material,
    enabling efficient SIMD/SIMT shading with minimal divergence.
    """

    def __init__(self, visibility_buffer: VisibilityBuffer) -> None:
        self._visibility_buffer = visibility_buffer
        self._material_lut: dict[int, int] = {}

        # Sorted pixel lists by material
        self._sorted_pixels: dict[int, list[tuple[int, int]]] = {}

    def set_material_lut(self, lut: dict[int, int]) -> None:
        """Set instance-to-material lookup table."""
        self._material_lut = lut

    def sort(self) -> dict[int, list[tuple[int, int]]]:
        """
        Sort pixels by material.

        Returns:
            Dictionary mapping material_id to list of (x, y) pixel coordinates
        """
        self._sorted_pixels.clear()

        w = self._visibility_buffer.width
        h = self._visibility_buffer.height

        for y in range(h):
            for x in range(w):
                vis = self._visibility_buffer.read_pixel(x, y)
                if not vis.valid:
                    continue

                mat_id = self._material_lut.get(vis.instance_id, 0)

                if mat_id not in self._sorted_pixels:
                    self._sorted_pixels[mat_id] = []

                self._sorted_pixels[mat_id].append((x, y))

        return self._sorted_pixels

    def get_pixel_count(self, material_id: int) -> int:
        """Get number of pixels using a material."""
        return len(self._sorted_pixels.get(material_id, []))

    def get_total_pixel_count(self) -> int:
        """Get total number of visible pixels."""
        return sum(len(pixels) for pixels in self._sorted_pixels.values())


# =============================================================================
# VISIBILITY BUFFER PIPELINE
# =============================================================================


class VisibilityBufferPipeline:
    """
    Complete visibility buffer rendering pipeline.

    Orchestrates:
    1. Visibility buffer pass (geometry to IDs)
    2. Material sorting (group by material)
    3. Deferred texturing (shade pixels)
    """

    def __init__(self, config: Optional[VisibilityBufferConfig] = None) -> None:
        self._config = config or VisibilityBufferConfig()
        self._visibility_buffer = VisibilityBuffer(self._config)
        self._visibility_pass = VisibilityBufferPass(self._visibility_buffer)
        self._tile_classifier = MaterialTileClassifier(
            screen_width=self._config.width,
            screen_height=self._config.height,
        )
        self._deferred_pass = DeferredTexturingPass(
            self._visibility_buffer,
            self._tile_classifier,
        )
        self._sorting_pass = MaterialSortingPass(self._visibility_buffer)

    @property
    def visibility_buffer(self) -> VisibilityBuffer:
        return self._visibility_buffer

    @property
    def visibility_pass(self) -> VisibilityBufferPass:
        return self._visibility_pass

    @property
    def deferred_pass(self) -> DeferredTexturingPass:
        return self._deferred_pass

    @property
    def sorting_pass(self) -> MaterialSortingPass:
        return self._sorting_pass

    def set_material_lut(self, lut: dict[int, int]) -> None:
        """Set instance-to-material lookup for all passes."""
        self._tile_classifier.set_material_lut(lut)
        self._deferred_pass.set_material_lut(lut)
        self._sorting_pass.set_material_lut(lut)

    def register_material_shader(
        self,
        material_id: int,
        shader: MaterialShader,
    ) -> None:
        """Register a material shader."""
        self._deferred_pass.register_material_shader(material_id, shader)

    def begin_frame(self) -> None:
        """Begin a new frame."""
        self._visibility_pass.begin()

    def end_visibility_pass(self) -> None:
        """End visibility pass and prepare for shading."""
        self._visibility_pass.end()

    def execute_shading(self) -> list[list[ShadingOutput]]:
        """Execute deferred texturing pass."""
        return self._deferred_pass.execute()

    def get_depth_buffer_for_hzb(self) -> list[list[float]]:
        """Get depth buffer for HZB building."""
        return self._visibility_buffer.get_depth_buffer()


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Constants
    "VisibilityBufferConstants",
    # Data formats
    "VisibilityBufferFormat",
    "VisibilityData",
    # Buffer
    "VisibilityBufferConfig",
    "VisibilityBuffer",
    # Tile classification
    "MaterialTile",
    "MaterialTileClassifier",
    # Passes
    "VisibilityBufferPass",
    "ShadingInput",
    "ShadingOutput",
    "MaterialShader",
    "DeferredTexturingPass",
    "MaterialSortingPass",
    # Pipeline
    "VisibilityBufferPipeline",
]
