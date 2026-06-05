"""
TRINITY Architecture SDF Module (T-DEMO-4.7 and T-DEMO-4.8)

This module provides procedural building generation using Signed Distance Fields.
It includes:
- BuildingSDF: Parametric building from combined primitives
- CityBlockSDF: Domain repetition for procedural city generation

Following the Trinity metaclass pattern with:
- Mirror: Introspection for field access and type information
- Tracker: Dirty tracking for cache invalidation

Reference:
- Rust primitives: crates/renderer-backend/src/sdf_primitives.rs
- Rust combinators: crates/renderer-backend/src/sdf_combinators.rs
- Rust domain ops: crates/renderer-backend/src/sdf_domain_ops.rs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, ClassVar
import threading

from .sdf_ast import (
    SDFNode,
    SDFNodeMeta,
    PrimitiveNode,
    CombinatorNode,
    Vec3,
    BoxNode,
    PyramidNode,
    SubtractionNode,
    UnionNode,
    SmoothUnionNode,
    Mirror,
    Tracker,
)


__all__ = [
    "RoofStyle",
    "BuildingSDF",
    "CityBlockSDF",
    "cell_hash",
    "hash_to_float",
]


# =============================================================================
# Constants
# =============================================================================

# Default building parameters
DEFAULT_WIDTH = 10.0
DEFAULT_HEIGHT = 15.0
DEFAULT_DEPTH = 8.0
DEFAULT_FLOORS = 3
DEFAULT_WINDOWS_PER_FLOOR = 4
DEFAULT_WINDOW_WIDTH = 0.8
DEFAULT_WINDOW_HEIGHT = 1.2
DEFAULT_WINDOW_DEPTH = 0.15
DEFAULT_DOOR_WIDTH = 1.5
DEFAULT_DOOR_HEIGHT = 2.5
DEFAULT_DOOR_DEPTH = 0.2
DEFAULT_ROOF_HEIGHT = 2.0
DEFAULT_EDGE_TRIM_SIZE = 0.3

# City block parameters
DEFAULT_CELL_SIZE = 20.0
DEFAULT_STREET_WIDTH = 5.0
DEFAULT_HEIGHT_VARIATION = 0.4
DEFAULT_MIN_FLOORS = 2
DEFAULT_MAX_FLOORS = 8


# =============================================================================
# Roof Style Enumeration
# =============================================================================

class RoofStyle(Enum):
    """Available roof styles for buildings."""
    FLAT = auto()
    PITCHED = auto()  # Triangular pitched roof
    DOME = auto()     # Hemispherical dome


# =============================================================================
# Cell Hash Functions for Deterministic Variation
# =============================================================================

def cell_hash(cell_x: int, cell_z: int, seed: int = 0) -> int:
    """
    Compute deterministic hash for a cell position.

    Uses a simple hash function to generate consistent values
    for the same cell coordinates.

    Args:
        cell_x: Cell X coordinate
        cell_z: Cell Z coordinate
        seed: Optional seed for variation

    Returns:
        Integer hash value
    """
    # Simple hash combining based on standard practices
    h = seed
    h = h ^ (cell_x * 374761393)
    h = ((h << 17) | (h >> 15)) & 0xFFFFFFFF
    h = h ^ (cell_z * 668265263)
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = h * 1274126177
    h = h & 0xFFFFFFFF
    return h


def hash_to_float(h: int, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Convert hash to float in range [min_val, max_val].

    Args:
        h: Integer hash value
        min_val: Minimum output value
        max_val: Maximum output value

    Returns:
        Float value in specified range
    """
    # Normalize to [0, 1] then scale
    normalized = (h & 0xFFFFFFFF) / 0xFFFFFFFF
    return min_val + normalized * (max_val - min_val)


# =============================================================================
# BuildingSDF - T-DEMO-4.7
# =============================================================================

class BuildingSDF(SDFNode):
    """
    Parametric building SDF composed of primitives.

    Creates a building with:
    - Main structure: Box primitive
    - Windows: Subtracted boxes in grid pattern
    - Roof: Configurable style (flat, pitched, dome)
    - Door: Single subtracted box at ground level

    The building is centered at the origin with:
    - X axis: width direction
    - Y axis: height direction (base at y=0)
    - Z axis: depth direction

    Attributes:
        width: Building width (X extent)
        height: Building height (Y extent, excluding roof)
        depth: Building depth (Z extent)
        floors: Number of floors
        windows_per_floor: Windows per floor per side
        roof_style: Type of roof
        roof_height: Height of pitched/dome roof
        window_width: Width of each window
        window_height: Height of each window
        window_depth: Carving depth for windows
        door_width: Width of entry door
        door_height: Height of entry door
        door_depth: Carving depth for door
        edge_trim_size: Size of edge trim for flat roofs
        material_id: Material ID for rendering
    """

    __slots__ = (
        "width", "height", "depth", "floors", "windows_per_floor",
        "roof_style", "roof_height", "window_width", "window_height",
        "window_depth", "door_width", "door_height", "door_depth",
        "edge_trim_size", "material_id", "_cached_sdf", "_wgsl_cache"
    )

    # Type annotations for metaclass
    width: float
    height: float
    depth: float
    floors: int
    windows_per_floor: int
    roof_style: RoofStyle
    roof_height: float
    window_width: float
    window_height: float
    window_depth: float
    door_width: float
    door_height: float
    door_depth: float
    edge_trim_size: float
    material_id: int

    def __init__(
        self,
        width: float = DEFAULT_WIDTH,
        height: float = DEFAULT_HEIGHT,
        depth: float = DEFAULT_DEPTH,
        floors: int = DEFAULT_FLOORS,
        windows_per_floor: int = DEFAULT_WINDOWS_PER_FLOOR,
        roof_style: RoofStyle = RoofStyle.FLAT,
        roof_height: float = DEFAULT_ROOF_HEIGHT,
        window_width: float = DEFAULT_WINDOW_WIDTH,
        window_height: float = DEFAULT_WINDOW_HEIGHT,
        window_depth: float = DEFAULT_WINDOW_DEPTH,
        door_width: float = DEFAULT_DOOR_WIDTH,
        door_height: float = DEFAULT_DOOR_HEIGHT,
        door_depth: float = DEFAULT_DOOR_DEPTH,
        edge_trim_size: float = DEFAULT_EDGE_TRIM_SIZE,
        material_id: int = 0,
    ) -> None:
        super().__init__()

        # Validate parameters
        if width <= 0:
            raise ValueError("width must be positive")
        if height <= 0:
            raise ValueError("height must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if floors < 1:
            raise ValueError("floors must be at least 1")
        if windows_per_floor < 0:
            raise ValueError("windows_per_floor must be non-negative")

        self.width = width
        self.height = height
        self.depth = depth
        self.floors = floors
        self.windows_per_floor = windows_per_floor
        self.roof_style = roof_style
        self.roof_height = roof_height
        self.window_width = window_width
        self.window_height = window_height
        self.window_depth = window_depth
        self.door_width = door_width
        self.door_height = door_height
        self.door_depth = door_depth
        self.edge_trim_size = edge_trim_size
        self.material_id = material_id
        self._cached_sdf = None
        self._wgsl_cache = None

        # Mark all fields dirty
        for field in self._field_names:
            self.tracker.mark_dirty(field)

    @property
    def floor_height(self) -> float:
        """Height of each floor."""
        return self.height / self.floors

    @property
    def total_height(self) -> float:
        """Total height including roof."""
        if self.roof_style == RoofStyle.FLAT:
            return self.height
        return self.height + self.roof_height

    def get_main_structure_bounds(self) -> Tuple[Vec3, Vec3]:
        """
        Get the bounding box of the main building structure.

        Returns:
            Tuple of (min_corner, max_corner) as Vec3
        """
        half_w = self.width / 2
        half_d = self.depth / 2
        return (
            Vec3(-half_w, 0.0, -half_d),
            Vec3(half_w, self.height, half_d)
        )

    def get_window_positions(self) -> List[Tuple[Vec3, int, int]]:
        """
        Get all window center positions with floor and column indices.

        Returns:
            List of (position, floor_index, column_index) tuples
        """
        positions = []

        if self.windows_per_floor == 0:
            return positions

        half_w = self.width / 2
        half_d = self.depth / 2
        floor_h = self.floor_height

        # Calculate horizontal spacing
        spacing_x = self.width / (self.windows_per_floor + 1)
        spacing_z = self.depth / (self.windows_per_floor + 1)

        for floor_idx in range(self.floors):
            # Center Y of this floor's windows
            y = floor_h * (floor_idx + 0.5)

            # Front face (positive Z)
            for col in range(self.windows_per_floor):
                x = -half_w + spacing_x * (col + 1)
                positions.append((Vec3(x, y, half_d), floor_idx, col))

            # Back face (negative Z)
            for col in range(self.windows_per_floor):
                x = -half_w + spacing_x * (col + 1)
                positions.append((Vec3(x, y, -half_d), floor_idx, col))

            # Left face (negative X)
            for col in range(self.windows_per_floor):
                z = -half_d + spacing_z * (col + 1)
                positions.append((Vec3(-half_w, y, z), floor_idx, col))

            # Right face (positive X)
            for col in range(self.windows_per_floor):
                z = -half_d + spacing_z * (col + 1)
                positions.append((Vec3(half_w, y, z), floor_idx, col))

        return positions

    def get_door_position(self) -> Vec3:
        """
        Get the door center position.

        Returns:
            Door center as Vec3
        """
        return Vec3(0.0, self.door_height / 2, self.depth / 2)

    def evaluate(self, p: Vec3) -> float:
        """
        Evaluate the SDF at position p.

        This is a Python model for testing. The actual GPU evaluation
        uses the WGSL code generated by to_wgsl().

        Args:
            p: Query position

        Returns:
            Signed distance to the building surface
        """
        # Main structure box
        half_w = self.width / 2
        half_h = self.height / 2
        half_d = self.depth / 2

        # Translate to building-local coordinates (building base at y=0)
        local_p = Vec3(p.x, p.y - half_h, p.z)

        # Box SDF
        d = self._sd_box(local_p, Vec3(half_w, half_h, half_d))

        # Subtract windows
        for pos, _, _ in self.get_window_positions():
            # Window carving - depends on which face
            if abs(pos.z) > abs(pos.x):
                # Front/back face window
                window_box = Vec3(
                    self.window_width / 2,
                    self.window_height / 2,
                    self.window_depth
                )
            else:
                # Side face window
                window_box = Vec3(
                    self.window_depth,
                    self.window_height / 2,
                    self.window_width / 2
                )

            window_center = Vec3(pos.x, pos.y - half_h, pos.z)
            window_p = Vec3(
                local_p.x - window_center.x,
                local_p.y - window_center.y,
                local_p.z - window_center.z
            )
            window_d = self._sd_box(window_p, window_box)
            # Subtraction: max(a, -b)
            d = max(d, -window_d)

        # Subtract door
        door_pos = self.get_door_position()
        door_box = Vec3(
            self.door_width / 2,
            self.door_height / 2,
            self.door_depth
        )
        door_center = Vec3(door_pos.x, door_pos.y - half_h, door_pos.z)
        door_p = Vec3(
            local_p.x - door_center.x,
            local_p.y - door_center.y,
            local_p.z - door_center.z
        )
        door_d = self._sd_box(door_p, door_box)
        d = max(d, -door_d)

        # Add roof
        if self.roof_style == RoofStyle.PITCHED:
            roof_d = self._sd_pyramid_roof(p)
            d = min(d, roof_d)
        elif self.roof_style == RoofStyle.DOME:
            roof_d = self._sd_dome_roof(p)
            d = min(d, roof_d)
        elif self.roof_style == RoofStyle.FLAT:
            # Flat roof with edge trim
            trim_d = self._sd_flat_roof_trim(p)
            d = min(d, trim_d)

        return d

    def _sd_box(self, p: Vec3, half_extents: Vec3) -> float:
        """Box SDF helper."""
        qx = abs(p.x) - half_extents.x
        qy = abs(p.y) - half_extents.y
        qz = abs(p.z) - half_extents.z

        outside = math.sqrt(
            max(qx, 0.0) ** 2 +
            max(qy, 0.0) ** 2 +
            max(qz, 0.0) ** 2
        )
        inside = min(max(qx, max(qy, qz)), 0.0)

        return outside + inside

    def _sd_pyramid_roof(self, p: Vec3) -> float:
        """Pitched roof SDF - triangular prism along Z axis."""
        # Translate to roof base
        local_p = Vec3(p.x, p.y - self.height, p.z)

        half_w = self.width / 2
        half_d = self.depth / 2

        # Triangular prism (ridge along Z)
        # Distance to angled faces
        slope = self.roof_height / half_w
        angle = math.atan(slope)
        nx = math.cos(angle)
        ny = math.sin(angle)

        # Left slope plane
        d_left = nx * (local_p.x + half_w) - ny * local_p.y
        # Right slope plane
        d_right = -nx * (local_p.x - half_w) - ny * local_p.y

        # Front/back caps
        d_front = abs(local_p.z) - half_d

        # Bottom cap
        d_bottom = -local_p.y

        # Roof distance is max of all constraints
        d = max(max(d_left, d_right), max(d_front, d_bottom))

        return d

    def _sd_dome_roof(self, p: Vec3) -> float:
        """Hemispherical dome roof SDF."""
        # Translate to roof base center
        local_p = Vec3(p.x, p.y - self.height, p.z)

        # Hemisphere radius based on building dimensions
        radius = min(self.width, self.depth) / 2

        # Sphere SDF capped at y=0
        sphere_d = math.sqrt(
            local_p.x ** 2 + local_p.y ** 2 + local_p.z ** 2
        ) - radius

        # Clamp to upper hemisphere
        if local_p.y < 0:
            return max(sphere_d, -local_p.y)
        return sphere_d

    def _sd_flat_roof_trim(self, p: Vec3) -> float:
        """Flat roof edge trim (parapet) SDF."""
        half_w = self.width / 2
        half_d = self.depth / 2
        trim = self.edge_trim_size

        # Translate to roof level
        local_p = Vec3(p.x, p.y - self.height, p.z)

        # Outer box of trim
        outer = self._sd_box(local_p, Vec3(
            half_w + trim,
            trim,
            half_d + trim
        ))

        # Inner box (to subtract)
        inner = self._sd_box(local_p, Vec3(
            half_w - trim,
            trim * 2,  # Deeper to cut through
            half_d - trim
        ))

        # Subtraction for hollow trim
        return max(outer, -inner)

    def to_wgsl(self, name: str = "building") -> str:
        """
        Generate WGSL code for this building SDF.

        Args:
            name: Function name prefix

        Returns:
            WGSL source code string
        """
        lines = [
            f"// Building SDF: {name}",
            f"// Dimensions: {self.width} x {self.height} x {self.depth}",
            f"// Floors: {self.floors}, Windows/floor: {self.windows_per_floor}",
            f"// Roof style: {self.roof_style.name}",
            "",
        ]

        # Main function
        lines.append(f"fn sd_{name}(p: vec3<f32>) -> f32 {{")

        # Building dimensions
        half_w = self.width / 2
        half_h = self.height / 2
        half_d = self.depth / 2

        lines.append(f"    let half_w = {half_w:.6f};")
        lines.append(f"    let half_h = {half_h:.6f};")
        lines.append(f"    let half_d = {half_d:.6f};")
        lines.append("")
        lines.append("    // Local coordinates (building base at y=0)")
        lines.append("    let local_p = vec3<f32>(p.x, p.y - half_h, p.z);")
        lines.append("")
        lines.append("    // Main structure box")
        lines.append("    var d = sdBox(local_p, vec3<f32>(half_w, half_h, half_d));")
        lines.append("")

        # Windows
        if self.windows_per_floor > 0:
            lines.append("    // Windows")
            for i, (pos, floor_idx, col_idx) in enumerate(self.get_window_positions()):
                # Determine window orientation
                if abs(pos.z) > abs(pos.x):
                    wx, wy, wz = self.window_width / 2, self.window_height / 2, self.window_depth
                else:
                    wx, wy, wz = self.window_depth, self.window_height / 2, self.window_width / 2

                cx = pos.x
                cy = pos.y - half_h
                cz = pos.z

                lines.append(f"    {{ // Window {i} (floor {floor_idx}, col {col_idx})")
                lines.append(f"        let wp = local_p - vec3<f32>({cx:.6f}, {cy:.6f}, {cz:.6f});")
                lines.append(f"        let wd = sdBox(wp, vec3<f32>({wx:.6f}, {wy:.6f}, {wz:.6f}));")
                lines.append(f"        d = max(d, -wd);")
                lines.append(f"    }}")
            lines.append("")

        # Door
        door_pos = self.get_door_position()
        lines.append("    // Door")
        lines.append("    {")
        lines.append(f"        let dp = local_p - vec3<f32>({door_pos.x:.6f}, {door_pos.y - half_h:.6f}, {door_pos.z:.6f});")
        lines.append(f"        let dd = sdBox(dp, vec3<f32>({self.door_width/2:.6f}, {self.door_height/2:.6f}, {self.door_depth:.6f}));")
        lines.append("        d = max(d, -dd);")
        lines.append("    }")
        lines.append("")

        # Roof
        if self.roof_style == RoofStyle.PITCHED:
            lines.append("    // Pitched roof")
            lines.append(f"    let roof_p = vec3<f32>(p.x, p.y - {self.height:.6f}, p.z);")
            slope = self.roof_height / half_w
            angle = math.atan(slope)
            nx = math.cos(angle)
            ny = math.sin(angle)
            lines.append(f"    let d_left = {nx:.6f} * (roof_p.x + {half_w:.6f}) - {ny:.6f} * roof_p.y;")
            lines.append(f"    let d_right = {-nx:.6f} * (roof_p.x - {half_w:.6f}) - {ny:.6f} * roof_p.y;")
            lines.append(f"    let d_front = abs(roof_p.z) - {half_d:.6f};")
            lines.append("    let d_bottom = -roof_p.y;")
            lines.append("    let roof_d = max(max(d_left, d_right), max(d_front, d_bottom));")
            lines.append("    d = min(d, roof_d);")
        elif self.roof_style == RoofStyle.DOME:
            lines.append("    // Dome roof")
            lines.append(f"    let roof_p = vec3<f32>(p.x, p.y - {self.height:.6f}, p.z);")
            radius = min(self.width, self.depth) / 2
            lines.append(f"    let sphere_d = length(roof_p) - {radius:.6f};")
            lines.append("    let roof_d = select(sphere_d, max(sphere_d, -roof_p.y), roof_p.y < 0.0);")
            lines.append("    d = min(d, roof_d);")
        elif self.roof_style == RoofStyle.FLAT:
            lines.append("    // Flat roof with edge trim")
            trim = self.edge_trim_size
            lines.append(f"    let roof_p = vec3<f32>(p.x, p.y - {self.height:.6f}, p.z);")
            lines.append(f"    let outer = sdBox(roof_p, vec3<f32>({half_w + trim:.6f}, {trim:.6f}, {half_d + trim:.6f}));")
            lines.append(f"    let inner = sdBox(roof_p, vec3<f32>({half_w - trim:.6f}, {trim * 2:.6f}, {half_d - trim:.6f}));")
            lines.append("    let trim_d = max(outer, -inner);")
            lines.append("    d = min(d, trim_d);")

        lines.append("")
        lines.append("    return d;")
        lines.append("}")

        return "\n".join(lines)

    def label(self) -> str:
        return (
            f"Building({self.width}x{self.height}x{self.depth}, "
            f"{self.floors}F, {self.windows_per_floor}W, {self.roof_style.name})"
        )

    def clone(self) -> "BuildingSDF":
        return BuildingSDF(
            width=self.width,
            height=self.height,
            depth=self.depth,
            floors=self.floors,
            windows_per_floor=self.windows_per_floor,
            roof_style=self.roof_style,
            roof_height=self.roof_height,
            window_width=self.window_width,
            window_height=self.window_height,
            window_depth=self.window_depth,
            door_width=self.door_width,
            door_height=self.door_height,
            door_depth=self.door_depth,
            edge_trim_size=self.edge_trim_size,
            material_id=self.material_id,
        )


# =============================================================================
# CityBlockSDF - T-DEMO-4.8
# =============================================================================

class CityBlockSDF(SDFNode):
    """
    City block generator using domain repetition.

    Creates an infinite grid of buildings using domain repetition
    with per-block pseudo-random variation for:
    - Building height
    - Window pattern
    - Roof style
    - Material/color ID

    Street gaps are maintained between blocks.

    Attributes:
        cell_size: Size of each cell (building + street)
        street_width: Width of streets between buildings
        base_width: Base building width
        base_height: Base building height (before variation)
        base_depth: Base building depth
        height_variation: Max height variation factor (0-1)
        min_floors: Minimum number of floors
        max_floors: Maximum number of floors
        min_windows: Minimum windows per floor
        max_windows: Maximum windows per floor
        seed: Random seed for variation
    """

    __slots__ = (
        "cell_size", "street_width", "base_width", "base_height", "base_depth",
        "height_variation", "min_floors", "max_floors", "min_windows",
        "max_windows", "seed", "_building_cache"
    )

    # Type annotations
    cell_size: float
    street_width: float
    base_width: float
    base_height: float
    base_depth: float
    height_variation: float
    min_floors: int
    max_floors: int
    min_windows: int
    max_windows: int
    seed: int

    def __init__(
        self,
        cell_size: float = DEFAULT_CELL_SIZE,
        street_width: float = DEFAULT_STREET_WIDTH,
        base_width: float = DEFAULT_WIDTH,
        base_height: float = DEFAULT_HEIGHT,
        base_depth: float = DEFAULT_DEPTH,
        height_variation: float = DEFAULT_HEIGHT_VARIATION,
        min_floors: int = DEFAULT_MIN_FLOORS,
        max_floors: int = DEFAULT_MAX_FLOORS,
        min_windows: int = 2,
        max_windows: int = 6,
        seed: int = 42,
    ) -> None:
        super().__init__()

        # Validate
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        if street_width < 0:
            raise ValueError("street_width must be non-negative")
        if street_width >= cell_size:
            raise ValueError("street_width must be less than cell_size")
        if base_width <= 0 or base_height <= 0 or base_depth <= 0:
            raise ValueError("base dimensions must be positive")
        if not (0.0 <= height_variation <= 1.0):
            raise ValueError("height_variation must be in [0, 1]")
        if min_floors < 1 or max_floors < min_floors:
            raise ValueError("floor range must be valid (min >= 1, max >= min)")
        if min_windows < 0 or max_windows < min_windows:
            raise ValueError("window range must be valid (min >= 0, max >= min)")

        self.cell_size = cell_size
        self.street_width = street_width
        self.base_width = base_width
        self.base_height = base_height
        self.base_depth = base_depth
        self.height_variation = height_variation
        self.min_floors = min_floors
        self.max_floors = max_floors
        self.min_windows = min_windows
        self.max_windows = max_windows
        self.seed = seed
        self._building_cache = {}

        # Mark dirty
        for field in self._field_names:
            self.tracker.mark_dirty(field)

    @property
    def building_area_size(self) -> float:
        """Size of the building area within each cell (excluding streets)."""
        return self.cell_size - self.street_width

    def get_cell_coords(self, p: Vec3) -> Tuple[int, int]:
        """
        Get the cell coordinates for a world position.

        Args:
            p: World position

        Returns:
            (cell_x, cell_z) integer coordinates
        """
        cell_x = int(math.floor(p.x / self.cell_size))
        cell_z = int(math.floor(p.z / self.cell_size))
        return (cell_x, cell_z)

    def get_cell_properties(self, cell_x: int, cell_z: int) -> dict:
        """
        Get deterministic properties for a specific cell.

        Uses cell hash to generate consistent random-looking values.

        Args:
            cell_x: Cell X coordinate
            cell_z: Cell Z coordinate

        Returns:
            Dict with height, floors, windows, roof_style, material_id
        """
        h = cell_hash(cell_x, cell_z, self.seed)

        # Height variation
        height_factor = 1.0 + hash_to_float(h, -self.height_variation, self.height_variation)
        height = self.base_height * height_factor

        # Floor count
        h2 = cell_hash(cell_x + 1000, cell_z + 1000, self.seed)
        floors = int(hash_to_float(h2, self.min_floors, self.max_floors + 1))
        floors = max(self.min_floors, min(self.max_floors, floors))

        # Windows per floor
        h3 = cell_hash(cell_x + 2000, cell_z + 2000, self.seed)
        windows = int(hash_to_float(h3, self.min_windows, self.max_windows + 1))
        windows = max(self.min_windows, min(self.max_windows, windows))

        # Roof style
        h4 = cell_hash(cell_x + 3000, cell_z + 3000, self.seed)
        roof_val = hash_to_float(h4, 0, 3)
        if roof_val < 1:
            roof_style = RoofStyle.FLAT
        elif roof_val < 2:
            roof_style = RoofStyle.PITCHED
        else:
            roof_style = RoofStyle.DOME

        # Material ID (for rendering variation)
        h5 = cell_hash(cell_x + 4000, cell_z + 4000, self.seed)
        material_id = int(hash_to_float(h5, 0, 8))

        return {
            "height": height,
            "floors": floors,
            "windows_per_floor": windows,
            "roof_style": roof_style,
            "material_id": material_id,
        }

    def get_building_for_cell(self, cell_x: int, cell_z: int) -> BuildingSDF:
        """
        Get or create the building for a specific cell.

        Results are cached for efficiency.

        Args:
            cell_x: Cell X coordinate
            cell_z: Cell Z coordinate

        Returns:
            BuildingSDF configured for this cell
        """
        key = (cell_x, cell_z)
        if key not in self._building_cache:
            props = self.get_cell_properties(cell_x, cell_z)

            # Scale width/depth to fit in building area
            area = self.building_area_size
            width = min(self.base_width, area * 0.9)
            depth = min(self.base_depth, area * 0.9)

            self._building_cache[key] = BuildingSDF(
                width=width,
                height=props["height"],
                depth=depth,
                floors=props["floors"],
                windows_per_floor=props["windows_per_floor"],
                roof_style=props["roof_style"],
                material_id=props["material_id"],
            )

        return self._building_cache[key]

    def domain_repeat(self, p: Vec3) -> Tuple[Vec3, int, int]:
        """
        Apply domain repetition to get local coordinates and cell ID.

        Args:
            p: World position

        Returns:
            (local_p, cell_x, cell_z) where local_p is centered in cell
        """
        cell_x, cell_z = self.get_cell_coords(p)

        # Local position within cell, centered
        local_x = p.x - (cell_x + 0.5) * self.cell_size
        local_z = p.z - (cell_z + 0.5) * self.cell_size

        return Vec3(local_x, p.y, local_z), cell_x, cell_z

    def evaluate(self, p: Vec3) -> float:
        """
        Evaluate the city SDF at position p.

        Uses domain repetition to sample the nearest building.

        Args:
            p: Query position

        Returns:
            Signed distance to nearest building surface
        """
        local_p, cell_x, cell_z = self.domain_repeat(p)

        # Get building for this cell
        building = self.get_building_for_cell(cell_x, cell_z)

        # Check if we're in street area
        half_area = self.building_area_size / 2
        if abs(local_p.x) > half_area or abs(local_p.z) > half_area:
            # In street - return distance to cell boundary
            dx = abs(local_p.x) - half_area
            dz = abs(local_p.z) - half_area
            street_d = math.sqrt(max(dx, 0) ** 2 + max(dz, 0) ** 2)
            return street_d

        # Evaluate building SDF at local position
        return building.evaluate(local_p)

    def evaluate_with_neighbors(self, p: Vec3, neighbor_range: int = 1) -> float:
        """
        Evaluate SDF considering neighboring cells for smooth transitions.

        Args:
            p: Query position
            neighbor_range: Number of neighbor cells to check (1 = 3x3 grid)

        Returns:
            Minimum signed distance across all considered buildings
        """
        cell_x, cell_z = self.get_cell_coords(p)

        min_d = float('inf')

        for dx in range(-neighbor_range, neighbor_range + 1):
            for dz in range(-neighbor_range, neighbor_range + 1):
                cx = cell_x + dx
                cz = cell_z + dz

                # Get local position relative to this cell
                cell_center_x = (cx + 0.5) * self.cell_size
                cell_center_z = (cz + 0.5) * self.cell_size
                local_p = Vec3(
                    p.x - cell_center_x,
                    p.y,
                    p.z - cell_center_z
                )

                # Check street boundary
                half_area = self.building_area_size / 2
                if abs(local_p.x) <= half_area and abs(local_p.z) <= half_area:
                    # Inside building area
                    building = self.get_building_for_cell(cx, cz)
                    d = building.evaluate(local_p)
                    min_d = min(min_d, d)
                else:
                    # In street - compute distance to building area
                    dx_dist = abs(local_p.x) - half_area
                    dz_dist = abs(local_p.z) - half_area
                    street_d = math.sqrt(max(dx_dist, 0) ** 2 + max(dz_dist, 0) ** 2)
                    # Also need to consider building inside
                    if dx_dist > 0 or dz_dist > 0:
                        min_d = min(min_d, street_d)

        return min_d

    def to_wgsl(self, name: str = "city") -> str:
        """
        Generate WGSL code for the city block SDF.

        Includes domain repetition and cell-based variation.

        Args:
            name: Function name prefix

        Returns:
            WGSL source code string
        """
        lines = [
            f"// City Block SDF: {name}",
            f"// Cell size: {self.cell_size}, Street width: {self.street_width}",
            f"// Height variation: {self.height_variation}",
            f"// Floors: {self.min_floors}-{self.max_floors}",
            "",
            "// Cell hash function",
            "fn cell_hash(cell_x: i32, cell_z: i32, seed: i32) -> u32 {",
            "    var h = u32(seed);",
            "    h = h ^ (u32(cell_x) * 374761393u);",
            "    h = ((h << 17u) | (h >> 15u));",
            "    h = h ^ (u32(cell_z) * 668265263u);",
            "    h = ((h << 13u) | (h >> 19u));",
            "    h = h * 1274126177u;",
            "    return h;",
            "}",
            "",
            "fn hash_to_float(h: u32, min_val: f32, max_val: f32) -> f32 {",
            "    let normalized = f32(h) / 4294967295.0;",
            "    return min_val + normalized * (max_val - min_val);",
            "}",
            "",
        ]

        # Main city SDF function
        lines.append(f"fn sd_{name}(p: vec3<f32>) -> vec2<f32> {{")
        lines.append(f"    let cell_size = {self.cell_size:.6f};")
        lines.append(f"    let street_width = {self.street_width:.6f};")
        lines.append(f"    let building_area = cell_size - street_width;")
        lines.append(f"    let half_area = building_area * 0.5;")
        lines.append(f"    let seed = {self.seed};")
        lines.append("")
        lines.append("    // Cell coordinates")
        lines.append("    let cell_x = i32(floor(p.x / cell_size));")
        lines.append("    let cell_z = i32(floor(p.z / cell_size));")
        lines.append("")
        lines.append("    // Local position in cell (centered)")
        lines.append("    let cell_center_x = (f32(cell_x) + 0.5) * cell_size;")
        lines.append("    let cell_center_z = (f32(cell_z) + 0.5) * cell_size;")
        lines.append("    let local_p = vec3<f32>(p.x - cell_center_x, p.y, p.z - cell_center_z);")
        lines.append("")
        lines.append("    // Check multiple cells for smooth transitions")
        lines.append("    var min_d = 1000000.0;")
        lines.append("    var material_id = 0.0;")
        lines.append("")
        lines.append("    for (var dx = -1; dx <= 1; dx++) {")
        lines.append("        for (var dz = -1; dz <= 1; dz++) {")
        lines.append("            let cx = cell_x + dx;")
        lines.append("            let cz = cell_z + dz;")
        lines.append("")
        lines.append("            // Cell-relative position")
        lines.append("            let ccx = (f32(cx) + 0.5) * cell_size;")
        lines.append("            let ccz = (f32(cz) + 0.5) * cell_size;")
        lines.append("            let lp = vec3<f32>(p.x - ccx, p.y, p.z - ccz);")
        lines.append("")
        lines.append("            // Skip if outside building area")
        lines.append("            if (abs(lp.x) > half_area || abs(lp.z) > half_area) {")
        lines.append("                continue;")
        lines.append("            }")
        lines.append("")
        lines.append("            // Cell properties from hash")
        lines.append("            let h1 = cell_hash(cx, cz, seed);")
        lines.append(f"            let height_factor = 1.0 + hash_to_float(h1, {-self.height_variation:.6f}, {self.height_variation:.6f});")
        lines.append(f"            let height = {self.base_height:.6f} * height_factor;")
        lines.append("")
        lines.append("            let h2 = cell_hash(cx + 1000, cz + 1000, seed);")
        lines.append(f"            let floors = i32(hash_to_float(h2, {float(self.min_floors):.1f}, {float(self.max_floors + 1):.1f}));")
        lines.append("")
        lines.append("            let h3 = cell_hash(cx + 2000, cz + 2000, seed);")
        lines.append(f"            let windows = i32(hash_to_float(h3, {float(self.min_windows):.1f}, {float(self.max_windows + 1):.1f}));")
        lines.append("")
        lines.append("            let h4 = cell_hash(cx + 3000, cz + 3000, seed);")
        lines.append("            let roof_type = i32(hash_to_float(h4, 0.0, 3.0));")
        lines.append("")
        lines.append("            let h5 = cell_hash(cx + 4000, cz + 4000, seed);")
        lines.append("            let mat_id = hash_to_float(h5, 0.0, 8.0);")
        lines.append("")
        lines.append("            // Evaluate building (simplified - full version has windows/door)")
        lines.append(f"            let half_w = min({self.base_width:.6f}, building_area * 0.9) * 0.5;")
        lines.append("            let half_h = height * 0.5;")
        lines.append(f"            let half_d = min({self.base_depth:.6f}, building_area * 0.9) * 0.5;")
        lines.append("")
        lines.append("            let bp = vec3<f32>(lp.x, lp.y - half_h, lp.z);")
        lines.append("            let d = sdBox(bp, vec3<f32>(half_w, half_h, half_d));")
        lines.append("")
        lines.append("            if (d < min_d) {")
        lines.append("                min_d = d;")
        lines.append("                material_id = mat_id;")
        lines.append("            }")
        lines.append("        }")
        lines.append("    }")
        lines.append("")
        lines.append("    return vec2<f32>(min_d, material_id);")
        lines.append("}")

        return "\n".join(lines)

    def label(self) -> str:
        return (
            f"CityBlock(cell={self.cell_size}, street={self.street_width}, "
            f"floors={self.min_floors}-{self.max_floors})"
        )

    def clone(self) -> "CityBlockSDF":
        return CityBlockSDF(
            cell_size=self.cell_size,
            street_width=self.street_width,
            base_width=self.base_width,
            base_height=self.base_height,
            base_depth=self.base_depth,
            height_variation=self.height_variation,
            min_floors=self.min_floors,
            max_floors=self.max_floors,
            min_windows=self.min_windows,
            max_windows=self.max_windows,
            seed=self.seed,
        )
