"""
Terrain Level of Detail (LOD) system.

Provides LOD selection, quadtree-based terrain chunking, and seamless
transitions between detail levels using various LOD methods and stitching
techniques.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple
from typing import runtime_checkable

from engine.world.terrain.constants import (
    DEFAULT_RESOLUTION,
    HEIGHT_EPSILON,
)

if TYPE_CHECKING:
    from typing import TypeAlias


class TerrainLODMethod(Enum):
    """LOD method for terrain rendering."""

    GEO_MIPMAPPING = auto()  # Classic geo-mipmapping
    CLIPMAPS = auto()  # GPU-based clipmaps
    CDLOD = auto()  # Continuous Distance-Dependent LOD
    QUADTREE = auto()  # Quadtree-based adaptive


class LODStitchMethod(Enum):
    """Method for stitching LOD transitions."""

    SKIRTS = auto()  # Vertical skirts to hide gaps
    MORPHING = auto()  # Vertex morphing for smooth transition
    INDEX_MODIFICATION = auto()  # Modified indices at boundaries


@dataclass
class BoundingBox:
    """Axis-aligned bounding box.

    Attributes:
        min_x: Minimum X coordinate.
        min_y: Minimum Y coordinate.
        min_z: Minimum Z coordinate.
        max_x: Maximum X coordinate.
        max_y: Maximum Y coordinate.
        max_z: Maximum Z coordinate.
    """

    min_x: float = 0.0
    min_y: float = 0.0
    min_z: float = 0.0
    max_x: float = 0.0
    max_y: float = 0.0
    max_z: float = 0.0

    @property
    def center(self) -> Tuple[float, float, float]:
        """Get center point of the box."""
        return (
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
            (self.min_z + self.max_z) / 2,
        )

    @property
    def size(self) -> Tuple[float, float, float]:
        """Get size of the box."""
        return (
            self.max_x - self.min_x,
            self.max_y - self.min_y,
            self.max_z - self.min_z,
        )

    @property
    def width(self) -> float:
        """Get width (X dimension)."""
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """Get height (Y dimension)."""
        return self.max_y - self.min_y

    @property
    def depth(self) -> float:
        """Get depth (Z dimension)."""
        return self.max_z - self.min_z

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if a point is inside the box."""
        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
            and self.min_z <= z <= self.max_z
        )

    def intersects(self, other: BoundingBox) -> bool:
        """Check if this box intersects another."""
        return (
            self.min_x <= other.max_x
            and self.max_x >= other.min_x
            and self.min_y <= other.max_y
            and self.max_y >= other.min_y
            and self.min_z <= other.max_z
            and self.max_z >= other.min_z
        )

    def distance_to_point(self, x: float, y: float, z: float) -> float:
        """Calculate distance from point to nearest box surface."""
        dx = max(self.min_x - x, 0, x - self.max_x)
        dy = max(self.min_y - y, 0, y - self.max_y)
        dz = max(self.min_z - z, 0, z - self.max_z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)


@runtime_checkable
class Frustum(Protocol):
    """Protocol for view frustum used in culling."""

    def contains_box(self, box: BoundingBox) -> bool:
        """Check if frustum contains or intersects a bounding box."""
        ...

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Check if frustum contains a point."""
        ...


@dataclass
class TerrainChunk:
    """A chunk of terrain at a specific LOD level.

    Attributes:
        bounds: Bounding box of the chunk.
        lod_level: LOD level (0 = highest detail).
        vertex_count: Number of vertices in the chunk.
        index_count: Number of indices in the chunk.
        max_error: Maximum geometric error for this LOD.
        neighbor_lods: LOD levels of adjacent chunks [north, east, south, west].
    """

    bounds: BoundingBox = field(default_factory=BoundingBox)
    lod_level: int = 0
    vertex_count: int = 0
    index_count: int = 0
    max_error: float = 0.0
    neighbor_lods: List[int] = field(default_factory=lambda: [-1, -1, -1, -1])

    def get_error_metric(self, camera_x: float, camera_y: float, camera_z: float) -> float:
        """Calculate error metric based on distance from camera.

        Args:
            camera_x: Camera X position.
            camera_y: Camera Y position.
            camera_z: Camera Z position.

        Returns:
            Error metric value (higher = more error visible).
        """
        distance = self.bounds.distance_to_point(camera_x, camera_y, camera_z)
        if distance < 1.0:
            distance = 1.0
        return self.max_error / distance

    def get_screen_space_error(
        self,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        fov_radians: float,
        screen_height: int,
    ) -> float:
        """Calculate screen-space error in pixels.

        Args:
            camera_x: Camera X position.
            camera_y: Camera Y position.
            camera_z: Camera Z position.
            fov_radians: Vertical field of view in radians.
            screen_height: Screen height in pixels.

        Returns:
            Approximate error in pixels.
        """
        distance = self.bounds.distance_to_point(camera_x, camera_y, camera_z)
        if distance < 1.0:
            distance = 1.0

        # Project error to screen space
        # error_pixels = (error_world / distance) * (screen_height / (2 * tan(fov/2)))
        projection_factor = screen_height / (2 * math.tan(fov_radians / 2))
        return (self.max_error / distance) * projection_factor


@dataclass
class QuadtreeNode:
    """Node in a terrain quadtree for LOD management.

    Attributes:
        bounds: Bounding box of this node.
        depth: Depth in the tree (0 = root).
        children: Four child nodes or None if leaf.
        chunk: Terrain chunk for this node.
        max_error: Maximum geometric error at this level.
    """

    bounds: BoundingBox = field(default_factory=BoundingBox)
    depth: int = 0
    children: Optional[List[QuadtreeNode]] = None
    chunk: Optional[TerrainChunk] = None
    max_error: float = 0.0

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.children is None

    def should_split(
        self,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        error_threshold: float,
        max_depth: int,
    ) -> bool:
        """Determine if this node should be split into children.

        Args:
            camera_x: Camera X position.
            camera_y: Camera Y position.
            camera_z: Camera Z position.
            error_threshold: Maximum acceptable error.
            max_depth: Maximum tree depth.

        Returns:
            True if the node should be split.
        """
        if self.depth >= max_depth:
            return False

        distance = self.bounds.distance_to_point(camera_x, camera_y, camera_z)
        if distance < 1.0:
            distance = 1.0

        error = self.max_error / distance
        return error > error_threshold

    def get_center(self) -> Tuple[float, float]:
        """Get XZ center of this node."""
        return (
            (self.bounds.min_x + self.bounds.max_x) / 2,
            (self.bounds.min_z + self.bounds.max_z) / 2,
        )


class TerrainQuadtree:
    """Quadtree structure for terrain LOD management.

    Organizes terrain into a hierarchical structure for efficient
    LOD selection and rendering.
    """

    def __init__(
        self,
        bounds: BoundingBox,
        max_depth: int = 8,
        base_error: float = 100.0,
    ) -> None:
        """Initialize the quadtree.

        Args:
            bounds: World bounds of the terrain.
            max_depth: Maximum depth of the quadtree.
            base_error: Error at the root level.
        """
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if base_error <= 0:
            raise ValueError("base_error must be > 0")

        self._max_depth = max_depth
        self._base_error = base_error
        self._root = QuadtreeNode(bounds=bounds, depth=0, max_error=base_error)
        self._build_tree(self._root)

    @property
    def root(self) -> QuadtreeNode:
        """Get the root node."""
        return self._root

    @property
    def max_depth(self) -> int:
        """Get maximum depth."""
        return self._max_depth

    def _build_tree(self, node: QuadtreeNode) -> None:
        """Recursively build the quadtree structure.

        Args:
            node: Current node to process.
        """
        if node.depth >= self._max_depth:
            # Create chunk for leaf node
            node.chunk = TerrainChunk(
                bounds=node.bounds,
                lod_level=node.depth,
                max_error=node.max_error,
            )
            return

        # Create children
        bounds = node.bounds
        mid_x = (bounds.min_x + bounds.max_x) / 2
        mid_z = (bounds.min_z + bounds.max_z) / 2
        child_error = node.max_error / 2

        node.children = [
            # NW
            QuadtreeNode(
                bounds=BoundingBox(
                    min_x=bounds.min_x,
                    min_y=bounds.min_y,
                    min_z=bounds.min_z,
                    max_x=mid_x,
                    max_y=bounds.max_y,
                    max_z=mid_z,
                ),
                depth=node.depth + 1,
                max_error=child_error,
            ),
            # NE
            QuadtreeNode(
                bounds=BoundingBox(
                    min_x=mid_x,
                    min_y=bounds.min_y,
                    min_z=bounds.min_z,
                    max_x=bounds.max_x,
                    max_y=bounds.max_y,
                    max_z=mid_z,
                ),
                depth=node.depth + 1,
                max_error=child_error,
            ),
            # SW
            QuadtreeNode(
                bounds=BoundingBox(
                    min_x=bounds.min_x,
                    min_y=bounds.min_y,
                    min_z=mid_z,
                    max_x=mid_x,
                    max_y=bounds.max_y,
                    max_z=bounds.max_z,
                ),
                depth=node.depth + 1,
                max_error=child_error,
            ),
            # SE
            QuadtreeNode(
                bounds=BoundingBox(
                    min_x=mid_x,
                    min_y=bounds.min_y,
                    min_z=mid_z,
                    max_x=bounds.max_x,
                    max_y=bounds.max_y,
                    max_z=bounds.max_z,
                ),
                depth=node.depth + 1,
                max_error=child_error,
            ),
        ]

        # Create chunk for this level (used when not splitting)
        node.chunk = TerrainChunk(
            bounds=node.bounds,
            lod_level=node.depth,
            max_error=node.max_error,
        )

        # Recursively build children
        for child in node.children:
            self._build_tree(child)

    def select_lod(
        self,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        error_threshold: float,
    ) -> List[TerrainChunk]:
        """Select LOD chunks based on camera position.

        Args:
            camera_x: Camera X position.
            camera_y: Camera Y position.
            camera_z: Camera Z position.
            error_threshold: Maximum acceptable screen-space error.

        Returns:
            List of chunks to render.
        """
        chunks: List[TerrainChunk] = []
        self._select_lod_recursive(
            self._root, camera_x, camera_y, camera_z, error_threshold, chunks
        )
        return chunks

    def _select_lod_recursive(
        self,
        node: QuadtreeNode,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        error_threshold: float,
        chunks: List[TerrainChunk],
    ) -> None:
        """Recursively select LOD chunks.

        Args:
            node: Current node.
            camera_x: Camera X position.
            camera_y: Camera Y position.
            camera_z: Camera Z position.
            error_threshold: Maximum acceptable error.
            chunks: List to append selected chunks.
        """
        if node.is_leaf:
            if node.chunk is not None:
                chunks.append(node.chunk)
            return

        if node.should_split(camera_x, camera_y, camera_z, error_threshold, self._max_depth):
            # Use children
            if node.children is not None:
                for child in node.children:
                    self._select_lod_recursive(
                        child, camera_x, camera_y, camera_z, error_threshold, chunks
                    )
        else:
            # Use this node's chunk
            if node.chunk is not None:
                chunks.append(node.chunk)

    def get_visible_chunks(
        self,
        frustum: Frustum,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        error_threshold: float,
    ) -> List[TerrainChunk]:
        """Get chunks that are visible in the frustum.

        Args:
            frustum: View frustum for culling.
            camera_x: Camera X position.
            camera_y: Camera Y position.
            camera_z: Camera Z position.
            error_threshold: Maximum acceptable error.

        Returns:
            List of visible chunks.
        """
        chunks: List[TerrainChunk] = []
        self._get_visible_recursive(
            self._root, frustum, camera_x, camera_y, camera_z, error_threshold, chunks
        )
        return chunks

    def _get_visible_recursive(
        self,
        node: QuadtreeNode,
        frustum: Frustum,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        error_threshold: float,
        chunks: List[TerrainChunk],
    ) -> None:
        """Recursively get visible chunks.

        Args:
            node: Current node.
            frustum: View frustum.
            camera_x: Camera X position.
            camera_y: Camera Y position.
            camera_z: Camera Z position.
            error_threshold: Maximum acceptable error.
            chunks: List to append visible chunks.
        """
        # Frustum culling
        if not frustum.contains_box(node.bounds):
            return

        if node.is_leaf:
            if node.chunk is not None:
                chunks.append(node.chunk)
            return

        if node.should_split(camera_x, camera_y, camera_z, error_threshold, self._max_depth):
            if node.children is not None:
                for child in node.children:
                    self._get_visible_recursive(
                        child,
                        frustum,
                        camera_x,
                        camera_y,
                        camera_z,
                        error_threshold,
                        chunks,
                    )
        else:
            if node.chunk is not None:
                chunks.append(node.chunk)


@dataclass
class ClipmapRing:
    """A ring in a clipmap-based terrain LOD system.

    Attributes:
        level: Clipmap level (0 = finest detail).
        inner_radius: Inner radius of the ring.
        outer_radius: Outer radius of the ring.
        resolution: Number of vertices per side at this level.
        cell_size: World size of each cell at this level.
    """

    level: int = 0
    inner_radius: float = 0.0
    outer_radius: float = 100.0
    resolution: int = 64
    cell_size: float = 1.0

    def __post_init__(self) -> None:
        """Validate ring settings."""
        if self.level < 0:
            raise ValueError("level must be >= 0")
        if self.inner_radius < 0:
            raise ValueError("inner_radius must be >= 0")
        if self.outer_radius <= self.inner_radius:
            raise ValueError("outer_radius must be > inner_radius")
        if self.resolution < 2:
            raise ValueError("resolution must be >= 2")
        if self.cell_size <= 0:
            raise ValueError("cell_size must be > 0")

    def get_mesh_for_ring(
        self, center_x: float, center_z: float
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int]]]:
        """Generate mesh vertices and indices for this ring.

        Args:
            center_x: World X center of the clipmap.
            center_z: World Z center of the clipmap.

        Returns:
            Tuple of (vertices, indices) where vertices are (x, z) positions
            and indices are triangles (i0, i1, i2).
        """
        vertices: List[Tuple[float, float]] = []
        indices: List[Tuple[int, int, int]] = []

        # Generate grid of vertices
        half_size = self.outer_radius
        step = (half_size * 2) / (self.resolution - 1)

        for iz in range(self.resolution):
            for ix in range(self.resolution):
                x = center_x - half_size + ix * step
                z = center_z - half_size + iz * step

                # Check if vertex is in the ring (not in inner area)
                dx = abs(x - center_x)
                dz = abs(z - center_z)
                max_dist = max(dx, dz)

                if max_dist >= self.inner_radius or self.inner_radius == 0:
                    vertices.append((x, z))

        # Generate indices for the ring area
        # Simplified - full grid minus inner grid
        vertex_index_map = {}
        for i, (vx, vz) in enumerate(vertices):
            ix = int((vx - (center_x - half_size)) / step + 0.5)
            iz = int((vz - (center_z - half_size)) / step + 0.5)
            vertex_index_map[(ix, iz)] = i

        for iz in range(self.resolution - 1):
            for ix in range(self.resolution - 1):
                # Check all four corners exist
                corners = [(ix, iz), (ix + 1, iz), (ix, iz + 1), (ix + 1, iz + 1)]
                if all(c in vertex_index_map for c in corners):
                    v00 = vertex_index_map[(ix, iz)]
                    v10 = vertex_index_map[(ix + 1, iz)]
                    v01 = vertex_index_map[(ix, iz + 1)]
                    v11 = vertex_index_map[(ix + 1, iz + 1)]

                    indices.append((v00, v10, v01))
                    indices.append((v10, v11, v01))

        return vertices, indices


@dataclass
class TerrainPatch:
    """A terrain patch for LOD management.

    Attributes:
        x: Grid X coordinate.
        z: Grid Z coordinate.
        bounds: World bounds of the patch.
        current_lod: Currently selected LOD level.
        target_lod: Target LOD level for transitions.
        morph_factor: Blending factor for LOD transitions (0-1).
    """

    x: int = 0
    z: int = 0
    bounds: BoundingBox = field(default_factory=BoundingBox)
    current_lod: int = 0
    target_lod: int = 0
    morph_factor: float = 0.0


class TerrainLODSystem:
    """Main terrain LOD management system.

    Handles LOD selection, transitions, and edge stitching for
    seamless terrain rendering.
    """

    def __init__(
        self,
        terrain_bounds: BoundingBox,
        patch_size: float = 64.0,
        method: TerrainLODMethod = TerrainLODMethod.QUADTREE,
        stitch_method: LODStitchMethod = LODStitchMethod.SKIRTS,
        max_lod_level: int = 6,
        error_threshold: float = 4.0,
    ) -> None:
        """Initialize the LOD system.

        Args:
            terrain_bounds: World bounds of the terrain.
            patch_size: Size of each terrain patch in world units.
            method: LOD method to use.
            stitch_method: Method for stitching LOD boundaries.
            max_lod_level: Maximum LOD level (0 = highest detail).
            error_threshold: Screen-space error threshold in pixels.
        """
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if max_lod_level < 1:
            raise ValueError("max_lod_level must be >= 1")
        if error_threshold <= 0:
            raise ValueError("error_threshold must be > 0")

        self._bounds = terrain_bounds
        self._patch_size = patch_size
        self._method = method
        self._stitch_method = stitch_method
        self._max_lod_level = max_lod_level
        self._error_threshold = error_threshold

        # Calculate patch grid dimensions
        self._patches_x = int(
            math.ceil((terrain_bounds.max_x - terrain_bounds.min_x) / patch_size)
        )
        self._patches_z = int(
            math.ceil((terrain_bounds.max_z - terrain_bounds.min_z) / patch_size)
        )

        # Create patches
        self._patches: List[List[TerrainPatch]] = []
        for pz in range(self._patches_z):
            row = []
            for px in range(self._patches_x):
                patch = TerrainPatch(
                    x=px,
                    z=pz,
                    bounds=BoundingBox(
                        min_x=terrain_bounds.min_x + px * patch_size,
                        min_y=terrain_bounds.min_y,
                        min_z=terrain_bounds.min_z + pz * patch_size,
                        max_x=terrain_bounds.min_x + (px + 1) * patch_size,
                        max_y=terrain_bounds.max_y,
                        max_z=terrain_bounds.min_z + (pz + 1) * patch_size,
                    ),
                    current_lod=max_lod_level,
                )
                row.append(patch)
            self._patches.append(row)

        # Initialize quadtree if using that method
        self._quadtree: Optional[TerrainQuadtree] = None
        if method == TerrainLODMethod.QUADTREE:
            self._quadtree = TerrainQuadtree(
                terrain_bounds,
                max_depth=max_lod_level,
                base_error=patch_size,
            )

        # Initialize clipmap rings if using that method
        self._clipmap_rings: List[ClipmapRing] = []
        if method == TerrainLODMethod.CLIPMAPS:
            for level in range(max_lod_level + 1):
                ring = ClipmapRing(
                    level=level,
                    inner_radius=patch_size * (2 ** level) / 2 if level > 0 else 0,
                    outer_radius=patch_size * (2 ** (level + 1)) / 2,
                    resolution=64,
                    cell_size=patch_size / 64 * (2 ** level),
                )
                self._clipmap_rings.append(ring)

    @property
    def method(self) -> TerrainLODMethod:
        """Get LOD method."""
        return self._method

    @property
    def stitch_method(self) -> LODStitchMethod:
        """Get stitch method."""
        return self._stitch_method

    @property
    def error_threshold(self) -> float:
        """Get error threshold."""
        return self._error_threshold

    @error_threshold.setter
    def error_threshold(self, value: float) -> None:
        """Set error threshold."""
        if value <= 0:
            raise ValueError("error_threshold must be > 0")
        self._error_threshold = value

    @property
    def max_lod_level(self) -> int:
        """Get maximum LOD level."""
        return self._max_lod_level

    @property
    def patches_x(self) -> int:
        """Get number of patches in X direction."""
        return self._patches_x

    @property
    def patches_z(self) -> int:
        """Get number of patches in Z direction."""
        return self._patches_z

    def get_patch(self, px: int, pz: int) -> Optional[TerrainPatch]:
        """Get a patch by grid coordinates.

        Args:
            px: Patch X coordinate.
            pz: Patch Z coordinate.

        Returns:
            The patch or None if out of bounds.
        """
        if 0 <= px < self._patches_x and 0 <= pz < self._patches_z:
            return self._patches[pz][px]
        return None

    def update(
        self,
        camera_x: float,
        camera_y: float,
        camera_z: float,
        fov_radians: float = math.pi / 3,
        screen_height: int = 1080,
    ) -> None:
        """Update LOD selection based on camera position.

        Args:
            camera_x: Camera X position.
            camera_y: Camera Y position.
            camera_z: Camera Z position.
            fov_radians: Vertical field of view in radians.
            screen_height: Screen height in pixels.
        """
        if self._method == TerrainLODMethod.QUADTREE and self._quadtree is not None:
            # Quadtree handles its own selection
            return

        # Update each patch's LOD level
        for row in self._patches:
            for patch in row:
                # Calculate distance to patch center
                center_x, _, center_z = patch.bounds.center
                distance = math.sqrt(
                    (camera_x - center_x) ** 2 + (camera_z - center_z) ** 2
                )
                if distance < 1.0:
                    distance = 1.0

                # Calculate target LOD based on distance
                # Each LOD level doubles the acceptable distance
                target_lod = 0
                check_distance = self._patch_size
                while target_lod < self._max_lod_level and distance > check_distance:
                    target_lod += 1
                    check_distance *= 2

                patch.target_lod = target_lod

                # Update morph factor for smooth transitions
                if patch.current_lod != patch.target_lod:
                    # Gradually transition
                    morph_speed = 0.1  # Adjust for smoother/faster transitions
                    if patch.target_lod > patch.current_lod:
                        patch.morph_factor += morph_speed
                        if patch.morph_factor >= 1.0:
                            patch.current_lod = patch.target_lod
                            patch.morph_factor = 0.0
                    else:
                        patch.morph_factor -= morph_speed
                        if patch.morph_factor <= 0.0:
                            patch.current_lod = patch.target_lod
                            patch.morph_factor = 0.0

    def get_render_chunks(self) -> List[TerrainChunk]:
        """Get list of chunks to render with their assigned LODs.

        Returns:
            List of terrain chunks with LOD information.
        """
        chunks: List[TerrainChunk] = []

        for pz, row in enumerate(self._patches):
            for px, patch in enumerate(row):
                # Determine neighbor LODs
                neighbor_lods = [
                    self._patches[pz - 1][px].current_lod
                    if pz > 0
                    else -1,  # North
                    self._patches[pz][px + 1].current_lod
                    if px < self._patches_x - 1
                    else -1,  # East
                    self._patches[pz + 1][px].current_lod
                    if pz < self._patches_z - 1
                    else -1,  # South
                    self._patches[pz][px - 1].current_lod
                    if px > 0
                    else -1,  # West
                ]

                chunk = TerrainChunk(
                    bounds=patch.bounds,
                    lod_level=patch.current_lod,
                    neighbor_lods=neighbor_lods,
                    max_error=self._patch_size / (2 ** patch.current_lod),
                )
                chunks.append(chunk)

        return chunks

    def get_stitch_indices(
        self,
        patch: TerrainPatch,
        vertices_per_side: int = 17,
    ) -> List[int]:
        """Get indices for stitching a patch to its neighbors.

        Args:
            patch: The patch to generate stitch indices for.
            vertices_per_side: Number of vertices per side at LOD 0.

        Returns:
            List of indices for the stitched mesh.
        """
        indices: List[int] = []
        lod = patch.current_lod
        step = 2 ** lod
        verts = (vertices_per_side - 1) // step + 1

        # Get neighbor LODs
        neighbor_lods = [
            self._patches[patch.z - 1][patch.x].current_lod
            if patch.z > 0
            else lod,
            self._patches[patch.z][patch.x + 1].current_lod
            if patch.x < self._patches_x - 1
            else lod,
            self._patches[patch.z + 1][patch.x].current_lod
            if patch.z < self._patches_z - 1
            else lod,
            self._patches[patch.z][patch.x - 1].current_lod
            if patch.x > 0
            else lod,
        ]

        # Generate base grid indices
        for iz in range(verts - 1):
            for ix in range(verts - 1):
                v00 = iz * verts + ix
                v10 = iz * verts + (ix + 1)
                v01 = (iz + 1) * verts + ix
                v11 = (iz + 1) * verts + (ix + 1)

                # Check if this quad is on an edge that needs stitching
                on_north = iz == 0 and neighbor_lods[0] > lod
                on_east = ix == verts - 2 and neighbor_lods[1] > lod
                on_south = iz == verts - 2 and neighbor_lods[2] > lod
                on_west = ix == 0 and neighbor_lods[3] > lod

                if self._stitch_method == LODStitchMethod.INDEX_MODIFICATION:
                    # Modify triangulation for T-junction fix
                    if on_north and ix % 2 == 0 and ix + 1 < verts - 1:
                        # Skip alternate vertex on north edge
                        indices.extend([v00, v10 + 1, v01])
                        indices.extend([v10 + 1, v11 + 1, v01])
                        continue
                    if on_south and ix % 2 == 0 and ix + 1 < verts - 1:
                        indices.extend([v00, v10, v01])
                        indices.extend([v10, v11, v01 + 1])
                        continue

                # Default triangulation
                indices.extend([v00, v10, v01])
                indices.extend([v10, v11, v01])

        return indices

    def get_skirt_vertices(
        self,
        patch: TerrainPatch,
        skirt_depth: float = 10.0,
        vertices_per_side: int = 17,
    ) -> List[Tuple[float, float, float]]:
        """Generate skirt vertices for hiding LOD gaps.

        Args:
            patch: The patch to generate skirts for.
            skirt_depth: Depth of the skirt below terrain.
            vertices_per_side: Number of vertices per side.

        Returns:
            List of (x, y, z) vertex positions for skirts.
        """
        skirt_verts: List[Tuple[float, float, float]] = []
        step = patch.bounds.width / (vertices_per_side - 1)

        # North edge
        for i in range(vertices_per_side):
            x = patch.bounds.min_x + i * step
            z = patch.bounds.min_z
            # Terrain height would be sampled here; using 0 as placeholder
            y = patch.bounds.min_y
            skirt_verts.append((x, y - skirt_depth, z))

        # East edge
        for i in range(vertices_per_side):
            x = patch.bounds.max_x
            z = patch.bounds.min_z + i * step
            y = patch.bounds.min_y
            skirt_verts.append((x, y - skirt_depth, z))

        # South edge
        for i in range(vertices_per_side):
            x = patch.bounds.max_x - i * step
            z = patch.bounds.max_z
            y = patch.bounds.min_y
            skirt_verts.append((x, y - skirt_depth, z))

        # West edge
        for i in range(vertices_per_side):
            x = patch.bounds.min_x
            z = patch.bounds.max_z - i * step
            y = patch.bounds.min_y
            skirt_verts.append((x, y - skirt_depth, z))

        return skirt_verts

    def get_morph_factor(
        self,
        patch: TerrainPatch,
        vertex_x: float,
        vertex_z: float,
        camera_x: float,
        camera_z: float,
    ) -> float:
        """Calculate vertex morph factor for smooth LOD transitions.

        Args:
            patch: The patch containing the vertex.
            vertex_x: Vertex X position.
            vertex_z: Vertex Z position.
            camera_x: Camera X position.
            camera_z: Camera Z position.

        Returns:
            Morph factor (0 = current LOD, 1 = next coarser LOD).
        """
        if self._stitch_method != LODStitchMethod.MORPHING:
            return 0.0

        # Calculate distance from camera
        distance = math.sqrt(
            (camera_x - vertex_x) ** 2 + (camera_z - vertex_z) ** 2
        )

        # Calculate morph range for this LOD level
        lod_distance = self._patch_size * (2 ** patch.current_lod)
        morph_start = lod_distance * 0.8
        morph_end = lod_distance

        if distance <= morph_start:
            return 0.0
        elif distance >= morph_end:
            return 1.0
        else:
            return (distance - morph_start) / (morph_end - morph_start)
