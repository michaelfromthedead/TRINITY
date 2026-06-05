"""Probe Atlas Management System.

Implements efficient GPU probe storage via atlas packing:
- Fixed-grid atlas packing (e.g., 4x4 probes)
- Per-probe atlas UV coordinates
- Atlas update on capture
- Slot allocation/deallocation
- Mip chain support in atlas

Reference: RENDERING_CONTEXT.md Section 6.4
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple

from engine.core.math.vec import Vec2, Vec3
from engine.rendering.lighting.baked_probes import (
    BakedProbeConstants,
    CubemapData,
    CubemapFace,
    CubemapFaceData,
    CubemapMipChain,
    HDRPixel,
    MipLevel,
)

if TYPE_CHECKING:
    pass


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

class AtlasConstants:
    """Constants for probe atlas management."""
    # Default grid size (4x4 = 16 probes)
    DEFAULT_GRID_ROWS: int = 4
    DEFAULT_GRID_COLS: int = 4
    # Default per-probe face resolution
    DEFAULT_PROBE_RESOLUTION: int = 128
    # Maximum atlas dimensions
    MAX_ATLAS_SIZE: int = 8192
    # Minimum atlas dimensions
    MIN_ATLAS_SIZE: int = 256
    # Cubemap face layout (3 wide x 2 tall per probe)
    FACES_PER_ROW: int = 3
    FACES_PER_COL: int = 2
    # Padding between probes (in pixels) to prevent bleeding
    DEFAULT_PADDING: int = 2
    # Maximum mip levels in atlas
    MAX_MIP_LEVELS: int = 8
    # Minimum resolution for mip generation
    MIN_MIP_RESOLUTION: int = 4


class AtlasFormat(Enum):
    """Atlas texture format."""
    RGBA16F = auto()  # Half-float RGBA
    RGBA32F = auto()  # Full float RGBA
    RGB16F = auto()   # Half-float RGB (no alpha)
    RGB32F = auto()   # Full float RGB (no alpha)


# -----------------------------------------------------------------------------
# Atlas Slot
# -----------------------------------------------------------------------------

@dataclass
class AtlasSlot:
    """A single slot in the probe atlas.

    Each slot contains space for one probe's 6 cubemap faces
    arranged in a 3x2 layout.

    Attributes:
        grid_x: X position in atlas grid
        grid_y: Y position in atlas grid
        probe_id: Assigned probe ID (-1 if unoccupied)
        _occupied: Whether the slot is in use
        _dirty: Whether the slot needs re-upload
    """
    grid_x: int
    grid_y: int
    probe_id: int = -1
    _occupied: bool = False
    _dirty: bool = False

    def is_occupied(self) -> bool:
        """Check if slot is currently occupied.

        Returns:
            True if slot is in use
        """
        return self._occupied

    def is_dirty(self) -> bool:
        """Check if slot needs re-upload.

        Returns:
            True if slot has pending updates
        """
        return self._dirty

    def mark_dirty(self) -> None:
        """Mark slot as needing re-upload."""
        self._dirty = True

    def mark_clean(self) -> None:
        """Mark slot as uploaded."""
        self._dirty = False

    def assign(self, probe_id: int) -> bool:
        """Assign a probe to this slot.

        Args:
            probe_id: ID of probe to assign

        Returns:
            True if assignment succeeded (slot was empty)
        """
        if self._occupied:
            return False
        self.probe_id = probe_id
        self._occupied = True
        self._dirty = True
        return True

    def release(self) -> int:
        """Release the slot and return the previous probe ID.

        Returns:
            Previous probe ID, or -1 if was unoccupied
        """
        old_id = self.probe_id
        self.probe_id = -1
        self._occupied = False
        self._dirty = False
        return old_id

    def get_face_uv(
        self,
        face: CubemapFace,
        layout: AtlasLayout,
    ) -> Tuple[float, float, float, float]:
        """Get UV coordinates for a specific face within this slot.

        The 6 cubemap faces are arranged in a 3x2 grid:
        +X | -X | +Y
        -Y | +Z | -Z

        Args:
            face: Which face to get UVs for
            layout: Atlas layout for size calculations

        Returns:
            Tuple of (u_min, v_min, u_max, v_max) in [0, 1] range
        """
        # Face offsets within probe slot (3x2 layout)
        face_offsets = {
            CubemapFace.POSITIVE_X: (0, 0),
            CubemapFace.NEGATIVE_X: (1, 0),
            CubemapFace.POSITIVE_Y: (2, 0),
            CubemapFace.NEGATIVE_Y: (0, 1),
            CubemapFace.POSITIVE_Z: (1, 1),
            CubemapFace.NEGATIVE_Z: (2, 1),
        }

        face_x, face_y = face_offsets[face]
        return layout.compute_face_uvs(self.grid_x, self.grid_y, face_x, face_y)

    def get_all_face_uvs(
        self,
        layout: AtlasLayout,
    ) -> Dict[CubemapFace, Tuple[float, float, float, float]]:
        """Get UV coordinates for all faces.

        Args:
            layout: Atlas layout for size calculations

        Returns:
            Dictionary mapping face to (u_min, v_min, u_max, v_max)
        """
        return {
            face: self.get_face_uv(face, layout)
            for face in CubemapFace
        }


# -----------------------------------------------------------------------------
# Atlas Layout
# -----------------------------------------------------------------------------

@dataclass
class AtlasLayout:
    """Defines the layout of probes within the atlas.

    Manages grid dimensions, resolution, and UV calculations.

    Attributes:
        grid_rows: Number of rows in the atlas grid
        grid_cols: Number of columns in the atlas grid
        probe_resolution: Resolution per cubemap face (pixels)
        padding: Padding between probes (pixels)
        include_mips: Whether to allocate space for mip levels
    """
    grid_rows: int = AtlasConstants.DEFAULT_GRID_ROWS
    grid_cols: int = AtlasConstants.DEFAULT_GRID_COLS
    probe_resolution: int = AtlasConstants.DEFAULT_PROBE_RESOLUTION
    padding: int = AtlasConstants.DEFAULT_PADDING
    include_mips: bool = True

    def __post_init__(self) -> None:
        """Validate layout configuration."""
        self.grid_rows = max(1, self.grid_rows)
        self.grid_cols = max(1, self.grid_cols)
        self.probe_resolution = max(
            AtlasConstants.MIN_MIP_RESOLUTION,
            self.probe_resolution
        )
        self.padding = max(0, self.padding)

    @property
    def max_probes(self) -> int:
        """Get maximum number of probes the atlas can hold."""
        return self.grid_rows * self.grid_cols

    @property
    def slot_width(self) -> int:
        """Get width of a single probe slot in pixels.

        Each probe needs 3 faces wide plus padding.
        """
        return (
            self.probe_resolution * AtlasConstants.FACES_PER_ROW +
            self.padding * 2
        )

    @property
    def slot_height(self) -> int:
        """Get height of a single probe slot in pixels.

        Each probe needs 2 faces tall plus padding.
        """
        return (
            self.probe_resolution * AtlasConstants.FACES_PER_COL +
            self.padding * 2
        )

    def get_slot(self, grid_x: int, grid_y: int) -> Optional[Tuple[int, int]]:
        """Get pixel position of a slot's top-left corner.

        Args:
            grid_x: X position in grid
            grid_y: Y position in grid

        Returns:
            Tuple of (pixel_x, pixel_y) or None if out of bounds
        """
        if not (0 <= grid_x < self.grid_cols and 0 <= grid_y < self.grid_rows):
            return None
        pixel_x = grid_x * self.slot_width
        pixel_y = grid_y * self.slot_height
        return (pixel_x, pixel_y)

    def get_total_size(self) -> Tuple[int, int]:
        """Get total atlas texture size.

        Returns:
            Tuple of (width, height) in pixels
        """
        base_width = self.grid_cols * self.slot_width
        base_height = self.grid_rows * self.slot_height

        if self.include_mips:
            # Reserve extra width for mip chain (adds ~33% width)
            mip_count = self._calculate_mip_count()
            mip_width = self._calculate_mip_chain_width()
            return (base_width + mip_width, base_height)

        return (base_width, base_height)

    def get_mip_region(self, mip_level: int) -> Optional[Tuple[int, int, int, int]]:
        """Get the region for a specific mip level.

        Args:
            mip_level: Mip level (0 = base, 1 = half, etc.)

        Returns:
            Tuple of (x, y, width, height) or None if level doesn't exist
        """
        if not self.include_mips:
            return None

        mip_count = self._calculate_mip_count()
        if mip_level >= mip_count:
            return None

        if mip_level == 0:
            # Base level is the main grid
            return (0, 0, self.grid_cols * self.slot_width, self.grid_rows * self.slot_height)

        # Mip levels are stored to the right of the base
        base_width = self.grid_cols * self.slot_width
        mip_x = base_width
        mip_y = 0

        # Each mip level is half the previous size
        for i in range(1, mip_level):
            mip_y += (self.grid_rows * self.slot_height) >> i

        mip_res = self.probe_resolution >> mip_level
        slot_w = mip_res * AtlasConstants.FACES_PER_ROW + self.padding * 2
        slot_h = mip_res * AtlasConstants.FACES_PER_COL + self.padding * 2

        return (
            mip_x,
            mip_y,
            self.grid_cols * slot_w,
            self.grid_rows * slot_h,
        )

    def _calculate_mip_count(self) -> int:
        """Calculate number of mip levels."""
        count = 0
        res = self.probe_resolution
        while res >= AtlasConstants.MIN_MIP_RESOLUTION:
            count += 1
            res //= 2
        return min(count, AtlasConstants.MAX_MIP_LEVELS)

    def _calculate_mip_chain_width(self) -> int:
        """Calculate total width needed for mip chain."""
        if not self.include_mips:
            return 0

        # Mips are stored vertically stacked to the right
        # Width is the width of mip level 1 (half base)
        mip_res = self.probe_resolution // 2
        slot_w = mip_res * AtlasConstants.FACES_PER_ROW + self.padding * 2
        return self.grid_cols * slot_w

    def compute_face_uvs(
        self,
        grid_x: int,
        grid_y: int,
        face_x: int,
        face_y: int,
        mip_level: int = 0,
    ) -> Tuple[float, float, float, float]:
        """Compute UV coordinates for a face.

        Args:
            grid_x: Probe X position in grid
            grid_y: Probe Y position in grid
            face_x: Face X offset within probe (0-2)
            face_y: Face Y offset within probe (0-1)
            mip_level: Mip level (0 = base)

        Returns:
            Tuple of (u_min, v_min, u_max, v_max) in [0, 1] range
        """
        atlas_width, atlas_height = self.get_total_size()

        # Calculate resolution at this mip level
        mip_res = self.probe_resolution >> mip_level
        mip_slot_w = mip_res * AtlasConstants.FACES_PER_ROW + self.padding * 2
        mip_slot_h = mip_res * AtlasConstants.FACES_PER_COL + self.padding * 2

        # Calculate base position
        if mip_level == 0:
            base_x = grid_x * self.slot_width + self.padding
            base_y = grid_y * self.slot_height + self.padding
        else:
            # Mip levels are stored to the right
            mip_region = self.get_mip_region(mip_level)
            if mip_region is None:
                # Fall back to base level
                base_x = grid_x * self.slot_width + self.padding
                base_y = grid_y * self.slot_height + self.padding
                mip_res = self.probe_resolution
            else:
                base_x = mip_region[0] + grid_x * mip_slot_w + self.padding
                base_y = mip_region[1] + grid_y * mip_slot_h + self.padding

        # Add face offset
        face_px_x = base_x + face_x * mip_res
        face_px_y = base_y + face_y * mip_res

        # Convert to UV coordinates
        u_min = face_px_x / atlas_width
        v_min = face_px_y / atlas_height
        u_max = (face_px_x + mip_res) / atlas_width
        v_max = (face_px_y + mip_res) / atlas_height

        return (u_min, v_min, u_max, v_max)

    def grid_to_slot_index(self, grid_x: int, grid_y: int) -> int:
        """Convert grid position to linear slot index.

        Args:
            grid_x: X position in grid
            grid_y: Y position in grid

        Returns:
            Linear slot index
        """
        return grid_y * self.grid_cols + grid_x

    def slot_index_to_grid(self, index: int) -> Tuple[int, int]:
        """Convert linear slot index to grid position.

        Args:
            index: Linear slot index

        Returns:
            Tuple of (grid_x, grid_y)
        """
        grid_y = index // self.grid_cols
        grid_x = index % self.grid_cols
        return (grid_x, grid_y)


# -----------------------------------------------------------------------------
# Atlas Config
# -----------------------------------------------------------------------------

@dataclass
class AtlasConfig:
    """Configuration for probe atlas creation.

    Attributes:
        grid_size: Tuple of (rows, cols) for the atlas grid
        probe_resolution: Resolution per cubemap face (pixels)
        include_mips: Whether to include mip levels in atlas
        format: Atlas texture format
        padding: Padding between probes (pixels)
    """
    grid_size: Tuple[int, int] = (
        AtlasConstants.DEFAULT_GRID_ROWS,
        AtlasConstants.DEFAULT_GRID_COLS,
    )
    probe_resolution: int = AtlasConstants.DEFAULT_PROBE_RESOLUTION
    include_mips: bool = True
    format: str = "rgba16f"
    padding: int = AtlasConstants.DEFAULT_PADDING

    def __post_init__(self) -> None:
        """Validate configuration."""
        if len(self.grid_size) != 2:
            self.grid_size = (
                AtlasConstants.DEFAULT_GRID_ROWS,
                AtlasConstants.DEFAULT_GRID_COLS,
            )
        self.probe_resolution = max(
            AtlasConstants.MIN_MIP_RESOLUTION,
            self.probe_resolution,
        )
        self.padding = max(0, self.padding)

    @property
    def grid_rows(self) -> int:
        """Get number of grid rows."""
        return self.grid_size[0]

    @property
    def grid_cols(self) -> int:
        """Get number of grid columns."""
        return self.grid_size[1]

    @property
    def max_probes(self) -> int:
        """Get maximum number of probes."""
        return self.grid_size[0] * self.grid_size[1]

    def to_layout(self) -> AtlasLayout:
        """Convert config to layout."""
        return AtlasLayout(
            grid_rows=self.grid_size[0],
            grid_cols=self.grid_size[1],
            probe_resolution=self.probe_resolution,
            padding=self.padding,
            include_mips=self.include_mips,
        )

    def get_atlas_format(self) -> AtlasFormat:
        """Get the atlas format enum."""
        format_map = {
            "rgba16f": AtlasFormat.RGBA16F,
            "rgba32f": AtlasFormat.RGBA32F,
            "rgb16f": AtlasFormat.RGB16F,
            "rgb32f": AtlasFormat.RGB32F,
        }
        return format_map.get(self.format.lower(), AtlasFormat.RGBA16F)


# -----------------------------------------------------------------------------
# Probe Atlas
# -----------------------------------------------------------------------------

@dataclass
class ProbeAtlas:
    """Atlas texture containing multiple probe cubemaps.

    Manages slot allocation, probe-to-slot mapping, and
    provides UV coordinates for shader access.

    Attributes:
        config: Atlas configuration
        layout: Atlas layout
        _slots: Grid of atlas slots
        _probe_to_slot: Mapping of probe ID to slot
        _texture_data: Raw texture data (if maintained on CPU)
    """
    config: AtlasConfig = field(default_factory=AtlasConfig)
    layout: AtlasLayout = field(default=None)
    _slots: List[List[AtlasSlot]] = field(default_factory=list, repr=False)
    _probe_to_slot: Dict[int, AtlasSlot] = field(default_factory=dict, repr=False)
    _texture_data: Optional[List[HDRPixel]] = field(default=None, repr=False)
    _free_slots: List[AtlasSlot] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Initialize atlas structures."""
        if self.layout is None:
            self.layout = self.config.to_layout()

        # Initialize slot grid
        if not self._slots:
            self._init_slots()

    def _init_slots(self) -> None:
        """Initialize the slot grid."""
        self._slots = []
        self._free_slots = []

        for y in range(self.layout.grid_rows):
            row = []
            for x in range(self.layout.grid_cols):
                slot = AtlasSlot(grid_x=x, grid_y=y)
                row.append(slot)
                self._free_slots.append(slot)
            self._slots.append(row)

    def _init_texture(self) -> None:
        """Initialize CPU-side texture data."""
        width, height = self.layout.get_total_size()
        self._texture_data = [HDRPixel() for _ in range(width * height)]

    @property
    def width(self) -> int:
        """Get atlas texture width."""
        return self.layout.get_total_size()[0]

    @property
    def height(self) -> int:
        """Get atlas texture height."""
        return self.layout.get_total_size()[1]

    @property
    def allocated_count(self) -> int:
        """Get number of allocated probe slots."""
        return len(self._probe_to_slot)

    @property
    def free_count(self) -> int:
        """Get number of free slots."""
        return len(self._free_slots)

    @property
    def capacity(self) -> int:
        """Get total slot capacity."""
        return self.layout.max_probes

    def is_full(self) -> bool:
        """Check if atlas is at capacity.

        Returns:
            True if no free slots available
        """
        return len(self._free_slots) == 0

    def has_probe(self, probe_id: int) -> bool:
        """Check if a probe is allocated in the atlas.

        Args:
            probe_id: ID of probe to check

        Returns:
            True if probe is in atlas
        """
        return probe_id in self._probe_to_slot

    def allocate_probe(self, probe_id: int) -> Optional[AtlasSlot]:
        """Allocate a slot for a probe.

        Args:
            probe_id: ID of probe to allocate

        Returns:
            Allocated slot, or None if atlas is full or probe already exists
        """
        if probe_id in self._probe_to_slot:
            return None  # Already allocated

        if not self._free_slots:
            return None  # Atlas is full

        slot = self._free_slots.pop(0)
        slot.assign(probe_id)
        self._probe_to_slot[probe_id] = slot
        return slot

    def deallocate_probe(self, probe_id: int) -> bool:
        """Deallocate a probe's slot.

        Args:
            probe_id: ID of probe to deallocate

        Returns:
            True if probe was deallocated
        """
        if probe_id not in self._probe_to_slot:
            return False

        slot = self._probe_to_slot.pop(probe_id)
        slot.release()
        self._free_slots.append(slot)
        return True

    def get_slot(self, probe_id: int) -> Optional[AtlasSlot]:
        """Get the slot for a probe.

        Args:
            probe_id: ID of probe

        Returns:
            Slot, or None if probe not in atlas
        """
        return self._probe_to_slot.get(probe_id)

    def get_slot_at(self, grid_x: int, grid_y: int) -> Optional[AtlasSlot]:
        """Get slot at grid position.

        Args:
            grid_x: X position in grid
            grid_y: Y position in grid

        Returns:
            Slot at position, or None if out of bounds
        """
        if not (0 <= grid_x < self.layout.grid_cols and
                0 <= grid_y < self.layout.grid_rows):
            return None
        return self._slots[grid_y][grid_x]

    def get_probe_uvs(
        self,
        probe_id: int,
        mip_level: int = 0,
    ) -> Optional[Dict[CubemapFace, Tuple[float, float, float, float]]]:
        """Get UV coordinates for all faces of a probe.

        Args:
            probe_id: ID of probe
            mip_level: Mip level (0 = base)

        Returns:
            Dictionary mapping face to UV bounds, or None if probe not found
        """
        slot = self._probe_to_slot.get(probe_id)
        if slot is None:
            return None

        result = {}
        for face in CubemapFace:
            # Get face offset
            face_offsets = {
                CubemapFace.POSITIVE_X: (0, 0),
                CubemapFace.NEGATIVE_X: (1, 0),
                CubemapFace.POSITIVE_Y: (2, 0),
                CubemapFace.NEGATIVE_Y: (0, 1),
                CubemapFace.POSITIVE_Z: (1, 1),
                CubemapFace.NEGATIVE_Z: (2, 1),
            }
            face_x, face_y = face_offsets[face]
            uvs = self.layout.compute_face_uvs(
                slot.grid_x, slot.grid_y, face_x, face_y, mip_level
            )
            result[face] = uvs

        return result

    def get_face_uvs(
        self,
        probe_id: int,
        face: CubemapFace,
        mip_level: int = 0,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Get UV coordinates for a specific face of a probe.

        Args:
            probe_id: ID of probe
            face: Which face
            mip_level: Mip level (0 = base)

        Returns:
            UV bounds (u_min, v_min, u_max, v_max), or None if not found
        """
        slot = self._probe_to_slot.get(probe_id)
        if slot is None:
            return None

        face_offsets = {
            CubemapFace.POSITIVE_X: (0, 0),
            CubemapFace.NEGATIVE_X: (1, 0),
            CubemapFace.POSITIVE_Y: (2, 0),
            CubemapFace.NEGATIVE_Y: (0, 1),
            CubemapFace.POSITIVE_Z: (1, 1),
            CubemapFace.NEGATIVE_Z: (2, 1),
        }
        face_x, face_y = face_offsets[face]
        return self.layout.compute_face_uvs(
            slot.grid_x, slot.grid_y, face_x, face_y, mip_level
        )

    def update_slot(
        self,
        probe_id: int,
        cubemap: CubemapData,
        mip_level: int = 0,
    ) -> bool:
        """Update a slot with cubemap data.

        Args:
            probe_id: ID of probe to update
            cubemap: Cubemap data to copy
            mip_level: Mip level to update

        Returns:
            True if update succeeded
        """
        slot = self._probe_to_slot.get(probe_id)
        if slot is None:
            return False

        # Ensure texture data exists
        if self._texture_data is None:
            self._init_texture()

        # Copy each face to the atlas
        for face in CubemapFace:
            face_data = cubemap.get_face(face)
            self._copy_face_to_atlas(slot, face, face_data, mip_level)

        slot.mark_dirty()
        return True

    def _copy_face_to_atlas(
        self,
        slot: AtlasSlot,
        face: CubemapFace,
        face_data: CubemapFaceData,
        mip_level: int = 0,
    ) -> None:
        """Copy face data to atlas texture.

        Args:
            slot: Target slot
            face: Which face
            face_data: Face pixel data
            mip_level: Mip level
        """
        if self._texture_data is None:
            return

        # Get UV coordinates (pixel position)
        face_offsets = {
            CubemapFace.POSITIVE_X: (0, 0),
            CubemapFace.NEGATIVE_X: (1, 0),
            CubemapFace.POSITIVE_Y: (2, 0),
            CubemapFace.NEGATIVE_Y: (0, 1),
            CubemapFace.POSITIVE_Z: (1, 1),
            CubemapFace.NEGATIVE_Z: (2, 1),
        }
        face_x, face_y = face_offsets[face]

        # Calculate pixel position
        mip_res = self.layout.probe_resolution >> mip_level

        if mip_level == 0:
            base_x = slot.grid_x * self.layout.slot_width + self.layout.padding
            base_y = slot.grid_y * self.layout.slot_height + self.layout.padding
        else:
            mip_region = self.layout.get_mip_region(mip_level)
            if mip_region is None:
                return
            mip_slot_w = mip_res * AtlasConstants.FACES_PER_ROW + self.layout.padding * 2
            mip_slot_h = mip_res * AtlasConstants.FACES_PER_COL + self.layout.padding * 2
            base_x = mip_region[0] + slot.grid_x * mip_slot_w + self.layout.padding
            base_y = mip_region[1] + slot.grid_y * mip_slot_h + self.layout.padding

        start_x = base_x + face_x * mip_res
        start_y = base_y + face_y * mip_res

        atlas_width = self.width

        # Copy pixels
        for y in range(min(face_data.resolution, mip_res)):
            for x in range(min(face_data.resolution, mip_res)):
                pixel = face_data.get_pixel(x, y)
                atlas_idx = (start_y + y) * atlas_width + (start_x + x)
                if 0 <= atlas_idx < len(self._texture_data):
                    self._texture_data[atlas_idx] = pixel

    def get_dirty_slots(self) -> List[AtlasSlot]:
        """Get all slots that need re-upload.

        Returns:
            List of dirty slots
        """
        dirty = []
        for row in self._slots:
            for slot in row:
                if slot.is_dirty():
                    dirty.append(slot)
        return dirty

    def clear_dirty_flags(self) -> None:
        """Clear dirty flags on all slots."""
        for row in self._slots:
            for slot in row:
                slot.mark_clean()

    def clear(self) -> None:
        """Deallocate all probes and reset atlas."""
        for probe_id in list(self._probe_to_slot.keys()):
            self.deallocate_probe(probe_id)

        if self._texture_data is not None:
            self._texture_data = [HDRPixel() for _ in range(len(self._texture_data))]

    def get_texture_data(self) -> Optional[List[HDRPixel]]:
        """Get raw texture data for GPU upload.

        Returns:
            List of HDRPixel, or None if not initialized
        """
        return self._texture_data

    def get_allocated_probes(self) -> List[int]:
        """Get list of allocated probe IDs.

        Returns:
            List of probe IDs
        """
        return list(self._probe_to_slot.keys())


# -----------------------------------------------------------------------------
# Atlas Updater
# -----------------------------------------------------------------------------

@dataclass
class PendingUpdate:
    """A pending update for the atlas.

    Attributes:
        probe_id: ID of probe to update
        face: Face to update (None for all faces)
        face_data: Face data (None if using cubemap)
        cubemap: Full cubemap data (None if using face)
        mip_level: Mip level to update
    """
    probe_id: int
    face: Optional[CubemapFace] = None
    face_data: Optional[CubemapFaceData] = None
    cubemap: Optional[CubemapData] = None
    mip_level: int = 0


class AtlasUpdater:
    """Handles batched updates to the probe atlas.

    Collects updates and flushes them efficiently.

    Attributes:
        _atlas: Target probe atlas
        _pending: List of pending updates
        _batch_size: Maximum updates per flush
    """

    def __init__(
        self,
        atlas: ProbeAtlas,
        batch_size: int = 16,
    ) -> None:
        """Initialize updater.

        Args:
            atlas: Target probe atlas
            batch_size: Maximum updates per flush
        """
        self._atlas = atlas
        self._pending: List[PendingUpdate] = []
        self._batch_size = batch_size

    @property
    def pending_count(self) -> int:
        """Get number of pending updates."""
        return len(self._pending)

    def update_probe(
        self,
        probe_id: int,
        cubemap: CubemapData,
        mip_level: int = 0,
    ) -> bool:
        """Queue a full probe update.

        Args:
            probe_id: ID of probe to update
            cubemap: Cubemap data
            mip_level: Mip level to update

        Returns:
            True if update was queued
        """
        if not self._atlas.has_probe(probe_id):
            return False

        update = PendingUpdate(
            probe_id=probe_id,
            cubemap=cubemap,
            mip_level=mip_level,
        )
        self._pending.append(update)

        # Auto-flush if batch is full
        if len(self._pending) >= self._batch_size:
            self.flush_updates()

        return True

    def update_face(
        self,
        probe_id: int,
        face: CubemapFace,
        face_data: CubemapFaceData,
        mip_level: int = 0,
    ) -> bool:
        """Queue a single face update.

        Args:
            probe_id: ID of probe to update
            face: Which face to update
            face_data: Face pixel data
            mip_level: Mip level to update

        Returns:
            True if update was queued
        """
        if not self._atlas.has_probe(probe_id):
            return False

        update = PendingUpdate(
            probe_id=probe_id,
            face=face,
            face_data=face_data,
            mip_level=mip_level,
        )
        self._pending.append(update)

        # Auto-flush if batch is full
        if len(self._pending) >= self._batch_size:
            self.flush_updates()

        return True

    def update_mip_chain(
        self,
        probe_id: int,
        mip_chain: CubemapMipChain,
    ) -> bool:
        """Queue updates for all mip levels.

        Args:
            probe_id: ID of probe to update
            mip_chain: Full mip chain data

        Returns:
            True if updates were queued
        """
        if not self._atlas.has_probe(probe_id):
            return False

        for mip_level, mip in enumerate(mip_chain.mips):
            update = PendingUpdate(
                probe_id=probe_id,
                cubemap=mip.cubemap,
                mip_level=mip_level,
            )
            self._pending.append(update)

        # Auto-flush if batch is full
        if len(self._pending) >= self._batch_size:
            self.flush_updates()

        return True

    def flush_updates(self) -> int:
        """Apply all pending updates to the atlas.

        Returns:
            Number of updates applied
        """
        count = 0

        for update in self._pending:
            if update.cubemap is not None:
                # Full cubemap update
                success = self._atlas.update_slot(
                    update.probe_id,
                    update.cubemap,
                    update.mip_level,
                )
                if success:
                    count += 1
            elif update.face is not None and update.face_data is not None:
                # Single face update
                slot = self._atlas.get_slot(update.probe_id)
                if slot is not None:
                    self._atlas._copy_face_to_atlas(
                        slot,
                        update.face,
                        update.face_data,
                        update.mip_level,
                    )
                    slot.mark_dirty()
                    count += 1

        self._pending.clear()
        return count

    def clear_pending(self) -> None:
        """Clear all pending updates without applying them."""
        self._pending.clear()


# -----------------------------------------------------------------------------
# Atlas Sampler
# -----------------------------------------------------------------------------

class AtlasSampler:
    """Samples probes from the atlas texture.

    Handles bilinear filtering and roughness-based mip sampling.

    Attributes:
        _atlas: Source probe atlas
    """

    def __init__(self, atlas: ProbeAtlas) -> None:
        """Initialize sampler.

        Args:
            atlas: Source probe atlas
        """
        self._atlas = atlas

    def sample(
        self,
        probe_id: int,
        face: CubemapFace,
        u: float,
        v: float,
    ) -> HDRPixel:
        """Sample a probe face at UV coordinates.

        Args:
            probe_id: ID of probe to sample
            face: Which face to sample
            u: U coordinate [0, 1]
            v: V coordinate [0, 1]

        Returns:
            Sampled HDR pixel
        """
        return self._sample_face(probe_id, face, u, v, 0)

    def sample_with_roughness(
        self,
        probe_id: int,
        face: CubemapFace,
        u: float,
        v: float,
        roughness: float,
    ) -> HDRPixel:
        """Sample with roughness-based mip selection.

        Args:
            probe_id: ID of probe to sample
            face: Which face to sample
            u: U coordinate [0, 1]
            v: V coordinate [0, 1]
            roughness: Surface roughness [0, 1]

        Returns:
            Sampled HDR pixel (trilinear filtered between mips)
        """
        if roughness <= 0.0:
            return self._sample_face(probe_id, face, u, v, 0)

        # Calculate mip level from roughness
        # Roughness of 1.0 maps to highest mip level
        max_mip = self._calculate_max_mip()
        mip_float = roughness * max_mip

        mip_low = int(mip_float)
        mip_high = min(mip_low + 1, max_mip)
        mip_frac = mip_float - mip_low

        # Sample both mip levels and blend
        sample_low = self._sample_face(probe_id, face, u, v, mip_low)
        if mip_frac < 0.001:
            return sample_low

        sample_high = self._sample_face(probe_id, face, u, v, mip_high)

        # Linear blend between mip levels
        return HDRPixel(
            sample_low.r * (1 - mip_frac) + sample_high.r * mip_frac,
            sample_low.g * (1 - mip_frac) + sample_high.g * mip_frac,
            sample_low.b * (1 - mip_frac) + sample_high.b * mip_frac,
        )

    def sample_direction(
        self,
        probe_id: int,
        direction: Vec3,
        roughness: float = 0.0,
    ) -> HDRPixel:
        """Sample probe using a world-space direction.

        Args:
            probe_id: ID of probe to sample
            direction: World-space direction vector
            roughness: Surface roughness [0, 1]

        Returns:
            Sampled HDR pixel
        """
        # Determine which face to sample
        face, u, v = self._direction_to_face_uv(direction)

        if roughness > 0.0:
            return self.sample_with_roughness(probe_id, face, u, v, roughness)
        return self.sample(probe_id, face, u, v)

    def _sample_face(
        self,
        probe_id: int,
        face: CubemapFace,
        u: float,
        v: float,
        mip_level: int,
    ) -> HDRPixel:
        """Internal face sampling with bilinear filtering.

        Args:
            probe_id: ID of probe
            face: Which face
            u: U coordinate [0, 1]
            v: V coordinate [0, 1]
            mip_level: Mip level

        Returns:
            Bilinear sampled pixel
        """
        texture_data = self._atlas.get_texture_data()
        if texture_data is None:
            return HDRPixel()

        # Get UV bounds for this face
        uvs = self._atlas.get_face_uvs(probe_id, face, mip_level)
        if uvs is None:
            return HDRPixel()

        u_min, v_min, u_max, v_max = uvs

        # Map local UV to atlas UV
        atlas_u = u_min + u * (u_max - u_min)
        atlas_v = v_min + v * (v_max - v_min)

        # Convert to pixel coordinates
        atlas_width = self._atlas.width
        atlas_height = self._atlas.height

        px = atlas_u * (atlas_width - 1)
        py = atlas_v * (atlas_height - 1)

        # Bilinear sample
        x0 = int(px)
        y0 = int(py)
        x1 = min(x0 + 1, atlas_width - 1)
        y1 = min(y0 + 1, atlas_height - 1)

        fx = px - x0
        fy = py - y0

        # Clamp to valid range
        x0 = max(0, min(x0, atlas_width - 1))
        y0 = max(0, min(y0, atlas_height - 1))

        # Get four surrounding pixels
        p00 = texture_data[y0 * atlas_width + x0]
        p10 = texture_data[y0 * atlas_width + x1]
        p01 = texture_data[y1 * atlas_width + x0]
        p11 = texture_data[y1 * atlas_width + x1]

        # Bilinear interpolation
        top = HDRPixel(
            p00.r * (1 - fx) + p10.r * fx,
            p00.g * (1 - fx) + p10.g * fx,
            p00.b * (1 - fx) + p10.b * fx,
        )
        bottom = HDRPixel(
            p01.r * (1 - fx) + p11.r * fx,
            p01.g * (1 - fx) + p11.g * fx,
            p01.b * (1 - fx) + p11.b * fx,
        )

        return HDRPixel(
            top.r * (1 - fy) + bottom.r * fy,
            top.g * (1 - fy) + bottom.g * fy,
            top.b * (1 - fy) + bottom.b * fy,
        )

    def _calculate_max_mip(self) -> int:
        """Calculate maximum mip level in atlas."""
        if not self._atlas.layout.include_mips:
            return 0

        count = 0
        res = self._atlas.layout.probe_resolution
        while res >= AtlasConstants.MIN_MIP_RESOLUTION:
            count += 1
            res //= 2
        return max(0, count - 1)

    def _direction_to_face_uv(
        self,
        direction: Vec3,
    ) -> Tuple[CubemapFace, float, float]:
        """Convert direction to face and UV coordinates.

        Args:
            direction: World-space direction

        Returns:
            Tuple of (face, u, v)
        """
        # Normalize direction
        length = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
        if length < 0.0001:
            return (CubemapFace.POSITIVE_Z, 0.5, 0.5)

        d = Vec3(direction.x / length, direction.y / length, direction.z / length)

        # Find dominant axis
        abs_x = abs(d.x)
        abs_y = abs(d.y)
        abs_z = abs(d.z)

        if abs_x >= abs_y and abs_x >= abs_z:
            # X is dominant
            if d.x > 0:
                face = CubemapFace.POSITIVE_X
                u = (-d.z / abs_x + 1) / 2
                v = (-d.y / abs_x + 1) / 2
            else:
                face = CubemapFace.NEGATIVE_X
                u = (d.z / abs_x + 1) / 2
                v = (-d.y / abs_x + 1) / 2
        elif abs_y >= abs_x and abs_y >= abs_z:
            # Y is dominant
            if d.y > 0:
                face = CubemapFace.POSITIVE_Y
                u = (d.x / abs_y + 1) / 2
                v = (d.z / abs_y + 1) / 2
            else:
                face = CubemapFace.NEGATIVE_Y
                u = (d.x / abs_y + 1) / 2
                v = (-d.z / abs_y + 1) / 2
        else:
            # Z is dominant
            if d.z > 0:
                face = CubemapFace.POSITIVE_Z
                u = (d.x / abs_z + 1) / 2
                v = (-d.y / abs_z + 1) / 2
            else:
                face = CubemapFace.NEGATIVE_Z
                u = (-d.x / abs_z + 1) / 2
                v = (-d.y / abs_z + 1) / 2

        # Clamp UVs
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))

        return (face, u, v)


# -----------------------------------------------------------------------------
# Atlas Manager
# -----------------------------------------------------------------------------

class ProbeAtlasManager:
    """High-level manager for probe atlases.

    Handles multiple atlases, automatic allocation, and updates.
    """

    def __init__(
        self,
        default_config: Optional[AtlasConfig] = None,
    ) -> None:
        """Initialize atlas manager.

        Args:
            default_config: Default config for new atlases
        """
        self._default_config = default_config or AtlasConfig()
        self._atlases: List[ProbeAtlas] = []
        self._probe_to_atlas: Dict[int, ProbeAtlas] = {}
        self._updaters: Dict[int, AtlasUpdater] = {}

    @property
    def atlas_count(self) -> int:
        """Get number of atlases."""
        return len(self._atlases)

    @property
    def total_capacity(self) -> int:
        """Get total probe capacity across all atlases."""
        return sum(atlas.capacity for atlas in self._atlases)

    @property
    def total_allocated(self) -> int:
        """Get total allocated probes across all atlases."""
        return sum(atlas.allocated_count for atlas in self._atlases)

    def create_atlas(
        self,
        config: Optional[AtlasConfig] = None,
    ) -> ProbeAtlas:
        """Create a new atlas.

        Args:
            config: Atlas configuration (uses default if None)

        Returns:
            Created atlas
        """
        atlas_config = config or self._default_config
        atlas = ProbeAtlas(config=atlas_config)
        self._atlases.append(atlas)
        return atlas

    def allocate_probe(self, probe_id: int) -> Optional[AtlasSlot]:
        """Allocate a probe in any available atlas.

        Creates a new atlas if needed.

        Args:
            probe_id: ID of probe to allocate

        Returns:
            Allocated slot, or None if probe already exists
        """
        if probe_id in self._probe_to_atlas:
            return None  # Already allocated

        # Try existing atlases
        for atlas in self._atlases:
            if not atlas.is_full():
                slot = atlas.allocate_probe(probe_id)
                if slot is not None:
                    self._probe_to_atlas[probe_id] = atlas
                    return slot

        # Create new atlas
        atlas = self.create_atlas()
        slot = atlas.allocate_probe(probe_id)
        if slot is not None:
            self._probe_to_atlas[probe_id] = atlas
        return slot

    def deallocate_probe(self, probe_id: int) -> bool:
        """Deallocate a probe.

        Args:
            probe_id: ID of probe to deallocate

        Returns:
            True if probe was deallocated
        """
        atlas = self._probe_to_atlas.get(probe_id)
        if atlas is None:
            return False

        success = atlas.deallocate_probe(probe_id)
        if success:
            del self._probe_to_atlas[probe_id]
        return success

    def get_probe_atlas(self, probe_id: int) -> Optional[ProbeAtlas]:
        """Get the atlas containing a probe.

        Args:
            probe_id: ID of probe

        Returns:
            Atlas containing the probe, or None
        """
        return self._probe_to_atlas.get(probe_id)

    def get_probe_uvs(
        self,
        probe_id: int,
        mip_level: int = 0,
    ) -> Optional[Dict[CubemapFace, Tuple[float, float, float, float]]]:
        """Get UV coordinates for a probe.

        Args:
            probe_id: ID of probe
            mip_level: Mip level

        Returns:
            UV mappings, or None if probe not found
        """
        atlas = self._probe_to_atlas.get(probe_id)
        if atlas is None:
            return None
        return atlas.get_probe_uvs(probe_id, mip_level)

    def update_probe(
        self,
        probe_id: int,
        cubemap: CubemapData,
        mip_level: int = 0,
    ) -> bool:
        """Update a probe in its atlas.

        Args:
            probe_id: ID of probe
            cubemap: Cubemap data
            mip_level: Mip level

        Returns:
            True if update succeeded
        """
        atlas = self._probe_to_atlas.get(probe_id)
        if atlas is None:
            return False
        return atlas.update_slot(probe_id, cubemap, mip_level)

    def get_updater(self, atlas: ProbeAtlas) -> AtlasUpdater:
        """Get or create an updater for an atlas.

        Args:
            atlas: Target atlas

        Returns:
            Atlas updater
        """
        atlas_id = id(atlas)
        if atlas_id not in self._updaters:
            self._updaters[atlas_id] = AtlasUpdater(atlas)
        return self._updaters[atlas_id]

    def flush_all_updates(self) -> int:
        """Flush all pending updates across all atlases.

        Returns:
            Total number of updates applied
        """
        total = 0
        for updater in self._updaters.values():
            total += updater.flush_updates()
        return total

    def clear(self) -> None:
        """Clear all atlases and probes."""
        for atlas in self._atlases:
            atlas.clear()
        self._atlases.clear()
        self._probe_to_atlas.clear()
        self._updaters.clear()

    def get_all_atlases(self) -> List[ProbeAtlas]:
        """Get all managed atlases.

        Returns:
            List of atlases
        """
        return list(self._atlases)

    def create_sampler(self, atlas: ProbeAtlas) -> AtlasSampler:
        """Create a sampler for an atlas.

        Args:
            atlas: Target atlas

        Returns:
            Atlas sampler
        """
        return AtlasSampler(atlas)
