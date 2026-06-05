"""
Image Widget - Texture display with scaling, tinting, and atlas support.

Provides flexible image rendering with multiple scale modes including
nine-slice for scalable UI frames and atlas UV coordinates for sprite sheets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union


class ScaleMode(Enum):
    """Image scaling modes for rendering."""
    STRETCH = auto()      # Stretch to fill bounds (may distort)
    FIT = auto()          # Scale uniformly to fit within bounds (letterbox)
    FILL = auto()         # Scale uniformly to cover bounds (may crop)
    TILE = auto()         # Repeat image to fill bounds
    NINE_SLICE = auto()   # Nine-slice scaling for UI frames


class UVCoordinates:
    """UV coordinates for texture atlas regions.

    Coordinates are normalized (0.0 to 1.0) representing position
    within the source texture.

    Attributes:
        u0: Left edge U coordinate
        v0: Top edge V coordinate
        u1: Right edge U coordinate
        v1: Bottom edge V coordinate
    """
    __slots__ = ('u0', 'v0', 'u1', 'v1', '_skip_validation')

    def __init__(
        self,
        u0: float = 0.0,
        v0: float = 0.0,
        u1: float = 1.0,
        v1: float = 1.0,
        _skip_validation: bool = False,
    ) -> None:
        """Initialize UV coordinates.

        Args:
            u0: Left edge U coordinate
            v0: Top edge V coordinate
            u1: Right edge U coordinate
            v1: Bottom edge V coordinate
            _skip_validation: Internal flag to skip validation for flips
        """
        object.__setattr__(self, 'u0', u0)
        object.__setattr__(self, 'v0', v0)
        object.__setattr__(self, 'u1', u1)
        object.__setattr__(self, 'v1', v1)
        object.__setattr__(self, '_skip_validation', _skip_validation)

        if not _skip_validation:
            if not (0.0 <= u0 <= 1.0):
                raise ValueError(f"u0 must be in range [0, 1], got {u0}")
            if not (0.0 <= v0 <= 1.0):
                raise ValueError(f"v0 must be in range [0, 1], got {v0}")
            if not (0.0 <= u1 <= 1.0):
                raise ValueError(f"u1 must be in range [0, 1], got {u1}")
            if not (0.0 <= v1 <= 1.0):
                raise ValueError(f"v1 must be in range [0, 1], got {v1}")
            if u1 < u0:
                raise ValueError(f"u1 ({u1}) must be >= u0 ({u0})")
            if v1 < v0:
                raise ValueError(f"v1 ({v1}) must be >= v0 ({v0})")

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent attribute modification (frozen-like behavior)."""
        raise AttributeError("UVCoordinates is immutable")

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, UVCoordinates):
            return NotImplemented
        return (
            self.u0 == other.u0 and
            self.v0 == other.v0 and
            self.u1 == other.u1 and
            self.v1 == other.v1
        )

    def __hash__(self) -> int:
        """Return hash."""
        return hash((self.u0, self.v0, self.u1, self.v1))

    def __repr__(self) -> str:
        """Return string representation."""
        return f"UVCoordinates(u0={self.u0}, v0={self.v0}, u1={self.u1}, v1={self.v1})"

    @property
    def width(self) -> float:
        """Get UV width."""
        return self.u1 - self.u0

    @property
    def height(self) -> float:
        """Get UV height."""
        return self.v1 - self.v0

    def flip_horizontal(self) -> "UVCoordinates":
        """Return UV coordinates flipped horizontally."""
        return UVCoordinates(
            u0=self.u1,
            v0=self.v0,
            u1=self.u0,
            v1=self.v1,
            _skip_validation=True,
        )

    def flip_vertical(self) -> "UVCoordinates":
        """Return UV coordinates flipped vertically."""
        return UVCoordinates(
            u0=self.u0,
            v0=self.v1,
            u1=self.u1,
            v1=self.v0,
            _skip_validation=True,
        )

    @classmethod
    def from_pixel_rect(
        cls,
        x: int,
        y: int,
        width: int,
        height: int,
        texture_width: int,
        texture_height: int,
    ) -> "UVCoordinates":
        """Create UV coordinates from pixel rectangle on texture.

        Args:
            x: Left edge in pixels
            y: Top edge in pixels
            width: Width in pixels
            height: Height in pixels
            texture_width: Total texture width in pixels
            texture_height: Total texture height in pixels

        Returns:
            Normalized UV coordinates
        """
        if texture_width <= 0 or texture_height <= 0:
            raise ValueError("Texture dimensions must be positive")
        if width < 0 or height < 0:
            raise ValueError("Width and height must be non-negative")

        return cls(
            u0=x / texture_width,
            v0=y / texture_height,
            u1=(x + width) / texture_width,
            v1=(y + height) / texture_height,
        )


@dataclass
class NineSliceConfig:
    """Configuration for nine-slice image scaling.

    Nine-slice divides an image into 9 regions:
    - 4 corners (fixed size)
    - 4 edges (stretch in one direction)
    - 1 center (stretch in both directions)

    Attributes:
        left: Left border width in pixels
        right: Right border width in pixels
        top: Top border height in pixels
        bottom: Bottom border height in pixels
        tile_center: If True, tile center instead of stretching
        tile_edges: If True, tile edges instead of stretching
    """
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0
    tile_center: bool = False
    tile_edges: bool = False

    def __post_init__(self) -> None:
        """Validate nine-slice configuration."""
        if self.left < 0:
            raise ValueError(f"left must be >= 0, got {self.left}")
        if self.right < 0:
            raise ValueError(f"right must be >= 0, got {self.right}")
        if self.top < 0:
            raise ValueError(f"top must be >= 0, got {self.top}")
        if self.bottom < 0:
            raise ValueError(f"bottom must be >= 0, got {self.bottom}")

    @property
    def horizontal_borders(self) -> int:
        """Get total horizontal border size."""
        return self.left + self.right

    @property
    def vertical_borders(self) -> int:
        """Get total vertical border size."""
        return self.top + self.bottom

    @classmethod
    def uniform(cls, size: int, tile_center: bool = False) -> NineSliceConfig:
        """Create uniform nine-slice with same border on all sides."""
        return cls(
            left=size,
            right=size,
            top=size,
            bottom=size,
            tile_center=tile_center,
        )


def _validate_color(value: Any) -> Tuple[float, float, float, float]:
    """Validate and normalize color value.

    Accepts:
    - Tuple of 3 floats (RGB, adds alpha=1.0)
    - Tuple of 4 floats (RGBA)
    - Hex string (#RGB, #RGBA, #RRGGBB, #RRGGBBAA)

    Returns:
        Tuple of 4 floats (RGBA) in range [0, 1]
    """
    if isinstance(value, str):
        return _parse_hex_color(value)
    elif isinstance(value, (tuple, list)):
        if len(value) == 3:
            r, g, b = value
            a = 1.0
        elif len(value) == 4:
            r, g, b, a = value
        else:
            raise ValueError(f"Color must have 3 or 4 components, got {len(value)}")

        # Validate ranges
        for component, name in [(r, "red"), (g, "green"), (b, "blue"), (a, "alpha")]:
            if not isinstance(component, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not (0.0 <= component <= 1.0):
                raise ValueError(f"{name} must be in range [0, 1], got {component}")

        return (float(r), float(g), float(b), float(a))
    else:
        raise ValueError(f"Invalid color type: {type(value)}")


def _parse_hex_color(hex_str: str) -> Tuple[float, float, float, float]:
    """Parse hex color string to RGBA tuple."""
    if not hex_str.startswith("#"):
        raise ValueError("Hex color must start with #")

    hex_str = hex_str[1:]
    length = len(hex_str)

    if length == 3:  # #RGB
        r = int(hex_str[0] * 2, 16) / 255.0
        g = int(hex_str[1] * 2, 16) / 255.0
        b = int(hex_str[2] * 2, 16) / 255.0
        a = 1.0
    elif length == 4:  # #RGBA
        r = int(hex_str[0] * 2, 16) / 255.0
        g = int(hex_str[1] * 2, 16) / 255.0
        b = int(hex_str[2] * 2, 16) / 255.0
        a = int(hex_str[3] * 2, 16) / 255.0
    elif length == 6:  # #RRGGBB
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        a = 1.0
    elif length == 8:  # #RRGGBBAA
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        a = int(hex_str[6:8], 16) / 255.0
    else:
        raise ValueError(f"Invalid hex color length: {length}")

    return (r, g, b, a)


class Image:
    """
    Image widget for displaying textures with various scaling modes.

    Supports texture atlas regions via UV coordinates, color tinting,
    opacity control, and nine-slice scaling for UI frames.

    Attributes:
        source: Path or ID of the source texture/atlas
        scale_mode: How the image should be scaled
        tint: Color tint applied multiplicatively (RGBA)
        opacity: Overall transparency (0.0 to 1.0)
        uv: UV coordinates for atlas region
        nine_slice: Nine-slice configuration (used when scale_mode is NINE_SLICE)
        preserve_aspect: Whether to preserve aspect ratio (for FIT/FILL modes)
        flip_horizontal: Flip image horizontally
        flip_vertical: Flip image vertically
    """

    __slots__ = (
        '_source',
        '_scale_mode',
        '_tint',
        '_opacity',
        '_uv',
        '_nine_slice',
        '_preserve_aspect',
        '_flip_horizontal',
        '_flip_vertical',
        '_width',
        '_height',
        '_natural_width',
        '_natural_height',
        '_cached_vertices',
        '_dirty',
        '_dirty_mesh',
        '_entity_id',
    )

    def __init__(
        self,
        source: str = "",
        scale_mode: ScaleMode = ScaleMode.STRETCH,
        tint: Union[Tuple[float, float, float, float], str] = (1.0, 1.0, 1.0, 1.0),
        opacity: float = 1.0,
        uv: Optional[UVCoordinates] = None,
        nine_slice: Optional[NineSliceConfig] = None,
        width: float = 0.0,
        height: float = 0.0,
        entity_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the Image widget.

        Args:
            source: Path or ID of the source texture/atlas
            scale_mode: How the image should be scaled
            tint: Color tint (RGBA tuple or hex string)
            opacity: Overall transparency (0.0 to 1.0)
            uv: UV coordinates for atlas region
            nine_slice: Nine-slice configuration
            width: Widget width
            height: Widget height
            entity_id: Optional entity ID for tracking
        """
        # Validate opacity
        if not (0.0 <= opacity <= 1.0):
            raise ValueError(f"opacity must be in range [0, 1], got {opacity}")

        self._source = source
        self._scale_mode = scale_mode
        self._tint = _validate_color(tint)
        self._opacity = opacity
        self._uv = uv or UVCoordinates()
        self._nine_slice = nine_slice
        self._preserve_aspect = True
        self._flip_horizontal = False
        self._flip_vertical = False
        self._width = max(0.0, width)
        self._height = max(0.0, height)
        self._natural_width: float = 0.0
        self._natural_height: float = 0.0
        self._cached_vertices: Optional[List[Any]] = None
        self._dirty = True
        self._dirty_mesh = True
        self._entity_id = entity_id

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def source(self) -> str:
        """Get source texture path or ID."""
        return self._source

    @source.setter
    def source(self, value: str) -> None:
        """Set source texture path or ID."""
        if self._source != value:
            self._source = value
            self._dirty = True

    @property
    def scale_mode(self) -> ScaleMode:
        """Get current scale mode."""
        return self._scale_mode

    @scale_mode.setter
    def scale_mode(self, value: ScaleMode) -> None:
        """Set scale mode."""
        if not isinstance(value, ScaleMode):
            raise ValueError(f"scale_mode must be a ScaleMode, got {type(value)}")
        if self._scale_mode != value:
            self._scale_mode = value
            self._dirty = True

    @property
    def tint(self) -> Tuple[float, float, float, float]:
        """Get current tint color (RGBA)."""
        return self._tint

    @tint.setter
    def tint(self, value: Union[Tuple[float, float, float, float], str]) -> None:
        """Set tint color."""
        validated = _validate_color(value)
        if self._tint != validated:
            self._tint = validated
            self._dirty = True

    @property
    def opacity(self) -> float:
        """Get opacity value."""
        return self._opacity

    @opacity.setter
    def opacity(self, value: float) -> None:
        """Set opacity value."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"opacity must be in range [0, 1], got {value}")
        if self._opacity != value:
            self._opacity = value
            self._dirty = True

    @property
    def uv(self) -> UVCoordinates:
        """Get current UV coordinates."""
        return self._uv

    @uv.setter
    def uv(self, value: UVCoordinates) -> None:
        """Set UV coordinates."""
        if not isinstance(value, UVCoordinates):
            raise ValueError(f"uv must be UVCoordinates, got {type(value)}")
        if self._uv != value:
            self._uv = value
            self._dirty = True

    @property
    def nine_slice(self) -> Optional[NineSliceConfig]:
        """Get nine-slice configuration."""
        return self._nine_slice

    @nine_slice.setter
    def nine_slice(self, value: Optional[NineSliceConfig]) -> None:
        """Set nine-slice configuration."""
        if value is not None and not isinstance(value, NineSliceConfig):
            raise ValueError(f"nine_slice must be NineSliceConfig, got {type(value)}")
        if self._nine_slice != value:
            self._nine_slice = value
            self._dirty = True

    @property
    def preserve_aspect(self) -> bool:
        """Get whether aspect ratio is preserved."""
        return self._preserve_aspect

    @preserve_aspect.setter
    def preserve_aspect(self, value: bool) -> None:
        """Set whether to preserve aspect ratio."""
        if self._preserve_aspect != value:
            self._preserve_aspect = bool(value)
            self._dirty = True

    @property
    def flip_horizontal(self) -> bool:
        """Get horizontal flip state."""
        return self._flip_horizontal

    @flip_horizontal.setter
    def flip_horizontal(self, value: bool) -> None:
        """Set horizontal flip."""
        if self._flip_horizontal != value:
            self._flip_horizontal = bool(value)
            self._dirty = True

    @property
    def flip_vertical(self) -> bool:
        """Get vertical flip state."""
        return self._flip_vertical

    @flip_vertical.setter
    def flip_vertical(self, value: bool) -> None:
        """Set vertical flip."""
        if self._flip_vertical != value:
            self._flip_vertical = bool(value)
            self._dirty = True

    @property
    def width(self) -> float:
        """Get widget width."""
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        """Set widget width."""
        value = max(0.0, float(value))
        if self._width != value:
            self._width = value
            self._dirty = True
            self._dirty_mesh = True

    @property
    def height(self) -> float:
        """Get widget height."""
        return self._height

    @height.setter
    def height(self, value: float) -> None:
        """Set widget height."""
        value = max(0.0, float(value))
        if self._height != value:
            self._height = value
            self._dirty = True

    @property
    def natural_width(self) -> float:
        """Get natural (source) image width."""
        return self._natural_width

    @property
    def natural_height(self) -> float:
        """Get natural (source) image height."""
        return self._natural_height

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width / height)."""
        if self._natural_height > 0:
            return self._natural_width / self._natural_height
        return 1.0

    @property
    def is_dirty(self) -> bool:
        """Check if mesh needs rebuilding."""
        return self._dirty

    @property
    def entity_id(self) -> Optional[str]:
        """Get entity ID."""
        return self._entity_id

    # =========================================================================
    # METHODS
    # =========================================================================

    def set_natural_size(self, width: float, height: float) -> None:
        """Set the natural size of the source image.

        Args:
            width: Natural width in pixels
            height: Natural height in pixels
        """
        if width < 0 or height < 0:
            raise ValueError("Natural dimensions must be non-negative")
        self._natural_width = float(width)
        self._natural_height = float(height)
        self._dirty = True

    def get_rendered_size(self) -> Tuple[float, float]:
        """Calculate the actual rendered size based on scale mode.

        Returns:
            Tuple of (width, height) after scaling
        """
        if self._width == 0 or self._height == 0:
            return (self._natural_width, self._natural_height)

        if self._scale_mode == ScaleMode.STRETCH:
            return (self._width, self._height)

        elif self._scale_mode == ScaleMode.FIT:
            if not self._preserve_aspect or self._natural_width == 0 or self._natural_height == 0:
                return (self._width, self._height)

            scale = min(
                self._width / self._natural_width,
                self._height / self._natural_height,
            )
            return (self._natural_width * scale, self._natural_height * scale)

        elif self._scale_mode == ScaleMode.FILL:
            if not self._preserve_aspect or self._natural_width == 0 or self._natural_height == 0:
                return (self._width, self._height)

            scale = max(
                self._width / self._natural_width,
                self._height / self._natural_height,
            )
            return (self._natural_width * scale, self._natural_height * scale)

        elif self._scale_mode in (ScaleMode.TILE, ScaleMode.NINE_SLICE):
            return (self._width, self._height)

        return (self._width, self._height)

    def get_effective_uv(self) -> UVCoordinates:
        """Get effective UV coordinates with flip applied.

        Returns:
            UV coordinates with horizontal/vertical flip applied
        """
        uv = self._uv
        if self._flip_horizontal:
            uv = uv.flip_horizontal()
        if self._flip_vertical:
            uv = uv.flip_vertical()
        return uv

    def clear_mesh_cache(self) -> None:
        """Clear cached mesh data."""
        self._cached_vertices = None
        self._dirty = True
        self._dirty_mesh = True

    def mark_clean(self) -> None:
        """Mark the image as clean (mesh up to date)."""
        self._dirty = False
        self._dirty_mesh = False

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize image to dictionary."""
        result: Dict[str, Any] = {
            "source": self._source,
            "scale_mode": self._scale_mode.name,
            "tint": self._tint,
            "opacity": self._opacity,
            "uv": {
                "u0": self._uv.u0,
                "v0": self._uv.v0,
                "u1": self._uv.u1,
                "v1": self._uv.v1,
            },
            "preserve_aspect": self._preserve_aspect,
            "flip_horizontal": self._flip_horizontal,
            "flip_vertical": self._flip_vertical,
            "width": self._width,
            "height": self._height,
        }

        if self._nine_slice is not None:
            result["nine_slice"] = {
                "left": self._nine_slice.left,
                "right": self._nine_slice.right,
                "top": self._nine_slice.top,
                "bottom": self._nine_slice.bottom,
                "tile_center": self._nine_slice.tile_center,
                "tile_edges": self._nine_slice.tile_edges,
            }

        if self._entity_id is not None:
            result["entity_id"] = self._entity_id

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Image:
        """Deserialize image from dictionary."""
        uv_data = data.get("uv", {})
        uv = UVCoordinates(
            u0=uv_data.get("u0", 0.0),
            v0=uv_data.get("v0", 0.0),
            u1=uv_data.get("u1", 1.0),
            v1=uv_data.get("v1", 1.0),
        )

        nine_slice = None
        if "nine_slice" in data:
            ns_data = data["nine_slice"]
            nine_slice = NineSliceConfig(
                left=ns_data.get("left", 0),
                right=ns_data.get("right", 0),
                top=ns_data.get("top", 0),
                bottom=ns_data.get("bottom", 0),
                tile_center=ns_data.get("tile_center", False),
                tile_edges=ns_data.get("tile_edges", False),
            )

        image = cls(
            source=data.get("source", ""),
            scale_mode=ScaleMode[data.get("scale_mode", "STRETCH")],
            tint=tuple(data.get("tint", (1.0, 1.0, 1.0, 1.0))),
            opacity=data.get("opacity", 1.0),
            uv=uv,
            nine_slice=nine_slice,
            width=data.get("width", 0.0),
            height=data.get("height", 0.0),
            entity_id=data.get("entity_id"),
        )

        image._preserve_aspect = data.get("preserve_aspect", True)
        image._flip_horizontal = data.get("flip_horizontal", False)
        image._flip_vertical = data.get("flip_vertical", False)

        return image

    def __repr__(self) -> str:
        return (
            f"Image(source={self._source!r}, scale_mode={self._scale_mode.name}, "
            f"size=({self._width}, {self._height}))"
        )


__all__ = [
    "Image",
    "ScaleMode",
    "NineSliceConfig",
    "UVCoordinates",
]
