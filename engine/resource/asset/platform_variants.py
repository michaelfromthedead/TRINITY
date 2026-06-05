"""Platform-specific asset variants with content-addressed storage.

Provides automatic generation of platform-optimized asset variants,
particularly for textures where different platforms require different
compression formats (BC7 for desktop, ASTC for iOS/macOS, ETC2 for Android).
"""
from __future__ import annotations

import logging
import struct
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, TypeVar

from engine.resource.asset.content_hash import (
    AssetHasher,
    ContentAddressedStorage,
    ContentHash,
    HashAlgorithm,
)

__all__ = [
    "Platform",
    "TextureFormat",
    "VariantKey",
    "VariantRule",
    "VariantRuleSet",
    "VariantGenerator",
    "VariantStorage",
    "VariantCache",
    "SourceAsset",
    "GeneratedVariant",
    "FormatConverter",
    "TextureFormatConverter",
    "DEFAULT_PLATFORM_FORMATS",
    "get_optimal_format",
    "is_format_supported",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Platform(Enum):
    """Supported target platforms."""
    WINDOWS = auto()
    LINUX = auto()
    MACOS = auto()
    IOS = auto()
    ANDROID = auto()
    WEBGL = auto()

    @classmethod
    def from_string(cls, name: str) -> Platform:
        """Create Platform from string name (case-insensitive)."""
        name_upper = name.upper()
        try:
            return cls[name_upper]
        except KeyError:
            raise ValueError(f"Unknown platform: {name}") from None

    @property
    def is_desktop(self) -> bool:
        """Return True if this is a desktop platform."""
        return self in (Platform.WINDOWS, Platform.LINUX, Platform.MACOS)

    @property
    def is_mobile(self) -> bool:
        """Return True if this is a mobile platform."""
        return self in (Platform.IOS, Platform.ANDROID)

    @property
    def is_apple(self) -> bool:
        """Return True if this is an Apple platform."""
        return self in (Platform.MACOS, Platform.IOS)

    def __str__(self) -> str:
        return self.name.lower()


class TextureFormat(Enum):
    """Texture compression formats supported across platforms."""
    # Uncompressed
    RGBA8 = auto()      # Universal fallback, 32 bits per pixel
    RGBA16F = auto()    # HDR format, 64 bits per pixel
    RGBA32F = auto()    # Full HDR, 128 bits per pixel

    # Desktop (DirectX / Vulkan on Windows/Linux)
    BC1 = auto()        # DXT1, 4 bpp, RGB with 1-bit alpha
    BC3 = auto()        # DXT5, 8 bpp, RGBA with full alpha
    BC4 = auto()        # 4 bpp, single channel
    BC5 = auto()        # 8 bpp, two channels (normal maps)
    BC6H = auto()       # 8 bpp, HDR
    BC7 = auto()        # 8 bpp, high-quality RGBA

    # Apple (iOS / macOS)
    ASTC_4x4 = auto()   # 8 bpp, highest quality
    ASTC_5x5 = auto()   # 5.12 bpp
    ASTC_6x6 = auto()   # 3.56 bpp
    ASTC_8x8 = auto()   # 2 bpp, most compressed

    # Android / OpenGL ES
    ETC1 = auto()       # 4 bpp, RGB only (legacy)
    ETC2_RGB = auto()   # 4 bpp, RGB
    ETC2_RGBA = auto()  # 8 bpp, RGBA

    @classmethod
    def from_string(cls, name: str) -> TextureFormat:
        """Create TextureFormat from string name (case-insensitive).

        Supports both 'astc-4x4' and 'astc_4x4' style names.
        """
        # Normalize: replace hyphens with underscores
        name_normalized = name.replace("-", "_")
        # Try exact match first (for case-sensitive names like ASTC_4x4)
        for member in cls:
            if member.name.upper() == name_normalized.upper():
                return member
        raise ValueError(f"Unknown texture format: {name}")

    @property
    def bits_per_pixel(self) -> float:
        """Return the bits per pixel for this format."""
        bpp_map: dict[TextureFormat, float] = {
            TextureFormat.RGBA8: 32.0,
            TextureFormat.RGBA16F: 64.0,
            TextureFormat.RGBA32F: 128.0,
            TextureFormat.BC1: 4.0,
            TextureFormat.BC3: 8.0,
            TextureFormat.BC4: 4.0,
            TextureFormat.BC5: 8.0,
            TextureFormat.BC6H: 8.0,
            TextureFormat.BC7: 8.0,
            TextureFormat.ASTC_4x4: 8.0,
            TextureFormat.ASTC_5x5: 5.12,
            TextureFormat.ASTC_6x6: 3.56,
            TextureFormat.ASTC_8x8: 2.0,
            TextureFormat.ETC1: 4.0,
            TextureFormat.ETC2_RGB: 4.0,
            TextureFormat.ETC2_RGBA: 8.0,
        }
        return bpp_map.get(self, 32.0)

    @property
    def has_alpha(self) -> bool:
        """Return True if this format supports alpha channel."""
        no_alpha = {
            TextureFormat.BC1,
            TextureFormat.BC4,
            TextureFormat.ETC1,
            TextureFormat.ETC2_RGB,
        }
        return self not in no_alpha

    @property
    def is_compressed(self) -> bool:
        """Return True if this is a compressed format."""
        uncompressed = {
            TextureFormat.RGBA8,
            TextureFormat.RGBA16F,
            TextureFormat.RGBA32F,
        }
        return self not in uncompressed

    @property
    def is_hdr(self) -> bool:
        """Return True if this format supports HDR."""
        return self in {
            TextureFormat.RGBA16F,
            TextureFormat.RGBA32F,
            TextureFormat.BC6H,
        }

    def __str__(self) -> str:
        return self.name.lower()


# Default format mappings per platform
DEFAULT_PLATFORM_FORMATS: dict[Platform, TextureFormat] = {
    Platform.WINDOWS: TextureFormat.BC7,
    Platform.LINUX: TextureFormat.BC7,
    Platform.MACOS: TextureFormat.ASTC_4x4,
    Platform.IOS: TextureFormat.ASTC_4x4,
    Platform.ANDROID: TextureFormat.ETC2_RGBA,
    Platform.WEBGL: TextureFormat.ETC2_RGBA,
}

# Format support matrix per platform
PLATFORM_SUPPORTED_FORMATS: dict[Platform, frozenset[TextureFormat]] = {
    Platform.WINDOWS: frozenset({
        TextureFormat.RGBA8, TextureFormat.RGBA16F, TextureFormat.RGBA32F,
        TextureFormat.BC1, TextureFormat.BC3, TextureFormat.BC4,
        TextureFormat.BC5, TextureFormat.BC6H, TextureFormat.BC7,
    }),
    Platform.LINUX: frozenset({
        TextureFormat.RGBA8, TextureFormat.RGBA16F, TextureFormat.RGBA32F,
        TextureFormat.BC1, TextureFormat.BC3, TextureFormat.BC4,
        TextureFormat.BC5, TextureFormat.BC6H, TextureFormat.BC7,
        TextureFormat.ETC2_RGB, TextureFormat.ETC2_RGBA,  # Some Linux GPUs support ETC2
    }),
    Platform.MACOS: frozenset({
        TextureFormat.RGBA8, TextureFormat.RGBA16F, TextureFormat.RGBA32F,
        TextureFormat.ASTC_4x4, TextureFormat.ASTC_5x5,
        TextureFormat.ASTC_6x6, TextureFormat.ASTC_8x8,
        TextureFormat.BC1, TextureFormat.BC3, TextureFormat.BC7,  # Metal supports BC
    }),
    Platform.IOS: frozenset({
        TextureFormat.RGBA8, TextureFormat.RGBA16F,
        TextureFormat.ASTC_4x4, TextureFormat.ASTC_5x5,
        TextureFormat.ASTC_6x6, TextureFormat.ASTC_8x8,
    }),
    Platform.ANDROID: frozenset({
        TextureFormat.RGBA8, TextureFormat.RGBA16F,
        TextureFormat.ETC1, TextureFormat.ETC2_RGB, TextureFormat.ETC2_RGBA,
        TextureFormat.ASTC_4x4, TextureFormat.ASTC_5x5,  # Modern Android
        TextureFormat.ASTC_6x6, TextureFormat.ASTC_8x8,
    }),
    Platform.WEBGL: frozenset({
        TextureFormat.RGBA8,
        TextureFormat.ETC1, TextureFormat.ETC2_RGB, TextureFormat.ETC2_RGBA,
        TextureFormat.ASTC_4x4, TextureFormat.ASTC_5x5,  # WebGL 2.0+ with extensions
    }),
}


def get_optimal_format(
    platform: Platform,
    has_alpha: bool = True,
    is_hdr: bool = False,
    quality_level: int = 2,
) -> TextureFormat:
    """Get the optimal texture format for a platform.

    Args:
        platform: Target platform
        has_alpha: Whether the texture needs alpha channel
        is_hdr: Whether the texture needs HDR
        quality_level: 0=lowest, 1=medium, 2=high, 3=highest

    Returns:
        Optimal TextureFormat for the given constraints
    """
    if is_hdr:
        if platform.is_desktop:
            return TextureFormat.BC6H
        return TextureFormat.RGBA16F

    if platform == Platform.WINDOWS or platform == Platform.LINUX:
        if has_alpha:
            return TextureFormat.BC7 if quality_level >= 2 else TextureFormat.BC3
        return TextureFormat.BC7 if quality_level >= 2 else TextureFormat.BC1

    if platform.is_apple:
        astc_formats = [
            TextureFormat.ASTC_8x8,
            TextureFormat.ASTC_6x6,
            TextureFormat.ASTC_5x5,
            TextureFormat.ASTC_4x4,
        ]
        return astc_formats[min(quality_level, 3)]

    if platform == Platform.ANDROID:
        if has_alpha:
            return TextureFormat.ETC2_RGBA
        return TextureFormat.ETC2_RGB

    if platform == Platform.WEBGL:
        if has_alpha:
            return TextureFormat.ETC2_RGBA
        return TextureFormat.ETC2_RGB

    # Fallback
    return TextureFormat.RGBA8


def is_format_supported(platform: Platform, fmt: TextureFormat) -> bool:
    """Check if a format is supported on a platform."""
    supported = PLATFORM_SUPPORTED_FORMATS.get(platform, frozenset())
    return fmt in supported


@dataclass(frozen=True, slots=True)
class VariantKey:
    """Unique identifier for a platform variant.

    Combines source content hash with platform and format
    to create a unique key for variant storage.
    """
    source_hash: ContentHash
    platform: Platform
    format: TextureFormat

    def __post_init__(self) -> None:
        if self.source_hash.is_null():
            raise ValueError("Source hash cannot be null")

    @property
    def cache_key(self) -> str:
        """Return a string key suitable for caching."""
        return f"{self.source_hash.hex}:{self.platform.name}:{self.format.name}"

    def compute_variant_hash(self) -> ContentHash:
        """Compute a derived hash for this variant key.

        The variant hash is derived from the source hash plus
        platform and format identifiers, ensuring unique storage.
        """
        key_data = (
            self.source_hash.digest +
            self.platform.name.encode("utf-8") +
            self.format.name.encode("utf-8")
        )
        return ContentHash.from_content(key_data)

    def __str__(self) -> str:
        return f"{self.source_hash.short_hex}@{self.platform.name}/{self.format.name}"


@dataclass
class VariantRule:
    """Rule for selecting texture format based on conditions.

    Supports platform matching, alpha channel requirements,
    HDR requirements, and custom predicates.
    """
    platform: Platform | None = None
    target_format: TextureFormat = TextureFormat.RGBA8
    requires_alpha: bool | None = None
    requires_hdr: bool | None = None
    quality_level: int = 2
    priority: int = 0
    predicate: Callable[[dict[str, Any]], bool] | None = None

    def matches(
        self,
        platform: Platform,
        has_alpha: bool = True,
        is_hdr: bool = False,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if this rule matches the given conditions."""
        # Platform match (None means any platform)
        if self.platform is not None and self.platform != platform:
            return False

        # Alpha requirement
        if self.requires_alpha is not None:
            if self.requires_alpha and not has_alpha:
                return False
            if not self.requires_alpha and has_alpha:
                return False

        # HDR requirement
        if self.requires_hdr is not None:
            if self.requires_hdr and not is_hdr:
                return False
            if not self.requires_hdr and is_hdr:
                return False

        # Custom predicate
        if self.predicate is not None:
            ctx = context or {}
            if not self.predicate(ctx):
                return False

        return True


class VariantRuleSet:
    """Collection of variant rules with priority-based matching."""
    __slots__ = ("_rules", "_lock")

    def __init__(self, rules: list[VariantRule] | None = None) -> None:
        self._rules: list[VariantRule] = list(rules) if rules else []
        self._lock = threading.RLock()
        self._sort_rules()

    def _sort_rules(self) -> None:
        """Sort rules by priority (highest first)."""
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def add_rule(self, rule: VariantRule) -> None:
        """Add a rule and re-sort."""
        with self._lock:
            self._rules.append(rule)
            self._sort_rules()

    def remove_rule(self, rule: VariantRule) -> bool:
        """Remove a rule. Returns True if found and removed."""
        with self._lock:
            try:
                self._rules.remove(rule)
                return True
            except ValueError:
                return False

    def clear(self) -> None:
        """Remove all rules."""
        with self._lock:
            self._rules.clear()

    def match(
        self,
        platform: Platform,
        has_alpha: bool = True,
        is_hdr: bool = False,
        context: dict[str, Any] | None = None,
    ) -> TextureFormat:
        """Find the best matching format for given conditions.

        Returns the format from the highest priority matching rule,
        or the platform default if no rules match.
        """
        with self._lock:
            for rule in self._rules:
                if rule.matches(platform, has_alpha, is_hdr, context):
                    return rule.target_format

        # Fall back to platform default
        return DEFAULT_PLATFORM_FORMATS.get(platform, TextureFormat.RGBA8)

    def __len__(self) -> int:
        with self._lock:
            return len(self._rules)

    def __iter__(self) -> Iterator[VariantRule]:
        with self._lock:
            return iter(list(self._rules))

    @classmethod
    def default_rules(cls) -> VariantRuleSet:
        """Create a rule set with sensible defaults for all platforms."""
        rules = [
            # Desktop BC7 for quality
            VariantRule(
                platform=Platform.WINDOWS,
                target_format=TextureFormat.BC7,
                priority=10,
            ),
            VariantRule(
                platform=Platform.LINUX,
                target_format=TextureFormat.BC7,
                priority=10,
            ),
            # Apple ASTC
            VariantRule(
                platform=Platform.MACOS,
                target_format=TextureFormat.ASTC_4x4,
                priority=10,
            ),
            VariantRule(
                platform=Platform.IOS,
                target_format=TextureFormat.ASTC_4x4,
                priority=10,
            ),
            # Android ETC2
            VariantRule(
                platform=Platform.ANDROID,
                target_format=TextureFormat.ETC2_RGBA,
                requires_alpha=True,
                priority=10,
            ),
            VariantRule(
                platform=Platform.ANDROID,
                target_format=TextureFormat.ETC2_RGB,
                requires_alpha=False,
                priority=10,
            ),
            # WebGL ETC2
            VariantRule(
                platform=Platform.WEBGL,
                target_format=TextureFormat.ETC2_RGBA,
                priority=10,
            ),
            # HDR overrides
            VariantRule(
                platform=Platform.WINDOWS,
                target_format=TextureFormat.BC6H,
                requires_hdr=True,
                priority=20,
            ),
            VariantRule(
                platform=Platform.LINUX,
                target_format=TextureFormat.BC6H,
                requires_hdr=True,
                priority=20,
            ),
            # Fallback HDR for other platforms
            VariantRule(
                target_format=TextureFormat.RGBA16F,
                requires_hdr=True,
                priority=5,
            ),
        ]
        return cls(rules)


@dataclass
class SourceAsset:
    """Represents a source asset before variant generation."""
    path: Path
    content_hash: ContentHash
    has_alpha: bool = True
    is_hdr: bool = False
    width: int = 0
    height: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        hasher: AssetHasher | None = None,
    ) -> SourceAsset:
        """Create from file path, computing content hash."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Source asset not found: {path}")

        hasher = hasher or AssetHasher()
        content_hash = hasher.hash_file(path)

        # Try to detect properties from file content
        has_alpha = True  # Default assumption
        is_hdr = False
        width = 0
        height = 0

        # Check file extension for HDR hint
        ext = path.suffix.lower()
        if ext in {".hdr", ".exr"}:
            is_hdr = True

        return cls(
            path=path,
            content_hash=content_hash,
            has_alpha=has_alpha,
            is_hdr=is_hdr,
            width=width,
            height=height,
        )


@dataclass
class GeneratedVariant:
    """Represents a generated platform variant."""
    key: VariantKey
    data: bytes
    variant_hash: ContentHash
    generation_time_ms: float = 0.0
    compressed_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.compressed_size == 0:
            self.compressed_size = len(self.data)


class FormatConverter(ABC):
    """Abstract base class for format conversion."""

    @abstractmethod
    def convert(
        self,
        source_data: bytes,
        source_format: TextureFormat,
        target_format: TextureFormat,
        width: int,
        height: int,
    ) -> bytes:
        """Convert data from source to target format."""
        ...

    @abstractmethod
    def can_convert(
        self,
        source_format: TextureFormat,
        target_format: TextureFormat,
    ) -> bool:
        """Check if this converter supports the conversion."""
        ...


class TextureFormatConverter(FormatConverter):
    """Simulated texture format converter.

    In a real implementation, this would use GPU compression
    libraries or external tools. This version provides a
    simulation for testing and development.
    """
    __slots__ = ("_simulation_overhead_ms",)

    def __init__(self, simulation_overhead_ms: float = 0.0) -> None:
        self._simulation_overhead_ms = simulation_overhead_ms

    def convert(
        self,
        source_data: bytes,
        source_format: TextureFormat,
        target_format: TextureFormat,
        width: int,
        height: int,
    ) -> bytes:
        """Convert texture data to target format.

        This is a simulation that produces format-specific headers
        followed by mock compressed data.
        """
        if not self.can_convert(source_format, target_format):
            raise ValueError(
                f"Cannot convert {source_format} to {target_format}"
            )

        if source_format == target_format:
            return source_data

        # Build format-specific header
        header = self._build_header(target_format, width, height)

        # Simulate compression ratio
        source_size = len(source_data)
        target_bpp = target_format.bits_per_pixel
        source_bpp = source_format.bits_per_pixel

        if target_bpp < source_bpp:
            # Compression - reduce size
            ratio = target_bpp / source_bpp
            compressed_size = max(int(source_size * ratio), len(header) + 16)
        else:
            # Decompression - size may increase or stay same
            compressed_size = source_size

        # Generate deterministic "compressed" data based on source
        body_size = compressed_size - len(header)
        body = self._generate_body(source_data, body_size, target_format)

        return header + body

    def can_convert(
        self,
        source_format: TextureFormat,
        target_format: TextureFormat,
    ) -> bool:
        """Check if conversion is supported.

        Most conversions are supported as we can always go through
        uncompressed RGBA8 as an intermediate format.
        """
        # Same format always supported
        if source_format == target_format:
            return True

        # HDR to non-HDR requires special handling
        if source_format.is_hdr and not target_format.is_hdr:
            # Tonemapping would be needed - still possible
            return True

        # All other conversions supported
        return True

    def _build_header(
        self,
        target_format: TextureFormat,
        width: int,
        height: int,
    ) -> bytes:
        """Build format-specific header."""
        # Simple header: magic + format_id + width + height
        magic = b"TXFM"  # Texture Format
        format_id = target_format.value
        return struct.pack("<4sIII", magic, format_id, width, height)

    def _generate_body(
        self,
        source_data: bytes,
        body_size: int,
        target_format: TextureFormat,
    ) -> bytes:
        """Generate deterministic body data based on source."""
        # Use source hash to generate deterministic output
        hasher = HashAlgorithm()
        hasher.update(source_data)
        hasher.update(target_format.name.encode())
        seed = hasher.digest()

        # Repeat seed to fill body
        repeats = (body_size // len(seed)) + 1
        full_data = seed * repeats
        return full_data[:body_size]


class VariantCache:
    """In-memory cache for generated variants with LRU eviction."""
    __slots__ = ("_cache", "_max_size", "_lock", "_access_order")

    def __init__(self, max_size: int = 100) -> None:
        if max_size <= 0:
            raise ValueError("Cache size must be positive")
        self._cache: dict[str, GeneratedVariant] = {}
        self._max_size = max_size
        self._lock = threading.RLock()
        self._access_order: list[str] = []

    def get(self, key: VariantKey) -> GeneratedVariant | None:
        """Get cached variant, updating access order."""
        cache_key = key.cache_key
        with self._lock:
            variant = self._cache.get(cache_key)
            if variant is not None:
                # Update access order for LRU
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
            return variant

    def put(self, variant: GeneratedVariant) -> None:
        """Cache a variant, evicting LRU if at capacity."""
        cache_key = variant.key.cache_key
        with self._lock:
            if cache_key in self._cache:
                # Update existing
                self._cache[cache_key] = variant
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
            else:
                # Evict if needed
                while len(self._cache) >= self._max_size:
                    oldest = self._access_order.pop(0)
                    del self._cache[oldest]
                # Add new
                self._cache[cache_key] = variant
                self._access_order.append(cache_key)

    def invalidate(self, key: VariantKey) -> bool:
        """Remove variant from cache. Returns True if found."""
        cache_key = key.cache_key
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._access_order.remove(cache_key)
                return True
            return False

    def invalidate_source(self, source_hash: ContentHash) -> int:
        """Remove all variants for a source hash. Returns count removed."""
        prefix = source_hash.hex + ":"
        with self._lock:
            to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in to_remove:
                del self._cache[k]
                self._access_order.remove(k)
            return len(to_remove)

    def clear(self) -> int:
        """Clear all cached variants. Returns count cleared."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            return count

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: VariantKey) -> bool:
        return self.get(key) is not None

    @property
    def max_size(self) -> int:
        return self._max_size


class VariantStorage:
    """Content-addressed storage for platform variants.

    Stores generated variants using content-addressed storage,
    enabling deduplication when multiple platforms produce
    identical output (e.g., when using fallback formats).
    """
    __slots__ = ("_storage", "_key_to_hash", "_lock")

    def __init__(self) -> None:
        self._storage: ContentAddressedStorage[bytes] = ContentAddressedStorage()
        self._key_to_hash: dict[str, ContentHash] = {}
        self._lock = threading.RLock()

    def store(self, variant: GeneratedVariant) -> ContentHash:
        """Store a variant, returning its content hash."""
        cache_key = variant.key.cache_key

        with self._lock:
            # Store the data using content-addressed storage
            stored_hash = self._storage.store_bytes(
                variant.data,
                virtual_path=f"variant://{cache_key}",
            )
            self._key_to_hash[cache_key] = stored_hash
            return stored_hash

    def get(self, key: VariantKey) -> bytes | None:
        """Get stored variant data by key."""
        cache_key = key.cache_key
        with self._lock:
            stored_hash = self._key_to_hash.get(cache_key)
            if stored_hash is None:
                return None
            return self._storage.get_by_hash(stored_hash)

    def get_by_hash(self, content_hash: ContentHash) -> bytes | None:
        """Get variant data directly by hash."""
        with self._lock:
            return self._storage.get_by_hash(content_hash)

    def contains(self, key: VariantKey) -> bool:
        """Check if variant is stored."""
        cache_key = key.cache_key
        with self._lock:
            return cache_key in self._key_to_hash

    def contains_hash(self, content_hash: ContentHash) -> bool:
        """Check if hash is stored."""
        with self._lock:
            return self._storage.contains_hash(content_hash)

    def get_hash(self, key: VariantKey) -> ContentHash | None:
        """Get the content hash for a variant key."""
        cache_key = key.cache_key
        with self._lock:
            return self._key_to_hash.get(cache_key)

    def remove(self, key: VariantKey) -> bool:
        """Remove a variant. Returns True if found."""
        cache_key = key.cache_key
        with self._lock:
            stored_hash = self._key_to_hash.pop(cache_key, None)
            if stored_hash is None:
                return False
            self._storage.release(stored_hash)
            return True

    def remove_for_source(self, source_hash: ContentHash) -> int:
        """Remove all variants for a source. Returns count removed."""
        prefix = source_hash.hex + ":"
        with self._lock:
            to_remove = [k for k in self._key_to_hash if k.startswith(prefix)]
            for cache_key in to_remove:
                stored_hash = self._key_to_hash.pop(cache_key)
                self._storage.release(stored_hash)
            return len(to_remove)

    def get_stats(self) -> dict[str, Any]:
        """Return storage statistics."""
        with self._lock:
            storage_stats = self._storage.get_stats()
            return {
                "variant_count": len(self._key_to_hash),
                **storage_stats,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._key_to_hash)

    def __contains__(self, item: VariantKey | ContentHash) -> bool:
        if isinstance(item, VariantKey):
            return self.contains(item)
        return self.contains_hash(item)


class VariantGenerator:
    """Generates platform-specific asset variants on demand.

    Coordinates rule-based format selection, conversion,
    caching, and content-addressed storage.
    """
    __slots__ = (
        "_rules",
        "_converter",
        "_cache",
        "_storage",
        "_hasher",
        "_lock",
        "_lazy_generation",
    )

    def __init__(
        self,
        rules: VariantRuleSet | None = None,
        converter: FormatConverter | None = None,
        cache: VariantCache | None = None,
        storage: VariantStorage | None = None,
        lazy_generation: bool = True,
    ) -> None:
        self._rules = rules or VariantRuleSet.default_rules()
        self._converter = converter or TextureFormatConverter()
        self._cache = cache or VariantCache()
        self._storage = storage or VariantStorage()
        self._hasher = AssetHasher()
        self._lock = threading.RLock()
        self._lazy_generation = lazy_generation

    @property
    def rules(self) -> VariantRuleSet:
        """Access the rule set."""
        return self._rules

    @property
    def cache(self) -> VariantCache:
        """Access the variant cache."""
        return self._cache

    @property
    def storage(self) -> VariantStorage:
        """Access the variant storage."""
        return self._storage

    @property
    def lazy_generation(self) -> bool:
        """Return True if lazy generation is enabled."""
        return self._lazy_generation

    @lazy_generation.setter
    def lazy_generation(self, value: bool) -> None:
        self._lazy_generation = value

    def select_format(
        self,
        platform: Platform,
        source: SourceAsset,
        context: dict[str, Any] | None = None,
    ) -> TextureFormat:
        """Select the optimal format for a platform and source."""
        return self._rules.match(
            platform=platform,
            has_alpha=source.has_alpha,
            is_hdr=source.is_hdr,
            context=context,
        )

    def generate(
        self,
        source: SourceAsset,
        platform: Platform,
        target_format: TextureFormat | None = None,
        force_regenerate: bool = False,
    ) -> GeneratedVariant:
        """Generate a platform variant for the source asset.

        Args:
            source: The source asset to convert
            platform: Target platform
            target_format: Explicit format, or None to auto-select
            force_regenerate: Bypass cache and storage

        Returns:
            Generated variant with converted data
        """
        import time

        # Determine target format
        if target_format is None:
            target_format = self.select_format(platform, source)

        # Validate format support
        if not is_format_supported(platform, target_format):
            logger.warning(
                "Format %s not officially supported on %s, proceeding anyway",
                target_format,
                platform,
            )

        # Create variant key
        key = VariantKey(
            source_hash=source.content_hash,
            platform=platform,
            format=target_format,
        )

        # Check cache first (unless forced)
        if not force_regenerate:
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("Cache hit for variant %s", key)
                return cached

            # Check storage
            stored_data = self._storage.get(key)
            if stored_data is not None:
                logger.debug("Storage hit for variant %s", key)
                variant_hash = self._storage.get_hash(key)
                variant = GeneratedVariant(
                    key=key,
                    data=stored_data,
                    variant_hash=variant_hash or ContentHash.from_content(stored_data),
                )
                self._cache.put(variant)
                return variant

        # Generate the variant
        start_time = time.perf_counter()

        # Read source data
        with open(source.path, "rb") as f:
            source_data = f.read()

        # Convert
        converted_data = self._converter.convert(
            source_data=source_data,
            source_format=TextureFormat.RGBA8,  # Assume source is uncompressed
            target_format=target_format,
            width=source.width or 256,
            height=source.height or 256,
        )

        generation_time_ms = (time.perf_counter() - start_time) * 1000

        # Create variant
        variant_hash = ContentHash.from_content(converted_data)
        variant = GeneratedVariant(
            key=key,
            data=converted_data,
            variant_hash=variant_hash,
            generation_time_ms=generation_time_ms,
            compressed_size=len(converted_data),
            metadata={
                "source_path": str(source.path),
                "source_size": len(source_data),
            },
        )

        # Store
        self._storage.store(variant)
        self._cache.put(variant)

        logger.debug(
            "Generated variant %s in %.2fms (%d -> %d bytes)",
            key,
            generation_time_ms,
            len(source_data),
            len(converted_data),
        )

        return variant

    def generate_all_platforms(
        self,
        source: SourceAsset,
        platforms: list[Platform] | None = None,
    ) -> dict[Platform, GeneratedVariant]:
        """Generate variants for multiple platforms.

        Args:
            source: Source asset
            platforms: Target platforms, or None for all

        Returns:
            Mapping of platform to generated variant
        """
        if platforms is None:
            platforms = list(Platform)

        results: dict[Platform, GeneratedVariant] = {}
        for platform in platforms:
            try:
                results[platform] = self.generate(source, platform)
            except Exception as e:
                logger.error(
                    "Failed to generate variant for %s on %s: %s",
                    source.path,
                    platform,
                    e,
                )
        return results

    def get_or_generate(
        self,
        source: SourceAsset,
        platform: Platform,
        target_format: TextureFormat | None = None,
    ) -> GeneratedVariant:
        """Get existing variant or generate on demand (lazy).

        This is the primary method for lazy generation mode.
        """
        if target_format is None:
            target_format = self.select_format(platform, source)

        key = VariantKey(
            source_hash=source.content_hash,
            platform=platform,
            format=target_format,
        )

        # Try cache
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Try storage
        stored_data = self._storage.get(key)
        if stored_data is not None:
            variant_hash = self._storage.get_hash(key)
            variant = GeneratedVariant(
                key=key,
                data=stored_data,
                variant_hash=variant_hash or ContentHash.from_content(stored_data),
            )
            self._cache.put(variant)
            return variant

        # Generate on demand if lazy mode enabled
        if self._lazy_generation:
            return self.generate(source, platform, target_format)

        raise KeyError(f"Variant not found and lazy generation disabled: {key}")

    def invalidate(self, source_hash: ContentHash) -> int:
        """Invalidate all variants for a source. Returns count removed."""
        cache_removed = self._cache.invalidate_source(source_hash)
        storage_removed = self._storage.remove_for_source(source_hash)
        return max(cache_removed, storage_removed)

    def get_stats(self) -> dict[str, Any]:
        """Return generator statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_max_size": self._cache.max_size,
            "rule_count": len(self._rules),
            "lazy_generation": self._lazy_generation,
            **self._storage.get_stats(),
        }
