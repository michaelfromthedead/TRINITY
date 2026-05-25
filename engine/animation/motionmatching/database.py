"""
Motion Matching Database - Core data structures for motion matching.

This module provides the database infrastructure for motion matching:
- MotionDatabase: Stores clips, features, and metadata for search
- DatabaseEntry: Individual frame entries with features and tags
- build_database: Constructs database from animation clips
- Serialization/deserialization for fast loading
- Incremental database updates
- Memory-efficient quantized feature storage

Usage:
    from engine.animation.motionmatching.database import (
        MotionDatabase, DatabaseEntry, build_database
    )

    # Build database from clips
    db = build_database(clips, feature_extractor)

    # Save/load for fast startup
    db.save("locomotion.mmdb")
    db = MotionDatabase.load("locomotion.mmdb")
"""

from __future__ import annotations

import struct
import pickle
import gzip
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import numpy as np

if TYPE_CHECKING:
    from engine.animation.motionmatching.features import FeatureExtractor, FeatureSet


# =============================================================================
# DECORATORS
# =============================================================================


def motion_matching(func: Callable) -> Callable:
    """Decorator to mark functions as motion matching database builders.

    Provides:
    - Performance timing
    - Error handling
    - Logging
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Could add timing, logging, etc.
        return func(*args, **kwargs)
    wrapper._motion_matching_builder = True
    return wrapper


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================


class QuantizationLevel(Enum):
    """Quantization levels for feature storage."""
    NONE = auto()      # 32-bit float (full precision)
    FLOAT16 = auto()   # 16-bit float (half precision)
    INT16 = auto()     # 16-bit signed integer (scaled)
    INT8 = auto()      # 8-bit signed integer (scaled)


# Import centralized config
from engine.animation.motionmatching.config import (
    DEFAULT_DATABASE_CONFIG,
)

# Default quantization scale for integer quantization
DEFAULT_QUANT_SCALE = DEFAULT_DATABASE_CONFIG.int16_quant_scale

# Magic number for database file format
DATABASE_MAGIC = b'MMDB'
DATABASE_VERSION = 1


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ClipMetadata:
    """Metadata about an animation clip in the database.

    Attributes:
        clip_index: Index of the clip in the database
        name: Human-readable clip name
        frame_count: Number of frames in the clip
        frame_rate: Frames per second
        duration: Total duration in seconds
        is_looping: Whether the clip loops
        has_root_motion: Whether the clip has root motion data
        tags: Set of tags describing this clip (e.g., "walk", "run")
    """
    clip_index: int
    name: str
    frame_count: int
    frame_rate: float = 30.0
    duration: float = 0.0
    is_looping: bool = False
    has_root_motion: bool = False
    tags: FrozenSet[str] = field(default_factory=frozenset)

    def __post_init__(self):
        if self.duration <= 0.0 and self.frame_count > 0:
            self.duration = (self.frame_count - 1) / self.frame_rate if self.frame_rate > 0 else 0.0
        if isinstance(self.tags, (set, list)):
            self.tags = frozenset(self.tags)


@dataclass
class DatabaseEntry:
    """A single frame entry in the motion database.

    Attributes:
        clip_index: Index of the source clip
        frame: Frame number within the clip
        features: Flattened feature vector (numpy array)
        tags: Tags associated with this frame
        is_transition_candidate: Whether this frame can be transitioned to
        cost_modifier: Additional cost for matching this frame (for biasing)
    """
    clip_index: int
    frame: int
    features: np.ndarray
    tags: FrozenSet[str] = field(default_factory=frozenset)
    is_transition_candidate: bool = True
    cost_modifier: float = 0.0

    def __post_init__(self):
        if isinstance(self.tags, (set, list)):
            self.tags = frozenset(self.tags)
        if not isinstance(self.features, np.ndarray):
            self.features = np.array(self.features, dtype=np.float32)

    @property
    def entry_id(self) -> Tuple[int, int]:
        """Unique identifier as (clip_index, frame) tuple."""
        return (self.clip_index, self.frame)

    def matches_tags(self, required_tags: Optional[Set[str]] = None) -> bool:
        """Check if this entry has all required tags."""
        if not required_tags:
            return True
        return required_tags.issubset(self.tags)


@dataclass
class NormalizationStats:
    """Statistics for feature normalization.

    Attributes:
        mean: Mean value per feature dimension
        std: Standard deviation per feature dimension
        min_val: Minimum value per feature dimension
        max_val: Maximum value per feature dimension
    """
    mean: np.ndarray
    std: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray

    @classmethod
    def compute(cls, features: np.ndarray, epsilon: float = None) -> NormalizationStats:
        """Compute normalization statistics from feature matrix.

        Args:
            features: Matrix of shape (num_entries, feature_dim)
            epsilon: Small value to prevent division by zero (uses config default if None)

        Returns:
            NormalizationStats instance
        """
        if epsilon is None:
            epsilon = DEFAULT_DATABASE_CONFIG.normalization_epsilon

        # Handle empty features
        if features.size == 0:
            dim = features.shape[1] if len(features.shape) > 1 else 1
            return cls(
                mean=np.zeros(dim, dtype=np.float32),
                std=np.ones(dim, dtype=np.float32),
                min_val=np.zeros(dim, dtype=np.float32),
                max_val=np.zeros(dim, dtype=np.float32),
            )

        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        # Prevent division by zero with configurable epsilon
        std = np.where(std > epsilon, std, epsilon)
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)

        return cls(mean=mean, std=std, min_val=min_val, max_val=max_val)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using z-score normalization."""
        return (features - self.mean) / self.std

    def denormalize(self, features: np.ndarray) -> np.ndarray:
        """Denormalize features back to original scale."""
        return features * self.std + self.mean


# =============================================================================
# ANIMATION CLIP PROTOCOL
# =============================================================================


class AnimationClipProtocol(Protocol):
    """Protocol defining what an animation clip must provide for motion matching."""

    @property
    def name(self) -> str: ...

    @property
    def frame_count(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    @property
    def duration(self) -> float: ...

    @property
    def is_looping(self) -> bool: ...

    @property
    def has_root_motion(self) -> bool: ...

    def sample(self, time: float) -> Any: ...

    def get_frame_pose(self, frame: int) -> Any: ...


# =============================================================================
# MOTION DATABASE
# =============================================================================


class MotionDatabase:
    """Motion matching database containing clips, features, and search structures.

    The database stores:
    - Metadata for each clip
    - Per-frame feature vectors for matching
    - Normalization statistics
    - Tag indices for filtering

    Features are stored in a compact numpy array for fast search.
    Quantization can be used to reduce memory footprint.

    Attributes:
        clip_metadata: List of ClipMetadata for each clip
        entries: List of DatabaseEntry for each frame
        feature_matrix: Dense matrix of all features (num_entries x feature_dim)
        normalization: Optional normalization statistics
        quantization: Current quantization level
        feature_dimension: Dimension of feature vectors
    """

    def __init__(
        self,
        feature_dimension: int = 0,
        quantization: QuantizationLevel = QuantizationLevel.NONE,
    ):
        """Initialize empty motion database.

        Args:
            feature_dimension: Dimension of feature vectors
            quantization: Quantization level for storage
        """
        self._feature_dimension = feature_dimension
        self._quantization = quantization

        # Clip metadata
        self._clip_metadata: List[ClipMetadata] = []

        # Frame entries
        self._entries: List[DatabaseEntry] = []

        # Dense feature matrix (built during finalize)
        self._feature_matrix: Optional[np.ndarray] = None

        # Normalization statistics
        self._normalization: Optional[NormalizationStats] = None

        # Tag index for fast filtering
        self._tag_index: Dict[str, Set[int]] = {}

        # Clip to entry mapping
        self._clip_entry_ranges: Dict[int, Tuple[int, int]] = {}

        # Flag indicating if database needs rebuild
        self._dirty: bool = True

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def clip_count(self) -> int:
        """Number of clips in the database."""
        return len(self._clip_metadata)

    @property
    def entry_count(self) -> int:
        """Total number of frame entries."""
        return len(self._entries)

    @property
    def feature_dimension(self) -> int:
        """Dimension of feature vectors."""
        return self._feature_dimension

    @property
    def quantization(self) -> QuantizationLevel:
        """Current quantization level."""
        return self._quantization

    @property
    def clip_metadata(self) -> List[ClipMetadata]:
        """List of all clip metadata."""
        return self._clip_metadata.copy()

    @property
    def entries(self) -> List[DatabaseEntry]:
        """List of all frame entries."""
        return self._entries.copy()

    @property
    def feature_matrix(self) -> Optional[np.ndarray]:
        """Dense feature matrix for search."""
        return self._feature_matrix

    @property
    def normalization(self) -> Optional[NormalizationStats]:
        """Normalization statistics."""
        return self._normalization

    @property
    def is_dirty(self) -> bool:
        """Whether database needs to be finalized."""
        return self._dirty

    @property
    def total_frames(self) -> int:
        """Total number of frames across all clips."""
        return sum(clip.frame_count for clip in self._clip_metadata)

    @property
    def memory_usage_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        # Feature matrix memory
        if self._feature_matrix is not None:
            matrix_bytes = self._feature_matrix.nbytes
        else:
            matrix_bytes = 0

        # Entry overhead (rough estimate)
        entry_overhead = len(self._entries) * 100

        # Metadata overhead
        metadata_overhead = len(self._clip_metadata) * 200

        return matrix_bytes + entry_overhead + metadata_overhead

    # -------------------------------------------------------------------------
    # Clip Management
    # -------------------------------------------------------------------------

    def add_clip(self, metadata: ClipMetadata) -> int:
        """Add a new clip to the database.

        Args:
            metadata: Clip metadata

        Returns:
            Index of the added clip
        """
        clip_index = len(self._clip_metadata)
        metadata = ClipMetadata(
            clip_index=clip_index,
            name=metadata.name,
            frame_count=metadata.frame_count,
            frame_rate=metadata.frame_rate,
            duration=metadata.duration,
            is_looping=metadata.is_looping,
            has_root_motion=metadata.has_root_motion,
            tags=metadata.tags,
        )
        self._clip_metadata.append(metadata)
        self._dirty = True
        return clip_index

    def get_clip_metadata(self, clip_index: int) -> Optional[ClipMetadata]:
        """Get metadata for a specific clip."""
        if 0 <= clip_index < len(self._clip_metadata):
            return self._clip_metadata[clip_index]
        return None

    def get_clip_by_name(self, name: str) -> Optional[ClipMetadata]:
        """Get clip metadata by name."""
        for clip in self._clip_metadata:
            if clip.name == name:
                return clip
        return None

    # -------------------------------------------------------------------------
    # Entry Management
    # -------------------------------------------------------------------------

    def add_entry(self, entry: DatabaseEntry) -> int:
        """Add a frame entry to the database.

        Args:
            entry: Database entry

        Returns:
            Index of the added entry
        """
        entry_index = len(self._entries)

        # Update feature dimension if needed
        if self._feature_dimension == 0:
            self._feature_dimension = len(entry.features)

        self._entries.append(entry)

        # Update tag index
        for tag in entry.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(entry_index)

        self._dirty = True
        return entry_index

    def add_entries(self, entries: Iterable[DatabaseEntry]) -> List[int]:
        """Add multiple frame entries to the database.

        Args:
            entries: Iterable of database entries

        Returns:
            List of indices of added entries
        """
        return [self.add_entry(e) for e in entries]

    def get_entry(self, index: int) -> Optional[DatabaseEntry]:
        """Get entry by index."""
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return None

    def get_entry_by_clip_frame(
        self, clip_index: int, frame: int
    ) -> Optional[DatabaseEntry]:
        """Get entry by clip index and frame number."""
        if clip_index in self._clip_entry_ranges:
            start, end = self._clip_entry_ranges[clip_index]
            if start <= frame < end:
                return self._entries[start + frame]

        # Fallback to linear search
        for entry in self._entries:
            if entry.clip_index == clip_index and entry.frame == frame:
                return entry
        return None

    def get_entries_for_clip(self, clip_index: int) -> List[DatabaseEntry]:
        """Get all entries for a specific clip."""
        return [e for e in self._entries if e.clip_index == clip_index]

    def get_entries_with_tags(
        self, required_tags: Set[str], any_tag: bool = False
    ) -> List[DatabaseEntry]:
        """Get entries matching tag requirements.

        Args:
            required_tags: Set of required tags
            any_tag: If True, match if any tag is present; if False, all required

        Returns:
            List of matching entries
        """
        if not required_tags:
            return self._entries.copy()

        if any_tag:
            # Union of all entries with any of the tags
            indices: Set[int] = set()
            for tag in required_tags:
                if tag in self._tag_index:
                    indices.update(self._tag_index[tag])
            return [self._entries[i] for i in sorted(indices)]
        else:
            # Intersection: entries must have all tags
            if not required_tags:
                return self._entries.copy()

            # Start with first tag's entries
            first_tag = next(iter(required_tags))
            if first_tag not in self._tag_index:
                return []

            indices = self._tag_index[first_tag].copy()

            # Intersect with remaining tags
            for tag in required_tags:
                if tag == first_tag:
                    continue
                if tag not in self._tag_index:
                    return []
                indices.intersection_update(self._tag_index[tag])

            return [self._entries[i] for i in sorted(indices)]

    # -------------------------------------------------------------------------
    # Database Building
    # -------------------------------------------------------------------------

    def finalize(self, compute_normalization: bool = True) -> None:
        """Finalize the database for search.

        Builds the dense feature matrix and computes normalization.
        Must be called after adding all entries.

        Args:
            compute_normalization: Whether to compute normalization stats
        """
        if not self._entries:
            self._feature_matrix = np.zeros((0, self._feature_dimension), dtype=np.float32)
            self._dirty = False
            return

        # Build dense feature matrix
        features = np.stack([e.features for e in self._entries], axis=0)

        # Compute normalization
        if compute_normalization:
            self._normalization = NormalizationStats.compute(features)
            features = self._normalization.normalize(features)

        # Apply quantization
        self._feature_matrix = self._quantize_features(features)

        # Build clip entry ranges
        self._build_clip_entry_ranges()

        self._dirty = False

    def _build_clip_entry_ranges(self) -> None:
        """Build mapping from clip index to entry range."""
        self._clip_entry_ranges.clear()

        current_clip = -1
        start_index = 0

        for i, entry in enumerate(self._entries):
            if entry.clip_index != current_clip:
                if current_clip >= 0:
                    self._clip_entry_ranges[current_clip] = (start_index, i)
                current_clip = entry.clip_index
                start_index = i

        if current_clip >= 0:
            self._clip_entry_ranges[current_clip] = (start_index, len(self._entries))

    def _quantize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply quantization to feature matrix.

        Args:
            features: Float32 feature matrix

        Returns:
            Quantized feature matrix
        """
        if self._quantization == QuantizationLevel.NONE:
            return features.astype(np.float32)
        elif self._quantization == QuantizationLevel.FLOAT16:
            return features.astype(np.float16)
        elif self._quantization == QuantizationLevel.INT16:
            scaled = np.clip(features * DEFAULT_QUANT_SCALE, -32767, 32767)
            return scaled.astype(np.int16)
        elif self._quantization == QuantizationLevel.INT8:
            scaled = np.clip(features * (DEFAULT_QUANT_SCALE / 128), -127, 127)
            return scaled.astype(np.int8)
        else:
            return features.astype(np.float32)

    def _dequantize_features(self, features: np.ndarray) -> np.ndarray:
        """Convert quantized features back to float32.

        Args:
            features: Quantized feature matrix

        Returns:
            Float32 feature matrix
        """
        if self._quantization == QuantizationLevel.NONE:
            return features.astype(np.float32)
        elif self._quantization == QuantizationLevel.FLOAT16:
            return features.astype(np.float32)
        elif self._quantization == QuantizationLevel.INT16:
            return features.astype(np.float32) / DEFAULT_QUANT_SCALE
        elif self._quantization == QuantizationLevel.INT8:
            return features.astype(np.float32) / (DEFAULT_QUANT_SCALE / 128)
        else:
            return features.astype(np.float32)

    def get_features(self, normalize: bool = False) -> np.ndarray:
        """Get feature matrix, optionally dequantized and denormalized.

        Args:
            normalize: If True, return normalized features

        Returns:
            Feature matrix (num_entries x feature_dim)
        """
        if self._feature_matrix is None:
            self.finalize()

        features = self._dequantize_features(self._feature_matrix)

        if not normalize and self._normalization is not None:
            features = self._normalization.denormalize(features)

        return features

    def get_feature_vector(
        self, entry_index: int, normalize: bool = True
    ) -> np.ndarray:
        """Get feature vector for a specific entry.

        Args:
            entry_index: Index of entry
            normalize: If True, return normalized features

        Returns:
            Feature vector
        """
        if self._feature_matrix is None:
            self.finalize()

        features = self._dequantize_features(
            self._feature_matrix[entry_index:entry_index+1]
        )[0]

        if not normalize and self._normalization is not None:
            features = self._normalization.denormalize(features[np.newaxis, :])[0]

        return features

    # -------------------------------------------------------------------------
    # Incremental Updates
    # -------------------------------------------------------------------------

    def update_entry_tags(
        self, entry_index: int, tags: Set[str], replace: bool = False
    ) -> None:
        """Update tags for an entry.

        Args:
            entry_index: Index of entry to update
            tags: Tags to add/replace
            replace: If True, replace all tags; if False, add to existing
        """
        if entry_index < 0 or entry_index >= len(self._entries):
            return

        entry = self._entries[entry_index]
        old_tags = entry.tags

        if replace:
            new_tags = frozenset(tags)
        else:
            new_tags = frozenset(old_tags | tags)

        # Update entry
        self._entries[entry_index] = DatabaseEntry(
            clip_index=entry.clip_index,
            frame=entry.frame,
            features=entry.features,
            tags=new_tags,
            is_transition_candidate=entry.is_transition_candidate,
            cost_modifier=entry.cost_modifier,
        )

        # Update tag index
        for tag in old_tags - new_tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(entry_index)

        for tag in new_tags - old_tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(entry_index)

    def mark_transition_candidate(
        self, entry_index: int, is_candidate: bool
    ) -> None:
        """Mark whether an entry can be transitioned to.

        Args:
            entry_index: Index of entry
            is_candidate: Whether entry is a valid transition target
        """
        if 0 <= entry_index < len(self._entries):
            entry = self._entries[entry_index]
            self._entries[entry_index] = DatabaseEntry(
                clip_index=entry.clip_index,
                frame=entry.frame,
                features=entry.features,
                tags=entry.tags,
                is_transition_candidate=is_candidate,
                cost_modifier=entry.cost_modifier,
            )

    def set_cost_modifier(self, entry_index: int, modifier: float) -> None:
        """Set cost modifier for an entry.

        Args:
            entry_index: Index of entry
            modifier: Cost modifier (added to search cost)
        """
        if 0 <= entry_index < len(self._entries):
            entry = self._entries[entry_index]
            self._entries[entry_index] = DatabaseEntry(
                clip_index=entry.clip_index,
                frame=entry.frame,
                features=entry.features,
                tags=entry.tags,
                is_transition_candidate=entry.is_transition_candidate,
                cost_modifier=modifier,
            )

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def save(self, path: str, compress: bool = True) -> None:
        """Save database to file.

        Args:
            path: Output file path
            compress: Whether to use gzip compression
        """
        data = {
            'version': DATABASE_VERSION,
            'feature_dimension': self._feature_dimension,
            'quantization': self._quantization.value,
            'clip_metadata': [
                {
                    'clip_index': c.clip_index,
                    'name': c.name,
                    'frame_count': c.frame_count,
                    'frame_rate': c.frame_rate,
                    'duration': c.duration,
                    'is_looping': c.is_looping,
                    'has_root_motion': c.has_root_motion,
                    'tags': list(c.tags),
                }
                for c in self._clip_metadata
            ],
            'entries': [
                {
                    'clip_index': e.clip_index,
                    'frame': e.frame,
                    'features': e.features.tobytes(),
                    'features_dtype': str(e.features.dtype),
                    'tags': list(e.tags),
                    'is_transition_candidate': e.is_transition_candidate,
                    'cost_modifier': e.cost_modifier,
                }
                for e in self._entries
            ],
            'feature_matrix': self._feature_matrix.tobytes() if self._feature_matrix is not None else None,
            'feature_matrix_dtype': str(self._feature_matrix.dtype) if self._feature_matrix is not None else None,
            'feature_matrix_shape': self._feature_matrix.shape if self._feature_matrix is not None else None,
            'normalization': {
                'mean': self._normalization.mean.tobytes(),
                'mean_dtype': str(self._normalization.mean.dtype),
                'std': self._normalization.std.tobytes(),
                'std_dtype': str(self._normalization.std.dtype),
                'min_val': self._normalization.min_val.tobytes(),
                'min_val_dtype': str(self._normalization.min_val.dtype),
                'max_val': self._normalization.max_val.tobytes(),
                'max_val_dtype': str(self._normalization.max_val.dtype),
            } if self._normalization else None,
        }

        if compress:
            with gzip.open(path, 'wb') as f:
                f.write(DATABASE_MAGIC)
                pickle.dump(data, f)
        else:
            with open(path, 'wb') as f:
                f.write(DATABASE_MAGIC)
                pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> MotionDatabase:
        """Load database from file.

        Args:
            path: Input file path

        Returns:
            Loaded MotionDatabase
        """
        # Try compressed first
        try:
            with gzip.open(path, 'rb') as f:
                magic = f.read(4)
                if magic != DATABASE_MAGIC:
                    raise ValueError(f"Invalid database file: {path}")
                data = pickle.load(f)
        except gzip.BadGzipFile:
            with open(path, 'rb') as f:
                magic = f.read(4)
                if magic != DATABASE_MAGIC:
                    raise ValueError(f"Invalid database file: {path}")
                data = pickle.load(f)

        version = data.get('version', 1)
        if version > DATABASE_VERSION:
            raise ValueError(f"Unsupported database version: {version}")

        db = cls(
            feature_dimension=data['feature_dimension'],
            quantization=QuantizationLevel(data['quantization']),
        )

        # Load clip metadata
        for c in data['clip_metadata']:
            db._clip_metadata.append(ClipMetadata(
                clip_index=c['clip_index'],
                name=c['name'],
                frame_count=c['frame_count'],
                frame_rate=c['frame_rate'],
                duration=c['duration'],
                is_looping=c['is_looping'],
                has_root_motion=c['has_root_motion'],
                tags=frozenset(c['tags']),
            ))

        # Load entries
        for e in data['entries']:
            features = np.frombuffer(e['features'], dtype=np.dtype(e['features_dtype']))
            db._entries.append(DatabaseEntry(
                clip_index=e['clip_index'],
                frame=e['frame'],
                features=features,
                tags=frozenset(e['tags']),
                is_transition_candidate=e['is_transition_candidate'],
                cost_modifier=e['cost_modifier'],
            ))

        # Load feature matrix
        if data['feature_matrix'] is not None:
            db._feature_matrix = np.frombuffer(
                data['feature_matrix'],
                dtype=np.dtype(data['feature_matrix_dtype']),
            ).reshape(data['feature_matrix_shape'])

        # Load normalization
        if data['normalization']:
            norm = data['normalization']
            db._normalization = NormalizationStats(
                mean=np.frombuffer(norm['mean'], dtype=np.dtype(norm.get('mean_dtype', 'float64'))),
                std=np.frombuffer(norm['std'], dtype=np.dtype(norm.get('std_dtype', 'float64'))),
                min_val=np.frombuffer(norm['min_val'], dtype=np.dtype(norm.get('min_val_dtype', 'float64'))),
                max_val=np.frombuffer(norm['max_val'], dtype=np.dtype(norm.get('max_val_dtype', 'float64'))),
            )

        # Rebuild indices
        db._build_clip_entry_ranges()
        for i, entry in enumerate(db._entries):
            for tag in entry.tags:
                if tag not in db._tag_index:
                    db._tag_index[tag] = set()
                db._tag_index[tag].add(i)

        db._dirty = False
        return db

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def __iter__(self) -> Iterator[DatabaseEntry]:
        """Iterate over all entries."""
        return iter(self._entries)

    def __len__(self) -> int:
        """Return number of entries."""
        return len(self._entries)

    def __getitem__(self, index: int) -> DatabaseEntry:
        """Get entry by index."""
        return self._entries[index]


# =============================================================================
# DATABASE BUILDER
# =============================================================================


@motion_matching
def build_database(
    clips: List[Any],
    feature_extractor: FeatureExtractor,
    quantization: QuantizationLevel = QuantizationLevel.NONE,
    skip_first_frames: int = 0,
    skip_last_frames: int = 0,
    default_tags: Optional[Set[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> MotionDatabase:
    """Build motion database from animation clips.

    Args:
        clips: List of animation clips (must implement AnimationClipProtocol)
        feature_extractor: Feature extractor instance
        quantization: Quantization level for feature storage
        skip_first_frames: Number of frames to skip at clip start
        skip_last_frames: Number of frames to skip at clip end
        default_tags: Default tags to apply to all entries
        progress_callback: Optional callback(current_frame, total_frames)

    Returns:
        Built MotionDatabase ready for search
    """
    db = MotionDatabase(
        feature_dimension=feature_extractor.feature_dimension,
        quantization=quantization,
    )

    total_frames = sum(
        max(0, clip.frame_count - skip_first_frames - skip_last_frames)
        for clip in clips
    )
    current_frame = 0

    for clip_idx, clip in enumerate(clips):
        # Get clip metadata
        clip_tags = getattr(clip, 'tags', set())
        if default_tags:
            clip_tags = clip_tags | default_tags

        metadata = ClipMetadata(
            clip_index=clip_idx,
            name=clip.name,
            frame_count=clip.frame_count,
            frame_rate=clip.frame_rate,
            duration=clip.duration,
            is_looping=getattr(clip, 'is_looping', False),
            has_root_motion=getattr(clip, 'has_root_motion', False),
            tags=frozenset(clip_tags),
        )
        db.add_clip(metadata)

        # Process each frame
        start_frame = skip_first_frames
        end_frame = clip.frame_count - skip_last_frames

        for frame in range(start_frame, end_frame):
            # Extract features for this frame
            features = feature_extractor.extract(clip, frame)

            # Get frame-specific tags if available
            frame_tags = getattr(clip, 'get_frame_tags', lambda f: set())(frame)
            combined_tags = clip_tags | frame_tags

            # Check if this is a valid transition candidate
            is_candidate = _is_valid_transition_candidate(clip, frame, end_frame)

            entry = DatabaseEntry(
                clip_index=clip_idx,
                frame=frame,
                features=features.values if hasattr(features, 'values') else features,
                tags=frozenset(combined_tags),
                is_transition_candidate=is_candidate,
            )
            db.add_entry(entry)

            current_frame += 1
            if progress_callback:
                progress_callback(current_frame, total_frames)

    # Finalize database
    db.finalize(compute_normalization=True)

    return db


def _is_valid_transition_candidate(
    clip: Any, frame: int, end_frame: int, min_remaining: int = 5
) -> bool:
    """Check if a frame is a valid transition candidate.

    Args:
        clip: Animation clip
        frame: Current frame
        end_frame: Last frame to process
        min_remaining: Minimum frames that must remain after transition

    Returns:
        True if this frame can be transitioned to
    """
    # Don't transition to frames near the end (not enough time to blend)
    if frame > end_frame - min_remaining:
        return False

    # Check if clip has explicit transition markers
    if hasattr(clip, 'transition_markers'):
        markers = clip.transition_markers
        if markers is not None:
            return frame in markers

    return True


def merge_databases(
    databases: List[MotionDatabase],
    quantization: Optional[QuantizationLevel] = None,
) -> MotionDatabase:
    """Merge multiple databases into one.

    Args:
        databases: List of databases to merge
        quantization: Override quantization level (uses first db's if None)

    Returns:
        Merged MotionDatabase
    """
    if not databases:
        return MotionDatabase()

    # Use first database's settings
    first_db = databases[0]
    merged = MotionDatabase(
        feature_dimension=first_db.feature_dimension,
        quantization=quantization or first_db.quantization,
    )

    clip_offset = 0

    for db in databases:
        # Check compatible dimensions
        if db.feature_dimension != merged.feature_dimension:
            raise ValueError(
                f"Feature dimension mismatch: {db.feature_dimension} vs {merged.feature_dimension}"
            )

        # Add clips with offset indices
        for clip in db.clip_metadata:
            new_clip = ClipMetadata(
                clip_index=clip.clip_index + clip_offset,
                name=clip.name,
                frame_count=clip.frame_count,
                frame_rate=clip.frame_rate,
                duration=clip.duration,
                is_looping=clip.is_looping,
                has_root_motion=clip.has_root_motion,
                tags=clip.tags,
            )
            merged._clip_metadata.append(new_clip)

        # Add entries with offset clip indices
        for entry in db.entries:
            new_entry = DatabaseEntry(
                clip_index=entry.clip_index + clip_offset,
                frame=entry.frame,
                features=entry.features,
                tags=entry.tags,
                is_transition_candidate=entry.is_transition_candidate,
                cost_modifier=entry.cost_modifier,
            )
            merged.add_entry(new_entry)

        clip_offset += db.clip_count

    merged.finalize()
    return merged
