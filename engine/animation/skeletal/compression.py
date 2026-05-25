"""Animation data compression.

This module provides compression algorithms for animation clips:
- Quantization: Reduce precision to save memory
- Curve fitting: Remove redundant keyframes
- ACL-style compression: High-quality variable bitrate encoding
- Error metrics for quality validation
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict, Callable

from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat
from engine.core.constants import MATH_EPSILON

from engine.animation.skeletal.constants import (
    COMPRESSION_BITS_LOW,
    COMPRESSION_BITS_MEDIUM,
    COMPRESSION_BITS_HIGH,
    DEFAULT_TRANSLATION_BITS,
    DEFAULT_ROTATION_BITS,
    DEFAULT_SCALE_BITS,
    MIN_COMPRESSION_BITS,
    DEFAULT_TRANSLATION_ERROR_THRESHOLD,
    DEFAULT_ROTATION_ERROR_THRESHOLD,
    DEFAULT_SCALE_ERROR_THRESHOLD,
    DEFAULT_CURVE_FITTING_TOLERANCE,
    QUANTIZATION_RANGE_PADDING_FACTOR,
    QUANTIZATION_MIN_PADDING,
    DEFAULT_FRAME_RATE,
    DEFAULT_SAMPLE_RATE,
    COMPRESSION_HEADER_SIZE_ESTIMATE,
    CONSTANT_VALUE_STORAGE_SIZE,
    TIME_EPSILON,
)

if TYPE_CHECKING:
    from engine.animation.skeletal.clip import AnimationClip


class CompressionMethod(Enum):
    """Animation compression methods."""
    NONE = auto()        # No compression, raw keyframes
    QUANTIZED = auto()   # Fixed-point quantization
    CURVE = auto()       # Curve fitting with keyframe reduction
    ACL = auto()         # ACL-style variable bitrate


class TrackType(Enum):
    """Type of animation track."""
    TRANSLATION = auto()
    ROTATION = auto()
    SCALE = auto()


@dataclass
class CompressionSettings:
    """Settings controlling compression behavior.

    Attributes:
        method: Compression method to use
        translation_error_threshold: Max allowed translation error
        rotation_error_threshold: Max allowed rotation error (radians)
        scale_error_threshold: Max allowed scale error
        translation_bits: Bits per component for quantized translation
        rotation_bits: Bits per component for quantized rotation
        scale_bits: Bits per component for quantized scale
        curve_fitting_tolerance: Error tolerance for curve fitting
        sample_rate: Target sample rate (0 = keep original)
    """
    method: CompressionMethod = CompressionMethod.QUANTIZED
    translation_error_threshold: float = DEFAULT_TRANSLATION_ERROR_THRESHOLD
    rotation_error_threshold: float = DEFAULT_ROTATION_ERROR_THRESHOLD
    scale_error_threshold: float = DEFAULT_SCALE_ERROR_THRESHOLD
    translation_bits: int = DEFAULT_TRANSLATION_BITS
    rotation_bits: int = DEFAULT_ROTATION_BITS
    scale_bits: int = DEFAULT_SCALE_BITS
    curve_fitting_tolerance: float = DEFAULT_CURVE_FITTING_TOLERANCE
    sample_rate: float = DEFAULT_SAMPLE_RATE


@dataclass
class Keyframe:
    """Single animation keyframe.

    Attributes:
        time: Time of keyframe in seconds
        value: Value at this keyframe (Vec3 or Quat)
    """
    time: float
    value: any  # Vec3, Quat, or float


@dataclass
class AnimationTrack:
    """Single bone track with keyframes.

    Attributes:
        bone_index: Index of bone this track affects
        track_type: Type of track (translation, rotation, scale)
        keyframes: List of keyframes
    """
    bone_index: int
    track_type: TrackType
    keyframes: List[Keyframe] = field(default_factory=list)

    @property
    def duration(self) -> float:
        if not self.keyframes:
            return 0.0
        return self.keyframes[-1].time

    def sample(self, time: float) -> any:
        """Sample track value at a specific time."""
        if not self.keyframes:
            return None

        # Before first keyframe
        if time <= self.keyframes[0].time:
            return self.keyframes[0].value

        # After last keyframe
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1].value

        # Find keyframe interval
        for i in range(len(self.keyframes) - 1):
            k0 = self.keyframes[i]
            k1 = self.keyframes[i + 1]
            if k0.time <= time <= k1.time:
                time_delta = k1.time - k0.time
                t = (time - k0.time) / time_delta if time_delta > TIME_EPSILON else 0.0
                return self._interpolate(k0.value, k1.value, t)

        return self.keyframes[-1].value

    def _interpolate(self, a: any, b: any, t: float) -> any:
        """Interpolate between two values."""
        if isinstance(a, Vec3):
            return a.lerp(b, t)
        elif isinstance(a, Quat):
            return a.slerp(b, t)
        elif isinstance(a, (int, float)):
            return a + (b - a) * t
        return b


@dataclass
class QuantizedValue:
    """Quantized value with min/max range.

    Stores values as fixed-point integers with a known range.
    """
    data: bytes  # Packed quantized values
    min_value: float
    max_value: float
    bits_per_sample: int
    sample_count: int

    @staticmethod
    def quantize_float(
        value: float,
        min_val: float,
        max_val: float,
        bits: int
    ) -> int:
        """Quantize a single float to integer."""
        if max_val <= min_val:
            return 0
        max_int = (1 << bits) - 1
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))
        return int(normalized * max_int + 0.5)

    @staticmethod
    def dequantize_float(
        quantized: int,
        min_val: float,
        max_val: float,
        bits: int
    ) -> float:
        """Dequantize integer back to float."""
        max_int = (1 << bits) - 1
        if max_int == 0:
            return min_val
        normalized = quantized / max_int
        return min_val + normalized * (max_val - min_val)


@dataclass
class QuantizedCurve:
    """Quantized animation curve.

    Attributes:
        min_values: Minimum values per component
        max_values: Maximum values per component
        bits_per_sample: Bits used per sample per component
        data: Packed quantized data
        sample_count: Number of samples
        component_count: Number of components (3 for Vec3, 4 for Quat)
    """
    min_values: List[float] = field(default_factory=list)
    max_values: List[float] = field(default_factory=list)
    bits_per_sample: int = 16
    data: bytes = b""
    sample_count: int = 0
    component_count: int = 3

    def get_value_at_index(self, index: int) -> List[float]:
        """Get decompressed values at a sample index."""
        if index < 0 or index >= self.sample_count:
            return [0.0] * self.component_count

        values = []
        for comp in range(self.component_count):
            # Calculate bit position
            bit_offset = (index * self.component_count + comp) * self.bits_per_sample
            byte_offset = bit_offset // 8
            bit_in_byte = bit_offset % 8

            # Extract quantized value (simplified - assumes aligned)
            if self.bits_per_sample == 16 and byte_offset + 1 < len(self.data):
                quantized = struct.unpack_from('<H', self.data, byte_offset)[0]
            elif self.bits_per_sample == 8 and byte_offset < len(self.data):
                quantized = self.data[byte_offset]
            else:
                quantized = 0

            # Dequantize
            value = QuantizedValue.dequantize_float(
                quantized,
                self.min_values[comp],
                self.max_values[comp],
                self.bits_per_sample
            )
            values.append(value)

        return values


@dataclass
class CompressedTrack:
    """Compressed animation track.

    Attributes:
        bone_index: Index of affected bone
        track_type: Type of track
        quantized_curve: Quantized curve data
        constant_value: Optional constant value (if track is static)
        is_constant: Whether track has constant value
        sample_times: Optional explicit sample times (for variable rate)
    """
    bone_index: int
    track_type: TrackType
    quantized_curve: Optional[QuantizedCurve] = None
    constant_value: Optional[any] = None
    is_constant: bool = False
    sample_times: Optional[List[float]] = None


@dataclass
class CompressedClip:
    """Complete compressed animation clip.

    Attributes:
        name: Clip name
        duration: Clip duration in seconds
        frame_rate: Original frame rate
        bone_count: Number of bones
        tracks: List of compressed tracks
        compression_method: Method used for compression
        compression_ratio: Achieved compression ratio
        error_metrics: Per-bone error metrics
    """
    name: str = ""
    duration: float = 0.0
    frame_rate: float = 30.0
    bone_count: int = 0
    tracks: List[CompressedTrack] = field(default_factory=list)
    compression_method: CompressionMethod = CompressionMethod.NONE
    compression_ratio: float = 1.0
    error_metrics: Optional[CompressionErrorMetrics] = None

    def get_track(self, bone_index: int, track_type: TrackType) -> Optional[CompressedTrack]:
        """Get track for a specific bone and type."""
        for track in self.tracks:
            if track.bone_index == bone_index and track.track_type == track_type:
                return track
        return None


@dataclass
class CompressionErrorMetrics:
    """Error metrics for compression quality.

    Attributes:
        max_translation_error: Maximum translation error across all bones
        mean_translation_error: Mean translation error
        max_rotation_error: Maximum rotation error (radians)
        mean_rotation_error: Mean rotation error
        max_scale_error: Maximum scale error
        mean_scale_error: Mean scale error
        per_bone_max_error: Maximum error per bone
        per_bone_mean_error: Mean error per bone
    """
    max_translation_error: float = 0.0
    mean_translation_error: float = 0.0
    max_rotation_error: float = 0.0
    mean_rotation_error: float = 0.0
    max_scale_error: float = 0.0
    mean_scale_error: float = 0.0
    per_bone_max_error: Dict[int, float] = field(default_factory=dict)
    per_bone_mean_error: Dict[int, float] = field(default_factory=dict)

    def meets_threshold(self, settings: CompressionSettings) -> bool:
        """Check if error metrics meet quality thresholds."""
        if self.max_translation_error > settings.translation_error_threshold:
            return False
        if self.max_rotation_error > settings.rotation_error_threshold:
            return False
        if self.max_scale_error > settings.scale_error_threshold:
            return False
        return True


@dataclass
class AnimationClipData:
    """Uncompressed animation clip data for compression input.

    Attributes:
        name: Clip name
        duration: Duration in seconds
        frame_rate: Frame rate
        bone_count: Number of bones
        tracks: List of animation tracks
    """
    name: str = ""
    duration: float = 0.0
    frame_rate: float = 30.0
    bone_count: int = 0
    tracks: List[AnimationTrack] = field(default_factory=list)


def _compute_range(values: List[float]) -> Tuple[float, float]:
    """Compute min/max range for a list of values."""
    if not values:
        return 0.0, 0.0
    return min(values), max(values)


def _quantize_track(
    track: AnimationTrack,
    settings: CompressionSettings
) -> CompressedTrack:
    """Quantize a single track."""
    if not track.keyframes:
        return CompressedTrack(
            bone_index=track.bone_index,
            track_type=track.track_type,
            is_constant=True,
            constant_value=None
        )

    # Check if track is constant
    first_value = track.keyframes[0].value
    is_constant = all(
        _values_equal(kf.value, first_value, MATH_EPSILON)
        for kf in track.keyframes
    )

    if is_constant:
        return CompressedTrack(
            bone_index=track.bone_index,
            track_type=track.track_type,
            is_constant=True,
            constant_value=first_value
        )

    # Determine bits and component count
    if track.track_type == TrackType.ROTATION:
        bits = settings.rotation_bits
        component_count = 4
    elif track.track_type == TrackType.SCALE:
        bits = settings.scale_bits
        component_count = 3
    else:
        bits = settings.translation_bits
        component_count = 3

    # Extract all component values
    all_components: List[List[float]] = [[] for _ in range(component_count)]
    for kf in track.keyframes:
        components = _value_to_components(kf.value)
        for i, c in enumerate(components[:component_count]):
            all_components[i].append(c)

    # Compute ranges
    min_values = []
    max_values = []
    for comp_values in all_components:
        min_v, max_v = _compute_range(comp_values)
        # Add small padding to avoid edge cases
        padding = max(QUANTIZATION_MIN_PADDING, (max_v - min_v) * QUANTIZATION_RANGE_PADDING_FACTOR)
        min_values.append(min_v - padding)
        max_values.append(max_v + padding)

    # Quantize and pack data
    data = bytearray()
    for kf in track.keyframes:
        components = _value_to_components(kf.value)
        for i in range(component_count):
            quantized = QuantizedValue.quantize_float(
                components[i], min_values[i], max_values[i], bits
            )
            if bits == 16:
                data.extend(struct.pack('<H', quantized))
            elif bits == 8:
                data.append(quantized & 0xFF)
            else:
                # Handle other bit depths
                data.extend(struct.pack('<I', quantized)[:((bits + 7) // 8)])

    quantized_curve = QuantizedCurve(
        min_values=min_values,
        max_values=max_values,
        bits_per_sample=bits,
        data=bytes(data),
        sample_count=len(track.keyframes),
        component_count=component_count
    )

    return CompressedTrack(
        bone_index=track.bone_index,
        track_type=track.track_type,
        quantized_curve=quantized_curve,
        sample_times=[kf.time for kf in track.keyframes]
    )


def _values_equal(a: any, b: any, tolerance: float) -> bool:
    """Check if two values are approximately equal."""
    if isinstance(a, Vec3):
        return (a - b).length() < tolerance
    elif isinstance(a, Quat):
        return abs(a.dot(b)) > 1.0 - tolerance
    else:
        return abs(a - b) < tolerance


def _value_to_components(value: any) -> List[float]:
    """Convert a value to list of float components."""
    if isinstance(value, Vec3):
        return [value.x, value.y, value.z]
    elif isinstance(value, Quat):
        return [value.x, value.y, value.z, value.w]
    elif isinstance(value, (int, float)):
        return [float(value)]
    return []


def _components_to_value(components: List[float], track_type: TrackType) -> any:
    """Convert components back to typed value."""
    if track_type == TrackType.ROTATION:
        if len(components) >= 4:
            return Quat(components[0], components[1], components[2], components[3]).normalized()
        return Quat.identity()
    elif track_type in (TrackType.TRANSLATION, TrackType.SCALE):
        if len(components) >= 3:
            return Vec3(components[0], components[1], components[2])
        return Vec3.zero()
    return 0.0


def compress_clip(
    clip: AnimationClipData,
    settings: Optional[CompressionSettings] = None
) -> CompressedClip:
    """Compress an animation clip.

    Args:
        clip: Uncompressed animation clip data
        settings: Compression settings

    Returns:
        Compressed clip
    """
    if settings is None:
        settings = CompressionSettings()

    if settings.method == CompressionMethod.NONE:
        return _compress_none(clip)
    elif settings.method == CompressionMethod.QUANTIZED:
        return _compress_quantized(clip, settings)
    elif settings.method == CompressionMethod.CURVE:
        return _compress_curve_fitting(clip, settings)
    elif settings.method == CompressionMethod.ACL:
        return _compress_acl(clip, settings)
    else:
        return _compress_none(clip)


def _compress_none(clip: AnimationClipData) -> CompressedClip:
    """No compression - store raw data."""
    compressed_tracks = []

    for track in clip.tracks:
        if not track.keyframes:
            continue

        # Store as quantized with full precision
        compressed = _quantize_track(track, CompressionSettings(
            translation_bits=COMPRESSION_BITS_HIGH,
            rotation_bits=COMPRESSION_BITS_HIGH,
            scale_bits=COMPRESSION_BITS_HIGH
        ))
        compressed_tracks.append(compressed)

    return CompressedClip(
        name=clip.name,
        duration=clip.duration,
        frame_rate=clip.frame_rate,
        bone_count=clip.bone_count,
        tracks=compressed_tracks,
        compression_method=CompressionMethod.NONE,
        compression_ratio=1.0
    )


def _compress_quantized(
    clip: AnimationClipData,
    settings: CompressionSettings
) -> CompressedClip:
    """Quantization-based compression."""
    compressed_tracks = []
    original_size = 0
    compressed_size = 0

    for track in clip.tracks:
        if not track.keyframes:
            continue

        # Calculate original size (approximate)
        if track.track_type == TrackType.ROTATION:
            original_size += len(track.keyframes) * 4 * 4  # 4 floats
        else:
            original_size += len(track.keyframes) * 3 * 4  # 3 floats

        compressed = _quantize_track(track, settings)
        compressed_tracks.append(compressed)

        # Calculate compressed size
        if compressed.quantized_curve:
            compressed_size += len(compressed.quantized_curve.data)
        elif compressed.is_constant:
            compressed_size += 16  # Constant value storage

    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

    return CompressedClip(
        name=clip.name,
        duration=clip.duration,
        frame_rate=clip.frame_rate,
        bone_count=clip.bone_count,
        tracks=compressed_tracks,
        compression_method=CompressionMethod.QUANTIZED,
        compression_ratio=compression_ratio
    )


def _compress_curve_fitting(
    clip: AnimationClipData,
    settings: CompressionSettings
) -> CompressedClip:
    """Curve fitting compression with keyframe reduction."""
    compressed_tracks = []

    for track in clip.tracks:
        if not track.keyframes:
            continue

        # Reduce keyframes using curve fitting
        reduced_keyframes = _reduce_keyframes(
            track.keyframes,
            track.track_type,
            settings.curve_fitting_tolerance
        )

        # Create reduced track
        reduced_track = AnimationTrack(
            bone_index=track.bone_index,
            track_type=track.track_type,
            keyframes=reduced_keyframes
        )

        # Quantize the reduced track
        compressed = _quantize_track(reduced_track, settings)
        compressed_tracks.append(compressed)

    # Calculate compression ratio
    original_kf_count = sum(len(t.keyframes) for t in clip.tracks)
    compressed_kf_count = sum(
        len(t.sample_times) if t.sample_times else 1
        for t in compressed_tracks
    )
    compression_ratio = original_kf_count / compressed_kf_count if compressed_kf_count > 0 else 1.0

    return CompressedClip(
        name=clip.name,
        duration=clip.duration,
        frame_rate=clip.frame_rate,
        bone_count=clip.bone_count,
        tracks=compressed_tracks,
        compression_method=CompressionMethod.CURVE,
        compression_ratio=compression_ratio
    )


def _reduce_keyframes(
    keyframes: List[Keyframe],
    track_type: TrackType,
    tolerance: float
) -> List[Keyframe]:
    """Reduce keyframes using Ramer-Douglas-Peucker-like algorithm."""
    if len(keyframes) <= 2:
        return list(keyframes)

    # Always keep first and last
    result = [keyframes[0]]

    def can_remove(start_idx: int, end_idx: int) -> bool:
        """Check if intermediate keyframes can be removed."""
        if end_idx - start_idx <= 1:
            return True

        start_kf = keyframes[start_idx]
        end_kf = keyframes[end_idx]
        time_span = end_kf.time - start_kf.time

        for i in range(start_idx + 1, end_idx):
            mid_kf = keyframes[i]
            t = (mid_kf.time - start_kf.time) / time_span if time_span > TIME_EPSILON else 0.0

            # Interpolate expected value
            if isinstance(start_kf.value, Vec3):
                expected = start_kf.value.lerp(end_kf.value, t)
                error = (mid_kf.value - expected).length()
            elif isinstance(start_kf.value, Quat):
                expected = start_kf.value.slerp(end_kf.value, t)
                error = 1.0 - abs(mid_kf.value.dot(expected))
            else:
                expected = start_kf.value + (end_kf.value - start_kf.value) * t
                error = abs(mid_kf.value - expected)

            if error > tolerance:
                return False

        return True

    # Simplified algorithm - check each potential removal
    i = 0
    while i < len(keyframes) - 1:
        # Try to skip ahead as far as possible
        best_end = i + 1
        for j in range(i + 2, len(keyframes)):
            if can_remove(i, j):
                best_end = j
            else:
                break

        if best_end < len(keyframes) - 1:
            result.append(keyframes[best_end])
        i = best_end

    result.append(keyframes[-1])
    return result


def _compress_acl(
    clip: AnimationClipData,
    settings: CompressionSettings
) -> CompressedClip:
    """ACL-style variable bitrate compression.

    This is a simplified version of ACL compression that uses
    adaptive quantization per-track based on error thresholds.
    """
    compressed_tracks = []

    for track in clip.tracks:
        if not track.keyframes:
            continue

        # Determine optimal bit depth per track
        if track.track_type == TrackType.ROTATION:
            threshold = settings.rotation_error_threshold
            base_bits = settings.rotation_bits
        elif track.track_type == TrackType.SCALE:
            threshold = settings.scale_error_threshold
            base_bits = settings.scale_bits
        else:
            threshold = settings.translation_error_threshold
            base_bits = settings.translation_bits

        # Try progressively lower bit depths
        optimal_bits = base_bits
        for bits in [base_bits, base_bits - 4, base_bits - 8, MIN_COMPRESSION_BITS]:
            if bits < MIN_COMPRESSION_BITS:
                continue

            test_settings = CompressionSettings(
                translation_bits=bits,
                rotation_bits=bits,
                scale_bits=bits
            )
            test_compressed = _quantize_track(track, test_settings)

            # Verify error is within threshold
            error = _compute_track_error(track, test_compressed)
            if error <= threshold:
                optimal_bits = bits
            else:
                break

        # Compress with optimal bits
        final_settings = CompressionSettings(
            translation_bits=optimal_bits,
            rotation_bits=optimal_bits,
            scale_bits=optimal_bits
        )
        compressed = _quantize_track(track, final_settings)
        compressed_tracks.append(compressed)

    return CompressedClip(
        name=clip.name,
        duration=clip.duration,
        frame_rate=clip.frame_rate,
        bone_count=clip.bone_count,
        tracks=compressed_tracks,
        compression_method=CompressionMethod.ACL,
        compression_ratio=1.0  # Would need proper calculation
    )


def _compute_track_error(
    original: AnimationTrack,
    compressed: CompressedTrack
) -> float:
    """Compute maximum error between original and compressed track."""
    if compressed.is_constant or not original.keyframes:
        return 0.0

    max_error = 0.0

    for i, kf in enumerate(original.keyframes):
        # Get decompressed value
        if compressed.quantized_curve:
            components = compressed.quantized_curve.get_value_at_index(i)
            decompressed = _components_to_value(components, original.track_type)
        else:
            continue

        # Compute error
        if isinstance(kf.value, Vec3):
            error = (kf.value - decompressed).length()
        elif isinstance(kf.value, Quat):
            error = 1.0 - abs(kf.value.dot(decompressed))
        else:
            error = abs(kf.value - decompressed)

        max_error = max(max_error, error)

    return max_error


def decompress_track(
    track: CompressedTrack,
    sample_count: Optional[int] = None
) -> AnimationTrack:
    """Decompress a single track back to keyframes.

    Args:
        track: Compressed track
        sample_count: Optional target sample count (for resampling)

    Returns:
        Decompressed animation track
    """
    if track.is_constant:
        # Return single keyframe with constant value
        return AnimationTrack(
            bone_index=track.bone_index,
            track_type=track.track_type,
            keyframes=[Keyframe(0.0, track.constant_value)] if track.constant_value else []
        )

    if track.quantized_curve is None:
        return AnimationTrack(
            bone_index=track.bone_index,
            track_type=track.track_type
        )

    keyframes = []
    curve = track.quantized_curve
    times = track.sample_times or [i / DEFAULT_FRAME_RATE for i in range(curve.sample_count)]

    for i in range(curve.sample_count):
        components = curve.get_value_at_index(i)
        value = _components_to_value(components, track.track_type)
        time = times[i] if i < len(times) else i / DEFAULT_FRAME_RATE
        keyframes.append(Keyframe(time, value))

    return AnimationTrack(
        bone_index=track.bone_index,
        track_type=track.track_type,
        keyframes=keyframes
    )


def decompress_clip(compressed: CompressedClip) -> AnimationClipData:
    """Decompress a complete animation clip.

    Args:
        compressed: Compressed clip

    Returns:
        Decompressed animation clip data
    """
    tracks = [decompress_track(t) for t in compressed.tracks]

    return AnimationClipData(
        name=compressed.name,
        duration=compressed.duration,
        frame_rate=compressed.frame_rate,
        bone_count=compressed.bone_count,
        tracks=tracks
    )


def compute_compression_error(
    original: AnimationClipData,
    compressed: CompressedClip
) -> CompressionErrorMetrics:
    """Compute error metrics between original and compressed clips.

    Args:
        original: Original uncompressed clip
        compressed: Compressed clip

    Returns:
        Error metrics
    """
    metrics = CompressionErrorMetrics()

    translation_errors = []
    rotation_errors = []
    scale_errors = []
    per_bone_errors: Dict[int, List[float]] = {}

    # Decompress for comparison
    decompressed = decompress_clip(compressed)

    # Build track lookup
    original_tracks: Dict[Tuple[int, TrackType], AnimationTrack] = {
        (t.bone_index, t.track_type): t for t in original.tracks
    }
    decompressed_tracks: Dict[Tuple[int, TrackType], AnimationTrack] = {
        (t.bone_index, t.track_type): t for t in decompressed.tracks
    }

    # Compare each track
    for key, orig_track in original_tracks.items():
        decomp_track = decompressed_tracks.get(key)
        if decomp_track is None:
            continue

        bone_idx = key[0]
        if bone_idx not in per_bone_errors:
            per_bone_errors[bone_idx] = []

        # Sample at multiple points
        sample_times = [kf.time for kf in orig_track.keyframes]
        for time in sample_times:
            orig_value = orig_track.sample(time)
            decomp_value = decomp_track.sample(time)

            if orig_value is None or decomp_value is None:
                continue

            if isinstance(orig_value, Vec3):
                error = (orig_value - decomp_value).length()
                if key[1] == TrackType.TRANSLATION:
                    translation_errors.append(error)
                else:
                    scale_errors.append(error)
            elif isinstance(orig_value, Quat):
                error = 1.0 - abs(orig_value.dot(decomp_value))
                rotation_errors.append(error)
            else:
                error = abs(orig_value - decomp_value)
                translation_errors.append(error)

            per_bone_errors[bone_idx].append(error)

    # Compute aggregate metrics
    if translation_errors:
        metrics.max_translation_error = max(translation_errors)
        metrics.mean_translation_error = sum(translation_errors) / len(translation_errors)

    if rotation_errors:
        metrics.max_rotation_error = max(rotation_errors)
        metrics.mean_rotation_error = sum(rotation_errors) / len(rotation_errors)

    if scale_errors:
        metrics.max_scale_error = max(scale_errors)
        metrics.mean_scale_error = sum(scale_errors) / len(scale_errors)

    for bone_idx, errors in per_bone_errors.items():
        if errors:
            metrics.per_bone_max_error[bone_idx] = max(errors)
            metrics.per_bone_mean_error[bone_idx] = sum(errors) / len(errors)

    return metrics


def estimate_compressed_size(
    clip: AnimationClipData,
    settings: CompressionSettings
) -> int:
    """Estimate compressed size in bytes without actually compressing.

    Args:
        clip: Clip to estimate
        settings: Compression settings

    Returns:
        Estimated size in bytes
    """
    total_bytes = 0

    # Header (approximate)
    total_bytes += COMPRESSION_HEADER_SIZE_ESTIMATE

    for track in clip.tracks:
        if not track.keyframes:
            continue

        # Check if constant
        first_value = track.keyframes[0].value
        is_constant = all(
            _values_equal(kf.value, first_value, MATH_EPSILON)
            for kf in track.keyframes
        )

        if is_constant:
            # Just store one value
            total_bytes += CONSTANT_VALUE_STORAGE_SIZE
            continue

        # Estimate based on keyframe count and bit depth
        kf_count = len(track.keyframes)
        if settings.method == CompressionMethod.CURVE:
            # Assume 50% keyframe reduction
            kf_count = max(2, kf_count // 2)

        if track.track_type == TrackType.ROTATION:
            bits = settings.rotation_bits
            components = 4
        else:
            bits = settings.translation_bits if track.track_type == TrackType.TRANSLATION else settings.scale_bits
            components = 3

        bytes_per_kf = (bits * components + 7) // 8
        total_bytes += kf_count * bytes_per_kf

        # Time data
        total_bytes += kf_count * 2  # 16-bit times

    return total_bytes
