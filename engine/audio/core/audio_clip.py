"""
Audio Clip Management

Handles audio clip loading, format detection, and metadata.
"""

from __future__ import annotations

import hashlib
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, BinaryIO
from enum import IntEnum, auto

from .config import (
    AudioFormat,
    ChannelLayout,
    AudioCategory,
    MemoryPoolType,
    DEFAULT_SAMPLE_RATE,
    COMPRESSED_FORMATS,
    FORMAT_BITS_PER_SAMPLE,
)


class ClipLoadState(IntEnum):
    """Loading state of an audio clip."""
    UNLOADED = 0
    LOADING = auto()
    LOADED = auto()
    STREAMING = auto()
    ERROR = auto()


@dataclass
class AudioClipMetadata:
    """Metadata for an audio clip."""
    duration_seconds: float = 0.0
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1
    format: AudioFormat = AudioFormat.PCM_INT16
    total_samples: int = 0
    file_size: int = 0
    compressed_size: int = 0
    loop_start: int = 0
    loop_end: int = 0
    has_loop_points: bool = False


@dataclass
class AudioClip:
    """
    Represents a loaded audio clip.

    Manages audio data, metadata, and reference counting for memory management.
    """

    id: str
    name: str
    path: Optional[Path] = None
    category: AudioCategory = AudioCategory.SFX
    pool_type: MemoryPoolType = MemoryPoolType.RESIDENT

    # Metadata
    metadata: AudioClipMetadata = field(default_factory=AudioClipMetadata)

    # State
    load_state: ClipLoadState = ClipLoadState.UNLOADED

    # Audio data (raw samples or compressed)
    _data: Optional[bytes] = field(default=None, repr=False)
    _decoded_data: Optional[bytes] = field(default=None, repr=False)

    # Reference counting
    _ref_count: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Priority for eviction (higher = keep longer)
    priority: int = 50

    # Last access time for LRU eviction
    last_access_time: float = 0.0

    def __post_init__(self) -> None:
        """Initialize after dataclass creation."""
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique ID for this clip."""
        content = f"{self.name}:{self.path}:{id(self)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    @property
    def duration(self) -> float:
        """Get clip duration in seconds."""
        return self.metadata.duration_seconds

    @property
    def sample_rate(self) -> int:
        """Get sample rate in Hz."""
        return self.metadata.sample_rate

    @property
    def channels(self) -> int:
        """Get number of channels."""
        return self.metadata.channels

    @property
    def format(self) -> AudioFormat:
        """Get audio format."""
        return self.metadata.format

    @property
    def is_compressed(self) -> bool:
        """Check if the clip uses a compressed format."""
        return self.metadata.format in COMPRESSED_FORMATS

    @property
    def is_loaded(self) -> bool:
        """Check if clip data is loaded."""
        return self.load_state == ClipLoadState.LOADED

    @property
    def is_streaming(self) -> bool:
        """Check if clip is set up for streaming."""
        return self.pool_type == MemoryPoolType.STREAMING

    @property
    def memory_size(self) -> int:
        """Get memory usage in bytes."""
        size = 0
        if self._data:
            size += len(self._data)
        if self._decoded_data:
            size += len(self._decoded_data)
        return size

    @property
    def data(self) -> Optional[bytes]:
        """Get raw audio data."""
        return self._data

    @property
    def decoded_data(self) -> Optional[bytes]:
        """Get decoded audio data (for compressed formats)."""
        if self._decoded_data:
            return self._decoded_data
        return self._data

    def add_ref(self) -> int:
        """Add a reference to this clip."""
        with self._lock:
            self._ref_count += 1
            return self._ref_count

    def release_ref(self) -> int:
        """Release a reference from this clip."""
        with self._lock:
            self._ref_count = max(0, self._ref_count - 1)
            return self._ref_count

    @property
    def ref_count(self) -> int:
        """Get current reference count."""
        with self._lock:
            return self._ref_count

    def load_from_file(self, file_path: Path) -> bool:
        """
        Load audio data from a file.

        Args:
            file_path: Path to the audio file

        Returns:
            True if loaded successfully
        """
        self.path = file_path
        self.load_state = ClipLoadState.LOADING

        try:
            if not file_path.exists():
                self.load_state = ClipLoadState.ERROR
                return False

            # Read file data
            with open(file_path, 'rb') as f:
                self._data = f.read()

            # Detect format and parse metadata
            self._detect_format_and_metadata()

            self.load_state = ClipLoadState.LOADED
            return True

        except Exception:
            self.load_state = ClipLoadState.ERROR
            return False

    def load_from_memory(
        self,
        data: bytes,
        format: AudioFormat,
        sample_rate: int,
        channels: int,
    ) -> bool:
        """
        Load audio data from memory.

        Args:
            data: Raw audio data
            format: Audio format
            sample_rate: Sample rate in Hz
            channels: Number of channels

        Returns:
            True if loaded successfully
        """
        self.load_state = ClipLoadState.LOADING

        try:
            self._data = data
            self.metadata.format = format
            self.metadata.sample_rate = sample_rate
            self.metadata.channels = channels
            self.metadata.file_size = len(data)

            # Calculate duration based on format
            self._calculate_duration()

            self.load_state = ClipLoadState.LOADED
            return True

        except Exception:
            self.load_state = ClipLoadState.ERROR
            return False

    def unload(self) -> None:
        """Unload audio data from memory."""
        with self._lock:
            if self._ref_count > 0:
                return  # Can't unload while in use

            self._data = None
            self._decoded_data = None
            self.load_state = ClipLoadState.UNLOADED

    def _detect_format_and_metadata(self) -> None:
        """Detect audio format and extract metadata from loaded data."""
        if not self._data or len(self._data) < 12:
            return

        # Check for WAV format
        if self._data[:4] == b'RIFF' and self._data[8:12] == b'WAVE':
            self._parse_wav_header()
        # Check for OGG/Vorbis
        elif self._data[:4] == b'OggS':
            self._parse_ogg_header()
        # Check for FLAC
        elif self._data[:4] == b'fLaC':
            self._parse_flac_header()
        # Check for MP3 (ID3 tag or sync word)
        elif self._data[:3] == b'ID3' or (self._data[0] == 0xFF and (self._data[1] & 0xE0) == 0xE0):
            self._parse_mp3_header()
        else:
            # Assume raw PCM
            self.metadata.format = AudioFormat.PCM_INT16
            self._calculate_duration()

    def _parse_wav_header(self) -> None:
        """Parse WAV file header for metadata."""
        if not self._data:
            return

        try:
            # Find fmt chunk
            pos = 12
            while pos < len(self._data) - 8:
                chunk_id = self._data[pos:pos+4]
                chunk_size = struct.unpack('<I', self._data[pos+4:pos+8])[0]

                if chunk_id == b'fmt ':
                    self._parse_wav_fmt_chunk(pos + 8, chunk_size)
                elif chunk_id == b'data':
                    self.metadata.file_size = chunk_size
                    break

                pos += 8 + chunk_size
                if chunk_size % 2:  # Padding
                    pos += 1

            self._calculate_duration()

        except Exception:
            pass

    def _parse_wav_fmt_chunk(self, offset: int, size: int) -> None:
        """Parse WAV fmt chunk."""
        if not self._data or offset + 16 > len(self._data):
            return

        fmt_data = self._data[offset:offset+size]

        audio_format = struct.unpack('<H', fmt_data[0:2])[0]
        channels = struct.unpack('<H', fmt_data[2:4])[0]
        sample_rate = struct.unpack('<I', fmt_data[4:8])[0]
        bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]

        self.metadata.channels = channels
        self.metadata.sample_rate = sample_rate

        # Map WAV format to AudioFormat
        if audio_format == 1:  # PCM
            if bits_per_sample == 16:
                self.metadata.format = AudioFormat.PCM_INT16
            elif bits_per_sample == 24:
                self.metadata.format = AudioFormat.PCM_INT24
            elif bits_per_sample == 32:
                self.metadata.format = AudioFormat.PCM_FLOAT32
        elif audio_format == 3:  # IEEE Float
            self.metadata.format = AudioFormat.PCM_FLOAT32
        elif audio_format == 17:  # ADPCM
            self.metadata.format = AudioFormat.ADPCM

    def _parse_ogg_header(self) -> None:
        """Parse OGG/Vorbis header for metadata."""
        self.metadata.format = AudioFormat.VORBIS
        # Simplified - would need full OGG parser for accurate metadata
        self.metadata.sample_rate = DEFAULT_SAMPLE_RATE
        self.metadata.channels = 2
        self._calculate_duration()

    def _parse_flac_header(self) -> None:
        """Parse FLAC header for metadata."""
        if not self._data or len(self._data) < 42:
            return

        try:
            # Parse STREAMINFO block
            metadata_block = self._data[4:42]

            # Sample rate is 20 bits starting at byte 10
            sample_rate = (metadata_block[10] << 12) | (metadata_block[11] << 4) | (metadata_block[12] >> 4)
            channels = ((metadata_block[12] >> 1) & 0x07) + 1
            bits_per_sample = ((metadata_block[12] & 0x01) << 4) | (metadata_block[13] >> 4) + 1

            total_samples = ((metadata_block[13] & 0x0F) << 32) | struct.unpack('>I', metadata_block[14:18])[0]

            self.metadata.sample_rate = sample_rate
            self.metadata.channels = channels
            self.metadata.total_samples = total_samples
            self.metadata.format = AudioFormat.PCM_INT16 if bits_per_sample <= 16 else AudioFormat.PCM_INT24

            if sample_rate > 0:
                self.metadata.duration_seconds = total_samples / sample_rate

        except Exception:
            pass

    def _parse_mp3_header(self) -> None:
        """Parse MP3 header for metadata."""
        self.metadata.format = AudioFormat.MP3
        self.metadata.sample_rate = DEFAULT_SAMPLE_RATE
        self.metadata.channels = 2
        # Would need full MP3 parser for accurate duration
        self._calculate_duration()

    def _calculate_duration(self) -> None:
        """Calculate duration based on format and data size."""
        if not self._data:
            return

        data_size = self.metadata.file_size or len(self._data)
        sample_rate = self.metadata.sample_rate
        channels = self.metadata.channels

        if sample_rate <= 0 or channels <= 0:
            return

        bits_per_sample = FORMAT_BITS_PER_SAMPLE.get(self.metadata.format, 16)

        if bits_per_sample > 0:
            bytes_per_sample = bits_per_sample // 8
            total_samples = data_size // (bytes_per_sample * channels)
            self.metadata.total_samples = total_samples
            self.metadata.duration_seconds = total_samples / sample_rate

    def get_samples(self, start_sample: int, num_samples: int) -> Optional[bytes]:
        """
        Get a range of samples from the clip.

        Args:
            start_sample: Starting sample index
            num_samples: Number of samples to retrieve

        Returns:
            Bytes containing the requested samples, or None if not available
        """
        if not self.is_loaded or not self._data:
            return None

        bits_per_sample = FORMAT_BITS_PER_SAMPLE.get(self.metadata.format, 16)
        bytes_per_sample = bits_per_sample // 8 if bits_per_sample > 0 else 2
        frame_size = bytes_per_sample * self.metadata.channels

        start_byte = start_sample * frame_size
        end_byte = start_byte + (num_samples * frame_size)

        data = self.decoded_data or self._data
        if start_byte >= len(data):
            return None

        return data[start_byte:end_byte]

    def set_loop_points(self, start_sample: int, end_sample: int) -> None:
        """
        Set loop points for the clip.

        Args:
            start_sample: Loop start sample
            end_sample: Loop end sample
        """
        self.metadata.loop_start = max(0, start_sample)
        self.metadata.loop_end = min(end_sample, self.metadata.total_samples)
        self.metadata.has_loop_points = True

    def __hash__(self) -> int:
        """Hash based on clip ID."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on clip ID."""
        if not isinstance(other, AudioClip):
            return False
        return self.id == other.id


class AudioClipManager:
    """
    Manages audio clip loading and caching.
    """

    def __init__(self) -> None:
        """Initialize the clip manager."""
        self._clips: dict[str, AudioClip] = {}
        self._path_to_id: dict[Path, str] = {}
        self._lock = threading.RLock()

    def load_clip(
        self,
        path: Path,
        name: Optional[str] = None,
        category: AudioCategory = AudioCategory.SFX,
        pool_type: MemoryPoolType = MemoryPoolType.RESIDENT,
    ) -> Optional[AudioClip]:
        """
        Load an audio clip from file.

        Args:
            path: Path to audio file
            name: Optional name (defaults to filename)
            category: Audio category
            pool_type: Memory pool type

        Returns:
            Loaded AudioClip or None on failure
        """
        path = Path(path).resolve()

        with self._lock:
            # Check if already loaded
            if path in self._path_to_id:
                clip = self._clips.get(self._path_to_id[path])
                if clip:
                    clip.add_ref()
                    return clip

            # Create new clip
            clip_name = name or path.stem
            clip = AudioClip(
                id="",
                name=clip_name,
                path=path,
                category=category,
                pool_type=pool_type,
            )

            if not clip.load_from_file(path):
                return None

            self._clips[clip.id] = clip
            self._path_to_id[path] = clip.id
            clip.add_ref()

            return clip

    def create_clip(
        self,
        name: str,
        data: bytes,
        format: AudioFormat,
        sample_rate: int,
        channels: int,
        category: AudioCategory = AudioCategory.SFX,
    ) -> Optional[AudioClip]:
        """
        Create a clip from raw audio data.

        Args:
            name: Clip name
            data: Raw audio data
            format: Audio format
            sample_rate: Sample rate
            channels: Number of channels
            category: Audio category

        Returns:
            Created AudioClip or None on failure
        """
        with self._lock:
            clip = AudioClip(
                id="",
                name=name,
                category=category,
                pool_type=MemoryPoolType.TEMPORARY,
            )

            if not clip.load_from_memory(data, format, sample_rate, channels):
                return None

            self._clips[clip.id] = clip
            clip.add_ref()

            return clip

    def get_clip(self, clip_id: str) -> Optional[AudioClip]:
        """Get a clip by ID."""
        with self._lock:
            return self._clips.get(clip_id)

    def release_clip(self, clip: AudioClip) -> None:
        """
        Release a reference to a clip.

        Args:
            clip: The clip to release
        """
        with self._lock:
            ref_count = clip.release_ref()

            # If no more references and temporary, unload
            if ref_count == 0 and clip.pool_type == MemoryPoolType.TEMPORARY:
                self._unload_clip(clip)

    def _unload_clip(self, clip: AudioClip) -> None:
        """Internal method to unload a clip."""
        clip.unload()
        if clip.id in self._clips:
            del self._clips[clip.id]
        if clip.path and clip.path in self._path_to_id:
            del self._path_to_id[clip.path]

    def unload_all(self) -> None:
        """Unload all clips."""
        with self._lock:
            for clip in list(self._clips.values()):
                clip.unload()
            self._clips.clear()
            self._path_to_id.clear()

    def get_loaded_clips(self) -> list[AudioClip]:
        """Get all loaded clips."""
        with self._lock:
            return [c for c in self._clips.values() if c.is_loaded]

    def get_memory_usage(self) -> int:
        """Get total memory usage of all clips."""
        with self._lock:
            return sum(c.memory_size for c in self._clips.values())

    def get_clip_count(self) -> int:
        """Get number of loaded clips."""
        with self._lock:
            return len(self._clips)
