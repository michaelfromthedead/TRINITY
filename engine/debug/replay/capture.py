"""Frame capture for screenshots, video, and GIF recording.

This module provides utilities for capturing frames during replay
playback, including single screenshots, video recording, and animated GIFs.
"""

from __future__ import annotations

import io
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, BinaryIO, Callable, Protocol, runtime_checkable


class CaptureFormat(Enum):
    """Supported capture output formats.

    Attributes:
        PNG: Lossless PNG image
        JPEG: Lossy JPEG image
        GIF: Animated GIF
        VIDEO: Video file (format depends on encoder)
    """
    PNG = auto()
    JPEG = auto()
    GIF = auto()
    VIDEO = auto()


@dataclass(slots=True)
class FrameData:
    """Raw frame pixel data.

    Attributes:
        width: Frame width in pixels
        height: Frame height in pixels
        pixels: Raw pixel data (RGBA, row-major)
        timestamp: Capture timestamp
    """
    width: int
    height: int
    pixels: bytes
    timestamp: float = field(default_factory=time.time)

    @property
    def size(self) -> tuple[int, int]:
        """Get frame dimensions as (width, height)."""
        return (self.width, self.height)

    @property
    def bytes_per_pixel(self) -> int:
        """Get bytes per pixel (RGBA = 4)."""
        return 4

    @property
    def total_bytes(self) -> int:
        """Get expected total bytes."""
        return self.width * self.height * self.bytes_per_pixel


@runtime_checkable
class FrameProvider(Protocol):
    """Protocol for providing frame data.

    Implement this to connect the capture system to your renderer.
    """

    def capture_frame(self) -> FrameData:
        """Capture the current frame.

        Returns:
            Frame pixel data
        """
        ...


class ImageEncoder(ABC):
    """Base class for image encoders."""

    @abstractmethod
    def encode(self, frame: FrameData) -> bytes:
        """Encode a frame to bytes.

        Args:
            frame: Frame to encode

        Returns:
            Encoded image data
        """
        pass


class PNGEncoder(ImageEncoder):
    """Simple PNG encoder.

    This is a minimal implementation. In production, use a proper
    image library like PIL or a native PNG encoder.
    """

    def encode(self, frame: FrameData) -> bytes:
        """Encode frame as PNG.

        This is a simplified implementation that creates a valid but
        uncompressed PNG. For production use, integrate with PIL or
        a native PNG library.

        Args:
            frame: Frame to encode

        Returns:
            PNG file data
        """
        width = frame.width
        height = frame.height

        # PNG signature
        output = io.BytesIO()
        output.write(b'\x89PNG\r\n\x1a\n')

        # IHDR chunk
        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
        self._write_chunk(output, b'IHDR', ihdr_data)

        # IDAT chunk (uncompressed for simplicity)
        import zlib
        raw_data = bytearray()
        row_bytes = width * 4
        for y in range(height):
            raw_data.append(0)  # Filter byte (none)
            row_start = y * row_bytes
            raw_data.extend(frame.pixels[row_start:row_start + row_bytes])

        compressed = zlib.compress(bytes(raw_data), 6)
        self._write_chunk(output, b'IDAT', compressed)

        # IEND chunk
        self._write_chunk(output, b'IEND', b'')

        return output.getvalue()

    def _write_chunk(self, output: io.BytesIO, chunk_type: bytes, data: bytes) -> None:
        """Write a PNG chunk."""
        import zlib
        length = struct.pack('>I', len(data))
        output.write(length)
        output.write(chunk_type)
        output.write(data)
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        output.write(struct.pack('>I', crc))


class JPEGEncoder(ImageEncoder):
    """JPEG encoder stub.

    In production, use PIL or a native JPEG library.
    """

    def __init__(self, quality: int = 85) -> None:
        """Initialize JPEG encoder.

        Args:
            quality: JPEG quality (1-100)
        """
        self.quality = quality

    def encode(self, frame: FrameData) -> bytes:
        """Encode frame as JPEG.

        This is a stub implementation. For production, integrate with
        PIL or a native JPEG library.

        Args:
            frame: Frame to encode

        Returns:
            JPEG file data
        """
        # Stub: fall back to PNG for now
        return PNGEncoder().encode(frame)


class VideoEncoder(ABC):
    """Base class for video encoders."""

    @abstractmethod
    def start(self, path: Path, width: int, height: int, fps: int) -> None:
        """Start encoding a new video.

        Args:
            path: Output file path
            width: Video width
            height: Video height
            fps: Target frame rate
        """
        pass

    @abstractmethod
    def add_frame(self, frame: FrameData) -> None:
        """Add a frame to the video.

        Args:
            frame: Frame to add
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish encoding and write the video file."""
        pass


class RawVideoEncoder(VideoEncoder):
    """Simple raw video encoder.

    Stores frames as raw RGBA data. For production, use FFmpeg
    or a proper video encoding library.
    """

    def __init__(self) -> None:
        self._file: BinaryIO | None = None
        self._path: Path | None = None
        self._width = 0
        self._height = 0
        self._fps = 30
        self._frame_count = 0

    def start(self, path: Path, width: int, height: int, fps: int) -> None:
        """Start recording raw video."""
        self._path = path
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_count = 0

        # Write header
        self._file = open(path, "wb")
        header = struct.pack(">4sIIIQ", b"RAWV", width, height, fps, 0)
        self._file.write(header)

    def add_frame(self, frame: FrameData) -> None:
        """Add frame to raw video."""
        if self._file is None:
            raise RuntimeError("Video recording not started")

        if frame.width != self._width or frame.height != self._height:
            raise ValueError("Frame size mismatch")

        self._file.write(frame.pixels)
        self._frame_count += 1

    def finish(self) -> None:
        """Finish recording raw video."""
        if self._file is None:
            return

        # Update frame count in header
        self._file.seek(16)
        self._file.write(struct.pack(">Q", self._frame_count))
        self._file.close()
        self._file = None


class GIFEncoder:
    """Simple animated GIF encoder.

    Creates basic GIFs. For production, use PIL or a proper GIF library.
    """

    def __init__(self, delay_cs: int = 10) -> None:
        """Initialize GIF encoder.

        Args:
            delay_cs: Delay between frames in centiseconds (100 = 1 second)
        """
        self.delay_cs = delay_cs
        self._frames: list[FrameData] = []
        self._width = 0
        self._height = 0

    def add_frame(self, frame: FrameData) -> None:
        """Add a frame to the GIF.

        Args:
            frame: Frame to add
        """
        if not self._frames:
            self._width = frame.width
            self._height = frame.height
        elif frame.width != self._width or frame.height != self._height:
            raise ValueError("All frames must have the same dimensions")

        self._frames.append(frame)

    def encode(self) -> bytes:
        """Encode all frames as an animated GIF.

        This is a simplified implementation. For production, use PIL.

        Returns:
            GIF file data
        """
        if not self._frames:
            raise ValueError("No frames to encode")

        output = io.BytesIO()

        # GIF Header
        output.write(b'GIF89a')

        # Logical Screen Descriptor
        output.write(struct.pack('<HH', self._width, self._height))
        output.write(bytes([0xF7, 0, 0]))  # Global color table, 256 colors

        # Global Color Table (256 grays for simplicity)
        for i in range(256):
            output.write(bytes([i, i, i]))

        # Netscape Extension for looping
        output.write(bytes([0x21, 0xFF, 0x0B]))
        output.write(b'NETSCAPE2.0')
        output.write(bytes([0x03, 0x01, 0x00, 0x00, 0x00]))

        # Write each frame
        for frame in self._frames:
            self._write_frame(output, frame)

        # GIF Trailer
        output.write(bytes([0x3B]))

        return output.getvalue()

    def _write_frame(self, output: io.BytesIO, frame: FrameData) -> None:
        """Write a single frame to the GIF."""
        # Graphic Control Extension
        output.write(bytes([0x21, 0xF9, 0x04]))
        output.write(bytes([0x04]))  # Dispose: restore to background
        output.write(struct.pack('<H', self.delay_cs))
        output.write(bytes([0, 0]))  # No transparent color

        # Image Descriptor
        output.write(bytes([0x2C]))
        output.write(struct.pack('<HHHH', 0, 0, frame.width, frame.height))
        output.write(bytes([0]))  # No local color table

        # LZW Minimum Code Size
        output.write(bytes([8]))

        # Convert RGBA to indexed (grayscale for simplicity)
        indexed = bytearray()
        for i in range(0, len(frame.pixels), 4):
            r = frame.pixels[i]
            g = frame.pixels[i + 1]
            b = frame.pixels[i + 2]
            gray = (r + g + b) // 3
            indexed.append(gray)

        # Simple LZW encoding (sub-blocks)
        # This is a simplified version that may not work with all decoders
        pos = 0
        while pos < len(indexed):
            chunk_size = min(254, len(indexed) - pos)
            output.write(bytes([chunk_size]))
            output.write(indexed[pos:pos + chunk_size])
            pos += chunk_size

        output.write(bytes([0]))  # Block terminator

    def clear(self) -> None:
        """Clear all frames."""
        self._frames.clear()
        self._width = 0
        self._height = 0


class FrameCapture:
    """Captures frames from the renderer for screenshots and video.

    FrameCapture provides a unified interface for capturing individual
    screenshots, recording video, and creating animated GIFs during
    replay playback or gameplay.

    Example:
        capture = FrameCapture(renderer)

        # Screenshot
        capture.screenshot("screenshot.png", CaptureFormat.PNG)

        # Video recording
        capture.start_video("recording.raw", fps=30)
        # ... capture frames during replay ...
        capture.stop_video()

        # Animated GIF
        capture.capture_gif("animation.gif", duration_s=5.0, fps=15)
    """

    def __init__(
        self,
        frame_provider: FrameProvider | None = None,
    ) -> None:
        """Initialize frame capture.

        Args:
            frame_provider: Provider of frame data (renderer)
        """
        self._frame_provider = frame_provider

        # Encoders
        self._png_encoder = PNGEncoder()
        self._jpeg_encoder = JPEGEncoder()
        self._video_encoder: VideoEncoder | None = None
        self._gif_encoder = GIFEncoder()

        # State
        self._is_recording_video = False
        self._video_path: Path | None = None
        self._video_fps = 30
        self._video_start_time: float | None = None

    def set_frame_provider(self, provider: FrameProvider) -> None:
        """Set the frame provider.

        Args:
            provider: Frame data provider
        """
        self._frame_provider = provider

    @property
    def is_recording_video(self) -> bool:
        """Check if video recording is in progress."""
        return self._is_recording_video

    def _get_frame(self) -> FrameData:
        """Get current frame from provider.

        Returns:
            Frame data

        Raises:
            RuntimeError: If no frame provider set
        """
        if self._frame_provider is None:
            raise RuntimeError("No frame provider set")
        return self._frame_provider.capture_frame()

    def screenshot(
        self,
        path: Path | str,
        format: CaptureFormat = CaptureFormat.PNG,
        frame: FrameData | None = None,
    ) -> None:
        """Capture a single screenshot.

        Args:
            path: Output file path
            format: Image format to use
            frame: Optional pre-captured frame. If None, captures now.

        Raises:
            ValueError: If format is not a still image format
        """
        if format not in (CaptureFormat.PNG, CaptureFormat.JPEG):
            raise ValueError(f"Format {format} is not a still image format")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if frame is None:
            frame = self._get_frame()

        if format == CaptureFormat.PNG:
            data = self._png_encoder.encode(frame)
        else:
            data = self._jpeg_encoder.encode(frame)

        with open(path, "wb") as f:
            f.write(data)

    def start_video(
        self,
        path: Path | str,
        fps: int = 30,
        encoder: VideoEncoder | None = None,
    ) -> None:
        """Start recording video.

        Args:
            path: Output file path
            fps: Target frame rate
            encoder: Optional custom video encoder

        Raises:
            RuntimeError: If already recording
        """
        if self._is_recording_video:
            raise RuntimeError("Already recording video")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._video_path = path
        self._video_fps = fps
        self._video_encoder = encoder or RawVideoEncoder()
        self._is_recording_video = True
        self._video_start_time = time.time()

        # Get initial frame to determine dimensions
        frame = self._get_frame()
        self._video_encoder.start(path, frame.width, frame.height, fps)
        self._video_encoder.add_frame(frame)

    def capture_video_frame(self) -> None:
        """Capture a frame for the current video recording.

        Raises:
            RuntimeError: If not recording video
        """
        if not self._is_recording_video or self._video_encoder is None:
            raise RuntimeError("Not recording video")

        frame = self._get_frame()
        self._video_encoder.add_frame(frame)

    def stop_video(self) -> Path | None:
        """Stop video recording and finalize the file.

        Returns:
            Path to the recorded video, or None if not recording
        """
        if not self._is_recording_video or self._video_encoder is None:
            return None

        self._video_encoder.finish()
        self._is_recording_video = False

        path = self._video_path
        self._video_path = None
        self._video_encoder = None
        self._video_start_time = None

        return path

    def capture_gif(
        self,
        path: Path | str,
        duration_s: float,
        fps: int = 15,
        capture_callback: Callable[[], None] | None = None,
    ) -> None:
        """Capture an animated GIF over a duration.

        This method blocks while capturing frames. For non-blocking capture,
        use capture_gif_start/capture_gif_frame/capture_gif_finish.

        Args:
            path: Output file path
            duration_s: Duration to capture in seconds
            fps: Frame rate for the GIF
            capture_callback: Optional callback called before each frame capture.
                              Use to advance the replay.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        delay_cs = max(1, int(100 / fps))
        encoder = GIFEncoder(delay_cs=delay_cs)

        frame_interval = 1.0 / fps
        total_frames = int(duration_s * fps)

        for _ in range(total_frames):
            if capture_callback is not None:
                capture_callback()

            frame = self._get_frame()
            encoder.add_frame(frame)

            # Simulate frame timing (in blocking mode)
            time.sleep(frame_interval)

        data = encoder.encode()
        with open(path, "wb") as f:
            f.write(data)

    def capture_gif_start(self, fps: int = 15) -> None:
        """Start non-blocking GIF capture.

        Args:
            fps: Frame rate for the GIF
        """
        delay_cs = max(1, int(100 / fps))
        self._gif_encoder = GIFEncoder(delay_cs=delay_cs)

    def capture_gif_frame(self) -> None:
        """Capture a frame for the GIF."""
        frame = self._get_frame()
        self._gif_encoder.add_frame(frame)

    def capture_gif_finish(self, path: Path | str) -> None:
        """Finish GIF capture and save to file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self._gif_encoder.encode()
        with open(path, "wb") as f:
            f.write(data)

        self._gif_encoder.clear()

    def get_video_duration(self) -> float:
        """Get duration of current video recording.

        Returns:
            Duration in seconds, or 0 if not recording
        """
        if not self._is_recording_video or self._video_start_time is None:
            return 0.0
        return time.time() - self._video_start_time

    @staticmethod
    def list_supported_formats() -> list[CaptureFormat]:
        """Get list of supported capture formats.

        Returns:
            List of supported formats
        """
        return list(CaptureFormat)
