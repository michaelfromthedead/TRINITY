"""
Replay Export - Export replay to video and GIF formats.

Provides functionality to export replays as video files or animated
GIFs for sharing and analysis.
"""

from __future__ import annotations

import io
import struct
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, BinaryIO
import time


class ExportFormat(Enum):
    """Supported export formats."""
    MP4 = auto()
    WEBM = auto()
    AVI = auto()
    GIF = auto()
    PNG_SEQUENCE = auto()
    JPEG_SEQUENCE = auto()


class VideoCodec(Enum):
    """Supported video codecs."""
    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"
    VP8 = "libvpx"
    MJPEG = "mjpeg"
    PRORES = "prores"


@dataclass
class GifConfig:
    """Configuration for GIF export."""
    fps: int = 15
    width: int = 480
    height: int = 270
    colors: int = 256
    dither: bool = True
    loop: int = 0  # 0 = infinite loop
    optimize: bool = True
    quality: int = 85


@dataclass
class ExportConfig:
    """Configuration for replay export."""
    # Output settings
    format: ExportFormat = ExportFormat.MP4
    output_path: Optional[Path] = None

    # Video settings
    codec: VideoCodec = VideoCodec.H264
    fps: int = 30
    width: int = 1920
    height: int = 1080
    bitrate: str = "8M"
    crf: int = 23  # Constant Rate Factor (0-51, lower = better quality)

    # Audio settings
    include_audio: bool = True
    audio_bitrate: str = "192k"

    # Range settings
    start_frame: int = 0
    end_frame: int = -1  # -1 = end of replay

    # GIF settings
    gif_config: GifConfig = field(default_factory=GifConfig)

    # Progress callback
    on_progress: Optional[Callable[[float], None]] = None

    # Frame capture callback (required for export)
    capture_frame: Optional[Callable[[int], bytes]] = None

    # Overlay settings
    show_timestamp: bool = False
    show_frame_number: bool = False
    timestamp_format: str = "%M:%S.%f"

    # Post-processing
    add_watermark: bool = False
    watermark_text: Optional[str] = None
    watermark_position: str = "bottom-right"


@dataclass
class ExportProgress:
    """Progress information for export."""
    current_frame: int = 0
    total_frames: int = 0
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    bytes_written: int = 0
    is_complete: bool = False
    error: Optional[str] = None

    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total_frames <= 0:
            return 0.0
        return (self.current_frame / self.total_frames) * 100.0

    @property
    def fps(self) -> float:
        """Get current export FPS."""
        if self.elapsed_time <= 0:
            return 0.0
        return self.current_frame / self.elapsed_time


class ReplayExporter:
    """Export replays to video and GIF formats.

    Provides functionality to render and export replays as video
    files or animated GIFs.
    """
    __slots__ = (
        '_config', '_progress', '_is_exporting', '_temp_dir',
        '_frames_buffer', '_start_time'
    )

    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize exporter.

        Args:
            config: Export configuration
        """
        self._config = config or ExportConfig()
        self._progress = ExportProgress()
        self._is_exporting = False
        self._temp_dir: Optional[Path] = None
        self._frames_buffer: list[bytes] = []
        self._start_time = 0.0

    @property
    def config(self) -> ExportConfig:
        """Get export configuration."""
        return self._config

    @config.setter
    def config(self, value: ExportConfig) -> None:
        """Set export configuration."""
        self._config = value

    @property
    def progress(self) -> ExportProgress:
        """Get current export progress."""
        return self._progress

    @property
    def is_exporting(self) -> bool:
        """Check if export is in progress."""
        return self._is_exporting

    def export(
        self,
        total_frames: int,
        output_path: Optional[str | Path] = None
    ) -> ExportProgress:
        """Export replay to file.

        Args:
            total_frames: Total number of frames to export
            output_path: Output file path (overrides config)

        Returns:
            Final export progress
        """
        if self._is_exporting:
            raise RuntimeError("Export already in progress")

        if not self._config.capture_frame:
            raise ValueError("capture_frame callback is required")

        output = Path(output_path) if output_path else self._config.output_path
        if not output:
            raise ValueError("Output path is required")

        self._is_exporting = True
        self._progress = ExportProgress(total_frames=total_frames)
        self._start_time = time.perf_counter()

        try:
            if self._config.format == ExportFormat.GIF:
                self._export_gif(total_frames, output)
            elif self._config.format == ExportFormat.PNG_SEQUENCE:
                self._export_image_sequence(total_frames, output, 'png')
            elif self._config.format == ExportFormat.JPEG_SEQUENCE:
                self._export_image_sequence(total_frames, output, 'jpg')
            else:
                self._export_video(total_frames, output)

            self._progress.is_complete = True

        except Exception as e:
            self._progress.error = str(e)
            raise

        finally:
            self._is_exporting = False
            self._progress.elapsed_time = time.perf_counter() - self._start_time

        return self._progress

    def cancel(self) -> None:
        """Cancel ongoing export."""
        self._is_exporting = False

    def get_estimated_size(self, total_frames: int) -> int:
        """Estimate output file size.

        Args:
            total_frames: Number of frames

        Returns:
            Estimated size in bytes
        """
        duration = total_frames / self._config.fps

        if self._config.format == ExportFormat.GIF:
            # Rough estimate: 50KB per second at default settings
            bytes_per_second = 50 * 1024
            return int(duration * bytes_per_second)

        # Video estimate based on bitrate
        bitrate = self._parse_bitrate(self._config.bitrate)
        return int(duration * bitrate / 8)

    def _export_video(self, total_frames: int, output: Path) -> None:
        """Export to video format using frame buffer.

        Note: In a real implementation, this would use ffmpeg or similar.
        This is a simplified version that creates a raw frame dump.
        """
        start_frame = self._config.start_frame
        end_frame = self._config.end_frame if self._config.end_frame >= 0 else total_frames

        # Create output directory
        output.parent.mkdir(parents=True, exist_ok=True)

        # Collect frames
        self._frames_buffer.clear()

        for frame in range(start_frame, end_frame):
            if not self._is_exporting:
                break

            # Capture frame
            frame_data = self._config.capture_frame(frame)
            self._frames_buffer.append(frame_data)

            # Update progress
            self._update_progress(frame - start_frame + 1, end_frame - start_frame)

        # Write video container (simplified)
        self._write_video_container(output)

    def _export_gif(self, total_frames: int, output: Path) -> None:
        """Export to GIF format."""
        gif_config = self._config.gif_config
        start_frame = self._config.start_frame
        end_frame = self._config.end_frame if self._config.end_frame >= 0 else total_frames

        # Frame step based on FPS difference
        source_fps = self._config.fps
        target_fps = gif_config.fps
        frame_step = max(1, source_fps // target_fps)

        # Create output directory
        output.parent.mkdir(parents=True, exist_ok=True)

        # Collect frames
        self._frames_buffer.clear()

        frame_count = 0
        for frame in range(start_frame, end_frame, frame_step):
            if not self._is_exporting:
                break

            # Capture frame
            frame_data = self._config.capture_frame(frame)
            self._frames_buffer.append(frame_data)
            frame_count += 1

            # Update progress
            progress_pct = (frame - start_frame) / (end_frame - start_frame)
            self._update_progress(int(progress_pct * total_frames), total_frames)

        # Write GIF
        self._write_gif(output, gif_config)

    def _export_image_sequence(
        self,
        total_frames: int,
        output: Path,
        image_format: str
    ) -> None:
        """Export as image sequence."""
        start_frame = self._config.start_frame
        end_frame = self._config.end_frame if self._config.end_frame >= 0 else total_frames

        # Create output directory
        output.mkdir(parents=True, exist_ok=True)

        for frame in range(start_frame, end_frame):
            if not self._is_exporting:
                break

            # Capture frame
            frame_data = self._config.capture_frame(frame)

            # Write frame file
            frame_path = output / f"frame_{frame:06d}.{image_format}"
            with open(frame_path, 'wb') as f:
                f.write(frame_data)

            self._progress.bytes_written += len(frame_data)

            # Update progress
            self._update_progress(frame - start_frame + 1, end_frame - start_frame)

    def _write_video_container(self, output: Path) -> None:
        """Write frames to video container.

        Note: This is a simplified implementation. A real implementation
        would use ffmpeg or a video encoding library.
        """
        # Write a simple raw video container
        # Format: [frame_count:4][width:4][height:4][fps:4][frames...]
        with open(output, 'wb') as f:
            # Header
            f.write(struct.pack('<I', len(self._frames_buffer)))
            f.write(struct.pack('<I', self._config.width))
            f.write(struct.pack('<I', self._config.height))
            f.write(struct.pack('<I', self._config.fps))

            # Frames
            for frame_data in self._frames_buffer:
                f.write(struct.pack('<I', len(frame_data)))
                f.write(frame_data)
                self._progress.bytes_written += len(frame_data) + 4

    def _write_gif(self, output: Path, config: GifConfig) -> None:
        """Write frames to GIF file.

        Note: This is a simplified implementation. A real implementation
        would use PIL/Pillow or a GIF encoding library.
        """
        # Simplified GIF header
        with open(output, 'wb') as f:
            # GIF89a header
            f.write(b'GIF89a')

            # Logical screen descriptor
            f.write(struct.pack('<H', config.width))
            f.write(struct.pack('<H', config.height))
            f.write(bytes([0xF7, 0x00, 0x00]))  # Global color table, bg, aspect

            # Global color table (256 colors)
            for i in range(256):
                f.write(bytes([i, i, i]))  # Grayscale palette

            # Application extension for looping
            if config.loop >= 0:
                f.write(bytes([0x21, 0xFF, 0x0B]))
                f.write(b'NETSCAPE2.0')
                f.write(bytes([0x03, 0x01]))
                f.write(struct.pack('<H', config.loop))
                f.write(bytes([0x00]))

            # Frame delay (in centiseconds)
            delay = 100 // config.fps

            # Write each frame
            for frame_data in self._frames_buffer:
                # Graphic control extension
                f.write(bytes([0x21, 0xF9, 0x04, 0x00]))
                f.write(struct.pack('<H', delay))
                f.write(bytes([0x00, 0x00]))

                # Image descriptor
                f.write(bytes([0x2C]))
                f.write(struct.pack('<H', 0))  # Left
                f.write(struct.pack('<H', 0))  # Top
                f.write(struct.pack('<H', config.width))
                f.write(struct.pack('<H', config.height))
                f.write(bytes([0x00]))  # No local color table

                # Image data (simplified - just write length prefix)
                f.write(bytes([0x08]))  # LZW minimum code size
                # In a real implementation, we'd LZW compress the image data
                chunk_size = min(255, len(frame_data))
                f.write(bytes([chunk_size]))
                f.write(frame_data[:chunk_size])
                f.write(bytes([0x00]))  # Block terminator

                self._progress.bytes_written += len(frame_data)

            # GIF trailer
            f.write(bytes([0x3B]))

    def _update_progress(self, current: int, total: int) -> None:
        """Update export progress."""
        self._progress.current_frame = current
        self._progress.total_frames = total
        self._progress.elapsed_time = time.perf_counter() - self._start_time

        if current > 0:
            time_per_frame = self._progress.elapsed_time / current
            remaining_frames = total - current
            self._progress.estimated_remaining = time_per_frame * remaining_frames

        # Notify callback
        if self._config.on_progress:
            self._config.on_progress(self._progress.percentage / 100.0)

    def _parse_bitrate(self, bitrate: str) -> int:
        """Parse bitrate string to bits per second."""
        bitrate = bitrate.upper().strip()

        if bitrate.endswith('K'):
            return int(float(bitrate[:-1]) * 1000)
        elif bitrate.endswith('M'):
            return int(float(bitrate[:-1]) * 1000000)
        elif bitrate.endswith('G'):
            return int(float(bitrate[:-1]) * 1000000000)

        return int(bitrate)


class FrameCapture:
    """Utility class for capturing frames during export."""
    __slots__ = ('_width', '_height', '_format', '_buffer')

    def __init__(self, width: int, height: int, format: str = 'rgba'):
        """Initialize frame capture.

        Args:
            width: Frame width
            height: Frame height
            format: Pixel format
        """
        self._width = width
        self._height = height
        self._format = format
        self._buffer = bytearray(width * height * 4)  # RGBA

    @property
    def width(self) -> int:
        """Get frame width."""
        return self._width

    @property
    def height(self) -> int:
        """Get frame height."""
        return self._height

    def capture(self, render_callback: Callable[[int], None], frame: int) -> bytes:
        """Capture a frame.

        Args:
            render_callback: Function to render the frame
            frame: Frame number

        Returns:
            Frame data as bytes
        """
        # Call render to update frame buffer
        render_callback(frame)

        # Return copy of buffer
        return bytes(self._buffer)

    def set_pixel(self, x: int, y: int, r: int, g: int, b: int, a: int = 255) -> None:
        """Set a pixel value.

        Args:
            x: X coordinate
            y: Y coordinate
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            a: Alpha component (0-255)
        """
        if 0 <= x < self._width and 0 <= y < self._height:
            offset = (y * self._width + x) * 4
            self._buffer[offset:offset + 4] = bytes([r, g, b, a])

    def fill(self, r: int, g: int, b: int, a: int = 255) -> None:
        """Fill entire frame with color.

        Args:
            r: Red component
            g: Green component
            b: Blue component
            a: Alpha component
        """
        pixel = bytes([r, g, b, a])
        for i in range(0, len(self._buffer), 4):
            self._buffer[i:i + 4] = pixel

    def get_buffer(self) -> bytes:
        """Get raw buffer data."""
        return bytes(self._buffer)
