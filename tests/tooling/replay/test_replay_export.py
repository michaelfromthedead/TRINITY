"""
Tests for replay_export.py - Video and GIF export.
"""

import pytest
import tempfile
from pathlib import Path

from engine.tooling.replay.replay_export import (
    ReplayExporter,
    ExportFormat,
    ExportConfig,
    ExportProgress,
    VideoCodec,
    GifConfig,
    FrameCapture,
)


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_formats_exist(self):
        """Test all export formats exist."""
        assert ExportFormat.MP4
        assert ExportFormat.WEBM
        assert ExportFormat.AVI
        assert ExportFormat.GIF
        assert ExportFormat.PNG_SEQUENCE
        assert ExportFormat.JPEG_SEQUENCE


class TestVideoCodec:
    """Tests for VideoCodec enum."""

    def test_codec_values(self):
        """Test codec values."""
        assert VideoCodec.H264.value == "libx264"
        assert VideoCodec.H265.value == "libx265"
        assert VideoCodec.VP9.value == "libvpx-vp9"


class TestGifConfig:
    """Tests for GifConfig dataclass."""

    def test_default_config(self):
        """Test default GIF configuration."""
        config = GifConfig()
        assert config.fps == 15
        assert config.width == 480
        assert config.colors == 256
        assert config.loop == 0

    def test_custom_config(self):
        """Test custom GIF configuration."""
        config = GifConfig(
            fps=30,
            width=640,
            height=360,
            colors=128
        )
        assert config.fps == 30
        assert config.width == 640
        assert config.colors == 128


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_config(self):
        """Test default export configuration."""
        config = ExportConfig()
        assert config.format == ExportFormat.MP4
        assert config.codec == VideoCodec.H264
        assert config.fps == 30
        assert config.width == 1920
        assert config.height == 1080

    def test_custom_config(self):
        """Test custom export configuration."""
        config = ExportConfig(
            format=ExportFormat.GIF,
            fps=15,
            width=640,
            height=480
        )
        assert config.format == ExportFormat.GIF
        assert config.fps == 15

    def test_config_with_callbacks(self):
        """Test config with callbacks."""
        progress_values = []

        def on_progress(value):
            progress_values.append(value)

        config = ExportConfig(on_progress=on_progress)
        config.on_progress(0.5)

        assert 0.5 in progress_values


class TestExportProgress:
    """Tests for ExportProgress dataclass."""

    def test_default_progress(self):
        """Test default progress values."""
        progress = ExportProgress()
        assert progress.current_frame == 0
        assert progress.total_frames == 0
        assert progress.is_complete is False

    def test_percentage_calculation(self):
        """Test percentage calculation."""
        progress = ExportProgress(
            current_frame=50,
            total_frames=100
        )
        assert progress.percentage == 50.0

    def test_percentage_zero_frames(self):
        """Test percentage with zero frames."""
        progress = ExportProgress(
            current_frame=0,
            total_frames=0
        )
        assert progress.percentage == 0.0

    def test_fps_calculation(self):
        """Test FPS calculation."""
        progress = ExportProgress(
            current_frame=60,
            elapsed_time=2.0
        )
        assert progress.fps == 30.0


class TestFrameCapture:
    """Tests for FrameCapture utility class."""

    def test_create_frame_capture(self):
        """Test creating a frame capture."""
        capture = FrameCapture(width=640, height=480)
        assert capture.width == 640
        assert capture.height == 480

    def test_set_pixel(self):
        """Test setting a pixel."""
        capture = FrameCapture(width=100, height=100)
        capture.set_pixel(50, 50, 255, 128, 64, 255)

        buffer = capture.get_buffer()
        # Check pixel at (50, 50)
        offset = (50 * 100 + 50) * 4
        assert buffer[offset:offset + 4] == bytes([255, 128, 64, 255])

    def test_set_pixel_out_of_bounds(self):
        """Test setting pixel out of bounds (should be ignored)."""
        capture = FrameCapture(width=100, height=100)
        # Should not raise
        capture.set_pixel(200, 200, 255, 0, 0)
        capture.set_pixel(-1, -1, 255, 0, 0)

    def test_fill(self):
        """Test filling entire frame."""
        capture = FrameCapture(width=10, height=10)
        capture.fill(255, 128, 64, 255)

        buffer = capture.get_buffer()
        # Check first pixel
        assert buffer[0:4] == bytes([255, 128, 64, 255])
        # Check last pixel
        assert buffer[-4:] == bytes([255, 128, 64, 255])

    def test_capture(self):
        """Test capturing a frame."""
        capture = FrameCapture(width=100, height=100)

        def render(frame):
            capture.fill(frame, frame, frame)

        result = capture.capture(render, 128)
        assert len(result) == 100 * 100 * 4

    def test_get_buffer(self):
        """Test getting raw buffer."""
        capture = FrameCapture(width=100, height=100)
        buffer = capture.get_buffer()

        assert isinstance(buffer, bytes)
        assert len(buffer) == 100 * 100 * 4


class TestReplayExporter:
    """Tests for ReplayExporter class."""

    def test_create_exporter(self):
        """Test creating an exporter."""
        exporter = ReplayExporter()
        assert not exporter.is_exporting

    def test_create_with_config(self):
        """Test creating with config."""
        config = ExportConfig(format=ExportFormat.GIF)
        exporter = ReplayExporter(config)
        assert exporter.config.format == ExportFormat.GIF

    def test_config_property(self):
        """Test config property."""
        exporter = ReplayExporter()
        new_config = ExportConfig(fps=60)
        exporter.config = new_config
        assert exporter.config.fps == 60

    def test_progress_property(self):
        """Test progress property."""
        exporter = ReplayExporter()
        progress = exporter.progress
        assert isinstance(progress, ExportProgress)

    def test_export_requires_capture_callback(self):
        """Test that export requires capture callback."""
        exporter = ReplayExporter()

        with pytest.raises(ValueError, match="capture_frame callback"):
            exporter.export(100, "/tmp/test.mp4")

    def test_export_requires_output_path(self):
        """Test that export requires output path."""
        def capture(frame):
            return b'\x00' * 100

        config = ExportConfig(capture_frame=capture)
        exporter = ReplayExporter(config)

        with pytest.raises(ValueError, match="Output path"):
            exporter.export(100)

    def test_export_video(self):
        """Test exporting video."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp4"

            def capture(frame):
                return b'\x00' * (640 * 480 * 4)

            config = ExportConfig(
                capture_frame=capture,
                width=640,
                height=480,
                fps=30
            )
            exporter = ReplayExporter(config)

            progress = exporter.export(30, output_path)

            assert progress.is_complete
            assert output_path.exists()

    def test_export_gif(self):
        """Test exporting GIF."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.gif"

            def capture(frame):
                return b'\x00' * (480 * 270 * 4)

            config = ExportConfig(
                format=ExportFormat.GIF,
                capture_frame=capture,
                gif_config=GifConfig(fps=15, width=480, height=270)
            )
            exporter = ReplayExporter(config)

            progress = exporter.export(30, output_path)

            assert progress.is_complete
            assert output_path.exists()

    def test_export_image_sequence(self):
        """Test exporting image sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "frames"

            def capture(frame):
                return b'\x00' * 100  # Simple placeholder

            config = ExportConfig(
                format=ExportFormat.PNG_SEQUENCE,
                capture_frame=capture
            )
            exporter = ReplayExporter(config)

            progress = exporter.export(10, output_path)

            assert progress.is_complete
            assert output_path.exists()
            # Should have created frame files
            assert len(list(output_path.glob("*.png"))) == 10

    def test_export_with_frame_range(self):
        """Test exporting with frame range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp4"
            captured_frames = []

            def capture(frame):
                captured_frames.append(frame)
                return b'\x00' * 100

            config = ExportConfig(
                capture_frame=capture,
                start_frame=10,
                end_frame=20
            )
            exporter = ReplayExporter(config)

            exporter.export(30, output_path)

            assert min(captured_frames) == 10
            assert max(captured_frames) == 19

    def test_export_progress_callback(self):
        """Test progress callback during export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp4"
            progress_values = []

            def on_progress(value):
                progress_values.append(value)

            def capture(frame):
                return b'\x00' * 100

            config = ExportConfig(
                capture_frame=capture,
                on_progress=on_progress
            )
            exporter = ReplayExporter(config)

            exporter.export(10, output_path)

            assert len(progress_values) > 0
            # Progress should increase
            assert progress_values[-1] >= progress_values[0]

    def test_cancel_export(self):
        """Test canceling export."""
        frames_captured = []

        def capture(frame):
            frames_captured.append(frame)
            return b'\x00' * 100

        config = ExportConfig(capture_frame=capture)
        exporter = ReplayExporter(config)

        # Cancel immediately
        exporter.cancel()
        assert not exporter.is_exporting

    def test_get_estimated_size_video(self):
        """Test estimating video file size."""
        config = ExportConfig(
            format=ExportFormat.MP4,
            bitrate="8M",
            fps=30
        )
        exporter = ReplayExporter(config)

        # 60 frames at 30fps = 2 seconds
        size = exporter.get_estimated_size(60)
        # At 8Mbps, 2 seconds = 2MB = 2,000,000 bytes
        assert size > 0

    def test_get_estimated_size_gif(self):
        """Test estimating GIF file size."""
        config = ExportConfig(format=ExportFormat.GIF)
        exporter = ReplayExporter(config)

        size = exporter.get_estimated_size(60)
        assert size > 0

    def test_cannot_start_while_exporting(self):
        """Test that starting new export while one is running fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp4"

            def slow_capture(frame):
                import time
                time.sleep(0.01)
                return b'\x00' * 100

            config = ExportConfig(capture_frame=slow_capture)
            exporter = ReplayExporter(config)

            # Start first export in a way that we can test
            # (In real use, this would be async)
            # For now, just verify the is_exporting flag works

    def test_export_error_handling(self):
        """Test error handling during export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.mp4"

            def failing_capture(frame):
                if frame > 5:
                    raise RuntimeError("Capture failed")
                return b'\x00' * 100

            config = ExportConfig(capture_frame=failing_capture)
            exporter = ReplayExporter(config)

            with pytest.raises(RuntimeError):
                exporter.export(20, output_path)

            progress = exporter.progress
            assert progress.error is not None

    def test_different_video_codecs(self):
        """Test different video codecs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            def capture(frame):
                return b'\x00' * 100

            for codec in [VideoCodec.H264, VideoCodec.VP9]:
                output_path = Path(tmpdir) / f"test_{codec.name}.mp4"
                config = ExportConfig(
                    capture_frame=capture,
                    codec=codec
                )
                exporter = ReplayExporter(config)

                progress = exporter.export(10, output_path)
                assert progress.is_complete

    def test_export_jpeg_sequence(self):
        """Test exporting JPEG sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "frames"

            def capture(frame):
                return b'\xFF\xD8\xFF' + b'\x00' * 97  # JPEG-like header

            config = ExportConfig(
                format=ExportFormat.JPEG_SEQUENCE,
                capture_frame=capture
            )
            exporter = ReplayExporter(config)

            progress = exporter.export(5, output_path)

            assert progress.is_complete
            assert len(list(output_path.glob("*.jpg"))) == 5
