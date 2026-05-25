"""
Comprehensive tests for ImportPipeline functionality.

Tests import formats, settings, presets, and batch operations.
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.assettools.import_pipeline import (
    ImportPipeline,
    ImportFormat,
    ImportStatus,
    ImportSettings,
    FBXImportSettings,
    OBJImportSettings,
    GLTFImportSettings,
    TextureImportSettings,
    AudioImportSettings,
    ImportPreset,
    ImportResult,
)


@pytest.fixture
def temp_import_dir():
    """Create a temporary directory for import tests."""
    path = Path(tempfile.mkdtemp())

    # Create source files
    (path / "source").mkdir()
    (path / "output").mkdir()

    (path / "source" / "model.fbx").write_text("fbx data")
    (path / "source" / "model.obj").write_text("obj data")
    (path / "source" / "model.gltf").write_text("gltf data")
    (path / "source" / "model.glb").write_bytes(b"glb data")
    (path / "source" / "texture.png").write_bytes(b"png data")
    (path / "source" / "texture.jpg").write_bytes(b"jpg data")
    (path / "source" / "texture.tga").write_bytes(b"tga data")
    (path / "source" / "audio.wav").write_bytes(b"wav data")
    (path / "source" / "audio.ogg").write_bytes(b"ogg data")
    (path / "source" / "audio.mp3").write_bytes(b"mp3 data")
    (path / "source" / "unknown.xyz").write_text("unknown")

    yield path
    shutil.rmtree(path)


class TestImportFormat:
    """Test ImportFormat enum."""

    def test_from_extension_valid(self):
        """from_extension should return correct format."""
        assert ImportFormat.from_extension(".fbx") == ImportFormat.FBX
        assert ImportFormat.from_extension(".obj") == ImportFormat.OBJ
        assert ImportFormat.from_extension(".png") == ImportFormat.PNG
        assert ImportFormat.from_extension(".wav") == ImportFormat.WAV

    def test_from_extension_case_insensitive(self):
        """from_extension should be case insensitive."""
        assert ImportFormat.from_extension(".FBX") == ImportFormat.FBX
        assert ImportFormat.from_extension(".PNG") == ImportFormat.PNG

    def test_from_extension_invalid(self):
        """from_extension should return None for unknown."""
        assert ImportFormat.from_extension(".xyz") is None
        assert ImportFormat.from_extension("invalid") is None

    def test_is_mesh(self):
        """is_mesh should identify mesh formats."""
        assert ImportFormat.FBX.is_mesh
        assert ImportFormat.OBJ.is_mesh
        assert ImportFormat.GLTF.is_mesh
        assert not ImportFormat.PNG.is_mesh
        assert not ImportFormat.WAV.is_mesh

    def test_is_texture(self):
        """is_texture should identify texture formats."""
        assert ImportFormat.PNG.is_texture
        assert ImportFormat.JPG.is_texture
        assert ImportFormat.DDS.is_texture
        assert not ImportFormat.FBX.is_texture
        assert not ImportFormat.WAV.is_texture

    def test_is_audio(self):
        """is_audio should identify audio formats."""
        assert ImportFormat.WAV.is_audio
        assert ImportFormat.OGG.is_audio
        assert ImportFormat.MP3.is_audio
        assert not ImportFormat.PNG.is_audio
        assert not ImportFormat.FBX.is_audio


class TestImportSettings:
    """Test import settings classes."""

    def test_base_settings_defaults(self):
        """Base settings should have sensible defaults."""
        settings = ImportSettings()

        assert settings.overwrite_existing is False
        assert settings.create_backup is True
        assert settings.generate_metadata is True
        assert settings.track_provenance is True

    def test_fbx_settings_defaults(self):
        """FBX settings should have sensible defaults."""
        settings = FBXImportSettings()

        assert settings.import_meshes is True
        assert settings.import_materials is True
        assert settings.import_animations is True
        assert settings.scale_factor == 1.0
        assert settings.up_axis == "Y"

    def test_obj_settings_defaults(self):
        """OBJ settings should have sensible defaults."""
        settings = OBJImportSettings()

        assert settings.import_materials is True
        assert settings.flip_uvs is True
        assert settings.smooth_normals is True

    def test_gltf_settings_defaults(self):
        """glTF settings should have sensible defaults."""
        settings = GLTFImportSettings()

        assert settings.import_meshes is True
        assert settings.import_materials is True
        assert settings.draco_decode is True

    def test_texture_settings_defaults(self):
        """Texture settings should have sensible defaults."""
        settings = TextureImportSettings()

        assert settings.generate_mipmaps is True
        assert settings.srgb is True
        assert settings.power_of_two is True

    def test_audio_settings_defaults(self):
        """Audio settings should have sensible defaults."""
        settings = AudioImportSettings()

        assert settings.sample_rate == 44100
        assert settings.channels == 2
        assert settings.target_format == "ogg"


class TestImportPreset:
    """Test ImportPreset functionality."""

    def test_preset_creation(self):
        """Preset should store all attributes."""
        preset = ImportPreset(
            name="test_preset",
            description="Test preset",
            format=ImportFormat.FBX,
            settings={"import_meshes": True},
        )

        assert preset.name == "test_preset"
        assert preset.format == ImportFormat.FBX
        assert preset.settings["import_meshes"] is True

    def test_preset_to_settings_fbx(self):
        """to_settings() should return FBX settings for FBX format."""
        preset = ImportPreset(
            name="fbx_preset",
            format=ImportFormat.FBX,
            settings={"import_meshes": True, "scale_factor": 0.01},
        )

        settings = preset.to_settings()

        assert isinstance(settings, FBXImportSettings)
        assert settings.import_meshes is True
        assert settings.scale_factor == 0.01

    def test_preset_to_settings_texture(self):
        """to_settings() should return texture settings for texture format."""
        preset = ImportPreset(
            name="tex_preset",
            format=ImportFormat.PNG,
            settings={"srgb": False, "generate_mipmaps": False},
        )

        settings = preset.to_settings()

        assert isinstance(settings, TextureImportSettings)
        assert settings.srgb is False
        assert settings.generate_mipmaps is False

    def test_preset_to_settings_audio(self):
        """to_settings() should return audio settings for audio format."""
        preset = ImportPreset(
            name="audio_preset",
            format=ImportFormat.WAV,
            settings={"channels": 1, "normalize": True},
        )

        settings = preset.to_settings()

        assert isinstance(settings, AudioImportSettings)
        assert settings.channels == 1
        assert settings.normalize is True


class TestImportResult:
    """Test ImportResult functionality."""

    def test_result_creation(self):
        """Result should store attributes."""
        result = ImportResult(source_path=Path("/test/file.fbx"))

        assert result.source_path == Path("/test/file.fbx")
        assert result.status == ImportStatus.PENDING

    def test_success_property(self):
        """success property should check status."""
        result = ImportResult(source_path=Path("/test"))

        assert not result.success

        result.status = ImportStatus.COMPLETED
        assert result.success

    def test_has_warnings(self):
        """has_warnings should check warnings list."""
        result = ImportResult(source_path=Path("/test"))

        assert not result.has_warnings

        result.warnings.append("Warning message")
        assert result.has_warnings


class TestImportPipeline:
    """Test ImportPipeline main class."""

    def test_pipeline_creation(self, temp_import_dir):
        """Pipeline should initialize correctly."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")

        assert pipeline.default_destination == temp_import_dir / "output"

    def test_import_fbx(self, temp_import_dir):
        """import_file() should import FBX files."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "model.fbx"

        result = pipeline.import_file(source)

        assert result.status == ImportStatus.COMPLETED
        assert result.success

    def test_import_obj(self, temp_import_dir):
        """import_file() should import OBJ files."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "model.obj"

        result = pipeline.import_file(source)

        assert result.success

    def test_import_gltf(self, temp_import_dir):
        """import_file() should import glTF files."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "model.gltf"

        result = pipeline.import_file(source)

        assert result.success

    def test_import_texture(self, temp_import_dir):
        """import_file() should import texture files."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "texture.png"

        result = pipeline.import_file(source)

        assert result.success

    def test_import_audio(self, temp_import_dir):
        """import_file() should import audio files."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "audio.wav"

        result = pipeline.import_file(source)

        assert result.success

    def test_import_nonexistent_file(self, temp_import_dir):
        """import_file() should fail for missing files."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")

        result = pipeline.import_file(temp_import_dir / "nonexistent.fbx")

        assert result.status == ImportStatus.FAILED
        assert "not found" in result.error_message.lower()

    def test_import_unsupported_format(self, temp_import_dir):
        """import_file() should fail for unsupported formats."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "unknown.xyz"

        result = pipeline.import_file(source)

        assert result.status == ImportStatus.FAILED
        assert "unsupported" in result.error_message.lower()

    def test_import_with_settings(self, temp_import_dir):
        """import_file() should use provided settings."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "model.fbx"
        settings = FBXImportSettings(scale_factor=0.01)

        result = pipeline.import_file(source, settings=settings)

        assert result.success

    def test_import_with_preset(self, temp_import_dir):
        """import_file() should use preset settings."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "model.fbx"

        # Use built-in preset
        result = pipeline.import_file(source, preset_name="game_ready_mesh")

        assert result.success

    def test_import_batch(self, temp_import_dir):
        """import_batch() should import multiple files."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        sources = [
            temp_import_dir / "source" / "model.fbx",
            temp_import_dir / "source" / "texture.png",
            temp_import_dir / "source" / "audio.wav",
        ]

        results = pipeline.import_batch(sources)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_import_generates_metadata(self, temp_import_dir):
        """Import should generate metadata when enabled."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "model.fbx"
        settings = FBXImportSettings(generate_metadata=True)

        result = pipeline.import_file(source, settings=settings)

        assert result.success
        assert "source_path" in result.metadata

    def test_import_records_time(self, temp_import_dir):
        """Import should record timing."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "model.fbx"

        result = pipeline.import_file(source)

        assert result.import_time_ms > 0

    def test_cancel_batch_import(self, temp_import_dir):
        """cancel() should cancel batch import."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        sources = [temp_import_dir / "source" / f for f in ["model.fbx", "texture.png", "audio.wav"]]

        # Cancel before batch
        pipeline.cancel()
        results = pipeline.import_batch(sources)

        cancelled = [r for r in results if r.status == ImportStatus.CANCELLED]
        assert len(cancelled) > 0


class TestImportPresets:
    """Test preset management."""

    def test_builtin_presets_loaded(self, temp_import_dir):
        """Pipeline should load built-in presets."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")

        assert "game_ready_mesh" in pipeline.presets
        assert "normal_map" in pipeline.presets
        assert "voice_dialogue" in pipeline.presets

    def test_save_preset(self, temp_import_dir):
        """save_preset() should save a preset."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        preset = ImportPreset(
            name="custom_preset",
            description="Custom preset",
            format=ImportFormat.FBX,
            settings={"scale_factor": 0.1},
        )

        pipeline.save_preset(preset)

        assert "custom_preset" in pipeline.presets

    def test_delete_preset(self, temp_import_dir):
        """delete_preset() should remove a preset."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        preset = ImportPreset(name="to_delete", format=ImportFormat.FBX, settings={})
        pipeline.save_preset(preset)

        success = pipeline.delete_preset("to_delete")

        assert success
        assert "to_delete" not in pipeline.presets

    def test_get_preset(self, temp_import_dir):
        """get_preset() should return preset by name."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")

        preset = pipeline.get_preset("game_ready_mesh")

        assert preset is not None
        assert preset.name == "game_ready_mesh"

    def test_list_presets(self, temp_import_dir):
        """list_presets() should return all presets."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")

        presets = pipeline.list_presets()

        assert len(presets) > 0

    def test_list_presets_by_format(self, temp_import_dir):
        """list_presets() should filter by format."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")

        fbx_presets = pipeline.list_presets(format=ImportFormat.FBX)
        audio_presets = pipeline.list_presets(format=ImportFormat.WAV)

        assert all(p.format == ImportFormat.FBX for p in fbx_presets)


class TestImportHistory:
    """Test import history tracking."""

    def test_history_tracking(self, temp_import_dir):
        """Imports should be tracked in history."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "model.fbx"

        pipeline.import_file(source)

        history = pipeline.get_history()
        assert len(history) == 1
        assert history[0].source_path == source

    def test_history_limit(self, temp_import_dir):
        """get_history() should respect limit."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        sources = [
            temp_import_dir / "source" / "model.fbx",
            temp_import_dir / "source" / "texture.png",
            temp_import_dir / "source" / "audio.wav",
        ]
        pipeline.import_batch(sources)

        history = pipeline.get_history(limit=2)

        assert len(history) == 2

    def test_clear_history(self, temp_import_dir):
        """clear_history() should clear all history."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "model.fbx"
        pipeline.import_file(source)

        pipeline.clear_history()

        assert len(pipeline.get_history()) == 0


class TestProgressCallback:
    """Test progress callback functionality."""

    def test_progress_callback(self, temp_import_dir):
        """Progress callback should be called during import."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")
        source = temp_import_dir / "source" / "model.fbx"
        progress_updates = []

        pipeline.on_progress(lambda name, prog: progress_updates.append((name, prog)))
        pipeline.import_file(source)

        assert len(progress_updates) > 0
        # Should start at 0 and end at 1
        assert any(p[1] == 0.0 for p in progress_updates)
        assert any(p[1] == 1.0 for p in progress_updates)


class TestFormatSettings:
    """Test format-specific settings retrieval."""

    def test_get_supported_formats(self, temp_import_dir):
        """get_supported_formats() should return all formats."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")

        formats = pipeline.get_supported_formats()

        assert ImportFormat.FBX in formats
        assert ImportFormat.PNG in formats
        assert ImportFormat.WAV in formats

    def test_get_format_settings_class(self, temp_import_dir):
        """get_format_settings_class() should return correct class."""
        pipeline = ImportPipeline(default_destination=temp_import_dir / "output")

        assert pipeline.get_format_settings_class(ImportFormat.FBX) == FBXImportSettings
        assert pipeline.get_format_settings_class(ImportFormat.OBJ) == OBJImportSettings
        assert pipeline.get_format_settings_class(ImportFormat.GLTF) == GLTFImportSettings
        assert pipeline.get_format_settings_class(ImportFormat.PNG) == TextureImportSettings
        assert pipeline.get_format_settings_class(ImportFormat.WAV) == AudioImportSettings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
