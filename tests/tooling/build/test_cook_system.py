"""Tests for asset cooking system."""
import pytest
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from engine.tooling.build.cook_system import (
    AssetCookState,
    CookResult,
    AssetInfo,
    AssetCooker,
    TextureCooker,
    MeshCooker,
    AudioCooker,
    ShaderCooker,
    CookRegistry,
    CookPipeline,
    cook_project,
)


class TestAssetCookState:
    """Tests for AssetCookState enum."""

    def test_all_states_exist(self):
        """Test all cook states exist."""
        assert AssetCookState.DISCOVERED
        assert AssetCookState.FILTERED
        assert AssetCookState.CONVERTING
        assert AssetCookState.CONVERTED
        assert AssetCookState.COMPRESSING
        assert AssetCookState.COMPRESSED
        assert AssetCookState.PACKAGED
        assert AssetCookState.FAILED
        assert AssetCookState.SKIPPED


class TestCookResult:
    """Tests for CookResult dataclass."""

    def test_success_result(self):
        """Test creating success result."""
        result = CookResult(
            success=True,
            source_path="/src/texture.png",
            output_path="/out/texture.dds",
            output_size=1024,
            elapsed_time=0.5,
        )
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        """Test creating failure result."""
        result = CookResult(
            success=False,
            source_path="/src/texture.png",
            error="Unsupported format",
            elapsed_time=0.1,
        )
        assert result.success is False
        assert result.error == "Unsupported format"


class TestAssetInfo:
    """Tests for AssetInfo dataclass."""

    def test_asset_creation(self):
        """Test creating asset info."""
        asset = AssetInfo(
            source_path="/assets/texture.png",
            asset_type="texture",
        )
        assert asset.source_path == "/assets/texture.png"
        assert asset.asset_type == "texture"
        assert asset.state == AssetCookState.DISCOVERED

    def test_compute_hash(self):
        """Test computing asset hash."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            asset = AssetInfo(source_path=temp_path, asset_type="test")
            hash_value = asset.compute_hash()
            assert hash_value != ""
            assert len(hash_value) == 64  # SHA256 hex length
        finally:
            os.unlink(temp_path)


class TestTextureCooker:
    """Tests for TextureCooker."""

    def test_supported_extensions(self):
        """Test supported texture extensions."""
        cooker = TextureCooker()
        exts = cooker.supported_extensions
        assert ".png" in exts
        assert ".jpg" in exts
        assert ".dds" in exts
        assert ".tga" in exts

    def test_asset_type(self):
        """Test asset type."""
        cooker = TextureCooker()
        assert cooker.asset_type == "texture"

    def test_can_cook_texture(self):
        """Test can_cook for textures."""
        cooker = TextureCooker()
        asset = AssetInfo(source_path="/textures/diffuse.png", asset_type="texture")
        assert cooker.can_cook(asset) is True

    def test_cannot_cook_non_texture(self):
        """Test can_cook for non-textures."""
        cooker = TextureCooker()
        asset = AssetInfo(source_path="/models/mesh.fbx", asset_type="mesh")
        assert cooker.can_cook(asset) is False

    def test_cook_texture(self):
        """Test cooking a texture."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            temp_path = f.name

        try:
            cooker = TextureCooker()
            asset = AssetInfo(source_path=temp_path, asset_type="texture")
            result = cooker.cook(asset, "/tmp/output", "windows", {})

            assert result.success is True
            assert result.output_path.endswith(".dds")
        finally:
            os.unlink(temp_path)

    def test_cook_texture_mobile_format(self):
        """Test cooking texture for mobile platforms."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            temp_path = f.name

        try:
            cooker = TextureCooker()
            asset = AssetInfo(source_path=temp_path, asset_type="texture")
            result = cooker.cook(asset, "/tmp/output", "android", {})

            assert result.success is True
            assert result.output_path.endswith(".astc")
        finally:
            os.unlink(temp_path)


class TestMeshCooker:
    """Tests for MeshCooker."""

    def test_supported_extensions(self):
        """Test supported mesh extensions."""
        cooker = MeshCooker()
        exts = cooker.supported_extensions
        assert ".fbx" in exts
        assert ".obj" in exts
        assert ".gltf" in exts
        assert ".blend" in exts

    def test_asset_type(self):
        """Test asset type."""
        cooker = MeshCooker()
        assert cooker.asset_type == "mesh"

    def test_cook_mesh(self):
        """Test cooking a mesh."""
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
            f.write(b"# OBJ file\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3")
            temp_path = f.name

        try:
            cooker = MeshCooker()
            asset = AssetInfo(source_path=temp_path, asset_type="mesh")
            result = cooker.cook(asset, "/tmp/output", "windows", {"generate_lods": True})

            assert result.success is True
            assert result.output_path.endswith(".mesh")
            assert result.metadata.get("lod_count", 0) > 0
        finally:
            os.unlink(temp_path)


class TestAudioCooker:
    """Tests for AudioCooker."""

    def test_supported_extensions(self):
        """Test supported audio extensions."""
        cooker = AudioCooker()
        exts = cooker.supported_extensions
        assert ".wav" in exts
        assert ".mp3" in exts
        assert ".ogg" in exts
        assert ".flac" in exts

    def test_asset_type(self):
        """Test asset type."""
        cooker = AudioCooker()
        assert cooker.asset_type == "audio"

    def test_cook_audio_platform_formats(self):
        """Test audio format selection by platform."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF" + b"\x00" * 100)
            temp_path = f.name

        try:
            cooker = AudioCooker()
            asset = AssetInfo(source_path=temp_path, asset_type="audio")

            # Test different platforms
            result_pc = cooker.cook(asset, "/tmp/out", "windows", {})
            assert result_pc.output_path.endswith(".ogg")

            result_ps5 = cooker.cook(asset, "/tmp/out", "ps5", {})
            assert result_ps5.output_path.endswith(".at9")

            result_xbox = cooker.cook(asset, "/tmp/out", "xbox", {})
            assert result_xbox.output_path.endswith(".xma2")
        finally:
            os.unlink(temp_path)


class TestShaderCooker:
    """Tests for ShaderCooker."""

    def test_supported_extensions(self):
        """Test supported shader extensions."""
        cooker = ShaderCooker()
        exts = cooker.supported_extensions
        assert ".hlsl" in exts
        assert ".glsl" in exts
        assert ".metal" in exts
        assert ".vert" in exts
        assert ".frag" in exts

    def test_asset_type(self):
        """Test asset type."""
        cooker = ShaderCooker()
        assert cooker.asset_type == "shader"

    def test_cook_shader_platform_formats(self):
        """Test shader format selection by platform."""
        with tempfile.NamedTemporaryFile(suffix=".hlsl", delete=False) as f:
            f.write(b"float4 main() : SV_Target { return 0; }")
            temp_path = f.name

        try:
            cooker = ShaderCooker()
            asset = AssetInfo(source_path=temp_path, asset_type="shader")

            result_dx = cooker.cook(asset, "/tmp/out", "windows", {})
            assert ".dxil" in result_dx.output_path or ".dxbc" in result_dx.output_path

            result_vk = cooker.cook(asset, "/tmp/out", "linux", {})
            assert result_vk.output_path.endswith(".spirv")

            result_mtl = cooker.cook(asset, "/tmp/out", "macos", {})
            assert result_mtl.output_path.endswith(".metallib")
        finally:
            os.unlink(temp_path)


class TestCookRegistry:
    """Tests for CookRegistry."""

    def test_register_cooker(self):
        """Test registering a cooker."""
        registry = CookRegistry()
        cooker = TextureCooker()
        registry.register(cooker)

        assert registry.get_cooker("texture") is cooker

    def test_get_cooker_for_extension(self):
        """Test getting cooker by extension."""
        registry = CookRegistry()
        registry.register(TextureCooker())
        registry.register(MeshCooker())

        assert registry.get_cooker_for_extension(".png").asset_type == "texture"
        assert registry.get_cooker_for_extension(".fbx").asset_type == "mesh"

    def test_unregister_cooker(self):
        """Test unregistering a cooker."""
        registry = CookRegistry()
        registry.register(TextureCooker())

        result = registry.unregister("texture")
        assert result is True
        assert registry.get_cooker("texture") is None

    def test_get_supported_extensions(self):
        """Test getting all supported extensions."""
        registry = CookRegistry()
        registry.register(TextureCooker())
        registry.register(MeshCooker())

        exts = registry.get_supported_extensions()
        assert ".png" in exts
        assert ".fbx" in exts


class TestCookPipeline:
    """Tests for CookPipeline."""

    @pytest.fixture
    def temp_source_dir(self):
        """Create temporary source directory with test assets."""
        temp_dir = tempfile.mkdtemp()

        # Create test files
        with open(os.path.join(temp_dir, "texture.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        with open(os.path.join(temp_dir, "model.obj"), "w") as f:
            f.write("# OBJ\nv 0 0 0\n")
        with open(os.path.join(temp_dir, "sound.wav"), "wb") as f:
            f.write(b"RIFF" + b"\x00" * 100)

        yield temp_dir

        shutil.rmtree(temp_dir)

    def test_pipeline_creation(self):
        """Test creating cook pipeline."""
        pipeline = CookPipeline()
        assert pipeline.registry is not None

    def test_discover_assets(self, temp_source_dir):
        """Test discovering assets."""
        pipeline = CookPipeline()
        assets = pipeline.discover(temp_source_dir)

        assert len(assets) >= 3
        types = {a.asset_type for a in assets}
        assert "texture" in types
        assert "mesh" in types
        assert "audio" in types

    def test_filter_assets(self, temp_source_dir):
        """Test filtering assets."""
        pipeline = CookPipeline()
        assets = pipeline.discover(temp_source_dir)

        # Filter only textures
        filtered = pipeline.filter(assets, lambda a: a.asset_type == "texture")
        assert len(filtered) == 1
        assert filtered[0].asset_type == "texture"

    def test_cook_assets(self, temp_source_dir):
        """Test cooking assets."""
        pipeline = CookPipeline()
        assets = pipeline.discover(temp_source_dir)

        output_dir = tempfile.mkdtemp()
        try:
            results = pipeline.cook(assets, output_dir, "windows", {})

            assert len(results) == len(assets)
            assert all(r.success for r in results.values())
        finally:
            shutil.rmtree(output_dir)

    def test_event_callbacks(self, temp_source_dir):
        """Test event callbacks during cooking."""
        pipeline = CookPipeline()
        events = []

        pipeline.on("asset_discovered", lambda a: events.append(("discovered", a.source_path)))
        pipeline.on("asset_cooked", lambda a, r: events.append(("cooked", a.source_path)))
        pipeline.on("cook_started", lambda n: events.append(("started", n)))
        pipeline.on("cook_completed", lambda r: events.append(("completed", len(r))))

        assets = pipeline.discover(temp_source_dir)
        output_dir = tempfile.mkdtemp()
        try:
            pipeline.cook(assets, output_dir, "windows", {})

            assert any(e[0] == "discovered" for e in events)
            assert any(e[0] == "cooked" for e in events)
            assert any(e[0] == "started" for e in events)
            assert any(e[0] == "completed" for e in events)
        finally:
            shutil.rmtree(output_dir)

    def test_cancel_cooking(self):
        """Test cancelling cooking process."""
        pipeline = CookPipeline()
        # Just verify cancel method exists and doesn't raise
        pipeline.cancel()

    def test_get_asset(self, temp_source_dir):
        """Test getting asset by path."""
        pipeline = CookPipeline()
        assets = pipeline.discover(temp_source_dir)

        texture_path = os.path.join(temp_source_dir, "texture.png")
        asset = pipeline.get_asset(texture_path)
        assert asset is not None
        assert asset.asset_type == "texture"

    def test_clear_assets(self, temp_source_dir):
        """Test clearing discovered assets."""
        pipeline = CookPipeline()
        pipeline.discover(temp_source_dir)

        pipeline.clear()
        assert len(pipeline.get_all_assets()) == 0


class TestCookProjectFunction:
    """Tests for cook_project convenience function."""

    @pytest.fixture
    def temp_project(self):
        """Create temporary project structure."""
        temp_dir = tempfile.mkdtemp()

        with open(os.path.join(temp_dir, "texture.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        yield temp_dir

        shutil.rmtree(temp_dir)

    def test_cook_project(self, temp_project):
        """Test cooking entire project."""
        output_dir = tempfile.mkdtemp()
        try:
            results = cook_project(temp_project, output_dir, "windows")
            assert len(results) > 0
            assert all(r.success for r in results.values())
        finally:
            shutil.rmtree(output_dir)
