"""Tests for GLES capability detection and workarounds (T-CC-0.9)."""
import pytest

from engine.rendering.quality.gles_capabilities import (
    GLESVersion,
    GLESCapabilities,
    GLESWorkaround,
    GLESWorkaroundRegistry,
    GLES_WORKAROUNDS,
)


class TestGLESVersion:
    """Test GLESVersion enum."""

    def test_version_ordering(self):
        """Test versions are ordered correctly."""
        assert GLESVersion.GLES_20 < GLESVersion.GLES_30
        assert GLESVersion.GLES_30 < GLESVersion.GLES_31
        assert GLESVersion.GLES_31 < GLESVersion.GLES_32

    def test_version_values(self):
        """Test version numeric values."""
        assert GLESVersion.GLES_20 == 20
        assert GLESVersion.GLES_30 == 30
        assert GLESVersion.GLES_31 == 31
        assert GLESVersion.GLES_32 == 32


class TestGLESCapabilitiesCreation:
    """Test GLESCapabilities creation."""

    def test_from_version_gles30(self):
        """Test GLES 3.0 capabilities."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_30)
        assert caps.version == GLESVersion.GLES_30
        assert not caps.has_compute
        assert not caps.has_ssbo
        assert not caps.has_image_load_store
        assert not caps.has_geometry_shader
        assert caps.has_etc2_compression

    def test_from_version_gles31(self):
        """Test GLES 3.1 capabilities."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_31)
        assert caps.version == GLESVersion.GLES_31
        assert caps.has_compute
        assert caps.has_ssbo
        assert caps.has_image_load_store
        assert not caps.has_geometry_shader
        assert caps.max_compute_work_group_size[0] == 1024

    def test_from_version_gles32(self):
        """Test GLES 3.2 capabilities."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_32)
        assert caps.version == GLESVersion.GLES_32
        assert caps.has_compute
        assert caps.has_geometry_shader
        assert caps.has_tessellation
        assert caps.has_texture_buffer

    def test_from_version_gles20(self):
        """Test GLES 2.0 capabilities (very limited)."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_20)
        assert caps.version == GLESVersion.GLES_20
        assert not caps.has_compute
        assert not caps.has_etc2_compression


class TestGLESCapabilitiesDetection:
    """Test capability detection from adapter info."""

    def test_detect_gles30_no_compute(self):
        """Test detection of GLES 3.0 (no compute)."""
        adapter_info = {
            "backend": "OpenGL ES",
            "features": {
                "compute_shader": False,
                "texture_compression_etc2": True,
            },
            "limits": {},
        }
        caps = GLESCapabilities.detect_from_adapter(adapter_info)
        assert caps.version == GLESVersion.GLES_30
        assert not caps.has_compute

    def test_detect_gles31_with_compute(self):
        """Test detection of GLES 3.1 (compute available)."""
        adapter_info = {
            "backend": "OpenGL ES",
            "features": {
                "compute_shader": True,
                "geometry_shader": False,
            },
            "limits": {
                "max_compute_work_group_size_x": 512,
                "max_compute_work_group_size_y": 512,
                "max_compute_work_group_size_z": 64,
                "max_compute_shared_memory_size": 16384,
            },
        }
        caps = GLESCapabilities.detect_from_adapter(adapter_info)
        assert caps.version == GLESVersion.GLES_31
        assert caps.has_compute
        assert caps.max_compute_work_group_size == (512, 512, 64)
        assert caps.max_compute_shared_memory == 16384

    def test_detect_gles32_with_geometry(self):
        """Test detection of GLES 3.2 (geometry shader)."""
        adapter_info = {
            "backend": "OpenGL ES",
            "features": {
                "compute_shader": True,
                "geometry_shader": True,
            },
            "limits": {},
        }
        caps = GLESCapabilities.detect_from_adapter(adapter_info)
        assert caps.version == GLESVersion.GLES_32

    def test_detect_astc_compression(self):
        """Test ASTC compression detection."""
        adapter_info = {
            "backend": "OpenGL ES",
            "features": {
                "compute_shader": True,
                "texture_compression_astc": True,
            },
            "limits": {},
        }
        caps = GLESCapabilities.detect_from_adapter(adapter_info)
        assert caps.has_astc_compression


class TestGLESCapabilitiesWorkarounds:
    """Test workaround requirements."""

    def test_gles30_requires_compute_workaround(self):
        """Test GLES 3.0 requires compute workaround."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_30)
        assert caps.requires_workaround("compute_shader")
        assert caps.requires_workaround("ssbo")
        assert caps.requires_workaround("gpu_culling")
        assert caps.requires_workaround("gpu_particles")
        assert caps.requires_workaround("clustered_lighting")

    def test_gles31_no_compute_workaround(self):
        """Test GLES 3.1 doesn't require compute workaround."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_31)
        assert not caps.requires_workaround("compute_shader")
        assert not caps.requires_workaround("ssbo")
        assert caps.requires_workaround("geometry_shader")
        assert caps.requires_workaround("tessellation")

    def test_gles32_minimal_workarounds(self):
        """Test GLES 3.2 has minimal workarounds."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_32)
        assert not caps.requires_workaround("compute_shader")
        assert not caps.requires_workaround("geometry_shader")
        assert not caps.requires_workaround("tessellation")


class TestGLESCapabilitiesToFeatureFlags:
    """Test conversion to FeatureFlags."""

    def test_gles30_feature_flags(self):
        """Test GLES 3.0 feature flags."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_30)
        flags = caps.to_feature_flags()
        assert not flags.compute_shader
        assert not flags.storage_buffers
        assert flags.texture_compression_etc2
        assert not flags.ray_tracing

    def test_gles31_feature_flags(self):
        """Test GLES 3.1 feature flags."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_31)
        flags = caps.to_feature_flags()
        assert flags.compute_shader
        assert flags.storage_buffers
        assert flags.indirect_draw


class TestGLESWorkaround:
    """Test GLESWorkaround dataclass."""

    def test_workaround_fields(self):
        """Test workaround has all required fields."""
        workaround = GLESWorkaround(
            feature="test_feature",
            min_version=GLESVersion.GLES_31,
            workaround_strategy="Use fallback",
            performance_impact="2x slower",
            implementation_notes="Details here",
        )
        assert workaround.feature == "test_feature"
        assert workaround.min_version == GLESVersion.GLES_31
        assert "fallback" in workaround.workaround_strategy


class TestGLESWorkaroundRegistry:
    """Test GLESWorkaroundRegistry."""

    def test_get_compute_workaround(self):
        """Test getting compute shader workaround."""
        workaround = GLESWorkaroundRegistry.get("compute_shader")
        assert workaround is not None
        assert workaround.min_version == GLESVersion.GLES_31
        assert "ping-pong" in workaround.workaround_strategy.lower()

    def test_get_nonexistent_workaround(self):
        """Test getting non-existent workaround returns None."""
        workaround = GLESWorkaroundRegistry.get("nonexistent_feature")
        assert workaround is None

    def test_list_required_gles30(self):
        """Test listing required workarounds for GLES 3.0."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_30)
        required = GLESWorkaroundRegistry.list_required(caps)
        features = {w.feature for w in required}
        assert "compute_shader" in features
        assert "ssbo" in features
        assert "gpu_culling" in features

    def test_list_required_gles31(self):
        """Test listing required workarounds for GLES 3.1."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_31)
        required = GLESWorkaroundRegistry.list_required(caps)
        features = {w.feature for w in required}
        assert "compute_shader" not in features
        assert "geometry_shader" in features

    def test_list_required_gles32(self):
        """Test listing required workarounds for GLES 3.2."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_32)
        required = GLESWorkaroundRegistry.list_required(caps)
        assert len(required) == 0  # GLES 3.2 has all features

    def test_get_strategy(self):
        """Test getting workaround strategy string."""
        strategy = GLESWorkaroundRegistry.get_strategy("ssbo")
        assert strategy is not None
        assert "uniform buffer" in strategy.lower()

    def test_all_workarounds(self):
        """Test getting all workarounds."""
        all_workarounds = GLESWorkaroundRegistry.all_workarounds()
        assert len(all_workarounds) == len(GLES_WORKAROUNDS)


class TestGLESWorkaroundDocumentation:
    """Test workaround documentation is complete."""

    def test_all_workarounds_have_notes(self):
        """Test all workarounds have implementation notes."""
        for workaround in GLES_WORKAROUNDS:
            assert workaround.implementation_notes, f"{workaround.feature} missing notes"
            assert len(workaround.implementation_notes) > 20

    def test_all_workarounds_have_impact(self):
        """Test all workarounds have performance impact."""
        for workaround in GLES_WORKAROUNDS:
            assert workaround.performance_impact, f"{workaround.feature} missing impact"

    def test_all_workarounds_have_strategy(self):
        """Test all workarounds have strategy."""
        for workaround in GLES_WORKAROUNDS:
            assert workaround.workaround_strategy, f"{workaround.feature} missing strategy"


class TestComputelessRenderingPath:
    """Test compute-less rendering path for GLES 3.0."""

    def test_gles30_can_render_forward(self):
        """Test GLES 3.0 can use forward rendering path."""
        caps = GLESCapabilities.from_version(GLESVersion.GLES_30)
        # Forward rendering doesn't require compute
        assert not caps.requires_workaround("forward_rendering")

    def test_gles30_particle_workaround_exists(self):
        """Test particle workaround exists for GLES 3.0."""
        workaround = GLESWorkaroundRegistry.get("gpu_particles")
        assert workaround is not None
        assert "transform feedback" in workaround.workaround_strategy.lower()

    def test_gles30_lighting_workaround_exists(self):
        """Test lighting workaround exists for GLES 3.0."""
        workaround = GLESWorkaroundRegistry.get("clustered_lighting")
        assert workaround is not None
        assert "forward" in workaround.workaround_strategy.lower()

    def test_gles30_culling_workaround_exists(self):
        """Test culling workaround exists for GLES 3.0."""
        workaround = GLESWorkaroundRegistry.get("gpu_culling")
        assert workaround is not None
        assert "cpu" in workaround.workaround_strategy.lower()
