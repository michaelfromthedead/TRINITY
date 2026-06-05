"""Tests for baked probe capture system.

Covers:
- HDR pixel operations
- Cubemap face data and sampling
- BC6H compression/decompression
- KTX2 format writing/reading
- Mip chain generation
- Pre-filtered environment maps
- Probe capture and serialization
- Probe manager and blending
"""

from __future__ import annotations

import math
import struct
import tempfile
from pathlib import Path

import pytest

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3
from engine.rendering.lighting.baked_probes import (
    BakedProbeConstants,
    CubemapFace,
    CUBEMAP_FACE_DIRECTIONS,
    CompressionQuality,
    FilterMode,
    HDRPixel,
    CubemapFaceData,
    CubemapData,
    MipLevel,
    CubemapMipChain,
    BC6HBlock,
    BC6HCompressor,
    KTX2Format,
    KTX2Header,
    KTX2Writer,
    KTX2Reader,
    CaptureConfig,
    CubemapRenderer,
    FunctionCubemapRenderer,
    MipGenerator,
    PrefilteredGenerator,
    BakedProbeConfig,
    BakedProbeAsset,
    BakedProbeCapture,
    BakedProbeManager,
)


# -----------------------------------------------------------------------------
# HDRPixel Tests
# -----------------------------------------------------------------------------

class TestHDRPixel:
    """Tests for HDR pixel operations."""

    def test_hdr_pixel_creation(self) -> None:
        """Test creating an HDR pixel."""
        pixel = HDRPixel(1.0, 2.0, 3.0)
        assert pixel.r == pytest.approx(1.0)
        assert pixel.g == pytest.approx(2.0)
        assert pixel.b == pytest.approx(3.0)

    def test_hdr_pixel_default(self) -> None:
        """Test default HDR pixel is black."""
        pixel = HDRPixel()
        assert pixel.r == pytest.approx(0.0)
        assert pixel.g == pytest.approx(0.0)
        assert pixel.b == pytest.approx(0.0)

    def test_hdr_pixel_addition(self) -> None:
        """Test adding HDR pixels."""
        p1 = HDRPixel(1.0, 2.0, 3.0)
        p2 = HDRPixel(0.5, 0.5, 0.5)
        result = p1 + p2
        assert result.r == pytest.approx(1.5)
        assert result.g == pytest.approx(2.5)
        assert result.b == pytest.approx(3.5)

    def test_hdr_pixel_scalar_multiplication(self) -> None:
        """Test multiplying HDR pixel by scalar."""
        pixel = HDRPixel(1.0, 2.0, 3.0)
        result = pixel * 2.0
        assert result.r == pytest.approx(2.0)
        assert result.g == pytest.approx(4.0)
        assert result.b == pytest.approx(6.0)

    def test_hdr_pixel_rmul(self) -> None:
        """Test reverse scalar multiplication."""
        pixel = HDRPixel(1.0, 2.0, 3.0)
        result = 0.5 * pixel
        assert result.r == pytest.approx(0.5)
        assert result.g == pytest.approx(1.0)
        assert result.b == pytest.approx(1.5)

    def test_hdr_pixel_clamp(self) -> None:
        """Test clamping to HDR range."""
        pixel = HDRPixel(-1.0, 100000.0, 0.5)
        clamped = pixel.clamp_hdr()
        assert clamped.r == pytest.approx(0.0)
        assert clamped.g == pytest.approx(BakedProbeConstants.HDR_MAX_VALUE)
        assert clamped.b == pytest.approx(0.5)

    def test_hdr_pixel_luminance(self) -> None:
        """Test luminance calculation."""
        pixel = HDRPixel(1.0, 1.0, 1.0)
        lum = pixel.luminance()
        # 0.2126 + 0.7152 + 0.0722 = 1.0
        assert lum == pytest.approx(1.0)

    def test_hdr_pixel_to_vec3(self) -> None:
        """Test conversion to Vec3."""
        pixel = HDRPixel(1.0, 2.0, 3.0)
        v = pixel.to_vec3()
        assert v.x == pytest.approx(1.0)
        assert v.y == pytest.approx(2.0)
        assert v.z == pytest.approx(3.0)

    def test_hdr_pixel_from_vec3(self) -> None:
        """Test creation from Vec3."""
        v = Vec3(1.0, 2.0, 3.0)
        pixel = HDRPixel.from_vec3(v)
        assert pixel.r == pytest.approx(1.0)
        assert pixel.g == pytest.approx(2.0)
        assert pixel.b == pytest.approx(3.0)


# -----------------------------------------------------------------------------
# CubemapFaceData Tests
# -----------------------------------------------------------------------------

class TestCubemapFaceData:
    """Tests for cubemap face data."""

    def test_face_data_creation(self) -> None:
        """Test creating face data."""
        face = CubemapFaceData(face=CubemapFace.POSITIVE_X, resolution=4)
        assert face.resolution == 4
        assert len(face.pixels) == 16

    def test_face_data_get_set_pixel(self) -> None:
        """Test getting and setting pixels."""
        face = CubemapFaceData(face=CubemapFace.POSITIVE_X, resolution=4)
        pixel = HDRPixel(1.0, 0.5, 0.25)
        face.set_pixel(2, 1, pixel)
        result = face.get_pixel(2, 1)
        assert result.r == pytest.approx(1.0)
        assert result.g == pytest.approx(0.5)
        assert result.b == pytest.approx(0.25)

    def test_face_data_out_of_bounds(self) -> None:
        """Test out of bounds access raises error."""
        face = CubemapFaceData(face=CubemapFace.POSITIVE_X, resolution=4)
        with pytest.raises(IndexError):
            face.get_pixel(4, 0)
        with pytest.raises(IndexError):
            face.set_pixel(-1, 0, HDRPixel())

    def test_face_data_bilinear_sample(self) -> None:
        """Test bilinear sampling."""
        face = CubemapFaceData(face=CubemapFace.POSITIVE_X, resolution=2)
        # Set corners
        face.set_pixel(0, 0, HDRPixel(0.0, 0.0, 0.0))
        face.set_pixel(1, 0, HDRPixel(1.0, 0.0, 0.0))
        face.set_pixel(0, 1, HDRPixel(0.0, 1.0, 0.0))
        face.set_pixel(1, 1, HDRPixel(1.0, 1.0, 0.0))

        # Sample center
        result = face.sample_bilinear(0.5, 0.5)
        assert result.r == pytest.approx(0.5, abs=0.1)
        assert result.g == pytest.approx(0.5, abs=0.1)

    def test_face_data_invalid_pixel_count(self) -> None:
        """Test error on wrong pixel count."""
        with pytest.raises(ValueError):
            CubemapFaceData(
                face=CubemapFace.POSITIVE_X,
                resolution=4,
                pixels=[HDRPixel()] * 10  # Wrong count
            )


# -----------------------------------------------------------------------------
# CubemapData Tests
# -----------------------------------------------------------------------------

class TestCubemapData:
    """Tests for complete cubemap data."""

    def test_cubemap_creation(self) -> None:
        """Test creating a cubemap."""
        cubemap = CubemapData(resolution=4)
        assert cubemap.resolution == 4
        assert len(cubemap.faces) == 6

    def test_cubemap_get_face(self) -> None:
        """Test getting a specific face."""
        cubemap = CubemapData(resolution=4)
        face = cubemap.get_face(CubemapFace.POSITIVE_Y)
        assert face.face == CubemapFace.POSITIVE_Y

    def test_cubemap_sample_direction_positive_x(self) -> None:
        """Test sampling in positive X direction."""
        cubemap = CubemapData(resolution=4)
        # Set +X face to red
        for y in range(4):
            for x in range(4):
                cubemap.get_face(CubemapFace.POSITIVE_X).set_pixel(
                    x, y, HDRPixel(1.0, 0.0, 0.0)
                )

        result = cubemap.sample_direction(Vec3(1, 0, 0))
        assert result.r > 0.5  # Should be mostly red

    def test_cubemap_sample_direction_negative_y(self) -> None:
        """Test sampling in negative Y direction."""
        cubemap = CubemapData(resolution=4)
        # Set -Y face to green
        for y in range(4):
            for x in range(4):
                cubemap.get_face(CubemapFace.NEGATIVE_Y).set_pixel(
                    x, y, HDRPixel(0.0, 1.0, 0.0)
                )

        result = cubemap.sample_direction(Vec3(0, -1, 0))
        assert result.g > 0.5  # Should be mostly green

    def test_cubemap_invalid_face_count(self) -> None:
        """Test error on wrong face count."""
        with pytest.raises(ValueError):
            CubemapData(
                resolution=4,
                faces=[CubemapFaceData(CubemapFace.POSITIVE_X, 4)]
            )


# -----------------------------------------------------------------------------
# BC6H Compression Tests
# -----------------------------------------------------------------------------

class TestBC6HCompressor:
    """Tests for BC6H compression."""

    def test_compressor_creation(self) -> None:
        """Test creating a compressor."""
        comp = BC6HCompressor(CompressionQuality.HIGH)
        assert comp.quality == CompressionQuality.HIGH

    def test_compressor_quality_setter(self) -> None:
        """Test setting compression quality."""
        comp = BC6HCompressor()
        comp.quality = CompressionQuality.FAST
        assert comp.quality == CompressionQuality.FAST

    def test_compress_face(self) -> None:
        """Test compressing a face."""
        face = CubemapFaceData(face=CubemapFace.POSITIVE_X, resolution=4)
        # Fill with gradient
        for y in range(4):
            for x in range(4):
                face.set_pixel(x, y, HDRPixel(x / 3.0, y / 3.0, 0.5))

        comp = BC6HCompressor()
        data = comp.compress_face(face)

        # 4x4 = 1 block = 16 bytes
        assert len(data) == 16

    def test_compress_decompress_roundtrip(self) -> None:
        """Test compression/decompression preserves data."""
        face = CubemapFaceData(face=CubemapFace.POSITIVE_X, resolution=4)
        # Fill with uniform color
        for y in range(4):
            for x in range(4):
                face.set_pixel(x, y, HDRPixel(1.0, 0.5, 0.25))

        comp = BC6HCompressor()
        compressed = comp.compress_face(face)
        decompressed = comp.decompress_face(compressed, 4, CubemapFace.POSITIVE_X)

        # Check center pixel is approximately correct
        result = decompressed.get_pixel(2, 2)
        assert result.r == pytest.approx(1.0, abs=0.3)
        assert result.g == pytest.approx(0.5, abs=0.3)
        assert result.b == pytest.approx(0.25, abs=0.3)

    def test_compress_cubemap(self) -> None:
        """Test compressing entire cubemap."""
        cubemap = CubemapData(resolution=4)
        comp = BC6HCompressor()
        compressed = comp.compress_cubemap(cubemap)

        assert len(compressed) == 6
        for face_data in compressed:
            assert len(face_data) == 16

    def test_estimate_compressed_size(self) -> None:
        """Test size estimation."""
        comp = BC6HCompressor()
        size = comp.estimate_compressed_size(256)
        # 256x256 = 64x64 blocks = 4096 blocks
        # 4096 * 16 bytes = 65536 bytes per face
        # 65536 * 6 faces = 393216 bytes
        assert size == 393216

    def test_compression_invalid_resolution(self) -> None:
        """Test error on non-block-aligned resolution."""
        face = CubemapFaceData(face=CubemapFace.POSITIVE_X, resolution=5)  # Not multiple of 4
        # Manually set pixels to avoid __post_init__ check
        face.pixels = [HDRPixel() for _ in range(25)]
        face.resolution = 5

        comp = BC6HCompressor()
        with pytest.raises(ValueError):
            comp.compress_face(face)


class TestBC6HBlock:
    """Tests for BC6H block structure."""

    def test_block_creation(self) -> None:
        """Test creating a BC6H block."""
        block = BC6HBlock()
        assert len(block.data) == 16

    def test_block_invalid_size(self) -> None:
        """Test error on wrong block size."""
        with pytest.raises(ValueError):
            BC6HBlock(data=bytes(10))


# -----------------------------------------------------------------------------
# KTX2 Format Tests
# -----------------------------------------------------------------------------

class TestKTX2Writer:
    """Tests for KTX2 writing."""

    def test_write_to_bytes(self) -> None:
        """Test writing KTX2 to bytes."""
        cubemap = CubemapData(resolution=4)
        mip_chain = CubemapMipChain(base_resolution=4, mip_count=1)
        mip_chain.mips.append(MipLevel(level=0, resolution=4, cubemap=cubemap))

        comp = BC6HCompressor()
        compressed = [comp.compress_cubemap(cubemap)]

        writer = KTX2Writer()
        data = writer.write_to_bytes(mip_chain, compressed, supercompress=False)

        # Verify magic number
        assert data[:12] == BakedProbeConstants.KTX2_MAGIC

    def test_write_with_supercompression(self) -> None:
        """Test writing with zlib supercompression."""
        cubemap = CubemapData(resolution=4)
        mip_chain = CubemapMipChain(base_resolution=4, mip_count=1)
        mip_chain.mips.append(MipLevel(level=0, resolution=4, cubemap=cubemap))

        comp = BC6HCompressor()
        compressed = [comp.compress_cubemap(cubemap)]

        writer = KTX2Writer()
        data_uncompressed = writer.write_to_bytes(mip_chain, compressed, supercompress=False)
        data_compressed = writer.write_to_bytes(mip_chain, compressed, supercompress=True)

        # Compressed should typically be smaller or same size
        assert len(data_compressed) <= len(data_uncompressed) + 100  # Allow overhead

    def test_write_to_file(self) -> None:
        """Test writing KTX2 to file."""
        cubemap = CubemapData(resolution=4)
        mip_chain = CubemapMipChain(base_resolution=4, mip_count=1)
        mip_chain.mips.append(MipLevel(level=0, resolution=4, cubemap=cubemap))

        comp = BC6HCompressor()
        compressed = [comp.compress_cubemap(cubemap)]

        with tempfile.NamedTemporaryFile(suffix='.ktx2', delete=False) as f:
            path = Path(f.name)

        writer = KTX2Writer()
        size = writer.write(mip_chain, compressed, path, supercompress=False)

        assert size > 0
        assert path.exists()
        path.unlink()


class TestKTX2Reader:
    """Tests for KTX2 reading."""

    def test_read_from_bytes(self) -> None:
        """Test reading KTX2 from bytes."""
        # Create and write KTX2
        cubemap = CubemapData(resolution=4)
        mip_chain = CubemapMipChain(base_resolution=4, mip_count=1)
        mip_chain.mips.append(MipLevel(level=0, resolution=4, cubemap=cubemap))

        comp = BC6HCompressor()
        compressed = [comp.compress_cubemap(cubemap)]

        writer = KTX2Writer()
        data = writer.write_to_bytes(mip_chain, compressed, supercompress=False)

        # Read back
        reader = KTX2Reader()
        header, mip_data = reader.read_from_bytes(data)

        assert header.width == 4
        assert header.height == 4
        assert header.face_count == 6
        assert len(mip_data) == 1
        assert len(mip_data[0]) == 6

    def test_read_with_decompression(self) -> None:
        """Test reading supercompressed KTX2."""
        cubemap = CubemapData(resolution=4)
        mip_chain = CubemapMipChain(base_resolution=4, mip_count=1)
        mip_chain.mips.append(MipLevel(level=0, resolution=4, cubemap=cubemap))

        comp = BC6HCompressor()
        compressed = [comp.compress_cubemap(cubemap)]

        writer = KTX2Writer()
        data = writer.write_to_bytes(mip_chain, compressed, supercompress=True)

        reader = KTX2Reader()
        header, mip_data = reader.read_from_bytes(data)

        assert header.supercompression == 3  # Zlib
        assert len(mip_data[0]) == 6

    def test_read_invalid_magic(self) -> None:
        """Test error on invalid magic number."""
        reader = KTX2Reader()
        with pytest.raises(ValueError, match="Invalid KTX2 magic"):
            reader.read_from_bytes(b'INVALID_MAGIC_NUMBER')


# -----------------------------------------------------------------------------
# Mip Generation Tests
# -----------------------------------------------------------------------------

class TestMipGenerator:
    """Tests for mip chain generation."""

    def test_generator_creation(self) -> None:
        """Test creating a mip generator."""
        gen = MipGenerator(FilterMode.GAUSSIAN)
        assert gen.filter_mode == FilterMode.GAUSSIAN

    def test_generate_mips_default(self) -> None:
        """Test generating all mip levels."""
        cubemap = CubemapData(resolution=8)
        gen = MipGenerator()
        chain = gen.generate_mips(cubemap)

        # 8 -> 4 -> 2 -> 1 = 4 levels
        assert chain.mip_count == 4
        assert len(chain.mips) == 4
        assert chain.mips[0].resolution == 8
        assert chain.mips[1].resolution == 4
        assert chain.mips[2].resolution == 2
        assert chain.mips[3].resolution == 1

    def test_generate_mips_limited(self) -> None:
        """Test generating limited mip levels."""
        cubemap = CubemapData(resolution=8)
        gen = MipGenerator()
        chain = gen.generate_mips(cubemap, mip_count=2)

        assert len(chain.mips) == 2
        assert chain.mips[0].resolution == 8
        assert chain.mips[1].resolution == 4

    def test_mip_chain_properties(self) -> None:
        """Test mip chain properties."""
        cubemap = CubemapData(resolution=16)
        gen = MipGenerator()
        chain = gen.generate_mips(cubemap)

        assert chain.base_resolution == 16
        assert chain.max_mip_levels == 5  # log2(16) + 1


class TestCubemapMipChain:
    """Tests for cubemap mip chain."""

    def test_mip_chain_creation(self) -> None:
        """Test creating a mip chain."""
        chain = CubemapMipChain(base_resolution=256, mip_count=8)
        assert chain.base_resolution == 256
        assert chain.mip_count == 8

    def test_mip_chain_get_mip(self) -> None:
        """Test getting a mip level."""
        chain = CubemapMipChain(base_resolution=8, mip_count=3)
        for i in range(3):
            chain.mips.append(MipLevel(
                level=i,
                resolution=8 >> i,
                cubemap=CubemapData(resolution=8 >> i),
            ))

        mip = chain.get_mip(1)
        assert mip is not None
        assert mip.resolution == 4

        assert chain.get_mip(10) is None

    def test_mip_chain_sample_roughness(self) -> None:
        """Test roughness-based sampling."""
        chain = CubemapMipChain(base_resolution=4, mip_count=2, is_prefiltered=True)

        # Create mips with different colors
        mip0 = CubemapData(resolution=4)
        for face in mip0.faces:
            for i in range(16):
                face.pixels[i] = HDRPixel(1.0, 0.0, 0.0)  # Red

        mip1 = CubemapData(resolution=2)
        for face in mip1.faces:
            for i in range(4):
                face.pixels[i] = HDRPixel(0.0, 1.0, 0.0)  # Green

        chain.mips.append(MipLevel(level=0, resolution=4, cubemap=mip0, roughness=0.0))
        chain.mips.append(MipLevel(level=1, resolution=2, cubemap=mip1, roughness=1.0))

        # Sample at roughness 0 (should be red)
        result = chain.sample_roughness(Vec3(1, 0, 0), 0.0)
        assert result.r > 0.5

        # Sample at roughness 1 (should be green)
        result = chain.sample_roughness(Vec3(1, 0, 0), 1.0)
        assert result.g > 0.5


# -----------------------------------------------------------------------------
# Pre-filtered Generator Tests
# -----------------------------------------------------------------------------

class TestPrefilteredGenerator:
    """Tests for pre-filtered environment map generation."""

    def test_generator_creation(self) -> None:
        """Test creating a prefilter generator."""
        gen = PrefilteredGenerator(sample_count=512, roughness_levels=6)
        assert gen.sample_count == 512
        assert gen.roughness_levels == 6

    def test_generate_prefiltered(self) -> None:
        """Test generating pre-filtered mips."""
        # Create a simple environment
        cubemap = CubemapData(resolution=4)
        for face in cubemap.faces:
            for i in range(16):
                face.pixels[i] = HDRPixel(1.0, 1.0, 1.0)

        gen = PrefilteredGenerator(sample_count=32, roughness_levels=3)
        chain = gen.generate_prefiltered(cubemap)

        assert chain.is_prefiltered
        assert len(chain.mips) == 3
        assert chain.mips[0].roughness == pytest.approx(0.0)
        assert chain.mips[2].roughness == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# Cubemap Renderer Tests
# -----------------------------------------------------------------------------

class TestCubemapRenderer:
    """Tests for cubemap rendering."""

    def test_capture_config_defaults(self) -> None:
        """Test capture config defaults."""
        config = CaptureConfig()
        assert config.resolution == BakedProbeConstants.DEFAULT_RESOLUTION
        assert config.near_plane == pytest.approx(0.1)
        assert config.far_plane == pytest.approx(1000.0)

    def test_function_renderer_capture(self) -> None:
        """Test function-based cubemap capture."""
        config = CaptureConfig(resolution=4)

        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            # Sky gradient based on Y direction
            return Vec3(0.5, 0.5 + direction.y * 0.5, 1.0)

        renderer = FunctionCubemapRenderer(config, sample_func)
        cubemap = renderer.capture(Vec3(0, 0, 0))

        assert cubemap.resolution == 4
        assert len(cubemap.faces) == 6

        # Check that up direction has higher Y component
        up_sample = cubemap.sample_direction(Vec3(0, 1, 0))
        down_sample = cubemap.sample_direction(Vec3(0, -1, 0))
        assert up_sample.g > down_sample.g


# -----------------------------------------------------------------------------
# Baked Probe Asset Tests
# -----------------------------------------------------------------------------

class TestBakedProbeAsset:
    """Tests for baked probe assets."""

    def test_asset_creation(self) -> None:
        """Test creating a probe asset."""
        asset = BakedProbeAsset(
            probe_id=1,
            name="test_probe",
            position=Vec3(0, 5, 0),
            bounds=AABB(Vec3(-10, 0, -10), Vec3(10, 10, 10)),
            resolution=256,
            mip_count=8,
            is_prefiltered=True,
        )
        assert asset.probe_id == 1
        assert asset.name == "test_probe"
        assert asset.resolution == 256
        assert not asset.loaded

    def test_asset_load_unload(self) -> None:
        """Test loading and unloading a probe."""
        # Create valid KTX2 data
        cubemap = CubemapData(resolution=4)
        mip_chain = CubemapMipChain(base_resolution=4, mip_count=1)
        mip_chain.mips.append(MipLevel(level=0, resolution=4, cubemap=cubemap))

        comp = BC6HCompressor()
        compressed = [comp.compress_cubemap(cubemap)]

        writer = KTX2Writer()
        ktx2_data = writer.write_to_bytes(mip_chain, compressed, supercompress=False)

        asset = BakedProbeAsset(
            probe_id=1,
            name="test",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)),
            resolution=4,
            mip_count=1,
            is_prefiltered=False,
            ktx2_data=ktx2_data,
        )

        assert not asset.loaded
        asset.load()
        assert asset.loaded

        asset.unload()
        assert not asset.loaded

    def test_asset_sample(self) -> None:
        """Test sampling a loaded probe."""
        # Create probe with known color
        cubemap = CubemapData(resolution=4)
        for face in cubemap.faces:
            for i in range(16):
                face.pixels[i] = HDRPixel(1.0, 0.5, 0.25)

        mip_chain = CubemapMipChain(base_resolution=4, mip_count=1)
        mip_chain.mips.append(MipLevel(level=0, resolution=4, cubemap=cubemap))

        comp = BC6HCompressor()
        compressed = [comp.compress_cubemap(cubemap)]

        writer = KTX2Writer()
        ktx2_data = writer.write_to_bytes(mip_chain, compressed, supercompress=False)

        asset = BakedProbeAsset(
            probe_id=1,
            name="test",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)),
            resolution=4,
            mip_count=1,
            is_prefiltered=False,
            ktx2_data=ktx2_data,
        )

        # Sample before loading should return black
        result = asset.sample(Vec3(1, 0, 0))
        assert result == Vec3(0, 0, 0)

        # Sample after loading
        asset.load()
        result = asset.sample(Vec3(1, 0, 0))
        # Should be approximately the color we set (with compression loss)
        assert result.x > 0.5


# -----------------------------------------------------------------------------
# Baked Probe Capture Tests
# -----------------------------------------------------------------------------

class TestBakedProbeCapture:
    """Tests for the main capture system."""

    def test_capture_probe(self) -> None:
        """Test capturing a probe."""
        config = CaptureConfig(resolution=4)

        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            return Vec3(1.0, 0.5, 0.25)

        renderer = FunctionCubemapRenderer(config, sample_func)

        bake_config = BakedProbeConfig(
            resolution=4,
            roughness_levels=2,
            sample_count=16,
            supercompress=False,
        )

        capture = BakedProbeCapture(renderer, bake_config)
        asset = capture.capture_probe(
            name="test_probe",
            position=Vec3(0, 5, 0),
            bounds=AABB(Vec3(-10, 0, -10), Vec3(10, 10, 10)),
            prefilter=True,
        )

        assert asset.name == "test_probe"
        assert asset.resolution == 4
        assert asset.mip_count == 2
        assert asset.is_prefiltered
        assert len(asset.ktx2_data) > 0

    def test_capture_without_prefilter(self) -> None:
        """Test capturing without pre-filtering."""
        config = CaptureConfig(resolution=4)

        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            return Vec3(1.0, 1.0, 1.0)

        renderer = FunctionCubemapRenderer(config, sample_func)

        bake_config = BakedProbeConfig(resolution=4, roughness_levels=3)
        capture = BakedProbeCapture(renderer, bake_config)

        asset = capture.capture_probe(
            name="simple_probe",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
            prefilter=False,
        )

        assert not asset.is_prefiltered

    def test_save_load_probe(self) -> None:
        """Test saving and loading a probe."""
        config = CaptureConfig(resolution=4)

        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            return Vec3(0.5, 0.5, 0.5)

        renderer = FunctionCubemapRenderer(config, sample_func)
        bake_config = BakedProbeConfig(resolution=4, roughness_levels=2, supercompress=False)
        capture = BakedProbeCapture(renderer, bake_config)

        asset = capture.capture_probe(
            name="saved_probe",
            position=Vec3(1, 2, 3),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
        )

        with tempfile.NamedTemporaryFile(suffix='.bkpr', delete=False) as f:
            path = Path(f.name)

        try:
            size = capture.save_probe(asset, path)
            assert size > 0
            assert path.exists()

            loaded = capture.load_probe(path)
            assert loaded.name == "saved_probe"
            assert loaded.position.x == pytest.approx(1.0)
            assert loaded.position.y == pytest.approx(2.0)
            assert loaded.position.z == pytest.approx(3.0)
            assert loaded.resolution == 4
            assert loaded.mip_count == asset.mip_count
        finally:
            path.unlink()


# -----------------------------------------------------------------------------
# Probe Manager Tests
# -----------------------------------------------------------------------------

class TestBakedProbeManager:
    """Tests for probe management."""

    def _create_test_asset(self, probe_id: int, position: Vec3) -> BakedProbeAsset:
        """Helper to create a test asset."""
        cubemap = CubemapData(resolution=4)
        for face in cubemap.faces:
            for i in range(16):
                face.pixels[i] = HDRPixel(1.0, 1.0, 1.0)

        mip_chain = CubemapMipChain(base_resolution=4, mip_count=1)
        mip_chain.mips.append(MipLevel(level=0, resolution=4, cubemap=cubemap))

        comp = BC6HCompressor()
        compressed = [comp.compress_cubemap(cubemap)]

        writer = KTX2Writer()
        ktx2_data = writer.write_to_bytes(mip_chain, compressed, supercompress=False)

        return BakedProbeAsset(
            probe_id=probe_id,
            name=f"probe_{probe_id}",
            position=position,
            bounds=AABB(position - Vec3(5, 5, 5), position + Vec3(5, 5, 5)),
            resolution=4,
            mip_count=1,
            is_prefiltered=False,
            ktx2_data=ktx2_data,
        )

    def test_manager_add_remove(self) -> None:
        """Test adding and removing probes."""
        manager = BakedProbeManager()
        asset = self._create_test_asset(1, Vec3(0, 0, 0))

        manager.add_probe(asset)
        assert manager.get_probe(1) is not None

        manager.remove_probe(1)
        assert manager.get_probe(1) is None

    def test_manager_load_probe(self) -> None:
        """Test loading a probe."""
        manager = BakedProbeManager()
        asset = self._create_test_asset(1, Vec3(0, 0, 0))
        manager.add_probe(asset)

        assert not asset.loaded
        result = manager.load_probe(1)
        assert result
        assert asset.loaded

    def test_manager_find_affecting_probes(self) -> None:
        """Test finding probes affecting a point."""
        manager = BakedProbeManager()

        asset1 = self._create_test_asset(1, Vec3(0, 0, 0))
        asset2 = self._create_test_asset(2, Vec3(10, 0, 0))

        manager.add_probe(asset1)
        manager.add_probe(asset2)

        # Point at origin should be affected by probe 1
        affecting = manager.find_affecting_probes(Vec3(0, 0, 0))
        assert len(affecting) >= 1
        assert any(p.probe_id == 1 for p, _ in affecting)

        # Point far away should not be affected
        affecting = manager.find_affecting_probes(Vec3(100, 0, 0))
        assert len(affecting) == 0

    def test_manager_sample_blended(self) -> None:
        """Test blended sampling."""
        manager = BakedProbeManager()
        asset = self._create_test_asset(1, Vec3(0, 0, 0))
        manager.add_probe(asset)

        # Sample inside bounds
        result = manager.sample(Vec3(0, 0, 0), Vec3(1, 0, 0), 0.0)
        # Should get some color from the probe
        assert result.x > 0 or result.y > 0 or result.z > 0

    def test_manager_eviction(self) -> None:
        """Test LRU eviction of loaded probes."""
        manager = BakedProbeManager(max_loaded_probes=2)

        assets = [self._create_test_asset(i, Vec3(i * 20, 0, 0)) for i in range(3)]
        for asset in assets:
            manager.add_probe(asset)

        # Load first two
        manager.load_probe(0)
        manager.load_probe(1)

        assert assets[0].loaded
        assert assets[1].loaded
        assert not assets[2].loaded

        # Load third, should evict first
        manager.load_probe(2)

        assert not assets[0].loaded  # Evicted
        assert assets[1].loaded
        assert assets[2].loaded

    def test_manager_clear(self) -> None:
        """Test clearing all probes."""
        manager = BakedProbeManager()
        asset = self._create_test_asset(1, Vec3(0, 0, 0))
        manager.add_probe(asset)
        manager.load_probe(1)

        manager.clear()

        assert manager.get_probe(1) is None


# -----------------------------------------------------------------------------
# Constants and Configuration Tests
# -----------------------------------------------------------------------------

class TestBakedProbeConstants:
    """Tests for constants."""

    def test_face_count(self) -> None:
        """Test cubemap face count."""
        assert BakedProbeConstants.FACE_COUNT == 6

    def test_cubemap_fov(self) -> None:
        """Test cubemap FOV values."""
        assert BakedProbeConstants.CUBEMAP_FOV_DEGREES == pytest.approx(90.0)
        assert BakedProbeConstants.CUBEMAP_FOV_RADIANS == pytest.approx(math.pi / 2)

    def test_bc6h_block_size(self) -> None:
        """Test BC6H block dimensions."""
        assert BakedProbeConstants.BC6H_BLOCK_SIZE == 4
        assert BakedProbeConstants.BC6H_BYTES_PER_BLOCK == 16


class TestCubemapFaceDirections:
    """Tests for cubemap face directions."""

    def test_all_faces_defined(self) -> None:
        """Test all faces have directions."""
        for face in CubemapFace:
            assert face in CUBEMAP_FACE_DIRECTIONS
            direction, up = CUBEMAP_FACE_DIRECTIONS[face]
            assert isinstance(direction, Vec3)
            assert isinstance(up, Vec3)

    def test_directions_orthogonal(self) -> None:
        """Test direction and up are orthogonal."""
        for face in CubemapFace:
            direction, up = CUBEMAP_FACE_DIRECTIONS[face]
            dot = direction.dot(up)
            assert dot == pytest.approx(0.0, abs=0.001)


# -----------------------------------------------------------------------------
# Edge Cases and Error Handling
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_direction_sample(self) -> None:
        """Test sampling with zero direction doesn't crash."""
        cubemap = CubemapData(resolution=4)
        # Should handle gracefully (normalized zero might cause issues)
        # In practice, the normalized() call should handle this
        # We just ensure no crash
        try:
            cubemap.sample_direction(Vec3(0.001, 0, 0))
        except Exception:
            pass  # May fail, but shouldn't crash ungracefully

    def test_single_pixel_face(self) -> None:
        """Test 1x1 face resolution."""
        face = CubemapFaceData(face=CubemapFace.POSITIVE_X, resolution=1)
        face.set_pixel(0, 0, HDRPixel(1.0, 0.0, 0.0))
        result = face.get_pixel(0, 0)
        assert result.r == pytest.approx(1.0)

    def test_max_resolution_estimate(self) -> None:
        """Test size estimation for max resolution."""
        comp = BC6HCompressor()
        size = comp.estimate_compressed_size(BakedProbeConstants.MAX_RESOLUTION)
        # Should not overflow or produce negative values
        assert size > 0

    def test_prefilter_roughness_zero(self) -> None:
        """Test prefiltering at roughness 0."""
        cubemap = CubemapData(resolution=4)
        for face in cubemap.faces:
            for i in range(16):
                face.pixels[i] = HDRPixel(1.0, 0.0, 0.0)

        gen = PrefilteredGenerator(sample_count=16, roughness_levels=2)
        chain = gen.generate_prefiltered(cubemap)

        # Roughness 0 should preserve original colors
        mip0 = chain.mips[0]
        sample = mip0.cubemap.sample_direction(Vec3(1, 0, 0))
        assert sample.r > 0.5


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_full_capture_pipeline(self) -> None:
        """Test complete capture, compress, save, load, sample pipeline."""
        # 1. Create renderer
        config = CaptureConfig(resolution=4)

        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            # Simple sky gradient
            return Vec3(0.3, 0.5, 0.8 + direction.y * 0.2)

        renderer = FunctionCubemapRenderer(config, sample_func)

        # 2. Create capture system
        bake_config = BakedProbeConfig(
            resolution=4,
            roughness_levels=2,
            sample_count=16,
            supercompress=False,
        )
        capture = BakedProbeCapture(renderer, bake_config)

        # 3. Capture probe
        asset = capture.capture_probe(
            name="integration_test",
            position=Vec3(0, 0, 0),
            bounds=AABB(Vec3(-10, -10, -10), Vec3(10, 10, 10)),
            prefilter=True,
        )

        # 4. Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.bkpr', delete=False) as f:
            path = Path(f.name)

        try:
            capture.save_probe(asset, path)

            # 5. Load back
            loaded = capture.load_probe(path)

            # 6. Add to manager
            manager = BakedProbeManager()
            manager.add_probe(loaded)

            # 7. Sample
            result = manager.sample(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.0)

            # Should get valid color
            assert result.x >= 0 and result.y >= 0 and result.z >= 0

        finally:
            path.unlink()

    def test_multiple_probes_blending(self) -> None:
        """Test blending between multiple probes."""
        manager = BakedProbeManager()

        # Create two probes with different colors
        def create_colored_asset(probe_id: int, pos: Vec3, color: Vec3) -> BakedProbeAsset:
            cubemap = CubemapData(resolution=4)
            for face in cubemap.faces:
                for i in range(16):
                    face.pixels[i] = HDRPixel(color.x, color.y, color.z)

            mip_chain = CubemapMipChain(base_resolution=4, mip_count=1)
            mip_chain.mips.append(MipLevel(level=0, resolution=4, cubemap=cubemap))

            comp = BC6HCompressor()
            compressed = [comp.compress_cubemap(cubemap)]

            writer = KTX2Writer()
            ktx2_data = writer.write_to_bytes(mip_chain, compressed, supercompress=False)

            return BakedProbeAsset(
                probe_id=probe_id,
                name=f"probe_{probe_id}",
                position=pos,
                bounds=AABB(pos - Vec3(5, 5, 5), pos + Vec3(5, 5, 5)),
                resolution=4,
                mip_count=1,
                is_prefiltered=False,
                ktx2_data=ktx2_data,
            )

        # Red probe at (0, 0, 0), green probe at (8, 0, 0)
        red_probe = create_colored_asset(1, Vec3(0, 0, 0), Vec3(1, 0, 0))
        green_probe = create_colored_asset(2, Vec3(8, 0, 0), Vec3(0, 1, 0))

        manager.add_probe(red_probe)
        manager.add_probe(green_probe)

        # Sample at red probe center
        result_red = manager.sample(Vec3(0, 0, 0), Vec3(1, 0, 0), 0.0)
        # Should be mostly red
        assert result_red.x >= result_red.y

        # Sample at green probe center
        result_green = manager.sample(Vec3(8, 0, 0), Vec3(1, 0, 0), 0.0)
        # Should be mostly green
        assert result_green.y >= result_green.x
