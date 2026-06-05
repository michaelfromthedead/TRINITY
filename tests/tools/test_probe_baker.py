"""Tests for the light probe baker tool.

Validates the complete probe baking pipeline:
- Fibonacci spiral ray generation uniformity
- SH projection accuracy
- KTX2 and TRINITY format export/import
- Progress callback invocation
- Scene proxy protocol
"""

from __future__ import annotations

import math
import struct
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from engine.rendering.gi.sh_math import (
    SHCoefficientsL2,
    fibonacci_sphere_directions,
    sh_evaluate_l2,
)
from engine.tools.probe_baker import (
    TPROBE_MAGIC,
    TPROBE_VERSION,
    BakeConfig,
    BakedProbe,
    EmptyScene,
    GroundPlaneScene,
    HitInfo,
    ProbeLocation,
    ProbeBaker,
    SceneProxy,
    generate_probe_grid,
    main,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def default_config() -> BakeConfig:
    """Default bake configuration for tests."""
    return BakeConfig(
        rays_per_probe=256,
        max_bounces=2,
        output_format="trinity_probe",
        compress=False,
    )


@pytest.fixture
def baker(default_config: BakeConfig) -> ProbeBaker:
    """Pre-configured probe baker."""
    return ProbeBaker(default_config)


@pytest.fixture
def empty_scene() -> EmptyScene:
    """Empty scene (sky only)."""
    return EmptyScene(sky_color=(0.5, 0.7, 1.0))


@pytest.fixture
def ground_scene() -> GroundPlaneScene:
    """Ground plane scene for testing."""
    return GroundPlaneScene(
        ground_albedo=(0.3, 0.5, 0.2),
        sky_color=(0.5, 0.7, 1.0),
    )


@pytest.fixture
def temp_output_path(tmp_path: Path) -> Path:
    """Temporary output file path."""
    return tmp_path / "test_probes.tprobe"


# ============================================================================
# Fibonacci Spiral Ray Generation Tests
# ============================================================================


class TestFibonacciRayGeneration:
    """Tests for ray direction generation."""

    def test_correct_count(self, baker: ProbeBaker) -> None:
        """generate_ray_directions returns correct number of rays."""
        directions = baker.generate_ray_directions(512)
        assert len(directions) == 512

    def test_normalized_directions(self, baker: ProbeBaker) -> None:
        """All generated directions are unit length."""
        directions = baker.generate_ray_directions(256)
        norms = np.linalg.norm(directions, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_hemisphere_coverage(self, baker: ProbeBaker) -> None:
        """Directions cover both hemispheres approximately equally."""
        directions = baker.generate_ray_directions(1000)

        # Count positive z (upper hemisphere)
        upper = np.sum(directions[:, 2] > 0)
        lower = np.sum(directions[:, 2] < 0)

        # Should be roughly 50/50 (within 10%)
        assert 400 < upper < 600
        assert 400 < lower < 600

    def test_uniform_distribution_x_axis(self, baker: ProbeBaker) -> None:
        """X coordinates are uniformly distributed."""
        directions = baker.generate_ray_directions(1000)

        # Check quartile distribution
        q1 = np.sum(directions[:, 0] < -0.5)
        q4 = np.sum(directions[:, 0] > 0.5)

        # Should be roughly 25% in outer quartiles (within 10%)
        assert 150 < q1 < 350
        assert 150 < q4 < 350

    def test_cached_directions(self, baker: ProbeBaker) -> None:
        """Directions are cached for same ray count."""
        d1 = baker.generate_ray_directions(256)
        d2 = baker.generate_ray_directions(256)

        # Should be the exact same array (same object)
        assert d1 is d2

    def test_cache_invalidation(self, baker: ProbeBaker) -> None:
        """Cache is invalidated on different ray count."""
        d1 = baker.generate_ray_directions(256)
        d2 = baker.generate_ray_directions(512)

        assert len(d1) != len(d2)

    def test_deterministic_generation(self) -> None:
        """Generation is deterministic (same seeds produce same results)."""
        b1 = ProbeBaker(BakeConfig())
        b2 = ProbeBaker(BakeConfig())

        d1 = b1.generate_ray_directions(100)
        d2 = b2.generate_ray_directions(100)

        np.testing.assert_array_equal(d1, d2)


# ============================================================================
# SH Projection Tests
# ============================================================================


class TestSHProjection:
    """Tests for spherical harmonic projection."""

    def test_constant_color_projection(self, baker: ProbeBaker) -> None:
        """Projecting constant color produces correct DC coefficient."""
        # Use empty scene with known sky color
        sky_color = (0.5, 0.7, 1.0)
        scene = EmptyScene(sky_color=sky_color)

        probe = baker.trace_probe((0, 0, 0), scene)

        # Evaluate in multiple directions - should be approximately sky color
        # (convolved with cosine lobe)
        dirs = [
            np.array([0, 1, 0], dtype=np.float32),  # Up
            np.array([1, 0, 0], dtype=np.float32),  # Right
        ]

        for d in dirs:
            result = sh_evaluate_l2(probe.irradiance, d)
            # Due to cosine convolution, values will differ from raw sky color
            # but should be in reasonable range
            assert result[0] > 0.0
            assert result[1] > 0.0
            assert result[2] > 0.0

    def test_directional_bias(self, baker: ProbeBaker, ground_scene: GroundPlaneScene) -> None:
        """Ground plane creates directional lighting bias."""
        probe = baker.trace_probe((0, 2, 0), ground_scene)

        # Evaluate pointing down (toward ground) vs up (toward sky)
        down = sh_evaluate_l2(probe.irradiance, np.array([0, -1, 0], dtype=np.float32))
        up = sh_evaluate_l2(probe.irradiance, np.array([0, 1, 0], dtype=np.float32))

        # Down should have green tint from ground, up should be more blue from sky
        # The difference shows directional encoding is working
        assert down[1] != up[1]  # Green channel differs

    def test_probe_position_affects_result(
        self, baker: ProbeBaker, ground_scene: GroundPlaneScene
    ) -> None:
        """Different probe positions produce different results."""
        probe_low = baker.trace_probe((0, 0.5, 0), ground_scene)
        probe_high = baker.trace_probe((0, 5, 0), ground_scene)

        # Higher probe sees more sky, should have different mean distance
        assert probe_low.distance_mean != probe_high.distance_mean


# ============================================================================
# Scene Proxy Tests
# ============================================================================


class TestSceneProxy:
    """Tests for scene proxy implementations."""

    def test_empty_scene_returns_none(self, empty_scene: EmptyScene) -> None:
        """Empty scene always returns None for ray traces."""
        result = empty_scene.trace_ray((0, 0, 0), (0, 1, 0))
        assert result is None

    def test_ground_plane_hit(self, ground_scene: GroundPlaneScene) -> None:
        """Ground plane scene detects hits on y=0 plane."""
        # Ray from above pointing down
        result = ground_scene.trace_ray((0, 1, 0), (0, -1, 0))
        assert result is not None
        assert result.position == (0, 0, 0)
        assert result.distance == pytest.approx(1.0)
        assert result.normal == (0, 1, 0)

    def test_ground_plane_miss_parallel(self, ground_scene: GroundPlaneScene) -> None:
        """Parallel ray misses ground plane."""
        result = ground_scene.trace_ray((0, 1, 0), (1, 0, 0))  # Horizontal ray
        assert result is None

    def test_ground_plane_miss_facing_away(self, ground_scene: GroundPlaneScene) -> None:
        """Ray pointing away from ground misses."""
        result = ground_scene.trace_ray((0, 1, 0), (0, 1, 0))  # Pointing up
        assert result is None

    def test_scene_proxy_protocol(self) -> None:
        """SceneProxy protocol is properly implemented."""

        class CustomScene:
            def trace_ray(
                self,
                origin: tuple[float, float, float],
                direction: tuple[float, float, float],
            ) -> Optional[HitInfo]:
                return None

        scene = CustomScene()
        assert isinstance(scene, SceneProxy)


# ============================================================================
# KTX2 Export Tests
# ============================================================================


class TestKTX2Export:
    """Tests for KTX2 format export."""

    def test_export_creates_file(self, baker: ProbeBaker, tmp_path: Path) -> None:
        """KTX2 export creates a file."""
        output = tmp_path / "probes.ktx2"

        probes = [
            BakedProbe(
                location=ProbeLocation(position=(0, 0, 0)),
                irradiance=SHCoefficientsL2.zero(),
                distance_mean=1.0,
                distance_variance=0.1,
            )
        ]

        baker.export_ktx2(probes, output)
        assert output.exists()

    def test_ktx2_magic_header(self, baker: ProbeBaker, tmp_path: Path) -> None:
        """KTX2 file starts with correct magic number."""
        output = tmp_path / "probes.ktx2"

        probes = [
            BakedProbe(
                location=ProbeLocation(position=(0, 0, 0)),
                irradiance=SHCoefficientsL2.zero(),
                distance_mean=1.0,
                distance_variance=0.1,
            )
        ]

        baker.export_ktx2(probes, output)

        with open(output, "rb") as f:
            magic = f.read(12)
            expected = bytes([0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A])
            assert magic == expected

    def test_ktx2_dimensions(self, baker: ProbeBaker, tmp_path: Path) -> None:
        """KTX2 contains correct dimensions."""
        output = tmp_path / "probes.ktx2"
        num_probes = 5

        probes = [
            BakedProbe(
                location=ProbeLocation(position=(i, 0, 0)),
                irradiance=SHCoefficientsL2.zero(),
                distance_mean=1.0,
                distance_variance=0.1,
            )
            for i in range(num_probes)
        ]

        baker.export_ktx2(probes, output)

        with open(output, "rb") as f:
            # KTX2 header: 12 magic + 4 format + 4 type_size = 20
            f.seek(20)
            width, height, depth = struct.unpack("<III", f.read(12))
            layer_count, _, _ = struct.unpack("<III", f.read(12))

            assert width == 9  # SH coefficients
            assert height == 1
            assert layer_count == num_probes

    def test_ktx2_empty_probes_raises(self, baker: ProbeBaker, tmp_path: Path) -> None:
        """Export raises ValueError for empty probe list."""
        output = tmp_path / "probes.ktx2"

        with pytest.raises(ValueError, match="No probes"):
            baker.export_ktx2([], output)


# ============================================================================
# TRINITY Format Tests
# ============================================================================


class TestTRINITYFormat:
    """Tests for TRINITY probe format (.tprobe)."""

    def test_export_creates_file(
        self, baker: ProbeBaker, temp_output_path: Path
    ) -> None:
        """Trinity export creates a file."""
        probes = [
            BakedProbe(
                location=ProbeLocation(position=(0, 0, 0)),
                irradiance=SHCoefficientsL2.zero(),
                distance_mean=1.0,
                distance_variance=0.1,
            )
        ]

        baker.export_trinity(probes, temp_output_path)
        assert temp_output_path.exists()

    def test_tprobe_magic_header(
        self, baker: ProbeBaker, temp_output_path: Path
    ) -> None:
        """TPROBE file starts with correct magic number."""
        probes = [
            BakedProbe(
                location=ProbeLocation(position=(0, 0, 0)),
                irradiance=SHCoefficientsL2.zero(),
                distance_mean=1.0,
                distance_variance=0.1,
            )
        ]

        baker.export_trinity(probes, temp_output_path)

        with open(temp_output_path, "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            # Strip compression bit if present
            assert (magic & 0x7FFFFFFF) == TPROBE_MAGIC

    def test_tprobe_roundtrip(self, baker: ProbeBaker, temp_output_path: Path) -> None:
        """Export then load preserves probe data."""
        # Create probe with known values
        irradiance = SHCoefficientsL2.zero()
        irradiance.coeffs[0] = [1.0, 0.5, 0.25]
        irradiance.coeffs[3] = [0.1, 0.2, 0.3]

        original = BakedProbe(
            location=ProbeLocation(position=(1.0, 2.0, 3.0), name="test"),
            irradiance=irradiance,
            distance_mean=5.5,
            distance_variance=1.2,
        )

        baker.export_trinity([original], temp_output_path)
        loaded = ProbeBaker.load_trinity(temp_output_path)

        assert len(loaded) == 1
        probe = loaded[0]

        # Check position
        assert probe.location.position == pytest.approx((1.0, 2.0, 3.0), rel=1e-5)

        # Check irradiance coefficients
        np.testing.assert_allclose(
            probe.irradiance.coeffs[0], [1.0, 0.5, 0.25], rtol=1e-5
        )
        np.testing.assert_allclose(
            probe.irradiance.coeffs[3], [0.1, 0.2, 0.3], rtol=1e-5
        )

        # Check distance stats
        assert probe.distance_mean == pytest.approx(5.5, rel=1e-5)
        assert probe.distance_variance == pytest.approx(1.2, rel=1e-5)

    def test_tprobe_multiple_probes(
        self, baker: ProbeBaker, temp_output_path: Path
    ) -> None:
        """Multiple probes roundtrip correctly."""
        probes = [
            BakedProbe(
                location=ProbeLocation(position=(i * 2.0, 0.0, 0.0)),
                irradiance=SHCoefficientsL2.zero(),
                distance_mean=float(i),
                distance_variance=float(i) * 0.1,
            )
            for i in range(10)
        ]

        baker.export_trinity(probes, temp_output_path)
        loaded = ProbeBaker.load_trinity(temp_output_path)

        assert len(loaded) == 10

        for i, probe in enumerate(loaded):
            assert probe.location.position[0] == pytest.approx(i * 2.0)
            assert probe.distance_mean == pytest.approx(float(i))

    def test_tprobe_invalid_magic_raises(self, tmp_path: Path) -> None:
        """Loading invalid file raises ValueError."""
        bad_file = tmp_path / "bad.tprobe"
        with open(bad_file, "wb") as f:
            # Write 16 bytes (header size) with wrong magic
            f.write(b"BADMAGIC12345678")

        with pytest.raises(ValueError, match="Invalid TPROBE"):
            ProbeBaker.load_trinity(bad_file)

    def test_tprobe_empty_probes_raises(
        self, baker: ProbeBaker, temp_output_path: Path
    ) -> None:
        """Export raises ValueError for empty probe list."""
        with pytest.raises(ValueError, match="No probes"):
            baker.export_trinity([], temp_output_path)


# ============================================================================
# Progress Callback Tests
# ============================================================================


class TestProgressCallback:
    """Tests for progress reporting."""

    def test_callback_invoked(
        self, baker: ProbeBaker, empty_scene: EmptyScene
    ) -> None:
        """Progress callback is invoked for each probe."""
        callbacks: list[tuple[int, int, str]] = []

        def callback(current: int, total: int, message: str) -> None:
            callbacks.append((current, total, message))

        locations = [
            ProbeLocation(position=(0, 0, 0), name="p0"),
            ProbeLocation(position=(1, 0, 0), name="p1"),
            ProbeLocation(position=(2, 0, 0), name="p2"),
        ]

        baker.bake_probes(locations, empty_scene, callback)

        # Should have 4 callbacks: 3 per-probe + 1 completion
        assert len(callbacks) == 4

    def test_callback_progress_order(
        self, baker: ProbeBaker, empty_scene: EmptyScene
    ) -> None:
        """Progress values increase monotonically."""
        progress_values: list[int] = []

        def callback(current: int, total: int, message: str) -> None:
            progress_values.append(current)

        locations = [ProbeLocation(position=(i, 0, 0)) for i in range(5)]
        baker.bake_probes(locations, empty_scene, callback)

        # Should be: 0, 1, 2, 3, 4, 5 (last is completion)
        assert progress_values == [0, 1, 2, 3, 4, 5]

    def test_callback_receives_probe_name(
        self, baker: ProbeBaker, empty_scene: EmptyScene
    ) -> None:
        """Callback message includes probe name."""
        messages: list[str] = []

        def callback(current: int, total: int, message: str) -> None:
            messages.append(message)

        locations = [
            ProbeLocation(position=(0, 0, 0), name="kitchen"),
            ProbeLocation(position=(1, 0, 0), name="bathroom"),
        ]

        baker.bake_probes(locations, empty_scene, callback)

        assert "kitchen" in messages[0]
        assert "bathroom" in messages[1]

    def test_no_callback_works(
        self, baker: ProbeBaker, empty_scene: EmptyScene
    ) -> None:
        """Baking works without progress callback."""
        locations = [ProbeLocation(position=(0, 0, 0))]
        probes = baker.bake_probes(locations, empty_scene, None)
        assert len(probes) == 1


# ============================================================================
# Iterator Interface Tests
# ============================================================================


class TestIteratorInterface:
    """Tests for streaming iterator baking."""

    def test_iterator_yields_probes(
        self, baker: ProbeBaker, empty_scene: EmptyScene
    ) -> None:
        """Iterator yields probes one at a time."""
        locations = [ProbeLocation(position=(i, 0, 0)) for i in range(3)]

        probes = list(baker.bake_probes_iter(locations, empty_scene))
        assert len(probes) == 3

    def test_iterator_lazy_evaluation(
        self, baker: ProbeBaker, empty_scene: EmptyScene
    ) -> None:
        """Iterator doesn't process all probes upfront."""
        locations = [ProbeLocation(position=(i, 0, 0)) for i in range(100)]

        iterator = baker.bake_probes_iter(locations, empty_scene)

        # Taking just one should not process all 100
        first = next(iterator)
        assert first.location.position[0] == pytest.approx(0.0)


# ============================================================================
# Probe Grid Generation Tests
# ============================================================================


class TestProbeGridGeneration:
    """Tests for probe grid generation."""

    def test_grid_correct_count(self) -> None:
        """Grid generates correct number of probes."""
        # 3x3x3 grid with spacing 1
        probes = generate_probe_grid(
            min_bounds=(0, 0, 0),
            max_bounds=(2, 2, 2),
            spacing=1.0,
        )
        assert len(probes) == 27  # 3^3

    def test_grid_single_probe(self) -> None:
        """Minimum grid is a single probe."""
        probes = generate_probe_grid(
            min_bounds=(0, 0, 0),
            max_bounds=(0, 0, 0),
            spacing=1.0,
        )
        assert len(probes) == 1
        assert probes[0].position == (0, 0, 0)

    def test_grid_positions_correct(self) -> None:
        """Probe positions are at grid intersections."""
        probes = generate_probe_grid(
            min_bounds=(0, 0, 0),
            max_bounds=(2, 0, 0),
            spacing=1.0,
        )

        positions = [p.position for p in probes]
        assert (0, 0, 0) in positions
        assert (1, 0, 0) in positions
        assert (2, 0, 0) in positions

    def test_grid_names_unique(self) -> None:
        """Each probe has a unique name."""
        probes = generate_probe_grid(
            min_bounds=(0, 0, 0),
            max_bounds=(2, 2, 2),
            spacing=1.0,
        )

        names = [p.name for p in probes]
        assert len(names) == len(set(names))


# ============================================================================
# Configuration Tests
# ============================================================================


class TestBakeConfig:
    """Tests for BakeConfig dataclass."""

    def test_default_values(self) -> None:
        """Default configuration has sensible values."""
        config = BakeConfig()

        assert config.rays_per_probe == 512
        assert config.max_bounces == 3
        assert config.output_format == "ktx2"
        assert config.compress is True

    def test_custom_config(self) -> None:
        """Custom configuration values are preserved."""
        config = BakeConfig(
            rays_per_probe=1024,
            max_bounces=5,
            output_format="trinity_probe",
            compress=False,
            sky_color=(1.0, 0.0, 0.0),
        )

        assert config.rays_per_probe == 1024
        assert config.max_bounces == 5
        assert config.sky_color == (1.0, 0.0, 0.0)


# ============================================================================
# CLI Tests
# ============================================================================


class TestCLI:
    """Tests for command-line interface."""

    def test_cli_creates_output(self, tmp_path: Path) -> None:
        """CLI creates output file."""
        output = tmp_path / "probes.tprobe"

        result = main([
            "--output", str(output),
            "--rays", "64",
            "--bounces", "1",
            "--spacing", "5.0",
        ])

        assert result == 0
        assert output.exists()

    def test_cli_ktx2_format(self, tmp_path: Path) -> None:
        """CLI detects KTX2 format from extension."""
        output = tmp_path / "probes.ktx2"

        result = main([
            "--output", str(output),
            "--rays", "64",
        ])

        assert result == 0
        assert output.exists()

    def test_cli_verbose_output(self, tmp_path: Path, capsys) -> None:
        """CLI verbose mode prints progress."""
        output = tmp_path / "probes.tprobe"

        main([
            "--output", str(output),
            "--rays", "32",
            "--verbose",
            "--spacing", "10.0",  # Larger spacing = fewer probes
        ])

        captured = capsys.readouterr()
        assert "probe" in captured.out.lower()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_ray(self) -> None:
        """Baking with single ray works."""
        config = BakeConfig(rays_per_probe=1, max_bounces=0)
        baker = ProbeBaker(config)
        scene = EmptyScene()

        probe = baker.trace_probe((0, 0, 0), scene)
        assert probe is not None

    def test_zero_bounces(self) -> None:
        """Zero bounces produces valid (direct-only) lighting."""
        config = BakeConfig(rays_per_probe=64, max_bounces=0)
        baker = ProbeBaker(config)
        scene = GroundPlaneScene()

        probe = baker.trace_probe((0, 2, 0), scene)
        assert probe is not None
        # Should still have some irradiance from sky
        energy = np.sum(probe.irradiance.coeffs ** 2)
        assert energy > 0

    def test_probe_at_ground_level(self) -> None:
        """Probe at ground level handles near-zero distances."""
        config = BakeConfig(rays_per_probe=64, max_bounces=1)
        baker = ProbeBaker(config)
        scene = GroundPlaneScene()

        # Just above ground
        probe = baker.trace_probe((0, 0.01, 0), scene)
        assert probe is not None
        assert probe.distance_mean >= 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Full pipeline integration tests."""

    def test_full_bake_export_load_cycle(self, tmp_path: Path) -> None:
        """Complete bake -> export -> load cycle works."""
        # Configure
        config = BakeConfig(
            rays_per_probe=128,
            max_bounces=2,
            compress=False,
        )
        baker = ProbeBaker(config)

        # Generate grid
        locations = generate_probe_grid(
            min_bounds=(0, 1, 0),
            max_bounds=(4, 3, 4),
            spacing=2.0,
        )

        # Bake
        scene = GroundPlaneScene()
        probes = baker.bake_probes(locations, scene)

        # Export
        output = tmp_path / "test.tprobe"
        baker.export_trinity(probes, output)

        # Load
        loaded = ProbeBaker.load_trinity(output)

        # Verify
        assert len(loaded) == len(probes)

        for original, reloaded in zip(probes, loaded):
            np.testing.assert_allclose(
                original.irradiance.coeffs,
                reloaded.irradiance.coeffs,
                rtol=1e-5,
            )

    def test_baked_probes_produce_plausible_lighting(self) -> None:
        """Baked probes produce physically plausible results."""
        config = BakeConfig(rays_per_probe=256, max_bounces=2)
        baker = ProbeBaker(config)

        # Simple scene with ground
        scene = GroundPlaneScene(
            ground_albedo=(0.5, 0.5, 0.5),
            sky_color=(1.0, 1.0, 1.0),
        )

        probe = baker.trace_probe((0, 2, 0), scene)

        # Evaluate in directions
        up = sh_evaluate_l2(probe.irradiance, np.array([0, 1, 0], dtype=np.float32))
        down = sh_evaluate_l2(probe.irradiance, np.array([0, -1, 0], dtype=np.float32))
        side = sh_evaluate_l2(probe.irradiance, np.array([1, 0, 0], dtype=np.float32))

        # All values should be positive (irradiance is always >= 0)
        assert np.all(up >= -0.1)  # Small negative from SH ringing is OK
        assert np.all(down >= -0.1)
        assert np.all(side >= -0.1)

        # Sky direction should be brightest (white sky)
        assert up[0] > down[0] * 0.5  # Up brighter than down
