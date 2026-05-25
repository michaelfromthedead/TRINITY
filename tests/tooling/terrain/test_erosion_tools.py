"""Tests for erosion simulation tools."""

import pytest
from engine.tooling.terrain.erosion_tools import (
    ErosionType,
    ErosionParams,
    HydraulicErosionParams,
    ThermalErosionParams,
    ErosionBrush,
    WaterDroplet,
    ErosionSimulator,
)


class TestErosionParams:
    """Tests for erosion parameters."""

    def test_base_params(self):
        """Test base erosion parameters."""
        params = ErosionParams()
        assert params.iterations == 50000
        assert params.seed == 42
        assert params.brush_radius == 3

    def test_hydraulic_params(self):
        """Test hydraulic erosion parameters."""
        params = HydraulicErosionParams()
        assert params.inertia == 0.05
        assert params.sediment_capacity_factor == 4.0
        assert params.erosion_speed == 0.3
        assert params.max_droplet_lifetime == 30

    def test_thermal_params(self):
        """Test thermal erosion parameters."""
        params = ThermalErosionParams()
        assert params.talus_angle == 0.5
        assert params.erosion_rate == 0.5
        assert params.cell_size == 1.0

    def test_custom_hydraulic_params(self):
        """Test custom hydraulic parameters."""
        params = HydraulicErosionParams(
            iterations=10000,
            erosion_speed=0.5,
            deposition_speed=0.5,
        )
        assert params.iterations == 10000
        assert params.erosion_speed == 0.5


class TestWaterDroplet:
    """Tests for water droplet."""

    def test_creation(self):
        """Test droplet creation."""
        droplet = WaterDroplet(x=10.0, y=10.0)
        assert droplet.x == 10.0
        assert droplet.y == 10.0
        assert droplet.water == 1.0
        assert droplet.sediment == 0.0

    def test_initial_values(self):
        """Test initial droplet values."""
        droplet = WaterDroplet(x=0.0, y=0.0, speed=2.0, water=0.5)
        assert droplet.speed == 2.0
        assert droplet.water == 0.5


class TestErosionSimulator:
    """Tests for erosion simulator."""

    def setup_method(self):
        """Set up test terrain."""
        # Create 32x32 terrain with a hill
        self.heights = []
        for y in range(32):
            row = []
            for x in range(32):
                # Create a cone in the center
                dx = x - 16
                dy = y - 16
                dist = (dx * dx + dy * dy) ** 0.5
                height = max(0.0, 1.0 - dist / 16.0)
                row.append(height)
            self.heights.append(row)

        self.simulator = ErosionSimulator(32, 32, self.heights)

    def test_creation(self):
        """Test simulator creation."""
        assert self.simulator.width == 32
        assert self.simulator.height == 32
        assert len(self.simulator.heights) == 32

    def test_heights_copy(self):
        """Test heights are copied."""
        original = self.heights[16][16]
        self.simulator.heights[16][16] = 0.0
        assert self.heights[16][16] == original

    def test_reset(self):
        """Test reset functionality."""
        self.simulator.heights[16][16] = 0.0
        self.simulator.reset(self.heights)
        assert self.simulator.heights[16][16] == self.heights[16][16]

    def test_hydraulic_erosion_basic(self):
        """Test basic hydraulic erosion."""
        # Run minimal erosion
        params = HydraulicErosionParams(iterations=100, seed=42)
        self.simulator.simulate_hydraulic(params)

        # Terrain should be modified
        total_diff = 0.0
        for y in range(32):
            for x in range(32):
                total_diff += abs(self.heights[y][x] - self.simulator.heights[y][x])

        assert total_diff > 0  # Some change occurred

    def test_hydraulic_erosion_reduces_peaks(self):
        """Test that hydraulic erosion reduces peaks."""
        original_peak = self.heights[16][16]

        params = HydraulicErosionParams(iterations=1000, seed=42)
        self.simulator.simulate_hydraulic(params)

        # Peak should be lower
        assert self.simulator.heights[16][16] < original_peak

    def test_thermal_erosion_basic(self):
        """Test basic thermal erosion."""
        # Use a steep terrain with low talus angle for guaranteed erosion
        steep_heights = [[0.0 for _ in range(32)] for _ in range(32)]
        # Create steep cone in center
        for y in range(32):
            for x in range(32):
                dx = x - 16
                dy = y - 16
                dist = (dx * dx + dy * dy) ** 0.5
                steep_heights[y][x] = max(0.0, 5.0 - dist / 3.0)  # Steeper slope

        sim = ErosionSimulator(32, 32, steep_heights)
        original = [row[:] for row in steep_heights]

        # Low talus angle means more erosion
        params = ThermalErosionParams(iterations=50, seed=42, talus_angle=0.1)
        sim.simulate_thermal(params)

        # Terrain should be modified
        total_diff = 0.0
        for y in range(32):
            for x in range(32):
                total_diff += abs(original[y][x] - sim.heights[y][x])

        assert total_diff > 0

    def test_thermal_erosion_smooths_slopes(self):
        """Test that thermal erosion smooths steep slopes."""
        # Create a very steep terrain
        steep_heights = [[0.0 for _ in range(32)] for _ in range(32)]
        steep_heights[16][16] = 10.0  # Very tall spike

        sim = ErosionSimulator(32, 32, steep_heights)
        params = ThermalErosionParams(iterations=100, talus_angle=0.1)
        sim.simulate_thermal(params)

        # Spike should be reduced
        assert sim.heights[16][16] < 10.0

    def test_combined_erosion(self):
        """Test combined erosion simulation."""
        original_heights = [row[:] for row in self.heights]

        self.simulator.simulate_combined(
            hydraulic_params=HydraulicErosionParams(iterations=100),
            thermal_params=ThermalErosionParams(iterations=10),
            hydraulic_weight=0.5,
        )

        # Terrain should be modified
        total_diff = 0.0
        for y in range(32):
            for x in range(32):
                total_diff += abs(original_heights[y][x] - self.simulator.heights[y][x])

        assert total_diff > 0

    def test_erosion_map(self):
        """Test erosion map calculation."""
        original = [row[:] for row in self.heights]

        params = HydraulicErosionParams(iterations=100)
        self.simulator.simulate_hydraulic(params)

        erosion_map = self.simulator.get_erosion_map(original)

        assert len(erosion_map) == 32
        assert len(erosion_map[0]) == 32

    def test_progress_callback(self):
        """Test progress callback."""
        progress_values = []

        def callback(progress):
            progress_values.append(progress)

        params = HydraulicErosionParams(iterations=1000)
        self.simulator.simulate_hydraulic(params, progress_callback=callback)

        assert len(progress_values) > 0
        assert progress_values[-1] == 1.0

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        heights1 = [row[:] for row in self.heights]
        sim1 = ErosionSimulator(32, 32, heights1)
        sim1.simulate_hydraulic(HydraulicErosionParams(iterations=100, seed=42))

        heights2 = [row[:] for row in self.heights]
        sim2 = ErosionSimulator(32, 32, heights2)
        sim2.simulate_hydraulic(HydraulicErosionParams(iterations=100, seed=42))

        # Results should be identical
        for y in range(32):
            for x in range(32):
                assert sim1.heights[y][x] == sim2.heights[y][x]

    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        heights1 = [row[:] for row in self.heights]
        sim1 = ErosionSimulator(32, 32, heights1)
        sim1.simulate_hydraulic(HydraulicErosionParams(iterations=100, seed=42))

        heights2 = [row[:] for row in self.heights]
        sim2 = ErosionSimulator(32, 32, heights2)
        sim2.simulate_hydraulic(HydraulicErosionParams(iterations=100, seed=123))

        # Results should differ
        different = False
        for y in range(32):
            for x in range(32):
                if sim1.heights[y][x] != sim2.heights[y][x]:
                    different = True
                    break

        assert different
