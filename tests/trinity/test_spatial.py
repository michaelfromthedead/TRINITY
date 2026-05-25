"""
Tests for spatial decorators (spatial.py).

Tests the 2 spatial decorators built on Ops:
    @spatial, @partitioned

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry
from trinity.decorators.spatial import (
    VALID_PARTITION_DIMENSIONS,
    VALID_SPATIAL_STRUCTURES,
    partitioned,
    spatial,
)


# =============================================================================
# @spatial
# =============================================================================


class TestSpatial:
    def test_default_cell_size(self):
        @spatial(structure="grid")
        class Foo:
            pass

        assert Foo._spatial is True
        assert Foo._spatial_structure == "grid"
        assert Foo._spatial_cell_size == 1.0

    def test_custom_cell_size(self):
        @spatial(structure="quadtree", cell_size=2.5)
        class Bar:
            pass

        assert Bar._spatial_structure == "quadtree"
        assert Bar._spatial_cell_size == 2.5

    def test_all_valid_structures(self):
        for s in VALID_SPATIAL_STRUCTURES:

            @spatial(structure=s)
            class C:
                pass

            assert C._spatial_structure == s

    def test_invalid_structure(self):
        with pytest.raises(ValueError, match="invalid structure"):

            @spatial(structure="rtree")
            class Bad:
                pass

    def test_missing_structure(self):
        with pytest.raises(ValueError, match="'structure' parameter is required"):

            @spatial(structure="")
            class Bad:
                pass

    def test_zero_cell_size(self):
        with pytest.raises(ValueError, match="cell_size must be > 0"):

            @spatial(structure="grid", cell_size=0)
            class Bad:
                pass

    def test_negative_cell_size(self):
        with pytest.raises(ValueError, match="cell_size must be > 0"):

            @spatial(structure="grid", cell_size=-1.0)
            class Bad:
                pass

    def test_applied_decorators(self):
        @spatial(structure="grid")
        class C:
            pass

        assert "spatial" in C._applied_decorators

    def test_steps_recorded(self):
        @spatial(structure="grid")
        class C:
            pass

        assert len(C._applied_steps) > 0

    def test_decompose(self):
        steps = decompose(spatial)
        assert len(steps) > 0
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_tags_contain_structure(self):
        @spatial(structure="hash")
        class C:
            pass

        assert C._tags["spatial"] is True
        assert C._tags["spatial_structure"] == "hash"

    def test_tags_contain_cell_size(self):
        @spatial(structure="grid", cell_size=5.0)
        class C:
            pass

        assert C._tags["spatial_cell_size"] == 5.0

    def test_octree_structure(self):
        @spatial(structure="octree", cell_size=0.5)
        class C:
            pass

        assert C._spatial_structure == "octree"
        assert C._spatial_cell_size == 0.5

    def test_small_cell_size(self):
        @spatial(structure="grid", cell_size=0.001)
        class C:
            pass

        assert C._spatial_cell_size == 0.001

    def test_large_cell_size(self):
        @spatial(structure="grid", cell_size=10000.0)
        class C:
            pass

        assert C._spatial_cell_size == 10000.0

    def test_registry_entry(self):
        assert "spatial" in registry._decorators
        spec = registry._decorators["spatial"]
        assert spec.tier == Tier.SPATIAL
        assert spec.target_types == ("class",)


# =============================================================================
# @partitioned
# =============================================================================


class TestPartitioned:
    def test_default_params(self):
        @partitioned()
        class Foo:
            pass

        assert Foo._partitioned is True
        assert Foo._partition_dimensions == 2
        assert Foo._partition_max_entities == 1000

    def test_custom_dimensions_3d(self):
        @partitioned(dimensions=3)
        class Bar:
            pass

        assert Bar._partition_dimensions == 3

    def test_custom_max_entities(self):
        @partitioned(max_entities=500)
        class C:
            pass

        assert C._partition_max_entities == 500

    def test_custom_all_params(self):
        @partitioned(dimensions=3, max_entities=2000)
        class C:
            pass

        assert C._partition_dimensions == 3
        assert C._partition_max_entities == 2000

    def test_invalid_dimensions_1(self):
        with pytest.raises(ValueError, match="invalid dimensions"):

            @partitioned(dimensions=1)
            class Bad:
                pass

    def test_invalid_dimensions_4(self):
        with pytest.raises(ValueError, match="invalid dimensions"):

            @partitioned(dimensions=4)
            class Bad:
                pass

    def test_zero_max_entities(self):
        with pytest.raises(ValueError, match="max_entities must be > 0"):

            @partitioned(max_entities=0)
            class Bad:
                pass

    def test_negative_max_entities(self):
        with pytest.raises(ValueError, match="max_entities must be > 0"):

            @partitioned(max_entities=-10)
            class Bad:
                pass

    def test_applied_decorators(self):
        @partitioned()
        class C:
            pass

        assert "partitioned" in C._applied_decorators

    def test_steps_recorded(self):
        @partitioned()
        class C:
            pass

        assert len(C._applied_steps) > 0

    def test_decompose(self):
        steps = decompose(partitioned)
        assert len(steps) > 0
        ops = [s.op for s in steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_tags_contain_dimensions(self):
        @partitioned(dimensions=3)
        class C:
            pass

        assert C._tags["partitioned"] is True
        assert C._tags["partition_dimensions"] == 3

    def test_tags_contain_max_entities(self):
        @partitioned(max_entities=777)
        class C:
            pass

        assert C._tags["partition_max_entities"] == 777

    def test_registry_entry(self):
        assert "partitioned" in registry._decorators
        spec = registry._decorators["partitioned"]
        assert spec.tier == Tier.SPATIAL
        assert spec.target_types == ("class",)

    def test_min_max_entities(self):
        @partitioned(max_entities=1)
        class C:
            pass

        assert C._partition_max_entities == 1

    def test_large_max_entities(self):
        @partitioned(max_entities=1_000_000)
        class C:
            pass

        assert C._partition_max_entities == 1_000_000
