"""Tests for LOD & Streaming decorators (Tier 24)."""

import pytest

from trinity.decorators.lod_streaming import (
    VALID_STREAM_PRIORITIES,
    chunk,
    lod,
    streamable,
)
from trinity.decorators.registry import Tier, registry


# =============================================================================
# @lod
# =============================================================================


class TestLod:
    """Tests for the @lod decorator."""

    def test_basic_defaults(self):
        @lod()
        class Terrain:
            pass

        assert Terrain._lod is True
        assert Terrain._lod_levels == 4
        assert Terrain._lod_distances is None
        assert Terrain._lod_bias == 0.0

    def test_custom_levels(self):
        @lod(levels=6)
        class Mesh:
            pass

        assert Mesh._lod_levels == 6

    def test_custom_distances(self):
        @lod(levels=3, distances=[10.0, 50.0, 200.0])
        class Tree:
            pass

        assert Tree._lod_distances == [10.0, 50.0, 200.0]
        assert Tree._lod_levels == 3

    def test_custom_bias(self):
        @lod(bias=1.5)
        class Rock:
            pass

        assert Rock._lod_bias == 1.5

    def test_all_params(self):
        @lod(levels=2, distances=[5.0, 100.0], bias=-0.5)
        class Building:
            pass

        assert Building._lod_levels == 2
        assert Building._lod_distances == [5.0, 100.0]
        assert Building._lod_bias == -0.5

    def test_tags_set(self):
        @lod(levels=3)
        class Item:
            pass

        assert Item._tags["lod"] is True
        assert Item._tags["lod_levels"] == 3

    def test_applied_decorators(self):
        @lod()
        class Obj:
            pass

        assert "lod" in Obj._applied_decorators

    def test_registered(self):
        @lod()
        class Reg:
            pass

        assert "lod_streaming" in Reg._registries

    # Validation tests

    def test_levels_zero(self):
        with pytest.raises(ValueError, match="levels"):
            lod(levels=0)

    def test_levels_negative(self):
        with pytest.raises(ValueError, match="levels"):
            lod(levels=-1)

    def test_distances_wrong_length(self):
        with pytest.raises(ValueError, match="length"):
            lod(levels=3, distances=[10.0, 50.0])

    def test_distances_negative(self):
        with pytest.raises(ValueError, match="distances must be > 0"):
            lod(levels=2, distances=[-1.0, 10.0])

    def test_distances_not_ascending(self):
        with pytest.raises(ValueError, match="ascending"):
            lod(levels=3, distances=[10.0, 5.0, 20.0])

    def test_distances_equal_values(self):
        with pytest.raises(ValueError, match="ascending"):
            lod(levels=2, distances=[10.0, 10.0])

    def test_no_args_on_class(self):
        @lod
        class Auto:
            pass

        assert Auto._lod is True
        assert Auto._lod_levels == 4


# =============================================================================
# @streamable
# =============================================================================


class TestStreamable:
    """Tests for the @streamable decorator."""

    def test_basic_defaults(self):
        @streamable()
        class Texture:
            pass

        assert Texture._streamable is True
        assert Texture._stream_priority == "normal"
        assert Texture._stream_keep_loaded is False

    def test_custom_priority(self):
        @streamable(priority="critical")
        class Font:
            pass

        assert Font._stream_priority == "critical"

    def test_keep_loaded(self):
        @streamable(keep_loaded=True)
        class UI:
            pass

        assert UI._stream_keep_loaded is True

    def test_all_valid_priorities(self):
        for p in VALID_STREAM_PRIORITIES:

            @streamable(priority=p)
            class C:
                pass

            assert C._stream_priority == p

    def test_invalid_priority(self):
        with pytest.raises(ValueError, match="priority"):
            streamable(priority="urgent")

    def test_tags(self):
        @streamable(priority="high")
        class S:
            pass

        assert S._tags["streamable"] is True
        assert S._tags["stream_priority"] == "high"

    def test_applied_decorators(self):
        @streamable()
        class A:
            pass

        assert "streamable" in A._applied_decorators

    def test_no_args(self):
        @streamable
        class Auto:
            pass

        assert Auto._streamable is True
        assert Auto._stream_priority == "normal"


# =============================================================================
# @chunk
# =============================================================================


class TestChunk:
    """Tests for the @chunk decorator."""

    def test_basic(self):
        @chunk(size=(100.0, 100.0, 100.0))
        class World:
            pass

        assert World._chunk is True
        assert World._chunk_size == (100.0, 100.0, 100.0)
        assert World._chunk_overlap == 0.0

    def test_custom_overlap(self):
        @chunk(size=(50.0, 50.0, 50.0), overlap=5.0)
        class Terrain:
            pass

        assert Terrain._chunk_overlap == 5.0

    def test_size_stored_as_tuple(self):
        @chunk(size=[10.0, 20.0, 30.0])
        class C:
            pass

        assert isinstance(C._chunk_size, tuple)
        assert C._chunk_size == (10.0, 20.0, 30.0)

    def test_tags(self):
        @chunk(size=(1.0, 2.0, 3.0))
        class T:
            pass

        assert T._tags["chunk"] is True
        assert T._tags["chunk_size"] == (1.0, 2.0, 3.0)

    def test_applied_decorators(self):
        @chunk(size=(1.0, 1.0, 1.0))
        class A:
            pass

        assert "chunk" in A._applied_decorators

    # Validation tests

    def test_size_required(self):
        with pytest.raises(ValueError, match="required"):
            chunk()

    def test_size_wrong_length(self):
        with pytest.raises(ValueError, match="3 floats"):
            chunk(size=(1.0, 2.0))

    def test_size_negative(self):
        with pytest.raises(ValueError, match="> 0"):
            chunk(size=(1.0, -1.0, 1.0))

    def test_size_zero(self):
        with pytest.raises(ValueError, match="> 0"):
            chunk(size=(1.0, 0.0, 1.0))

    def test_overlap_negative(self):
        with pytest.raises(ValueError, match="overlap"):
            chunk(size=(1.0, 1.0, 1.0), overlap=-1.0)


# =============================================================================
# Registry
# =============================================================================


class TestLodStreamingRegistry:
    """Registry tests for Tier 24 decorators."""

    def test_lod_registered(self):
        spec = registry.get("lod")
        assert spec is not None
        assert spec.tier == Tier.LOD_STREAMING

    def test_streamable_registered(self):
        spec = registry.get("streamable")
        assert spec is not None
        assert spec.tier == Tier.LOD_STREAMING

    def test_chunk_registered(self):
        spec = registry.get("chunk")
        assert spec is not None
        assert spec.tier == Tier.LOD_STREAMING

    def test_all_in_tier(self):
        specs = registry.by_tier(Tier.LOD_STREAMING)
        names = {s.name for s in specs}
        assert {"lod", "streamable", "chunk"} <= names

    def test_target_types(self):
        assert registry.get("lod").target_types == ("class",)
        assert registry.get("streamable").target_types == ("class",)
        assert registry.get("chunk").target_types == ("class",)
