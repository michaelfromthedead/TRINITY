"""Tests for AssetHandle."""
import pytest

from engine.resource.asset.asset_handle import AssetHandle, AssetState
from engine.resource.constants import NULL_ASSET_INDEX, ASSET_GENERATION_MASK


class TestAssetHandle:
    def test_creation_stores_index_and_generation(self) -> None:
        h = AssetHandle(5, 3, int)
        assert h.index == 5
        assert h.generation == 3

    def test_null_handle_is_invalid(self) -> None:
        h = AssetHandle.null()
        assert not h.is_valid()
        assert h.index == NULL_ASSET_INDEX

    def test_valid_handle(self) -> None:
        h = AssetHandle(0, 0, str)
        assert h.is_valid()

    def test_asset_type_property(self) -> None:
        h = AssetHandle(1, 0, bytes)
        assert h.asset_type is bytes

    def test_asset_type_none(self) -> None:
        h = AssetHandle(1, 0)
        assert h.asset_type is None

    def test_equality(self) -> None:
        a = AssetHandle(3, 7, int)
        b = AssetHandle(3, 7, str)  # type doesn't affect equality
        assert a == b

    def test_inequality_different_index(self) -> None:
        a = AssetHandle(1, 0)
        b = AssetHandle(2, 0)
        assert a != b

    def test_inequality_different_generation(self) -> None:
        a = AssetHandle(1, 0)
        b = AssetHandle(1, 1)
        assert a != b

    def test_hash_consistency(self) -> None:
        a = AssetHandle(10, 5)
        b = AssetHandle(10, 5)
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_from_packed_roundtrip(self) -> None:
        original = AssetHandle(42, 7, float)
        packed = original.id
        restored = AssetHandle.from_packed(packed, float)
        assert restored.index == 42
        assert restored.generation == 7
        assert restored == original

    def test_generation_mask_wraps(self) -> None:
        h = AssetHandle(0, ASSET_GENERATION_MASK + 1)
        assert h.generation == 0  # wraps around

    def test_repr_valid(self) -> None:
        h = AssetHandle(1, 2, int)
        r = repr(h)
        assert "index=1" in r
        assert "gen=2" in r

    def test_repr_null(self) -> None:
        assert "null" in repr(AssetHandle.null())


class TestAssetState:
    def test_all_states_exist(self) -> None:
        expected = {"REQUESTED", "QUEUED", "LOADING", "LOADED", "READY", "FAILED", "UNLOADING", "UNLOADED"}
        actual = {s.name for s in AssetState}
        assert actual == expected

    def test_state_values_unique(self) -> None:
        values = [s.value for s in AssetState]
        assert len(values) == len(set(values))
