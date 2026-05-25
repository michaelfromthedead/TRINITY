"""Tests for DataTableAsset."""
import pytest

from engine.resource.types.data_table_asset import DataTableAsset


def _table(**kw):
    rows = [
        {"id": 1, "name": "sword", "damage": 10},
        {"id": 2, "name": "shield", "damage": 0},
        {"id": 3, "name": "bow", "damage": 8},
    ]
    defaults = dict(
        asset_id=70, name="items", path="/d.json", size_bytes=256,
        columns=["id", "name", "damage"], row_type="Item", rows=rows,
    )
    defaults.update(kw)
    return DataTableAsset(**defaults)


class TestDataTableAsset:
    def test_creation(self):
        t = _table()
        assert t.row_type == "Item"
        assert len(t.columns) == 3

    def test_get_row(self):
        t = _table()
        row = t.get_row(0)
        assert row["name"] == "sword"

    def test_get_row_out_of_range(self):
        t = _table()
        with pytest.raises(IndexError):
            t.get_row(99)

    def test_get_column_values(self):
        t = _table()
        names = t.get_column_values("name")
        assert names == ["sword", "shield", "bow"]

    def test_get_column_unknown(self):
        t = _table()
        with pytest.raises(KeyError):
            t.get_column_values("nonexistent")

    def test_find_rows(self):
        t = _table()
        results = t.find_rows(lambda r: r["damage"] > 5)
        assert len(results) == 2
        assert all(r["damage"] > 5 for r in results)

    def test_load_unload(self):
        t = _table()
        assert not t.is_loaded()
        t.load(b"")
        assert t.is_loaded()
