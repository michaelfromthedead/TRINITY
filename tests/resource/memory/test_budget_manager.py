"""Tests for BudgetManager."""

import pytest

from engine.resource.memory.budget_manager import (
    AssetCategory,
    BudgetManager,
    DEFAULT_AUDIO_BUDGET,
    DEFAULT_MESH_BUDGET,
    DEFAULT_TEXTURE_BUDGET,
)

_1_MB = 1024 * 1024
_SMALL_BUDGET = 10 * _1_MB


class TestBudgetManager:
    def test_default_budgets(self) -> None:
        mgr = BudgetManager()
        tex = mgr.get_usage(AssetCategory.TEXTURE)
        assert tex.budget_bytes == DEFAULT_TEXTURE_BUDGET
        mesh = mgr.get_usage(AssetCategory.MESH)
        assert mesh.budget_bytes == DEFAULT_MESH_BUDGET
        audio = mgr.get_usage(AssetCategory.AUDIO)
        assert audio.budget_bytes == DEFAULT_AUDIO_BUDGET

    def test_set_budget(self) -> None:
        mgr = BudgetManager()
        mgr.set_budget(AssetCategory.SHADER, _SMALL_BUDGET)
        entry = mgr.get_usage(AssetCategory.SHADER)
        assert entry.budget_bytes == _SMALL_BUDGET
        assert entry.used_bytes == 0

    def test_allocate_within_budget(self) -> None:
        mgr = BudgetManager()
        mgr.set_budget(AssetCategory.ANIMATION, _SMALL_BUDGET)
        assert mgr.allocate(AssetCategory.ANIMATION, _1_MB)
        entry = mgr.get_usage(AssetCategory.ANIMATION)
        assert entry.used_bytes == _1_MB

    def test_allocate_over_budget_returns_false(self) -> None:
        mgr = BudgetManager()
        mgr.set_budget(AssetCategory.ANIMATION, _SMALL_BUDGET)
        result = mgr.allocate(AssetCategory.ANIMATION, _SMALL_BUDGET + 1)
        assert result is False

    def test_free_reduces_usage(self) -> None:
        mgr = BudgetManager()
        mgr.set_budget(AssetCategory.MATERIAL, _SMALL_BUDGET)
        mgr.allocate(AssetCategory.MATERIAL, 5 * _1_MB)
        mgr.free(AssetCategory.MATERIAL, 3 * _1_MB)
        entry = mgr.get_usage(AssetCategory.MATERIAL)
        assert entry.used_bytes == 2 * _1_MB

    def test_is_over_budget(self) -> None:
        mgr = BudgetManager()
        mgr.set_budget(AssetCategory.OTHER, _SMALL_BUDGET)
        # Manually force over-budget by setting used > budget
        entry = mgr.get_usage(AssetCategory.OTHER)
        entry.used_bytes = _SMALL_BUDGET + 1
        assert mgr.is_over_budget(AssetCategory.OTHER)

    def test_pressure_calculation(self) -> None:
        mgr = BudgetManager()
        # Use only texture category
        mgr.set_budget(AssetCategory.TEXTURE, 100)
        mgr.set_budget(AssetCategory.MESH, 100)
        # Remove defaults by overwriting
        mgr.set_budget(AssetCategory.AUDIO, 100)
        mgr.allocate(AssetCategory.TEXTURE, 50)
        mgr.allocate(AssetCategory.MESH, 50)
        # 100 used / 300 total
        pressure = mgr.get_pressure()
        assert abs(pressure - 100.0 / 300.0) < 1e-9

    def test_peak_tracking(self) -> None:
        mgr = BudgetManager()
        mgr.set_budget(AssetCategory.SHADER, _SMALL_BUDGET)
        mgr.allocate(AssetCategory.SHADER, 5 * _1_MB)
        mgr.free(AssetCategory.SHADER, 3 * _1_MB)
        entry = mgr.get_usage(AssetCategory.SHADER)
        assert entry.peak_bytes == 5 * _1_MB
        assert entry.used_bytes == 2 * _1_MB

    def test_get_total_usage_returns_all(self) -> None:
        mgr = BudgetManager()
        total = mgr.get_total_usage()
        assert AssetCategory.TEXTURE in total
        assert AssetCategory.MESH in total

    def test_allocate_unknown_category_returns_false(self) -> None:
        mgr = BudgetManager()
        result = mgr.allocate(AssetCategory.ANIMATION, _1_MB)
        assert result is False
