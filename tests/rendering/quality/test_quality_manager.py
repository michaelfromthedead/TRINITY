"""Tests for QualityManager (T-CC-0.3)."""

import pytest

from trinity.types import QualityTier

from engine.rendering.quality.capability_scorer import (
    AdapterInfo,
    FeatureFlags,
    GPUBackend,
    GPUDeviceType,
    GPULimits,
)
from engine.rendering.quality.quality_manager import (
    QualityManager,
    QualityManagerConfig,
)


class TestQualityManagerBasic:
    """Test basic QualityManager functionality."""

    def test_default_tier(self):
        """Test default tier without adapter info."""
        manager = QualityManager()
        assert manager.current_tier == QualityTier.HIGH

    def test_custom_default_tier(self):
        """Test custom default tier via config."""
        config = QualityManagerConfig(default_tier=QualityTier.MEDIUM)
        manager = QualityManager(config=config)
        assert manager.current_tier == QualityTier.MEDIUM

    def test_base_tier_matches_current(self):
        """Test base tier matches current initially."""
        manager = QualityManager()
        assert manager.base_tier == manager.current_tier


class TestQualityManagerTierSelection:
    """Test tier selection from adapter info."""

    def test_high_end_gpu_selects_ultra(self):
        """Test high-end GPU selects ULTRA tier."""
        info = AdapterInfo(
            name="RTX 4090",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
            features=FeatureFlags(ray_tracing=True, bindless=True, mesh_shader=True),
            limits=GPULimits(
                max_texture_dimension_2d=16384,
                max_compute_invocations_per_workgroup=1024,
                max_storage_buffer_binding_size=2147483648,
            ),
        )
        manager = QualityManager(adapter_info=info)
        assert manager.current_tier == QualityTier.ULTRA

    def test_integrated_gpu_selects_medium(self):
        """Test integrated GPU selects MEDIUM tier."""
        info = AdapterInfo(
            name="Intel UHD",
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.INTEGRATED,
            features=FeatureFlags(ray_tracing=False),
            limits=GPULimits(
                max_texture_dimension_2d=4096,
                max_storage_buffer_binding_size=268435456,
            ),
        )
        manager = QualityManager(adapter_info=info)
        assert manager.current_tier in (QualityTier.MEDIUM, QualityTier.HIGH)

    def test_mobile_gpu_selects_low(self):
        """Test mobile GPU selects LOW tier."""
        info = AdapterInfo(
            name="Adreno 650",
            backend=GPUBackend.OPENGLES,
            device_type=GPUDeviceType.INTEGRATED,
            features=FeatureFlags(
                ray_tracing=False,
                compute_shader=True,
                texture_compression_etc2=True,
            ),
            limits=GPULimits(
                max_texture_dimension_2d=4096,
                max_storage_buffer_binding_size=67108864,
            ),
        )
        manager = QualityManager(adapter_info=info)
        assert manager.current_tier in (QualityTier.LOW, QualityTier.MEDIUM)


class TestQualityManagerOverrides:
    """Test per-subsystem tier overrides."""

    def test_get_tier_without_override(self):
        """Test get_tier returns global tier without override."""
        manager = QualityManager()
        manager.set_tier(QualityTier.HIGH)
        assert manager.get_tier("lighting") == QualityTier.HIGH

    def test_set_override(self):
        """Test setting subsystem override."""
        manager = QualityManager()
        manager.set_tier(QualityTier.HIGH)
        manager.set_override("lighting", QualityTier.MEDIUM)
        assert manager.get_tier("lighting") == QualityTier.MEDIUM

    def test_override_does_not_affect_others(self):
        """Test override only affects that subsystem."""
        manager = QualityManager()
        manager.set_tier(QualityTier.HIGH)
        manager.set_override("lighting", QualityTier.MEDIUM)
        assert manager.get_tier("shadows") == QualityTier.HIGH

    def test_clear_override(self):
        """Test clearing subsystem override."""
        manager = QualityManager()
        manager.set_tier(QualityTier.HIGH)
        manager.set_override("lighting", QualityTier.LOW)
        manager.clear_override("lighting")
        assert manager.get_tier("lighting") == QualityTier.HIGH

    def test_locked_override(self):
        """Test locked override prevents auto-adjustment."""
        manager = QualityManager()
        manager.set_override("lighting", QualityTier.LOW, locked=True)
        assert manager.is_locked("lighting") is True

    def test_unlocked_by_default(self):
        """Test subsystems are not locked by default."""
        manager = QualityManager()
        assert manager.is_locked("lighting") is False


class TestQualityManagerManualTier:
    """Test manual tier setting."""

    def test_set_tier(self):
        """Test manually setting tier."""
        manager = QualityManager()
        manager.set_tier(QualityTier.LOW)
        assert manager.current_tier == QualityTier.LOW

    def test_set_tier_updates_base(self):
        """Test setting tier updates base tier."""
        manager = QualityManager()
        manager.set_tier(QualityTier.MEDIUM)
        assert manager.base_tier == QualityTier.MEDIUM

    def test_set_tier_notifies_listeners(self):
        """Test setting tier notifies listeners."""
        manager = QualityManager()
        manager.set_tier(QualityTier.HIGH)

        changes = []

        def on_change(old_tier, new_tier):
            changes.append((old_tier, new_tier))

        manager.add_listener(on_change)
        manager.set_tier(QualityTier.MEDIUM)

        assert len(changes) == 1
        assert changes[0] == (QualityTier.HIGH, QualityTier.MEDIUM)


class TestQualityManagerDynamicAdjustment:
    """Test dynamic tier adjustment based on frame budget."""

    def test_no_adjustment_within_budget(self):
        """Test no adjustment when within budget."""
        manager = QualityManager()
        manager.set_tier(QualityTier.HIGH)

        for _ in range(100):
            manager.record_frame_time(15.0)  # Within 16.67ms budget

        assert manager.current_tier == QualityTier.HIGH

    def test_downgrade_on_budget_violation(self):
        """Test downgrade after sustained budget violation."""
        config = QualityManagerConfig(
            budget_violation_threshold=10,
            frame_budget_ms=16.67,
        )
        manager = QualityManager(config=config)
        manager.set_tier(QualityTier.HIGH)

        # Simulate 15 frames at 25ms (50% over budget)
        for _ in range(15):
            manager.record_frame_time(25.0)

        assert manager.current_tier == QualityTier.MEDIUM

    def test_no_downgrade_below_low(self):
        """Test cannot downgrade below LOW tier."""
        config = QualityManagerConfig(budget_violation_threshold=5)
        manager = QualityManager(config=config)
        manager.set_tier(QualityTier.LOW)

        # Many frames over budget
        for _ in range(20):
            manager.record_frame_time(50.0)

        assert manager.current_tier == QualityTier.LOW

    def test_upgrade_on_budget_recovery(self):
        """Test upgrade after sustained under-budget frames."""
        config = QualityManagerConfig(
            budget_violation_threshold=10,
            budget_recovery_threshold=10,
        )
        manager = QualityManager(config=config)
        manager.set_tier(QualityTier.HIGH)

        # Downgrade first (10 frames over threshold)
        for _ in range(12):
            manager.record_frame_time(25.0)
        assert manager.current_tier == QualityTier.MEDIUM

        # Then recover
        for _ in range(15):
            manager.record_frame_time(8.0)  # Well under budget

        assert manager.current_tier == QualityTier.HIGH

    def test_no_upgrade_above_base(self):
        """Test cannot upgrade above base tier."""
        config = QualityManagerConfig(budget_recovery_threshold=5)
        manager = QualityManager()
        manager.set_tier(QualityTier.MEDIUM)

        # Many frames well under budget
        for _ in range(20):
            manager.record_frame_time(5.0)

        # Should stay at MEDIUM (base tier)
        assert manager.current_tier == QualityTier.MEDIUM

    def test_auto_adjust_disabled(self):
        """Test auto-adjustment can be disabled."""
        config = QualityManagerConfig(
            auto_adjust=False,
            budget_violation_threshold=5,
        )
        manager = QualityManager(config=config)
        manager.set_tier(QualityTier.HIGH)

        # Many frames over budget
        for _ in range(20):
            manager.record_frame_time(50.0)

        # Should not downgrade
        assert manager.current_tier == QualityTier.HIGH


class TestQualityManagerListeners:
    """Test tier change listener functionality."""

    def test_add_listener(self):
        """Test adding a listener."""
        manager = QualityManager()
        manager.set_tier(QualityTier.HIGH)

        called = [False]

        def on_change(old_tier, new_tier):
            called[0] = True

        manager.add_listener(on_change)
        manager.set_tier(QualityTier.LOW)
        assert called[0] is True

    def test_remove_listener(self):
        """Test removing a listener."""
        manager = QualityManager()
        manager.set_tier(QualityTier.HIGH)

        called = [False]

        def on_change(old_tier, new_tier):
            called[0] = True

        manager.add_listener(on_change)
        manager.remove_listener(on_change)
        manager.set_tier(QualityTier.LOW)
        assert called[0] is False

    def test_multiple_listeners(self):
        """Test multiple listeners all called."""
        manager = QualityManager()
        manager.set_tier(QualityTier.HIGH)

        calls = []

        def listener1(old_tier, new_tier):
            calls.append("listener1")

        def listener2(old_tier, new_tier):
            calls.append("listener2")

        manager.add_listener(listener1)
        manager.add_listener(listener2)
        manager.set_tier(QualityTier.LOW)

        assert "listener1" in calls
        assert "listener2" in calls

    def test_no_notification_on_same_tier(self):
        """Test no notification when tier unchanged."""
        manager = QualityManager()
        manager.set_tier(QualityTier.HIGH)

        called = [False]

        def on_change(old_tier, new_tier):
            called[0] = True

        manager.add_listener(on_change)
        manager.set_tier(QualityTier.HIGH)  # Same tier
        assert called[0] is False


class TestQualityManagerScoring:
    """Test capability score access."""

    def test_capability_score_without_adapter(self):
        """Test default score without adapter info."""
        manager = QualityManager()
        assert manager.capability_score == 0.5

    def test_capability_score_with_adapter(self):
        """Test score with adapter info."""
        info = AdapterInfo(
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
        )
        manager = QualityManager(adapter_info=info)
        assert 0.0 <= manager.capability_score <= 1.0

    def test_explain_score_without_adapter(self):
        """Test explain without adapter."""
        manager = QualityManager()
        breakdown = manager.explain_score()
        assert "total" in breakdown

    def test_explain_score_with_adapter(self):
        """Test explain with adapter info."""
        info = AdapterInfo(
            backend=GPUBackend.VULKAN,
            device_type=GPUDeviceType.DISCRETE,
        )
        manager = QualityManager(adapter_info=info)
        breakdown = manager.explain_score()
        assert "device_type" in breakdown
        assert "features" in breakdown
        assert "limits" in breakdown
        assert "backend" in breakdown
