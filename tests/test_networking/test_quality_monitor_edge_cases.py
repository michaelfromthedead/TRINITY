"""
White-box tests for quality monitor edge cases.

Tests QualityLevel, QualityMetrics, QualityMonitor, NetworkQualityAdapter,
and AdaptiveSettings edge cases including RTT sampling, loss calculation,
and adaptation hysteresis.
"""

from __future__ import annotations

import time
from unittest import mock
import pytest

from engine.networking.transport.quality import (
    QualityLevel, QualityMetrics, QualityMonitor,
    NetworkQualityAdapter, AdaptiveSettings,
)
from engine.networking.config import DEFAULT_CONFIG


class TestQualityLevel:
    """QualityLevel enum tests."""

    def test_has_correct_values(self):
        assert QualityLevel.CRITICAL == 0
        assert QualityLevel.POOR == 1
        assert QualityLevel.FAIR == 2
        assert QualityLevel.GOOD == 3
        assert QualityLevel.EXCELLENT == 4

    def test_can_access_by_name(self):
        assert QualityLevel['CRITICAL'] == QualityLevel.CRITICAL
        assert QualityLevel['EXCELLENT'] == QualityLevel.EXCELLENT


class TestQualityMetrics:
    """QualityMetrics edge case tests."""

    def test_defaults_all_zero(self):
        m = QualityMetrics()
        assert m.rtt == 0.0
        assert m.rtt_variance == 0.0
        assert m.jitter == 0.0
        assert m.packet_loss == 0.0
        assert m.bandwidth_up == 0.0
        assert m.bandwidth_down == 0.0

    def test_quality_level_property(self):
        m = QualityMetrics()
        assert m.quality_level == QualityLevel.EXCELLENT

    def test_quality_level_excellent_at_low_rtt(self):
        m = QualityMetrics(rtt=0.01, packet_loss=0.0)
        assert m.quality_level == QualityLevel.EXCELLENT

    def test_quality_level_good_at_moderate_rtt(self):
        m = QualityMetrics(rtt=0.06, packet_loss=0.0)
        assert m.quality_level == QualityLevel.GOOD

    def test_quality_level_fair_at_higher_rtt(self):
        m = QualityMetrics(rtt=0.12, packet_loss=0.0)
        assert m.quality_level == QualityLevel.FAIR

    def test_to_dict_contains_rtt(self):
        m = QualityMetrics(rtt=0.05)
        d = m.to_dict()
        assert 'rtt' in d
        assert d['rtt'] == 0.05

    def test_to_dict_contains_quality_level(self):
        m = QualityMetrics()
        d = m.to_dict()
        assert 'quality_level' in d


class TestQualityMonitor:
    """QualityMonitor edge case tests."""

    def test_initial_metrics_exist(self):
        monitor = QualityMonitor()
        metrics = monitor.get_metrics()
        assert isinstance(metrics, QualityMetrics)

    def test_initial_quality_excellent(self):
        monitor = QualityMonitor()
        assert monitor.get_quality_level() == QualityLevel.EXCELLENT

    def test_record_packet_sent(self):
        monitor = QualityMonitor()
        monitor.record_packet_sent(100)
        assert monitor._packets_sent == 1

    def test_record_packet_received(self):
        monitor = QualityMonitor()
        monitor.record_packet_received(100)
        assert monitor._packets_received == 1

    def test_record_packet_lost(self):
        monitor = QualityMonitor()
        monitor.record_packet_lost()
        assert monitor._packets_lost == 1

    def test_update_returns_quality_metrics(self):
        monitor = QualityMonitor()
        result = monitor.update()
        assert isinstance(result, QualityMetrics)

    def test_add_rtt_sample_updates_rtt_current(self):
        monitor = QualityMonitor()
        monitor.add_rtt_sample(0.05)
        stats = monitor.get_statistics()
        assert stats['rtt_current'] == 0.05

    def test_get_metrics_returns_current(self):
        monitor = QualityMonitor()
        metrics = monitor.get_metrics()
        assert isinstance(metrics, QualityMetrics)

    def test_get_statistics_keys_present(self):
        monitor = QualityMonitor()
        monitor.add_rtt_sample(0.05)
        stats = monitor.get_statistics()
        assert 'rtt_current' in stats
        assert 'rtt_min' in stats
        assert 'rtt_max' in stats
        assert 'rtt_avg' in stats
        assert 'jitter' in stats
        assert 'packet_loss' in stats
        assert 'packets_sent' in stats
        assert 'packets_received' in stats
        assert 'packets_lost' in stats
        assert 'bandwidth_up' in stats
        assert 'bandwidth_down' in stats

    def test_on_quality_change_callback(self):
        monitor = QualityMonitor()
        callbacks = []
        monitor.on_quality_change(lambda old, new: callbacks.append((old, new)))
        monitor._rtt_estimate = 0.5
        monitor._packets_lost = 50
        monitor._loss_window_sent = 100
        monitor._loss_window_received = 50
        monitor.update()
        assert len(callbacks) >= 1

    def test_update_with_loss_calculates_correctly(self):
        monitor = QualityMonitor()
        for _ in range(10):
            monitor.record_packet_sent()
        for _ in range(7):
            monitor.record_packet_received()
        metrics = monitor.update()
        assert metrics.packet_loss == pytest.approx(0.3, abs=0.01)

    def test_reset_clears_state(self):
        monitor = QualityMonitor()
        monitor.add_rtt_sample(0.05)
        monitor.record_packet_sent()
        monitor.reset()
        assert monitor._packets_sent == 0
        assert monitor._rtt_estimate == 0.0

    def test_multiple_rtt_samples_tracked(self):
        monitor = QualityMonitor()
        for rtt in [0.05, 0.06, 0.04]:
            monitor.add_rtt_sample(rtt)
        assert len(monitor._rtt_samples) == 3


class TestNetworkQualityAdapter:
    """NetworkQualityAdapter edge case tests."""

    def test_initial_level_good(self):
        adapter = NetworkQualityAdapter()
        assert adapter.current_level == QualityLevel.GOOD

    def test_adapt_returns_adaptive_settings(self):
        adapter = NetworkQualityAdapter()
        metrics = QualityMetrics(rtt=0.05)
        result = adapter.adapt(metrics)
        assert isinstance(result, AdaptiveSettings)

    def test_adapt_with_good_metrics(self):
        adapter = NetworkQualityAdapter()
        metrics = QualityMetrics(rtt=0.05, packet_loss=0.0)
        result = adapter.adapt(metrics)
        assert result is not None
        assert hasattr(result, 'update_rate')

    def test_adapt_with_poor_metrics(self):
        adapter = NetworkQualityAdapter()
        metrics = QualityMetrics(rtt=0.3, packet_loss=0.08)
        result = adapter.adapt(metrics)
        assert result is not None

    def test_force_level_changes_level(self):
        adapter = NetworkQualityAdapter()
        adapter.force_level(QualityLevel.CRITICAL)
        assert adapter.current_level == QualityLevel.CRITICAL

    def test_reset_returns_to_good(self):
        adapter = NetworkQualityAdapter()
        adapter.force_level(QualityLevel.CRITICAL)
        adapter.reset()
        assert adapter.current_level == QualityLevel.GOOD

    def test_presets_for_all_levels(self):
        for level in QualityLevel:
            assert level in NetworkQualityAdapter.PRESETS
            settings = NetworkQualityAdapter.PRESETS[level]
            assert isinstance(settings, AdaptiveSettings)

    def test_set_update_rate_limits(self):
        adapter = NetworkQualityAdapter()
        adapter.set_update_rate_limits(10.0, 30.0)
        assert adapter._min_update_rate == 10.0
        assert adapter._max_update_rate == 30.0

    def test_adapt_same_level_twice_hysteresis(self):
        """adapt() with same level returns unchanged settings initially."""
        adapter = NetworkQualityAdapter()
        metrics = QualityMetrics(rtt=0.05, packet_loss=0.0)
        first = adapter.adapt(metrics)
        second = adapter.adapt(metrics)
        assert second.update_rate == first.update_rate

    # ------------------------------------------------------------------
    # H-1: Quality adapter hysteresis — _current_level updated immediately
    # ------------------------------------------------------------------

    def test_current_level_updates_immediately_on_change(self):
        """_current_level reflects new quality instantly (H-1)."""
        adapter = NetworkQualityAdapter()
        assert adapter.current_level == QualityLevel.GOOD

        metrics = QualityMetrics(rtt=0.400, packet_loss=0.0)  # CRITICAL
        with mock.patch('time.time', return_value=100.0):
            result = adapter.adapt(metrics)

        assert adapter.current_level == QualityLevel.CRITICAL, \
            "_current_level should update immediately despite hysteresis"

    def test_settings_unchanged_during_hysteresis_after_level_change(self):
        """adapt() returns old settings during hysteresis period after level change (H-1)."""
        adapter = NetworkQualityAdapter(hysteresis_threshold=5.0, adaptation_delay=1.0)
        initial_settings = adapter.current_settings

        metrics = QualityMetrics(rtt=0.400, packet_loss=0.15)  # CRITICAL
        with mock.patch('time.time', return_value=100.0):
            result = adapter.adapt(metrics)

        # Level changed immediately
        assert adapter.current_level == QualityLevel.CRITICAL
        # But settings are unchanged (still within hysteresis)
        assert result is initial_settings

    def test_settings_adapt_after_hysteresis_and_delay(self):
        """adapt() applies new settings after hysteresis + adaptation delay (H-1)."""
        adapter = NetworkQualityAdapter(hysteresis_threshold=0.001, adaptation_delay=0.001)

        metrics_excellent = QualityMetrics(rtt=0.01, packet_loss=0.0)
        adapter.adapt(metrics_excellent)

        # Send poor metrics — level changes immediately
        metrics_poor = QualityMetrics(rtt=0.400, packet_loss=0.15)
        with mock.patch('time.time') as mock_time:
            mock_time.side_effect = [100.0, 100.0, 102.0]  # 2s later = past both delays
            adapter.adapt(metrics_poor)

        # After enough time, settings should adapt to CRITICAL
        with mock.patch('time.time', return_value=102.0):
            settings = adapter.adapt(metrics_poor)

        assert settings.update_rate == DEFAULT_CONFIG.UPDATE_RATE_CRITICAL, \
            "Settings should adapt after hysteresis + delay elapse"

    def test_level_stability_timer_resets_on_each_change(self):
        """Stability timer resets each time level changes (H-1)."""
        adapter = NetworkQualityAdapter(hysteresis_threshold=2.0, adaptation_delay=0.5)

        with mock.patch('time.time') as mock_time:
            mock_time.return_value = 100.0
            metrics_good = QualityMetrics(rtt=0.05, packet_loss=0.0)
            adapter.adapt(metrics_good)  # No change

            # Change to FAIR at t=100
            metrics_fair = QualityMetrics(rtt=0.15, packet_loss=0.03)
            adapter.adapt(metrics_fair)
            assert adapter.current_level == QualityLevel.FAIR
            t1 = adapter._level_stable_since

            # Change to POOR at t=101
            mock_time.return_value = 101.0
            metrics_poor = QualityMetrics(rtt=0.300, packet_loss=0.08)
            adapter.adapt(metrics_poor)
            assert adapter.current_level == QualityLevel.POOR
            t2 = adapter._level_stable_since

            # Timer was reset at t=101, so t2 should be later than t1
            assert t2 > t1, "Stability timer should reset on each level change"

    def test_adaptation_uses_tracked_level_not_fresh_metric(self):
        """Adaptation _create_settings uses self._current_level not fresh metric level (H-1)."""
        adapter = NetworkQualityAdapter(hysteresis_threshold=0.01, adaptation_delay=0.01)

        # Part 1: push level to POOR and let adaptation fire there
        with mock.patch('time.time') as mock_time:
            mock_time.return_value = 100.0
            metrics_poor = QualityMetrics(rtt=0.300, packet_loss=0.08)
            adapter.adapt(metrics_poor)
            # Level tracks immediately — POOR (H-1)
            assert adapter.current_level == QualityLevel.POOR

        with mock.patch('time.time') as mock_time:
            mock_time.return_value = 103.0
            # Same level, hysteresis+delay elapsed → adaptation fires
            result = adapter.adapt(metrics_poor)
            assert result.update_rate == DEFAULT_CONFIG.UPDATE_RATE_POOR

        # Part 2: push to CRITICAL so level is CRITICAL with CRITICAL-tuned settings
        with mock.patch('time.time') as mock_time:
            mock_time.return_value = 104.0
            metrics_critical = QualityMetrics(rtt=0.500, packet_loss=0.20)
            adapter.adapt(metrics_critical)
            assert adapter.current_level == QualityLevel.CRITICAL

        with mock.patch('time.time') as mock_time:
            mock_time.return_value = 108.0
            result = adapter.adapt(metrics_critical)
            # CRITICAL settings tuned
            assert result.update_rate == DEFAULT_CONFIG.UPDATE_RATE_CRITICAL

        # Part 3: now pass FAIR metrics at t=109 — level changes to FAIR immediately (H-1)
        with mock.patch('time.time', return_value=109.0):
            metrics_fair = QualityMetrics(rtt=0.15, packet_loss=0.03)
            result = adapter.adapt(metrics_fair)

        # Level follows the metric immediately (H-1)
        assert adapter.current_level == QualityLevel.FAIR
        # Returned settings are the OLD CRITICAL-tuned ones (from Part 2)
        assert result.update_rate == DEFAULT_CONFIG.UPDATE_RATE_CRITICAL


class TestAdaptiveSettings:
    """AdaptiveSettings defaults and structure tests."""

    def test_has_all_required_attributes(self):
        s = AdaptiveSettings()
        assert hasattr(s, 'update_rate')
        assert hasattr(s, 'compression_level')
        assert hasattr(s, 'delta_compression')
        assert hasattr(s, 'interpolation_delay')
        assert hasattr(s, 'extrapolation_limit')
        assert hasattr(s, 'packet_aggregation')
        assert hasattr(s, 'priority_queue')

    def test_default_update_rate(self):
        s = AdaptiveSettings()
        assert s.update_rate == DEFAULT_CONFIG.UPDATE_RATE_FAIR

    def test_default_compression_level(self):
        s = AdaptiveSettings()
        assert s.compression_level == DEFAULT_CONFIG.COMPRESSION_LEVEL

    def test_default_delta_compression(self):
        s = AdaptiveSettings()
        assert s.delta_compression is True

    def test_default_interpolation_delay(self):
        s = AdaptiveSettings()
        assert s.interpolation_delay == DEFAULT_CONFIG.CHANNEL_INITIAL_RTT

    def test_default_priority_queue(self):
        s = AdaptiveSettings()
        assert s.priority_queue is True
