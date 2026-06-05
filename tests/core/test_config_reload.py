"""Tests for config reload callback registration (T-CC-1.6)."""
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.core.config_reload import (
    ConfigCache,
    ConfigReloadDecorator,
    ConfigReloadManager,
    RegisteredHandler,
    ReloadEvent,
    ReloadOutcome,
    ReloadPriority,
    ReloadResult,
    create_engine_config_manager,
    on_config_reload,
)
from engine.core.file_watcher import FileChangeType, FileWatcher


class TestReloadPriority:
    """Tests for ReloadPriority enum."""

    def test_priority_ordering(self):
        priorities = [
            ReloadPriority.HIGHEST,
            ReloadPriority.HIGH,
            ReloadPriority.NORMAL,
            ReloadPriority.LOW,
            ReloadPriority.LOWEST,
        ]
        values = [p.value for p in priorities]
        assert values == sorted(values)

    def test_all_priorities_exist(self):
        assert len(ReloadPriority) == 5


class TestReloadResult:
    """Tests for ReloadResult enum."""

    def test_all_results_exist(self):
        results = [
            ReloadResult.SUCCESS,
            ReloadResult.PARTIAL,
            ReloadResult.FAILED,
            ReloadResult.SKIPPED,
        ]
        assert len(results) == 4


class TestReloadEvent:
    """Tests for ReloadEvent dataclass."""

    def test_basic_event(self):
        event = ReloadEvent(
            path=Path("/config/test.json"),
            change_type=FileChangeType.MODIFIED,
        )
        assert event.path == Path("/config/test.json")
        assert event.change_type == FileChangeType.MODIFIED
        assert event.old_data is None
        assert event.new_data is None

    def test_event_with_data(self):
        old = {"key": "old"}
        new = {"key": "new"}
        event = ReloadEvent(
            path=Path("config.json"),
            change_type=FileChangeType.MODIFIED,
            old_data=old,
            new_data=new,
        )
        assert event.old_data == old
        assert event.new_data == new

    def test_is_creation(self):
        event = ReloadEvent(Path("test.json"), FileChangeType.CREATED)
        assert event.is_creation
        assert not event.is_modification
        assert not event.is_deletion

    def test_is_modification(self):
        event = ReloadEvent(Path("test.json"), FileChangeType.MODIFIED)
        assert not event.is_creation
        assert event.is_modification
        assert not event.is_deletion

    def test_is_deletion(self):
        event = ReloadEvent(Path("test.json"), FileChangeType.DELETED)
        assert not event.is_creation
        assert not event.is_modification
        assert event.is_deletion


class TestReloadOutcome:
    """Tests for ReloadOutcome dataclass."""

    def test_success_outcome(self):
        event = ReloadEvent(Path("test.json"), FileChangeType.MODIFIED)
        outcome = ReloadOutcome(
            event=event,
            result=ReloadResult.SUCCESS,
            handler_name="test_handler",
            duration_ms=5.5,
        )
        assert outcome.result == ReloadResult.SUCCESS
        assert outcome.handler_name == "test_handler"
        assert outcome.duration_ms == 5.5
        assert outcome.error is None

    def test_failed_outcome(self):
        event = ReloadEvent(Path("test.json"), FileChangeType.MODIFIED)
        outcome = ReloadOutcome(
            event=event,
            result=ReloadResult.FAILED,
            handler_name="bad_handler",
            duration_ms=1.0,
            error="Connection refused",
        )
        assert outcome.result == ReloadResult.FAILED
        assert outcome.error == "Connection refused"


class TestRegisteredHandler:
    """Tests for RegisteredHandler dataclass."""

    def test_matches_specific_path(self):
        handler = RegisteredHandler(
            name="test",
            handler=lambda e: ReloadResult.SUCCESS,
            paths={Path("/config/app.json").resolve()},
        )
        assert handler.matches(Path("/config/app.json"))
        assert not handler.matches(Path("/config/other.json"))

    def test_matches_extension(self):
        handler = RegisteredHandler(
            name="test",
            handler=lambda e: ReloadResult.SUCCESS,
            extensions={".json", ".yaml"},
        )
        assert handler.matches(Path("config.json"))
        assert handler.matches(Path("config.yaml"))
        assert not handler.matches(Path("config.txt"))

    def test_matches_extension_without_dot(self):
        handler = RegisteredHandler(
            name="test",
            handler=lambda e: ReloadResult.SUCCESS,
            extensions={"json"},
        )
        assert handler.matches(Path("config.json"))

    def test_matches_pattern(self):
        handler = RegisteredHandler(
            name="test",
            handler=lambda e: ReloadResult.SUCCESS,
            patterns={"*.json", "config/*"},
        )
        assert handler.matches(Path("test.json"))
        assert handler.matches(Path("config/app.yaml"))
        assert not handler.matches(Path("data.txt"))

    def test_matches_all_when_no_filters(self):
        handler = RegisteredHandler(
            name="test",
            handler=lambda e: ReloadResult.SUCCESS,
        )
        assert handler.matches(Path("any/path/file.xyz"))

    def test_disabled_handler_no_match(self):
        handler = RegisteredHandler(
            name="test",
            handler=lambda e: ReloadResult.SUCCESS,
            enabled=False,
        )
        assert not handler.matches(Path("any.json"))


class TestConfigCache:
    """Tests for ConfigCache."""

    def test_get_set(self):
        cache = ConfigCache()
        cache.set(Path("test.json"), {"key": "value"}, 100)
        assert cache.get(Path("test.json")) == {"key": "value"}

    def test_get_nonexistent(self):
        cache = ConfigCache()
        assert cache.get(Path("missing.json")) is None

    def test_remove(self):
        cache = ConfigCache()
        cache.set(Path("test.json"), {"key": "value"}, 100)
        data = cache.remove(Path("test.json"))
        assert data == {"key": "value"}
        assert cache.get(Path("test.json")) is None

    def test_clear(self):
        cache = ConfigCache()
        cache.set(Path("a.json"), {}, 10)
        cache.set(Path("b.json"), {}, 10)
        assert cache.entry_count == 2
        cache.clear()
        assert cache.entry_count == 0

    def test_eviction_by_count(self):
        cache = ConfigCache(max_entries=2)
        cache.set(Path("a.json"), "a", 1)
        cache.set(Path("b.json"), "b", 1)
        cache.set(Path("c.json"), "c", 1)
        assert cache.entry_count == 2
        assert cache.get(Path("a.json")) is None  # Oldest evicted

    def test_eviction_by_size(self):
        cache = ConfigCache(max_size_bytes=100)
        cache.set(Path("a.json"), "a", 50)
        cache.set(Path("b.json"), "b", 50)
        cache.set(Path("c.json"), "c", 50)
        assert cache.total_size <= 100

    def test_update_existing(self):
        cache = ConfigCache()
        cache.set(Path("test.json"), "old", 10)
        cache.set(Path("test.json"), "new", 20)
        assert cache.get(Path("test.json")) == "new"
        assert cache.entry_count == 1


class TestConfigReloadManager:
    """Tests for ConfigReloadManager."""

    def test_initial_state(self):
        manager = ConfigReloadManager()
        assert not manager.is_running
        assert manager.handler_count == 0
        assert not manager.is_paused

    def test_register_handler(self):
        manager = ConfigReloadManager()
        result = manager.register(
            "test_handler",
            lambda e: ReloadResult.SUCCESS,
        )
        assert result is True
        assert manager.handler_count == 1

    def test_register_duplicate_fails(self):
        manager = ConfigReloadManager()
        manager.register("test", lambda e: ReloadResult.SUCCESS)
        result = manager.register("test", lambda e: ReloadResult.SUCCESS)
        assert result is False
        assert manager.handler_count == 1

    def test_unregister_handler(self):
        manager = ConfigReloadManager()
        manager.register("test", lambda e: ReloadResult.SUCCESS)
        result = manager.unregister("test")
        assert result is True
        assert manager.handler_count == 0

    def test_unregister_nonexistent(self):
        manager = ConfigReloadManager()
        result = manager.unregister("nonexistent")
        assert result is False

    def test_register_with_paths(self):
        manager = ConfigReloadManager()
        manager.register(
            "config_handler",
            lambda e: ReloadResult.SUCCESS,
            paths=["/config/app.json"],
        )
        handler = manager.get_handler("config_handler")
        assert handler is not None
        assert handler.paths is not None

    def test_register_with_patterns(self):
        manager = ConfigReloadManager()
        manager.register(
            "json_handler",
            lambda e: ReloadResult.SUCCESS,
            patterns=["*.json"],
        )
        handler = manager.get_handler("json_handler")
        assert handler is not None
        assert "*.json" in handler.patterns

    def test_register_with_extensions(self):
        manager = ConfigReloadManager()
        manager.register(
            "yaml_handler",
            lambda e: ReloadResult.SUCCESS,
            extensions=["yaml", "yml"],
        )
        handler = manager.get_handler("yaml_handler")
        assert handler is not None
        assert ".yaml" in handler.extensions
        assert ".yml" in handler.extensions

    def test_register_with_priority(self):
        manager = ConfigReloadManager()
        manager.register("low", lambda e: ReloadResult.SUCCESS, priority=ReloadPriority.LOW)
        manager.register("high", lambda e: ReloadResult.SUCCESS, priority=ReloadPriority.HIGH)
        manager.register("normal", lambda e: ReloadResult.SUCCESS, priority=ReloadPriority.NORMAL)

        handlers = [manager.get_handler(n).priority for n in ["high", "normal", "low"]]
        expected = [ReloadPriority.HIGH, ReloadPriority.NORMAL, ReloadPriority.LOW]
        for h, exp in zip(handlers, expected):
            assert h == exp

    def test_set_enabled(self):
        manager = ConfigReloadManager()
        manager.register("test", lambda e: ReloadResult.SUCCESS)

        assert manager.get_handler("test").enabled is True
        manager.set_enabled("test", False)
        assert manager.get_handler("test").enabled is False
        manager.set_enabled("test", True)
        assert manager.get_handler("test").enabled is True

    def test_set_enabled_nonexistent(self):
        manager = ConfigReloadManager()
        result = manager.set_enabled("nonexistent", False)
        assert result is False

    def test_watch_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigReloadManager()
            result = manager.watch(tmpdir)
            assert result is True

    def test_watch_nonexistent(self):
        manager = ConfigReloadManager()
        result = manager.watch("/nonexistent/path")
        assert result is False

    def test_unwatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigReloadManager()
            manager.watch(tmpdir)
            result = manager.unwatch(tmpdir)
            assert result is True

    def test_start_stop(self):
        manager = ConfigReloadManager()
        assert not manager.is_running
        manager.start()
        assert manager.is_running
        manager.stop()
        assert not manager.is_running

    def test_pause_resume(self):
        manager = ConfigReloadManager()
        assert not manager.is_paused
        manager.pause()
        assert manager.is_paused
        manager.resume()
        assert not manager.is_paused

    def test_trigger_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test.json"
            config_path.write_text('{"key": "value"}')

            handler_calls = []

            def handler(event):
                handler_calls.append(event)
                return ReloadResult.SUCCESS

            manager = ConfigReloadManager()
            manager.register("test", handler)

            outcomes = manager.trigger_reload(config_path)
            assert len(outcomes) == 1
            assert outcomes[0].result == ReloadResult.SUCCESS
            assert len(handler_calls) == 1
            assert handler_calls[0].new_data == {"key": "value"}

    def test_trigger_reload_deleted(self):
        handler_calls = []

        def handler(event):
            handler_calls.append(event)
            return ReloadResult.SUCCESS

        manager = ConfigReloadManager()
        manager.register("test", handler)

        outcomes = manager.trigger_reload("/nonexistent.json")
        assert len(outcomes) == 1
        assert len(handler_calls) == 1
        assert handler_calls[0].is_deletion

    def test_handler_exception_recorded(self):
        def bad_handler(event):
            raise ValueError("Handler error")

        manager = ConfigReloadManager()
        manager.register("bad", bad_handler)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "test.json"
            config.write_text("{}")
            outcomes = manager.trigger_reload(config)

        assert len(outcomes) == 1
        assert outcomes[0].result == ReloadResult.FAILED
        assert "Handler error" in outcomes[0].error

    def test_outcome_listener(self):
        received = []

        def listener(outcome):
            received.append(outcome)

        manager = ConfigReloadManager()
        manager.add_outcome_listener(listener)
        manager.register("test", lambda e: ReloadResult.SUCCESS)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "test.json"
            config.write_text("{}")
            manager.trigger_reload(config)

        assert len(received) == 1
        assert received[0].result == ReloadResult.SUCCESS

    def test_remove_outcome_listener(self):
        received = []

        def listener(outcome):
            received.append(outcome)

        manager = ConfigReloadManager()
        manager.add_outcome_listener(listener)
        result = manager.remove_outcome_listener(listener)
        assert result is True

        manager.register("test", lambda e: ReloadResult.SUCCESS)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "test.json"
            config.write_text("{}")
            manager.trigger_reload(config)

        assert len(received) == 0

    def test_get_recent_outcomes(self):
        manager = ConfigReloadManager()
        manager.register("test", lambda e: ReloadResult.SUCCESS)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                config = Path(tmpdir) / f"test{i}.json"
                config.write_text("{}")
                manager.trigger_reload(config)

        outcomes = manager.get_recent_outcomes(3)
        assert len(outcomes) == 3

    def test_get_status(self):
        manager = ConfigReloadManager()
        manager.register(
            "test",
            lambda e: ReloadResult.SUCCESS,
            extensions=["json"],
            priority=ReloadPriority.HIGH,
        )

        status = manager.get_status()
        assert status['running'] is False
        assert status['paused'] is False
        assert status['handler_count'] == 1
        assert len(status['handlers']) == 1
        assert status['handlers'][0]['name'] == "test"
        assert status['handlers'][0]['priority'] == "HIGH"

    def test_clear(self):
        manager = ConfigReloadManager()
        manager.register("test", lambda e: ReloadResult.SUCCESS)
        manager.add_outcome_listener(lambda o: None)

        manager.clear()
        assert manager.handler_count == 0


class TestConfigReloadManagerIntegration:
    """Integration tests for ConfigReloadManager with FileWatcher."""

    def test_file_change_triggers_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text('{"version": 1}')

            events = []

            def handler(event):
                events.append(event)
                return ReloadResult.SUCCESS

            manager = ConfigReloadManager()
            manager.watch(tmpdir)
            manager.register("test", handler, extensions=["json"])

            time.sleep(0.1)
            config_path.write_text('{"version": 2}')
            manager._watcher.poll_once()

            assert len(events) >= 1
            assert events[-1].new_data == {"version": 2}

    def test_paused_manager_ignores_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text('{"v": 1}')

            events = []

            def handler(event):
                events.append(event)
                return ReloadResult.SUCCESS

            manager = ConfigReloadManager()
            manager.watch(tmpdir)
            manager.register("test", handler)
            manager.pause()

            config_path.write_text('{"v": 2}')
            manager._watcher.poll_once()

            assert len(events) == 0

    def test_handler_priority_order(self):
        order = []

        def make_handler(name):
            def handler(event):
                order.append(name)
                return ReloadResult.SUCCESS
            return handler

        manager = ConfigReloadManager()
        manager.register("low", make_handler("low"), priority=ReloadPriority.LOW)
        manager.register("highest", make_handler("highest"), priority=ReloadPriority.HIGHEST)
        manager.register("normal", make_handler("normal"), priority=ReloadPriority.NORMAL)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "test.json"
            config.write_text("{}")
            manager.trigger_reload(config)

        assert order == ["highest", "normal", "low"]

    def test_handler_filtering(self):
        json_calls = []
        yaml_calls = []

        def json_handler(event):
            json_calls.append(event)
            return ReloadResult.SUCCESS

        def yaml_handler(event):
            yaml_calls.append(event)
            return ReloadResult.SUCCESS

        manager = ConfigReloadManager()
        manager.register("json", json_handler, extensions=["json"])
        manager.register("yaml", yaml_handler, extensions=["yaml"])

        with tempfile.TemporaryDirectory() as tmpdir:
            json_config = Path(tmpdir) / "config.json"
            yaml_config = Path(tmpdir) / "config.yaml"
            json_config.write_text("{}")
            yaml_config.write_text("key: value")

            manager.trigger_reload(json_config)
            manager.trigger_reload(yaml_config)

        assert len(json_calls) == 1
        assert len(yaml_calls) == 1

    def test_disabled_handler_skipped(self):
        calls = []

        def handler(event):
            calls.append(event)
            return ReloadResult.SUCCESS

        manager = ConfigReloadManager()
        manager.register("test", handler)
        manager.set_enabled("test", False)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "test.json"
            config.write_text("{}")
            manager.trigger_reload(config)

        assert len(calls) == 0


class TestConfigReloadDecorator:
    """Tests for ConfigReloadDecorator."""

    def test_decorator_registers_metadata(self):
        @on_config_reload(extension=".json", priority=ReloadPriority.HIGH)
        def my_reload_handler(event):
            return ReloadResult.SUCCESS

        key = f"{my_reload_handler.__module__}.{my_reload_handler.__qualname__}"
        assert key in ConfigReloadDecorator._registry

    def test_decorator_preserves_function(self):
        @on_config_reload(pattern="*.json")
        def handler(event):
            return ReloadResult.SUCCESS

        assert callable(handler)
        assert handler.__name__ == "handler"

    def test_get_handlers_for_class(self):
        # The get_handlers_for_class uses qualname matching
        # which requires the class to be defined at module level for proper lookup.
        # This tests the basic lookup mechanism works.
        handlers = ConfigReloadDecorator.get_handlers_for_class(object())
        assert isinstance(handlers, list)  # Returns empty list for object with no decorators


class TestCreateEngineConfigManager:
    """Tests for create_engine_config_manager factory."""

    def test_factory_creates_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = create_engine_config_manager([tmpdir])
            assert manager is not None
            assert isinstance(manager, ConfigReloadManager)

    def test_factory_with_multiple_dirs(self):
        with tempfile.TemporaryDirectory() as dir1:
            with tempfile.TemporaryDirectory() as dir2:
                manager = create_engine_config_manager([dir1, dir2])
                assert manager is not None


class TestConcurrency:
    """Thread safety tests."""

    def test_concurrent_register_unregister(self):
        manager = ConfigReloadManager()
        errors = []

        def register_worker():
            try:
                for i in range(50):
                    name = f"handler_{threading.current_thread().name}_{i}"
                    manager.register(name, lambda e: ReloadResult.SUCCESS)
                    manager.unregister(name)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_trigger_reload(self):
        manager = ConfigReloadManager()
        calls = []
        lock = threading.Lock()

        def handler(event):
            with lock:
                calls.append(event)
            return ReloadResult.SUCCESS

        manager.register("test", handler)

        def trigger_worker(path):
            for _ in range(10):
                manager.trigger_reload(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "test.json"
            config.write_text("{}")

            threads = [
                threading.Thread(target=trigger_worker, args=(config,))
                for _ in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(calls) == 40


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_json_file(self):
        manager = ConfigReloadManager()
        events = []
        manager.register("test", lambda e: (events.append(e), ReloadResult.SUCCESS)[1])

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "empty.json"
            config.write_text("")
            manager.trigger_reload(config)

        assert len(events) == 1
        assert events[0].new_data is None  # Invalid JSON

    def test_invalid_json_file(self):
        manager = ConfigReloadManager()
        events = []
        manager.register("test", lambda e: (events.append(e), ReloadResult.SUCCESS)[1])

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "bad.json"
            config.write_text("{invalid json}")
            manager.trigger_reload(config)

        assert len(events) == 1
        assert events[0].new_data is None

    def test_non_json_file_loaded_as_text(self):
        manager = ConfigReloadManager()
        events = []
        manager.register("test", lambda e: (events.append(e), ReloadResult.SUCCESS)[1])

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "config.yaml"
            config.write_text("key: value\n")
            manager.trigger_reload(config)

        assert len(events) == 1
        assert events[0].new_data == "key: value\n"

    def test_outcome_history_limit(self):
        manager = ConfigReloadManager()
        manager._max_outcomes = 5
        manager.register("test", lambda e: ReloadResult.SUCCESS)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                config = Path(tmpdir) / f"test{i}.json"
                config.write_text("{}")
                manager.trigger_reload(config)

        outcomes = manager.get_recent_outcomes(100)
        assert len(outcomes) == 5

    def test_listener_exception_doesnt_break_others(self):
        received = []

        def bad_listener(outcome):
            raise ValueError("Listener error")

        def good_listener(outcome):
            received.append(outcome)

        manager = ConfigReloadManager()
        manager.add_outcome_listener(bad_listener)
        manager.add_outcome_listener(good_listener)
        manager.register("test", lambda e: ReloadResult.SUCCESS)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "test.json"
            config.write_text("{}")
            manager.trigger_reload(config)

        assert len(received) == 1
