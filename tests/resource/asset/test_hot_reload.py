"""Tests for HotReloadWatcher."""
import os
import tempfile
import time

import pytest

from engine.resource.asset.hot_reload import HotReloadWatcher


class TestHotReloadWatcher:
    def test_register_and_poll_no_change(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            fired: list[str] = []
            w = HotReloadWatcher()
            w.register(path, fired.append)
            changed = w.poll()
            assert changed == []
            assert fired == []
        finally:
            os.unlink(path)

    def test_poll_detects_change(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"v1")
            path = f.name
        try:
            fired: list[str] = []
            w = HotReloadWatcher()
            w.register(path, fired.append)
            # Modify
            time.sleep(0.05)
            with open(path, "wb") as f:
                f.write(b"v2")
            # Force mtime to differ (some FS have 1s granularity)
            os.utime(path, (time.time() + 2, time.time() + 2))
            changed = w.poll()
            assert path in changed
            assert path in fired
        finally:
            os.unlink(path)

    def test_poll_no_double_fire(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            fired: list[str] = []
            w = HotReloadWatcher()
            w.register(path, fired.append)
            os.utime(path, (time.time() + 2, time.time() + 2))
            w.poll()
            # Poll again without change
            second = w.poll()
            assert second == []
            assert len(fired) == 1
        finally:
            os.unlink(path)

    def test_unregister_stops_watching(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            fired: list[str] = []
            w = HotReloadWatcher()
            w.register(path, fired.append)
            w.unregister(path)
            os.utime(path, (time.time() + 2, time.time() + 2))
            changed = w.poll()
            assert changed == []
            assert fired == []
        finally:
            os.unlink(path)

    def test_start_stop_no_crash(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"v1")
            path = f.name
        try:
            fired: list[str] = []
            w = HotReloadWatcher(interval=0.01)
            w.register(path, fired.append)
            w.start()
            # Modify file while watcher is running
            time.sleep(0.02)
            with open(path, "wb") as f:
                f.write(b"v2")
            os.utime(path, (time.time() + 2, time.time() + 2))
            time.sleep(0.05)
            w.stop()
            assert w._running is False
            assert path in fired
        finally:
            os.unlink(path)
