"""Tests for Session save/load."""

import json
import os
import tempfile

import pytest

from engine.core.session.session import Session, SessionData, Result
from engine.core.constants import SESSION_VERSION


class TestSessionData:
    def test_to_dict_from_dict_roundtrip(self):
        sd = SessionData(
            version=SESSION_VERSION,
            timestamp=123.456,
            frame_count=100,
            total_time=10.5,
            world_snapshot={"player": {"hp": 100}},
            metadata={"level": "1"},
        )
        d = sd.to_dict()
        restored = SessionData.from_dict(d)
        assert restored.version == sd.version
        assert restored.frame_count == sd.frame_count
        assert restored.total_time == sd.total_time
        assert restored.world_snapshot == sd.world_snapshot
        assert restored.metadata == sd.metadata

    def test_defaults(self):
        sd = SessionData()
        assert sd.version == SESSION_VERSION
        assert sd.frame_count == 0


class TestSession:
    def test_save_load_roundtrip(self, tmp_path):
        filepath = str(tmp_path / "test.json")
        session = Session(
            frame_count=42,
            total_time=3.14,
            world_snapshot={"entities": [1, 2, 3]},
            metadata={"name": "test"},
        )
        result = session.save(filepath)
        assert result.success

        session2 = Session()
        result2 = session2.load(filepath)
        assert result2.success
        assert session2.frame_count == 42
        assert session2.world_snapshot == {"entities": [1, 2, 3]}

    def test_save_version_stamp(self, tmp_path):
        filepath = str(tmp_path / "v.json")
        Session(frame_count=1).save(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert data["version"] == SESSION_VERSION

    def test_load_nonexistent_file(self):
        session = Session()
        result = session.load("/nonexistent/path.json")
        assert not result.success
        assert result.error is not None
