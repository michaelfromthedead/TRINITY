"""Tests for DeltaEncoder."""

import pytest

from engine.core.session.delta import DeltaEncoder, DeltaData
from engine.core.session.session import SessionData


class TestDeltaEncoder:
    def test_encode_added(self):
        old = SessionData(world_snapshot={"a": 1})
        new = SessionData(world_snapshot={"a": 1, "b": 2})
        delta = DeltaEncoder.encode_delta(old, new)
        assert delta.added == {"b": 2}
        assert delta.removed == {}
        assert delta.modified == {}

    def test_encode_removed(self):
        old = SessionData(world_snapshot={"a": 1, "b": 2})
        new = SessionData(world_snapshot={"a": 1})
        delta = DeltaEncoder.encode_delta(old, new)
        assert delta.removed == {"b": 2}

    def test_encode_modified(self):
        old = SessionData(world_snapshot={"a": 1})
        new = SessionData(world_snapshot={"a": 99})
        delta = DeltaEncoder.encode_delta(old, new)
        assert delta.modified == {"a": 99}

    def test_apply_delta(self):
        base = SessionData(world_snapshot={"a": 1, "b": 2})
        delta = DeltaData(added={"c": 3}, removed={"b": 2}, modified={"a": 10})
        result = DeltaEncoder.apply_delta(base, delta)
        assert result.world_snapshot == {"a": 10, "c": 3}

    def test_no_change_delta(self):
        sd = SessionData(world_snapshot={"x": 1})
        delta = DeltaEncoder.encode_delta(sd, sd)
        assert delta.added == {}
        assert delta.removed == {}
        assert delta.modified == {}
