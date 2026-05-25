"""Tests for Phase 7 descriptors: RateLimited, Conditional, Transform."""

from __future__ import annotations

import time

import pytest

from trinity.decorators.ops import Op, Step
from trinity.descriptors.rate_limiting import RateLimitedDescriptor, RateLimitExceeded
from trinity.descriptors.conditional import ConditionalDescriptor, WriteConditionError
from trinity.descriptors.transform import TransformDescriptor


# =============================================================================
# RateLimitedDescriptor
# =============================================================================

class TestRateLimitedDescriptor:
    """Tests for RateLimitedDescriptor."""

    def _make_class(self, **kwargs):
        desc = RateLimitedDescriptor(**kwargs)

        class Obj:
            val = desc

        desc.__set_name__(Obj, "val")
        return Obj

    def test_under_limit_passes(self):
        Cls = self._make_class(max_writes_per_second=5)
        obj = Cls()
        for i in range(5):
            obj.val = i
        assert obj.val == 4

    def test_exceeding_limit_raises(self):
        Cls = self._make_class(max_writes_per_second=3)
        obj = Cls()
        obj.val = 1
        obj.val = 2
        obj.val = 3
        with pytest.raises(RateLimitExceeded):
            obj.val = 4

    def test_on_exceed_drop(self):
        Cls = self._make_class(max_writes_per_second=2, on_exceed="drop")
        obj = Cls()
        obj.val = 10
        obj.val = 20
        # This write should be silently dropped
        obj.val = 30
        assert obj.val == 20

    def test_window_expires(self):
        Cls = self._make_class(max_writes_per_second=1)
        obj = Cls()
        obj.val = 1
        # Wait for window to expire
        time.sleep(1.05)
        obj.val = 2  # Should succeed
        assert obj.val == 2

    def test_invalid_on_exceed(self):
        with pytest.raises(ValueError, match="on_exceed"):
            RateLimitedDescriptor(on_exceed="invalid")

    def test_descriptor_steps(self):
        desc = RateLimitedDescriptor(max_writes_per_second=5)
        steps = desc.descriptor_steps
        assert isinstance(steps, list)
        assert all(isinstance(s, Step) for s in steps)
        ops = [s.op for s in steps]
        assert Op.INTERCEPT in ops
        assert Op.VALIDATE in ops

    def test_get_metadata(self):
        desc = RateLimitedDescriptor(max_writes_per_second=10, on_exceed="drop")
        meta = desc.get_metadata()
        assert isinstance(meta, dict)
        assert meta["max_writes_per_second"] == 10
        assert meta["on_exceed"] == "drop"


# =============================================================================
# ConditionalDescriptor
# =============================================================================

class TestConditionalDescriptor:
    """Tests for ConditionalDescriptor."""

    def _make_class(self, predicate):
        desc = ConditionalDescriptor(predicate=predicate)

        class Obj:
            val = desc

        desc.__set_name__(Obj, "val")
        return Obj

    def test_predicate_passes(self):
        # Only allow increasing values
        Cls = self._make_class(lambda obj, name, old, new: old is None or new > old)
        obj = Cls()
        obj.val = 1
        obj.val = 5
        assert obj.val == 5

    def test_predicate_fails_raises(self):
        Cls = self._make_class(lambda obj, name, old, new: old is None or new > old)
        obj = Cls()
        obj.val = 10
        with pytest.raises(WriteConditionError):
            obj.val = 5

    def test_requires_predicate(self):
        with pytest.raises(ValueError, match="requires a predicate"):
            ConditionalDescriptor(predicate=None)

    def test_predicate_receives_correct_args(self):
        received = {}

        def capture(obj, name, old, new):
            received.update(name=name, old=old, new=new)
            return True

        Cls = self._make_class(capture)
        obj = Cls()
        obj.val = 42
        assert received["name"] == "val"
        assert received["old"] is None
        assert received["new"] == 42

    def test_descriptor_steps(self):
        desc = ConditionalDescriptor(predicate=lambda obj, name, old, new: True)
        steps = desc.descriptor_steps
        assert isinstance(steps, list)
        assert all(isinstance(s, Step) for s in steps)
        ops = [s.op for s in steps]
        assert Op.INTERCEPT in ops

    def test_get_metadata(self):
        def my_pred(obj, name, old, new):
            return True

        desc = ConditionalDescriptor(predicate=my_pred)
        meta = desc.get_metadata()
        assert isinstance(meta, dict)
        assert "predicate" in meta


# =============================================================================
# TransformDescriptor
# =============================================================================

class TestTransformDescriptor:
    """Tests for TransformDescriptor."""

    def _make_class(self, **kwargs):
        desc = TransformDescriptor(**kwargs)

        class Obj:
            val = desc

        desc.__set_name__(Obj, "val")
        return Obj

    def test_write_transform(self):
        Cls = self._make_class(write_transform=lambda v: v * 2)
        obj = Cls()
        obj.val = 5
        # Stored value should be transformed
        assert obj.__dict__["val"] == 10

    def test_read_transform(self):
        Cls = self._make_class(read_transform=lambda v: str(v))
        obj = Cls()
        obj.val = 42
        assert obj.val == "42"
        # Raw stored value is untransformed
        assert obj.__dict__["val"] == 42

    def test_both_transforms(self):
        Cls = self._make_class(
            write_transform=lambda v: v.strip(),
            read_transform=lambda v: v.upper(),
        )
        obj = Cls()
        obj.val = "  hello  "
        assert obj.__dict__["val"] == "hello"
        assert obj.val == "HELLO"

    def test_no_transforms(self):
        Cls = self._make_class()
        obj = Cls()
        obj.val = "unchanged"
        assert obj.val == "unchanged"

    def test_descriptor_steps(self):
        desc = TransformDescriptor(write_transform=lambda v: v)
        steps = desc.descriptor_steps
        assert isinstance(steps, list)
        assert all(isinstance(s, Step) for s in steps)
        ops = [s.op for s in steps]
        assert Op.INTERCEPT in ops

    def test_get_metadata(self):
        desc = TransformDescriptor(write_transform=lambda v: v, read_transform=lambda v: v)
        meta = desc.get_metadata()
        assert isinstance(meta, dict)
        assert meta["has_read_transform"] is True
        assert meta["has_write_transform"] is True
