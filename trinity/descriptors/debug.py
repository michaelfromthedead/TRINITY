"""
Debug descriptors - development-only profiling and logging.
Intended to be stripped in release builds.
"""
from __future__ import annotations
import time
import logging
from typing import Any, Callable, Optional, TypeVar
from trinity.descriptors.base import BaseDescriptor
from trinity.decorators.ops import Op, Step

T = TypeVar("T")
logger = logging.getLogger("trinity.descriptors.debug")


class ProfiledDescriptor(BaseDescriptor[T]):
    """Time every get/set call, feed to profiler."""
    __slots__ = ("_get_times_attr", "_set_times_attr", "_max_samples")
    descriptor_id = "profiled"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(self, field_type=object, inner=None, max_samples=100, **config):
        super().__init__(field_type=field_type, inner=inner, **config)
        self._max_samples = max_samples
        self._get_times_attr = ""
        self._set_times_attr = ""

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        self._get_times_attr = f"_profile_get_{name}"
        self._set_times_attr = f"_profile_set_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        start = time.perf_counter_ns()
        result = super().__get__(obj, objtype)
        elapsed = time.perf_counter_ns() - start
        times = getattr(obj, self._get_times_attr, None)
        if times is None:
            times = []
            object.__setattr__(obj, self._get_times_attr, times)
        times.append(elapsed)
        if len(times) > self._max_samples:
            times.pop(0)
        return result

    def __set__(self, obj, value):
        start = time.perf_counter_ns()
        super().__set__(obj, value)
        elapsed = time.perf_counter_ns() - start
        times = getattr(obj, self._set_times_attr, None)
        if times is None:
            times = []
            object.__setattr__(obj, self._set_times_attr, times)
        times.append(elapsed)
        if len(times) > self._max_samples:
            times.pop(0)

    def get_stats(self, obj) -> dict:
        get_times = getattr(obj, self._get_times_attr, [])
        set_times = getattr(obj, self._set_times_attr, [])
        def _stats(times):
            if not times:
                return {"count": 0, "avg_ns": 0, "max_ns": 0, "min_ns": 0}
            return {"count": len(times), "avg_ns": sum(times) // len(times), "max_ns": max(times), "min_ns": min(times)}
        return {"get": _stats(get_times), "set": _stats(set_times)}

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.INTERCEPT, {"get": "profile_get", "set": "profile_set"}),
                Step(Op.TAG, {"key": "profiled", "value": True})]

    def get_metadata(self):
        meta = super().get_metadata()
        meta["max_samples"] = self._max_samples
        return meta


class LoggedDescriptor(BaseDescriptor[T]):
    """Log all field accesses with old->new values."""
    __slots__ = ("_log_level",)
    descriptor_id = "logged"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(self, field_type=object, inner=None, log_level="DEBUG", **config):
        super().__init__(field_type=field_type, inner=inner, **config)
        self._log_level = getattr(logging, log_level.upper(), logging.DEBUG)

    def post_get(self, obj, value):
        logger.log(self._log_level, "%s.%s -> %r", type(obj).__name__, self._name, value)
        return value

    def post_set(self, obj, value, old_value):
        logger.log(self._log_level, "%s.%s: %r -> %r", type(obj).__name__, self._name, old_value, value)

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.INTERCEPT, {"get": "log_get", "set": "log_set"}),
                Step(Op.TAG, {"key": "logged", "value": True})]

    def get_metadata(self):
        meta = super().get_metadata()
        meta["log_level"] = logging.getLevelName(self._log_level)
        return meta


class WatchedDescriptor(BaseDescriptor[T]):
    """Conditional breakpoint: trigger callback when value matches condition."""
    __slots__ = ("_condition", "_callback")
    descriptor_id = "watched"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(self, field_type=object, inner=None, condition=None, callback=None, **config):
        super().__init__(field_type=field_type, inner=inner, **config)
        self._condition = condition  # callable(value) -> bool
        self._callback = callback    # callable(obj, name, value) or None triggers breakpoint

    def post_set(self, obj, value, old_value):
        if self._condition and self._condition(value):
            if self._callback:
                self._callback(obj, self._name, value)
            else:
                import pdb; pdb.set_trace()  # noqa: E702

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.INTERCEPT, {"set": "watch_condition"}),
                Step(Op.TAG, {"key": "watched", "value": True})]

    def get_metadata(self):
        meta = super().get_metadata()
        meta["has_condition"] = self._condition is not None
        meta["has_callback"] = self._callback is not None
        return meta


__all__ = ["ProfiledDescriptor", "LoggedDescriptor", "WatchedDescriptor"]
