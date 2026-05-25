"""
Async descriptors - deferred and asynchronous field loading.
"""
from __future__ import annotations
import enum
from typing import Any, Callable, Optional, TypeVar
from trinity.descriptors.base import BaseDescriptor
from trinity.decorators.ops import Op, Step

T = TypeVar("T")


class LazyDescriptor(BaseDescriptor[T]):
    """Defer initialization until first access."""
    __slots__ = ("_factory", "_init_mode", "_initialized_attr")
    descriptor_id = "lazy"
    accepts_inner = ("storage",)
    accepts_outer = ("*",)
    excludes = ()

    VALID_MODES = frozenset({"first_access", "first_frame", "explicit"})

    def __init__(self, field_type=object, inner=None, factory=None, init_mode="first_access", **config):
        super().__init__(field_type=field_type, inner=inner, **config)
        if init_mode not in self.VALID_MODES:
            raise ValueError(f"Invalid init_mode '{init_mode}'. Valid: {sorted(self.VALID_MODES)}")
        self._factory = factory
        self._init_mode = init_mode
        self._initialized_attr = ""

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        self._initialized_attr = f"_lazy_init_{name}"

    def pre_get(self, obj):
        if self._init_mode == "first_access" and not getattr(obj, self._initialized_attr, False):
            if self._factory:
                value = self._factory()
                self.__set__(obj, value)
            object.__setattr__(obj, self._initialized_attr, True)

    def initialize(self, obj):
        """Explicitly initialize (for 'explicit' mode)."""
        if self._factory and not getattr(obj, self._initialized_attr, False):
            value = self._factory()
            self.__set__(obj, value)
            object.__setattr__(obj, self._initialized_attr, True)

    def is_initialized(self, obj) -> bool:
        return getattr(obj, self._initialized_attr, False)

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.INTERCEPT, {"get": "lazy_init"}),
                Step(Op.TAG, {"key": "lazy", "value": True}),
                Step(Op.TAG, {"key": "init_mode", "value": self._init_mode})]

    def get_metadata(self):
        meta = super().get_metadata()
        meta["init_mode"] = self._init_mode
        meta["has_factory"] = self._factory is not None
        return meta


class AsyncLoadState(enum.Enum):
    NOT_STARTED = "not_started"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class AsyncLoadDescriptor(BaseDescriptor[T]):
    """Load value asynchronously, return fallback until ready."""
    __slots__ = ("_loader", "_fallback", "_state_attr", "_error_attr")
    descriptor_id = "async_load"
    accepts_inner = ("storage",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(self, field_type=object, inner=None, loader=None, fallback=None, **config):
        super().__init__(field_type=field_type, inner=inner, **config)
        self._loader = loader
        self._fallback = fallback
        self._state_attr = ""
        self._error_attr = ""

    def __set_name__(self, owner, name):
        super().__set_name__(owner, name)
        self._state_attr = f"_async_state_{name}"
        self._error_attr = f"_async_error_{name}"

    def post_get(self, obj, value):
        state = getattr(obj, self._state_attr, AsyncLoadState.NOT_STARTED)
        if state == AsyncLoadState.NOT_STARTED and self._loader:
            object.__setattr__(obj, self._state_attr, AsyncLoadState.LOADING)
            try:
                loaded = self._loader()
                self.__set__(obj, loaded)
                object.__setattr__(obj, self._state_attr, AsyncLoadState.LOADED)
                return loaded
            except Exception as e:
                object.__setattr__(obj, self._state_attr, AsyncLoadState.ERROR)
                object.__setattr__(obj, self._error_attr, str(e))
                return self._fallback
        if state in (AsyncLoadState.NOT_STARTED, AsyncLoadState.LOADING, AsyncLoadState.ERROR):
            return self._fallback if value is None else value
        return value

    def get_state(self, obj) -> AsyncLoadState:
        return getattr(obj, self._state_attr, AsyncLoadState.NOT_STARTED)

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.INTERCEPT, {"get": "async_load"}),
                Step(Op.TAG, {"key": "async_load", "value": True})]

    def get_metadata(self):
        meta = super().get_metadata()
        meta["has_loader"] = self._loader is not None
        meta["has_fallback"] = self._fallback is not None
        return meta


__all__ = ["LazyDescriptor", "AsyncLoadDescriptor", "AsyncLoadState"]
