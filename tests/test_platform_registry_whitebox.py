"""
WHITEBOX tests for engine/platform/registry.py (T-P1-001 BackendRegistry).

WHITEBOX coverage plan:
  BackendRegistry.__init__:
    Path A1:  _lock is a threading.Lock instance
              -> test_lock_is_threading_lock
    Path A2:  _backends dict starts empty
              -> test_backends_starts_empty
    Path A3:  _default is None on fresh registry
              -> test_default_is_none_when_not_set

  BackendRegistry.register:
    Path B1:  single register stores backend class under name
              -> test_register_stores_backend_class
    Path B2:  register with set_default=True updates default
              -> test_register_sets_default
    Path B3:  register same name twice overwrites (last wins)
              -> test_register_overwrites_existing_name
    Path B4:  re-registering a different backend as default changes default
              -> test_register_changes_default_on_re_register
    Path B5:  register does NOT set default when set_default is not passed
              -> test_register_without_set_default_leaves_default_unchanged

  BackendRegistry.get:
    Path C1:  get existing backend returns its class
              -> test_get_returns_backend_class
    Path C2:  get unregistered name returns None
              -> test_get_unregistered_returns_none

  BackendRegistry.create:
    Path D1:  create with explicit name instantiates the class
              -> test_create_with_explicit_name
    Path D2:  create with no name uses default backend
              -> test_create_uses_default_when_name_is_none
    Path D3:  create forwards positional and keyword args to constructor
              -> test_create_forwards_constructor_args
    Path D4:  create when name is None and no default set -> ValueError
              -> test_create_no_name_no_default_raises_valueerror
    Path D5:  create with unknown name -> ValueError
              -> test_create_unknown_name_raises_valueerror
    Path D6:  create returns distinct instances each call
              -> test_create_returns_distinct_instances

  BackendRegistry.list:
    Path E1:  empty registry returns empty list
              -> test_list_empty_registry
    Path E2:  list returns sorted names with multiple backends
              -> test_list_returns_sorted_names
    Path E3:  list does not expose internal dict identity (snapshot)
              -> test_list_snapshot_isolation

  BackendRegistry.default:
    Path F1:  default returns None when no default was set
              -> test_default_returns_none_when_not_set
    Path F2:  default returns the name of the default backend when set
              -> test_default_returns_default_name

  Thread safety:
    Path G1:  concurrent register calls from multiple threads do not corrupt state
              -> test_concurrent_register_is_thread_safe
"""

from __future__ import annotations

import threading

import pytest

from engine.platform.registry import BackendRegistry


# =============================================================================
# Helper classes
# =============================================================================


class _AlphaBackend:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.value: int = 0


class _BetaBackend:
    pass


class _GammaBackend(_AlphaBackend):
    pass


# =============================================================================
# Path A — __init__
# =============================================================================


def test_lock_is_threading_lock() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    assert isinstance(reg._lock, threading.Lock)


def test_backends_starts_empty() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    assert reg._backends == {}


def test_default_is_none_when_not_set() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    assert reg._default is None


# =============================================================================
# Path B — register
# =============================================================================


def test_register_stores_backend_class() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("alpha", _AlphaBackend)
    assert reg._backends["alpha"] is _AlphaBackend


def test_register_sets_default() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("alpha", _AlphaBackend, set_default=True)
    assert reg._default == "alpha"


def test_register_overwrites_existing_name() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("x", _AlphaBackend)
    reg.register("x", _BetaBackend)
    assert reg._backends["x"] is _BetaBackend


def test_register_changes_default_on_re_register() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("alpha", _AlphaBackend, set_default=True)
    reg.register("beta", _BetaBackend, set_default=True)
    assert reg._default == "beta"


def test_register_without_set_default_leaves_default_unchanged() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("alpha", _AlphaBackend, set_default=True)
    reg.register("beta", _BetaBackend, set_default=False)
    assert reg._default == "alpha"


# =============================================================================
# Path C — get
# =============================================================================


def test_get_returns_backend_class() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("alpha", _AlphaBackend)
    assert reg.get("alpha") is _AlphaBackend


def test_get_unregistered_returns_none() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    assert reg.get("nonexistent") is None


# =============================================================================
# Path D — create
# =============================================================================


def test_create_with_explicit_name() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("alpha", _AlphaBackend)
    instance = reg.create("alpha")
    assert isinstance(instance, _AlphaBackend)


def test_create_uses_default_when_name_is_none() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("alpha", _AlphaBackend, set_default=True)
    instance = reg.create()
    assert isinstance(instance, _AlphaBackend)


def test_create_forwards_constructor_args() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("alpha", _AlphaBackend, set_default=True)
    instance = reg.create(None, 1, "two", key="value")
    assert instance.args == (1, "two")
    assert instance.kwargs == {"key": "value"}


def test_create_no_name_no_default_raises_valueerror() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    with pytest.raises(ValueError, match="No default backend set"):
        reg.create()


def test_create_unknown_name_raises_valueerror() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    with pytest.raises(ValueError, match="Unknown backend"):
        reg.create("missing")


def test_create_returns_distinct_instances() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("alpha", _AlphaBackend, set_default=True)
    a = reg.create()
    b = reg.create()
    assert a is not b


# =============================================================================
# Path E — list
# =============================================================================


def test_list_empty_registry() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    assert reg.list() == []


def test_list_returns_sorted_names() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("z", _AlphaBackend)
    reg.register("a", _BetaBackend)
    reg.register("m", _GammaBackend)
    assert reg.list() == ["a", "m", "z"]


def test_list_snapshot_isolation() -> None:
    """list() returns a snapshot, not a live reference."""
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("alpha", _AlphaBackend)
    snapshot = reg.list()
    reg.register("beta", _BetaBackend)
    # snapshot taken before beta was registered should not include it
    assert snapshot == ["alpha"]


# =============================================================================
# Path F — default
# =============================================================================


def test_default_returns_none_when_not_set() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    assert reg.default() is None


def test_default_returns_default_name() -> None:
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()
    reg.register("alpha", _AlphaBackend, set_default=True)
    assert reg.default() == "alpha"


# =============================================================================
# Path G — Thread safety
# =============================================================================


def test_concurrent_register_is_thread_safe() -> None:
    """Concurrent register() calls from multiple threads produce a consistent
    final state with all names present and no partial writes."""
    reg: BackendRegistry[_AlphaBackend] = BackendRegistry()

    # Zero-pad so lexicographic sort matches numeric order
    names = [f"b{i:03d}" for i in range(64)]

    def _register(n: str) -> None:
        reg.register(n, _AlphaBackend, set_default=(n == names[-1]))

    threads = [threading.Thread(target=_register, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    registered = reg.list()
    assert sorted(registered) == names
    assert reg.default() == names[-1]
