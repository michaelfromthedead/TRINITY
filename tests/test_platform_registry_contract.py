"""Contract tests for BackendRegistry (T-P1-001).

CLEANROOM: tests the PUBLIC API contract only.
No knowledge of internal representation, data structures, or algorithms.

Contract sources:
  - PHASE_1_ARCH.md section ADR-P1-001 (Backend Registry Pattern)
  - PHASE_1_TODO.md T-P1-001 (Create Generic Backend Registry)
  - PHASE_1_TODO.md T-P1-004 (Write Unit Tests for BackendRegistry)

BackendRegistry contract:
  register(name, backend_cls, set_default=False) -> None
  get(name) -> type[T] | None
  create(name=None, *args, **kwargs) -> T
  list() -> list[str]
  default() -> str | None
"""
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from engine.platform.registry import BackendRegistry


# ---- Test helpers -----------------------------------------------------------

class _BackendAlpha:
    """Concrete backend used for contract testing (no-op)."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _BackendBeta:
    """Another concrete backend for multi-backend scenarios."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _BackendGamma:
    """Third backend for list-order and default-change verification."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


# =============================================================================
# Equivalence Class: Registration
# =============================================================================

class TestRegister:
    """Backend classes can be registered by name."""

    def test_register_and_get(self):
        """register stores a class retrievable by get()."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha)
        cls = reg.get("alpha")
        assert cls is _BackendAlpha

    def test_register_multiple(self):
        """Multiple backends can be registered independently."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha)
        reg.register("beta", _BackendBeta)
        assert reg.get("alpha") is _BackendAlpha
        assert reg.get("beta") is _BackendBeta

    def test_register_overwrite_same_name(self):
        """Re-registering with the same name overwrites the stored class."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("x", _BackendAlpha)
        reg.register("x", _BackendBeta)
        assert reg.get("x") is _BackendBeta

    def test_register_with_set_default(self):
        """register with set_default=True makes this backend the default."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha, set_default=True)
        assert reg.default() == "alpha"

    def test_register_without_set_default_does_not_change_default(self):
        """register with set_default=False (or omitted) does not touch default."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha, set_default=True)
        reg.register("beta", _BackendBeta)  # no set_default
        assert reg.default() == "alpha"


# =============================================================================
# Equivalence Class: Get
# =============================================================================

class TestGet:
    """Backend classes can be retrieved by name."""

    def test_get_returns_none_for_unknown(self):
        """get returns None when the name has not been registered."""
        reg = BackendRegistry[_BackendAlpha]()
        assert reg.get("nonexistent") is None

    def test_get_returns_class_object(self):
        """get returns the class object, not an instance."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha)
        cls = reg.get("alpha")
        assert cls is _BackendAlpha
        # Verify it is the class itself (can be instantiated separately)
        instance = cls()
        assert isinstance(instance, _BackendAlpha)


# =============================================================================
# Equivalence Class: Create
# =============================================================================

class TestCreate:
    """Backend instances can be created via create()."""

    def test_create_with_default(self):
        """create() without arguments instantiates the default backend."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha, set_default=True)
        instance = reg.create()
        assert isinstance(instance, _BackendAlpha)

    def test_create_with_explicit_name(self):
        """create(name) instantiates the named backend."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha)
        reg.register("beta", _BackendBeta, set_default=True)
        instance = reg.create("alpha")
        assert isinstance(instance, _BackendAlpha)

    def test_create_with_positional_args(self):
        """create passes positional *args to the backend constructor."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha, set_default=True)
        instance = reg.create("alpha", 1, 2, 3)
        assert instance.args == (1, 2, 3)

    def test_create_with_keyword_args(self):
        """create passes **kwargs to the backend constructor."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha, set_default=True)
        instance = reg.create("alpha", x=10, y=20)
        assert instance.kwargs == {"x": 10, "y": 20}

    def test_create_with_combined_args(self):
        """create passes both positional and keyword arguments."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha, set_default=True)
        instance = reg.create("alpha", 42, mode="test")
        assert instance.args == (42,)
        assert instance.kwargs == {"mode": "test"}


# =============================================================================
# Error Cases: Create with invalid inputs
# =============================================================================

class TestCreateErrors:
    """create raises ValueError for invalid backend selection."""

    def test_create_unknown_raises(self):
        """create with a name that was never registered raises ValueError."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha, set_default=True)
        import pytest
        with pytest.raises(ValueError):
            reg.create("nonexistent")

    def test_create_no_default_raises(self):
        """create() with no default backend set raises ValueError."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha)  # not set as default
        # No default set, but we don't pass a name either
        import pytest
        with pytest.raises(ValueError):
            reg.create()

    def test_create_empty_registry_raises(self):
        """create() on an empty registry raises ValueError."""
        reg = BackendRegistry[_BackendAlpha]()
        import pytest
        with pytest.raises(ValueError):
            reg.create()


# =============================================================================
# Equivalence Class: List
# =============================================================================

class TestList:
    """Registered backend names can be listed."""

    def test_list_empty(self):
        """list returns an empty list when no backends are registered."""
        reg = BackendRegistry[_BackendAlpha]()
        assert reg.list() == []

    def test_list_after_single_registration(self):
        """list contains the name of a single registered backend."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha)
        names = reg.list()
        assert "alpha" in names
        assert len(names) == 1

    def test_list_after_multiple_registrations(self):
        """list contains all registered backend names."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha)
        reg.register("beta", _BackendBeta)
        reg.register("gamma", _BackendGamma)
        names = reg.list()
        assert sorted(names) == sorted(["alpha", "beta", "gamma"])

    def test_list_returns_copy(self):
        """list returns a new list, not an internal reference."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha)
        names = reg.list()
        names.append("spoof")
        # The registry should not be affected by mutating the returned list
        assert "spoof" not in reg.list()


# =============================================================================
# Equivalence Class: Default
# =============================================================================

class TestDefault:
    """Default backend name management."""

    def test_default_is_none_initially(self):
        """default returns None when no backend has been set as default."""
        reg = BackendRegistry[_BackendAlpha]()
        assert reg.default() is None

    def test_default_after_set_default(self):
        """default returns the name of the backend registered with set_default."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha, set_default=True)
        assert reg.default() == "alpha"

    def test_default_changes_with_new_set_default(self):
        """default updates when a new backend is registered with set_default."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha, set_default=True)
        reg.register("beta", _BackendBeta, set_default=True)
        assert reg.default() == "beta"

    def test_default_persists_without_set_default(self):
        """default stays unchanged when registering without set_default."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("alpha", _BackendAlpha, set_default=True)
        reg.register("beta", _BackendBeta)  # no set_default
        assert reg.default() == "alpha"


# =============================================================================
# Thread Safety
# =============================================================================

class TestThreadSafety:
    """Registry operations are thread-safe under concurrent access."""

    def test_concurrent_registration(self):
        """Multiple threads can register backends without data corruption."""
        reg = BackendRegistry[_BackendAlpha]()
        n_threads = 8

        def register_backend(i: int):
            name = f"thread_backend_{i}"
            reg.register(name, _BackendAlpha)

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [pool.submit(register_backend, i) for i in range(n_threads)]
            wait(futures)

        names = reg.list()
        assert len(names) == n_threads
        for i in range(n_threads):
            assert f"thread_backend_{i}" in names

    def test_concurrent_registration_and_read(self):
        """Concurrent registration and create do not cause crashes."""
        reg = BackendRegistry[_BackendAlpha]()
        reg.register("default", _BackendAlpha, set_default=True)

        n_ops = 10
        errors = []
        lock = threading.Lock()

        def register_task(i: int):
            try:
                name = f"dyn_{i}"
                reg.register(name, _BackendBeta)
            except Exception as e:
                with lock:
                    errors.append(e)

        def create_task():
            try:
                for _ in range(5):
                    reg.create("default")
            except Exception as e:
                with lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=6) as pool:
            reg_futures = [pool.submit(register_task, i) for i in range(n_ops)]
            create_futures = [pool.submit(create_task) for _ in range(4)]
            wait(reg_futures + create_futures)

        assert not errors, f"Errors during concurrent ops: {errors}"
        # All dynamic registrations visible
        for i in range(n_ops):
            assert f"dyn_{i}" in reg.list()


# =============================================================================
# Generic Type Propagation
# =============================================================================

class TestGenericType:
    """The generic type parameter T propagates correctly."""

    def test_registry_type_annotation(self):
        """BackendRegistry can be parameterised with different backend types.

        This test verifies the generic type is structurally preserved
        (runtime checking of generic origin).
        """
        from typing import Generic
        # Verify BackendRegistry is generic
        assert hasattr(BackendRegistry, "__class_getitem__") or hasattr(BackendRegistry, "__orig_bases__")
        # Verify we can parameterise it
        param_registry = BackendRegistry[_BackendAlpha]
        assert param_registry is not None

    def test_generic_specialisation(self):
        """BackendRegistry[_BackendAlpha] and BackendRegistry[_BackendBeta]
        are distinct parameterisations."""
        alpha_reg = BackendRegistry[_BackendAlpha]()
        beta_reg = BackendRegistry[_BackendBeta]()
        # They should each be valid BackendRegistry instances
        assert isinstance(alpha_reg, BackendRegistry)
        assert isinstance(beta_reg, BackendRegistry)
