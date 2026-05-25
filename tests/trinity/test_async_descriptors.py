"""
Tests for async descriptors: LazyDescriptor, AsyncLoadDescriptor.

Verifies:
- Lazy initialization on first access
- Explicit vs implicit initialization modes
- Async loading with fallback on error
- State transitions during async loading
"""
import pytest
from trinity.descriptors.async_descriptors import LazyDescriptor, AsyncLoadDescriptor, AsyncLoadState


class TestLazyDescriptor:
    """Test LazyDescriptor defers initialization until first access."""

    def test_first_access_initializes(self):
        """In implicit mode, first get should trigger the factory."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return 42

        class Foo:
            value = LazyDescriptor(field_type=int, factory=factory, init_mode="first_access")
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        assert call_count == 0
        result = f.value
        assert result == 42
        assert call_count == 1
        # Second access should not call factory again
        result2 = f.value
        assert result2 == 42
        assert call_count == 1

    def test_explicit_mode_initialize(self):
        """In explicit mode, initialize() should trigger the factory."""
        class Foo:
            value = LazyDescriptor(field_type=int, factory=lambda: 99, init_mode="explicit")
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        Foo.value.initialize(f)
        assert f.value == 99

    def test_is_initialized(self):
        """is_initialized should reflect whether the value has been created."""
        class Foo:
            value = LazyDescriptor(field_type=int, factory=lambda: 7, init_mode="first_access")
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        assert Foo.value.is_initialized(f) is False
        _ = f.value
        assert Foo.value.is_initialized(f) is True

    def test_set_stores_value(self):
        """Setting a value should store it via the descriptor."""
        class Foo:
            value = LazyDescriptor(field_type=int, factory=lambda: 42, init_mode="first_access")
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        # First access triggers factory
        assert f.value == 42
        # After init, set should work
        f.value = 100
        assert f.value == 100

    def test_no_factory_provided(self):
        """LazyDescriptor should work without a factory (manual initialization)."""
        class Foo:
            value = LazyDescriptor(field_type=int, init_mode="explicit")
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        # Manually set value
        f.value = 55
        assert f.value == 55
        assert not Foo.value.is_initialized(f)  # No factory, so no initialization flag

    def test_invalid_init_mode_raises(self):
        """Invalid init_mode should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid init_mode"):
            LazyDescriptor(field_type=int, factory=lambda: 1, init_mode="invalid_mode")

    def test_factory_returns_none(self):
        """Factory returning None should be handled correctly."""
        class Foo:
            value = LazyDescriptor(field_type=object, factory=lambda: None, init_mode="first_access")
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        result = f.value
        assert result is None
        assert Foo.value.is_initialized(f)

    def test_metadata_contains_init_mode(self):
        """Metadata should contain init_mode and factory status."""
        class Foo:
            value = LazyDescriptor(field_type=int, factory=lambda: 1, init_mode="explicit")
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["init_mode"] == "explicit"
        assert meta["has_factory"] is True


class TestAsyncLoadDescriptor:
    """Test AsyncLoadDescriptor loads values asynchronously with fallback."""

    def test_loader_called_on_first_get(self):
        """First access should trigger the loader function."""
        loaded = []

        def loader():
            loaded.append(True)
            return "data"

        class Foo:
            resource = AsyncLoadDescriptor(field_type=str, loader=loader)
        Foo.resource.__set_name__(Foo, 'resource')
        f = Foo()
        result = f.resource
        assert len(loaded) == 1
        assert result == "data"

    def test_fallback_on_error(self):
        """If loader raises, fallback value should be returned."""
        def bad_loader():
            raise IOError("network error")

        class Foo:
            resource = AsyncLoadDescriptor(
                field_type=str,
                loader=bad_loader,
                fallback="default",
            )
        Foo.resource.__set_name__(Foo, 'resource')
        f = Foo()
        result = f.resource
        assert result == "default"

    def test_state_transitions(self):
        """State should transition: pending -> loading -> loaded/error."""
        class Foo:
            resource = AsyncLoadDescriptor(field_type=str, loader=lambda: "ok")
        Foo.resource.__set_name__(Foo, 'resource')
        f = Foo()
        assert Foo.resource.get_state(f) == AsyncLoadState.NOT_STARTED
        _ = f.resource
        assert Foo.resource.get_state(f) == AsyncLoadState.LOADED

    def test_error_state(self):
        """Failed load should transition to error state."""
        def fail_loader():
            raise RuntimeError("fail")

        class Foo:
            resource = AsyncLoadDescriptor(
                field_type=str,
                loader=fail_loader,
                fallback="safe",
            )
        Foo.resource.__set_name__(Foo, 'resource')
        f = Foo()
        _ = f.resource
        assert Foo.resource.get_state(f) == AsyncLoadState.ERROR

    def test_loaded_value_cached(self):
        """Once loaded, subsequent gets should return the cached value."""
        count = 0

        def counting_loader():
            nonlocal count
            count += 1
            return f"v{count}"

        class Foo:
            resource = AsyncLoadDescriptor(field_type=str, loader=counting_loader)
        Foo.resource.__set_name__(Foo, 'resource')
        f = Foo()
        _ = f.resource
        assert count == 1
        _ = f.resource
        assert count == 1  # Should not re-invoke

    def test_no_loader_returns_none(self):
        """AsyncLoadDescriptor without loader should return None or fallback."""
        class Foo:
            resource = AsyncLoadDescriptor(field_type=str, fallback="default_val")
        Foo.resource.__set_name__(Foo, 'resource')
        f = Foo()
        result = f.resource
        assert result == "default_val"

    def test_loader_returns_none(self):
        """Loader returning None should be handled correctly."""
        class Foo:
            resource = AsyncLoadDescriptor(field_type=object, loader=lambda: None, fallback="fallback")
        Foo.resource.__set_name__(Foo, 'resource')
        f = Foo()
        result = f.resource
        assert result is None  # None is a valid loaded value
        assert Foo.resource.get_state(f) == AsyncLoadState.LOADED

    def test_metadata_contains_loader_info(self):
        """Metadata should indicate whether loader and fallback are present."""
        class Foo:
            resource = AsyncLoadDescriptor(field_type=str, loader=lambda: "x", fallback="y")
        Foo.resource.__set_name__(Foo, 'resource')
        meta = Foo.resource.get_metadata()
        assert meta["has_loader"] is True
        assert meta["has_fallback"] is True
