"""
Tests for reload callbacks and hooks.
"""
import pytest
import time

from engine.tooling.hotreload.reload_callbacks import (
    ReloadCallbacks,
    ReloadPhase,
    ReloadContext,
    CallbackPriority,
    CallbackRegistration,
    get_reload_callbacks,
    on_pre_reload,
    on_post_reload,
)


class TestReloadPhase:
    """Tests for ReloadPhase enum."""

    def test_all_phases_exist(self):
        """Test all expected phases exist."""
        assert hasattr(ReloadPhase, "PRE_RELOAD")
        assert hasattr(ReloadPhase, "STATE_PRESERVED")
        assert hasattr(ReloadPhase, "MODULE_RELOADED")
        assert hasattr(ReloadPhase, "STATE_RESTORED")
        assert hasattr(ReloadPhase, "POST_RELOAD")
        assert hasattr(ReloadPhase, "RELOAD_ERROR")
        assert hasattr(ReloadPhase, "RELOAD_CANCELLED")


class TestCallbackPriority:
    """Tests for CallbackPriority enum."""

    def test_priority_ordering(self):
        """Test priority values are ordered correctly."""
        assert CallbackPriority.HIGHEST.value < CallbackPriority.HIGH.value
        assert CallbackPriority.HIGH.value < CallbackPriority.NORMAL.value
        assert CallbackPriority.NORMAL.value < CallbackPriority.LOW.value
        assert CallbackPriority.LOW.value < CallbackPriority.LOWEST.value


class TestReloadContext:
    """Tests for ReloadContext."""

    def test_context_creation(self):
        """Test creating a reload context."""
        ctx = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="test_module",
        )

        assert ctx.phase == ReloadPhase.PRE_RELOAD
        assert ctx.module_name == "test_module"
        assert ctx.timestamp > 0
        assert ctx.abort is False
        assert ctx.error is None

    def test_context_abort(self):
        """Test aborting a reload via context."""
        ctx = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="test_module",
        )

        ctx.abort = True
        ctx.abort_reason = "Testing abort"

        assert ctx.abort is True
        assert ctx.abort_reason == "Testing abort"


class TestCallbackRegistration:
    """Tests for CallbackRegistration."""

    def test_registration_creation(self):
        """Test creating a callback registration."""
        def callback(ctx):
            pass

        reg = CallbackRegistration(
            callback=callback,
            priority=CallbackPriority.NORMAL,
            phases={ReloadPhase.PRE_RELOAD, ReloadPhase.POST_RELOAD},
        )

        assert reg.callback is callback
        assert reg.priority == CallbackPriority.NORMAL
        assert ReloadPhase.PRE_RELOAD in reg.phases

    def test_registration_matches(self):
        """Test matching logic."""
        def callback(ctx):
            pass

        reg = CallbackRegistration(
            callback=callback,
            priority=CallbackPriority.NORMAL,
            phases={ReloadPhase.PRE_RELOAD},
            module_filter="test_module",
        )

        ctx_match = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="test_module",
        )
        assert reg.matches(ctx_match) is True

        ctx_wrong_phase = ReloadContext(
            phase=ReloadPhase.POST_RELOAD,
            module_name="test_module",
        )
        assert reg.matches(ctx_wrong_phase) is False

        ctx_wrong_module = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="other_module",
        )
        assert reg.matches(ctx_wrong_module) is False

    def test_registration_disabled(self):
        """Test disabled registration doesn't match."""
        def callback(ctx):
            pass

        reg = CallbackRegistration(
            callback=callback,
            priority=CallbackPriority.NORMAL,
            phases={ReloadPhase.PRE_RELOAD},
            enabled=False,
        )

        ctx = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="test",
        )
        assert reg.matches(ctx) is False


class TestReloadCallbacks:
    """Tests for ReloadCallbacks manager."""

    def setup_method(self):
        """Create fresh callback manager for each test."""
        self.callbacks = ReloadCallbacks()
        self.invoked = []

    def test_callbacks_initialization(self):
        """Test ReloadCallbacks initializes correctly."""
        assert self.callbacks.callback_count == 0
        assert self.callbacks.invocation_count == 0

    def test_register_callback(self):
        """Test registering a callback."""
        def callback(ctx):
            self.invoked.append(ctx)

        reg_id = self.callbacks.register(callback)

        assert reg_id is not None
        assert self.callbacks.callback_count == 1

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        def callback(ctx):
            pass

        reg_id = self.callbacks.register(callback)
        result = self.callbacks.unregister(reg_id)

        assert result is True
        assert self.callbacks.callback_count == 0

    def test_unregister_by_function(self):
        """Test unregistering by callback function."""
        def callback(ctx):
            pass

        self.callbacks.register(callback)
        result = self.callbacks.unregister(callback)

        assert result is True
        assert self.callbacks.callback_count == 0

    def test_enable_disable_callback(self):
        """Test enabling/disabling a callback."""
        def callback(ctx):
            self.invoked.append(ctx)

        reg_id = self.callbacks.register(callback)

        # Disable
        self.callbacks.disable(reg_id)

        ctx = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="test",
        )
        self.callbacks.invoke(ctx)

        assert len(self.invoked) == 0

        # Enable
        self.callbacks.enable(reg_id)
        self.callbacks.invoke(ctx)

        assert len(self.invoked) == 1

    def test_invoke_callbacks(self):
        """Test invoking callbacks."""
        def callback(ctx):
            self.invoked.append(ctx.phase)

        self.callbacks.register(
            callback,
            phases={ReloadPhase.PRE_RELOAD},
        )

        ctx = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="test",
        )
        self.callbacks.invoke(ctx)

        assert ReloadPhase.PRE_RELOAD in self.invoked

    def test_invoke_respects_phases(self):
        """Test that callbacks only invoked for registered phases."""
        def callback(ctx):
            self.invoked.append(ctx.phase)

        self.callbacks.register(
            callback,
            phases={ReloadPhase.PRE_RELOAD},
        )

        # Invoke with POST_RELOAD - should not trigger
        ctx = ReloadContext(
            phase=ReloadPhase.POST_RELOAD,
            module_name="test",
        )
        self.callbacks.invoke(ctx)

        assert len(self.invoked) == 0

    def test_priority_ordering(self):
        """Test callbacks are invoked in priority order."""
        def low_callback(ctx):
            self.invoked.append("low")

        def high_callback(ctx):
            self.invoked.append("high")

        def normal_callback(ctx):
            self.invoked.append("normal")

        self.callbacks.register(low_callback, priority=CallbackPriority.LOW)
        self.callbacks.register(high_callback, priority=CallbackPriority.HIGH)
        self.callbacks.register(normal_callback, priority=CallbackPriority.NORMAL)

        ctx = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="test",
        )
        self.callbacks.invoke(ctx)

        assert self.invoked == ["high", "normal", "low"]

    def test_one_shot_callback(self):
        """Test one-shot callbacks are removed after invocation."""
        def callback(ctx):
            self.invoked.append(ctx)

        self.callbacks.register(callback, once=True)

        ctx = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="test",
        )

        self.callbacks.invoke(ctx)
        assert len(self.invoked) == 1
        assert self.callbacks.callback_count == 0

        self.callbacks.invoke(ctx)
        assert len(self.invoked) == 1  # Not invoked again

    def test_abort_stops_callbacks(self):
        """Test that aborting stops callback chain."""
        def abort_callback(ctx):
            self.invoked.append("abort")
            ctx.abort = True

        def after_callback(ctx):
            self.invoked.append("after")

        self.callbacks.register(
            abort_callback,
            priority=CallbackPriority.HIGH,
        )
        self.callbacks.register(
            after_callback,
            priority=CallbackPriority.LOW,
        )

        ctx = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="test",
        )
        self.callbacks.invoke(ctx)

        assert "abort" in self.invoked
        assert "after" not in self.invoked

    def test_callback_exception_captured(self):
        """Test that callback exceptions are captured."""
        def error_callback(ctx):
            raise ValueError("Test error")

        self.callbacks.register(error_callback)

        ctx = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="test",
        )
        result_ctx = self.callbacks.invoke(ctx)

        assert result_ctx.error is not None
        assert isinstance(result_ctx.error, ValueError)

    def test_module_filter(self):
        """Test module filtering."""
        def callback(ctx):
            self.invoked.append(ctx.module_name)

        self.callbacks.register(
            callback,
            module_filter="specific_module",
        )

        ctx1 = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="specific_module",
        )
        ctx2 = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="other_module",
        )

        self.callbacks.invoke(ctx1)
        self.callbacks.invoke(ctx2)

        assert "specific_module" in self.invoked
        assert "other_module" not in self.invoked

    def test_clear_callbacks(self):
        """Test clearing all callbacks."""
        def callback(ctx):
            pass

        self.callbacks.register(callback)
        self.callbacks.register(callback)

        count = self.callbacks.clear()

        assert count == 2
        assert self.callbacks.callback_count == 0


class TestDecorators:
    """Tests for callback decorators."""

    def setup_method(self):
        self.callbacks = ReloadCallbacks()

    def test_on_pre_reload_decorator(self):
        """Test @on_pre_reload decorator."""
        invoked = []

        @self.callbacks.on_pre_reload()
        def my_callback(ctx):
            invoked.append(ctx)

        ctx = ReloadContext(
            phase=ReloadPhase.PRE_RELOAD,
            module_name="test",
        )
        self.callbacks.invoke(ctx)

        assert len(invoked) == 1

    def test_on_post_reload_decorator(self):
        """Test @on_post_reload decorator."""
        invoked = []

        @self.callbacks.on_post_reload()
        def my_callback(ctx):
            invoked.append(ctx)

        ctx = ReloadContext(
            phase=ReloadPhase.POST_RELOAD,
            module_name="test",
        )
        self.callbacks.invoke(ctx)

        assert len(invoked) == 1

    def test_on_reload_error_decorator(self):
        """Test @on_reload_error decorator."""
        invoked = []

        @self.callbacks.on_reload_error()
        def my_callback(ctx):
            invoked.append(ctx)

        ctx = ReloadContext(
            phase=ReloadPhase.RELOAD_ERROR,
            module_name="test",
        )
        self.callbacks.invoke(ctx)

        assert len(invoked) == 1


class TestGlobalCallbacks:
    """Tests for global callback functions."""

    def test_get_reload_callbacks(self):
        """Test getting global callbacks instance."""
        callbacks = get_reload_callbacks()
        assert callbacks is not None
        assert isinstance(callbacks, ReloadCallbacks)

    def test_global_is_singleton(self):
        """Test global instance is singleton."""
        cb1 = get_reload_callbacks()
        cb2 = get_reload_callbacks()
        assert cb1 is cb2
