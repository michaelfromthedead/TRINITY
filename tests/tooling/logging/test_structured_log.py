"""Tests for structured logging.

Tests structured logger, context, and spans.
"""

import pytest
import time
import threading
from datetime import datetime

from engine.tooling.logging.log_system import LogSystem, LogLevel, LogCategory, LogConfig
from engine.tooling.logging.log_targets import RingBufferTarget
from engine.tooling.logging.structured_log import (
    StructuredLogger,
    LogContext,
    Span,
    SpanContext,
    TimedOperation,
)


@pytest.fixture(autouse=True)
def reset_context():
    """Reset thread-local context before and after each test."""
    LogContext.reset()
    yield
    LogContext.reset()


@pytest.fixture(autouse=True)
def reset_log_system():
    """Reset log system singleton."""
    LogSystem.reset_instance()
    yield
    LogSystem.reset_instance()


class TestSpanContext:
    """Tests for SpanContext."""

    def test_new_root(self):
        ctx = SpanContext.new_root()

        assert ctx.trace_id is not None
        assert ctx.span_id is not None
        assert ctx.parent_span_id is None

    def test_new_child(self):
        root = SpanContext.new_root()
        child = root.new_child()

        assert child.trace_id == root.trace_id
        assert child.span_id != root.span_id
        assert child.parent_span_id == root.span_id

    def test_baggage_inherited(self):
        root = SpanContext.new_root()
        root.baggage["key"] = "value"

        child = root.new_child()
        assert child.baggage["key"] == "value"


class TestSpan:
    """Tests for Span."""

    def test_basic_creation(self):
        ctx = SpanContext.new_root()
        span = Span(name="test_operation", context=ctx)

        assert span.name == "test_operation"
        assert span.is_finished is False
        assert span.status == "ok"

    def test_duration(self):
        ctx = SpanContext.new_root()
        span = Span(name="test", context=ctx)

        time.sleep(0.01)
        span.finish()

        assert span.duration_ms >= 10

    def test_set_attribute(self):
        ctx = SpanContext.new_root()
        span = Span(name="test", context=ctx)

        span.set_attribute("user_id", "123")
        span.set_attribute("count", 42)

        assert span.attributes["user_id"] == "123"
        assert span.attributes["count"] == 42

    def test_add_event(self):
        ctx = SpanContext.new_root()
        span = Span(name="test", context=ctx)

        span.add_event("checkpoint", {"step": 1})
        span.add_event("complete")

        assert len(span.events) == 2
        assert span.events[0][1] == "checkpoint"
        assert span.events[0][2]["step"] == 1

    def test_finish(self):
        ctx = SpanContext.new_root()
        span = Span(name="test", context=ctx)

        span.finish(status="ok")

        assert span.is_finished is True
        assert span.end_time is not None

    def test_finish_with_error(self):
        ctx = SpanContext.new_root()
        span = Span(name="test", context=ctx)

        span.finish(status="error", error_message="Something failed")

        assert span.status == "error"
        assert span.error_message == "Something failed"

    def test_to_dict(self):
        ctx = SpanContext.new_root()
        span = Span(name="test", context=ctx)
        span.set_attribute("key", "value")
        span.add_event("event1")
        span.finish()

        data = span.to_dict()

        assert data["name"] == "test"
        assert data["trace_id"] == ctx.trace_id
        assert data["span_id"] == ctx.span_id
        assert data["attributes"]["key"] == "value"
        assert len(data["events"]) == 1
        assert data["status"] == "ok"


class TestLogContext:
    """Tests for LogContext."""

    def test_current_context(self):
        ctx = LogContext.current()
        assert ctx is not None

        # Same thread gets same context
        ctx2 = LogContext.current()
        assert ctx is ctx2

    def test_set_and_get(self):
        ctx = LogContext.current()

        ctx.set("key", "value")
        assert ctx.get("key") == "value"
        assert ctx.get("missing", "default") == "default"

    def test_remove(self):
        ctx = LogContext.current()
        ctx.set("key", "value")

        ctx.remove("key")
        assert ctx.get("key") is None

    def test_clear(self):
        ctx = LogContext.current()
        ctx.set("key1", "value1")
        ctx.set("key2", "value2")

        ctx.clear()
        assert ctx.data == {}

    def test_scope_context_manager(self):
        ctx = LogContext.current()
        ctx.set("original", "value")

        with ctx.scope(temporary="temp", original="overridden"):
            assert ctx.get("temporary") == "temp"
            assert ctx.get("original") == "overridden"

        assert ctx.get("temporary") is None
        assert ctx.get("original") == "value"

    def test_span_stack(self):
        ctx = LogContext.current()
        span1 = Span(name="span1", context=SpanContext.new_root())
        span2 = Span(name="span2", context=span1.context.new_child())

        ctx.push_span(span1)
        assert ctx.current_span is span1

        ctx.push_span(span2)
        assert ctx.current_span is span2

        popped = ctx.pop_span()
        assert popped is span2
        assert ctx.current_span is span1

    def test_thread_isolation(self):
        results = {}

        def thread_func(thread_id):
            ctx = LogContext.current()
            ctx.set("thread_id", thread_id)
            time.sleep(0.01)
            results[thread_id] = ctx.get("thread_id")

        threads = [
            threading.Thread(target=thread_func, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should have its own value
        for i in range(5):
            assert results[i] == i


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    @pytest.fixture
    def logger_with_target(self):
        """Create a structured logger with a ring buffer target."""
        config = LogConfig(min_level=LogLevel.TRACE, async_logging=False)
        system = LogSystem(config)
        target = RingBufferTarget()
        system.add_target(target)
        logger = StructuredLogger(log_system=system)
        return logger, target

    def test_basic_log(self, logger_with_target):
        logger, target = logger_with_target

        logger.info("Test message")

        entries = target.get_entries()
        assert len(entries) == 1
        assert entries[0].message.message == "Test message"

    def test_all_levels(self, logger_with_target):
        logger, target = logger_with_target

        logger.trace("Trace")
        logger.debug("Debug")
        logger.info("Info")
        logger.warning("Warning")
        logger.error("Error")
        logger.fatal("Fatal")

        entries = target.get_entries()
        assert len(entries) == 6

    def test_extra_context(self, logger_with_target):
        logger, target = logger_with_target

        logger.info("Message", user_id="123", action="login")

        entries = target.get_entries()
        assert entries[0].message.context["user_id"] == "123"
        assert entries[0].message.context["action"] == "login"

    def test_bound_context(self, logger_with_target):
        logger, target = logger_with_target

        logger.bind(request_id="abc123")
        logger.info("First message")
        logger.info("Second message")

        entries = target.get_entries()
        assert entries[0].message.context["request_id"] == "abc123"
        assert entries[1].message.context["request_id"] == "abc123"

    def test_unbind(self, logger_with_target):
        logger, target = logger_with_target

        logger.bind(request_id="abc123")
        logger.unbind("request_id")
        logger.info("Message")

        entries = target.get_entries()
        assert "request_id" not in entries[0].message.context

    def test_context_manager(self, logger_with_target):
        logger, target = logger_with_target

        with logger.context(request_id="temp"):
            logger.info("Inside context")
            assert LogContext.current().get("request_id") == "temp"

        logger.info("Outside context")

        entries = target.get_entries()
        assert entries[0].message.context.get("request_id") == "temp"
        assert "request_id" not in entries[1].message.context


class TestStructuredLoggerSpans:
    """Tests for span management in StructuredLogger."""

    @pytest.fixture
    def logger_with_target(self):
        config = LogConfig(min_level=LogLevel.TRACE, async_logging=False)
        system = LogSystem(config)
        target = RingBufferTarget()
        system.add_target(target)
        logger = StructuredLogger(log_system=system)
        return logger, target

    def test_start_span(self, logger_with_target):
        logger, target = logger_with_target

        span = logger.start_span("test_operation")

        assert span is not None
        assert span.name == "test_operation"
        assert LogContext.current().current_span is span

    def test_end_span(self, logger_with_target):
        logger, target = logger_with_target

        logger.start_span("test_operation")
        span = logger.end_span()

        assert span is not None
        assert span.is_finished
        assert LogContext.current().current_span is None

    def test_nested_spans(self, logger_with_target):
        logger, target = logger_with_target

        span1 = logger.start_span("outer")
        span2 = logger.start_span("inner")

        assert span2.context.parent_span_id == span1.context.span_id

        logger.end_span()
        assert LogContext.current().current_span is span1

        logger.end_span()
        assert LogContext.current().current_span is None

    def test_span_context_manager(self, logger_with_target):
        logger, target = logger_with_target

        with logger.span("test_operation", key="value") as span:
            assert span.name == "test_operation"
            assert span.attributes["key"] == "value"
            logger.info("Inside span")

        # Span should be finished
        assert span.is_finished
        assert span.status == "ok"

        entries = target.get_entries()
        assert entries[0].message.context["trace_id"] == span.context.trace_id

    def test_span_with_exception(self, logger_with_target):
        logger, target = logger_with_target

        with pytest.raises(ValueError):
            with logger.span("failing_operation") as span:
                raise ValueError("Test error")

        assert span.status == "error"
        assert "ValueError" in span.events[0][2]["type"]

    def test_span_handler(self, logger_with_target):
        logger, target = logger_with_target
        received_spans = []

        def span_handler(span):
            received_spans.append(span)

        logger.add_span_handler(span_handler)

        with logger.span("test"):
            pass

        assert len(received_spans) == 1
        assert received_spans[0].name == "test"

    def test_remove_span_handler(self, logger_with_target):
        logger, target = logger_with_target
        received_spans = []

        def span_handler(span):
            received_spans.append(span)

        logger.add_span_handler(span_handler)
        logger.remove_span_handler(span_handler)

        with logger.span("test"):
            pass

        assert len(received_spans) == 0


class TestTimedOperation:
    """Tests for TimedOperation."""

    @pytest.fixture
    def logger_with_target(self):
        config = LogConfig(min_level=LogLevel.TRACE, async_logging=False)
        system = LogSystem(config)
        target = RingBufferTarget()
        system.add_target(target)
        logger = StructuredLogger(log_system=system)
        return logger, target

    def test_basic_timing(self, logger_with_target):
        logger, target = logger_with_target

        with logger.timed("test_operation"):
            time.sleep(0.01)

        entries = target.get_entries()
        assert len(entries) == 1
        assert "completed" in entries[0].message.message
        assert entries[0].message.context["duration_ms"] >= 10

    def test_timing_with_error(self, logger_with_target):
        logger, target = logger_with_target

        with pytest.raises(ValueError):
            with logger.timed("failing_operation"):
                raise ValueError("Error")

        entries = target.get_entries()
        assert "failed" in entries[0].message.message
        assert entries[0].message.level == LogLevel.ERROR

    def test_set_attribute(self, logger_with_target):
        logger, target = logger_with_target

        with logger.timed("operation") as op:
            op.set_attribute("items_processed", 100)

        entries = target.get_entries()
        assert entries[0].message.context["items_processed"] == 100

    def test_custom_level(self, logger_with_target):
        logger, target = logger_with_target

        with logger.timed("operation", level=LogLevel.INFO):
            pass

        entries = target.get_entries()
        assert entries[0].message.level == LogLevel.INFO
