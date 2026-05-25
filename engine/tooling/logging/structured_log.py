"""Structured logging with key-value pairs and spans for analytics.

Provides structured logging capabilities for distributed tracing and analytics.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Generator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .log_system import LogSystem, LogLevel, LogCategory


@dataclass
class SpanContext:
    """Context for distributed tracing spans."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: dict = field(default_factory=dict)

    @classmethod
    def new_root(cls) -> SpanContext:
        """Create a new root span context.

        Returns:
            New SpanContext with unique IDs
        """
        return cls(
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4())[:16],
        )

    def new_child(self) -> SpanContext:
        """Create a child span context.

        Returns:
            New SpanContext with this as parent
        """
        return SpanContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=self.span_id,
            baggage=dict(self.baggage)
        )


@dataclass
class Span:
    """A span representing a unit of work.

    Spans track timing and can have child spans.
    """
    name: str
    context: SpanContext
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    attributes: dict = field(default_factory=dict)
    events: list[tuple[datetime, str, dict]] = field(default_factory=list)
    status: str = "ok"  # "ok", "error", "cancelled"
    error_message: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds.

        Returns:
            Duration in milliseconds
        """
        end = self.end_time or datetime.now()
        delta = end - self.start_time
        return delta.total_seconds() * 1000

    @property
    def is_finished(self) -> bool:
        """Check if span has ended."""
        return self.end_time is not None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        """Add an event to the span.

        Args:
            name: Event name
            attributes: Event attributes
        """
        self.events.append((datetime.now(), name, attributes or {}))

    def finish(self, status: str = "ok", error_message: Optional[str] = None) -> None:
        """Finish the span.

        Args:
            status: Final status
            error_message: Error message if status is "error"
        """
        self.end_time = datetime.now()
        self.status = status
        if error_message:
            self.error_message = error_message

    def to_dict(self) -> dict:
        """Convert span to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": [
                {"time": t.isoformat(), "name": n, "attributes": a}
                for t, n, a in self.events
            ],
            "status": self.status,
            "error_message": self.error_message,
        }


class LogContext:
    """Thread-local logging context.

    Provides context data that is automatically included in log messages.
    """
    _local = threading.local()

    def __init__(self):
        """Initialize context."""
        self._data: dict[str, Any] = {}
        self._span_stack: list[Span] = []

    @classmethod
    def current(cls) -> LogContext:
        """Get the current thread's log context.

        Returns:
            Current LogContext
        """
        if not hasattr(cls._local, 'context'):
            cls._local.context = cls()
        return cls._local.context

    @classmethod
    def reset(cls) -> None:
        """Reset the current thread's context."""
        if hasattr(cls._local, 'context'):
            del cls._local.context

    @property
    def data(self) -> dict[str, Any]:
        """Get context data."""
        return dict(self._data)

    @property
    def current_span(self) -> Optional[Span]:
        """Get current span if any."""
        return self._span_stack[-1] if self._span_stack else None

    def set(self, key: str, value: Any) -> None:
        """Set context value.

        Args:
            key: Context key
            value: Context value
        """
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get context value.

        Args:
            key: Context key
            default: Default value

        Returns:
            Context value or default
        """
        return self._data.get(key, default)

    def remove(self, key: str) -> None:
        """Remove context value.

        Args:
            key: Context key
        """
        self._data.pop(key, None)

    def clear(self) -> None:
        """Clear all context data."""
        self._data.clear()

    def push_span(self, span: Span) -> None:
        """Push a span onto the stack.

        Args:
            span: Span to push
        """
        self._span_stack.append(span)

    def pop_span(self) -> Optional[Span]:
        """Pop a span from the stack.

        Returns:
            Popped span or None
        """
        return self._span_stack.pop() if self._span_stack else None

    @contextmanager
    def scope(self, **values) -> Generator[LogContext, None, None]:
        """Context manager for temporary context values.

        Args:
            **values: Values to set temporarily

        Yields:
            This LogContext
        """
        old_values = {}
        for key, value in values.items():
            if key in self._data:
                old_values[key] = self._data[key]
            self._data[key] = value

        try:
            yield self
        finally:
            for key in values:
                if key in old_values:
                    self._data[key] = old_values[key]
                else:
                    self._data.pop(key, None)


class StructuredLogger:
    """Structured logger with context and span support.

    Provides higher-level logging with automatic context injection.
    """
    __slots__ = (
        '_log_system', '_category', '_default_level',
        '_span_handlers', '_lock'
    )

    def __init__(
        self,
        log_system: Optional['LogSystem'] = None,
        category: Optional['LogCategory'] = None,
        default_level: Optional['LogLevel'] = None
    ):
        """Initialize structured logger.

        Args:
            log_system: Log system to use
            category: Default category
            default_level: Default log level
        """
        from .log_system import LogSystem, LogCategory, LogLevel

        self._log_system = log_system or LogSystem.get_instance()
        self._category = category or LogCategory.ENGINE
        self._default_level = default_level or LogLevel.INFO
        self._span_handlers: list[Callable[[Span], None]] = []
        self._lock = threading.Lock()

    @property
    def category(self) -> 'LogCategory':
        """Get default category."""
        return self._category

    @property
    def log_system(self) -> 'LogSystem':
        """Get log system."""
        return self._log_system

    def add_span_handler(self, handler: Callable[[Span], None]) -> None:
        """Add handler for finished spans.

        Args:
            handler: Function to call with finished spans
        """
        with self._lock:
            if handler not in self._span_handlers:
                self._span_handlers.append(handler)

    def remove_span_handler(self, handler: Callable[[Span], None]) -> None:
        """Remove span handler.

        Args:
            handler: Handler to remove
        """
        with self._lock:
            try:
                self._span_handlers.remove(handler)
            except ValueError:
                pass

    def log(
        self,
        message: str,
        level: Optional['LogLevel'] = None,
        category: Optional['LogCategory'] = None,
        exception: Optional[BaseException] = None,
        **extra
    ) -> None:
        """Log a message with context.

        Args:
            message: Log message
            level: Log level
            category: Log category
            exception: Exception to log
            **extra: Additional context data
        """
        ctx = LogContext.current()

        # Merge context
        context = dict(ctx.data)
        context.update(extra)

        # Add span info if present
        span = ctx.current_span
        if span:
            context["trace_id"] = span.context.trace_id
            context["span_id"] = span.context.span_id
            if span.context.parent_span_id:
                context["parent_span_id"] = span.context.parent_span_id

        self._log_system.log(
            level=level or self._default_level,
            category=category or self._category,
            message=message,
            exception=exception,
            context=context
        )

    def trace(self, message: str, **extra) -> None:
        """Log trace message."""
        from .log_system import LogLevel
        self.log(message, level=LogLevel.TRACE, **extra)

    def debug(self, message: str, **extra) -> None:
        """Log debug message."""
        from .log_system import LogLevel
        self.log(message, level=LogLevel.DEBUG, **extra)

    def info(self, message: str, **extra) -> None:
        """Log info message."""
        from .log_system import LogLevel
        self.log(message, level=LogLevel.INFO, **extra)

    def warning(self, message: str, **extra) -> None:
        """Log warning message."""
        from .log_system import LogLevel
        self.log(message, level=LogLevel.WARNING, **extra)

    def error(
        self,
        message: str,
        exception: Optional[BaseException] = None,
        **extra
    ) -> None:
        """Log error message."""
        from .log_system import LogLevel
        self.log(message, level=LogLevel.ERROR, exception=exception, **extra)

    def fatal(
        self,
        message: str,
        exception: Optional[BaseException] = None,
        **extra
    ) -> None:
        """Log fatal message."""
        from .log_system import LogLevel
        self.log(message, level=LogLevel.FATAL, exception=exception, **extra)

    def start_span(
        self,
        name: str,
        parent: Optional[SpanContext] = None,
        **attributes
    ) -> Span:
        """Start a new span.

        Args:
            name: Span name
            parent: Parent span context
            **attributes: Span attributes

        Returns:
            The new span
        """
        ctx = LogContext.current()

        # Create context
        if parent:
            span_context = parent.new_child()
        elif ctx.current_span:
            span_context = ctx.current_span.context.new_child()
        else:
            span_context = SpanContext.new_root()

        span = Span(
            name=name,
            context=span_context,
            attributes=dict(attributes)
        )

        ctx.push_span(span)
        return span

    def end_span(
        self,
        status: str = "ok",
        error_message: Optional[str] = None
    ) -> Optional[Span]:
        """End the current span.

        Args:
            status: Span status
            error_message: Error message if status is "error"

        Returns:
            The finished span
        """
        ctx = LogContext.current()
        span = ctx.pop_span()

        if span:
            span.finish(status, error_message)

            # Notify handlers
            with self._lock:
                for handler in self._span_handlers:
                    try:
                        handler(span)
                    except Exception:
                        pass

        return span

    @contextmanager
    def span(
        self,
        name: str,
        **attributes
    ) -> Generator[Span, None, None]:
        """Context manager for spans.

        Args:
            name: Span name
            **attributes: Span attributes

        Yields:
            The span
        """
        span = self.start_span(name, **attributes)
        try:
            yield span
        except Exception as e:
            span.add_event("exception", {
                "type": type(e).__name__,
                "message": str(e)
            })
            self.end_span("error", str(e))
            raise
        else:
            self.end_span("ok")

    def bind(self, **values) -> LogContext:
        """Bind values to the current context.

        Args:
            **values: Values to bind

        Returns:
            The LogContext
        """
        ctx = LogContext.current()
        for key, value in values.items():
            ctx.set(key, value)
        return ctx

    def unbind(self, *keys) -> None:
        """Unbind values from the current context.

        Args:
            *keys: Keys to remove
        """
        ctx = LogContext.current()
        for key in keys:
            ctx.remove(key)

    @contextmanager
    def context(self, **values) -> Generator[LogContext, None, None]:
        """Context manager for temporary context values.

        Args:
            **values: Temporary values

        Yields:
            LogContext
        """
        ctx = LogContext.current()
        with ctx.scope(**values):
            yield ctx

    def timed(
        self,
        operation: str,
        level: Optional['LogLevel'] = None
    ) -> 'TimedOperation':
        """Create a timed operation logger.

        Args:
            operation: Operation name
            level: Log level for timing message

        Returns:
            TimedOperation context manager
        """
        return TimedOperation(self, operation, level)


class TimedOperation:
    """Context manager for timing operations."""
    __slots__ = ('_logger', '_operation', '_level', '_start_time', '_attributes')

    def __init__(
        self,
        logger: StructuredLogger,
        operation: str,
        level: Optional['LogLevel'] = None
    ):
        """Initialize timed operation.

        Args:
            logger: Structured logger
            operation: Operation name
            level: Log level
        """
        self._logger = logger
        self._operation = operation
        self._level = level
        self._start_time: Optional[float] = None
        self._attributes: dict = {}

    def set_attribute(self, key: str, value: Any) -> None:
        """Set operation attribute.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self._attributes[key] = value

    def __enter__(self) -> TimedOperation:
        """Start timing."""
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing and log."""
        from .log_system import LogLevel

        elapsed = (time.perf_counter() - self._start_time) * 1000

        level = self._level or LogLevel.DEBUG

        if exc_val:
            self._logger.log(
                f"{self._operation} failed after {elapsed:.2f}ms",
                level=LogLevel.ERROR,
                exception=exc_val,
                duration_ms=elapsed,
                **self._attributes
            )
        else:
            self._logger.log(
                f"{self._operation} completed in {elapsed:.2f}ms",
                level=level,
                duration_ms=elapsed,
                **self._attributes
            )
