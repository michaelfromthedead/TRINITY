"""Tests for IPC module.

This module tests the IPC protocol parsing, handler routing, and error handling
for the FlowForge backend JSON-RPC 2.0 implementation.
"""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock

from ..ipc.protocol import IPCRequest, IPCResponse, IPCError
from ..ipc.handler import Handler, create_default_handler
from ..config import ErrorCodes, JSONRPC_VERSION


def run_async(coro):
    """Helper to run async functions in tests without pytest-asyncio."""
    return asyncio.new_event_loop().run_until_complete(coro)


# =============================================================================
# IPCError Tests
# =============================================================================


class TestIPCError:
    """Tests for IPCError class."""

    def test_basic_error_creation(self):
        """Test creating a basic error."""
        error = IPCError(code=-32600, message="Invalid request")

        assert error.code == -32600
        assert error.message == "Invalid request"
        assert error.data is None

    def test_error_with_data(self):
        """Test error with additional data."""
        error = IPCError(code=-32602, message="Invalid params", data={"field": "graph"})

        assert error.code == -32602
        assert error.data == {"field": "graph"}

    def test_error_to_dict_without_data(self):
        """Test to_dict without optional data."""
        error = IPCError(code=-32700, message="Parse error")
        result = error.to_dict()

        assert result == {"code": -32700, "message": "Parse error"}
        assert "data" not in result

    def test_error_to_dict_with_data(self):
        """Test to_dict with optional data."""
        error = IPCError(code=-32700, message="Parse error", data={"detail": "unexpected EOF"})
        result = error.to_dict()

        assert result == {
            "code": -32700,
            "message": "Parse error",
            "data": {"detail": "unexpected EOF"},
        }

    def test_error_is_frozen(self):
        """Test that IPCError is immutable (frozen dataclass)."""
        error = IPCError(code=-32600, message="Test")

        with pytest.raises(AttributeError):
            error.code = -32700

    def test_standard_error_codes(self):
        """Test that standard error codes are defined."""
        assert IPCError.PARSE_ERROR == ErrorCodes.PARSE_ERROR
        assert IPCError.INVALID_REQUEST == ErrorCodes.INVALID_REQUEST
        assert IPCError.METHOD_NOT_FOUND == ErrorCodes.METHOD_NOT_FOUND
        assert IPCError.INVALID_PARAMS == ErrorCodes.INVALID_PARAMS
        assert IPCError.INTERNAL_ERROR == ErrorCodes.INTERNAL_ERROR


class TestIPCErrorFactoryMethods:
    """Tests for IPCError factory methods."""

    def test_parse_error_default_message(self):
        """Test parse_error with default message."""
        error = IPCError.parse_error()

        assert error.code == ErrorCodes.PARSE_ERROR
        assert error.message == "Parse error"

    def test_parse_error_custom_message(self):
        """Test parse_error with custom message."""
        error = IPCError.parse_error("Unexpected token at position 5")

        assert error.code == ErrorCodes.PARSE_ERROR
        assert error.message == "Unexpected token at position 5"

    def test_invalid_request_default_message(self):
        """Test invalid_request with default message."""
        error = IPCError.invalid_request()

        assert error.code == ErrorCodes.INVALID_REQUEST
        assert error.message == "Invalid request"

    def test_invalid_request_custom_message(self):
        """Test invalid_request with custom message."""
        error = IPCError.invalid_request("Missing 'id' field")

        assert error.code == ErrorCodes.INVALID_REQUEST
        assert error.message == "Missing 'id' field"

    def test_method_not_found(self):
        """Test method_not_found factory."""
        error = IPCError.method_not_found("unknown_method")

        assert error.code == ErrorCodes.METHOD_NOT_FOUND
        assert "unknown_method" in error.message
        assert "Method not found" in error.message

    def test_invalid_params_default_message(self):
        """Test invalid_params with default message."""
        error = IPCError.invalid_params()

        assert error.code == ErrorCodes.INVALID_PARAMS
        assert error.message == "Invalid params"

    def test_invalid_params_custom_message(self):
        """Test invalid_params with custom message."""
        error = IPCError.invalid_params("Parameter 'source' must be a string")

        assert error.code == ErrorCodes.INVALID_PARAMS
        assert error.message == "Parameter 'source' must be a string"

    def test_internal_error_default_message(self):
        """Test internal_error with default message."""
        error = IPCError.internal_error()

        assert error.code == ErrorCodes.INTERNAL_ERROR
        assert error.message == "Internal error"

    def test_internal_error_custom_message(self):
        """Test internal_error with custom message."""
        error = IPCError.internal_error("Database connection failed")

        assert error.code == ErrorCodes.INTERNAL_ERROR
        assert error.message == "Database connection failed"


# =============================================================================
# IPCRequest Tests - Protocol Parsing
# =============================================================================


class TestIPCRequestFromDict:
    """Tests for IPCRequest.from_dict parsing."""

    def test_valid_jsonrpc_request(self):
        """Test parsing valid JSON-RPC 2.0 request."""
        data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "ping",
            "params": {"timeout": 5000},
        }
        request = IPCRequest.from_dict(data)

        assert request.jsonrpc == "2.0"
        assert request.id == 1
        assert request.method == "ping"
        assert request.params == {"timeout": 5000}

    def test_valid_request_with_string_id(self):
        """Test parsing request with string ID."""
        data = {
            "jsonrpc": "2.0",
            "id": "abc-123",
            "method": "test",
            "params": {},
        }
        request = IPCRequest.from_dict(data)

        assert request.id == "abc-123"

    def test_valid_request_with_integer_id(self):
        """Test parsing request with integer ID."""
        data = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "test",
            "params": {},
        }
        request = IPCRequest.from_dict(data)

        assert request.id == 42

    def test_simplified_format_without_jsonrpc(self):
        """Test parsing simplified format (jsonrpc field optional)."""
        data = {
            "id": "1",
            "method": "ping",
            "params": {},
        }
        request = IPCRequest.from_dict(data)

        assert request.jsonrpc == JSONRPC_VERSION
        assert request.method == "ping"

    def test_default_empty_params(self):
        """Test that params defaults to empty dict."""
        data = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "test",
        }
        request = IPCRequest.from_dict(data)

        assert request.params == {}

    def test_missing_id_raises_valueerror(self):
        """Test that missing id field raises ValueError."""
        data = {"method": "test", "params": {}}

        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_dict(data)

        assert "id" in str(exc_info.value).lower()

    def test_missing_method_raises_valueerror(self):
        """Test that missing method field raises ValueError."""
        data = {"id": 1, "params": {}}

        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_dict(data)

        assert "method" in str(exc_info.value).lower()

    def test_invalid_id_type_raises_valueerror(self):
        """Test that invalid id type raises ValueError."""
        data = {"id": [1, 2], "method": "test"}

        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_dict(data)

        assert "id" in str(exc_info.value).lower()

    def test_invalid_method_type_raises_valueerror(self):
        """Test that invalid method type raises ValueError."""
        data = {"id": 1, "method": 123}

        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_dict(data)

        assert "method" in str(exc_info.value).lower()

    def test_invalid_params_type_raises_valueerror(self):
        """Test that invalid params type raises ValueError."""
        data = {"id": 1, "method": "test", "params": "not an object"}

        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_dict(data)

        assert "params" in str(exc_info.value).lower()

    def test_invalid_jsonrpc_type_raises_valueerror(self):
        """Test that invalid jsonrpc type raises ValueError."""
        data = {"jsonrpc": 2.0, "id": 1, "method": "test"}

        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_dict(data)

        assert "jsonrpc" in str(exc_info.value).lower()

    def test_unsupported_jsonrpc_version_raises_valueerror(self):
        """Test that unsupported JSON-RPC version raises ValueError."""
        data = {"jsonrpc": "1.0", "id": 1, "method": "test"}

        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_dict(data)

        assert "version" in str(exc_info.value).lower() or "1.0" in str(exc_info.value)

    def test_null_id_type_raises_valueerror(self):
        """Test that null id type raises ValueError."""
        data = {"id": None, "method": "test"}

        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_dict(data)

        assert "id" in str(exc_info.value).lower()

    def test_float_id_type_raises_valueerror(self):
        """Test that float id type raises ValueError."""
        data = {"id": 1.5, "method": "test"}

        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_dict(data)

        assert "id" in str(exc_info.value).lower()


class TestIPCRequestFromJson:
    """Tests for IPCRequest.from_json parsing."""

    def test_valid_json_string(self):
        """Test parsing valid JSON string."""
        json_str = '{"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}'
        request = IPCRequest.from_json(json_str)

        assert request.id == 1
        assert request.method == "ping"

    def test_invalid_json_raises_valueerror(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_json("{invalid json}")

        assert "json" in str(exc_info.value).lower()

    def test_json_array_raises_valueerror(self):
        """Test that JSON array raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_json('[1, 2, 3]')

        assert "object" in str(exc_info.value).lower()

    def test_json_primitive_raises_valueerror(self):
        """Test that JSON primitive raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_json('"just a string"')

        assert "object" in str(exc_info.value).lower()

    def test_empty_string_raises_valueerror(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_json("")

        assert "json" in str(exc_info.value).lower()

    def test_whitespace_only_raises_valueerror(self):
        """Test that whitespace-only string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            IPCRequest.from_json("   \n\t   ")

        assert "json" in str(exc_info.value).lower()


class TestIPCRequestSerialization:
    """Tests for IPCRequest serialization methods."""

    def test_to_dict(self):
        """Test request to_dict serialization."""
        request = IPCRequest(id=1, method="test", params={"key": "value"})
        result = request.to_dict()

        assert result == {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "test",
            "params": {"key": "value"},
        }

    def test_to_json(self):
        """Test request to_json serialization."""
        request = IPCRequest(id=1, method="ping", params={})
        json_str = request.to_json()

        # Parse back to verify
        data = json.loads(json_str)
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert data["method"] == "ping"

    def test_roundtrip_dict(self):
        """Test dict serialization roundtrip."""
        original = IPCRequest(id="abc", method="test", params={"x": 1})
        data = original.to_dict()
        restored = IPCRequest.from_dict(data)

        assert restored.id == original.id
        assert restored.method == original.method
        assert restored.params == original.params

    def test_roundtrip_json(self):
        """Test JSON serialization roundtrip."""
        original = IPCRequest(id=42, method="test", params={"nested": {"key": "value"}})
        json_str = original.to_json()
        restored = IPCRequest.from_json(json_str)

        assert restored.id == original.id
        assert restored.method == original.method
        assert restored.params == original.params


# =============================================================================
# IPCResponse Tests
# =============================================================================


class TestIPCResponse:
    """Tests for IPCResponse class."""

    def test_success_response_creation(self):
        """Test creating a success response."""
        response = IPCResponse.success(1, {"data": "result"})

        assert response.id == 1
        assert response.result == {"data": "result"}
        assert response.error is None
        assert response.jsonrpc == "2.0"

    def test_failure_response_creation(self):
        """Test creating a failure response."""
        error = IPCError.internal_error("Something went wrong")
        response = IPCResponse.failure(1, error)

        assert response.id == 1
        assert response.result is None
        assert response.error == error

    def test_is_success_true_for_success(self):
        """Test is_success property for success response."""
        response = IPCResponse.success(1, "ok")

        assert response.is_success is True
        assert response.is_error is False

    def test_is_error_true_for_failure(self):
        """Test is_error property for failure response."""
        response = IPCResponse.failure(1, IPCError.parse_error())

        assert response.is_error is True
        assert response.is_success is False

    def test_success_with_string_id(self):
        """Test success response with string ID."""
        response = IPCResponse.success("req-123", {"status": "ok"})

        assert response.id == "req-123"

    def test_success_with_none_result(self):
        """Test success response with None result."""
        response = IPCResponse.success(1, None)

        assert response.result is None
        assert response.is_success is True


class TestIPCResponseSerialization:
    """Tests for IPCResponse serialization."""

    def test_success_to_dict(self):
        """Test success response to_dict."""
        response = IPCResponse.success(1, {"key": "value"})
        result = response.to_dict()

        assert result == {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"key": "value"},
        }
        assert "error" not in result

    def test_failure_to_dict(self):
        """Test failure response to_dict."""
        error = IPCError(code=-32600, message="Invalid request")
        response = IPCResponse.failure(1, error)
        result = response.to_dict()

        assert result == {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32600, "message": "Invalid request"},
        }
        assert "result" not in result

    def test_success_to_json(self):
        """Test success response to_json."""
        response = IPCResponse.success(1, "ok")
        json_str = response.to_json()
        data = json.loads(json_str)

        assert data["id"] == 1
        assert data["result"] == "ok"
        assert "error" not in data

    def test_failure_to_json(self):
        """Test failure response to_json."""
        error = IPCError.method_not_found("unknown")
        response = IPCResponse.failure(1, error)
        json_str = response.to_json()
        data = json.loads(json_str)

        assert data["id"] == 1
        assert "error" in data
        assert data["error"]["code"] == ErrorCodes.METHOD_NOT_FOUND


# =============================================================================
# Handler Tests - Registration and Routing
# =============================================================================


class TestHandlerRegistration:
    """Tests for Handler method registration."""

    def test_register_handler_method(self):
        """Test registering handler via register_handler."""
        handler = Handler()

        def my_handler(params):
            return {"ok": True}

        handler.register_handler("my_method", my_handler)

        assert handler.has_handler("my_method")

    def test_register_decorator(self):
        """Test registering handler via decorator."""
        handler = Handler()

        @handler.register("decorated_method")
        def my_handler(params):
            return {"ok": True}

        assert handler.has_handler("decorated_method")

    def test_decorator_returns_function(self):
        """Test that decorator returns the original function."""
        handler = Handler()

        @handler.register("test")
        def my_handler(params):
            return {"result": "test"}

        # Should be able to call directly
        assert my_handler({}) == {"result": "test"}

    def test_has_handler_false_for_unregistered(self):
        """Test has_handler returns False for unregistered methods."""
        handler = Handler()

        assert handler.has_handler("nonexistent") is False

    def test_list_methods_empty(self):
        """Test list_methods returns empty list for new handler."""
        handler = Handler()

        assert handler.list_methods() == []

    def test_list_methods_returns_registered(self):
        """Test list_methods returns all registered methods."""
        handler = Handler()
        handler.register_handler("method_a", lambda p: None)
        handler.register_handler("method_b", lambda p: None)
        handler.register_handler("method_c", lambda p: None)

        methods = handler.list_methods()

        assert "method_a" in methods
        assert "method_b" in methods
        assert "method_c" in methods
        assert len(methods) == 3

    def test_register_overwrites_existing(self):
        """Test that registering same method overwrites."""
        handler = Handler()

        handler.register_handler("method", lambda p: "first")
        handler.register_handler("method", lambda p: "second")

        request = IPCRequest(id=1, method="method")
        response = handler.handle(request)

        assert response.result == "second"


# =============================================================================
# Handler Tests - Request Handling
# =============================================================================


class TestHandlerHandle:
    """Tests for Handler.handle method."""

    def test_successful_handler_call(self):
        """Test successful handler invocation."""
        handler = Handler()
        handler.register_handler("echo", lambda p: {"echo": p.get("message")})

        request = IPCRequest(id=1, method="echo", params={"message": "hello"})
        response = handler.handle(request)

        assert response.is_success
        assert response.id == 1
        assert response.result == {"echo": "hello"}

    def test_handler_with_no_params(self):
        """Test handler that ignores params."""
        handler = Handler()
        handler.register_handler("ping", lambda p: {"pong": True})

        request = IPCRequest(id=1, method="ping")
        response = handler.handle(request)

        assert response.is_success
        assert response.result == {"pong": True}

    def test_method_not_found_error(self):
        """Test error for unregistered method."""
        handler = Handler()

        request = IPCRequest(id=1, method="nonexistent")
        response = handler.handle(request)

        assert response.is_error
        assert response.id == 1
        assert response.error.code == ErrorCodes.METHOD_NOT_FOUND
        assert "nonexistent" in response.error.message

    def test_valueerror_returns_invalid_params(self):
        """Test that ValueError in handler returns invalid params error."""
        handler = Handler()

        def bad_handler(params):
            raise ValueError("Parameter 'x' is required")

        handler.register_handler("bad", bad_handler)

        request = IPCRequest(id=1, method="bad")
        response = handler.handle(request)

        assert response.is_error
        assert response.error.code == ErrorCodes.INVALID_PARAMS
        assert "Parameter 'x' is required" in response.error.message

    def test_generic_exception_returns_internal_error(self):
        """Test that generic exception returns internal error."""
        handler = Handler()

        def failing_handler(params):
            raise RuntimeError("Unexpected failure")

        handler.register_handler("fail", failing_handler)

        request = IPCRequest(id=1, method="fail")
        response = handler.handle(request)

        assert response.is_error
        assert response.error.code == ErrorCodes.INTERNAL_ERROR
        assert "Unexpected failure" in response.error.message

    def test_handler_returning_none(self):
        """Test handler returning None."""
        handler = Handler()
        handler.register_handler("void", lambda p: None)

        request = IPCRequest(id=1, method="void")
        response = handler.handle(request)

        assert response.is_success
        assert response.result is None

    def test_handler_returning_primitive(self):
        """Test handler returning primitive value."""
        handler = Handler()
        handler.register_handler("count", lambda p: 42)

        request = IPCRequest(id=1, method="count")
        response = handler.handle(request)

        assert response.is_success
        assert response.result == 42

    def test_handler_returning_list(self):
        """Test handler returning list."""
        handler = Handler()
        handler.register_handler("list", lambda p: [1, 2, 3])

        request = IPCRequest(id=1, method="list")
        response = handler.handle(request)

        assert response.is_success
        assert response.result == [1, 2, 3]

    def test_preserves_request_id_string(self):
        """Test that string request ID is preserved."""
        handler = Handler()
        handler.register_handler("test", lambda p: {})

        request = IPCRequest(id="uuid-123-abc", method="test")
        response = handler.handle(request)

        assert response.id == "uuid-123-abc"

    def test_preserves_request_id_int(self):
        """Test that integer request ID is preserved."""
        handler = Handler()
        handler.register_handler("test", lambda p: {})

        request = IPCRequest(id=999, method="test")
        response = handler.handle(request)

        assert response.id == 999


class TestHandlerHandleAsync:
    """Tests for Handler.handle_async method."""

    def test_async_handler_success(self):
        """Test async handler invocation."""
        handler = Handler()

        async def async_handler(params):
            return {"async": True}

        handler.register_handler("async_method", async_handler)

        request = IPCRequest(id=1, method="async_method")
        response = run_async(handler.handle_async(request))

        assert response.is_success
        assert response.result == {"async": True}

    def test_sync_handler_via_async(self):
        """Test that sync handlers work through handle_async."""
        handler = Handler()
        handler.register_handler("sync", lambda p: {"sync": True})

        request = IPCRequest(id=1, method="sync")
        response = run_async(handler.handle_async(request))

        assert response.is_success
        assert response.result == {"sync": True}

    def test_async_method_not_found(self):
        """Test method not found in async handler."""
        handler = Handler()

        request = IPCRequest(id=1, method="missing")
        response = run_async(handler.handle_async(request))

        assert response.is_error
        assert response.error.code == ErrorCodes.METHOD_NOT_FOUND

    def test_async_valueerror_returns_invalid_params(self):
        """Test ValueError in async handler."""
        handler = Handler()

        async def bad_async(params):
            raise ValueError("Bad param")

        handler.register_handler("bad_async", bad_async)

        request = IPCRequest(id=1, method="bad_async")
        response = run_async(handler.handle_async(request))

        assert response.is_error
        assert response.error.code == ErrorCodes.INVALID_PARAMS

    def test_async_generic_exception(self):
        """Test generic exception in async handler."""
        handler = Handler()

        async def failing_async(params):
            raise RuntimeError("Async failure")

        handler.register_handler("fail_async", failing_async)

        request = IPCRequest(id=1, method="fail_async")
        response = run_async(handler.handle_async(request))

        assert response.is_error
        assert response.error.code == ErrorCodes.INTERNAL_ERROR

    def test_sync_call_with_async_handler_returns_error(self):
        """Test that calling async handler via sync handle returns error."""
        handler = Handler()

        async def async_handler(params):
            return {"async": True}

        handler.register_handler("async_method", async_handler)

        request = IPCRequest(id=1, method="async_method")
        response = handler.handle(request)

        assert response.is_error
        assert response.error.code == ErrorCodes.INTERNAL_ERROR
        assert "async" in response.error.message.lower()


# =============================================================================
# Default Handler Tests
# =============================================================================


class TestCreateDefaultHandler:
    """Tests for create_default_handler function."""

    def test_returns_handler_instance(self):
        """Test that function returns Handler instance."""
        handler = create_default_handler()

        assert isinstance(handler, Handler)

    def test_has_generate_code_handler(self):
        """Test that generate_code handler is registered."""
        handler = create_default_handler()

        assert handler.has_handler("generate_code")

    def test_has_validate_code_handler(self):
        """Test that validate_code handler is registered."""
        handler = create_default_handler()

        assert handler.has_handler("validate_code")

    def test_has_generate_diff_handler(self):
        """Test that generate_diff handler is registered."""
        handler = create_default_handler()

        assert handler.has_handler("generate_diff")

    def test_has_apply_changes_handler(self):
        """Test that apply_changes handler is registered."""
        handler = create_default_handler()

        assert handler.has_handler("apply_changes")

    def test_list_methods_includes_all_defaults(self):
        """Test that all default methods are listed."""
        handler = create_default_handler()
        methods = handler.list_methods()

        assert "generate_code" in methods
        assert "validate_code" in methods
        assert "generate_diff" in methods
        assert "apply_changes" in methods


class TestDefaultHandlerIntegration:
    """Integration tests for default handler methods."""

    def test_validate_code_success(self):
        """Test validate_code with valid Python code."""
        handler = create_default_handler()

        request = IPCRequest(
            id=1,
            method="validate_code",
            params={"source": "x = 1"}
        )
        response = handler.handle(request)

        assert response.is_success
        assert response.result["success"] is True

    def test_validate_code_with_syntax_error(self):
        """Test validate_code with invalid Python code."""
        handler = create_default_handler()

        request = IPCRequest(
            id=1,
            method="validate_code",
            params={"source": "def foo("}
        )
        response = handler.handle(request)

        assert response.is_success  # Handler succeeded, but validation failed
        assert response.result["success"] is False
        assert len(response.result["errors"]) > 0

    def test_validate_code_missing_source(self):
        """Test validate_code with missing source parameter."""
        handler = create_default_handler()

        request = IPCRequest(
            id=1,
            method="validate_code",
            params={}
        )
        response = handler.handle(request)

        assert response.is_error
        assert response.error.code == ErrorCodes.INVALID_PARAMS
        assert "source" in response.error.message.lower()

    def test_generate_diff_success(self):
        """Test generate_diff with valid parameters."""
        handler = create_default_handler()

        request = IPCRequest(
            id=1,
            method="generate_diff",
            params={
                "original": "x = 1",
                "modified": "x = 2",
            }
        )
        response = handler.handle(request)

        assert response.is_success
        # DiffResult.to_dict() uses camelCase keys
        assert "hasChanges" in response.result
        assert response.result["hasChanges"] is True

    def test_generate_diff_missing_original(self):
        """Test generate_diff with missing original parameter."""
        handler = create_default_handler()

        request = IPCRequest(
            id=1,
            method="generate_diff",
            params={"modified": "x = 2"}
        )
        response = handler.handle(request)

        assert response.is_error
        assert response.error.code == ErrorCodes.INVALID_PARAMS
        assert "original" in response.error.message.lower()

    def test_generate_diff_missing_modified(self):
        """Test generate_diff with missing modified parameter."""
        handler = create_default_handler()

        request = IPCRequest(
            id=1,
            method="generate_diff",
            params={"original": "x = 1"}
        )
        response = handler.handle(request)

        assert response.is_error
        assert response.error.code == ErrorCodes.INVALID_PARAMS
        assert "modified" in response.error.message.lower()


# =============================================================================
# Error Handling Edge Cases
# =============================================================================


class TestErrorHandlingEdgeCases:
    """Tests for error handling edge cases."""

    def test_handler_with_empty_params(self):
        """Test handler receives empty dict when no params."""
        handler = Handler()
        received_params = []

        def capture_handler(params):
            received_params.append(params)
            return {}

        handler.register_handler("capture", capture_handler)

        request = IPCRequest(id=1, method="capture")
        handler.handle(request)

        assert received_params == [{}]

    def test_handler_exception_does_not_crash(self):
        """Test that handler exceptions are caught."""
        handler = Handler()

        def crashing_handler(params):
            raise Exception("Kaboom!")

        handler.register_handler("crash", crashing_handler)

        request = IPCRequest(id=1, method="crash")

        # Should not raise
        response = handler.handle(request)

        assert response.is_error

    def test_handler_type_error_is_internal_error(self):
        """Test that TypeError is treated as internal error."""
        handler = Handler()

        def type_error_handler(params):
            return None + 1  # TypeError

        handler.register_handler("type_err", type_error_handler)

        request = IPCRequest(id=1, method="type_err")
        response = handler.handle(request)

        assert response.is_error
        assert response.error.code == ErrorCodes.INTERNAL_ERROR

    def test_handler_key_error_is_internal_error(self):
        """Test that KeyError is treated as internal error."""
        handler = Handler()

        def key_error_handler(params):
            d = {}
            return d["nonexistent"]

        handler.register_handler("key_err", key_error_handler)

        request = IPCRequest(id=1, method="key_err")
        response = handler.handle(request)

        assert response.is_error
        assert response.error.code == ErrorCodes.INTERNAL_ERROR

    def test_response_serialization_with_complex_data(self):
        """Test that complex result data serializes correctly."""
        handler = Handler()

        complex_result = {
            "nested": {"deeply": {"nested": "value"}},
            "list": [1, 2, {"key": "val"}],
            "unicode": "Hello!",
            "number": 3.14159,
            "boolean": True,
            "null": None,
        }

        handler.register_handler("complex", lambda p: complex_result)

        request = IPCRequest(id=1, method="complex")
        response = handler.handle(request)

        # Serialize and deserialize
        json_str = response.to_json()
        data = json.loads(json_str)

        assert data["result"]["nested"]["deeply"]["nested"] == "value"
        assert data["result"]["list"][2]["key"] == "val"
        assert data["result"]["boolean"] is True


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestJSONRPCCompliance:
    """Tests for JSON-RPC 2.0 protocol compliance."""

    def test_response_always_has_jsonrpc_field(self):
        """Test that responses always include jsonrpc field."""
        response = IPCResponse.success(1, {})
        data = response.to_dict()

        assert "jsonrpc" in data
        assert data["jsonrpc"] == "2.0"

    def test_response_always_has_id_field(self):
        """Test that responses always include id field."""
        response = IPCResponse.success(1, {})
        data = response.to_dict()

        assert "id" in data

    def test_success_has_result_not_error(self):
        """Test that success response has result, not error."""
        response = IPCResponse.success(1, {"data": "value"})
        data = response.to_dict()

        assert "result" in data
        assert "error" not in data

    def test_error_has_error_not_result(self):
        """Test that error response has error, not result."""
        response = IPCResponse.failure(1, IPCError.parse_error())
        data = response.to_dict()

        assert "error" in data
        assert "result" not in data

    def test_error_object_has_required_fields(self):
        """Test that error object has code and message."""
        response = IPCResponse.failure(1, IPCError(code=-32600, message="Test"))
        data = response.to_dict()

        assert "code" in data["error"]
        assert "message" in data["error"]

    def test_request_requires_method(self):
        """Test that request requires method field."""
        with pytest.raises(ValueError):
            IPCRequest.from_dict({"id": 1})

    def test_request_requires_id(self):
        """Test that request requires id field."""
        with pytest.raises(ValueError):
            IPCRequest.from_dict({"method": "test"})

    def test_standard_error_codes_are_negative(self):
        """Test that standard error codes are negative per JSON-RPC spec."""
        assert IPCError.PARSE_ERROR < 0
        assert IPCError.INVALID_REQUEST < 0
        assert IPCError.METHOD_NOT_FOUND < 0
        assert IPCError.INVALID_PARAMS < 0
        assert IPCError.INTERNAL_ERROR < 0

    def test_standard_error_codes_in_reserved_range(self):
        """Test that standard error codes are in reserved range (-32700 to -32600)."""
        assert -32700 <= IPCError.PARSE_ERROR <= -32600
        assert -32700 <= IPCError.INVALID_REQUEST <= -32600
        assert -32700 <= IPCError.METHOD_NOT_FOUND <= -32600
        assert -32700 <= IPCError.INVALID_PARAMS <= -32600
        assert -32700 <= IPCError.INTERNAL_ERROR <= -32600
