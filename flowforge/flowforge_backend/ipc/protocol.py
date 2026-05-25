"""IPC Protocol definitions for FlowForge Backend.

Defines the message format for communication between TypeScript frontend
and Python backend using line-delimited JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from typing_extensions import Self

from ..config import ErrorCodes, JSONRPC_VERSION


@dataclass(frozen=True)
class IPCError:
    """Represents an error in an IPC response.

    Attributes:
        code: Numeric error code for programmatic handling
        message: Human-readable error description
        data: Optional additional error context
    """
    code: int
    message: str
    data: Optional[Any] = None

    # Standard error codes (from centralized config)
    PARSE_ERROR = ErrorCodes.PARSE_ERROR
    INVALID_REQUEST = ErrorCodes.INVALID_REQUEST
    METHOD_NOT_FOUND = ErrorCodes.METHOD_NOT_FOUND
    INVALID_PARAMS = ErrorCodes.INVALID_PARAMS
    INTERNAL_ERROR = ErrorCodes.INTERNAL_ERROR

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def parse_error(cls, message: str = "Parse error") -> Self:
        """Create a parse error."""
        return cls(code=cls.PARSE_ERROR, message=message)

    @classmethod
    def invalid_request(cls, message: str = "Invalid request") -> Self:
        """Create an invalid request error."""
        return cls(code=cls.INVALID_REQUEST, message=message)

    @classmethod
    def method_not_found(cls, method: str) -> Self:
        """Create a method not found error."""
        return cls(code=cls.METHOD_NOT_FOUND, message=f"Method not found: {method}")

    @classmethod
    def invalid_params(cls, message: str = "Invalid params") -> Self:
        """Create an invalid params error."""
        return cls(code=cls.INVALID_PARAMS, message=message)

    @classmethod
    def internal_error(cls, message: str = "Internal error") -> Self:
        """Create an internal error."""
        return cls(code=cls.INTERNAL_ERROR, message=message)


@dataclass
class IPCRequest:
    """Represents an incoming IPC request.

    Supports both JSON-RPC 2.0 format and simplified format for backward compatibility.

    JSON-RPC 2.0 format:
        {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}

    Simplified format:
        {"id": "1", "method": "ping", "params": {}}

    Attributes:
        id: Unique request identifier for matching responses (string or int)
        method: The method name to invoke
        params: Optional parameters for the method
        jsonrpc: JSON-RPC version (optional, defaults to "2.0")
    """
    id: str | int
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization (JSON-RPC 2.0 format)."""
        return {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
            "params": self.params,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create an IPCRequest from a dictionary.

        Accepts both JSON-RPC 2.0 format and simplified format.

        Args:
            data: Dictionary containing id, method, and optional params/jsonrpc

        Returns:
            New IPCRequest instance

        Raises:
            ValueError: If required fields are missing or invalid
        """
        if "id" not in data:
            raise ValueError("Missing required field: id")
        if "method" not in data:
            raise ValueError("Missing required field: method")

        # Accept id as either string or number - keep original type
        request_id = data["id"]
        if not isinstance(request_id, (str, int)):
            raise ValueError("Field 'id' must be a string or number")

        method = data["method"]
        if not isinstance(method, str):
            raise ValueError("Field 'method' must be a string")

        params = data.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("Field 'params' must be an object")

        # Handle jsonrpc field - optional, defaults to JSONRPC_VERSION
        jsonrpc = data.get("jsonrpc", JSONRPC_VERSION)
        if not isinstance(jsonrpc, str):
            raise ValueError("Field 'jsonrpc' must be a string")
        if jsonrpc != JSONRPC_VERSION:
            raise ValueError(f"Unsupported JSON-RPC version: {jsonrpc} (expected '{JSONRPC_VERSION}')")

        return cls(id=request_id, method=method, params=params, jsonrpc=jsonrpc)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Parse an IPCRequest from a JSON string.

        Accepts both JSON-RPC 2.0 format and simplified format.

        Args:
            json_str: JSON string representation of the request

        Returns:
            New IPCRequest instance

        Raises:
            ValueError: If JSON is invalid or required fields are missing
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        if not isinstance(data, dict):
            raise ValueError("Request must be a JSON object")

        return cls.from_dict(data)


@dataclass
class IPCResponse:
    """Represents an outgoing IPC response in JSON-RPC 2.0 format.

    Response format:
        Success: {"jsonrpc": "2.0", "id": 1, "result": {...}}
        Error:   {"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "..."}}

    Attributes:
        id: Request identifier this response corresponds to (string or int)
        result: The successful result (None if error)
        error: Error information (None if success)
        jsonrpc: JSON-RPC version (always "2.0")
    """
    id: str | int
    result: Optional[Any] = None
    error: Optional[IPCError] = None
    jsonrpc: str = JSONRPC_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization (JSON-RPC 2.0 format)."""
        response: dict[str, Any] = {
            "jsonrpc": self.jsonrpc,
            "id": self.id,
        }

        if self.error is not None:
            response["error"] = self.error.to_dict()
        else:
            response["result"] = self.result

        return response

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def success(cls, request_id: str | int, result: Any) -> Self:
        """Create a successful response.

        Args:
            request_id: The ID of the request this responds to (string or int)
            result: The result data

        Returns:
            New IPCResponse with the result
        """
        return cls(id=request_id, result=result, error=None)

    @classmethod
    def failure(cls, request_id: str | int, error: IPCError) -> Self:
        """Create an error response.

        Args:
            request_id: The ID of the request this responds to (string or int)
            error: The error information

        Returns:
            New IPCResponse with the error
        """
        return cls(id=request_id, result=None, error=error)

    @property
    def is_success(self) -> bool:
        """Check if this response represents success."""
        return self.error is None

    @property
    def is_error(self) -> bool:
        """Check if this response represents an error."""
        return self.error is not None
