"""
ProtocolMeta - Metaclass for network protocol definitions.

Handles protocol versioning and message registration.
Protocols define the network communication format between clients and servers.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, ClassVar, Optional

from trinity.constants import DEFAULT_VERSION_HISTORY_LIMIT
from trinity.decorators.ops import Op, Step
from trinity.metaclasses.engine_meta import EngineMeta

logger = logging.getLogger(__name__)


class ProtocolMeta(EngineMeta):
    """
    Metaclass for network protocols.

    Created classes will:
    - Define a versioned network protocol
    - Register message types
    - Support protocol negotiation
    - Track compatibility requirements

    Required class attributes:
    - _protocol_version: int (current protocol version)

    Optional class attributes:
    - _protocol_min_version: int (minimum compatible version)
    - _protocol_messages: dict[int, type] (message ID -> message type)
    - _protocol_name: str (human-readable name)

    Attached attributes:
    - _protocol_id: int (unique identifier)
    - _protocol_qualified_name: str (module.Class name)
    """

    _registry: ClassVar[dict[int, type]] = {}
    _next_id: ClassVar[int] = 1
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _version_decoders: ClassVar[dict[tuple[int, int], Any]] = {}  # (protocol_id, version) -> decoder_fn

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> ProtocolMeta:
        """Create a new protocol type."""
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip base Protocol class
        if name == "Protocol":
            return cls

        with mcs._lock:
            # === 1. GENERATE UNIQUE ID ===
            cls._protocol_id = mcs._next_id
            mcs._next_id += 1
            cls._protocol_qualified_name = f"{cls.__module__}.{name}"

            # --- 3.8.2: Record TAG steps for id and name ---
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "protocol_id", "value": cls._protocol_id})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "protocol_name", "value": cls._protocol_qualified_name})
            )

            # === 2. VALIDATE VERSION ===
            if not hasattr(cls, "_protocol_version"):
                raise TypeError(
                    f"{cls.__name__}: Protocols must define _protocol_version. "
                    f"Example: _protocol_version = 1"
                )

            version = cls._protocol_version
            if not isinstance(version, int) or version < 1:
                raise TypeError(
                    f"{cls.__name__}: _protocol_version must be a positive integer, "
                    f"got {version!r}"
                )

            # --- 3.8.3: Record VALIDATE + TAG for version ---
            cls._metaclass_steps.append(
                Step(Op.VALIDATE, {"constraint": "protocol_version_valid"})
            )
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "protocol_version", "value": version})
            )

            # === 3. SET DEFAULTS ===
            if not hasattr(cls, "_protocol_min_version"):
                cls._protocol_min_version = (
                    cls._protocol_version
                )  # Only self-compatible by default
            if not hasattr(cls, "_protocol_messages"):
                cls._protocol_messages = {}
            if not hasattr(cls, "_protocol_name"):
                cls._protocol_name = name

            # --- 3.8.3 (cont): TAG for min_version ---
            cls._metaclass_steps.append(
                Step(Op.TAG, {"key": "protocol_min_version", "value": cls._protocol_min_version})
            )

            # Validate min_version
            if cls._protocol_min_version > cls._protocol_version:
                raise TypeError(
                    f"{cls.__name__}: _protocol_min_version ({cls._protocol_min_version}) "
                    f"cannot be greater than _protocol_version ({cls._protocol_version})"
                )

            # --- 3.8.4: Record VALIDATE for min_version <= version ---
            cls._metaclass_steps.append(
                Step(Op.VALIDATE, {"constraint": "min_version_lte_version"})
            )

            # === 4. REGISTER ===
            mcs._registry[cls._protocol_id] = cls

            # --- 3.8.5: Record REGISTER step ---
            cls._metaclass_steps.append(
                Step(Op.REGISTER, {"registry": "protocol_registry"})
            )

        return cls

    # =========================================================================
    # REGISTRY ACCESS CLASS METHODS
    # =========================================================================

    @classmethod
    def get_by_id(mcs, protocol_id: int) -> Optional[type]:
        """Get protocol class by ID."""
        return mcs._registry.get(protocol_id)

    @classmethod
    def get_by_name(mcs, name: str) -> Optional[type]:
        """Get protocol class by qualified name."""
        for proto_cls in mcs._registry.values():
            if proto_cls._protocol_qualified_name == name:
                return proto_cls
        return None

    @classmethod
    def all_protocols(mcs) -> list[type]:
        """Get all registered protocol classes."""
        return list(mcs._registry.values())

    @classmethod
    def is_compatible(mcs, protocol_cls: type, version: int) -> bool:
        """
        Check if a protocol version is compatible.

        Args:
            protocol_cls: The protocol class to check.
            version: The version to check compatibility with.

        Returns:
            True if the version is compatible.
        """
        if not hasattr(protocol_cls, "_protocol_min_version"):
            return False

        min_ver = protocol_cls._protocol_min_version
        max_ver = protocol_cls._protocol_version

        return min_ver <= version <= max_ver

    @classmethod
    def negotiate_version(
        mcs, protocol_cls: type, offered_versions: list[int]
    ) -> Optional[int]:
        """
        Negotiate the best compatible version.

        Args:
            protocol_cls: The protocol class.
            offered_versions: List of versions the other party supports.

        Returns:
            The highest compatible version, or None if no compatible version.
        """
        min_ver = getattr(protocol_cls, "_protocol_min_version", 1)
        max_ver = getattr(protocol_cls, "_protocol_version", 1)

        compatible = [v for v in offered_versions if min_ver <= v <= max_ver]
        return max(compatible) if compatible else None

    @classmethod
    def register_message(
        mcs, protocol_cls: type, message_id: int, message_type: type
    ) -> None:
        """
        Register a message type with a protocol.

        Args:
            protocol_cls: The protocol to register with.
            message_id: Unique ID for this message within the protocol.
            message_type: The message class.
        """
        if not hasattr(protocol_cls, "_protocol_messages"):
            protocol_cls._protocol_messages = {}

        if message_id in protocol_cls._protocol_messages:
            existing = protocol_cls._protocol_messages[message_id]
            raise ValueError(
                f"Message ID {message_id} already registered to {existing.__name__} "
                f"in protocol {protocol_cls.__name__}"
            )

        protocol_cls._protocol_messages[message_id] = message_type

    @classmethod
    def get_message_type(mcs, protocol_cls: type, message_id: int) -> Optional[type]:
        """Get the message type for a given ID."""
        messages = getattr(protocol_cls, "_protocol_messages", {})
        return messages.get(message_id)

    @classmethod
    def register_version_decoder(
        mcs, protocol_cls: type, version: int, decoder_fn: Any
    ) -> None:
        """
        Register a version-specific message decoder.

        Args:
            protocol_cls: The protocol class.
            version: The protocol version this decoder handles.
            decoder_fn: The decoder function (message_id, data) -> decoded_message.
        """
        with mcs._lock:
            protocol_id = getattr(protocol_cls, "_protocol_id", None)
            if protocol_id is None:
                raise ValueError(f"Protocol {protocol_cls.__name__} has no _protocol_id")

            key = (protocol_id, version)
            if key in mcs._version_decoders:
                raise ValueError(
                    f"Decoder for protocol {protocol_cls.__name__} version {version} "
                    f"already registered"
                )

            mcs._version_decoders[key] = decoder_fn

    @classmethod
    def decode_message(
        mcs, protocol_cls: type, version: int, message_id: int, data: Any
    ) -> Optional[Any]:
        """
        Decode a message using version-specific decoder if available.

        Args:
            protocol_cls: The protocol class.
            version: The protocol version of the message.
            message_id: The message ID.
            data: The raw message data.

        Returns:
            Decoded message, or None if no decoder is available.

        Raises:
            ValueError: If protocol_cls is not a valid registered protocol.
        """
        protocol_id = getattr(protocol_cls, "_protocol_id", None)
        if protocol_id is None:
            error_msg = f"Protocol {protocol_cls.__name__} has no _protocol_id (not registered)"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Try version-specific decoder first
        key = (protocol_id, version)
        decoder = mcs._version_decoders.get(key)
        if decoder is not None:
            try:
                return decoder(message_id, data)
            except Exception as e:
                logger.error(
                    f"Decoder for protocol {protocol_cls.__name__} version {version} "
                    f"failed to decode message {message_id}: {e}"
                )
                raise

        # Fall back to default message type lookup
        message_type = mcs.get_message_type(protocol_cls, message_id)
        if message_type is not None and hasattr(message_type, "decode"):
            try:
                return message_type.decode(data)
            except Exception as e:
                logger.error(
                    f"Message type {message_type.__name__} failed to decode: {e}"
                )
                raise

        # No decoder found
        logger.warning(
            f"No decoder available for protocol {protocol_cls.__name__} "
            f"version {version} message {message_id}"
        )
        return None

    @classmethod
    def get_migration_path(
        mcs, protocol_cls: type, from_version: int, to_version: int
    ) -> list[int]:
        """
        Get the migration path from one version to another.

        Args:
            protocol_cls: The protocol class.
            from_version: The starting version.
            to_version: The target version.

        Returns:
            List of version steps needed to migrate (inclusive of both endpoints).
            Returns empty list if either version is outside the supported range.
            Returns single-element list [version] if from_version == to_version.

        Raises:
            ValueError: If from_version or to_version is invalid (not a positive int).
        """
        if not isinstance(from_version, int) or from_version < 1:
            raise ValueError(f"from_version must be a positive integer, got {from_version}")
        if not isinstance(to_version, int) or to_version < 1:
            raise ValueError(f"to_version must be a positive integer, got {to_version}")

        min_ver = getattr(protocol_cls, "_protocol_min_version", 1)
        max_ver = getattr(protocol_cls, "_protocol_version", 1)

        # Validate versions are in supported range
        if from_version < min_ver or from_version > max_ver:
            logger.warning(
                f"from_version {from_version} is outside supported range "
                f"[{min_ver}, {max_ver}] for protocol {protocol_cls.__name__}"
            )
            return []
        if to_version < min_ver or to_version > max_ver:
            logger.warning(
                f"to_version {to_version} is outside supported range "
                f"[{min_ver}, {max_ver}] for protocol {protocol_cls.__name__}"
            )
            return []

        # Generate sequential path
        if from_version <= to_version:
            return list(range(from_version, to_version + 1))
        else:
            return list(range(from_version, to_version - 1, -1))

    @classmethod
    def clear_registry(mcs) -> None:
        """Clear the protocol registry. Useful for testing."""
        with mcs._lock:
            mcs._registry.clear()
            mcs._next_id = 1
            mcs._version_decoders.clear()
        super().clear_registry()
