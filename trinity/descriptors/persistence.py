"""
Persistence descriptors - field-level serialization control.

Provides custom serialization, transient marking, migration,
and encryption for individual fields.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

from trinity.constants import DEFAULT_SERIALIZATION_FORMAT, DEFAULT_SCHEMA_VERSION
from trinity.decorators.ops import Op, Step
from trinity.descriptors.base import BaseDescriptor

T = TypeVar("T")


class SerializableDescriptor(BaseDescriptor[T]):
    """Custom encode/decode for field serialization."""

    __slots__ = ("_encoder", "_decoder", "_format")

    descriptor_id = "serializable"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ("transient",)

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        encoder: Optional[Callable[[Any], Any]] = None,
        decoder: Optional[Callable[[Any], Any]] = None,
        format: str = DEFAULT_SERIALIZATION_FORMAT,
        **config: Any,
    ) -> None:
        """
        Initialize serializable descriptor.

        Args:
            field_type: The type annotation for this field.
            inner: Inner descriptor to wrap.
            encoder: Custom encoding function for serialization.
            decoder: Custom decoding function for deserialization.
            format: Serialization format identifier (e.g. 'binary', 'json').
            **config: Additional configuration.
        """
        super().__init__(field_type=field_type, inner=inner, **config)
        self._encoder = encoder
        self._decoder = decoder
        self._format = format

    def encode(self, value: Any) -> Any:
        """Encode a value for serialization."""
        if self._encoder:
            return self._encoder(value)
        return value

    def decode(self, data: Any) -> Any:
        """Decode serialized data back to a value."""
        if self._decoder:
            return self._decoder(data)
        return data

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [
            Step(Op.HOOK, {"event": "on_serialize", "callback": "encode"}),
            Step(Op.HOOK, {"event": "on_deserialize", "callback": "decode"}),
            Step(Op.TAG, {"key": "serialization_format", "value": self._format}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        """Return serialization configuration."""
        meta = super().get_metadata()
        meta["format"] = self._format
        meta["has_encoder"] = self._encoder is not None
        meta["has_decoder"] = self._decoder is not None
        return meta


class TransientDescriptor(BaseDescriptor[T]):
    """Marks field as non-serializable. Skipped on save/load."""

    __slots__ = ()

    descriptor_id = "transient"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ("serializable",)

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [Step(Op.TAG, {"key": "transient", "value": True})]

    def get_metadata(self) -> dict[str, Any]:
        """Return transient marker metadata."""
        meta = super().get_metadata()
        meta["transient"] = True
        return meta


class MigratedDescriptor(BaseDescriptor[T]):
    """Handle field renames across save file versions."""

    __slots__ = ("_from_name", "_version_added")

    descriptor_id = "migrated"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        from_name: str = "",
        version_added: int = DEFAULT_SCHEMA_VERSION,
        **config: Any,
    ) -> None:
        """
        Initialize migration descriptor.

        Args:
            field_type: The type annotation for this field.
            inner: Inner descriptor to wrap.
            from_name: The previous field name in older save versions.
            version_added: The schema version where this field was introduced.
            **config: Additional configuration.

        Raises:
            ValueError: If from_name is not provided.
        """
        super().__init__(field_type=field_type, inner=inner, **config)
        if not from_name:
            raise ValueError("MigratedDescriptor requires 'from_name' parameter")
        self._from_name = from_name
        self._version_added = version_added

    def migrate(self, old_data: dict[str, Any]) -> Any:
        """Extract value from old save data using previous field name."""
        if self._from_name in old_data:
            return old_data[self._from_name]
        return None

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [
            Step(Op.TAG, {"key": "migrated_from", "value": self._from_name}),
            Step(Op.TAG, {"key": "version_added", "value": self._version_added}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        """Return migration configuration."""
        meta = super().get_metadata()
        meta["from_name"] = self._from_name
        meta["version_added"] = self._version_added
        return meta


class EncryptedDescriptor(BaseDescriptor[T]):
    """Encrypts field value at rest. Plaintext in memory, encrypted on serialize."""

    __slots__ = ("_encrypt_fn", "_decrypt_fn", "_key")

    descriptor_id = "encrypted"
    accepts_inner = ("*",)
    accepts_outer = ("*",)
    excludes = ()

    def __init__(
        self,
        field_type: type = object,
        inner: Optional[BaseDescriptor[T]] = None,
        encrypt_fn: Optional[Callable[[Any], Any]] = None,
        decrypt_fn: Optional[Callable[[Any], Any]] = None,
        **config: Any,
    ) -> None:
        """
        Initialize encryption descriptor.

        Args:
            field_type: The type annotation for this field.
            inner: Inner descriptor to wrap.
            encrypt_fn: Custom encryption function.
            decrypt_fn: Custom decryption function.
            **config: Additional configuration.
        """
        key = config.pop("key", None)
        super().__init__(field_type=field_type, inner=inner, **config)
        self._key = key
        self._encrypt_fn = encrypt_fn
        self._decrypt_fn = decrypt_fn

    def encrypt(self, value: Any) -> Any:
        """Encrypt a value for storage at rest."""
        if self._encrypt_fn:
            return self._encrypt_fn(value)
        # Default: base64 for strings, identity for others
        if isinstance(value, str):
            import base64
            return base64.b64encode(value.encode()).decode()
        return value

    def decrypt(self, data: Any) -> Any:
        """Decrypt stored data back to plaintext."""
        if self._decrypt_fn:
            return self._decrypt_fn(data)
        if isinstance(data, str):
            import base64
            try:
                return base64.b64decode(data.encode()).decode()
            except (ValueError, UnicodeDecodeError) as e:
                raise ValueError(f"Failed to decrypt data: {e}") from e
        return data

    @property
    def descriptor_steps(self) -> list["Step"]:
        return [
            Step(Op.INTERCEPT, {"get": "decrypt", "set": "encrypt"}),
            Step(Op.TAG, {"key": "encrypted", "value": True}),
        ]

    def get_metadata(self) -> dict[str, Any]:
        """Return encryption configuration."""
        meta = super().get_metadata()
        meta["has_encrypt_fn"] = self._encrypt_fn is not None
        meta["has_decrypt_fn"] = self._decrypt_fn is not None
        return meta


__all__ = [
    "SerializableDescriptor",
    "TransientDescriptor",
    "MigratedDescriptor",
    "EncryptedDescriptor",
]
