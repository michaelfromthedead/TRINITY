"""
Tests for persistence descriptors: SerializableDescriptor, TransientDescriptor,
MigratedDescriptor, EncryptedDescriptor.

Verifies:
- Encode/decode round-trips
- Transient exclusion from serialization
- Field migration from old names
- Encryption/decryption of values
"""
import json
import pytest
import sys
sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from trinity.descriptors.persistence import (
    SerializableDescriptor,
    TransientDescriptor,
    MigratedDescriptor,
    EncryptedDescriptor,
)


class TestSerializableDescriptor:
    """Test SerializableDescriptor handles encode/decode for serialization."""

    def test_encode_decode(self):
        """Custom encoder/decoder should round-trip correctly."""
        class Foo:
            data = SerializableDescriptor(
                field_type=dict,
                encoder=lambda v: json.dumps(v),
                decoder=lambda v: json.loads(v),
            )
        Foo.data.__set_name__(Foo, 'data')
        f = Foo()
        f.data = {"key": "value"}
        encoded = Foo.data.encode(f.data)
        assert isinstance(encoded, str)
        decoded = Foo.data.decode(encoded)
        assert decoded == {"key": "value"}

    def test_default_passthrough(self):
        """Without custom encoder/decoder, values should pass through as-is."""
        class Foo:
            value = SerializableDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = 42
        encoded = Foo.value.encode(f.value)
        assert encoded == 42
        decoded = Foo.value.decode(encoded)
        assert decoded == 42

    def test_encode_none(self):
        """Encoding None should work correctly."""
        class Foo:
            value = SerializableDescriptor(field_type=int)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = None
        encoded = Foo.value.encode(f.value)
        assert encoded is None
        decoded = Foo.value.decode(encoded)
        assert decoded is None

    def test_encode_empty_string(self):
        """Encoding empty string should preserve it."""
        class Foo:
            value = SerializableDescriptor(field_type=str, encoder=lambda v: v.upper(), decoder=lambda v: v.lower())
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = ""
        encoded = Foo.value.encode(f.value)
        assert encoded == ""

    def test_encode_empty_collection(self):
        """Encoding empty collections should work."""
        class Foo:
            value = SerializableDescriptor(field_type=list, encoder=lambda v: str(len(v)), decoder=lambda v: [None] * int(v))
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = []
        encoded = Foo.value.encode(f.value)
        assert encoded == "0"
        decoded = Foo.value.decode(encoded)
        assert decoded == []

    def test_metadata_includes_format(self):
        """Metadata should include the format and encoder/decoder presence."""
        class Foo:
            value = SerializableDescriptor(field_type=int, format="json", encoder=lambda v: v)
        Foo.value.__set_name__(Foo, 'value')
        meta = Foo.value.get_metadata()
        assert meta["descriptor_id"] == "serializable"
        assert meta["format"] == "json"
        assert meta["has_encoder"] is True
        assert meta["has_decoder"] is False

    def test_set_and_get(self):
        """Basic set/get should work normally."""
        class Foo:
            value = SerializableDescriptor(field_type=str)
        Foo.value.__set_name__(Foo, 'value')
        f = Foo()
        f.value = "hello"
        assert f.value == "hello"


class TestTransientDescriptor:
    """Test TransientDescriptor marks fields as excluded from serialization."""

    def test_metadata_transient(self):
        """Metadata should indicate the field is transient."""
        class Foo:
            cache = TransientDescriptor(field_type=dict)
        Foo.cache.__set_name__(Foo, 'cache')
        meta = Foo.cache.get_metadata()
        assert meta.get("transient") is True
        assert meta["descriptor_id"] == "transient"

    def test_excludes_serializable(self):
        """TransientDescriptor should exclude 'serializable' in its excludes tuple."""
        desc = TransientDescriptor(field_type=dict)
        assert "serializable" in desc.excludes

    def test_set_and_get(self):
        """Transient fields should still store and retrieve values normally."""
        class Foo:
            temp = TransientDescriptor(field_type=int)
        Foo.temp.__set_name__(Foo, 'temp')
        f = Foo()
        f.temp = 99
        assert f.temp == 99

    def test_set_none(self):
        """Transient field should handle None correctly."""
        class Foo:
            temp = TransientDescriptor(field_type=int)
        Foo.temp.__set_name__(Foo, 'temp')
        f = Foo()
        f.temp = None
        assert f.temp is None

    def test_multiple_instances(self):
        """Multiple instances should have independent transient values."""
        class Foo:
            temp = TransientDescriptor(field_type=int)
        Foo.temp.__set_name__(Foo, 'temp')
        f1 = Foo()
        f2 = Foo()
        f1.temp = 10
        f2.temp = 20
        assert f1.temp == 10
        assert f2.temp == 20


class TestMigratedDescriptor:
    """Test MigratedDescriptor migrates data from old field names."""

    def test_migrate_from_old_name(self):
        """migrate() should extract value from old save data using previous field name."""
        desc = MigratedDescriptor(field_type=int, from_name="old_name")
        old_data = {"old_name": 42, "other": 99}
        result = desc.migrate(old_data)
        assert result == 42

    def test_migrate_missing_old_name(self):
        """migrate() should return None when old field name is not in data."""
        desc = MigratedDescriptor(field_type=int, from_name="old_name")
        old_data = {"other": 99}
        result = desc.migrate(old_data)
        assert result is None

    def test_migrate_empty_data(self):
        """migrate() should return None for empty data dict."""
        desc = MigratedDescriptor(field_type=int, from_name="old_name")
        result = desc.migrate({})
        assert result is None

    def test_requires_from_name(self):
        """MigratedDescriptor without from_name should raise ValueError."""
        with pytest.raises(ValueError, match="requires 'from_name' parameter"):
            MigratedDescriptor(field_type=int)

    def test_requires_from_name_empty_string(self):
        """MigratedDescriptor with empty from_name should raise ValueError."""
        with pytest.raises(ValueError, match="requires 'from_name' parameter"):
            MigratedDescriptor(field_type=int, from_name="")

    def test_new_value_takes_precedence(self):
        """Setting new name via descriptor should work normally."""
        class Foo:
            new_name = MigratedDescriptor(field_type=int, from_name="old_name")
        Foo.new_name.__set_name__(Foo, 'new_name')
        f = Foo()
        f.new_name = 100
        assert f.new_name == 100

    def test_metadata_includes_migration_info(self):
        """Metadata should include from_name and version_added."""
        class Foo:
            val = MigratedDescriptor(field_type=int, from_name="old_val", version_added=2)
        Foo.val.__set_name__(Foo, 'val')
        meta = Foo.val.get_metadata()
        assert meta["descriptor_id"] == "migrated"
        assert meta["from_name"] == "old_val"
        assert meta["version_added"] == 2


class TestEncryptedDescriptor:
    """Test EncryptedDescriptor encrypts/decrypts values transparently."""

    def test_encrypt_decrypt_string(self):
        """String should round-trip through default base64 encryption."""
        class Foo:
            secret = EncryptedDescriptor(field_type=str)
        Foo.secret.__set_name__(Foo, 'secret')
        f = Foo()
        original = "my secret data"
        encrypted = Foo.secret.encrypt(original)
        # Should be base64-encoded, different from original
        assert encrypted != original
        assert isinstance(encrypted, str)
        # Should decrypt back to original
        decrypted = Foo.secret.decrypt(encrypted)
        assert decrypted == original

    def test_encrypt_empty_string(self):
        """Empty string should be encrypted and decrypted correctly."""
        class Foo:
            secret = EncryptedDescriptor(field_type=str)
        Foo.secret.__set_name__(Foo, 'secret')
        f = Foo()
        encrypted = Foo.secret.encrypt("")
        decrypted = Foo.secret.decrypt(encrypted)
        assert decrypted == ""

    def test_encrypt_non_string_passthrough(self):
        """Non-string types should pass through without encryption by default."""
        class Foo:
            secret = EncryptedDescriptor(field_type=int)
        Foo.secret.__set_name__(Foo, 'secret')
        f = Foo()
        encrypted = Foo.secret.encrypt(42)
        assert encrypted == 42
        decrypted = Foo.secret.decrypt(42)
        assert decrypted == 42

    def test_custom_encrypt_fn(self):
        """Custom encrypt/decrypt functions should be used."""
        class Foo:
            secret = EncryptedDescriptor(
                field_type=str,
                encrypt_fn=lambda v: v[::-1],  # Reverse as "encryption"
                decrypt_fn=lambda v: v[::-1],  # Reverse back
            )
        Foo.secret.__set_name__(Foo, 'secret')
        f = Foo()
        encrypted = Foo.secret.encrypt("hello")
        assert encrypted == "olleh"
        decrypted = Foo.secret.decrypt(encrypted)
        assert decrypted == "hello"

    def test_decrypt_invalid_base64(self):
        """Decrypting invalid base64 should raise ValueError."""
        class Foo:
            secret = EncryptedDescriptor(field_type=str)
        Foo.secret.__set_name__(Foo, 'secret')
        f = Foo()
        # Invalid base64 data
        with pytest.raises(ValueError, match="Failed to decrypt data"):
            Foo.secret.decrypt("!@#$%^&*()_+")

    def test_encrypt_none(self):
        """Encrypting None should handle gracefully."""
        class Foo:
            secret = EncryptedDescriptor(field_type=str)
        Foo.secret.__set_name__(Foo, 'secret')
        f = Foo()
        # None is not a string, so it passes through
        encrypted = Foo.secret.encrypt(None)
        assert encrypted is None

    def test_metadata_does_not_expose_key(self):
        """Metadata should indicate encryption without exposing the key."""
        class Foo:
            secret = EncryptedDescriptor(field_type=str, key="secret-key-123")
        Foo.secret.__set_name__(Foo, 'secret')
        meta = Foo.secret.get_metadata()
        assert meta["descriptor_id"] == "encrypted"
        assert meta["has_encrypt_fn"] is False
        assert meta["has_decrypt_fn"] is False
        # Key should NOT be in metadata
        assert "secret-key-123" not in str(meta)
        assert "key" not in meta

    def test_metadata_with_custom_functions(self):
        """Metadata should indicate custom encrypt/decrypt function presence."""
        class Foo:
            secret = EncryptedDescriptor(
                field_type=str,
                encrypt_fn=lambda v: v,
                decrypt_fn=lambda v: v,
            )
        Foo.secret.__set_name__(Foo, 'secret')
        meta = Foo.secret.get_metadata()
        assert meta["has_encrypt_fn"] is True
        assert meta["has_decrypt_fn"] is True
