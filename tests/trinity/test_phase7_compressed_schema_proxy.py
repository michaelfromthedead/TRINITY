"""Tests for Phase 7 descriptors: Compressed, Schema, Proxy."""

import zlib

import pytest

from trinity.decorators.ops import Op, Step
from trinity.descriptors.compressed import CompressedDescriptor
from trinity.descriptors.schema import SchemaDescriptor
from trinity.descriptors.proxy import ProxyDescriptor


# =========================================================================
# CompressedDescriptor
# =========================================================================


class TestCompressedDescriptor:
    def _make_class(self, algorithm="zlib"):
        desc = CompressedDescriptor(algorithm=algorithm)

        class Obj:
            data = desc

        desc.__set_name__(Obj, "data")
        return Obj

    def test_roundtrip_zlib(self):
        Obj = self._make_class("zlib")
        obj = Obj()
        obj.data = b"hello world"
        result = obj.data
        assert result == b"hello world"

    def test_roundtrip_string(self):
        Obj = self._make_class("zlib")
        obj = Obj()
        obj.data = "hello world"
        assert obj.data == b"hello world"

    def test_zlib_actually_compresses(self):
        Obj = self._make_class("zlib")
        obj = Obj()
        obj.data = b"hello world"
        raw = obj.__dict__["data"]
        assert raw != b"hello world"
        assert zlib.decompress(raw) == b"hello world"

    def test_none_algorithm(self):
        Obj = self._make_class("none")
        obj = Obj()
        obj.data = b"raw bytes"
        assert obj.data == b"raw bytes"
        assert obj.__dict__["data"] == b"raw bytes"

    def test_none_value(self):
        Obj = self._make_class("zlib")
        obj = Obj()
        assert obj.data is None

    def test_descriptor_steps(self):
        desc = CompressedDescriptor(algorithm="zlib")
        steps = desc.descriptor_steps
        assert len(steps) == 2
        assert steps[0].op is Op.INTERCEPT
        assert steps[1].op is Op.TAG

    def test_invalid_algorithm(self):
        with pytest.raises(ValueError, match="algorithm"):
            CompressedDescriptor(algorithm="brotli")


# =========================================================================
# SchemaDescriptor
# =========================================================================


class TestSchemaDescriptor:
    def _make_class(self, schema):
        desc = SchemaDescriptor(schema=schema)

        class Obj:
            data = desc

        desc.__set_name__(Obj, "data")
        return Obj

    def test_valid_type_passes(self):
        Obj = self._make_class({"type": "string"})
        obj = Obj()
        obj.data = "hello"
        assert obj.data == "hello"

    def test_invalid_type_fails(self):
        Obj = self._make_class({"type": "string"})
        obj = Obj()
        with pytest.raises(TypeError, match="Expected type 'string'"):
            obj.data = 42

    def test_int_type(self):
        Obj = self._make_class({"type": "integer"})
        obj = Obj()
        obj.data = 10
        assert obj.data == 10
        with pytest.raises(TypeError, match="Expected type"):
            obj.data = "not an int"

    def test_required_keys(self):
        Obj = self._make_class({"type": "dict", "required": ["name", "age"]})
        obj = Obj()
        obj.data = {"name": "Alice", "age": 30}
        assert obj.data["name"] == "Alice"
        with pytest.raises(ValueError, match="Missing required key: 'age'"):
            obj.data = {"name": "Bob"}

    def test_no_type_constraint(self):
        Obj = self._make_class({})
        obj = Obj()
        obj.data = "anything"
        assert obj.data == "anything"

    def test_descriptor_steps(self):
        desc = SchemaDescriptor(schema={"type": "int"})
        steps = desc.descriptor_steps
        assert len(steps) == 2
        assert steps[0].op.value == "validate"


# =========================================================================
# ProxyDescriptor
# =========================================================================


class TestProxyDescriptor:
    def test_reads_from_target(self):
        class Inner:
            def __init__(self):
                self.value = 42

        desc = ProxyDescriptor(target_cls=Inner, target_field="value")

        class Outer:
            proxied = desc

        desc.__set_name__(Outer, "proxied")
        outer = Outer()
        outer.inner = Inner()
        assert outer.proxied == 42

    def test_writes_to_target(self):
        class Inner:
            def __init__(self):
                self.value = 0

        desc = ProxyDescriptor(target_cls=Inner, target_field="value")

        class Outer:
            proxied = desc

        desc.__set_name__(Outer, "proxied")
        outer = Outer()
        outer.inner = Inner()
        outer.proxied = 99
        assert outer.inner.value == 99

    def test_no_target_raises(self):
        class Inner:
            pass

        desc = ProxyDescriptor(target_cls=Inner, target_field="value")

        class Outer:
            proxied = desc

        desc.__set_name__(Outer, "proxied")
        outer = Outer()
        with pytest.raises(AttributeError, match="No attribute of type"):
            _ = outer.proxied

    def test_descriptor_steps(self):
        class Dummy:
            pass

        desc = ProxyDescriptor(target_cls=Dummy, target_field="x")
        steps = desc.descriptor_steps
        assert len(steps) == 1
        assert steps[0].op.value == "intercept"

    def test_class_access_returns_descriptor(self):
        class Inner:
            pass

        desc = ProxyDescriptor(target_cls=Inner, target_field="x")

        class Outer:
            proxied = desc

        desc.__set_name__(Outer, "proxied")
        assert Outer.proxied is desc
