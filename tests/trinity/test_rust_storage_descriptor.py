"""Whitebox tests for RustStorageDescriptor -- T-CORE-5.3a."""
from __future__ import annotations
import importlib.util
import os
from unittest import mock
import pytest

_CWD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RUST_STORAGE_PATH = os.path.join(_CWD, "trinity", "descriptors", "rust_storage.py")
_spec = importlib.util.spec_from_file_location("rust_storage", _RUST_STORAGE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
RSD = _mod.RustStorageDescriptor
_UNSET = _mod._UNSET
BaseDesc = _mod.BaseDescriptor


class Obj:
    def __init__(self, eid=None, cid=None, **kw):
        self._entity_id = eid
        self._component_id = cid
        self.__dict__.update(kw)


class TestIdentity:
    def test_descriptor_id(self):
        assert RSD.descriptor_id == "rust_storage"
    def test_accepts_inner_empty(self):
        assert RSD.accepts_inner == ()
    def test_accepts_outer_wildcard(self):
        assert RSD.accepts_outer == ("*",)
    def test_slots(self):
        assert RSD.__slots__ == ("_default", "_default_factory", "_rust_offset")
    def test_is_subclass(self):
        assert issubclass(RSD, BaseDesc)
    def test_defaults(self):
        d = RSD(int)
        assert d._default is _UNSET and d._default_factory is None and d._rust_offset is None
    def test_steps_empty(self):
        assert RSD(int).descriptor_steps == []


class TestCtor:
    def test_sentinel(self):
        d = RSD()
        assert d._default is _UNSET and _UNSET is not None
    def test_default_val(self):
        assert RSD(int, default=42)._default == 42
    def test_factory(self):
        assert callable(RSD(str, default_factory=lambda: "h")._default_factory)
    def test_field_type(self):
        assert RSD(float)._field_type is float
    def test_none_vs_unset(self):
        e = RSD(int, default=None)
        assert e._default is None and e._default is not _UNSET
        u = RSD(int)
        assert u._default is _UNSET
    def test_offset_none(self):
        assert RSD(int)._rust_offset is None
    def test_offset_settable(self):
        d = RSD(int)
        d._rust_offset = 0
        assert d._rust_offset == 0
        d._rust_offset = 7
        assert d._rust_offset == 7
    def test_config(self):
        assert RSD(int, extra="v")._config.get("extra") == "v"


class TestGetStored:
    def test_no_eid(self):
        d = RSD(int, default=0); d.__set_name__(Obj, "h"); d._rust_offset = 0
        assert d._get_stored(Obj(eid=None, cid=1)) == 0
    def test_no_cid(self):
        d = RSD(int, default=99); d.__set_name__(Obj, "s"); d._rust_offset = 0
        assert d._get_stored(Obj(eid=1, cid=None)) == 99
    def test_no_offset(self):
        d = RSD(int, default=42); d.__set_name__(Obj, "a")
        assert d._get_stored(Obj(eid=1, cid=2)) == 42
    def test_all_guards_no_omega(self):
        d = RSD(int, default=7); d.__set_name__(Obj, "l"); d._rust_offset = 0
        assert d._get_stored(Obj(eid=5, cid=1)) == 7
    def test_import_error(self):
        d = RSD(str, default="fb"); d.__set_name__(Obj, "n"); d._rust_offset = 0
        with mock.patch.dict("sys.modules", {"_omega": None}):
            assert d._get_stored(Obj(eid=1, cid=2)) == "fb"
    def test_missing_cr(self):
        d = RSD(str, default="fb"); d.__set_name__(Obj, "t"); d._rust_offset = 0
        with mock.patch.dict("sys.modules", {"_omega": object()}):
            assert d._get_stored(Obj(eid=1, cid=2)) == "fb"
    def test_runtime_error(self):
        d = RSD(float, default=1.5); d.__set_name__(Obj, "s"); d._rust_offset = 0
        class R:
            @staticmethod
            def component_read(*a, **kw):
                raise RuntimeError("x")
        with mock.patch.dict("sys.modules", {"_omega": R()}):
            assert d._get_stored(Obj(eid=10, cid=3)) == 1.5
    def test_eid_zero(self):
        d = RSD(int, default=999); d.__set_name__(Obj, "z"); d._rust_offset = 0
        assert d._get_stored(Obj(eid=0, cid=1)) == 999
    def test_offset_zero(self):
        d = RSD(int, default=77); d.__set_name__(Obj, "f"); d._rust_offset = 0
        assert d._get_stored(Obj(eid=5, cid=2)) == 77
    def test_no_eid_attr(self):
        d = RSD(int, default=0); d.__set_name__(Obj, "v"); d._rust_offset = 0
        class P: pass
        assert d._get_stored(P()) == 0
    def test_non_comp(self):
        d = RSD(str, default="p"); d.__set_name__(Obj, "l"); d._rust_offset = 0
        class P: pass
        assert d._get_stored(P()) == "p"
    def test_partial_no_eid(self):
        o = Obj(cid=1); del o._entity_id
        d = RSD(bool, default=False); d.__set_name__(Obj, "f"); d._rust_offset = 0
        assert d._get_stored(o) is False


class TestSetStored:
    def test_no_eid(self):
        d = RSD(int); d.__set_name__(Obj, "h"); d._rust_offset = 0
        o = Obj(eid=None, cid=1); d._set_stored(o, 100)
        assert o.__dict__["h"] == 100
    def test_no_cid(self):
        d = RSD(str); d.__set_name__(Obj, "t"); d._rust_offset = 0
        o = Obj(eid=1, cid=None); d._set_stored(o, "b")
        assert o.__dict__["t"] == "b"
    def test_no_offset(self):
        d = RSD(float); d.__set_name__(Obj, "w")
        o = Obj(eid=1, cid=2); d._set_stored(o, 3.5)
        assert o.__dict__["w"] == 3.5
    def test_no_omega(self):
        d = RSD(bool); d.__set_name__(Obj, "a"); d._rust_offset = 1
        o = Obj(eid=5, cid=1); d._set_stored(o, True)
        assert o.__dict__["a"] is True
    def test_import_error(self):
        d = RSD(int); d.__set_name__(Obj, "m"); d._rust_offset = 0
        o = Obj(eid=3, cid=1)
        with mock.patch.dict("sys.modules", {"_omega": None}):
            d._set_stored(o, 50)
        assert o.__dict__["m"] == 50
    def test_missing_cw(self):
        d = RSD(int); d.__set_name__(Obj, "c"); d._rust_offset = 0
        o = Obj(eid=1, cid=2)
        with mock.patch.dict("sys.modules", {"_omega": object()}):
            d._set_stored(o, 999)
        assert o.__dict__["c"] == 999
    def test_runtime_error(self):
        d = RSD(float); d.__set_name__(Obj, "e"); d._rust_offset = 0
        class R:
            @staticmethod
            def component_write(*a, **kw):
                raise RuntimeError("x")
        o = Obj(eid=10, cid=2)
        with mock.patch.dict("sys.modules", {"_omega": R()}):
            d._set_stored(o, 0.75)
        assert o.__dict__["e"] == 0.75
    def test_early_return(self):
        d = RSD(int); d.__set_name__(Obj, "a"); d._rust_offset = 0
        o = Obj(eid=1, cid=2)
        log = []
        def _cw(e, c, off, v):
            log.append((e, c, off, v))
        with mock.patch.object(_mod, "_HAVE_OMEGA", True), \
             mock.patch.object(_mod, "component_write", _cw, create=True):
            d._set_stored(o, 99)
        assert log == [(1, 2, 0, 99)]
        assert "a" not in o.__dict__


class TestDeleteStored:
    def test_no_eid(self):
        d = RSD(int); d.__set_name__(Obj, "h"); d._rust_offset = 0
        o = Obj(eid=None, cid=1); o.__dict__["h"] = 100
        d._delete_stored(o)
        assert "h" not in o.__dict__

    def test_no_cid(self):
        d = RSD(str); d.__set_name__(Obj, "t"); d._rust_offset = 0
        o = Obj(eid=1, cid=None); o.__dict__["t"] = "b"
        d._delete_stored(o)
        assert "t" not in o.__dict__

    def test_no_offset(self):
        d = RSD(float); d.__set_name__(Obj, "w")
        o = Obj(eid=1, cid=2); o.__dict__["w"] = 3.5
        d._delete_stored(o)
        assert "w" not in o.__dict__

    def test_no_omega(self):
        d = RSD(bool); d.__set_name__(Obj, "a"); d._rust_offset = 1
        o = Obj(eid=5, cid=1); o.__dict__["a"] = True
        d._delete_stored(o)
        assert "a" not in o.__dict__

    def test_import_error(self):
        d = RSD(int); d.__set_name__(Obj, "m"); d._rust_offset = 0
        o = Obj(eid=3, cid=1); o.__dict__["m"] = 50
        with mock.patch.dict("sys.modules", {"_omega": None}):
            d._delete_stored(o)
        assert "m" not in o.__dict__

    def test_missing_cd(self):
        d = RSD(int); d.__set_name__(Obj, "c"); d._rust_offset = 0
        o = Obj(eid=1, cid=2); o.__dict__["c"] = 999
        with mock.patch.dict("sys.modules", {"_omega": object()}):
            d._delete_stored(o)
        assert "c" not in o.__dict__

    def test_runtime_error(self):
        d = RSD(float); d.__set_name__(Obj, "e"); d._rust_offset = 0
        class R:
            @staticmethod
            def component_delete(*a, **kw):
                raise RuntimeError("x")
        o = Obj(eid=10, cid=2); o.__dict__["e"] = 0.75
        with mock.patch.dict("sys.modules", {"_omega": R()}):
            d._delete_stored(o)
        assert "e" not in o.__dict__

    def test_omega_delete_args(self):
        d = RSD(float); d.__set_name__(Obj, "x"); d._rust_offset = 4
        log = []
        def _cd(e, c, off):
            log.append((e, c, off))
        with mock.patch.object(_mod, "_HAVE_OMEGA", True), \
             mock.patch.object(_mod, "component_delete", _cd, create=True):
            d._delete_stored(Obj(eid=7, cid=3))
        assert log == [(7, 3, 4)]

    def test_early_return_no_dict_pop(self):
        d = RSD(int); d.__set_name__(Obj, "a"); d._rust_offset = 0
        o = Obj(eid=1, cid=2)
        log = []
        def _cd(e, c, off):
            log.append((e, c, off))
        with mock.patch.object(_mod, "_HAVE_OMEGA", True), \
             mock.patch.object(_mod, "component_delete", _cd, create=True):
            d._delete_stored(o)
        assert log == [(1, 2, 0)]
        assert "a" not in o.__dict__

    def test_non_component(self):
        d = RSD(str, default="p"); d.__set_name__(Obj, "l"); d._rust_offset = 0
        class P: pass
        p = P()
        d._delete_stored(p)

    def test_partial_no_eid_attr(self):
        d = RSD(int); d.__set_name__(Obj, "v"); d._rust_offset = 0
        class P: pass
        d._delete_stored(P())

    def test_fallback_on_runtime_error_cleans_dict(self):
        d = RSD(str); d.__set_name__(Obj, "n"); d._rust_offset = 0
        o = Obj(eid=1, cid=2); o.__dict__["n"] = "IN_DICT"
        def _cd(*a):
            raise RuntimeError("omega down")
        with mock.patch.object(_mod, "_HAVE_OMEGA", True), \
             mock.patch.object(_mod, "component_delete", _cd, create=True):
            d._delete_stored(o)
        assert "n" not in o.__dict__


class TestOmega:
    def test_read_args(self):
        d = RSD(float); d.__set_name__(Obj, "x"); d._rust_offset = 4
        log = []
        def _cr(e, c, off, ft):
            log.append((e, c, off, ft)); return 3.14
        with mock.patch.object(_mod, "_HAVE_OMEGA", True), \
             mock.patch.object(_mod, "component_read", _cr, create=True):
            assert d._get_stored(Obj(eid=7, cid=3)) == 3.14
        assert log == [(7, 3, 4, float)]
    def test_write_args(self):
        d = RSD(int); d.__set_name__(Obj, "s"); d._rust_offset = 8
        log = []
        def _cw2(e, c, off, v):
            log.append((e, c, off, v))
        with mock.patch.object(_mod, "_HAVE_OMEGA", True), \
             mock.patch.object(_mod, "component_write", _cw2, create=True):
            d._set_stored(Obj(eid=2, cid=5), 999)
        assert log == [(2, 5, 8, 999)]
    def test_read_ignores_dict(self):
        d = RSD(str); d.__set_name__(Obj, "n"); d._rust_offset = 0
        o = Obj(eid=1, cid=1); o.__dict__["n"] = "FROM_DICT"
        def _cr2(*a): return "FROM_RUST"
        with mock.patch.object(_mod, "_HAVE_OMEGA", True), \
             mock.patch.object(_mod, "component_read", _cr2, create=True):
            assert d._get_stored(o) == "FROM_RUST"


class TestDictGet:
    def test_exists(self):
        d = RSD(int); d.__set_name__(Obj, "l"); o = Obj(); o.__dict__["l"] = 5
        assert d._dict_get(o) == 5
    def test_factory(self):
        d = RSD(list, default_factory=list); d.__set_name__(Obj, "i")
        assert d._dict_get(Obj()) == []
    def test_default(self):
        d = RSD(str, default="h"); d.__set_name__(Obj, "g")
        assert d._dict_get(Obj()) == "h"
    def test_none(self):
        d = RSD(int); d.__set_name__(Obj, "u")
        assert d._dict_get(Obj()) is None
    def test_sets_on_dict(self):
        d = RSD(int, default=42); d.__set_name__(Obj, "a"); o = Obj()
        d._dict_get(o); assert o.__dict__.get("a") == 42
    def test_no_overwrite(self):
        d = RSD(int, default=0); d.__set_name__(Obj, "s"); o = Obj()
        o.__dict__["s"] = 999; assert d._dict_get(o) == 999


class TestDictSet:
    def test_write(self):
        d = RSD(str); d.__set_name__(Obj, "t"); o = Obj()
        d._dict_set(o, "b"); assert o.__dict__["t"] == "b"
    def test_overwrite(self):
        d = RSD(int); d.__set_name__(Obj, "a"); o = Obj()
        o.__dict__["a"] = 10; d._dict_set(o, 99); assert o.__dict__["a"] == 99


class TestResolve:
    def test_factory_wins(self):
        assert RSD(int, default=1, default_factory=lambda: 2)._resolve_default() == 2
    def test_default(self):
        assert RSD(str, default="fb")._resolve_default() == "fb"
    def test_neither(self):
        assert RSD(int)._resolve_default() is None
    def test_explicit_none(self):
        assert RSD(int, default=None)._resolve_default() is None
    def test_factory_called_each(self):
        calls = []
        d = RSD(list, default_factory=lambda: calls.append(1) or [])
        d._resolve_default(); d._resolve_default()
        assert len(calls) == 2


class TestMeta:
    def test_rust_offset(self):
        d = RSD(int); d._rust_offset = 3
        assert d.get_metadata().get("rust_offset") == 3
    def test_has_default_true(self):
        assert RSD(int, default=42).get_metadata().get("has_default") is True
    def test_has_default_false(self):
        assert RSD(int).get_metadata().get("has_default") is False
    def test_super_fields(self):
        d = RSD(float); d.__set_name__(Obj, "x")
        m = d.get_metadata()
        assert m["descriptor_id"] == "rust_storage" and m["name"] == "x" and m["field_type"] == "float"
    def test_steps(self):
        assert RSD(int).descriptor_steps == []


class TestSetName:
    def test_sets(self):
        d = RSD(int); d.__set_name__(Obj, "h")
        assert d._name == "h" and d._owner is Obj
    def test_prop(self):
        d = RSD(int); d.__set_name__(Obj, "m")
        assert d.name == "m"


class TestEdge:
    def test_round_trip(self):
        d = RSD(int, default=0); d.__set_name__(Obj, "c"); o = Obj()
        d._set_stored(o, 42); assert d._get_stored(o) == 42
    def test_field_type(self):
        d = RSD(bool); d.__set_name__(Obj, "f")
        assert d._field_type is bool
    def test_chain(self):
        d = RSD(int)
        assert len(d.get_chain()) == 1 and d.get_chain()[0] is d
    def test_repr(self):
        d = RSD(int); d.__set_name__(Obj, "x")
        assert "rust_storage" in repr(d) and "x" in repr(d)
    def test_isolation(self):
        d1 = RSD(int, default=1); d2 = RSD(int, default=2)
        d1.__set_name__(Obj, "a"); d2.__set_name__(Obj, "b")
        o = Obj()
        assert d1._get_stored(o) == 1 and d2._get_stored(o) == 2
        d1._set_stored(o, 10)
        assert o.__dict__["a"] == 10 and o.__dict__["b"] == 2


class TestCMeta:
    def test_has_rust_offset(self):
        assert hasattr(RSD(int), "_rust_offset")
    def test_meta_key(self):
        assert "rust_offset" in RSD(int).get_metadata()
