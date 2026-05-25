"""
Tests for trinity.decorators.base module.

Covers decorator tracking, attribute attachment, merge modes,
validation helpers, introspection, platform detection, stub generation,
and deprecated factory functions.
"""

import warnings
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from trinity.decorators.base import (
    DecoratorValidationError,
    PlatformUnavailableError,
    Tier,
    attach_attributes,
    check_excluded_decorators,
    check_required_decorators,
    create_unavailable_stub,
    get_applied_decorators,
    get_attribute,
    get_current_arch,
    get_current_platform,
    get_decorator_chain,
    has_decorator,
    inspect_decorated,
    make_configurable_decorator,
    make_marker_decorator,
    merge_attributes,
    registry,
    track_decorator,
    validate_parameters,
    validate_target_type,
)


# =============================================================================
# HELPERS
# =============================================================================


def _fresh_class(name="Target"):
    """Return a new class with no decorators attached."""
    return type(name, (), {})


def _fresh_function():
    """Return a new plain function."""

    def fn():
        pass

    return fn


# =============================================================================
# DECORATOR TRACKING
# =============================================================================


class TestTrackDecorator:
    def test_single_decorator(self):
        cls = _fresh_class()
        track_decorator(cls, "alpha")
        assert get_applied_decorators(cls) == ["alpha"]

    def test_multiple_decorators_preserve_order(self):
        cls = _fresh_class()
        track_decorator(cls, "first")
        track_decorator(cls, "second")
        track_decorator(cls, "third")
        assert get_applied_decorators(cls) == ["first", "second", "third"]


class TestGetAppliedDecorators:
    def test_no_decorators_returns_empty(self):
        cls = _fresh_class()
        assert get_applied_decorators(cls) == []

    def test_returns_copy_not_reference(self):
        cls = _fresh_class()
        track_decorator(cls, "dec")
        result = get_applied_decorators(cls)
        result.append("injected")
        # The internal list must be unaffected
        assert get_applied_decorators(cls) == ["dec"]


class TestHasDecorator:
    def test_present(self):
        cls = _fresh_class()
        track_decorator(cls, "present")
        assert has_decorator(cls, "present") is True

    def test_absent(self):
        cls = _fresh_class()
        assert has_decorator(cls, "nonexistent") is False

    def test_absent_on_decorated_target(self):
        cls = _fresh_class()
        track_decorator(cls, "other")
        assert has_decorator(cls, "missing") is False


# =============================================================================
# ATTRIBUTE ATTACHMENT
# =============================================================================


class TestAttachAttributes:
    def test_single_attribute(self):
        cls = _fresh_class()
        attach_attributes(cls, _enabled=True)
        assert cls._enabled is True

    def test_multiple_attributes(self):
        cls = _fresh_class()
        attach_attributes(cls, _backend="cython", _nogil=True, _level=3)
        assert cls._backend == "cython"
        assert cls._nogil is True
        assert cls._level == 3


class TestGetAttribute:
    def test_existing_attribute(self):
        cls = _fresh_class()
        cls._mode = "fast"
        assert get_attribute(cls, "_mode") == "fast"

    def test_missing_returns_default(self):
        cls = _fresh_class()
        assert get_attribute(cls, "_missing", "fallback") == "fallback"

    def test_missing_returns_none_by_default(self):
        cls = _fresh_class()
        assert get_attribute(cls, "_missing") is None


# =============================================================================
# MERGE ATTRIBUTES
# =============================================================================


class TestMergeAttributesReplace:
    def test_replace_overwrites_existing(self):
        cls = _fresh_class()
        cls._val = "old"
        merge_attributes(cls, "_val", "new", merge_type="replace")
        assert cls._val == "new"

    def test_replace_is_default_mode(self):
        cls = _fresh_class()
        cls._val = "old"
        merge_attributes(cls, "_val", "new")
        assert cls._val == "new"


class TestMergeAttributesUnion:
    def test_union_merges_sets(self):
        cls = _fresh_class()
        cls._tags = {"a", "b"}
        merge_attributes(cls, "_tags", {"b", "c"}, merge_type="union")
        assert cls._tags == {"a", "b", "c"}

    def test_union_on_non_set_raises_type_error(self):
        cls = _fresh_class()
        cls._tags = "not a set"
        with pytest.raises(TypeError, match="merge_type='union' requires.*set"):
            merge_attributes(cls, "_tags", {"x"}, merge_type="union")


class TestMergeAttributesExtend:
    def test_extend_appends_list(self):
        cls = _fresh_class()
        cls._items = [1, 2]
        merge_attributes(cls, "_items", [3, 4], merge_type="extend")
        assert cls._items == [1, 2, 3, 4]

    def test_extend_on_non_list_raises_type_error(self):
        cls = _fresh_class()
        cls._items = "not a list"
        with pytest.raises(TypeError, match="merge_type='extend' requires.*list"):
            merge_attributes(cls, "_items", [1], merge_type="extend")

    def test_extend_with_string_value_wraps_in_list(self):
        """A string value is wrapped in a list before extending (not exploded into chars)."""
        cls = _fresh_class()
        cls._items = ["a"]
        merge_attributes(cls, "_items", "bc", merge_type="extend")
        assert cls._items == ["a", "bc"]


class TestMergeAttributesUpdate:
    def test_update_merges_dicts(self):
        cls = _fresh_class()
        cls._cfg = {"a": 1}
        merge_attributes(cls, "_cfg", {"b": 2}, merge_type="update")
        assert cls._cfg == {"a": 1, "b": 2}

    def test_update_on_non_dict_raises_type_error(self):
        cls = _fresh_class()
        cls._cfg = 42
        with pytest.raises(TypeError, match="merge_type='update' requires.*dict"):
            merge_attributes(cls, "_cfg", {"x": 1}, merge_type="update")


class TestMergeAttributesFirstCall:
    def test_first_call_sets_regardless_of_mode(self):
        """When no existing value, any merge_type just sets it."""
        for mode in ("replace", "union", "extend", "update"):
            cls = _fresh_class()
            merge_attributes(cls, "_val", "initial", merge_type=mode)
            assert cls._val == "initial", f"Failed for mode={mode}"


# =============================================================================
# VALIDATION: TARGET TYPE
# =============================================================================


class TestValidateTargetType:
    def test_class_with_class_allowed(self):
        cls = _fresh_class()
        validate_target_type(cls, "test_dec", ("class",))  # should not raise

    def test_class_with_function_only_raises(self):
        cls = _fresh_class()
        with pytest.raises(DecoratorValidationError, match="cannot be applied to classes"):
            validate_target_type(cls, "test_dec", ("function",))

    def test_function_with_function_allowed(self):
        fn = _fresh_function()
        validate_target_type(fn, "test_dec", ("function",))  # should not raise

    def test_function_with_class_only_raises(self):
        fn = _fresh_function()
        with pytest.raises(DecoratorValidationError, match="cannot be applied to functions"):
            validate_target_type(fn, "test_dec", ("class",))

    def test_any_allows_class(self):
        cls = _fresh_class()
        validate_target_type(cls, "test_dec", ("any",))  # should not raise

    def test_any_allows_function(self):
        fn = _fresh_function()
        validate_target_type(fn, "test_dec", ("any",))  # should not raise

    def test_both_class_and_function_allow_class(self):
        cls = _fresh_class()
        validate_target_type(cls, "test_dec", ("class", "function"))

    def test_both_class_and_function_allow_function(self):
        fn = _fresh_function()
        validate_target_type(fn, "test_dec", ("class", "function"))


# =============================================================================
# VALIDATION: PARAMETERS
# =============================================================================


class TestValidateParameters:
    def test_valid_type_and_validator(self):
        validate_parameters(
            "test_dec",
            backend=("cython", str, lambda x: x in ("cython", "numba")),
        )  # should not raise

    def test_wrong_type_raises(self):
        with pytest.raises(DecoratorValidationError, match="must be str"):
            validate_parameters(
                "test_dec",
                backend=(123, str, None),
            )

    def test_right_type_failing_validator_raises(self):
        with pytest.raises(DecoratorValidationError, match="invalid value"):
            validate_parameters(
                "test_dec",
                backend=("bad", str, lambda x: x in ("cython", "numba")),
            )

    def test_multiple_params_first_fails(self):
        with pytest.raises(DecoratorValidationError, match="must be int"):
            validate_parameters(
                "test_dec",
                level=("not_int", int, None),
                name=("ok", str, None),
            )

    def test_no_validator_passes_with_correct_type(self):
        validate_parameters(
            "test_dec",
            flag=(True, bool, None),
        )


# =============================================================================
# REQUIRED / EXCLUDED DECORATORS
# =============================================================================


class TestCheckRequiredDecorators:
    def test_required_present_passes(self):
        cls = _fresh_class()
        track_decorator(cls, "base")
        check_required_decorators(cls, "child", ("base",))  # should not raise

    def test_required_missing_raises(self):
        cls = _fresh_class()
        with pytest.raises(DecoratorValidationError, match="requires @base"):
            check_required_decorators(cls, "child", ("base",))


class TestCheckExcludedDecorators:
    def test_excluded_absent_passes(self):
        cls = _fresh_class()
        check_excluded_decorators(cls, "child", ("conflict",))  # should not raise

    def test_excluded_present_raises(self):
        cls = _fresh_class()
        track_decorator(cls, "conflict")
        with pytest.raises(DecoratorValidationError, match="cannot be combined with @conflict"):
            check_excluded_decorators(cls, "child", ("conflict",))


# =============================================================================
# INTROSPECTION
# =============================================================================


class TestInspectDecorated:
    def test_with_tracked_decorator(self):
        cls = _fresh_class("MyClass")
        track_decorator(cls, "some_dec")
        info = inspect_decorated(cls)
        assert info["target"] == "MyClass"
        assert info["type"] == "class"
        assert info["decorator_count"] == 1
        assert isinstance(info["decorators"], list)
        assert len(info["decorators"]) == 1
        # The decorator may or may not be in the registry; verify structure
        assert info["decorators"][0]["name"] == "some_dec"

    def test_no_decorators(self):
        cls = _fresh_class("Empty")
        info = inspect_decorated(cls)
        assert info["decorator_count"] == 0
        assert info["decorators"] == []

    def test_return_dict_structure(self):
        fn = _fresh_function()
        info = inspect_decorated(fn)
        assert set(info.keys()) == {
            "target",
            "type",
            "decorators",
            "attributes",
            "decorator_count",
        }
        assert info["type"] == "function"


class TestGetDecoratorChain:
    def test_matches_get_applied_decorators(self):
        cls = _fresh_class()
        track_decorator(cls, "a")
        track_decorator(cls, "b")
        assert get_decorator_chain(cls) == get_applied_decorators(cls)


# =============================================================================
# PLATFORM DETECTION
# =============================================================================


class TestGetCurrentPlatform:
    @pytest.mark.parametrize(
        "sys_value, expected",
        [
            ("win32", "windows"),
            ("darwin", "macos"),
            ("linux", "linux"),
            ("linux2", "linux"),  # startswith("linux")
            ("emscripten", "web"),
            ("freebsd12", "freebsd12"),  # passthrough
        ],
    )
    def test_platform_mapping(self, sys_value, expected):
        with patch("trinity.decorators.base.sys") as mock_sys:
            mock_sys.platform = sys_value
            assert get_current_platform() == expected


class TestGetCurrentArch:
    @pytest.mark.parametrize(
        "machine_value, expected",
        [
            ("x86_64", "x86_64"),
            ("AMD64", "x86_64"),  # case-insensitive via .lower()
            ("amd64", "x86_64"),
            ("arm64", "arm64"),
            ("aarch64", "arm64"),
            ("i386", "x86"),
            ("i686", "x86"),
            ("x86", "x86"),
            ("armv7l", "arm"),
            ("riscv64", "riscv64"),  # passthrough
        ],
    )
    def test_arch_mapping(self, machine_value, expected):
        # platform is imported locally inside get_current_arch, so patch the stdlib module
        with patch("platform.machine", return_value=machine_value):
            assert get_current_arch() == expected


# =============================================================================
# STUB GENERATOR
# =============================================================================


class TestCreateUnavailableStub:
    def test_stub_raises_platform_unavailable_error(self):
        stub = create_unavailable_stub("my_func", "platform", "not on linux")
        with pytest.raises(PlatformUnavailableError, match="my_func is not available"):
            stub()

    def test_stub_raises_with_args(self):
        stub = create_unavailable_stub("f", "d", "reason")
        with pytest.raises(PlatformUnavailableError):
            stub(1, 2, key="val")

    def test_stub_has_unavailable_flag(self):
        stub = create_unavailable_stub("f", "dec", "reason")
        assert stub._unavailable is True

    def test_stub_has_unavailable_reason(self):
        stub = create_unavailable_stub("f", "dec", "no GPU")
        assert stub._unavailable_reason == "no GPU"

    def test_stub_has_correct_name(self):
        stub = create_unavailable_stub("render_frame", "dec", "reason")
        assert stub.__name__ == "render_frame"

    def test_stub_has_doc(self):
        stub = create_unavailable_stub("f", "dec", "some reason")
        assert "UNAVAILABLE" in stub.__doc__
        assert "some reason" in stub.__doc__


class TestPlatformUnavailableError:
    def test_is_runtime_error(self):
        assert issubclass(PlatformUnavailableError, RuntimeError)


# =============================================================================
# DEPRECATED FACTORIES
# =============================================================================


class TestMakeMarkerDecoratorDeprecation:
    def test_emits_deprecation_warning(self):
        unique_name = "_test_marker_deprecated_check"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dec = make_marker_decorator(
                name=unique_name,
                tier=Tier.COMPILATION,
                attribute_name="_test_marker",
                doc="test marker",
            )
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1, "Expected DeprecationWarning from make_marker_decorator"

    def test_marker_decorator_works(self):
        unique_name = "_test_marker_works"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dec = make_marker_decorator(
                name=unique_name,
                tier=Tier.COMPILATION,
                attribute_name="_test_marker_w",
                doc="test marker",
            )
        cls = _fresh_class()
        result = dec(cls)
        assert result._test_marker_w is True
        assert has_decorator(result, unique_name)


class TestMakeConfigurableDecoratorDeprecation:
    def test_emits_deprecation_warning(self):
        unique_name = "_test_cfg_dep_warn"

        @dataclass
        class Cfg:
            level: int = 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            make_configurable_decorator(
                name=unique_name,
                tier=Tier.COMPILATION,
                config_class=Cfg,
                attribute_name="_test_cfg_w",
                doc="test",
            )
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) >= 1, "Expected DeprecationWarning from make_configurable_decorator"

    def test_configurable_decorator_works(self):
        unique_name = "_test_cfg_works"

        @dataclass
        class TestConfig:
            level: int = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            factory = make_configurable_decorator(
                name=unique_name,
                tier=Tier.COMPILATION,
                config_class=TestConfig,
                attribute_name="_test_config",
                doc="test configurable",
            )
        cls = _fresh_class()
        decorated = factory(level=5)(cls)
        assert decorated._test_config.level == 5
        assert has_decorator(decorated, unique_name)
