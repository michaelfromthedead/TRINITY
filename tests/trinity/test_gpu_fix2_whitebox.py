"""
T-GPU-1.6 FIX#2 WHITEBOX verification.

FIX#2 applied M3:
  - H1  _resolve_wgpu_usage_flags auto-adds COPY_DST (0x0008) when indirect
        usage is present. Storage alone does NOT auto-add COPY_DST as it
        may be GPU-only (compute shader write, render pass read).
  - M3  All 6 after-step functions guard target._tags access with
        getattr(target, "_tags", {}).get() instead of bare target._tags.get().
  - M4  Unknown wgpu usage flag name triggers warnings.warn().

Whitebox tests:
  1. H1: _resolve_wgpu_usage_flags auto-adds COPY_DST for {"indirect"},
         but NOT for {"storage"} alone.
  2. H1: @gpu_buffer(usage={"storage","indirect"}) produces resolved usage
         that includes COPY_DST (0x0008) in _gpu_wgpu_usage.
  3. M3: All 6 after-step functions use getattr(target, "_tags", {}) guard.
  4. M3: Class without _tags does not raise AttributeError from getattr
         guard in _after_bind_group and _after_dispatch (full fallback).
  5. M4: Unknown flag name triggers UserWarning.
  6. COPY_DST auto-add: three usage combinations verified.
"""

import inspect
import warnings

import pytest

from trinity.decorators.gpu import (
    _WGPU_USAGE_FLAGS,
    _after_bind_group,
    _after_dispatch,
    _after_gpu_buffer,
    _after_gpu_kernel,
    _after_render_pass,
    _after_shader,
    _resolve_wgpu_usage_flags,
    gpu_buffer,
)


# =============================================================================
# H1: COPY_DST auto-add in _resolve_wgpu_usage_flags
# =============================================================================


class TestH1CopyDstAutoAddInResolver:
    """H1: _resolve_wgpu_usage_flags auto-adds COPY_DST (0x0008).

    WebGPU spec requires STORAGE and INDIRECT buffers to also have COPY_DST.
    """

    # --- Basic combinations ---

    def test_indirect_gets_copy_dst(self):
        """{"indirect"} -> INDIRECT | COPY_DST = 0x0108"""
        flags = _resolve_wgpu_usage_flags(frozenset({"indirect"}))
        assert flags == (0x0100 | 0x0008), (
            f"Got {hex(flags)}, expected 0x0108"
        )

    def test_storage_no_copy_dst(self):
        """{"storage"} -> STORAGE only (no auto COPY_DST per FIX#2)"""
        flags = _resolve_wgpu_usage_flags(frozenset({"storage"}))
        # FIX#2: storage alone does NOT auto-add COPY_DST (may be GPU-only)
        assert flags == 0x0080, (
            f"Got {hex(flags)}, expected 0x0080"
        )

    def test_storage_and_indirect_gets_copy_dst_once(self):
        """{"storage","indirect"} -> STORAGE | INDIRECT | COPY_DST = 0x0188"""
        flags = _resolve_wgpu_usage_flags(
            frozenset({"storage", "indirect"})
        )
        assert flags == (0x0080 | 0x0100 | 0x0008), (
            f"Got {hex(flags)}, expected 0x0188"
        )

    # --- Non-affected flag names ---

    def test_vertex_no_copy_dst(self):
        assert _resolve_wgpu_usage_flags(frozenset({"vertex"})) == 0x0020

    def test_uniform_no_copy_dst(self):
        assert _resolve_wgpu_usage_flags(frozenset({"uniform"})) == 0x0040

    def test_index_no_copy_dst(self):
        assert _resolve_wgpu_usage_flags(frozenset({"index"})) == 0x0010

    def test_copy_src_no_copy_dst(self):
        assert _resolve_wgpu_usage_flags(frozenset({"copy_src"})) == 0x0004

    def test_map_read_no_copy_dst(self):
        assert _resolve_wgpu_usage_flags(frozenset({"map_read"})) == 0x0001

    def test_map_write_no_copy_dst(self):
        assert _resolve_wgpu_usage_flags(frozenset({"map_write"})) == 0x0002

    def test_empty_set_no_copy_dst(self):
        assert _resolve_wgpu_usage_flags(frozenset()) == 0

    # --- Combined with other flags ---

    def test_explicit_copy_dst_not_duplicated(self):
        """storage + copy_dst -> same bitmask as auto-add."""
        flags = _resolve_wgpu_usage_flags(
            frozenset({"storage", "copy_dst"})
        )
        assert flags == (0x0080 | 0x0008)

    def test_storage_with_other_flags_no_copy_dst(self):
        # FIX#2: storage does NOT auto-add COPY_DST even with other flags
        flags = _resolve_wgpu_usage_flags(
            frozenset({"storage", "uniform", "vertex"})
        )
        assert not (flags & 0x0008), "COPY_DST should NOT be present"
        assert flags == (0x0080 | 0x0040 | 0x0020)

    def test_indirect_with_other_flags_still_gets_copy_dst(self):
        flags = _resolve_wgpu_usage_flags(
            frozenset({"indirect", "index", "vertex"})
        )
        assert flags & 0x0008, "COPY_DST should be present"
        assert flags == (0x0100 | 0x0010 | 0x0020 | 0x0008)

    def test_all_nine_flags_includes_copy_dst(self):
        """All 9 flags: COPY_DST included once, not double-counted."""
        all_names = set(_WGPU_USAGE_FLAGS.keys())
        flags = _resolve_wgpu_usage_flags(frozenset(all_names))
        expected = sum(_WGPU_USAGE_FLAGS.values())
        assert flags == expected, (
            f"Got {hex(flags)}, expected {hex(expected)}"
        )


# =============================================================================
# H1: @gpu_buffer decorator -> _gpu_wgpu_usage includes COPY_DST
# =============================================================================


class TestH1CopyDstOnDecoratedClass:
    """Verify @gpu_buffer classes get COPY_DST in resolved wgpu usage."""

    def test_storage_buffer_no_copy_dst(self):
        # FIX#2: storage alone does NOT auto-add COPY_DST
        @gpu_buffer(usage={"storage"})
        class Buf:
            data: float

        assert not (Buf._gpu_wgpu_usage & 0x0008), "COPY_DST should NOT be present"
        assert Buf._gpu_wgpu_usage == 0x0080

    def test_indirect_buffer_gets_copy_dst(self):
        @gpu_buffer(usage={"indirect"})
        class Buf:
            data: float

        assert Buf._gpu_wgpu_usage & 0x0008, "COPY_DST bit missing"
        assert Buf._gpu_wgpu_usage == (0x0100 | 0x0008)

    def test_storage_and_indirect_gets_copy_dst(self):
        @gpu_buffer(usage={"storage", "indirect"})
        class Buf:
            data: float

        assert Buf._gpu_wgpu_usage & 0x0008, "COPY_DST bit missing"
        assert Buf._gpu_wgpu_usage == (0x0080 | 0x0100 | 0x0008)

    def test_vertex_does_not_get_copy_dst(self):
        @gpu_buffer(usage={"vertex"})
        class Buf:
            data: float

        assert not (Buf._gpu_wgpu_usage & 0x0008), (
            "COPY_DST should NOT be present on vertex buffer"
        )
        assert Buf._gpu_wgpu_usage == 0x0020

    def test_uniform_does_not_get_copy_dst(self):
        @gpu_buffer(usage={"uniform"})
        class Buf:
            data: float

        assert not (Buf._gpu_wgpu_usage & 0x0008), (
            "COPY_DST should NOT be present on uniform buffer"
        )
        assert Buf._gpu_wgpu_usage == 0x0040

    def test_gpu_usage_preserves_original_names(self):
        """_gpu_usage stores original user-supplied names (not mutated)."""
        @gpu_buffer(usage={"storage", "indirect"})
        class Buf:
            data: float

        assert "storage" in Buf._gpu_usage
        assert "indirect" in Buf._gpu_usage
        assert "copy_dst" not in Buf._gpu_usage, (
            "_gpu_usage should not be mutated"
        )

    def test_default_storage_no_copy_dst(self):
        """Default @gpu_buffer() uses usage={"storage"} -> no COPY_DST per FIX#2."""
        @gpu_buffer()
        class Buf:
            data: float

        # FIX#2: storage alone does NOT auto-add COPY_DST
        assert not (Buf._gpu_wgpu_usage & 0x0008), (
            "Default storage buffer should NOT get COPY_DST per FIX#2"
        )
        assert Buf._gpu_wgpu_usage == 0x0080


# =============================================================================
# M3: getattr(_tags, {}) guard in all 6 after-step functions
# =============================================================================


class TestM3GetattrGuardSourceCode:
    """Verify source code of all 6 after-step functions uses getattr guard."""

    AFTER_FUNCS = [
        (_after_gpu_buffer, "_after_gpu_buffer"),
        (_after_gpu_kernel, "_after_gpu_kernel"),
        (_after_bind_group, "_after_bind_group"),
        (_after_dispatch, "_after_dispatch"),
        (_after_shader, "_after_shader"),
        (_after_render_pass, "_after_render_pass"),
    ]

    @pytest.mark.parametrize("func,name", AFTER_FUNCS)
    def test_getattr_in_source(self, func, name):
        """Function source contains getattr(target, '_tags', {})."""
        src = inspect.getsource(func)
        assert "getattr" in src, f"{name} does not use getattr()"
        assert '"_tags"' in src or "'_tags'" in src, (
            f"{name} does not reference _tags"
        )

    def test_all_six_use_getattr_not_bare(self):
        """Verify NONE of the 6 functions use bare target._tags.get()."""
        for func, name in self.AFTER_FUNCS:
            src = inspect.getsource(func)
            found_guard = (
                'getattr(target, "_tags"' in src
                or "getattr(target, '_tags'" in src
            )
            assert found_guard, (
                f"{name} missing getattr(target, '_tags', {{}}) guard"
            )
            assert "target._tags.get(" not in src, (
                f"{name} still has bare target._tags.get()"
            )


class TestM3NakedClassNoTags:
    """Verify getattr guard prevents AttributeError for missing _tags.

    _after_bind_group and _after_dispatch have full fallback defaults
    (0 and False respectively) and work completely with a bare class.
    """

    # --- _after_bind_group ---

    def test_bind_group_no_tags_index_defaults_to_zero(self):
        class NoTags:
            pass

        _after_bind_group(NoTags, {})
        assert NoTags._bind_group is True
        assert NoTags._bind_group_index == 0

    def test_bind_group_no_tags_no_attribute_error_for_tags(self):
        class NoTags:
            pass

        try:
            _after_bind_group(NoTags, {})
        except AttributeError as e:
            if "_tags" in str(e):
                pytest.fail(
                    f"_after_bind_group raised AttributeError for _tags: {e}"
                )

    # --- _after_dispatch ---

    def test_dispatch_no_tags_indirect_defaults_to_false(self):
        class NoTags:
            pass

        _after_dispatch(NoTags, {})
        assert NoTags._dispatch is True
        assert NoTags._dispatch_indirect is False

    def test_dispatch_no_tags_no_attribute_error_for_tags(self):
        class NoTags:
            pass

        try:
            _after_dispatch(NoTags, {})
        except AttributeError as e:
            if "_tags" in str(e):
                pytest.fail(
                    f"_after_dispatch raised AttributeError for _tags: {e}"
                )

    # --- Remaining 4: getattr prevents _tags AttributeError ---
    # These may still fail on config.xxx (None) when _tags is missing,
    # but the specific AttributeError for _tags is prevented by the guard.

    @pytest.mark.parametrize("func,name", [
        (_after_gpu_buffer, "_after_gpu_buffer"),
        (_after_gpu_kernel, "_after_gpu_kernel"),
        (_after_shader, "_after_shader"),
        (_after_render_pass, "_after_render_pass"),
    ])
    def test_remaining_four_no_tags_attribute_error(self, func, name):
        """_tags AttributeError is prevented by getattr guard.

        These may fail on None.config.xxx (config = getattr(..,{}).get() = None),
        but NOT on target._tags AttributeError.
        """
        class NoTags:
            pass

        try:
            func(NoTags, {})
        except AttributeError as e:
            if "_tags" in str(e):
                pytest.fail(
                    f"{name} raised AttributeError for _tags "
                    f"despite getattr guard: {e}"
                )
        except (TypeError, Exception):
            pass  # Other errors are acceptable


# =============================================================================
# M4: Unknown flag name triggers UserWarning
# =============================================================================


class TestM4UnknownFlagWarning:
    """M4: Unknown wgpu usage flag name triggers UserWarning."""

    def test_unknown_flag_emits_warning(self):
        with pytest.warns(UserWarning, match="Unknown wgpu usage flag"):
            _resolve_wgpu_usage_flags(frozenset({"quantum"}))

    def test_unknown_flag_ignored_in_result(self):
        with pytest.warns(UserWarning):
            flags = _resolve_wgpu_usage_flags(frozenset({"quantum"}))
        assert flags == 0, "Unknown flag should not affect bitmask"

    def test_known_and_unknown_mixed_warns(self):
        with pytest.warns(UserWarning, match="quantum"):
            _resolve_wgpu_usage_flags(
                frozenset({"storage", "quantum"})
            )

    def test_known_and_unknown_ignores_unknown(self):
        with pytest.warns(UserWarning):
            flags = _resolve_wgpu_usage_flags(
                frozenset({"storage", "quantum"})
            )
        # FIX#2: storage alone does NOT auto-add COPY_DST
        assert flags == 0x0080, f"Got {hex(flags)}"

    def test_multiple_unknown_emits_warning(self):
        with pytest.warns(UserWarning):
            _resolve_wgpu_usage_flags(frozenset({"foo", "bar"}))

    def test_all_known_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            flags = _resolve_wgpu_usage_flags(
                frozenset({"storage", "indirect"})
            )
            assert flags & 0x0008, "COPY_DST should be present"

    def test_warning_message_lists_valid_flags(self):
        with pytest.warns(UserWarning) as record:
            _resolve_wgpu_usage_flags(frozenset({"quantum"}))
        msg = str(record[0].message)
        assert "copy_dst" in msg, "Warning should list valid flags"
        assert "storage" in msg, "Warning should list valid flags"
        assert "valid" in msg.lower()

    def test_warning_stacklevel(self):
        """stacklevel=2 ensures warning file/line points to caller."""
        with pytest.warns(UserWarning) as record:
            _resolve_wgpu_usage_flags(frozenset({"quantum"}))
        assert len(record) == 1
        # The warning filename should reference our test file
        assert "test_gpu_fix2_whitebox" in record[0].filename, (
            f"Warning filename is {record[0].filename}, "
            "expected test_gpu_fix2_whitebox"
        )

    def test_warning_on_decorator_raises_value_error_not_warning(self):
        """Unknown flag through @gpu_buffer raises ValueError from validator,
        not UserWarning from resolver. The validator catches unknown names
        before _resolve_wgpu_usage_flags is called."""
        with pytest.raises(ValueError, match="invalid usage flag"):

            @gpu_buffer(usage={"quantum"})
            class Bad:
                data: float

    def test_decorator_mixed_known_bad_raises_value_error(self):
        """Mixed known+unknown flags raise ValueError from validator."""
        with pytest.raises(ValueError, match="invalid usage flag"):

            @gpu_buffer(usage={"storage", "nope"})
            class Mixed:
                data: float


# =============================================================================
# ALL SIX METHODS: Decorator-level smoke tests
# =============================================================================


class TestAllSixDecoratorsWorkWithTags:
    """Smoke test: all 6 after-step functions work normally via decorator."""

    def test_gpu_buffer_works(self):
        @gpu_buffer(usage={"storage"})
        class Buf:
            x: float

        assert Buf._gpu_buffer is True
        # FIX#2: storage alone does NOT auto-add COPY_DST
        assert not (Buf._gpu_wgpu_usage & 0x0008)
        assert Buf._gpu_wgpu_usage == 0x0080

    def test_gpu_kernel_works(self):
        from trinity.decorators.gpu import gpu_kernel

        @gpu_kernel()
        class K:
            pass

        assert K._gpu_kernel is True
        assert K._workgroup_size == (64, 1, 1)

    def test_bind_group_works(self):
        from trinity.decorators.gpu import bind_group

        @bind_group(index=2)
        class BG:
            pass

        assert BG._bind_group is True
        assert BG._bind_group_index == 2

    def test_dispatch_works(self):
        from trinity.decorators.gpu import dispatch

        @dispatch()
        def fn():
            pass

        assert fn._dispatch is True
        assert fn._dispatch_indirect is False

    def test_shader_works(self):
        from trinity.decorators.gpu import shader

        @shader(stage="compute")
        class S:
            pass

        assert S._shader is True
        assert S._shader_stage == "compute"

    def test_render_pass_works(self):
        from trinity.decorators.gpu import render_pass

        @render_pass()
        class RP:
            pass

        assert RP._render_pass is True
        assert RP._render_pass_colors == 1
