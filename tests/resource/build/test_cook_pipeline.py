"""Tests for the cook pipeline."""
from typing import Any

from engine.resource.build.cook_pipeline import (
    CompressionType,
    CookManager,
    CookResult,
    CookSettings,
    Cooker,
    TargetPlatform,
)


class SimpleCooker(Cooker):
    def cook(self, data: Any, settings: CookSettings) -> CookResult:
        raw = data if isinstance(data, bytes) else str(data).encode()
        cooked = raw  # passthrough
        if settings.strip_debug:
            cooked = cooked.replace(b"DEBUG", b"")
        return CookResult(
            success=True,
            output_data=cooked,
            original_size=len(raw),
            cooked_size=len(cooked),
        )


class BrokenCooker(Cooker):
    def cook(self, data: Any, settings: CookSettings) -> CookResult:
        raise IOError("write error")


class TestCookSettings:
    def test_defaults(self) -> None:
        s = CookSettings(target_platform=TargetPlatform.WINDOWS)
        assert s.compression == CompressionType.NONE
        assert s.strip_debug is False

    def test_all_platforms(self) -> None:
        for name in ("WINDOWS", "LINUX", "MACOS", "ANDROID", "IOS", "WEB"):
            assert hasattr(TargetPlatform, name), f"TargetPlatform.{name} missing"


class TestCookManager:
    def test_register_and_cook(self) -> None:
        mgr = CookManager()
        mgr.register("texture", SimpleCooker())
        settings = CookSettings(target_platform=TargetPlatform.LINUX)
        result = mgr.cook("texture", b"pixel_data", settings)
        assert result.success is True
        assert result.output_data == b"pixel_data"
        assert result.original_size == 10

    def test_no_cooker_registered(self) -> None:
        mgr = CookManager()
        settings = CookSettings(target_platform=TargetPlatform.WEB)
        result = mgr.cook("unknown", b"data", settings)
        assert result.success is False
        assert "No cooker registered" in result.errors[0]

    def test_strip_debug(self) -> None:
        mgr = CookManager()
        mgr.register("shader", SimpleCooker())
        settings = CookSettings(target_platform=TargetPlatform.WINDOWS, strip_debug=True)
        result = mgr.cook("shader", b"code_DEBUG_info", settings)
        assert result.success is True
        assert b"DEBUG" not in result.output_data
        assert result.cooked_size < result.original_size

    def test_cooker_exception_handled(self) -> None:
        mgr = CookManager()
        mgr.register("bad", BrokenCooker())
        settings = CookSettings(target_platform=TargetPlatform.ANDROID)
        result = mgr.cook("bad", b"data", settings)
        assert result.success is False
        assert "Cook failed" in result.errors[0]

    def test_size_tracking(self) -> None:
        mgr = CookManager()
        mgr.register("mesh", SimpleCooker())
        settings = CookSettings(target_platform=TargetPlatform.MACOS)
        result = mgr.cook("mesh", b"vertices", settings)
        assert result.original_size == 8
        assert result.cooked_size == 8

    def test_compression_enum_values(self) -> None:
        for name in ("NONE", "LZ4", "ZSTD", "DEFLATE"):
            assert hasattr(CompressionType, name), f"CompressionType.{name} missing"
        # Default cook settings should use no compression
        s = CookSettings(target_platform=TargetPlatform.WINDOWS)
        assert s.compression == CompressionType.NONE
