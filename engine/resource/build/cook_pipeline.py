"""Cook pipeline — platform-specific asset cooking."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TargetPlatform(Enum):
    """Supported target platforms."""

    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    ANDROID = "android"
    IOS = "ios"
    WEB = "web"


class CompressionType(Enum):
    """Supported compression types."""

    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    DEFLATE = "deflate"


@dataclass(slots=True)
class CookSettings:
    """Settings for cooking an asset."""

    target_platform: TargetPlatform
    compression: CompressionType = CompressionType.NONE
    strip_debug: bool = False


@dataclass(slots=True)
class CookResult:
    """Result of a cook operation."""

    success: bool
    output_data: Optional[bytes]
    original_size: int
    cooked_size: int
    errors: list[str] = field(default_factory=list)


class Cooker(ABC):
    """Abstract base for asset cookers."""

    __slots__ = ()

    @abstractmethod
    def cook(self, data: Any, settings: CookSettings) -> CookResult:
        """Cook *data* according to *settings*."""


class CookManager:
    """Manages cookers registered per asset type."""

    __slots__ = ("_cookers",)

    def __init__(self) -> None:
        self._cookers: dict[str, Cooker] = {}

    def register(self, asset_type: str, cooker: Cooker) -> None:
        """Register a cooker for an asset type."""
        self._cookers[asset_type] = cooker

    def get_cooker(self, asset_type: str) -> Optional[Cooker]:
        """Return the cooker for *asset_type*, or None."""
        return self._cookers.get(asset_type)

    def cook(self, asset_type: str, data: Any, settings: CookSettings) -> CookResult:
        """Cook an asset of the given type."""
        cooker = self._cookers.get(asset_type)
        if cooker is None:
            return CookResult(
                success=False,
                output_data=None,
                original_size=0,
                cooked_size=0,
                errors=[f"No cooker registered for asset type '{asset_type}'"],
            )
        try:
            return cooker.cook(data, settings)
        except Exception as exc:  # noqa: BLE001
            return CookResult(
                success=False,
                output_data=None,
                original_size=0,
                cooked_size=0,
                errors=[f"Cook failed: {exc}"],
            )
