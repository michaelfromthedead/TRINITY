"""Platform service provider abstraction and implementations."""
from __future__ import annotations

import subprocess
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Protocol, runtime_checkable


class ServiceType(Enum):
    """Types of platform services."""

    CLIPBOARD = auto()
    FILE_DIALOG = auto()
    NOTIFICATION = auto()


@dataclass(slots=True)
class FileDialogResult:
    """Result from a file dialog."""

    paths: list[Path]
    cancelled: bool = False


@dataclass(slots=True)
class NotificationResult:
    """Result from showing a notification."""

    shown: bool
    notification_id: str | None = None


@runtime_checkable
class ClipboardService(Protocol):
    """Protocol for clipboard operations."""

    def copy(self, text: str) -> bool: ...
    def paste(self) -> str | None: ...
    def clear(self) -> bool: ...


@runtime_checkable
class FileDialogService(Protocol):
    """Protocol for file dialog operations."""

    def open_file(
        self,
        title: str = "Open File",
        filters: list[tuple[str, str]] | None = None,
        initial_dir: Path | None = None,
        multiple: bool = False,
    ) -> FileDialogResult: ...

    def save_file(
        self,
        title: str = "Save File",
        filters: list[tuple[str, str]] | None = None,
        initial_dir: Path | None = None,
        default_name: str = "",
    ) -> FileDialogResult: ...

    def select_folder(
        self,
        title: str = "Select Folder",
        initial_dir: Path | None = None,
    ) -> FileDialogResult: ...


@runtime_checkable
class NotificationService(Protocol):
    """Protocol for notification operations."""

    def show(
        self,
        title: str,
        message: str,
        icon: str | None = None,
        timeout_ms: int = 5000,
    ) -> NotificationResult: ...


class ServiceProvider(ABC):
    """Abstract base for platform service providers."""

    @abstractmethod
    def get_clipboard(self) -> ClipboardService: ...

    @abstractmethod
    def get_file_dialog(self) -> FileDialogService: ...

    @abstractmethod
    def get_notification(self) -> NotificationService: ...

    @abstractmethod
    def is_available(self, service_type: ServiceType) -> bool: ...


class NullClipboardService:
    """No-op clipboard service."""

    def copy(self, text: str) -> bool:
        return False

    def paste(self) -> str | None:
        return None

    def clear(self) -> bool:
        return False


class NullFileDialogService:
    """No-op file dialog service."""

    def open_file(
        self,
        title: str = "Open File",
        filters: list[tuple[str, str]] | None = None,
        initial_dir: Path | None = None,
        multiple: bool = False,
    ) -> FileDialogResult:
        return FileDialogResult(paths=[], cancelled=True)

    def save_file(
        self,
        title: str = "Save File",
        filters: list[tuple[str, str]] | None = None,
        initial_dir: Path | None = None,
        default_name: str = "",
    ) -> FileDialogResult:
        return FileDialogResult(paths=[], cancelled=True)

    def select_folder(
        self,
        title: str = "Select Folder",
        initial_dir: Path | None = None,
    ) -> FileDialogResult:
        return FileDialogResult(paths=[], cancelled=True)


class NullNotificationService:
    """No-op notification service."""

    def show(
        self,
        title: str,
        message: str,
        icon: str | None = None,
        timeout_ms: int = 5000,
    ) -> NotificationResult:
        return NotificationResult(shown=False)


class NullServiceProvider(ServiceProvider):
    """Service provider that returns no-op implementations for all services."""

    def __init__(self) -> None:
        self._clipboard = NullClipboardService()
        self._file_dialog = NullFileDialogService()
        self._notification = NullNotificationService()

    def get_clipboard(self) -> ClipboardService:
        return self._clipboard

    def get_file_dialog(self) -> FileDialogService:
        return self._file_dialog

    def get_notification(self) -> NotificationService:
        return self._notification

    def is_available(self, service_type: ServiceType) -> bool:
        return False


class LinuxClipboardService:
    """Linux clipboard using xclip or xsel."""

    def __init__(self) -> None:
        self._tool = self._find_tool()

    def _find_tool(self) -> str | None:
        for tool in ("xclip", "xsel", "wl-copy"):
            if shutil.which(tool):
                return tool
        return None

    def copy(self, text: str) -> bool:
        if not self._tool:
            return False
        try:
            if self._tool == "xclip":
                subprocess.run(
                    ["xclip", "-selection", "clipboard"],
                    input=text.encode(),
                    check=True,
                    capture_output=True,
                )
            elif self._tool == "xsel":
                subprocess.run(
                    ["xsel", "--clipboard", "--input"],
                    input=text.encode(),
                    check=True,
                    capture_output=True,
                )
            elif self._tool == "wl-copy":
                subprocess.run(
                    ["wl-copy"],
                    input=text.encode(),
                    check=True,
                    capture_output=True,
                )
            return True
        except subprocess.CalledProcessError:
            return False

    def paste(self) -> str | None:
        if not self._tool:
            return None
        try:
            if self._tool == "xclip":
                result = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"],
                    capture_output=True,
                    check=True,
                )
            elif self._tool == "xsel":
                result = subprocess.run(
                    ["xsel", "--clipboard", "--output"],
                    capture_output=True,
                    check=True,
                )
            elif self._tool == "wl-copy":
                result = subprocess.run(
                    ["wl-paste"],
                    capture_output=True,
                    check=True,
                )
            else:
                return None
            return result.stdout.decode()
        except subprocess.CalledProcessError:
            return None

    def clear(self) -> bool:
        return self.copy("")


class LinuxFileDialogService:
    """Linux file dialogs using zenity or kdialog."""

    def __init__(self) -> None:
        self._tool = self._find_tool()

    def _find_tool(self) -> str | None:
        for tool in ("zenity", "kdialog"):
            if shutil.which(tool):
                return tool
        return None

    def _build_filter_arg(self, filters: list[tuple[str, str]] | None) -> list[str]:
        if not filters or self._tool != "zenity":
            return []
        args = []
        for name, pattern in filters:
            args.extend(["--file-filter", f"{name}|{pattern}"])
        return args

    def open_file(
        self,
        title: str = "Open File",
        filters: list[tuple[str, str]] | None = None,
        initial_dir: Path | None = None,
        multiple: bool = False,
    ) -> FileDialogResult:
        if not self._tool:
            return FileDialogResult(paths=[], cancelled=True)

        try:
            if self._tool == "zenity":
                cmd = ["zenity", "--file-selection", "--title", title]
                if multiple:
                    cmd.append("--multiple")
                cmd.extend(self._build_filter_arg(filters))
            else:  # kdialog
                cmd = ["kdialog", "--getopenfilename"]
                if initial_dir:
                    cmd.append(str(initial_dir))
                cmd.extend(["--title", title])

            result = subprocess.run(cmd, capture_output=True, check=True)
            paths_str = result.stdout.decode().strip()
            if paths_str:
                paths = [Path(p) for p in paths_str.split("|" if multiple else "\n") if p]
                return FileDialogResult(paths=paths)
            return FileDialogResult(paths=[], cancelled=True)
        except subprocess.CalledProcessError:
            return FileDialogResult(paths=[], cancelled=True)

    def save_file(
        self,
        title: str = "Save File",
        filters: list[tuple[str, str]] | None = None,
        initial_dir: Path | None = None,
        default_name: str = "",
    ) -> FileDialogResult:
        if not self._tool:
            return FileDialogResult(paths=[], cancelled=True)

        try:
            if self._tool == "zenity":
                cmd = ["zenity", "--file-selection", "--save", "--title", title]
                if default_name:
                    cmd.extend(["--filename", default_name])
                cmd.extend(self._build_filter_arg(filters))
            else:  # kdialog
                cmd = ["kdialog", "--getsavefilename"]
                if initial_dir:
                    cmd.append(str(initial_dir / default_name) if default_name else str(initial_dir))
                cmd.extend(["--title", title])

            result = subprocess.run(cmd, capture_output=True, check=True)
            path_str = result.stdout.decode().strip()
            if path_str:
                return FileDialogResult(paths=[Path(path_str)])
            return FileDialogResult(paths=[], cancelled=True)
        except subprocess.CalledProcessError:
            return FileDialogResult(paths=[], cancelled=True)

    def select_folder(
        self,
        title: str = "Select Folder",
        initial_dir: Path | None = None,
    ) -> FileDialogResult:
        if not self._tool:
            return FileDialogResult(paths=[], cancelled=True)

        try:
            if self._tool == "zenity":
                cmd = ["zenity", "--file-selection", "--directory", "--title", title]
            else:  # kdialog
                cmd = ["kdialog", "--getexistingdirectory"]
                if initial_dir:
                    cmd.append(str(initial_dir))
                cmd.extend(["--title", title])

            result = subprocess.run(cmd, capture_output=True, check=True)
            path_str = result.stdout.decode().strip()
            if path_str:
                return FileDialogResult(paths=[Path(path_str)])
            return FileDialogResult(paths=[], cancelled=True)
        except subprocess.CalledProcessError:
            return FileDialogResult(paths=[], cancelled=True)


class LinuxNotificationService:
    """Linux notifications using notify-send."""

    def __init__(self) -> None:
        self._available = shutil.which("notify-send") is not None

    def show(
        self,
        title: str,
        message: str,
        icon: str | None = None,
        timeout_ms: int = 5000,
    ) -> NotificationResult:
        if not self._available:
            return NotificationResult(shown=False)

        try:
            cmd = ["notify-send", title, message, "-t", str(timeout_ms)]
            if icon:
                cmd.extend(["-i", icon])
            subprocess.run(cmd, check=True, capture_output=True)
            return NotificationResult(shown=True)
        except subprocess.CalledProcessError:
            return NotificationResult(shown=False)


class LinuxServiceProvider(ServiceProvider):
    """Linux-specific service provider."""

    def __init__(self) -> None:
        self._clipboard = LinuxClipboardService()
        self._file_dialog = LinuxFileDialogService()
        self._notification = LinuxNotificationService()

    def get_clipboard(self) -> ClipboardService:
        return self._clipboard

    def get_file_dialog(self) -> FileDialogService:
        return self._file_dialog

    def get_notification(self) -> NotificationService:
        return self._notification

    def is_available(self, service_type: ServiceType) -> bool:
        if service_type == ServiceType.CLIPBOARD:
            return self._clipboard._tool is not None
        if service_type == ServiceType.FILE_DIALOG:
            return self._file_dialog._tool is not None
        if service_type == ServiceType.NOTIFICATION:
            return self._notification._available
        return False


class WindowsServiceProvider(ServiceProvider):
    """Windows service provider stub."""

    def __init__(self) -> None:
        self._null = NullServiceProvider()

    def get_clipboard(self) -> ClipboardService:
        return self._null.get_clipboard()

    def get_file_dialog(self) -> FileDialogService:
        return self._null.get_file_dialog()

    def get_notification(self) -> NotificationService:
        return self._null.get_notification()

    def is_available(self, service_type: ServiceType) -> bool:
        return False


class MacOSServiceProvider(ServiceProvider):
    """macOS service provider stub."""

    def __init__(self) -> None:
        self._null = NullServiceProvider()

    def get_clipboard(self) -> ClipboardService:
        return self._null.get_clipboard()

    def get_file_dialog(self) -> FileDialogService:
        return self._null.get_file_dialog()

    def get_notification(self) -> NotificationService:
        return self._null.get_notification()

    def is_available(self, service_type: ServiceType) -> bool:
        return False


def create_service_provider() -> ServiceProvider:
    """Auto-detect platform and create appropriate service provider."""
    from .platform_detect import detect, PlatformType

    info = detect()

    if info.type == PlatformType.LINUX:
        return LinuxServiceProvider()
    if info.type == PlatformType.WINDOWS:
        return WindowsServiceProvider()
    if info.type == PlatformType.MACOS:
        return MacOSServiceProvider()

    return NullServiceProvider()
