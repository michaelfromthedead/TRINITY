"""Integration tests for platform services."""

from pathlib import Path

import pytest

from engine.platform.services import (
    ClipboardService,
    FileDialogResult,
    FileDialogService,
    LinuxClipboardService,
    LinuxFileDialogService,
    LinuxNotificationService,
    LinuxServiceProvider,
    NotificationResult,
    NotificationService,
    NullClipboardService,
    NullFileDialogService,
    NullNotificationService,
    NullServiceProvider,
    ServiceType,
    create_service_provider,
)


class TestNullClipboardService:
    """Tests for NullClipboardService."""

    def test_copy_returns_false(self) -> None:
        """Copy always returns False."""
        svc = NullClipboardService()
        assert svc.copy("text") is False

    def test_paste_returns_none(self) -> None:
        """Paste always returns None."""
        svc = NullClipboardService()
        assert svc.paste() is None

    def test_clear_returns_false(self) -> None:
        """Clear always returns False."""
        svc = NullClipboardService()
        assert svc.clear() is False

    def test_implements_protocol(self) -> None:
        """NullClipboardService implements ClipboardService protocol."""
        svc = NullClipboardService()
        assert isinstance(svc, ClipboardService)


class TestNullFileDialogService:
    """Tests for NullFileDialogService."""

    def test_open_file_returns_cancelled(self) -> None:
        """Open file always returns cancelled result."""
        svc = NullFileDialogService()
        result = svc.open_file()
        assert result.cancelled is True
        assert result.paths == []

    def test_save_file_returns_cancelled(self) -> None:
        """Save file always returns cancelled result."""
        svc = NullFileDialogService()
        result = svc.save_file()
        assert result.cancelled is True
        assert result.paths == []

    def test_select_folder_returns_cancelled(self) -> None:
        """Select folder always returns cancelled result."""
        svc = NullFileDialogService()
        result = svc.select_folder()
        assert result.cancelled is True
        assert result.paths == []

    def test_implements_protocol(self) -> None:
        """NullFileDialogService implements FileDialogService protocol."""
        svc = NullFileDialogService()
        assert isinstance(svc, FileDialogService)


class TestNullNotificationService:
    """Tests for NullNotificationService."""

    def test_show_returns_not_shown(self) -> None:
        """Show always returns not shown."""
        svc = NullNotificationService()
        result = svc.show("Title", "Message")
        assert result.shown is False

    def test_implements_protocol(self) -> None:
        """NullNotificationService implements NotificationService protocol."""
        svc = NullNotificationService()
        assert isinstance(svc, NotificationService)


class TestNullServiceProvider:
    """Tests for NullServiceProvider."""

    def test_get_clipboard_returns_null_service(self) -> None:
        """get_clipboard returns NullClipboardService."""
        provider = NullServiceProvider()
        clipboard = provider.get_clipboard()
        assert isinstance(clipboard, NullClipboardService)

    def test_get_file_dialog_returns_null_service(self) -> None:
        """get_file_dialog returns NullFileDialogService."""
        provider = NullServiceProvider()
        dialog = provider.get_file_dialog()
        assert isinstance(dialog, NullFileDialogService)

    def test_get_notification_returns_null_service(self) -> None:
        """get_notification returns NullNotificationService."""
        provider = NullServiceProvider()
        notification = provider.get_notification()
        assert isinstance(notification, NullNotificationService)

    def test_is_available_always_false(self) -> None:
        """is_available always returns False for null provider."""
        provider = NullServiceProvider()
        assert provider.is_available(ServiceType.CLIPBOARD) is False
        assert provider.is_available(ServiceType.FILE_DIALOG) is False
        assert provider.is_available(ServiceType.NOTIFICATION) is False


class TestLinuxServiceProvider:
    """Tests for LinuxServiceProvider."""

    def test_get_clipboard_returns_linux_service(self) -> None:
        """get_clipboard returns LinuxClipboardService."""
        provider = LinuxServiceProvider()
        clipboard = provider.get_clipboard()
        assert isinstance(clipboard, LinuxClipboardService)

    def test_get_file_dialog_returns_linux_service(self) -> None:
        """get_file_dialog returns LinuxFileDialogService."""
        provider = LinuxServiceProvider()
        dialog = provider.get_file_dialog()
        assert isinstance(dialog, LinuxFileDialogService)

    def test_get_notification_returns_linux_service(self) -> None:
        """get_notification returns LinuxNotificationService."""
        provider = LinuxServiceProvider()
        notification = provider.get_notification()
        assert isinstance(notification, LinuxNotificationService)


class TestFileDialogResult:
    """Tests for FileDialogResult dataclass."""

    def test_default_not_cancelled(self) -> None:
        """Default cancelled is False."""
        result = FileDialogResult(paths=[Path("/tmp/test.txt")])
        assert result.cancelled is False

    def test_paths_stored(self) -> None:
        """Paths are correctly stored."""
        paths = [Path("/a.txt"), Path("/b.txt")]
        result = FileDialogResult(paths=paths)
        assert result.paths == paths


class TestNotificationResult:
    """Tests for NotificationResult dataclass."""

    def test_shown_flag(self) -> None:
        """Shown flag is correctly stored."""
        result = NotificationResult(shown=True, notification_id="123")
        assert result.shown is True
        assert result.notification_id == "123"


class TestCreateServiceProvider:
    """Tests for create_service_provider factory."""

    def test_returns_service_provider(self) -> None:
        """Factory returns a ServiceProvider instance."""
        from engine.platform.services import ServiceProvider
        provider = create_service_provider()
        assert isinstance(provider, ServiceProvider)

    def test_linux_returns_linux_provider(self) -> None:
        """On Linux, returns LinuxServiceProvider."""
        import platform
        if platform.system() == "Linux":
            provider = create_service_provider()
            assert isinstance(provider, LinuxServiceProvider)


class TestServiceType:
    """Tests for ServiceType enum."""

    def test_all_types_defined(self) -> None:
        """All expected service types are defined."""
        assert ServiceType.CLIPBOARD
        assert ServiceType.FILE_DIALOG
        assert ServiceType.NOTIFICATION
