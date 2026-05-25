"""Platform services module."""

from .platform_detect import (
    PlatformType,
    PlatformInfo,
    detect,
)

from .app_lifecycle import (
    AppState,
    AppLifecycle,
)

from .permissions import (
    Permission,
    PermissionStatus,
    request,
    check,
)

from .service_provider import (
    ServiceType,
    FileDialogResult,
    NotificationResult,
    ClipboardService,
    FileDialogService,
    NotificationService,
    ServiceProvider,
    NullClipboardService,
    NullFileDialogService,
    NullNotificationService,
    NullServiceProvider,
    LinuxClipboardService,
    LinuxFileDialogService,
    LinuxNotificationService,
    LinuxServiceProvider,
    create_service_provider,
)

__all__ = [
    # Platform detection
    'PlatformType',
    'PlatformInfo',
    'detect',
    # App lifecycle
    'AppState',
    'AppLifecycle',
    # Permissions
    'Permission',
    'PermissionStatus',
    'request',
    'check',
    # Service providers
    'ServiceType',
    'FileDialogResult',
    'NotificationResult',
    'ClipboardService',
    'FileDialogService',
    'NotificationService',
    'ServiceProvider',
    'NullClipboardService',
    'NullFileDialogService',
    'NullNotificationService',
    'NullServiceProvider',
    'LinuxClipboardService',
    'LinuxFileDialogService',
    'LinuxNotificationService',
    'LinuxServiceProvider',
    'create_service_provider',
]
