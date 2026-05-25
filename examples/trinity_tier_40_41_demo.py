"""
Demonstration of Trinity Pattern Tier 40 (ERROR_HANDLING) and Tier 41 (BUILD_DEPLOY) decorators.

This file shows practical usage examples of all 8 decorators across both tiers.
"""

from trinity.decorators.error_handling import (
    bug_report,
    crash_safe,
    error_boundary,
    recoverable,
)
from trinity.decorators.build_deploy import (
    asset_bundle,
    build_only,
    feature_flag,
    strip_in_release,
)
from trinity.decorators.registry import inspect_decorated


# =============================================================================
# ERROR_HANDLING Examples (Tier 40)
# =============================================================================


@crash_safe(recovery="retry")
@error_boundary(scope="system")
class NetworkManager:
    """Manages network connections with crash recovery and error isolation."""

    def __init__(self):
        self.connections = []

    def connect(self, url: str):
        """Connect to a server with automatic retry on failure."""
        pass


@recoverable(checkpoint=True)
class GameState:
    """Game state that can be recovered after a crash."""

    def __init__(self):
        self.player_position = (0, 0, 0)
        self.score = 0
        self.inventory = []


@bug_report(include={"screenshot", "logs", "save", "replay"})
@error_boundary(scope="entity")
class PlayerController:
    """Player controller that auto-reports bugs with full context."""

    def __init__(self):
        self.velocity = (0, 0, 0)
        self.health = 100


@crash_safe(recovery="fallback")
@recoverable(checkpoint=False)
class AIController:
    """AI controller with fallback behavior on crash."""

    def __init__(self):
        self.state = "idle"
        self.target = None


# =============================================================================
# BUILD_DEPLOY Examples (Tier 41)
# =============================================================================


@build_only(configurations={"debug", "development"})
@strip_in_release
class DebugConsole:
    """Debug console only available in debug builds."""

    def __init__(self):
        self.commands = {}
        self.history = []

    def execute(self, command: str):
        """Execute a debug command."""
        pass


@feature_flag(id="experimental_renderer", default=False)
class ExperimentalRenderer:
    """Experimental renderer controlled by feature flag."""

    def __init__(self):
        self.pipelines = []

    def render(self, scene):
        """Render scene with experimental techniques."""
        pass


@asset_bundle(name="ui_assets", platforms={"windows", "linux", "macos"})
class UIAssetPack:
    """UI assets bundled for desktop platforms."""

    def __init__(self):
        self.textures = []
        self.fonts = []
        self.layouts = []


@build_only(configurations={"debug", "profile"})
@feature_flag(id="advanced_profiling", default=True)
@strip_in_release
class AdvancedProfiler:
    """Advanced profiler for debug and profile builds only."""

    def __init__(self):
        self.samples = []
        self.markers = []

    def start_capture(self):
        """Start profiling capture."""
        pass


# =============================================================================
# Combined Examples (Both Tiers)
# =============================================================================


@crash_safe(recovery="skip")
@bug_report(include={"logs", "save"})
@build_only(configurations={"debug", "test"})
@feature_flag(id="beta_features", default=False)
class BetaFeatureManager:
    """
    Beta features manager that:
    - Skips on crash (doesn't break the game)
    - Reports bugs with logs and save data
    - Only exists in debug/test builds
    - Controlled by feature flag
    """

    def __init__(self):
        self.enabled_features = set()


@recoverable(checkpoint=True)
@error_boundary(scope="global")
@asset_bundle(name="save_data", platforms=None)
class SaveSystem:
    """
    Save system that:
    - Can be recovered from crashes
    - Has global error boundary
    - Assets bundled for all platforms
    """

    def __init__(self):
        self.current_save = None
        self.auto_save_enabled = True


# =============================================================================
# Introspection Demo
# =============================================================================


def demo_introspection():
    """Demonstrate decorator introspection."""
    print("=" * 60)
    print("TIER 40 & 41 DECORATOR INTROSPECTION")
    print("=" * 60)

    # NetworkManager
    info = inspect_decorated(NetworkManager)
    print(f"\nNetworkManager:")
    print(f"  Decorators: {info.decorators}")
    print(f"  Attributes: {list(info.attributes.keys())}")
    print(f"  Crash safe: {NetworkManager._crash_safe}")
    print(f"  Recovery: {NetworkManager._crash_recovery}")
    print(f"  Error scope: {NetworkManager._error_scope}")

    # GameState
    info = inspect_decorated(GameState)
    print(f"\nGameState:")
    print(f"  Decorators: {info.decorators}")
    print(f"  Recoverable: {GameState._recoverable}")
    print(f"  Checkpoint: {GameState._recoverable_checkpoint}")

    # DebugConsole
    info = inspect_decorated(DebugConsole)
    print(f"\nDebugConsole:")
    print(f"  Decorators: {info.decorators}")
    print(f"  Build only: {DebugConsole._build_only}")
    print(f"  Configurations: {DebugConsole._build_configurations}")
    print(f"  Strip in release: {DebugConsole._strip_in_release}")

    # ExperimentalRenderer
    info = inspect_decorated(ExperimentalRenderer)
    print(f"\nExperimentalRenderer:")
    print(f"  Decorators: {info.decorators}")
    print(f"  Feature flag: {ExperimentalRenderer._feature_flag}")
    print(f"  Flag ID: {ExperimentalRenderer._feature_flag_id}")
    print(f"  Default: {ExperimentalRenderer._feature_flag_default}")

    # BetaFeatureManager (multiple decorators)
    info = inspect_decorated(BetaFeatureManager)
    print(f"\nBetaFeatureManager (Combined):")
    print(f"  Decorators: {info.decorators}")
    print(f"  Crash safe: {BetaFeatureManager._crash_safe}")
    print(f"  Bug report: {BetaFeatureManager._bug_report}")
    print(f"  Build only: {BetaFeatureManager._build_only}")
    print(f"  Feature flag: {BetaFeatureManager._feature_flag}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo_introspection()
