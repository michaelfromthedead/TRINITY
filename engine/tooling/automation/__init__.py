"""
Automation subsystem for the AI Game Engine.

Provides comprehensive automation infrastructure including:
- Automation framework with @automation_test decorator
- Commandlets for build, cook, test, and validation
- Python API for scripting automation
- CI/CD integration for Jenkins, GitHub Actions, TeamCity
- Build agent management for distributed builds
- Automated gameplay testing with bots
"""

from .automation_framework import (
    automation_test,
    AutomationTest,
    AutomationTestResult,
    AutomationTestRunner,
    AutomationTestSuite,
    automation_step,
    requires,
    timeout as automation_timeout,
)

from .commandlets import (
    Commandlet,
    CommandletResult,
    CommandletRunner,
    CookCommandlet,
    BuildCommandlet,
    TestCommandlet,
    ValidateCommandlet,
    CleanCommandlet,
    PackageCommandlet,
    run_commandlet,
)

from .python_api import (
    AutomationAPI,
    BuildAPI,
    TestAPI,
    AssetAPI,
    DeployAPI,
    ScriptContext,
    run_script,
    execute_command,
)

from .ci_integration import (
    CIProvider,
    JenkinsIntegration,
    GitHubActionsIntegration,
    TeamCityIntegration,
    CIBuildStatus,
    CITestResult,
    create_ci_provider,
    publish_results,
)

from .build_agents import (
    BuildAgent,
    BuildAgentPool,
    BuildAgentManager,
    BuildJob,
    BuildJobResult,
    AgentCapability,
    dispatch_build,
    get_available_agents,
)

from .automated_testing import (
    GameBot,
    BotBehavior,
    BotAction,
    BotController,
    PlaytestSession,
    PlaytestRecorder,
    PlaytestReporter,
    create_bot,
    run_playtest,
)

__all__ = [
    # Automation framework
    "automation_test",
    "AutomationTest",
    "AutomationTestResult",
    "AutomationTestRunner",
    "AutomationTestSuite",
    "automation_step",
    "requires",
    "automation_timeout",
    # Commandlets
    "Commandlet",
    "CommandletResult",
    "CommandletRunner",
    "CookCommandlet",
    "BuildCommandlet",
    "TestCommandlet",
    "ValidateCommandlet",
    "CleanCommandlet",
    "PackageCommandlet",
    "run_commandlet",
    # Python API
    "AutomationAPI",
    "BuildAPI",
    "TestAPI",
    "AssetAPI",
    "DeployAPI",
    "ScriptContext",
    "run_script",
    "execute_command",
    # CI Integration
    "CIProvider",
    "JenkinsIntegration",
    "GitHubActionsIntegration",
    "TeamCityIntegration",
    "CIBuildStatus",
    "CITestResult",
    "create_ci_provider",
    "publish_results",
    # Build agents
    "BuildAgent",
    "BuildAgentPool",
    "BuildAgentManager",
    "BuildJob",
    "BuildJobResult",
    "AgentCapability",
    "dispatch_build",
    "get_available_agents",
    # Automated testing
    "GameBot",
    "BotBehavior",
    "BotAction",
    "BotController",
    "PlaytestSession",
    "PlaytestRecorder",
    "PlaytestReporter",
    "create_bot",
    "run_playtest",
]
