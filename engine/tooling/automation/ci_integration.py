"""
CI/CD integration for Jenkins, GitHub Actions, and TeamCity.

Provides integrations with popular CI/CD systems for automated
builds, tests, and deployments.
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Type, Union


class CIBuildStatus(Enum):
    """Status of a CI build."""

    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    CANCELLED = auto()
    UNSTABLE = auto()


@dataclass
class CITestResult:
    """Test result for CI systems."""

    name: str
    passed: bool
    duration: float = 0.0
    message: str = ""
    output: str = ""
    suite: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "passed": self.passed,
            "duration": self.duration,
            "message": self.message,
            "suite": self.suite,
        }


@dataclass
class CIBuildResult:
    """Build result for CI systems."""

    status: CIBuildStatus
    build_number: int = 0
    duration: float = 0.0
    url: str = ""
    commit: str = ""
    branch: str = ""
    artifacts: List[str] = field(default_factory=list)
    test_results: List[CITestResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if build succeeded."""
        return self.status == CIBuildStatus.SUCCESS

    @property
    def tests_passed(self) -> int:
        """Number of tests passed."""
        return sum(1 for t in self.test_results if t.passed)

    @property
    def tests_failed(self) -> int:
        """Number of tests failed."""
        return sum(1 for t in self.test_results if not t.passed)


class CIProvider(ABC):
    """
    Base class for CI/CD provider integrations.

    Provides a common interface for interacting with different
    CI/CD systems.
    """

    name: str = "ci"

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        project: Optional[str] = None,
    ):
        self.url = url or os.environ.get(f"{self.name.upper()}_URL", "")
        self.token = token or os.environ.get(f"{self.name.upper()}_TOKEN", "")
        self.project = project or os.environ.get(f"{self.name.upper()}_PROJECT", "")

    @abstractmethod
    def get_build_status(self, build_id: Union[int, str]) -> CIBuildStatus:
        """Get status of a build."""
        pass

    @abstractmethod
    def get_build_info(self, build_id: Union[int, str]) -> CIBuildResult:
        """Get full build information."""
        pass

    @abstractmethod
    def trigger_build(
        self,
        branch: str = "main",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[int, str]:
        """Trigger a new build."""
        pass

    @abstractmethod
    def cancel_build(self, build_id: Union[int, str]) -> bool:
        """Cancel a running build."""
        pass

    @abstractmethod
    def publish_test_results(
        self,
        results: List[CITestResult],
        build_id: Optional[Union[int, str]] = None,
    ) -> bool:
        """Publish test results."""
        pass

    @abstractmethod
    def upload_artifact(
        self,
        path: str,
        name: str,
        build_id: Optional[Union[int, str]] = None,
    ) -> str:
        """Upload a build artifact."""
        pass

    def set_build_status(
        self,
        status: CIBuildStatus,
        description: str = "",
        context: str = "build",
    ) -> bool:
        """Set build status (for commit status updates)."""
        # Override in subclasses that support commit status
        return False

    def post_comment(
        self,
        message: str,
        pr_number: Optional[int] = None,
    ) -> bool:
        """Post a comment on a PR/MR."""
        # Override in subclasses that support comments
        return False


class JenkinsIntegration(CIProvider):
    """
    Jenkins CI integration.

    Provides integration with Jenkins CI server.
    """

    name = "jenkins"

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        project: Optional[str] = None,
        username: Optional[str] = None,
    ):
        super().__init__(url, token, project)
        self.username = username or os.environ.get("JENKINS_USERNAME", "")

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make authenticated request to Jenkins."""
        url = f"{self.url}/{endpoint}"

        request = urllib.request.Request(url, method=method)

        # Add authentication
        if self.username and self.token:
            import base64
            credentials = base64.b64encode(
                f"{self.username}:{self.token}".encode()
            ).decode()
            request.add_header("Authorization", f"Basic {credentials}")

        if data:
            request.add_header("Content-Type", "application/json")
            request.data = json.dumps(data).encode()

        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode())
        except Exception:
            return {}

    def get_build_status(self, build_id: Union[int, str]) -> CIBuildStatus:
        """Get status of a Jenkins build."""
        info = self._make_request(
            f"job/{self.project}/{build_id}/api/json"
        )

        result = info.get("result")
        if result is None:
            return CIBuildStatus.RUNNING
        elif result == "SUCCESS":
            return CIBuildStatus.SUCCESS
        elif result == "FAILURE":
            return CIBuildStatus.FAILED
        elif result == "ABORTED":
            return CIBuildStatus.CANCELLED
        elif result == "UNSTABLE":
            return CIBuildStatus.UNSTABLE
        else:
            return CIBuildStatus.PENDING

    def get_build_info(self, build_id: Union[int, str]) -> CIBuildResult:
        """Get full Jenkins build information."""
        info = self._make_request(
            f"job/{self.project}/{build_id}/api/json"
        )

        return CIBuildResult(
            status=self.get_build_status(build_id),
            build_number=info.get("number", 0),
            duration=info.get("duration", 0) / 1000,  # Convert to seconds
            url=info.get("url", ""),
            artifacts=[a["fileName"] for a in info.get("artifacts", [])],
        )

    def trigger_build(
        self,
        branch: str = "main",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[int, str]:
        """Trigger a new Jenkins build."""
        endpoint = f"job/{self.project}/build"
        if parameters:
            endpoint = f"job/{self.project}/buildWithParameters"

        self._make_request(endpoint, method="POST", data=parameters)

        # Get the queue item and return build number
        # In practice, you'd poll the queue to get the actual build number
        return 0

    def cancel_build(self, build_id: Union[int, str]) -> bool:
        """Cancel a Jenkins build."""
        try:
            self._make_request(
                f"job/{self.project}/{build_id}/stop",
                method="POST",
            )
            return True
        except Exception:
            return False

    def publish_test_results(
        self,
        results: List[CITestResult],
        build_id: Optional[Union[int, str]] = None,
    ) -> bool:
        """Publish test results to Jenkins."""
        # Jenkins typically reads JUnit XML files
        # This would generate and upload the XML
        return True

    def upload_artifact(
        self,
        path: str,
        name: str,
        build_id: Optional[Union[int, str]] = None,
    ) -> str:
        """Upload artifact to Jenkins."""
        # Artifacts are typically archived during the build
        return f"{self.url}/job/{self.project}/{build_id}/artifact/{name}"


class GitHubActionsIntegration(CIProvider):
    """
    GitHub Actions integration.

    Provides integration with GitHub Actions CI/CD.
    """

    name = "github"

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        project: Optional[str] = None,
    ):
        super().__init__(
            url or "https://api.github.com",
            token or os.environ.get("GITHUB_TOKEN", ""),
            project or os.environ.get("GITHUB_REPOSITORY", ""),
        )

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make authenticated request to GitHub API."""
        url = f"{self.url}/{endpoint}"

        request = urllib.request.Request(url, method=method)
        request.add_header("Accept", "application/vnd.github.v3+json")
        request.add_header("Authorization", f"token {self.token}")

        if data:
            request.add_header("Content-Type", "application/json")
            request.data = json.dumps(data).encode()

        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode())
        except Exception:
            return {}

    def get_build_status(self, build_id: Union[int, str]) -> CIBuildStatus:
        """Get status of a GitHub Actions workflow run."""
        info = self._make_request(
            f"repos/{self.project}/actions/runs/{build_id}"
        )

        status = info.get("status")
        conclusion = info.get("conclusion")

        if status == "queued":
            return CIBuildStatus.PENDING
        elif status == "in_progress":
            return CIBuildStatus.RUNNING
        elif conclusion == "success":
            return CIBuildStatus.SUCCESS
        elif conclusion == "failure":
            return CIBuildStatus.FAILED
        elif conclusion == "cancelled":
            return CIBuildStatus.CANCELLED
        else:
            return CIBuildStatus.PENDING

    def get_build_info(self, build_id: Union[int, str]) -> CIBuildResult:
        """Get full GitHub Actions build information."""
        info = self._make_request(
            f"repos/{self.project}/actions/runs/{build_id}"
        )

        # Parse timing
        created = info.get("created_at", "")
        updated = info.get("updated_at", "")
        duration = 0.0
        if created and updated:
            try:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                duration = (updated_dt - created_dt).total_seconds()
            except Exception:
                pass

        return CIBuildResult(
            status=self.get_build_status(build_id),
            build_number=info.get("run_number", 0),
            duration=duration,
            url=info.get("html_url", ""),
            commit=info.get("head_sha", ""),
            branch=info.get("head_branch", ""),
        )

    def trigger_build(
        self,
        branch: str = "main",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[int, str]:
        """Trigger a GitHub Actions workflow."""
        # Need workflow ID or filename
        workflow = parameters.get("workflow", "ci.yml") if parameters else "ci.yml"

        data = {
            "ref": branch,
            "inputs": parameters.get("inputs", {}) if parameters else {},
        }

        result = self._make_request(
            f"repos/{self.project}/actions/workflows/{workflow}/dispatches",
            method="POST",
            data=data,
        )

        return result.get("id", 0)

    def cancel_build(self, build_id: Union[int, str]) -> bool:
        """Cancel a GitHub Actions workflow run."""
        try:
            self._make_request(
                f"repos/{self.project}/actions/runs/{build_id}/cancel",
                method="POST",
            )
            return True
        except Exception:
            return False

    def publish_test_results(
        self,
        results: List[CITestResult],
        build_id: Optional[Union[int, str]] = None,
    ) -> bool:
        """Publish test results using GitHub Checks API."""
        # Create a check run with test results
        data = {
            "name": "Test Results",
            "head_sha": os.environ.get("GITHUB_SHA", ""),
            "status": "completed",
            "conclusion": "success" if all(r.passed for r in results) else "failure",
            "output": {
                "title": "Test Results",
                "summary": f"{sum(1 for r in results if r.passed)}/{len(results)} tests passed",
                "annotations": [
                    {
                        "path": r.suite or "tests",
                        "start_line": 1,
                        "end_line": 1,
                        "annotation_level": "notice" if r.passed else "failure",
                        "message": r.message or r.name,
                        "title": r.name,
                    }
                    for r in results
                    if not r.passed or r.message
                ],
            },
        }

        try:
            self._make_request(
                f"repos/{self.project}/check-runs",
                method="POST",
                data=data,
            )
            return True
        except Exception:
            return False

    def upload_artifact(
        self,
        path: str,
        name: str,
        build_id: Optional[Union[int, str]] = None,
    ) -> str:
        """Upload artifact using GitHub Actions artifact API."""
        # Artifacts are typically uploaded via the actions/upload-artifact action
        # This is a placeholder for direct API usage
        return f"https://github.com/{self.project}/actions/runs/{build_id}"

    def set_build_status(
        self,
        status: CIBuildStatus,
        description: str = "",
        context: str = "build",
    ) -> bool:
        """Set commit status on GitHub."""
        state_map = {
            CIBuildStatus.PENDING: "pending",
            CIBuildStatus.RUNNING: "pending",
            CIBuildStatus.SUCCESS: "success",
            CIBuildStatus.FAILED: "failure",
            CIBuildStatus.CANCELLED: "error",
            CIBuildStatus.UNSTABLE: "failure",
        }

        sha = os.environ.get("GITHUB_SHA", "")
        if not sha:
            return False

        data = {
            "state": state_map.get(status, "pending"),
            "description": description[:140],  # GitHub limit
            "context": context,
        }

        try:
            self._make_request(
                f"repos/{self.project}/statuses/{sha}",
                method="POST",
                data=data,
            )
            return True
        except Exception:
            return False

    def post_comment(
        self,
        message: str,
        pr_number: Optional[int] = None,
    ) -> bool:
        """Post a comment on a PR."""
        if not pr_number:
            # Try to get from environment
            pr_number = int(os.environ.get("GITHUB_PR_NUMBER", 0))

        if not pr_number:
            return False

        try:
            self._make_request(
                f"repos/{self.project}/issues/{pr_number}/comments",
                method="POST",
                data={"body": message},
            )
            return True
        except Exception:
            return False


class TeamCityIntegration(CIProvider):
    """
    TeamCity CI integration.

    Provides integration with JetBrains TeamCity.
    """

    name = "teamcity"

    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make authenticated request to TeamCity."""
        url = f"{self.url}/app/rest/{endpoint}"

        request = urllib.request.Request(url, method=method)
        request.add_header("Accept", "application/json")
        request.add_header("Authorization", f"Bearer {self.token}")

        if data:
            request.add_header("Content-Type", "application/json")
            request.data = json.dumps(data).encode()

        try:
            with urllib.request.urlopen(request) as response:
                return json.loads(response.read().decode())
        except Exception:
            return {}

    def get_build_status(self, build_id: Union[int, str]) -> CIBuildStatus:
        """Get status of a TeamCity build."""
        info = self._make_request(f"builds/id:{build_id}")

        state = info.get("state")
        status = info.get("status")

        if state == "queued":
            return CIBuildStatus.PENDING
        elif state == "running":
            return CIBuildStatus.RUNNING
        elif status == "SUCCESS":
            return CIBuildStatus.SUCCESS
        elif status == "FAILURE":
            return CIBuildStatus.FAILED
        else:
            return CIBuildStatus.PENDING

    def get_build_info(self, build_id: Union[int, str]) -> CIBuildResult:
        """Get full TeamCity build information."""
        info = self._make_request(f"builds/id:{build_id}")

        return CIBuildResult(
            status=self.get_build_status(build_id),
            build_number=info.get("number", 0),
            url=info.get("webUrl", ""),
        )

    def trigger_build(
        self,
        branch: str = "main",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Union[int, str]:
        """Trigger a TeamCity build."""
        data = {
            "buildType": {"id": self.project},
            "branchName": branch,
        }

        if parameters:
            data["properties"] = {
                "property": [
                    {"name": k, "value": v}
                    for k, v in parameters.items()
                ]
            }

        result = self._make_request(
            "buildQueue",
            method="POST",
            data=data,
        )

        return result.get("id", 0)

    def cancel_build(self, build_id: Union[int, str]) -> bool:
        """Cancel a TeamCity build."""
        try:
            self._make_request(
                f"builds/id:{build_id}",
                method="POST",
                data={"comment": "Cancelled via API", "readdIntoQueue": False},
            )
            return True
        except Exception:
            return False

    def publish_test_results(
        self,
        results: List[CITestResult],
        build_id: Optional[Union[int, str]] = None,
    ) -> bool:
        """Publish test results to TeamCity."""
        # TeamCity uses service messages for test reporting
        for result in results:
            status = "testPassed" if result.passed else "testFailed"
            print(f"##teamcity[{status} name='{result.name}' duration='{int(result.duration * 1000)}']")
        return True

    def upload_artifact(
        self,
        path: str,
        name: str,
        build_id: Optional[Union[int, str]] = None,
    ) -> str:
        """Upload artifact to TeamCity."""
        # TeamCity artifacts are typically published via service messages
        print(f"##teamcity[publishArtifacts '{path} => {name}']")
        return f"{self.url}/repository/download/{self.project}/{build_id}/{name}"


def create_ci_provider(
    provider_type: str = "auto",
    **kwargs,
) -> CIProvider:
    """
    Create a CI provider based on environment or explicit type.

    Args:
        provider_type: Provider type (auto, jenkins, github, teamcity)
        **kwargs: Provider-specific arguments

    Returns:
        CI provider instance
    """
    if provider_type == "auto":
        # Auto-detect based on environment
        if os.environ.get("GITHUB_ACTIONS"):
            provider_type = "github"
        elif os.environ.get("JENKINS_URL"):
            provider_type = "jenkins"
        elif os.environ.get("TEAMCITY_VERSION"):
            provider_type = "teamcity"
        else:
            provider_type = "github"  # Default

    providers: Dict[str, Type[CIProvider]] = {
        "jenkins": JenkinsIntegration,
        "github": GitHubActionsIntegration,
        "teamcity": TeamCityIntegration,
    }

    provider_class = providers.get(provider_type, GitHubActionsIntegration)
    return provider_class(**kwargs)


def publish_results(
    results: List[CITestResult],
    provider: Optional[CIProvider] = None,
    **kwargs,
) -> bool:
    """
    Publish test results to CI system.

    Args:
        results: Test results to publish
        provider: CI provider (auto-detected if not provided)
        **kwargs: Additional arguments

    Returns:
        Success status
    """
    if provider is None:
        provider = create_ci_provider(**kwargs)

    return provider.publish_test_results(results)
