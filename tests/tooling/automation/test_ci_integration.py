"""
Tests for ci_integration.py - CI/CD integration.
"""

import os

import pytest

from engine.tooling.automation.ci_integration import (
    CIBuildResult,
    CIBuildStatus,
    CIProvider,
    CITestResult,
    GitHubActionsIntegration,
    JenkinsIntegration,
    TeamCityIntegration,
    create_ci_provider,
    publish_results,
)


class TestCIBuildStatus:
    """Tests for CIBuildStatus enum."""

    def test_status_values_exist(self):
        assert CIBuildStatus.PENDING
        assert CIBuildStatus.RUNNING
        assert CIBuildStatus.SUCCESS
        assert CIBuildStatus.FAILED
        assert CIBuildStatus.CANCELLED
        assert CIBuildStatus.UNSTABLE


class TestCITestResult:
    """Tests for CITestResult dataclass."""

    def test_create_result(self):
        result = CITestResult(
            name="test_example",
            passed=True,
            duration=1.5,
        )

        assert result.name == "test_example"
        assert result.passed is True
        assert result.duration == 1.5

    def test_to_dict(self):
        result = CITestResult(
            name="test_example",
            passed=True,
            duration=1.5,
            message="Success",
            suite="MySuite",
        )
        data = result.to_dict()

        assert data["name"] == "test_example"
        assert data["passed"] is True
        assert data["suite"] == "MySuite"


class TestCIBuildResult:
    """Tests for CIBuildResult dataclass."""

    def test_create_result(self):
        result = CIBuildResult(
            status=CIBuildStatus.SUCCESS,
            build_number=123,
            duration=300.0,
        )

        assert result.status == CIBuildStatus.SUCCESS
        assert result.build_number == 123
        assert result.success is True

    def test_success_property(self):
        success = CIBuildResult(status=CIBuildStatus.SUCCESS)
        failed = CIBuildResult(status=CIBuildStatus.FAILED)

        assert success.success is True
        assert failed.success is False

    def test_test_counts(self):
        result = CIBuildResult(status=CIBuildStatus.SUCCESS)
        result.test_results = [
            CITestResult("t1", True),
            CITestResult("t2", True),
            CITestResult("t3", False),
        ]

        assert result.tests_passed == 2
        assert result.tests_failed == 1


class TestJenkinsIntegration:
    """Tests for JenkinsIntegration class."""

    def test_create_integration(self):
        jenkins = JenkinsIntegration(
            url="http://jenkins.example.com",
            token="test-token",
            project="my-project",
        )

        assert jenkins.url == "http://jenkins.example.com"
        assert jenkins.project == "my-project"

    def test_name_attribute(self):
        jenkins = JenkinsIntegration()
        assert jenkins.name == "jenkins"


class TestGitHubActionsIntegration:
    """Tests for GitHubActionsIntegration class."""

    def test_create_integration(self):
        github = GitHubActionsIntegration(
            token="ghp_test-token",
            project="owner/repo",
        )

        assert github.project == "owner/repo"

    def test_default_url(self):
        github = GitHubActionsIntegration()
        assert "api.github.com" in github.url

    def test_name_attribute(self):
        github = GitHubActionsIntegration()
        assert github.name == "github"


class TestTeamCityIntegration:
    """Tests for TeamCityIntegration class."""

    def test_create_integration(self):
        tc = TeamCityIntegration(
            url="http://teamcity.example.com",
            token="test-token",
            project="MyProject",
        )

        assert tc.url == "http://teamcity.example.com"
        assert tc.project == "MyProject"

    def test_name_attribute(self):
        tc = TeamCityIntegration()
        assert tc.name == "teamcity"


class TestCreateCIProvider:
    """Tests for create_ci_provider function."""

    def test_create_jenkins(self):
        provider = create_ci_provider("jenkins")
        assert isinstance(provider, JenkinsIntegration)

    def test_create_github(self):
        provider = create_ci_provider("github")
        assert isinstance(provider, GitHubActionsIntegration)

    def test_create_teamcity(self):
        provider = create_ci_provider("teamcity")
        assert isinstance(provider, TeamCityIntegration)

    def test_auto_detect_github_actions(self, monkeypatch):
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        provider = create_ci_provider("auto")
        assert isinstance(provider, GitHubActionsIntegration)

    def test_auto_detect_jenkins(self, monkeypatch):
        monkeypatch.setenv("JENKINS_URL", "http://jenkins.example.com")
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        provider = create_ci_provider("auto")
        assert isinstance(provider, JenkinsIntegration)

    def test_auto_detect_teamcity(self, monkeypatch):
        monkeypatch.setenv("TEAMCITY_VERSION", "2021.1")
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("JENKINS_URL", raising=False)
        provider = create_ci_provider("auto")
        assert isinstance(provider, TeamCityIntegration)


class TestPublishResults:
    """Tests for publish_results function."""

    def test_publish_empty_results(self):
        results = []
        # Should not raise even with empty results
        success = publish_results(results)
        # Result depends on implementation


class TestCIProviderInterface:
    """Tests for CIProvider interface methods."""

    def test_provider_has_required_methods(self):
        # All providers should implement these methods
        providers = [
            JenkinsIntegration(),
            GitHubActionsIntegration(),
            TeamCityIntegration(),
        ]

        for provider in providers:
            assert hasattr(provider, "get_build_status")
            assert hasattr(provider, "get_build_info")
            assert hasattr(provider, "trigger_build")
            assert hasattr(provider, "cancel_build")
            assert hasattr(provider, "publish_test_results")
            assert hasattr(provider, "upload_artifact")
