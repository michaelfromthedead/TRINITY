"""
Distributed build agent management.

Provides infrastructure for managing distributed build agents
that can execute build jobs across multiple machines.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union


class AgentStatus(Enum):
    """Status of a build agent."""

    OFFLINE = auto()
    IDLE = auto()
    BUSY = auto()
    ERROR = auto()
    MAINTENANCE = auto()


class AgentCapability(Enum):
    """Capabilities that an agent can have."""

    WINDOWS = auto()
    LINUX = auto()
    MACOS = auto()
    GPU = auto()
    HIGH_MEMORY = auto()
    SSD = auto()
    DOCKER = auto()
    ANDROID_SDK = auto()
    IOS_SDK = auto()
    CONSOLE_DEVKIT = auto()


class BuildJobStatus(Enum):
    """Status of a build job."""

    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


@dataclass
class BuildAgent:
    """
    A build agent that can execute build jobs.

    Represents a machine that can be used for distributed builds.
    """

    id: str
    name: str
    hostname: str
    capabilities: Set[AgentCapability] = field(default_factory=set)
    status: AgentStatus = AgentStatus.OFFLINE
    current_job: Optional[str] = None
    max_concurrent_jobs: int = 1
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: float = 0.0

    @property
    def available(self) -> bool:
        """Check if agent is available for jobs."""
        return self.status == AgentStatus.IDLE

    @property
    def online(self) -> bool:
        """Check if agent is online."""
        return self.status not in (AgentStatus.OFFLINE, AgentStatus.ERROR)

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a capability."""
        return capability in self.capabilities

    def has_all_capabilities(self, capabilities: Set[AgentCapability]) -> bool:
        """Check if agent has all required capabilities."""
        return capabilities.issubset(self.capabilities)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "hostname": self.hostname,
            "capabilities": [c.name for c in self.capabilities],
            "status": self.status.name,
            "current_job": self.current_job,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildAgent":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            hostname=data["hostname"],
            capabilities={AgentCapability[c] for c in data.get("capabilities", [])},
            status=AgentStatus[data.get("status", "OFFLINE")],
            current_job=data.get("current_job"),
            tags=set(data.get("tags", [])),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BuildJob:
    """
    A build job to be executed by an agent.

    Contains all information needed to execute a build.
    """

    id: str
    name: str
    command: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_capabilities: Set[AgentCapability] = field(default_factory=set)
    priority: int = 0
    timeout: float = 3600.0  # 1 hour default
    status: BuildJobStatus = BuildJobStatus.PENDING
    agent_id: Optional[str] = None
    queued_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get job duration."""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return 0.0

    @property
    def wait_time(self) -> float:
        """Get time spent waiting in queue."""
        if self.started_at and self.queued_at:
            return self.started_at - self.queued_at
        elif self.queued_at:
            return time.time() - self.queued_at
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "command": self.command,
            "parameters": self.parameters,
            "required_capabilities": [c.name for c in self.required_capabilities],
            "priority": self.priority,
            "timeout": self.timeout,
            "status": self.status.name,
            "agent_id": self.agent_id,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


@dataclass
class BuildJobResult:
    """Result of a build job execution."""

    job_id: str
    status: BuildJobStatus
    exit_code: int = 0
    duration: float = 0.0
    output: str = ""
    errors: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if job succeeded."""
        return self.status == BuildJobStatus.COMPLETED and self.exit_code == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.name,
            "exit_code": self.exit_code,
            "duration": self.duration,
            "errors": self.errors,
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }


class BuildAgentPool:
    """
    Pool of build agents for job distribution.

    Manages a collection of agents and distributes jobs based
    on capabilities and availability.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._agents: Dict[str, BuildAgent] = {}
        self._job_queue: List[BuildJob] = []
        self._running_jobs: Dict[str, BuildJob] = {}
        self._completed_jobs: Dict[str, BuildJobResult] = {}

    def register_agent(self, agent: BuildAgent) -> None:
        """Register an agent with the pool."""
        self._agents[agent.id] = agent

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the pool."""
        self._agents.pop(agent_id, None)

    def get_agent(self, agent_id: str) -> Optional[BuildAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_available_agents(
        self,
        capabilities: Optional[Set[AgentCapability]] = None,
    ) -> List[BuildAgent]:
        """Get available agents with required capabilities."""
        agents = [a for a in self._agents.values() if a.available]

        if capabilities:
            agents = [a for a in agents if a.has_all_capabilities(capabilities)]

        return agents

    def submit_job(self, job: BuildJob) -> str:
        """Submit a job to the queue."""
        job.queued_at = time.time()
        job.status = BuildJobStatus.QUEUED
        self._job_queue.append(job)

        # Sort by priority (higher first)
        self._job_queue.sort(key=lambda j: j.priority, reverse=True)

        return job.id

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job."""
        # Check queue
        for i, job in enumerate(self._job_queue):
            if job.id == job_id:
                job.status = BuildJobStatus.CANCELLED
                self._job_queue.pop(i)
                return True

        # Check running jobs
        if job_id in self._running_jobs:
            job = self._running_jobs[job_id]
            job.status = BuildJobStatus.CANCELLED
            if job.agent_id:
                agent = self._agents.get(job.agent_id)
                if agent:
                    agent.status = AgentStatus.IDLE
                    agent.current_job = None
            del self._running_jobs[job_id]
            return True

        return False

    def dispatch_jobs(self) -> List[Tuple[str, str]]:
        """
        Dispatch queued jobs to available agents.

        Returns list of (job_id, agent_id) tuples for dispatched jobs.
        """
        dispatched = []

        remaining_queue = []
        for job in self._job_queue:
            agents = self.get_available_agents(job.required_capabilities)

            if agents:
                # Select agent (simple: first available)
                agent = agents[0]

                # Assign job
                job.status = BuildJobStatus.RUNNING
                job.started_at = time.time()
                job.agent_id = agent.id

                agent.status = AgentStatus.BUSY
                agent.current_job = job.id

                self._running_jobs[job.id] = job
                dispatched.append((job.id, agent.id))
            else:
                remaining_queue.append(job)

        self._job_queue = remaining_queue
        return dispatched

    def complete_job(
        self,
        job_id: str,
        result: BuildJobResult,
    ) -> None:
        """Mark a job as completed."""
        if job_id in self._running_jobs:
            job = self._running_jobs.pop(job_id)
            job.status = result.status
            job.completed_at = time.time()
            job.artifacts = result.artifacts

            # Free the agent
            if job.agent_id:
                agent = self._agents.get(job.agent_id)
                if agent:
                    agent.status = AgentStatus.IDLE
                    agent.current_job = None

            self._completed_jobs[job_id] = result

    def get_job_result(self, job_id: str) -> Optional[BuildJobResult]:
        """Get result of a completed job."""
        return self._completed_jobs.get(job_id)

    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status information."""
        return {
            "queued": len(self._job_queue),
            "running": len(self._running_jobs),
            "completed": len(self._completed_jobs),
            "agents_total": len(self._agents),
            "agents_available": len(self.get_available_agents()),
        }

    @property
    def agents(self) -> Dict[str, BuildAgent]:
        """Get all agents."""
        return self._agents.copy()


class BuildAgentManager:
    """
    Manager for coordinating build agents and jobs.

    Provides high-level management of distributed builds.
    """

    def __init__(self):
        self._pools: Dict[str, BuildAgentPool] = {}
        self._default_pool: Optional[str] = None

    def create_pool(self, name: str) -> BuildAgentPool:
        """Create a new agent pool."""
        pool = BuildAgentPool(name)
        self._pools[name] = pool
        if self._default_pool is None:
            self._default_pool = name
        return pool

    def get_pool(self, name: Optional[str] = None) -> Optional[BuildAgentPool]:
        """Get an agent pool by name."""
        if name is None:
            name = self._default_pool
        return self._pools.get(name) if name else None

    def register_agent(
        self,
        agent: BuildAgent,
        pool_name: Optional[str] = None,
    ) -> None:
        """Register an agent with a pool."""
        pool = self.get_pool(pool_name)
        if pool:
            pool.register_agent(agent)

    def submit_job(
        self,
        job: BuildJob,
        pool_name: Optional[str] = None,
    ) -> str:
        """Submit a job to a pool."""
        pool = self.get_pool(pool_name)
        if pool:
            return pool.submit_job(job)
        raise ValueError(f"Pool not found: {pool_name}")

    def dispatch_all(self) -> Dict[str, List[Tuple[str, str]]]:
        """Dispatch jobs in all pools."""
        results = {}
        for name, pool in self._pools.items():
            results[name] = pool.dispatch_jobs()
        return results

    def get_overall_status(self) -> Dict[str, Any]:
        """Get status across all pools."""
        total_agents = 0
        total_available = 0
        total_queued = 0
        total_running = 0

        for pool in self._pools.values():
            status = pool.get_queue_status()
            total_agents += status["agents_total"]
            total_available += status["agents_available"]
            total_queued += status["queued"]
            total_running += status["running"]

        return {
            "pools": len(self._pools),
            "agents_total": total_agents,
            "agents_available": total_available,
            "jobs_queued": total_queued,
            "jobs_running": total_running,
        }


def dispatch_build(
    command: str,
    parameters: Optional[Dict[str, Any]] = None,
    required_capabilities: Optional[Set[AgentCapability]] = None,
    manager: Optional[BuildAgentManager] = None,
    pool_name: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Dispatch a build job to available agents.

    Args:
        command: Build command to execute
        parameters: Build parameters
        required_capabilities: Required agent capabilities
        manager: Build agent manager (creates default if not provided)
        pool_name: Target pool name
        **kwargs: Additional job parameters

    Returns:
        Job ID
    """
    if manager is None:
        manager = BuildAgentManager()
        manager.create_pool("default")

    job = BuildJob(
        id=str(uuid.uuid4()),
        name=kwargs.get("name", command.split()[0]),
        command=command,
        parameters=parameters or {},
        required_capabilities=required_capabilities or set(),
        priority=kwargs.get("priority", 0),
        timeout=kwargs.get("timeout", 3600.0),
    )

    return manager.submit_job(job, pool_name)


def get_available_agents(
    manager: Optional[BuildAgentManager] = None,
    pool_name: Optional[str] = None,
    capabilities: Optional[Set[AgentCapability]] = None,
) -> List[BuildAgent]:
    """
    Get list of available build agents.

    Args:
        manager: Build agent manager
        pool_name: Pool to query
        capabilities: Required capabilities

    Returns:
        List of available agents
    """
    if manager is None:
        return []

    pool = manager.get_pool(pool_name)
    if pool:
        return pool.get_available_agents(capabilities)
    return []
