"""
Async client for the Team Collaboration Environment.

Supports:
  - from_docker_image(image_name)  → starts container, connects via HTTP
  - from_space_url(url)            → connects to HF Space via HTTP
  - Direct instantiation            → connects to local server

Follows the exact same pattern as the sample inference script.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import httpx


@dataclass
class StepResult:
    """Result from reset() or step() — matches OpenEnv spec."""
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class TeamCollabEnv:
    """Async HTTP client for the Team Collaboration Environment.

    Usage:
        env = await TeamCollabEnv.from_docker_image(IMAGE_NAME)
        result = await env.reset(task_name="solo_sprint")
        result = await env.step(action)
        await env.close()
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        self._container_id: Optional[str] = None

    # ── Factory methods ───────────────────────────────────────────────
    @classmethod
    async def from_docker_image(cls, image_name: str) -> "TeamCollabEnv":
        """Start a Docker container and return a connected client."""
        port = cls._find_free_port()
        print(f"[DEBUG] Starting container {image_name} on port {port}...", file=sys.stderr)

        container_id = subprocess.check_output(
            ["docker", "run", "-d", "-p", f"{port}:8000", image_name],
            stderr=subprocess.STDOUT,
        ).decode().strip()

        env = cls(base_url=f"http://localhost:{port}")
        env._container_id = container_id

        # Wait for health check
        for attempt in range(60):
            try:
                r = await env._client.get("/health")
                if r.status_code == 200:
                    print(f"[DEBUG] Container healthy after {attempt + 1}s", file=sys.stderr)
                    return env
            except Exception:
                pass
            await asyncio.sleep(1.0)

        raise RuntimeError(f"Container {image_name} failed to start within 60s")

    @classmethod
    async def from_space_url(cls, url: str) -> "TeamCollabEnv":
        """Connect to a running HF Space."""
        env = cls(base_url=url.rstrip("/"))
        r = await env._client.get("/health")
        if r.status_code != 200:
            raise RuntimeError(f"Space at {url} not healthy: HTTP {r.status_code}")
        return env

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    # ── Core API ──────────────────────────────────────────────────────
    async def reset(self, **kwargs) -> StepResult:
        """Reset environment. Returns StepResult with initial observation."""
        r = await self._client.post("/reset", json=kwargs)
        r.raise_for_status()
        data = r.json()
        return StepResult(
            observation=data.get("observation", {}),
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )

    async def step(self, action) -> StepResult:
        """Execute one step. action can be a Pydantic model or a dict."""
        if hasattr(action, "model_dump"):
            action_dict = action.model_dump(exclude_none=True)
        elif isinstance(action, dict):
            action_dict = action
        else:
            action_dict = {"action_type": "noop"}

        r = await self._client.post("/step", json={"action": action_dict})
        r.raise_for_status()
        data = r.json()
        return StepResult(
            observation=data.get("observation", {}),
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )

    async def state(self) -> Dict[str, Any]:
        """Get current environment state."""
        r = await self._client.get("/state")
        r.raise_for_status()
        return r.json()

    async def grade(self) -> float:
        """Get the deterministic grader score from the environment."""
        try:
            r = await self._client.get("/grade")
            r.raise_for_status()
            data = r.json()
            return float(data.get("score", 0.0))
        except Exception:
            return 0.0

    async def close(self) -> None:
        """Close HTTP client and stop Docker container if started."""
        await self._client.aclose()
        if self._container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self._container_id],
                    capture_output=True, timeout=15,
                )
                subprocess.run(
                    ["docker", "rm", "-f", self._container_id],
                    capture_output=True, timeout=15,
                )
                print(f"[DEBUG] Container {self._container_id[:12]} stopped", file=sys.stderr)
            except Exception as e:
                print(f"[DEBUG] Container cleanup error: {e}", file=sys.stderr)
            self._container_id = None
