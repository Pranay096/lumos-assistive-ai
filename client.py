"""
client.py — LUMOS Assistive AI OpenEnv Client
==============================================
Python client wrapper for the LUMOS environment HTTP server.
Matches the pattern seen in reasoning_gym_env/client.py and calendar_env/client.py.

Usage:
    from client import LumosClient

    client = LumosClient(base_url="http://localhost:7860")

    # Start a new episode
    obs = client.reset("blind_mode")

    # Take an action
    result = client.step({
        "action_type": "scan",
        "target": None,
        "content": None,
        "confidence": 0.9,
        "priority": None,
    })
    print(result["observation"], result["reward"], result["done"])

    # Inspect world state (debug)
    state = client.get_state()

    # Get final grader score
    score = client.grader()
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class LumosClient:
    """
    HTTP client for the LUMOS Assistive AI OpenEnv environment.

    All methods correspond 1-to-1 with the server endpoints:
        POST /reset  → reset()
        POST /step   → step()
        GET  /state  → get_state()
        POST /grader → grader()
        GET  /tasks  → list_tasks()
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Core OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "blind_mode") -> Dict[str, Any]:
        """
        Start a new episode for the given task.

        Args:
            task_id: One of "blind_mode", "deaf_mode", "mute_mode"

        Returns:
            Initial observation dict (partial view of the world).
        """
        resp = self._session.post(
            f"{self.base_url}/reset",
            params={"task_id": task_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take one action in the current episode.

        Args:
            action: dict with keys:
                - action_type (str): e.g. "scan", "listen", "observe_letter"
                - target (str | None): object name or letter
                - content (str | None): text generated for the user
                - confidence (float): agent self-reported confidence [0, 1]
                - priority (str | None): "current_task" | "handle_interrupt"

        Returns:
            dict with keys:
                - observation: updated (partial) world view
                - reward: float step reward
                - done: bool
                - info: dict with grader_score, partial_credits, success, etc.
        """
        resp = self._session.post(
            f"{self.base_url}/step",
            json=action,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_state(self) -> Dict[str, Any]:
        """
        Return the current hidden world state (for debugging / visualisation).
        Includes ground truth not visible to the agent (active hazards, target word, etc.)
        """
        resp = self._session.get(f"{self.base_url}/state", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def grader(self) -> Dict[str, Any]:
        """
        Return the final grader score for the current episode.

        Returns:
            dict with grader_score (float), success (bool), steps_taken (int).
        """
        resp = self._session.post(f"{self.base_url}/grader", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_tasks(self) -> list:
        """List all available tasks with valid actions and metadata."""
        resp = self._session.get(f"{self.base_url}/tasks", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        """Check if the environment server is running."""
        resp = self._session.get(f"{self.base_url}/", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "LumosClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self._session.close()

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()


# ------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------

def make_client(base_url: Optional[str] = None, timeout: int = 30) -> LumosClient:
    """
    Create a LumosClient, with optional URL override.

    Checks the BASE_ENV_URL environment variable if base_url is not given.
    Defaults to http://localhost:7860.
    """
    import os
    url = base_url or os.getenv("BASE_ENV_URL", "http://localhost:7860")
    return LumosClient(base_url=url, timeout=timeout)
