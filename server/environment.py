"""
Team Collaboration Environment — OpenEnv compatible.

Wraps the simulation engine with reset() / step() / state() API.
Works standalone (direct Python) and via the FastAPI HTTP server.

step() returns observation with: observation data, reward, done flag, and info dict.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from models import TeamCollabAction, TeamCollabObservation, TeamCollabState
from server.rewards import compute_step_reward
from server.simulation import Simulation
from server.tasks import TASKS, get_task_config, grade_task


class TeamCollabEnvironment:
    """OpenEnv-compatible environment for adaptive team collaboration."""

    def __init__(self) -> None:
        self.sim = Simulation()
        self._state = TeamCollabState()
        self._task_name: str = ""
        self._closed: bool = False

    # ── reset ─────────────────────────────────────────────────────────
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "solo_sprint",
        **kwargs: Any,
    ) -> TeamCollabObservation:
        """Initialise a new episode for the given task."""
        self._closed = False
        self._task_name = task_name
        config = get_task_config(task_name)

        if seed is not None:
            config["seed"] = seed

        self.sim.setup(
            members=config["members"],
            projects=config["projects"],
            max_steps=config["max_steps"],
            seed=config["seed"],
            enable_events=config.get("enable_events", True),
            event_mult=config.get("event_mult", 1.0),
        )

        self._state = TeamCollabState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
        )

        return self._build_observation(reward=0.0, done=False, events=[], error=None)

    # ── step ──────────────────────────────────────────────────────────
    def step(self, action: TeamCollabAction, **kwargs: Any) -> TeamCollabObservation:
        """Execute one management action, then advance the simulation.

        Returns an Observation containing:
          - observation data (members, projects, teams, etc.)
          - reward (float)
          - done (bool)
          - info in metadata dict
        """
        action_error: Optional[str] = None

        # 1) Execute the management action
        at = action.action_type
        if at == "assign":
            ok, msg = self.sim.assign_member(action.member_id or "", action.task_id or "")
            if not ok:
                action_error = msg
        elif at == "unassign":
            ok, msg = self.sim.unassign_member(action.member_id or "")
            if not ok:
                action_error = msg
        elif at == "form_team":
            ok, msg = self.sim.form_team(action.team_members or [])
            if not ok:
                action_error = msg
        elif at == "disband_team":
            ok, msg = self.sim.disband_team(action.team_id or "")
            if not ok:
                action_error = msg
        elif at == "rest":
            ok, msg = self.sim.rest_member(action.member_id or "")
            if not ok:
                action_error = msg
        elif at == "noop":
            pass
        else:
            action_error = f"Unknown action_type '{at}'"

        # 2) Advance simulation (everyone works, events fire, deadlines check)
        step_info = self.sim.advance_step()

        # 3) Compute reward
        reward = compute_step_reward(
            self.sim,
            step_info["progress_deltas"],
            step_info["events"],
            action_error,
        )

        # 4) Update state
        self._state.step_count = self.sim.current_step
        self._state.total_reward += reward
        self._state.projects_completed = self.sim.completed_count
        self._state.projects_failed = self.sim.failed_count
        self._state.burnout_incidents = self.sim.burnout_incidents
        self._state.events_triggered += len(step_info["events"])

        done = self.sim.is_done

        return self._build_observation(
            reward=reward,
            done=done,
            events=step_info["events"],
            error=action_error,
        )

    # ── state ─────────────────────────────────────────────────────────
    @property
    def state(self) -> TeamCollabState:
        return self._state

    def get_state(self) -> TeamCollabState:
        """Callable version of state for API compatibility."""
        return self._state

    # ── grading ───────────────────────────────────────────────────────
    def grade(self) -> float:
        """Return the deterministic grader score for the current task."""
        return grade_task(self._task_name, self.sim)

    # ── close ─────────────────────────────────────────────────────────
    def close(self) -> None:
        """Clean up environment resources."""
        self._closed = True

    async def aclose(self) -> None:
        """Async close for compatibility."""
        self.close()

    # ── helpers ───────────────────────────────────────────────────────
    def _build_observation(
        self,
        reward: float,
        done: bool,
        events: list[str],
        error: Optional[str],
    ) -> TeamCollabObservation:
        return TeamCollabObservation(
            members=[m.to_dict() for m in self.sim.members],
            projects=[p.to_dict() for p in self.sim.projects],
            teams=[t.to_dict() for t in self.sim.teams],
            current_step=self.sim.current_step,
            max_steps=self.sim.max_steps,
            active_events=events,
            last_action_error=error,
            reward=reward,
            done=done,
            metadata={
                "task_name": self._task_name,
                "projects_completed": self.sim.completed_count,
                "projects_failed": self.sim.failed_count,
                "burnout_incidents": self.sim.burnout_incidents,
            },
        )
