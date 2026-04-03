"""
Dense reward shaping functions for the Team Collaboration Environment.

Provides continuous gradient signal every step — not just binary end-of-episode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation import Project, Simulation, TeamMember


# ── Reward constants ──────────────────────────────────────────────────────
R_PROGRESS = 2.0           # per unit of progress delta
R_PROJECT_COMPLETE = 5.0
R_DEPENDENCY_UNBLOCKED = 1.0
R_TEAM_SYNERGY = 0.5
R_SMART_REST = 0.3         # resting when energy < 0.3
R_BURNOUT_PENALTY = -1.0   # working with energy < 0.2
R_IDLE_PENALTY = -0.2      # member with no assignment, not resting
R_DEADLINE_MISSED = -3.0
R_CASCADE_FAILURE = -1.5
R_BLOCKED_WORK_ATTEMPT = -0.5
R_ALL_COMPLETE_BONUS = 10.0


def compute_step_reward(
    sim: "Simulation",
    progress_deltas: dict[str, float],
    events: list[str],
    action_error: str | None,
) -> float:
    """Compute the total reward for a single step."""
    reward = 0.0

    # ── Progress rewards ──────────────────────────────────────────────
    for pid, delta in progress_deltas.items():
        reward += delta * R_PROGRESS

    # ── Completion rewards ────────────────────────────────────────────
    for ev in events:
        if "completed" in ev:
            reward += R_PROJECT_COMPLETE
        if "unblocked" in ev:
            reward += R_DEPENDENCY_UNBLOCKED
        if "missed deadline" in ev:
            reward += R_DEADLINE_MISSED
        if "Cascade" in ev:
            reward += R_CASCADE_FAILURE

    # ── Burnout penalties ─────────────────────────────────────────────
    for member in sim.members:
        if member.unavailable_steps > 0:
            continue
        if member.is_resting and member.energy < 0.3:
            reward += R_SMART_REST
        elif (
            member.current_task_id
            and not member.is_resting
            and member.energy < 0.2
        ):
            reward += R_BURNOUT_PENALTY

    # ── Idle penalty ──────────────────────────────────────────────────
    for member in sim.members:
        if member.unavailable_steps > 0:
            continue
        if not member.current_task_id and not member.is_resting:
            reward += R_IDLE_PENALTY

    # ── Team synergy ──────────────────────────────────────────────────
    for project in sim.projects:
        if project.status != "in_progress":
            continue
        working = [
            m
            for m in sim.members
            if m.current_task_id == project.id
            and not m.is_resting
            and m.unavailable_steps <= 0
        ]
        if len(working) >= 2:
            team_ids = {m.team_id for m in working if m.team_id}
            if team_ids:
                reward += R_TEAM_SYNERGY

    # ── Blocked work attempt ──────────────────────────────────────────
    if action_error and "blocked" in action_error.lower():
        reward += R_BLOCKED_WORK_ATTEMPT

    # ── All-complete bonus ────────────────────────────────────────────
    if all(p.status in ("completed", "failed") for p in sim.projects):
        if all(p.status == "completed" for p in sim.projects):
            reward += R_ALL_COMPLETE_BONUS

    return round(reward, 4)
