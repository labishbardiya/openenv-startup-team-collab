"""
Simulation engine for the Team Collaboration Environment.

Models team members, projects (with dependencies), teams, energy/burnout,
personality traits, and random disruption events.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal, Optional


# ── Constants ─────────────────────────────────────────────────────────────
BASE_WORK_RATE = 0.15
ENERGY_DRAIN_PER_DIFFICULTY = 0.06
REST_RECOVERY = 0.25
BURNOUT_THRESHOLD = 0.2
BURNOUT_EFFICIENCY = 0.3
COLLAB_BONUS = 1.3
WORK_STYLE_MULTIPLIER: dict[str, float] = {
    "lazy": 0.6,
    "normal": 1.0,
    "efficient": 1.3,
}

# Random event probabilities (per entity per step)
EVENT_BUG_PROB = 0.08
EVENT_UNAVAIL_PROB = 0.05
EVENT_SCOPE_CREEP_PROB = 0.03
EVENT_BREAKTHROUGH_PROB = 0.04


# ── Data Models ───────────────────────────────────────────────────────────
@dataclass
class TeamMember:
    id: str
    name: str
    skills: dict[str, float]
    energy: float = 1.0
    current_task_id: Optional[str] = None
    team_id: Optional[str] = None
    work_style: Literal["lazy", "normal", "efficient"] = "normal"
    reliability: float = 0.95
    unavailable_steps: int = 0
    is_resting: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "skills": dict(self.skills),
            "energy": round(self.energy, 3),
            "current_task_id": self.current_task_id,
            "team_id": self.team_id,
            "work_style": self.work_style,
            "reliability": self.reliability,
            "unavailable_steps": self.unavailable_steps,
            "is_resting": self.is_resting,
        }


@dataclass
class Project:
    id: str
    name: str
    required_skills: list[str]
    difficulty: float = 1.0
    deadline: int = 20
    progress: float = 0.0
    status: str = "pending"
    dependencies: list[str] = field(default_factory=list)
    original_deadline: int = 20
    blocked_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "required_skills": list(self.required_skills),
            "difficulty": round(self.difficulty, 2),
            "deadline": self.deadline,
            "original_deadline": self.original_deadline,
            "progress": round(self.progress, 4),
            "status": self.status,
            "dependencies": list(self.dependencies),
            "blocked_reason": self.blocked_reason,
        }


@dataclass
class Team:
    id: str
    member_ids: list[str]

    def to_dict(self) -> dict:
        return {"id": self.id, "member_ids": list(self.member_ids)}


# ── Simulation ────────────────────────────────────────────────────────────
class Simulation:
    """Core simulation engine – pure Python, no framework dependency."""

    def __init__(self) -> None:
        self.members: list[TeamMember] = []
        self.projects: list[Project] = []
        self.teams: list[Team] = []
        self.current_step: int = 0
        self.max_steps: int = 20
        self.rng: random.Random = random.Random(42)
        self.events_this_step: list[str] = []
        self.burnout_incidents: int = 0
        self.total_work_actions: int = 0
        self.enable_random_events: bool = True
        self.event_probability_mult: float = 1.0

    # ── Setup ─────────────────────────────────────────────────────────
    def setup(
        self,
        members: list[TeamMember],
        projects: list[Project],
        max_steps: int,
        seed: int = 42,
        enable_events: bool = True,
        event_mult: float = 1.0,
    ) -> None:
        self.members = members
        self.projects = projects
        self.teams = []
        self.current_step = 0
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.events_this_step = []
        self.burnout_incidents = 0
        self.total_work_actions = 0
        self.enable_random_events = enable_events
        self.event_probability_mult = event_mult
        self._update_project_statuses()

    # ── Lookups ───────────────────────────────────────────────────────
    def get_member(self, mid: str) -> Optional[TeamMember]:
        return next((m for m in self.members if m.id == mid), None)

    def get_project(self, pid: str) -> Optional[Project]:
        return next((p for p in self.projects if p.id == pid), None)

    def get_team(self, tid: str) -> Optional[Team]:
        return next((t for t in self.teams if t.id == tid), None)

    def get_member_team(self, mid: str) -> Optional[Team]:
        return next((t for t in self.teams if mid in t.member_ids), None)

    # ── Actions ───────────────────────────────────────────────────────
    def assign_member(self, member_id: str, task_id: str) -> tuple[bool, str]:
        member = self.get_member(member_id)
        if not member:
            return False, f"Member '{member_id}' not found"
        project = self.get_project(task_id)
        if not project:
            return False, f"Project '{task_id}' not found"
        if project.status == "blocked":
            return False, f"Project '{task_id}' is blocked: {project.blocked_reason}"
        if project.status == "completed":
            return False, f"Project '{task_id}' is already completed"
        if project.status == "failed":
            return False, f"Project '{task_id}' has failed"
        if member.unavailable_steps > 0:
            return False, f"{member.name} is unavailable for {member.unavailable_steps} step(s)"
        member.current_task_id = task_id
        member.is_resting = False
        if project.status == "pending":
            project.status = "in_progress"
        return True, f"Assigned {member.name} to {project.name}"

    def unassign_member(self, member_id: str) -> tuple[bool, str]:
        member = self.get_member(member_id)
        if not member:
            return False, f"Member '{member_id}' not found"
        if not member.current_task_id:
            return False, f"{member.name} has no assignment"
        old = member.current_task_id
        member.current_task_id = None
        return True, f"Unassigned {member.name} from {old}"

    def form_team(self, member_ids: list[str]) -> tuple[bool, str]:
        if len(member_ids) < 2:
            return False, "Need at least 2 members to form a team"
        members = [self.get_member(mid) for mid in member_ids]
        if any(m is None for m in members):
            return False, "One or more member IDs not found"
        for m in members:  # type: ignore[union-attr]
            if m.team_id:
                return False, f"{m.name} is already in team {m.team_id}"
        tid = f"team_{len(self.teams) + 1}"
        team = Team(id=tid, member_ids=list(member_ids))
        self.teams.append(team)
        for m in members:  # type: ignore[union-attr]
            m.team_id = tid
        names = ", ".join(m.name for m in members)  # type: ignore[union-attr]
        return True, f"Formed {tid} with {names}"

    def disband_team(self, team_id: str) -> tuple[bool, str]:
        team = self.get_team(team_id)
        if not team:
            return False, f"Team '{team_id}' not found"
        for mid in team.member_ids:
            member = self.get_member(mid)
            if member:
                member.team_id = None
        self.teams.remove(team)
        return True, f"Disbanded {team_id}"

    def rest_member(self, member_id: str) -> tuple[bool, str]:
        member = self.get_member(member_id)
        if not member:
            return False, f"Member '{member_id}' not found"
        if member.unavailable_steps > 0:
            return False, f"{member.name} is unavailable"
        member.is_resting = True
        return True, f"{member.name} will rest this step"

    # ── Step Advance ──────────────────────────────────────────────────
    def advance_step(self) -> dict:
        """Advance one step: everyone works, events fire, deadlines check."""
        self.current_step += 1
        self.events_this_step = []
        progress_deltas: dict[str, float] = {}

        # 1) Work phase – every assigned, non-resting, available member works
        for member in self.members:
            if member.unavailable_steps > 0:
                member.unavailable_steps -= 1
                continue
            if member.is_resting:
                member.energy = min(1.0, member.energy + REST_RECOVERY)
                member.is_resting = False
                continue
            if not member.current_task_id:
                continue
            project = self.get_project(member.current_task_id)
            if not project or project.status in ("completed", "failed", "blocked"):
                continue

            delta = self._compute_work_progress(member, project)
            self.total_work_actions += 1

            # Track burnout
            if member.energy < BURNOUT_THRESHOLD:
                self.burnout_incidents += 1

            # Drain energy
            member.energy = max(0.0, member.energy - ENERGY_DRAIN_PER_DIFFICULTY * project.difficulty)

            pid = project.id
            progress_deltas[pid] = progress_deltas.get(pid, 0.0) + delta

        # 2) Apply collaboration bonuses & update progress
        for pid, raw_delta in progress_deltas.items():
            project = self.get_project(pid)
            if not project:
                continue
            working = [m for m in self.members if m.current_task_id == pid and not m.is_resting and m.unavailable_steps <= 0]
            boosted = self._apply_collab_bonus(project, raw_delta, working)
            project.progress = min(1.0, project.progress + boosted)
            if project.progress >= 1.0:
                project.status = "completed"
                self.events_this_step.append(f"🎉 Project '{project.name}' completed!")

        # 3) Random events
        if self.enable_random_events:
            self._roll_random_events()

        # 4) Update project statuses (deps, deadlines)
        self._update_project_statuses()

        return {
            "step": self.current_step,
            "progress_deltas": {k: round(v, 4) for k, v in progress_deltas.items()},
            "events": list(self.events_this_step),
        }

    # ── Progress Computation ──────────────────────────────────────────
    def _compute_work_progress(self, member: TeamMember, project: Project) -> float:
        skill_match = max(
            (member.skills.get(s, 0.0) for s in project.required_skills),
            default=0.1,
        )
        style_mult = WORK_STYLE_MULTIPLIER.get(member.work_style, 1.0)
        if member.energy < BURNOUT_THRESHOLD:
            energy_factor = member.energy * BURNOUT_EFFICIENCY
        else:
            energy_factor = member.energy
        if self.rng.random() > member.reliability:
            return 0.0
        return skill_match * style_mult * energy_factor / project.difficulty * BASE_WORK_RATE

    def _apply_collab_bonus(
        self, project: Project, total: float, working: list[TeamMember]
    ) -> float:
        if len(working) < 2:
            return total
        team_ids = {m.team_id for m in working if m.team_id}
        if not team_ids:
            return total
        covered = sum(
            1
            for s in project.required_skills
            if any(m.skills.get(s, 0) > 0.3 for m in working)
        )
        if covered >= 2 or len(working) >= 2:
            return total * COLLAB_BONUS
        return total

    # ── Random Events ─────────────────────────────────────────────────
    def _roll_random_events(self) -> None:
        mult = self.event_probability_mult
        for project in self.projects:
            if project.status not in ("in_progress",):
                continue
            if self.rng.random() < EVENT_BUG_PROB * mult:
                lost = project.progress * 0.15
                project.progress = max(0.0, project.progress - lost)
                self.events_this_step.append(
                    f"🐛 Bug in '{project.name}': lost {lost:.0%} progress"
                )
            if self.rng.random() < EVENT_SCOPE_CREEP_PROB * mult:
                project.difficulty = min(3.0, project.difficulty + 0.3)
                self.events_this_step.append(
                    f"📈 Scope creep on '{project.name}': difficulty → {project.difficulty:.1f}"
                )
            if self.rng.random() < EVENT_BREAKTHROUGH_PROB * mult:
                bonus = 0.10
                project.progress = min(1.0, project.progress + bonus)
                self.events_this_step.append(
                    f"💡 Breakthrough on '{project.name}': +10% progress"
                )
        for member in self.members:
            if member.unavailable_steps > 0:
                continue
            if self.rng.random() < EVENT_UNAVAIL_PROB * mult:
                member.unavailable_steps = 1
                self.events_this_step.append(
                    f"🤒 {member.name} unavailable for 1 step"
                )

    # ── Status Updates ────────────────────────────────────────────────
    def _update_project_statuses(self) -> None:
        for project in self.projects:
            if project.status in ("completed", "failed"):
                continue
            if project.dependencies:
                deps = [self.get_project(d) for d in project.dependencies]
                deps = [d for d in deps if d is not None]
                any_failed = any(d.status == "failed" for d in deps)
                all_complete = all(d.status == "completed" for d in deps)
                if any_failed and project.status != "blocked":
                    project.deadline = min(project.deadline + 3, self.max_steps)
                    project.difficulty = min(project.difficulty + 0.2, 3.0)
                    self.events_this_step.append(
                        f"⚠️ Cascade: '{project.name}' deadline+3, difficulty+0.2"
                    )
                if not all_complete:
                    project.status = "blocked"
                    pending = [d.id for d in deps if d.status != "completed"]
                    project.blocked_reason = f"Waiting on: {', '.join(pending)}"
                else:
                    if project.status == "blocked":
                        project.status = "pending"
                        project.blocked_reason = None
                        self.events_this_step.append(
                            f"🔓 '{project.name}' unblocked!"
                        )
            if (
                project.status not in ("completed", "blocked")
                and self.current_step > project.deadline
            ):
                project.status = "failed"
                self.events_this_step.append(
                    f"❌ Project '{project.name}' missed deadline!"
                )

    # ── Queries ───────────────────────────────────────────────────────
    @property
    def is_done(self) -> bool:
        if self.current_step >= self.max_steps:
            return True
        return all(p.status in ("completed", "failed") for p in self.projects)

    @property
    def completed_count(self) -> int:
        return sum(1 for p in self.projects if p.status == "completed")

    @property
    def failed_count(self) -> int:
        return sum(1 for p in self.projects if p.status == "failed")
