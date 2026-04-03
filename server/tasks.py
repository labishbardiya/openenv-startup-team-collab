"""
Task definitions and deterministic graders for the three difficulty levels.

Each task returns a config dict consumed by the environment's reset().
Each grader takes a Simulation and returns a float score in [0.0, 1.0].
"""

from __future__ import annotations

from .simulation import Project, Simulation, TeamMember

TASKS = ("solo_sprint", "team_crunch", "deadline_hell")


# ══════════════════════════════════════════════════════════════════════════
# Task 1 — solo_sprint (Easy)
# ══════════════════════════════════════════════════════════════════════════
def _solo_sprint_config() -> dict:
    members = [
        TeamMember(
            id="alice", name="Alice",
            skills={"frontend": 0.9, "design": 0.6},
            work_style="normal", reliability=0.95, energy=1.0,
        ),
        TeamMember(
            id="bob", name="Bob",
            skills={"backend": 0.8, "devops": 0.7},
            work_style="normal", reliability=0.95, energy=1.0,
        ),
        TeamMember(
            id="carol", name="Carol",
            skills={"design": 0.8, "frontend": 0.5},
            work_style="normal", reliability=0.90, energy=1.0,
        ),
        TeamMember(
            id="dave", name="Dave",
            skills={"backend": 0.6, "devops": 0.8},
            work_style="normal", reliability=0.95, energy=1.0,
        ),
    ]
    projects = [
        Project(
            id="p1", name="Homepage Redesign",
            required_skills=["frontend", "design"],
            difficulty=0.8, deadline=15, original_deadline=15,
        ),
    ]
    return {
        "members": members,
        "projects": projects,
        "max_steps": 15,
        "seed": 42,
        "enable_events": False,
        "event_mult": 0.0,
    }


def grade_solo_sprint(sim: Simulation) -> float:
    p = sim.get_project("p1")
    return round(min(1.0, p.progress), 4) if p else 0.0


# ══════════════════════════════════════════════════════════════════════════
# Task 2 — team_crunch (Medium)
# ══════════════════════════════════════════════════════════════════════════
def _team_crunch_config() -> dict:
    members = [
        TeamMember(
            id="alice", name="Alice",
            skills={"frontend": 0.9, "design": 0.6},
            work_style="efficient", reliability=0.90, energy=1.0,
        ),
        TeamMember(
            id="bob", name="Bob",
            skills={"backend": 0.85, "devops": 0.5},
            work_style="normal", reliability=0.95, energy=1.0,
        ),
        TeamMember(
            id="carol", name="Carol",
            skills={"design": 0.7, "frontend": 0.4},
            work_style="lazy", reliability=0.85, energy=1.0,
        ),
        TeamMember(
            id="dave", name="Dave",
            skills={"devops": 0.8, "backend": 0.6},
            work_style="normal", reliability=1.0, energy=1.0,
        ),
    ]
    projects = [
        Project(
            id="p1", name="Backend API",
            required_skills=["backend"],
            difficulty=0.8, deadline=12, original_deadline=12,
        ),
        Project(
            id="p2", name="Frontend App",
            required_skills=["frontend", "design"],
            difficulty=1.2, deadline=18, original_deadline=18,
            dependencies=["p1"],
        ),
        Project(
            id="p3", name="Landing Page",
            required_skills=["frontend", "design"],
            difficulty=1.0, deadline=16, original_deadline=16,
        ),
    ]
    return {
        "members": members,
        "projects": projects,
        "max_steps": 20,
        "seed": 42,
        "enable_events": True,
        "event_mult": 0.5,
    }


def grade_team_crunch(sim: Simulation) -> float:
    total = len(sim.projects)
    completed = sim.completed_count
    avg_progress = sum(p.progress for p in sim.projects) / max(total, 1)
    score = (completed / total) * 0.7 + avg_progress * 0.3
    return round(min(1.0, score), 4)


# ══════════════════════════════════════════════════════════════════════════
# Task 3 — deadline_hell (Hard)
# ══════════════════════════════════════════════════════════════════════════
def _deadline_hell_config() -> dict:
    members = [
        TeamMember(
            id="alice", name="Alice",
            skills={"frontend": 0.8, "design": 0.5},
            work_style="efficient", reliability=0.90, energy=0.7,
        ),
        TeamMember(
            id="bob", name="Bob",
            skills={"backend": 0.75, "devops": 0.4},
            work_style="normal", reliability=0.80, energy=0.6,
        ),
        TeamMember(
            id="carol", name="Carol",
            skills={"design": 0.6, "backend": 0.3},
            work_style="lazy", reliability=0.70, energy=0.9,
        ),
        TeamMember(
            id="dave", name="Dave",
            skills={"devops": 0.7, "frontend": 0.5},
            work_style="normal", reliability=0.85, energy=0.6,
        ),
    ]
    projects = [
        Project(
            id="p1", name="Core API",
            required_skills=["backend"],
            difficulty=1.0, deadline=10, original_deadline=10,
        ),
        Project(
            id="p2", name="Integration Layer",
            required_skills=["backend", "devops"],
            difficulty=1.5, deadline=16, original_deadline=16,
            dependencies=["p1"],
        ),
        Project(
            id="p3", name="Marketing Site",
            required_skills=["frontend", "design"],
            difficulty=1.2, deadline=12, original_deadline=12,
        ),
        Project(
            id="p4", name="Full Platform",
            required_skills=["frontend", "backend", "devops"],
            difficulty=2.0, deadline=24, original_deadline=24,
            dependencies=["p2"],
        ),
    ]
    return {
        "members": members,
        "projects": projects,
        "max_steps": 25,
        "seed": 42,
        "enable_events": True,
        "event_mult": 1.0,
    }


def grade_deadline_hell(sim: Simulation) -> float:
    total = len(sim.projects)
    completed = sim.completed_count
    completion_rate = completed / total
    avg_progress = sum(p.progress for p in sim.projects) / max(total, 1)
    burnout_rate = (
        sim.burnout_incidents / max(1, sim.total_work_actions)
    )
    energy_mgmt = 1.0 - min(1.0, burnout_rate)
    score = completion_rate * 0.5 + avg_progress * 0.25 + energy_mgmt * 0.25
    return round(min(1.0, score), 4)


# ══════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════
_CONFIGS = {
    "solo_sprint": _solo_sprint_config,
    "team_crunch": _team_crunch_config,
    "deadline_hell": _deadline_hell_config,
}

_GRADERS = {
    "solo_sprint": grade_solo_sprint,
    "team_crunch": grade_team_crunch,
    "deadline_hell": grade_deadline_hell,
}


def get_task_config(task_name: str) -> dict:
    if task_name not in _CONFIGS:
        raise ValueError(f"Unknown task '{task_name}'. Choose from: {TASKS}")
    return _CONFIGS[task_name]()


def grade_task(task_name: str, sim: Simulation) -> float:
    if task_name not in _GRADERS:
        raise ValueError(f"Unknown task '{task_name}'. Choose from: {TASKS}")
    return _GRADERS[task_name](sim)
