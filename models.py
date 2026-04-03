"""
Typed Pydantic models for the Team Collaboration Environment.

Defines the Action, Observation, and State types used by the OpenEnv spec.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Base types (compatible with openenv-core when installed, standalone otherwise)
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.types import Action as _ActionBase
    from openenv.core.env_server.types import Observation as _ObsBase
    from openenv.core.env_server.types import State as _StateBase
except ImportError:
    # Standalone fallback so the env works without openenv-core installed
    class _ActionBase(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class _ObsBase(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        done: bool = Field(default=False)
        reward: float = Field(default=0.0)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class _StateBase(BaseModel):
        model_config = ConfigDict(extra="allow", validate_assignment=True)
        episode_id: Optional[str] = None
        step_count: int = 0


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class TeamCollabAction(_ActionBase):
    """Single management action the agent takes each step.

    action_type is one of:
        assign      – assign a member to a project
        unassign    – remove a member from their project
        form_team   – group members for collaboration bonus
        disband_team – break up a team
        rest        – let a member rest (skip work, recover energy)
        noop        – do nothing (everyone works as assigned)
    """

    action_type: str = Field(
        ...,
        description="One of: assign, unassign, form_team, disband_team, rest, noop",
    )
    member_id: Optional[str] = Field(None, description="Target member ID")
    task_id: Optional[str] = Field(None, description="Target project ID")
    team_members: Optional[List[str]] = Field(
        None, description="Member IDs for form_team"
    )
    team_id: Optional[str] = Field(None, description="Team ID for disband_team")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class TeamCollabObservation(_ObsBase):
    """What the agent sees after each step."""

    members: List[Dict[str, Any]] = Field(default_factory=list)
    projects: List[Dict[str, Any]] = Field(default_factory=list)
    teams: List[Dict[str, Any]] = Field(default_factory=list)
    current_step: int = 0
    max_steps: int = 0
    active_events: List[str] = Field(default_factory=list)
    last_action_error: Optional[str] = None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class TeamCollabState(_StateBase):
    """Episode-level metadata."""

    task_name: str = ""
    total_reward: float = 0.0
    projects_completed: int = 0
    projects_failed: int = 0
    burnout_incidents: int = 0
    events_triggered: int = 0
