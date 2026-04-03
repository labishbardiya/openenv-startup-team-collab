# 🏢 Team Collaboration Environment

**Simulates real startup team coordination under deadlines.**
**Models burnout, task dependencies, and collaboration dynamics.**
**Trains agents to make human-like resource management decisions.**

---

## Why This Matters

Every startup faces the same problem: too many projects, too few people, too little time. Bad decisions — overworking your best engineer, ignoring dependencies, skipping rest — cascade into missed deadlines and failed launches.

This environment captures that reality. It's not a toy grid-world. It's not a game. It's the actual decision problem that engineering managers face daily — and it's almost entirely absent from existing RL benchmarks.

An agent trained here learns to:
- **Prioritize** dependency chains (backend before frontend)
- **Manage energy** (rest before burnout)
- **Collaborate** (form teams for synergy bonuses)
- **Triage** (accept strategic failures to save critical projects)

---

## Quick Start

```bash
# Install
pip install -e .

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference
HF_TOKEN=your_token python inference.py
```

---

## Environment Mechanics

### Team Members
Each member has:
| Attribute | Description |
|-----------|-------------|
| `skills` | Proficiency per domain (0.0–1.0): frontend, backend, design, devops |
| `energy` | 0.0–1.0. Below 0.2 = **burnout** (30% efficiency) |
| `work_style` | `lazy` (0.6×), `normal` (1.0×), or `efficient` (1.3× speed) |
| `reliability` | Probability that a work action produces output |

### Projects (with Dependencies)
| Attribute | Description |
|-----------|-------------|
| `dependencies` | Projects that must complete before this one can start |
| `difficulty` | 0.5–2.0, affects work speed and energy drain |
| `deadline` | Step number — miss it and the project fails |
| `status` | `blocked` → `pending` → `in_progress` → `completed` / `failed` |

**Cascade rule:** If a dependency fails, all downstream projects get +3 step deadline extension and +0.2 difficulty increase.

### Random Events
| Event | Probability | Effect |
|-------|-------------|--------|
| 🐛 Bug discovered | 8%/project | Lose 15% progress |
| 🤒 Member unavailable | 5%/member | Skip 1 step |
| 📈 Scope creep | 3%/project | +0.3 difficulty |
| 💡 Breakthrough | 4%/project | +10% progress |

Events are seeded for reproducibility.

---

## Action Space

Each step, the agent takes **one management action**. All assigned members then auto-work.

| Action | Parameters | Description |
|--------|-----------|-------------|
| `assign` | `member_id`, `task_id` | Assign member to project |
| `unassign` | `member_id` | Remove assignment |
| `form_team` | `team_members[]` | Group for 1.3× collaboration bonus |
| `disband_team` | `team_id` | Break up team |
| `rest` | `member_id` | Member recovers energy instead of working |
| `noop` | — | Skip (everyone works as assigned) |

---

## Observation Space

```json
{
  "members": [{"id": "alice", "energy": 0.7, "skills": {...}, ...}],
  "projects": [{"id": "p1", "progress": 0.6, "status": "in_progress", ...}],
  "teams": [{"id": "team_1", "member_ids": ["alice", "bob"]}],
  "current_step": 5,
  "max_steps": 20,
  "active_events": ["🐛 Bug in 'Backend API': lost 15% progress"],
  "reward": 0.32,
  "done": false
}
```

---

## Tasks

### 🟢 `solo_sprint` (Easy)
- 4 members, 1 project, no dependencies, no random events
- 15 max steps, difficulty 0.8
- **Goal:** Complete the single project
- **Grader:** `score = project.progress`

### 🟡 `team_crunch` (Medium)
- 4 members (mixed personalities), 3 projects with dependency chain (P1→P2)
- 20 max steps, random events at 50%
- **Goal:** Complete all projects respecting dependencies
- **Grader:** `completion_rate × 0.7 + avg_progress × 0.3`

### 🔴 `deadline_hell` (Hard)
- 4 members (varied energy, reliability), 4 projects with deep dependency chain (P1→P2→P4)
- 25 max steps, tight deadlines, full random events
- **Goal:** Maximize completions under extreme pressure
- **Grader:** `completion × 0.5 + efficiency × 0.25 + energy_mgmt × 0.25`

---

## Reward System

Dense, continuous signal every step:

| Event | Reward |
|-------|--------|
| Progress on project | `+Δ × 2.0` |
| Project completed | `+5.0` |
| Dependency unblocked | `+1.0` |
| Team synergy work | `+0.5` |
| Smart rest (energy < 0.3) | `+0.3` |
| Burnout work (energy < 0.2) | `-1.0` |
| Idle member | `-0.2` |
| Deadline missed | `-3.0` |
| Cascade failure | `-1.5` |
| All projects completed | `+10.0` |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              inference.py                    │
│         (LLM agent via OpenAI)               │
└──────────────┬──────────────────────────────┘
               │ TeamCollabAction
┌──────────────▼──────────────────────────────┐
│          TeamCollabEnvironment               │
│   reset() │ step() │ state │ grade()         │
├──────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │Simulation│ │ Rewards  │ │ Tasks+Graders│ │
│  │ Engine   │ │ Shaping  │ │ (3 levels)   │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
├──────────────────────────────────────────────┤
│              FastAPI Server                  │
│    /reset  /step  /state  /health  /ws       │
└──────────────────────────────────────────────┘
```

---

## Baseline Scores

| Task | Score | Steps |
|------|-------|-------|
| solo_sprint | ~0.9–1.0 | 6–8 |
| team_crunch | ~0.6–0.8 | 15–20 |
| deadline_hell | ~0.3–0.5 | 20–25 |

---

## Setup & Deployment

```bash
# Docker
docker build -t team-collab-env .
docker run -p 8000:8000 team-collab-env

# Hugging Face Space — Dockerized
# Tag with openenv, deploy to HF Spaces
```

### Environment Variables
| Variable | Required | Default |
|----------|---------|---------|
| `API_BASE_URL` | Optional | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Optional | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | **Yes** (for inference) | — |

---

## License

MIT
