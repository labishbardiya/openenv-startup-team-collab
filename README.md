# Team Collaboration Environment

An RL OpenEnv environment simulating software team coordination under constraints. The simulation models resource management variables like burnout, discrete task dependencies, and collaborative skill mapping.

## Overview

Unlike standard grid-worlds, this environment isolates the continuous decision-problem of resource allocation. Agents must prioritize complex dependency chains, regulate team energy levels to prevent burnout, and triage failing projects to maximize overall completion rates.

## Environment Details

### Action Space

The agent acts as a team manager. Each step, it submits a single action:

| Action | Description |
|---|---|
| `assign` | Allocate a team member to a project |
| `unassign` | Remove a member from their current project |
| `form_team` | Group members for a 1.3× synergy bonus |
| `disband_team` | Break up a team |
| `rest` | Let a member recover energy |
| `noop` | Do nothing |

### Observation Space

Each step returns:
- **Members**: `skills`, `energy` (0.0–1.0), `work_style`, `reliability`, `current_task_id`
- **Projects**: `progress`, `status`, `dependencies`, `difficulty`, `deadline`
- **Teams**: Active team groupings
- **Events**: Random disruptions (bugs, scope creep, breakthroughs)
- **Reward**: Dense per-step reward signal
- **Done**: Whether the episode has ended

### Reward Function

The environment provides dense reward shaping every step:
- **+2.0 × progress**: Per unit of project progress made
- **+5.0**: Completing a project
- **+1.0**: Unblocking a dependent project
- **+0.5**: Team synergy bonus when >2 members collaborate
- **+0.3**: Smart rest (resting when energy < 0.3)
- **-1.0**: Working a member with dangerously low energy
- **-0.2**: Leaving a member idle
- **-3.0**: Missing a deadline
- **+10.0**: Bonus for completing ALL projects

## Tasks (Easy → Medium → Hard)

| Task | Difficulty | Members | Projects | Steps | Events |
|---|---|---|---|---|---|
| `solo_sprint` | 🟢 Easy | 4 | 1 | 15 | Off |
| `team_crunch` | 🟡 Medium | 4 | 3 (chained) | 20 | 50% |
| `deadline_hell` | 🔴 Hard | 4 | 4 (cascading deps) | 25 | 100% |

Each task has a deterministic grader returning scores in [0.0, 1.0]:
- **solo_sprint**: Score = project progress (simple)
- **team_crunch**: Score = 70% completion rate + 30% average progress
- **deadline_hell**: Score = 50% completion + 25% avg progress + 25% energy management

## Setup

```bash
pip install -e .
```

### Run locally

```bash
# Start the backend server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal, run inference
export API_BASE_URL="https://router.huggingface.co/v1"
export API_KEY="<your_api_key>"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

### Docker Deployment

```bash
docker build -t team-collab-env .
docker run -p 7860:7860 team-collab-env
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/` | Root info |
| POST | `/reset` | Start new episode (`{"task_name": "solo_sprint"}`) |
| POST | `/step` | Execute action (`{"action": {"action_type": "assign", ...}}`) |
| GET | `/state` | Get episode metadata |
| GET | `/schema` | JSON schemas for models |
| GET | `/grade` | Current task grader score |
| WS | `/ws` | WebSocket interface |

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint (injected by hackathon) | ✅ |
| `API_KEY` | API key (injected by hackathon) | ✅ |
| `MODEL_NAME` | Model identifier | ✅ |
| `HF_TOKEN` | Fallback API key for local testing | Optional |

## Project Structure

```
openenv-startup-team-collab/
├── __init__.py
├── models.py           ← Pydantic models (Action, Observation, State)
├── client.py           ← Async HTTP client
├── openenv.yaml        ← OpenEnv manifest
├── pyproject.toml      ← Package config
├── Dockerfile          ← Docker deployment
├── inference.py        ← Baseline inference script
├── README.md           ← This file
└── server/
    ├── __init__.py
    ├── app.py           ← FastAPI endpoints
    ├── environment.py   ← Core environment logic
    ├── simulation.py    ← Simulation engine
    ├── tasks.py         ← Task configs + graders
    └── rewards.py       ← Dense reward shaping
```

## License

MIT
