---
title: Team Collab Environment
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
---

# Team Collaboration Environment

An RL OpenEnv environment simulating software team coordination under constraints. The simulation models resource management variables like burnout, discrete task dependencies, and collaborative skill mapping.

## Overview

Unlike standard grid-worlds, this environment isolates the continuous decision-problem of resource allocation. Agents must prioritize complex dependency chains, regulate team energy levels to prevent burnout, and triage failing projects to maximize overall completion rates.

## Setup

```bash
pip install -e .
```

To run the full stack:
```bash
# 1. Start backend server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 2. Run inference baseline (LLM required)
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="<your_api_key>"
python inference.py
```

## RL Mechanics

### Agent Control (Manager)
The agent acts as the orchestrator. Each step, it submits a single management action, after which the environment steps forward and assigned members perform work probabilistically.

Supported actions (JSON payload to `/step`):
- `assign`: Allocate ID to task.
- `unassign`: Remove ID from task.
- `form_team`: Link IDs for a `1.3x` synergy bonus.
- `disband_team`: Sever team link.
- `rest`: Member halts work to regenerate energy natively.
- `noop`: Pass the step.

### State Space
Entities exposed via `/state`:
1. **Members**: Attributes include `skills` (frontend, backend, design, devops), `energy` (0.0–1.0, sub-0.2 causes a 30% efficiency penalty), `work_style` (`lazy`, `normal`, `efficient`), and `reliability`.
2. **Projects**: Contains `dependencies` (blocked until unblocked downstream), `difficulty` multiplier, `deadline`, and `status`.

If a baseline dependency fails, cascading penalties apply to downstream projects (+3 deadline extension, +0.2 difficulty). 

The environment emits discrete events per step (e.g. bugs, scope creep) seeded deterministically.

## Benchmarks

The inference suite evaluates the agent across 3 progressive difficulty tracks:

1. `solo_sprint` (Easy): 4 members, 1 isolated project. No dependencies. Evaluate baseline API capabilities.
2. `team_crunch` (Medium): 4 members, 3 chained projects. Random events enabled at 50%.
3. `deadline_hell` (Hard): 4 members, 4 rigid dependency graphs. Tight deadlines and full stochastic events enabled. Agent must exhibit robust long-term forecasting.

## Architecture Guidelines

This project exposes exact OpenEnv endpoints natively mapping to Gymnasium abstractions:

- `/reset`: Triggers `env.reset(task_name)`
- `/step`: Processes JSON action → discrete state delta
- `/state`: Serializes environment data
- `inference.py`: Baseline client demonstrating HF, OpenAI, and internal heuristic fallbacks.

## Docker Deployment

To wrap the environment for Space deployment:

```bash
docker build -t team-collab-env .
docker run -p 8000:8000 team-collab-env
```

## License
MIT
