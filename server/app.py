"""
FastAPI server for the Team Collaboration Environment.

Exposes HTTP + WebSocket endpoints compatible with OpenEnv spec.
All responses include: observation, reward, done, info.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from models import TeamCollabAction, TeamCollabObservation, TeamCollabState
from server.environment import TeamCollabEnvironment

# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Team Collaboration Environment",
    description="OpenEnv RL environment simulating startup team coordination",
    version="1.0.0",
)

# Single-instance environment for HTTP endpoints
_env = TeamCollabEnvironment()


# ── Request / Response models ─────────────────────────────────────────────
class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_name: str = "solo_sprint"


class StepRequest(BaseModel):
    action: Dict[str, Any]


# ── HTTP endpoints ────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: Request):
    """Reset the environment. Accepts empty body {} or ResetRequest fields."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    req = ResetRequest(**body)
    obs = _env.reset(
        seed=req.seed,
        episode_id=req.episode_id,
        task_name=req.task_name,
    )
    obs_data = obs.model_dump()
    return {
        "observation": obs_data,
        "reward": obs.reward,
        "done": obs.done,
        "info": obs_data.get("metadata", {}),
    }


@app.post("/step")
async def step(req: StepRequest):
    """Execute one step. Returns observation, reward, done, info."""
    try:
        action = TeamCollabAction(**req.action)
    except Exception as e:
        return {
            "observation": {},
            "reward": 0.0,
            "done": False,
            "info": {"error": f"Invalid action: {e}"},
        }
    obs = _env.step(action)
    obs_data = obs.model_dump()
    return {
        "observation": obs_data,
        "reward": obs.reward,
        "done": obs.done,
        "info": obs_data.get("metadata", {}),
    }


@app.get("/state")
async def get_state():
    """Return current environment state."""
    return _env.state.model_dump()


@app.get("/schema")
async def schema():
    """Return JSON schemas for action, observation, and state."""
    return {
        "action": TeamCollabAction.model_json_schema(),
        "observation": TeamCollabObservation.model_json_schema(),
        "state": TeamCollabState.model_json_schema(),
    }


@app.get("/grade")
async def grade():
    """Return the deterministic grader score for the current task."""
    return {"score": _env.grade(), "task": _env._task_name}


# ── WebSocket endpoint ───────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    env = TeamCollabEnvironment()
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type", "")

            if msg_type == "reset":
                data = msg.get("data", {})
                obs = env.reset(**data)
                obs_data = obs.model_dump()
                await ws.send_json({
                    "type": "observation",
                    "data": {
                        "observation": obs_data,
                        "reward": obs.reward,
                        "done": obs.done,
                        "info": obs_data.get("metadata", {}),
                    },
                })
            elif msg_type == "step":
                data = msg.get("data", {})
                try:
                    action = TeamCollabAction(**data)
                except Exception as e:
                    await ws.send_json({
                        "type": "error",
                        "data": {"message": f"Invalid action: {e}"},
                    })
                    continue
                obs = env.step(action)
                obs_data = obs.model_dump()
                await ws.send_json({
                    "type": "observation",
                    "data": {
                        "observation": obs_data,
                        "reward": obs.reward,
                        "done": obs.done,
                        "info": obs_data.get("metadata", {}),
                    },
                })
            elif msg_type == "state":
                await ws.send_json({
                    "type": "state",
                    "data": env.state.model_dump(),
                })
            elif msg_type == "close":
                env.close()
                await ws.close()
                break
            else:
                await ws.send_json({
                    "type": "error",
                    "data": {"message": f"Unknown type: {msg_type}"},
                })
    except WebSocketDisconnect:
        env.close()
    except Exception as e:
        try:
            await ws.send_json({
                "type": "error",
                "data": {"message": str(e), "traceback": traceback.format_exc()},
            })
        except Exception:
            pass
        env.close()


# ── Direct run ────────────────────────────────────────────────────────────
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
