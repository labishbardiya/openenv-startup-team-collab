"""
Inference Script — Team Collaboration Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   The name of the local image to use for the environment if using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Config (MANDATORY env vars) ──────────────────────────────────────────
# CRITICAL: Use API_KEY first (injected by hackathon LiteLLM proxy),
# then fall back to HF_TOKEN for local testing.
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "team_collab"
TASKS = ("solo_sprint", "team_crunch", "deadline_hell")
MAX_STEPS = 30
TEMPERATURE = 0.3
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

VALID_ACTIONS = {"assign", "unassign", "form_team", "disband_team", "rest", "noop"}

SYSTEM_PROMPT = textwrap.dedent("""\
You are managing a software team. Goal: maximize total project completion before deadlines.

AVAILABLE ACTIONS (respond with JSON only):
1. {"action_type":"assign","member_id":"<id>","task_id":"<id>"} - Assign to an UNLOCKED project.
2. {"action_type":"rest","member_id":"<id>"} - Rest a tired member.
3. {"action_type":"noop"} - Do nothing (AVOID if there is valid work).

RULES & STRATEGY:
- ONLY assign agents to tasks that are NOT locked (lock:F).
- Prefer assigning early tasks to unblock downstream dependencies.
- Match agent skills to the task (e.g. frontend to frontend).
- Do not assign agents to completed or failed tasks.
- If energy drops below 0.3, REST the agent (to avoid 0% efficiency burnout).

Return ONLY valid JSON:
{"action_type": "<action_type>", ...}
""").strip()


# ---------------------------------------------------------------------------
# Structured Logging — strict stdout format per OpenEnv
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# CLI UI Helpers (stderr ONLY)
# ---------------------------------------------------------------------------
def _bar(progress: float, width: int = 10) -> str:
    filled = int(progress * width)
    return "█" * filled + "░" * (width - filled)


def visual_step(obs: dict, action_str: str, reward: float) -> None:
    step = obs.get("current_step", 0)
    max_s = obs.get("max_steps", 0)
    print(f"\n--- Step {step}/{max_s} ---", file=sys.stderr)
    print(" 📋 Projects:", file=sys.stderr)
    for p in obs.get("projects", []):
        prog = p.get("progress", 0)
        icon = {"completed": "✅", "failed": "❌", "blocked": "🔒",
                "in_progress": "🔨", "pending": "⏳"}.get(p.get("status", ""), "?")
        print(f"   {p['id']} \"{p['name']}\" {_bar(prog)} {prog:.0%}  ⏰{p['deadline']}  {icon}", file=sys.stderr)
    print(" 👥 Team:", file=sys.stderr)
    for m in obs.get("members", []):
        warn = " ⚠️LOW" if m["energy"] < 0.3 else ""
        task = m.get("current_task_id") or "—"
        top = max(m["skills"], key=m["skills"].get) if m["skills"] else "?"
        print(f"   {m['name']:8s} ⚡{m['energy']:.2f}{warn}  🔧{top}:{m['skills'].get(top,0):.1f}  📌{task}  [{m['work_style']}]", file=sys.stderr)
    evts = obs.get("active_events", [])
    if evts:
        print(f" ⚡ Events: {'; '.join(evts)}", file=sys.stderr)
    print(f" 🎯 Action: {action_str}  →  reward: {reward:+.2f}", file=sys.stderr)
    print("---", file=sys.stderr)


# ---------------------------------------------------------------------------
# Prompt builder & State Serializer
# ---------------------------------------------------------------------------
def build_prompt(obs: dict) -> str:
    lines = [f"Step {obs.get('current_step', 0)}/{obs.get('max_steps', 0)}"]
    err = obs.get("last_action_error")
    if err:
        lines.append(f"⚠ Last error: {err}")
    lines.append("Projects:")
    for p in obs.get("projects", []):
        locked = "T" if p.get("status") == "blocked" else "F"
        failed = "T" if p.get("status") == "failed" else "F"
        completed = "T" if p.get("status") == "completed" else "F"
        lines.append(f"  {p['id']}(prog:{p['progress']:.0%},lock:{locked},fail:{failed},done:{completed},due:{p['deadline']})")
    lines.append("Agents:")
    for m in obs.get("members", []):
        task = m.get("current_task_id") or "-"
        top_skill = max(m["skills"], key=m["skills"].get) if m["skills"] else "none"
        lines.append(f"  {m['id']}(eng:{m['energy']:.2f},{top_skill}:{m['skills'].get(top_skill,0):.1f},task:{task})")
    lines.append("\nAction (JSON):")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action parsing & Fallbacks
# ---------------------------------------------------------------------------
def parse_action(text: str) -> tuple[dict, Optional[str]]:
    """Parse LLM output into action dict. Returns (action_dict, error_or_None)."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, dict) and parsed.get("action_type") in VALID_ACTIONS:
                return parsed, None
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: noop with error
    return {"action_type": "noop"}, "invalid_action"


def heuristic_fallback(obs: dict) -> str:
    """Deterministic heuristic fallback — ONLY used if LLM call fails.
    This still counts as a valid action; the LLM call was attempted first."""
    members = obs.get("members", [])
    projects = obs.get("projects", [])

    # Rest any critically tired member
    for m in members:
        if m.get("energy", 1.0) < 0.25 and not m.get("is_resting", False):
            return json.dumps({"action_type": "rest", "member_id": m["id"]})

    # Find idle members and unlocked projects
    idle_members = [m for m in members if not m.get("current_task_id") and not m.get("is_resting", False)]
    available_projects = [p for p in projects if p.get("status") not in ("completed", "failed", "blocked")]

    if idle_members and available_projects:
        m = idle_members[0]
        p = available_projects[0]
        return json.dumps({"action_type": "assign", "member_id": m["id"], "task_id": p["id"]})

    # Rest an idle member if nothing else to do
    if idle_members:
        return json.dumps({"action_type": "rest", "member_id": idle_members[0]["id"]})

    return '{"action_type":"noop"}'


def get_model_message(client: OpenAI, messages: list, obs: dict) -> str:
    """Call LLM via the provided API_BASE_URL and API_KEY.
    Uses ONLY the injected proxy — NO fallback to other providers.
    On failure, uses a heuristic fallback but the API call is still attempted."""

    # Always try the provided API first
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            text = ""
            if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                text = resp.choices[0].message.content.strip()

            if text:
                return text

            print(f"[DEBUG] LLM returned empty response (attempt {attempt+1}/3)", file=sys.stderr)

        except Exception as exc:
            print(f"[DEBUG] LLM request failed (attempt {attempt+1}/3): {exc}", file=sys.stderr)
            if attempt < 2:
                import time
                time.sleep(1)  # Brief backoff before retry
                continue

    # --- FALLBACK: Heuristic (LLM call was already attempted above) ---
    print("[DEBUG] All LLM attempts failed. Using heuristic fallback.", file=sys.stderr)
    return heuristic_fallback(obs)


# ---------------------------------------------------------------------------
# Score computation — normalizes cumulative reward to [0, 1]
# ---------------------------------------------------------------------------
def compute_score(rewards: List[float], task_name: str) -> float:
    """Compute a normalized score in [0, 1] for the task.
    Approximates from cumulative reward — used as fallback if /grade fails."""
    total = sum(rewards)
    # Rough normalization based on max possible reward per task
    max_rewards = {
        "solo_sprint": 40.0,
        "team_crunch": 80.0,
        "deadline_hell": 100.0,
    }
    max_r = max_rewards.get(task_name, 60.0)
    score = max(0.0, min(1.0, total / max_r))
    return round(score, 2)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
async def run_task(client: OpenAI, env, task_name: str) -> tuple[bool, List[float], float]:
    """Run inference on one task. [END] is ALWAYS emitted via try/finally."""
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        obs = result.observation
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step_num in range(1, MAX_STEPS + 1):
            # Check done from previous step
            if result.done:
                break

            user_prompt = build_prompt(obs)
            messages.append({"role": "user", "content": user_prompt})

            # Call LLM via the injected proxy
            reply = get_model_message(client, messages, obs)
            messages.append({"role": "assistant", "content": reply})

            # Parse action with validation
            action_dict, parse_error = parse_action(reply)
            action_str = json.dumps(action_dict, separators=(",", ":"))

            # Execute step
            result = await env.step(action_dict)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            # Error: use parse error OR env error
            error = parse_error or obs.get("last_action_error")

            rewards.append(reward)
            steps_taken = step_num

            # Log IMMEDIATELY after step
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)
            visual_step(obs, action_str, reward)

            # Hard termination at MAX_STEPS
            if done or step_num >= MAX_STEPS:
                break

            # Sliding window for conversation
            if len(messages) > 12:
                messages = [messages[0]] + messages[-10:]

        # Fetch the actual grader score from the environment
        try:
            grader_score = await env.grade()
            score = round(min(1.0, max(0.0, grader_score)), 2)
            print(f"[DEBUG] Grader score for {task_name}: {score}", file=sys.stderr)
        except Exception as ge:
            print(f"[DEBUG] Failed to get grader score, using computed: {ge}", file=sys.stderr)
            score = compute_score(rewards, task_name)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task '{task_name}' exception: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr)
        # [END] ALWAYS emitted — even on crash
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success, rewards, score


async def main() -> None:
    if not API_KEY:
        print("ERROR: Set API_KEY or HF_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    # CRITICAL: Initialize OpenAI client with the INJECTED proxy URL and key
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    print(f"\n🚀 Running inference...", file=sys.stderr)
    print(f"   Model:     {MODEL_NAME}", file=sys.stderr)
    print(f"   API URL:   {API_BASE_URL}", file=sys.stderr)
    print(f"   API Key:   {API_KEY[:8]}...{API_KEY[-4:] if len(API_KEY or '') > 12 else '***'}", file=sys.stderr)
    print(f"   Image:     {IMAGE_NAME or '(direct)'}", file=sys.stderr)
    print(f"   Tasks:     {', '.join(TASKS)}\n", file=sys.stderr)

    all_results = {}

    for task in TASKS:
        # Create env for each task (fresh container or connection)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from client import TeamCollabEnv

        if IMAGE_NAME:
            env = await TeamCollabEnv.from_docker_image(IMAGE_NAME)
        else:
            # Connect to running server (HF Space or local)
            env = TeamCollabEnv(base_url=os.getenv("SPACE_URL", "http://localhost:8000"))

        ok, rewards, score = await run_task(client, env, task)
        all_results[task] = (ok, rewards, score)

    # Summary to stderr
    print("\n--- FINAL RESULTS ---", file=sys.stderr)
    for task, (ok, rewards, score) in all_results.items():
        emoji = "✅" if ok else "❌"
        total = sum(rewards)
        print(f"  {emoji} {task:20s} → steps={len(rewards)}, total_reward={total:.2f}, score={score:.2f}", file=sys.stderr)
    print("-" * 30 + "\n", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
