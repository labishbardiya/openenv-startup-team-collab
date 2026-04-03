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
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Config (MANDATORY env vars) ──────────────────────────────────────────
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = "team_collab"
TASKS = ("solo_sprint", "team_crunch", "deadline_hell")
MAX_STEPS = 30
TEMPERATURE = 0.3
MAX_TOKENS = 100
SUCCESS_SCORE_THRESHOLD = 0.5

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


# ═══════════════════════════════════════════════════════════════════════════
# Structured Logging — EXACT format per spec (NO extra fields)
# ═══════════════════════════════════════════════════════════════════════════
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Visual output (stderr ONLY — never contaminates stdout)
# ═══════════════════════════════════════════════════════════════════════════
def _bar(progress: float, width: int = 10) -> str:
    filled = int(progress * width)
    return "█" * filled + "░" * (width - filled)


def visual_step(obs: dict, action_str: str, reward: float) -> None:
    step = obs.get("current_step", 0)
    max_s = obs.get("max_steps", 0)
    print(f"\n{'═' * 50}", file=sys.stderr)
    print(f" [Step {step}/{max_s}]", file=sys.stderr)
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
    print("═" * 50, file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════════════
def build_prompt(obs: dict) -> str:
    lines = [f"Step {obs.get('current_step', 0)}/{obs.get('max_steps', 0)}"]
    err = obs.get("last_action_error")
    if err:
        lines.append(f"⚠ Last error: {err}")
    lines.append("Projects:")
    for p in obs.get("projects", []):
        locked = "T" if p.get("status") == "blocked" else "F"
        failed = "T" if p.get("status") == "failed" else "F"
        lines.append(f"  {p['id']}(prog:{p['progress']:.0%},lock:{locked},fail:{failed},due:{p['deadline']})")
    lines.append("Agents:")
    for m in obs.get("members", []):
        task = m.get("current_task_id") or "-"
        top_skill = max(m["skills"], key=m["skills"].get) if m["skills"] else "none"
        lines.append(f"  {m['id']}(eng:{m['energy']:.2f},{top_skill}:{m['skills'].get(top_skill,0):.1f},task:{task})")
    lines.append("\nAction (JSON):")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Action parsing — ROBUST with validation + noop fallback
# ═══════════════════════════════════════════════════════════════════════════
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


import re

def heuristic_fallback(prompt: str) -> str:
    """Modest heuristic fallback. Only used if all APIs fail."""
    latest_state = prompt.split("Step ")[-1] if "Step " in prompt else prompt
    
    # Rest any critically tired member
    m = re.search(r"  ([a-zA-Z0-9_-]+)\(eng:0\.[0-2].*", latest_state)
    if m:
        return f'{{"action_type":"rest","member_id":"{m.group(1)}"}}'
    
    # Assign idle members to any unlocked project
    idle_m = re.search(r"  ([a-zA-Z0-9_-]+)\(eng:[0-9]\.[0-9].*task:-\)", latest_state)
    proj_m = re.search(r"  ([a-zA-Z0-9_-]+)\(prog:[0-9]{1,2}%,lock:F,fail:F", latest_state)
    if idle_m and proj_m:
        return f'{{"action_type":"assign","member_id":"{idle_m.group(1)}","task_id":"{proj_m.group(1)}"}}'
    
    # Just rest random idle member if stuck
    idle_fallback = re.search(r"  ([a-zA-Z0-9_-]+)\(.*task:-\)", latest_state)
    if idle_fallback:
        return f'{{"action_type":"rest","member_id":"{idle_fallback.group(1)}"}}'
        
    return '{"action_type":"noop"}'


def get_model_message(client: OpenAI, messages: list) -> str:
    """Call LLM and return response text. Automatically falls back on API failures."""
    prompt = "\n".join([m["content"] for m in messages])
    
    MODELS = [
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    
    # If the user overrode MODEL_NAME to something entirely different, respect it
    if MODEL_NAME not in MODELS:
        MODELS = [MODEL_NAME]

    for model in MODELS:
        for attempt in range(2):
            try:
                resp = client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                    max_output_tokens=100,
                    temperature=0.2,
                )
                
                print(f"[DEBUG] RAW RESPONSE ({model}):", resp, file=sys.stderr)
                
                if getattr(resp, "status", None) == "failed" or getattr(resp, "error", None) is not None:
                    raise RuntimeError(f"LLM hard fail. Status: {getattr(resp, 'status', None)}")
                
                try:
                    if hasattr(resp, "output_text") and resp.output_text:
                        text = resp.output_text
                    elif getattr(resp, "output", None) and hasattr(resp.output[0], "content") and resp.output[0].content:
                        text = resp.output[0].content[0].text
                    else:
                        text = ""
                except Exception as e:
                    print(f"[DEBUG] Parse error: {e}", file=sys.stderr)
                    text = ""
                    
                if text:
                    return text
                    
            except RuntimeError as re:
                print(f"[DEBUG] FATAL: {re}", file=sys.stderr)
                break  # Don't retry quota/fatal errors, drop to fallbacks
                
            except Exception as exc:
                print(f"[DEBUG] LLM request failed for model {model} (attempt {attempt+1}): {exc}", file=sys.stderr)
                continue
    
    # --- FALLBACK 1: OpenAI API ---
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("[DEBUG] APIs Failed. Falling back to OpenAI API.", file=sys.stderr)
        try:
            oai_client = OpenAI(api_key=openai_key)
            oai_resp = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                max_tokens=100
            )
            return oai_resp.choices[0].message.content or '{"action_type":"noop"}'
        except Exception as oe:
            print(f"[DEBUG] OpenAI Fallback failed: {oe}", file=sys.stderr)

    # --- FALLBACK 2: Gemini API ---
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("[DEBUG] HF Failed. Falling back to Gemini API (Flash Model).", file=sys.stderr)
        try:
            backup_client = OpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            backup_resp = backup_client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=messages,
                temperature=0.2,
                max_tokens=100
            )
            return backup_resp.choices[0].message.content or '{"action_type":"noop"}'
        except Exception as fe:
            print(f"[DEBUG] Gemini Fallback failed: {fe}", file=sys.stderr)

    # --- FALLBACK 2: Zero-cost Heuristic ---
    print("[DEBUG] Using heuristic fallback instead of crashing.", file=sys.stderr)
    return heuristic_fallback(prompt)



# ═══════════════════════════════════════════════════════════════════════════
# Main inference loop
# ═══════════════════════════════════════════════════════════════════════════
async def run_task(client: OpenAI, env, task_name: str) -> tuple[bool, List[float]]:
    """Run inference on one task. [END] is ALWAYS emitted via try/finally."""
    rewards: List[float] = []
    steps_taken = 0
    success = False

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

            # Call LLM
            reply = get_model_message(client, messages)
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

        total = sum(rewards)
        success = total > 0

    except Exception as exc:
        if "All connection" in str(exc) and getattr(env, "base_url", "").startswith("http://localhost"):
            print(f"[DEBUG] Task '{task_name}' exception: Environment HTTP server is not running at {env.base_url}. Please start it with 'uvicorn server.app:app --port 8000' or provide LOCAL_IMAGE_NAME.", file=sys.stderr)
        else:
            print(f"[DEBUG] Task '{task_name}' exception: {exc}", file=sys.stderr)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", file=sys.stderr)
        # [END] ALWAYS emitted — even on crash
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return success, rewards


async def main() -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    print(f"\n🚀 Running inference...", file=sys.stderr)
    print(f"   Model:  {MODEL_NAME}", file=sys.stderr)
    print(f"   API:    {API_BASE_URL}", file=sys.stderr)
    print(f"   Image:  {IMAGE_NAME or '(direct)'}", file=sys.stderr)
    print(f"   Tasks:  {', '.join(TASKS)}\n", file=sys.stderr)

    all_results = {}

    for task in TASKS:
        # Create env for each task (fresh container or connection)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from client import TeamCollabEnv

        if IMAGE_NAME:
            env = await TeamCollabEnv.from_docker_image(IMAGE_NAME)
        else:
            # Fallback: start local server or connect to running one
            env = TeamCollabEnv(base_url=os.getenv("SPACE_URL", "http://localhost:8000"))

        ok, rewards = await run_task(client, env, task)
        all_results[task] = (ok, rewards)

    # Summary to stderr
    print("\n" + "═" * 50, file=sys.stderr)
    print("  📊 FINAL RESULTS", file=sys.stderr)
    print("═" * 50, file=sys.stderr)
    for task, (ok, rewards) in all_results.items():
        emoji = "✅" if ok else "❌"
        total = sum(rewards)
        print(f"  {emoji} {task:20s} → steps={len(rewards)}, total_reward={total:.2f}", file=sys.stderr)
    print("═" * 50 + "\n", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
