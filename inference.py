#!/usr/bin/env python3
"""
Baseline inference script for the DarkStore Inbound Environment.

Runs an LLM agent against all 5 tasks and reports scores.

Setup (uses Hugging Face router — no OpenAI key needed):
    Get a free HF token at https://huggingface.co/settings/tokens, then:

        set HF_TOKEN=your_hf_token_here
        set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct   # optional, has a default
        python inference.py

    Or set env vars in .env file:
        HF_TOKEN=your_hf_token_here
        MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Auto-load .env if it exists
# ---------------------------------------------------------------------------

_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_file):
    with open(_env_file, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL_NAME") or os.getenv("MODEL", "openai/gpt-oss-120b:novita")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

TASKS = [
    {"name": "clean_delivery",       "seed": 42, "difficulty": "easy"},
    {"name": "quantity_mismatch",     "seed": 42, "difficulty": "medium"},
    {"name": "hidden_violation",      "seed": 42, "difficulty": "hard"},
    {"name": "price_discrepancy",     "seed": 42, "difficulty": "medium_hard"},
    {"name": "multi_violation_chaos", "seed": 42, "difficulty": "expert"},
]

MAX_TOKENS = 1024
VERBOSE = True

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a receiving dock agent at a dark store (quick-commerce warehouse).
A supplier delivery has arrived. Your job is to inspect it thoroughly and decide what to accept, flag, or reject.

## YOUR WORKFLOW

Step 1 — Gather all available data before making any decisions:
  - Request the supplier invoice
  - Request the physical scan data
  - Request the cold chain temperature log

Step 2 — For each SKU in pending_skus, make exactly one decision:
  - accept_sku: item is fine, accept it
  - flag_shortage: fewer units arrived than ordered (provide the shortage_qty)
  - reject_sku: item fails inspection (provide the rejection reason)

Step 3 — When all pending_skus are empty, call finalize.

## VALID REJECTION REASONS
"damaged" | "expired" | "shelf_life_violation" | "cold_chain_violation" | "unauthorized_substitution"

## OUTPUT FORMAT
Respond with exactly one JSON action per turn. No text, no explanation, just JSON.

Examples:
{"action_type": "request_invoice"}
{"action_type": "request_scan"}
{"action_type": "request_cold_chain"}
{"action_type": "accept_sku", "sku_id": "SKU001"}
{"action_type": "flag_shortage", "sku_id": "SKU002", "shortage_qty": 5}
{"action_type": "reject_sku", "sku_id": "SKU003", "reason": "damaged"}
{"action_type": "finalize"}
"""

# ---------------------------------------------------------------------------
# Format observation as text for LLM
# ---------------------------------------------------------------------------


def format_observation(obs: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 55)
    lines.append(f"Phase: {obs.get('phase')} | Steps left: {obs.get('steps_remaining')}")
    lines.append(f"Message: {obs.get('message', '')}")
    lines.append("")

    # Purchase Order
    po = obs.get("purchase_order", [])
    if po:
        lines.append("PURCHASE ORDER:")
        for item in po:
            lines.append(f"  {item['sku_id']}: {item['name']} | Qty ordered: {item['quantity']}")
        lines.append("")

    # Policy
    policy = obs.get("policy_rules")
    if policy:
        lines.append("POLICY RULES:")
        lines.append(f"  Min shelf life: {policy['min_shelf_life_days']} days")
        lines.append(f"  Max transit temp: {policy['max_transit_temp_celsius']} C")
        subs = policy.get("approved_substitutions", {})
        lines.append(f"  Approved substitutions: {subs if subs else 'NONE'}")
        lines.append("")

    # Invoice
    invoice = obs.get("invoice")
    if invoice is None:
        lines.append("INVOICE: [not yet requested — call request_invoice]")
    else:
        lines.append("INVOICE:")
        for item in invoice:
            lines.append(f"  {item['sku_id']}: {item['name']} | Qty invoiced: {item['quantity']}")
    lines.append("")

    # Scan
    scan = obs.get("scan_data")
    if scan is None:
        lines.append("SCAN DATA: [not yet requested — call request_scan]")
    else:
        lines.append("SCAN DATA:")
        for item in scan:
            lines.append(
                f"  {item['sku_id']}: Scanned qty: {item['scanned_qty']} | "
                f"Expiry: {item['expiry_date']} | Condition: {item['condition']}"
            )
    lines.append("")

    # Cold chain
    cold = obs.get("cold_chain_log")
    if cold is None:
        lines.append("COLD CHAIN LOG: [not yet requested — call request_cold_chain]")
    else:
        lines.append("COLD CHAIN LOG:")
        for r in cold:
            temp = r["temperature_celsius"]
            flag = " *** ABOVE LIMIT ***" if temp > 8.0 else ""
            lines.append(f"  {r['timestamp']}: {temp} C{flag}")
    lines.append("")

    pending = obs.get("pending_skus", [])
    resolved = obs.get("resolved_skus", [])
    lines.append(f"Pending (need decision): {pending}")
    lines.append(f"Resolved: {resolved}")
    if not pending:
        lines.append("All SKUs resolved. Call finalize.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parse LLM response to JSON action
# ---------------------------------------------------------------------------


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    # Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Any JSON object in text
    m = re.search(r"\{[^{}]+\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Play one full episode via WebSocket
# ---------------------------------------------------------------------------


def play_episode(client: OpenAI, env_url: str, task: Dict[str, Any]) -> Dict[str, Any]:
    import websockets.sync.client  # type: ignore

    task_name = task["name"]
    seed = task["seed"]
    difficulty = task["difficulty"]
    env_name = "darkstore_inbound"

    if VERBOSE:
        print(f"\n{'=' * 55}")
        print(f"Task: {task_name}  ({difficulty})")
        print("=" * 55)

    ws_url = env_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    step_count = 0
    final_score = 0.0
    rewards: List[float] = []
    end_emitted = False
    success = False

    # --- Structured output: START (always emitted before anything else) ---
    print(f"[START] task={task_name} env={env_name} model={MODEL}", flush=True)

    try:
        with websockets.sync.client.connect(ws_url) as ws:
            # Reset
            ws.send(json.dumps({"type": "reset", "data": {"task": task_name, "seed": seed}}))
            r = json.loads(ws.recv())
            obs = r["data"]["observation"]
            done = r["data"].get("done", False)

            sku_count = len(obs.get("purchase_order", []))
            pending = obs.get("pending_skus", [])
            if VERBOSE:
                print(f"Reset OK. {sku_count} SKUs to inspect. Pending: {pending}. Done: {done}")

            history: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

            while not done:
                step_count += 1
                obs_text = format_observation(obs)

                history.append({"role": "user", "content": obs_text})

                # Small delay to avoid rate limiting
                if step_count > 1:
                    time.sleep(1)

                # Call LLM (with retry on rate limit)
                raw = ""
                last_error: Optional[str] = None
                for attempt in range(2):
                    try:
                        resp = client.chat.completions.create(
                            model=MODEL,
                            messages=history,
                            max_tokens=MAX_TOKENS,
                            temperature=0.0,
                        )
                        raw = resp.choices[0].message.content or ""
                        break
                    except Exception as e:
                        err_msg = str(e).lower()
                        if "rate limit" in err_msg or "429" in err_msg:
                            if attempt == 0:
                                print(f"  Rate limited, waiting 10s...", flush=True)
                                time.sleep(10)
                            else:
                                print(f"  Rate limit persists, finalizing", flush=True)
                                raw = '{"action_type": "finalize"}'
                        else:
                            print(f"  LLM error: {e}", flush=True)
                            raw = '{"action_type": "finalize"}'
                            break

                action = parse_action(raw)
                if action is None:
                    if VERBOSE:
                        print(f"  Step {step_count}: parse failed, finalizing. Raw: {raw[:100]}")
                    action = {"action_type": "finalize"}

                action_str = action.get("action_type", "unknown")
                if VERBOSE:
                    summary = json.dumps(action)[:90]
                    print(f"  Step {step_count}: {summary}")

                history.append({"role": "assistant", "content": raw})

                # Step environment
                ws.send(json.dumps({"type": "step", "data": action}))
                r = json.loads(ws.recv())
                payload = r["data"]
                obs = payload["observation"]
                done = payload.get("done", False)
                reward = payload.get("reward", 0.0)
                error_msg = payload.get("last_action_error", None)

                rewards.append(reward)

                # --- Structured output: STEP ---
                done_str = str(done).lower()
                error_str = error_msg if error_msg else "null"
                print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

                if done:
                    final_score = reward
                    success = final_score > 0
                    if VERBOSE:
                        print(f"\n  Done! Score: {final_score}")
                        print(f"  {obs.get('message', '')}")

                # Trim history to avoid token explosion
                if len(history) > 24:
                    history = [history[0]] + history[-22:]

    finally:
        # --- Structured output: END (ALWAYS emitted, even on exception) ---
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(f"[END] success={str(success).lower()} steps={step_count} score={final_score:.2f} rewards={rewards_str}", flush=True)
        end_emitted = True

    return {"task": task_name, "difficulty": difficulty, "score": final_score, "steps": step_count}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not API_KEY:
        print("ERROR: No API key. Edit .env or set API_KEY / HF_TOKEN env var.")
        sys.exit(1)

    print(f"API: {API_BASE_URL}")
    print(f"Model: {MODEL}")
    print(f"Env: {ENV_URL}")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    results = []
    t0 = time.time()

    for i, task in enumerate(TASKS):
        if i > 0:
            print(f"\nWaiting 15s between tasks (rate limit buffer)...")
            time.sleep(15)
        try:
            results.append(play_episode(client, ENV_URL, task))
        except Exception as e:
            print(f"\nERROR in {task['name']}: {e}")
            results.append({"task": task["name"], "difficulty": task["difficulty"], "score": 0.0, "steps": 0})

    elapsed = time.time() - t0

    print(f"\n{'=' * 55}")
    print("SUMMARY")
    print(f"{'=' * 55}")
    print(f"{'Task':<25} {'Diff':<10} {'Score':<8} Steps")
    print("-" * 50)
    for r in results:
        print(f"{r['task']:<25} {r['difficulty']:<10} {r['score']:<8.4f} {r['steps']}")
    print("-" * 50)
    avg = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"{'Average':<25} {'':10} {avg:<8.4f}")
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Status: {'PASS' if all(r['score'] > 0 for r in results) else 'PARTIAL'}")


if __name__ == "__main__":
    main()
