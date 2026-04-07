#!/usr/bin/env python3
"""
Baseline inference script for the DarkStore Inbound Environment.

Runs an LLM agent against all 3 tasks (clean_delivery, quantity_mismatch,
hidden_violation) and reports scores.

Setup:
    Edit the .env file in this directory with your credentials, then run:
        python inference.py

    Or set env vars manually:
        set API_BASE_URL=https://lightning.ai/api/v1/
        set API_KEY=your_key_here
        set MODEL_NAME=openai/gpt-4.1
        python inference.py
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
    {"name": "clean_delivery",    "seed": 42, "difficulty": "easy"},
    {"name": "quantity_mismatch", "seed": 42, "difficulty": "medium"},
    {"name": "hidden_violation",  "seed": 42, "difficulty": "hard"},
]

MAX_TOKENS = 1024
VERBOSE = True

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a warehouse receiving dock agent for a dark store.

## MANDATORY FIRST STEPS — do these three BEFORE any decisions:
1. {"action_type": "request_invoice"}
2. {"action_type": "request_scan"}
3. {"action_type": "request_cold_chain"}

## THEN for each SKU in pending_skus, make ONE decision:
- accept_sku: everything is fine
- flag_shortage: scanned quantity is less than ordered  (shortage_qty = ordered - scanned)
- reject_sku: item fails a check  (must include reason)

## CHECKS to apply to every SKU:
1. If scanned_qty < PO quantity   -> flag_shortage with shortage_qty = (PO_qty - scanned_qty)
2. If invoice SKU != PO SKU       -> reject, reason="unauthorized_substitution"
3. If condition is "damaged" or "leaking" -> reject, reason="damaged"
4. If days until expiry < min_shelf_life_days (today is 2026-04-01) -> reject, reason="shelf_life_violation"
5. If ANY cold_chain temperature > max_transit_temp_celsius -> reject, reason="cold_chain_violation"

## Valid reason values (copy exactly):
"damaged" | "expired" | "shelf_life_violation" | "cold_chain_violation" | "unauthorized_substitution"

## When pending_skus is empty, call:
{"action_type": "finalize"}

## RESPONSE FORMAT: JSON ONLY — no text, no markdown, no explanation.
{"action_type": "request_invoice"}
{"action_type": "request_scan"}
{"action_type": "request_cold_chain"}
{"action_type": "accept_sku", "sku_id": "SKU001"}
{"action_type": "flag_shortage", "sku_id": "SKU002", "shortage_qty": 6}
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

    if VERBOSE:
        print(f"\n{'=' * 55}")
        print(f"Task: {task_name}  ({difficulty})")
        print("=" * 55)

    ws_url = env_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    with websockets.sync.client.connect(ws_url) as ws:
        # Reset
        ws.send(json.dumps({"type": "reset", "data": {"task": task_name, "seed": seed}}))
        r = json.loads(ws.recv())
        obs = r["data"]["observation"]
        done = r["data"].get("done", False)  # check if reset returned done=True

        sku_count = len(obs.get("purchase_order", []))
        pending = obs.get("pending_skus", [])
        if VERBOSE:
            print(f"Reset OK. {sku_count} SKUs to inspect. Pending: {pending}. Done: {done}")

        # --- Required structured output block: START ---
        print(f"[START] task={task_name}", flush=True)

        history: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        step_count = 0
        final_score = 0.0

        while not done:
            step_count += 1
            obs_text = format_observation(obs)

            history.append({"role": "user", "content": obs_text})

            # Small delay to avoid rate limiting
            if step_count > 1:
                time.sleep(1)

            # Call LLM (with retry on rate limit)
            raw = ""
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
                            print(f"  Rate limited, waiting 10s...")
                            time.sleep(10)
                        else:
                            print(f"  Rate limit persists, finalizing")
                            raw = '{"action_type": "finalize"}'
                    else:
                        print(f"  LLM error: {e}")
                        raw = '{"action_type": "finalize"}'
                        break

            action = parse_action(raw)
            if action is None:
                if VERBOSE:
                    print(f"  Step {step_count}: parse failed, finalizing. Raw: {raw[:100]}")
                action = {"action_type": "finalize"}

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

            # --- Required structured output block: STEP ---
            print(f"[STEP] step={step_count} reward={reward}", flush=True)

            if done:
                final_score = reward
                # --- Required structured output block: END ---
                print(f"[END] task={task_name} score={final_score} steps={step_count}", flush=True)
                if VERBOSE:
                    print(f"\n  Done! Score: {final_score}")
                    print(f"  {obs.get('message', '')}")

            # Trim history to avoid token explosion
            if len(history) > 24:
                history = [history[0]] + history[-22:]

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
