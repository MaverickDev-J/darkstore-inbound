#!/usr/bin/env python3
"""
Multi-Model Evaluation Script for DarkStore Inbound Environment.

Runs all 5 tasks against multiple models via the Hugging Face router
and produces a model comparison table saved to outputs/model_comparison.json.

Usage:
    set HF_TOKEN=your_hf_token_here
    python evaluate_models.py

Requirements:
    - Local server running: uv run server
    - HF_TOKEN set in env or .env file
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
MAX_TOKENS = 1024

TASKS = [
    {"name": "clean_delivery",       "seed": 42, "difficulty": "easy"},
    {"name": "quantity_mismatch",    "seed": 42, "difficulty": "medium"},
    {"name": "hidden_violation",     "seed": 42, "difficulty": "hard"},
    {"name": "price_discrepancy",    "seed": 42, "difficulty": "medium_hard"},
    {"name": "multi_violation_chaos", "seed": 42, "difficulty": "expert"},
]

# Models available on HF router
MODELS = [
    {
        "id": "Qwen/Qwen3-32B",
        "label": "Qwen3-32B (Latest frontier)",
    },
    {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "label": "Qwen2.5-72B-Instruct (Strong)",
    },
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "label": "Llama-3.3-70B-Instruct (Strong)",
    },
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "label": "Llama-3.1-8B-Instruct (Weak)",
    },
]

SYSTEM_PROMPT = """You are a receiving dock agent at a dark store (quick-commerce warehouse).
A supplier delivery has arrived. Your job is to inspect it thoroughly and decide what to accept, flag, or reject.

## YOUR WORKFLOW

Step 1 — Gather all available data before making any decisions:
  - Request the supplier invoice
  - Request the physical scan data
  - Request the cold chain temperature log

Step 2 — For each SKU in pending_skus, make exactly one decision:
  - accept_sku: item is fine, accept it
  - flag_shortage: fewer units arrived than ordered (provide the shortage_qty = ordered - scanned)
  - reject_sku: item fails inspection (provide the rejection reason)

Step 3 — When all pending_skus are empty, call finalize.

## IMPORTANT RULES
- ALWAYS gather ALL three data sources (invoice, scan, cold chain) BEFORE making any decisions
- Compare quantities: if scanned_qty < ordered qty, flag_shortage with the difference
- Check conditions: if condition is "damaged", reject with reason "damaged"
- Check cold chain: if any temperature > max_transit_temp, reject perishable items with "cold_chain_violation"
- Check expiry: if expiry is too soon (< min_shelf_life_days from today), reject with "shelf_life_violation"
- Check prices: if invoice unit_price > PO unit_price, reject with "price_discrepancy"
- If a SKU on invoice doesn't match PO, reject with "unauthorized_substitution"

## VALID REJECTION REASONS
"damaged" | "expired" | "shelf_life_violation" | "cold_chain_violation" | "unauthorized_substitution" | "price_discrepancy"

## OUTPUT FORMAT
Respond with exactly one JSON action per turn. No markdown, no explanation, just the raw JSON object.

{"action_type": "request_invoice"}
{"action_type": "request_scan"}
{"action_type": "request_cold_chain"}
{"action_type": "accept_sku", "sku_id": "SKU001"}
{"action_type": "flag_shortage", "sku_id": "SKU002", "shortage_qty": 5}
{"action_type": "reject_sku", "sku_id": "SKU003", "reason": "damaged"}
{"action_type": "finalize"}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_observation(obs: Dict[str, Any]) -> str:
    """Format observation into clear readable text for the LLM."""
    lines = []
    if obs.get("message"):
        lines.append(f"MESSAGE: {obs['message']}")
    lines.append(f"Phase: {obs.get('phase')} | Steps remaining: {obs.get('steps_remaining')}")
    lines.append(f"Pending SKUs: {obs.get('pending_skus')} | Resolved: {obs.get('resolved_skus')}")

    if obs.get("policy_rules"):
        pr = obs["policy_rules"]
        lines.append(f"POLICY: Min shelf life: {pr.get('min_shelf_life_days')} days | "
                     f"Max transit temp: {pr.get('max_transit_temp_celsius')}°C")
        if pr.get("approved_substitutions"):
            lines.append(f"  Approved substitutions: {pr['approved_substitutions']}")

    if obs.get("purchase_order"):
        lines.append("PURCHASE ORDER:")
        for item in obs["purchase_order"]:
            lines.append(f"  {item['sku_id']}: {item['name']} qty={item['quantity']} price=₹{item['unit_price']}")

    if obs.get("invoice"):
        lines.append("INVOICE:")
        for item in obs["invoice"]:
            lines.append(f"  {item['sku_id']}: {item['name']} qty={item['quantity']} price=₹{item['unit_price']}")

    if obs.get("scan_data"):
        lines.append("SCAN DATA:")
        for item in obs["scan_data"]:
            lines.append(f"  {item['sku_id']}: scanned_qty={item['scanned_qty']} "
                         f"condition={item['condition']} expiry={item['expiry_date']}")

    if obs.get("cold_chain_log"):
        readings = obs["cold_chain_log"]
        max_temp = max(r["temperature_celsius"] for r in readings)
        lines.append(f"COLD CHAIN LOG: {len(readings)} readings, max={max_temp:.1f}°C")
        for r in readings:
            flag = " *** ABOVE THRESHOLD ***" if r["temperature_celsius"] > 8.0 else ""
            lines.append(f"  {r['timestamp']}: {r['temperature_celsius']}°C{flag}")

    return "\n".join(lines)


def parse_action(raw: str) -> Optional[Dict[str, Any]]:
    """Extract JSON action from model response, handling markdown fences and Qwen3 think tags."""
    raw = raw.strip()
    # Strip Qwen3-style <think>...</think> reasoning tags (closed and unclosed)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()  # unclosed thinking
    # Try extracting from markdown code blocks first
    for pattern in [r"```json\s*(\{[^{}]+\})\s*```", r"```\s*(\{[^{}]+\})\s*```"]:
        matches = re.findall(pattern, raw, re.DOTALL)
        for m in matches:
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue
    # Try finding bare JSON
    matches = re.findall(r"\{[^{}]+\}", raw, re.DOTALL)
    for m in matches:
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    # Try the whole string
    try:
        return json.loads(raw)
    except Exception:
        return None


def call_llm(client: OpenAI, model_id: str, messages: List[Dict], retries: int = 3) -> str:
    """Call LLM with robust retry logic. Returns raw response text."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = "rate limit" in err or "429" in err or "too many" in err
            is_overloaded = "overloaded" in err or "503" in err or "502" in err
            if is_rate_limit or is_overloaded:
                wait = (attempt + 1) * 15  # 15s, 30s, 45s
                print(f"\n    [Rate limited, waiting {wait}s...]", end="", flush=True)
                time.sleep(wait)
            else:
                print(f"\n    [LLM error: {str(e)[:80]}]", end="", flush=True)
                if attempt < retries - 1:
                    time.sleep(5)
    return ""  # All retries failed


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, model_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
    import websockets.sync.client  # type: ignore

    task_name = task["name"]
    seed = task["seed"]
    ws_url = ENV_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws"

    step_count = 0
    final_score = 0.0
    rewards: List[float] = []
    success = False

    print(f"  [{task_name}]", end=" ", flush=True)

    try:
        with websockets.sync.client.connect(ws_url) as ws:
            ws.send(json.dumps({"type": "reset", "data": {"task": task_name, "seed": seed}}))
            r = json.loads(ws.recv())

            if r.get("type") == "error":
                print(f"RESET ERROR: {r['data'].get('message', 'unknown')}", end=" ", flush=True)
                print(f"score={final_score:.3f}", flush=True)
                return {"task": task_name, "difficulty": task["difficulty"], "score": 0.0, "steps": 0, "success": False}

            obs = r["data"]["observation"]
            done = r["data"].get("done", False)

            history: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

            while not done and step_count < 25:
                step_count += 1

                # Rate limit: 2s between steps to avoid HF router throttling
                if step_count > 1:
                    time.sleep(2)

                history.append({"role": "user", "content": format_observation(obs)})

                raw = call_llm(client, model_id, history)

                if not raw:
                    # All retries exhausted — finalize gracefully
                    print(f"(LLM failed, finalizing)", end=" ", flush=True)
                    action = {"action_type": "finalize"}
                    raw = json.dumps(action)
                else:
                    action = parse_action(raw)
                    if action is None:
                        print(f"(parse fail: {raw[:50]})", end=" ", flush=True)
                        action = {"action_type": "finalize"}

                history.append({"role": "assistant", "content": raw})

                # Trim history to avoid token explosion
                if len(history) > 30:
                    history = [history[0]] + history[-28:]

                ws.send(json.dumps({"type": "step", "data": action}))
                r = json.loads(ws.recv())

                if r.get("type") == "error":
                    print(f"(step error: {r['data'].get('message', '')[:50]})", end=" ", flush=True)
                    continue

                payload = r["data"]
                obs = payload["observation"]
                done = payload.get("done", False)
                reward = payload.get("reward", 0.0)
                rewards.append(reward)

                if done:
                    final_score = reward
                    success = final_score > 0

    except Exception as e:
        print(f"ERROR: {e}", end=" ", flush=True)

    print(f"steps={step_count} score={final_score:.3f}", flush=True)
    return {
        "task": task_name,
        "difficulty": task["difficulty"],
        "score": round(final_score, 4),
        "steps": step_count,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("ERROR: No HF token. Set HF_TOKEN in .env or env var.")
        sys.exit(1)

    print(f"\nDarkStore Inbound — Multi-Model Evaluation")
    print(f"API: {API_BASE_URL}")
    print(f"Env: {ENV_URL}")
    print(f"Tasks: {[t['name'] for t in TASKS]}")
    print(f"Models: {[m['id'] for m in MODELS]}")
    print("=" * 80)

    all_results = []
    timestamp = datetime.now().isoformat()

    for mi, model_info in enumerate(MODELS):
        model_id = model_info["id"]
        label = model_info["label"]
        print(f"\n[{mi+1}/{len(MODELS)}] Model: {label}")

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        model_results = []

        for i, task in enumerate(TASKS):
            if i > 0:
                time.sleep(8)  # 8s between tasks
            result = run_episode(client, model_id, task)
            model_results.append(result)

        avg = sum(r["score"] for r in model_results) / len(model_results)
        print(f"  >>> Average: {avg:.3f}")

        all_results.append({
            "model_id": model_id,
            "model_label": label,
            "task_results": model_results,
            "average_score": round(avg, 4),
        })

        if mi < len(MODELS) - 1:
            print(f"  Waiting 45s before next model...")
            time.sleep(45)

    # Save results
    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/model_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "env": "darkstore_inbound",
            "api_base": API_BASE_URL,
            "tasks": [t["name"] for t in TASKS],
            "results": all_results,
        }, f, indent=2)

    # Print table
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)

    task_names = [t["name"] for t in TASKS]
    header = f"{'Model':<38}" + "".join(f"{n[:12]:>13}" for n in task_names) + f"{'AVG':>8}"
    print(header)
    print("-" * 80)

    for md in all_results:
        scores = {r["task"]: r["score"] for r in md["task_results"]}
        row = f"{md['model_label'][:36]:<38}"
        row += "".join(f"{scores.get(t, 0.0):>13.3f}" for t in task_names)
        row += f"{md['average_score']:>8.3f}"
        print(row)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
