#!/usr/bin/env python3
"""
Multi-Model Evaluation Script for DarkStore Inbound Environment.

Runs all 5 tasks against multiple models via the Hugging Face router
and produces a model comparison table saved to outputs/model_comparison.json.

This script proves the environment differentiates between AI capability levels.

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

# All 5 tasks, evaluated with the same seed for reproducibility
TASKS = [
    {"name": "clean_delivery",       "seed": 42, "difficulty": "easy"},
    {"name": "quantity_mismatch",    "seed": 42, "difficulty": "medium"},
    {"name": "hidden_violation",     "seed": 42, "difficulty": "hard"},
    {"name": "price_discrepancy",    "seed": 42, "difficulty": "medium_hard"},
    {"name": "multi_violation_chaos", "seed": 42, "difficulty": "expert"},
]

# Models to compare — all available via HF router with just an HF token
# Strong  -> Medium -> Weak: should show clear score gradient to prove env value
MODELS = [
    {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "label": "Qwen2.5-72B (Strong frontier)",
    },
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "label": "Llama-3.3-70B (Strong open-source)",
    },
    {
        "id": "Qwen/Qwen2.5-32B-Instruct",
        "label": "Qwen2.5-32B (Medium)",
    },
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "label": "Llama-3.1-8B (Weak small model)",
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
  - flag_shortage: fewer units arrived than ordered (provide the shortage_qty)
  - reject_sku: item fails inspection (provide the rejection reason)

Step 3 — When all pending_skus are empty, call finalize.

## VALID REJECTION REASONS
"damaged" | "expired" | "shelf_life_violation" | "cold_chain_violation" | "unauthorized_substitution" | "price_discrepancy"

## OUTPUT FORMAT
Respond with exactly one JSON action per turn. No text, no explanation, just JSON.

Examples:
{"action_type": "request_invoice"}
{"action_type": "accept_sku", "sku_id": "SKU001"}
{"action_type": "flag_shortage", "sku_id": "SKU002", "shortage_qty": 5}
{"action_type": "reject_sku", "sku_id": "SKU003", "reason": "damaged"}
{"action_type": "finalize"}
"""


# ---------------------------------------------------------------------------
# Helpers (shared with inference.py)
# ---------------------------------------------------------------------------


def format_observation(obs: Dict[str, Any]) -> str:
    lines = []
    if obs.get("message"):
        lines.append(f"[MESSAGE] {obs['message']}")
    lines.append(f"[STATUS] Phase: {obs.get('phase')} | Steps remaining: {obs.get('steps_remaining')}")
    lines.append(f"[STATUS] Pending SKUs: {obs.get('pending_skus')} | Resolved: {obs.get('resolved_skus')}")

    if obs.get("policy_rules"):
        pr = obs["policy_rules"]
        lines.append(f"[POLICY] Min shelf life: {pr.get('min_shelf_life_days')} days | "
                     f"Max transit temp: {pr.get('max_transit_temp_celsius')}°C")

    if obs.get("purchase_order"):
        lines.append("[PURCHASE ORDER]")
        for item in obs["purchase_order"]:
            lines.append(f"  {item['sku_id']}: {item['name']} x{item['quantity']} @ ₹{item['unit_price']}")

    if obs.get("invoice"):
        lines.append("[INVOICE]")
        for item in obs["invoice"]:
            lines.append(f"  {item['sku_id']}: {item['name']} x{item['quantity']} @ ₹{item['unit_price']}")

    if obs.get("scan_data"):
        lines.append("[SCAN DATA]")
        for item in obs["scan_data"]:
            lines.append(f"  {item['sku_id']}: scanned={item['scanned_qty']} "
                         f"condition={item['condition']} expiry={item['expiry_date']}")

    if obs.get("cold_chain_log"):
        readings = obs["cold_chain_log"]
        max_temp = max(r["temperature_celsius"] for r in readings)
        lines.append(f"[COLD CHAIN] {len(readings)} readings, max temp: {max_temp:.1f}°C")
        for r in readings:
            if r["temperature_celsius"] > 8.0:
                lines.append(f"  *** SPIKE: {r['timestamp']} = {r['temperature_celsius']}°C ***")

    return "\n".join(lines)


def parse_action(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    import re
    for pattern in [r"\{[^{}]+\}", r"```json\s*(\{[^{}]+\})\s*```", r"```\s*(\{[^{}]+\})\s*```"]:
        matches = re.findall(pattern, raw, re.DOTALL)
        for m in matches:
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(raw)
    except Exception:
        return None


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
            obs = r["data"]["observation"]
            done = r["data"].get("done", False)

            history: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

            while not done and step_count < 25:
                step_count += 1
                if step_count > 1:
                    time.sleep(0.5)

                history.append({"role": "user", "content": format_observation(obs)})

                raw = ""
                for attempt in range(2):
                    try:
                        resp = client.chat.completions.create(
                            model=model_id,
                            messages=history,
                            max_tokens=MAX_TOKENS,
                            temperature=0.0,
                        )
                        raw = resp.choices[0].message.content or ""
                        break
                    except Exception as e:
                        if "rate limit" in str(e).lower() or "429" in str(e):
                            if attempt == 0:
                                time.sleep(10)
                            else:
                                raw = '{"action_type": "finalize"}'
                        else:
                            raw = '{"action_type": "finalize"}'
                            break

                action = parse_action(raw) or {"action_type": "finalize"}
                history.append({"role": "assistant", "content": raw})

                if len(history) > 24:
                    history = [history[0]] + history[-22:]

                ws.send(json.dumps({"type": "step", "data": action}))
                r = json.loads(ws.recv())
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

    print(f"score={final_score:.3f}", flush=True)
    return {
        "task": task_name,
        "difficulty": task["difficulty"],
        "score": final_score,
        "steps": step_count,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def main() -> None:
    if not API_KEY:
        print("ERROR: No HF token found. Set HF_TOKEN env var or in .env file.")
        print("Get a free token at https://huggingface.co/settings/tokens")
        sys.exit(1)

    print(f"\nDarkStore Inbound — Multi-Model Evaluation")
    print(f"API: {API_BASE_URL}")
    print(f"Env: {ENV_URL}")
    print(f"Tasks: {[t['name'] for t in TASKS]}")
    print(f"Models: {len(MODELS)}")
    print("=" * 65)

    all_results = []
    timestamp = datetime.now().isoformat()

    for model_info in MODELS:
        model_id = model_info["id"]
        label = model_info["label"]
        print(f"\nModel: {label}")
        print(f"  ID: {model_id}")

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        model_results = []

        for i, task in enumerate(TASKS):
            if i > 0:
                time.sleep(5)  # Rate limit buffer between tasks
            result = run_episode(client, model_id, task)
            model_results.append(result)

        avg = sum(r["score"] for r in model_results) / len(model_results)
        print(f"  Average score: {avg:.3f}")

        all_results.append({
            "model_id": model_id,
            "model_label": label,
            "task_results": model_results,
            "average_score": round(avg, 4),
        })

        # Wait between models to avoid rate limiting
        print(f"  Waiting 30s before next model...")
        time.sleep(30)

    # Save results to outputs/
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

    print(f"\n{'=' * 65}")
    print("MODEL COMPARISON RESULTS")
    print("=" * 65)

    # Print comparison table
    task_names = [t["name"] for t in TASKS]
    header = f"{'Model':<35}" + "".join(f"{n[:8]:>10}" for n in task_names) + f"{'AVG':>8}"
    print(header)
    print("-" * 65)

    for model_data in all_results:
        scores = {r["task"]: r["score"] for r in model_data["task_results"]}
        row = f"{model_data['model_label'][:33]:<35}"
        row += "".join(f"{scores.get(t, 0.0):>10.3f}" for t in task_names)
        row += f"{model_data['average_score']:>8.3f}"
        print(row)

    print(f"\nResults saved to: {output_path}")
    print("\nCopy this table into README.md to show model differentiation evidence.")


if __name__ == "__main__":
    main()
