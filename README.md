---
title: Darkstore Inbound Environment Server
emoji: "\U0001F4E6"
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# DarkStore Inbound Receiving Environment

A real-world OpenEnv simulation of a quick-commerce (dark store) receiving dock. An LLM agent must inspect incoming deliveries, compare physical scan data against purchase orders, and check cold chain logs to make accept, reject, or shortage decisions per SKU.

## Motivation

Supply chain reconciliation (three-way matching) is a $100B+ industry problem. When goods arrive at a warehouse or dark store, receiving agents must rapidly verify that:

1. What arrived matches what was ordered (Purchase Order).
2. What the supplier is charging (Invoice) matches the physical count.
3. The products are safe (Cold chain temperatures and shelf life requirements).

This environment simulates this exact process, testing an agent's ability to iteratively gather information, perform arithmetic checks, follow specific business policies, and identify hidden safety violations.

## Task Descriptions and Expected Difficulty

This environment comes with 3 built-in tasks, each scored by a deterministic grader from 0.0 to 1.0.

| Task | Difficulty | SKUs | Description |
|---|---|---|---|
| `clean_delivery` | Easy | 5 | A perfect delivery. The agent must request the data, verify everything matches, and accept all SKUs. Tests basic instruction following and prevents over-cautious rejection. |
| `quantity_mismatch` | Medium | 8 | A delivery with discrepancies. Contains quantity shortages and unauthorized substitutions. The agent must flag shortages with the correct missing amount and reject substitutions. Tests arithmetic and comparison reasoning. |
| `hidden_violation` | Hard | 10 | Surface documents (PO and scan count) look perfect, but there is a hidden temperature spike in the cold chain log and an impending expiry date policy violation. Tests proactive investigation of non-obvious data sources. |

## Baseline Scores

Baseline scores using the included `inference.py` script with `gpt-4.1` via Lightning AI:

| Task | Difficulty | Score |
|---|---|---|
| `clean_delivery` | Easy | **1.00** |
| `quantity_mismatch` | Medium | **1.00** |
| `hidden_violation` | Hard | **0.82** |
| **Average** | | **0.94** |

## Observation Space

The observation is progressively revealed. Initially the agent only sees the Purchase Order and Policy Rules. Other documents must be explicitly requested via actions.

| Field | Type | Description |
|---|---|---|
| `phase` | str | Current phase: `inspect`, `decide`, `complete` |
| `steps_remaining` | int | Steps remaining before auto-finalize (starts at 25) |
| `purchase_order` | List[LineItem] | What was ordered: SKU ID, name, qty, unit price |
| `policy_rules` | PolicyRules | Min shelf life (days), max transit temp (C), approved substitutions |
| `invoice` | List[LineItem] or None | Revealed after `request_invoice`. Supplier billing data |
| `scan_data` | List[ScanItem] or None | Revealed after `request_scan`. Physical count, condition, expiry |
| `cold_chain_log` | List[ColdChainReading] or None | Revealed after `request_cold_chain`. Transit temperatures |
| `pending_skus` | List[str] | SKUs that still need a decision |
| `resolved_skus` | List[str] | SKUs already decided upon |
| `message` | str | Feedback from the last action |

## Action Space

All actions are JSON objects with an `action_type` field.

### Data Gathering (no reward, reveals data)

```json
{"action_type": "request_invoice"}
{"action_type": "request_scan"}
{"action_type": "request_cold_chain"}
```

### SKU Decisions (one per SKU)

```json
{"action_type": "accept_sku", "sku_id": "SKU001"}
{"action_type": "flag_shortage", "sku_id": "SKU002", "shortage_qty": 5}
{"action_type": "reject_sku", "sku_id": "SKU003", "reason": "damaged"}
```

Valid rejection reasons: `"damaged"`, `"expired"`, `"shelf_life_violation"`, `"cold_chain_violation"`, `"unauthorized_substitution"`

### Episode Control

```json
{"action_type": "finalize"}
```

## Reward Structure

| Action | Reward |
|---|---|
| Data gathering (request_invoice, etc.) | 0 |
| Correct accept/reject/shortage | +0.1 to +0.2 |
| Wrong decision | -0.05 to -0.15 |
| Correct rejection with right reason | +0.2 (bonus for exact reason) |
| Finalize | Triggers grader, returns final score 0.0-1.0 |

## Project Structure

```
darkstore_inbound/
  README.md              # This file (HF Spaces metadata + docs)
  openenv.yaml           # OpenEnv manifest
  pyproject.toml         # Dependencies and project config
  uv.lock                # Locked dependencies
  models.py              # Pydantic types: Action, Observation, State
  client.py              # WebSocket client for programmatic use
  __init__.py            # Package exports
  inference.py           # Baseline LLM agent (openai client)
  .env                   # Local API keys (not committed)
  server/
    app.py               # FastAPI application (HTTP + WebSocket)
    darkstore_inbound_environment.py  # Core reset()/step() logic
    scenario_generator.py             # Seeded scenario generation
    graders.py                        # Deterministic scoring (0.0-1.0)
    Dockerfile                        # Container for HF Spaces
    __init__.py
```

## Setup and Usage

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Install

```bash
git clone https://github.com/MaverickDev-J/darkstore-inbound.git
cd darkstore-inbound
uv sync
```

### 2. Run Server Locally

```bash
uv run server
# Server: http://localhost:8000
# Web UI: http://localhost:8000/web
# API docs: http://localhost:8000/docs
```

### 3. Run Baseline Inference

Create a `.env` file in the project root:

```env
API_BASE_URL=https://lightning.ai/api/v1/
API_KEY=your_api_key_here
MODEL_NAME=openai/gpt-4.1
```

Then run:

```bash
python inference.py
```

The script reads `API_BASE_URL`, `MODEL_NAME`, and `API_KEY` (or `HF_TOKEN`) from environment variables or `.env`, connects to the local server, and runs all 3 tasks.

### 4. Deploy to Hugging Face Spaces

```bash
# Login to HF
huggingface-cli login

# Push
openenv push --repo-id your-username/darkstore-inbound
```

### 5. Environment Variables (for judges)

The inference script accepts these standard env vars:

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | OpenAI-compatible API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model name | `openai/gpt-oss-120b:novita` |
| `HF_TOKEN` | API key (also reads `API_KEY`) | -- |
| `ENV_URL` | Environment server URL | `http://localhost:8000` |

## Live Deployment

- HF Space: https://huggingface.co/spaces/Maverick006/darkstore-inbound
- Health: https://maverick006-darkstore-inbound.hf.space/health
- Web UI: https://maverick006-darkstore-inbound.hf.space/web
- API Docs: https://maverick006-darkstore-inbound.hf.space/docs
