---
title: Darkstore Inbound Environment Server
emoji: "📦"
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

A real-world OpenEnv RL environment simulating a quick-commerce (dark store) receiving dock. An LLM agent must inspect incoming deliveries, compare physical scan data against purchase orders, check cold chain logs, and verify invoice pricing to make accept, reject, or shortage decisions per SKU.

## Motivation

Supply chain reconciliation (three-way matching) is a $100B+ industry problem. Every dark store and warehouse faces this daily: when goods arrive, receiving agents must verify that:

1. What arrived matches what was ordered (Purchase Order)
2. What the supplier charges (Invoice) matches the physical count and agreed prices
3. The products are safe — cold chain compliance, shelf life, physical condition

This environment simulates the exact cognitive process, testing an agent's ability to iteratively gather information, perform arithmetic checks, follow business policies, and identify both obvious and hidden violations.

## Tasks

5 built-in tasks with genuine difficulty progression, each graded 0.0–1.0 by a deterministic grader with anti-gaming penalties.

| Task | Difficulty | SKUs | What It Tests |
|---|---|---|---|
| `clean_delivery` | Easy | 4–6 | Basic instruction following — gather data, accept all. Prevents over-cautious rejection. |
| `quantity_mismatch` | Medium | 6–10 | Arithmetic + comparison — detect shortages and unauthorized substitutions at randomized positions. |
| `hidden_violation` | Hard | 8–12 | Proactive investigation — surface documents look clean but cold chain log has a hidden spike. Agent must gather all data sources. |
| `price_discrepancy` | Medium-Hard | 7–10 | Financial reasoning — detect invoice prices that exceed PO prices. Agent must compare unit prices and flag overcharging. |
| `multi_violation_chaos` | Expert | 10–12 | Priority reasoning — some SKUs have multiple simultaneous violations. Agent must apply the correct priority rule (cold chain > shelf life > damaged > shortage). |

## Model Comparison (Empirical Evidence)

Evaluated 7 April 2026 via HF router (`router.huggingface.co/v1`). All models run with seed=42, temperature=0.0.

| Model | Easy | Medium | Hard | Med-Hard | Expert | **Average** |
|---|---|---|---|---|---|---|
| Qwen2.5-72B-Instruct | **1.000** | **0.867** | 0.572 | **1.000** | **0.441** | **0.776** |
| Llama-3.3-70B-Instruct | **1.000** | 0.578 | **0.733** | 0.828 | 0.090 | 0.646 |
| Qwen3-32B | **1.000** | 0.150 | 0.067 | 0.475 | 0.067 | 0.352 |
| Llama-3.1-8B-Instruct | 0.300 | 0.000 | 0.000 | 0.150 | 0.067 | 0.103 |

**Key observations:**
- Clear **7.5× score gradient** between strongest (0.776) and weakest (0.103) models
- Qwen3-32B scores lower due to reasoning-token overhead (`<think>` tags consuming output budget)
- **Expert task** (multi-violation chaos) is genuinely hard — even frontier models score < 0.5
- The environment meaningfully separates model capability at every difficulty level
- Full results in [`outputs/model_comparison.json`](outputs/model_comparison.json)

## Grader Design (Anti-Gaming)

Each grader has multiple independent scoring components and an **anti-gaming penalty** that collapses trivial strategies:

- A "reject everything" agent scores ~37% on the hard task (was ~73% before penalty)
- A "accept everything" agent scores 0% on tasks with violations
- Per-step rewards are **uniform and non-informative** (0.02 per valid action) — ground truth is only revealed at `finalize`, preventing reward exploitation

## Observation Space

Progressive information reveal — the agent starts with only the Purchase Order and Policy Rules. Other documents must be explicitly requested.

| Field | Type | Revealed |
|---|---|---|
| `purchase_order` | List[LineItem] | Always |
| `policy_rules` | PolicyRules | Always |
| `pending_skus` | List[str] | Always |
| `invoice` | List[LineItem] or None | After `request_invoice` |
| `scan_data` | List[ScanItem] or None | After `request_scan` |
| `cold_chain_log` | List[ColdChainReading] or None | After `request_cold_chain` |
| `message` | str | Always |
| `steps_remaining` | int | Always (max 25) |

## Action Space

| Action | Parameters | Notes |
|---|---|---|
| `request_invoice` | — | Reveals invoice data |
| `request_scan` | — | Reveals scan/count data |
| `request_cold_chain` | — | Reveals temperature log |
| `accept_sku` | `sku_id` | Accept as correctly delivered |
| `flag_shortage` | `sku_id`, `shortage_qty` | Flag quantity discrepancy |
| `reject_sku` | `sku_id`, `reason` | Reject with reason |
| `finalize` | — | End episode, trigger grader |

**Valid rejection reasons:** `damaged` · `expired` · `shelf_life_violation` · `cold_chain_violation` · `unauthorized_substitution` · `price_discrepancy`

## Reward Structure

| Event | Reward |
|---|---|
| Any valid data request | 0.0 |
| Any valid SKU decision (accept/flag/reject) | +0.02 (progress, non-informative) |
| `finalize` | Returns grader score 0.0–1.0 |

The grader score is the sum of weighted sub-scores per task (correctness of decisions, precision of quantities, reason accuracy, data thoroughness). Correctness is only revealed at `finalize` to prevent reward hacking.

## Project Structure

```
darkstore_inbound/
  README.md              # This file
  openenv.yaml           # OpenEnv manifest
  pyproject.toml         # Dependencies
  requirements.txt       # Pip-installable dependency list
  models.py              # Pydantic types: Action, Observation, State
  inference.py           # Baseline LLM agent (runs all 5 tasks)
  evaluate_models.py     # Multi-model comparison script
  outputs/               # Evaluation results (JSON)
  server/
    app.py               # FastAPI server (HTTP + WebSocket)
    darkstore_inbound_environment.py  # Core reset()/step() logic
    scenario_generator.py             # 5 seeded scenario generators
    graders.py                        # 5 deterministic graders (0.0–1.0)
    Dockerfile
```

## Setup

### 1. Install

```bash
git clone https://github.com/MaverickDev-J/darkstore-inbound.git
cd darkstore-inbound
uv sync
```

### 2. Run Server Locally

```bash
uv run server
# Server:  http://localhost:8000
# Web UI:  http://localhost:8000/web
# API docs: http://localhost:8000/docs
```

### 3. Run Baseline Inference

No OpenAI key needed — uses the free Hugging Face router.

```bash
# Get a free token at https://huggingface.co/settings/tokens
set HF_TOKEN=your_hf_token_here
python inference.py
```

### 4. Run Multi-Model Evaluation

```bash
set HF_TOKEN=your_hf_token_here
python evaluate_models.py
# Results saved to outputs/model_comparison.json
```

### 5. Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | OpenAI-compatible endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `openai/gpt-oss-120b:novita` |
| `HF_TOKEN` | Authentication (also reads `API_KEY`) | — |
| `ENV_URL` | Environment server URL | `http://localhost:8000` |

## Live Deployment

- **HF Space:** https://huggingface.co/spaces/Maverick006/darkstore-inbound
- **Health:** https://maverick006-darkstore-inbound.hf.space/health
- **Web UI:** https://maverick006-darkstore-inbound.hf.space/web
- **API Docs:** https://maverick006-darkstore-inbound.hf.space/docs
