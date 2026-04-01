---
title: Darkstore Inbound Environment Server
emoji: 📦
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

A real-world OpenEnv simulation of a quick-commerce (dark store) receiving dock. An LLM agent must inspect incoming deliveries, compare physical scan data against purchase orders, and check cold chain logs to make accept, reject, or shortage decisions.

## Motivation

Supply chain reconciliation (three-way matching) is a $100B+ industry problem. When goods arrive at a warehouse or dark store, receiving agents must rapidly verify that:
1. What arrived matches what was ordered (Purchase Order).
2. What the supplier is charging (Invoice) matches the physical count.
3. The products are safe (Cold chain temperatures & Shelf life requirements).

This environment simulates this exact process, testing an agent's ability to iteratively gather information, perform arithmetic checks, follow specific business policies, and identify hidden safety violations.

##  Task Descriptions & Expected Difficulty

This environment comes with 3 built-in tasks, each scored by a deterministic grader from 0.0 to 1.0.

| Task | Difficulty | Description |
|---|---|---|
| `clean_delivery` | **Easy** | A perfect delivery. The agent must request the data, verify everything matches, and accept all SKUs. Tests basic instruction following and prevents over-cautious rejection. |
| `quantity_mismatch` | **Medium** | A delivery with discrepancies. Contains quantity shortages and unauthorized substitutions. The agent must flag shortages with the correct missing amount and reject substitutions. Tests arithmetic and comparison reasoning. |
| `hidden_violation` | **Hard** | Surface documents (PO and scan count) look perfect, but there is a hidden temperature spike in the cold chain log and an impending expiry date policy violation. Tests proactive investigation of non-obvious data sources. |

---

##  Baseline Scores

Here are the baseline scores using the included `inference.py` script.

| Task | Difficulty | Score (`gpt-4.1` on Lightning AI) |
|---|---|---|
| `clean_delivery` | Easy | **1.00**  |
| `quantity_mismatch` | Medium | **1.00**  |
| `hidden_violation` | Hard | **0.82**  |
| **Average** | | **0.94** |

---

##  Observation Space

The state is progressively revealed. The agent only sees the Purchase Order initially and must take actions to reveal other documents.

The observation is provided as a JSON dictionary (via Pydantic):

- `phase` (str): Current session phase (`inspect`, `decide`, `complete`).
- `step_budget` (int): Number of steps remaining (starts at 25).
- `purchase_order` (List[Dict]): What was ordered (SKU, name, ordered qty, price).
- `policy_rules` (Dict): Strict rules for the dock (e.g., `min_shelf_life_days`, `max_transit_temp_celsius`).
- `invoice` (List[Dict] | None): Revealed only after `request_invoice`. Shows what supplier billed.
- `scan_data` (List[Dict] | None): Revealed only after `request_scan`. Shows physical dock count, condition, and expiry dates.
- `cold_chain_log` (List[Dict] | None): Revealed only after `request_cold_chain`. Shows transit temperatures.
- `pending_skus` (List[str]): SKUs that still need an accept/reject decision.
- `resolved_skus` (List[str]): SKUs already decided upon.
- `message` (str): System feedback from the previous action.

##  Action Space

The agent can take the following actions by sending a JSON object containing an `action_type` and relevant parameters.

### Data Gathering Actions
- `{"action_type": "request_invoice"}` - Retrieves the supplier's invoice.
- `{"action_type": "request_scan"}` - Retrieves the physical receiving scan data.
- `{"action_type": "request_cold_chain"}` - Retrieves the transit temperature log.

### Decision Actions (One per SKU)
- `{"action_type": "accept_sku", "sku_id": "SKU001"}` - Accepts the specified SKU.
- `{"action_type": "flag_shortage", "sku_id": "SKU002", "shortage_qty": 5}` - Flags a shortage (must specify exact missing quantity).
- `{"action_type": "reject_sku", "sku_id": "SKU003", "reason": "damaged"}` - Rejects a SKU. Valid reasons: `"damaged"`, `"expired"`, `"shelf_life_violation"`, `"cold_chain_violation"`, `"unauthorized_substitution"`.

### Episode Control
- `{"action_type": "finalize"}` - Ends the episode and triggers the grader.

---

##  Setup & Usage Instructions

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/MaverickDev-J/darkstore-inbound.git
cd darkstore-inbound

# Install requirements via uv
uv sync
```

### 2. Running the Server Locally

```bash
uv run server
# The environment is now running at http://localhost:8000
# Web UI is available at http://localhost:8000/web
```

### 3. Running Baseline Inference

The `inference.py` script automatically runs an LLM agent against all 3 tasks.

1. Create a `.env` file in the project root:
```env
API_BASE_URL=https://api.lightning.ai/v1/
API_KEY=your_api_key_here
MODEL_NAME=gpt-oss-20b
ENV_URL=http://localhost:8000
```

2. Run the script:
```bash
python inference.py
```

### 4. Deploying to Hugging Face Spaces

```bash
openenv push --repo-id your-hf-username/darkstore-inbound
```

