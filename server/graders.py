# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graders for the DarkStore Inbound Environment.

Three deterministic graders, one per task difficulty.
Each compares the agent's decisions against ground truth and returns 0.0–1.0.

Scoring philosophy:
- Partial credit for correct action type even if details are wrong
- Bonus for proactively requesting data sources
- Penalty for false rejections and missed violations
"""

from typing import Dict, List


def grade_clean_delivery(
    agent_decisions: Dict[str, Dict],
    ground_truth: Dict[str, Dict],
    data_requested: Dict[str, bool],
) -> float:
    """
    Grade the Easy task: clean delivery where everything matches.

    Score breakdown (weights sum to 1.0):
      - 70%: Correctly accepted SKUs / total SKUs
      - 30%: No false flags (1.0 if no incorrect rejections/flags, 0.0 otherwise)

    Args:
        agent_decisions: Agent's per-SKU decisions, e.g. {"SKU001": {"action": "accept"}}
        ground_truth: Expected correct decisions
        data_requested: Which data sources the agent requested

    Returns:
        Score between 0.0 and 1.0
    """
    if not ground_truth:
        return 0.0

    total_skus = len(ground_truth)
    correct_accepts = 0
    false_flags = 0

    for sku_id, expected in ground_truth.items():
        agent_decision = agent_decisions.get(sku_id, {})
        agent_action = agent_decision.get("action", "none")

        if expected["action"] == "accept":
            if agent_action == "accept":
                correct_accepts += 1
            elif agent_action in ("flag_shortage", "reject"):
                false_flags += 1

    accept_score = correct_accepts / total_skus if total_skus > 0 else 0.0
    no_false_flags_score = 1.0 if false_flags == 0 else 0.0

    score = (accept_score * 0.7) + (no_false_flags_score * 0.3)

    # Anti-gaming: penalize agents that false-flag/reject clean items
    if total_skus > 0:
        false_rejection_rate = false_flags / total_skus
        score *= (1.0 - 0.5 * false_rejection_rate)

    return round(min(max(score, 0.0), 1.0), 4)


def grade_quantity_mismatch(
    agent_decisions: Dict[str, Dict],
    ground_truth: Dict[str, Dict],
    data_requested: Dict[str, bool],
) -> float:
    """
    Grade the Medium task: shortages + unauthorized substitution.

    Score breakdown (weights sum to 1.0):
      - 40%: Correctly flagged discrepancies (shortages + substitution) / total discrepancies
      - 25%: Shortage quantity accuracy (within 20% tolerance)
      - 20%: Correctly accepted clean items / total clean items
      - 15%: Data thoroughness (requested invoice + scan data)

    Args:
        agent_decisions: Agent's per-SKU decisions
        ground_truth: Expected correct decisions
        data_requested: Which data sources the agent requested

    Returns:
        Score between 0.0 and 1.0
    """
    if not ground_truth:
        return 0.0

    # Separate ground truth into discrepancies and clean items
    discrepancies = {k: v for k, v in ground_truth.items() if v["action"] != "accept"}
    clean_items = {k: v for k, v in ground_truth.items() if v["action"] == "accept"}

    # 1. Correctly flagged discrepancies
    correct_flags = 0
    for sku_id, expected in discrepancies.items():
        agent_decision = agent_decisions.get(sku_id, {})
        agent_action = agent_decision.get("action", "none")

        if expected["action"] == "flag_shortage" and agent_action == "flag_shortage":
            correct_flags += 1
        elif expected["action"] == "reject" and agent_action == "reject":
            correct_flags += 1

    flag_score = correct_flags / len(discrepancies) if discrepancies else 1.0

    # 2. Shortage quantity accuracy
    shortage_items = {k: v for k, v in discrepancies.items() if v["action"] == "flag_shortage"}
    qty_correct = 0
    qty_total = len(shortage_items)

    for sku_id, expected in shortage_items.items():
        agent_decision = agent_decisions.get(sku_id, {})
        if agent_decision.get("action") == "flag_shortage":
            expected_qty = expected.get("shortage_qty", 0)
            agent_qty = agent_decision.get("shortage_qty", 0)
            if expected_qty > 0:
                error = abs(agent_qty - expected_qty) / expected_qty
                if error <= 0.2:  # Within 20% tolerance
                    qty_correct += 1

    qty_score = qty_correct / qty_total if qty_total > 0 else 1.0

    # 3. Correctly accepted clean items
    correct_accepts = 0
    for sku_id in clean_items:
        agent_decision = agent_decisions.get(sku_id, {})
        if agent_decision.get("action") == "accept":
            correct_accepts += 1

    accept_score = correct_accepts / len(clean_items) if clean_items else 1.0

    # 4. Data thoroughness
    thoroughness = 0.0
    if data_requested.get("invoice", False):
        thoroughness += 0.5
    if data_requested.get("scan", False):
        thoroughness += 0.5

    score = (flag_score * 0.40) + (qty_score * 0.25) + (accept_score * 0.20) + (thoroughness * 0.15)

    # Anti-gaming: penalize agents that incorrectly flag/reject clean items
    if clean_items:
        false_rejections = sum(
            1 for sku_id in clean_items
            if agent_decisions.get(sku_id, {}).get("action") in ("reject", "flag_shortage")
        )
        false_rejection_rate = false_rejections / len(clean_items)
        score *= (1.0 - 0.5 * false_rejection_rate)

    return round(min(max(score, 0.0), 1.0), 4)


def grade_hidden_violation(
    agent_decisions: Dict[str, Dict],
    ground_truth: Dict[str, Dict],
    data_requested: Dict[str, bool],
) -> float:
    """
    Grade the Hard task: hidden safety violations.

    Score breakdown (weights sum to 1.0):
      - 35%: Detected safety violations (rejected with correct reason) / total violations
      - 25%: Proactively checked cold chain (critical for this task)
      - 20%: Correctly handled other SKUs (accepted clean items) / total clean
      - 10%: Detected violations with correct reason codes
      - 10%: Data thoroughness (requested all 3 data sources)

    Args:
        agent_decisions: Agent's per-SKU decisions
        ground_truth: Expected correct decisions
        data_requested: Which data sources the agent requested

    Returns:
        Score between 0.0 and 1.0
    """
    if not ground_truth:
        return 0.0

    violations = {k: v for k, v in ground_truth.items() if v["action"] == "reject"}
    clean_items = {k: v for k, v in ground_truth.items() if v["action"] == "accept"}

    # 1. Detected violations (agent rejected when should have rejected)
    detected = 0
    for sku_id, expected in violations.items():
        agent_decision = agent_decisions.get(sku_id, {})
        if agent_decision.get("action") == "reject":
            detected += 1

    violation_score = detected / len(violations) if violations else 1.0

    # 2. Proactively checked cold chain
    cold_chain_score = 1.0 if data_requested.get("cold_chain", False) else 0.0

    # 3. Correctly accepted clean items
    correct_accepts = 0
    for sku_id in clean_items:
        agent_decision = agent_decisions.get(sku_id, {})
        if agent_decision.get("action") == "accept":
            correct_accepts += 1

    accept_score = correct_accepts / len(clean_items) if clean_items else 1.0

    # 4. Correct reason codes for violations
    correct_reasons = 0
    for sku_id, expected in violations.items():
        agent_decision = agent_decisions.get(sku_id, {})
        if (
            agent_decision.get("action") == "reject"
            and agent_decision.get("reason") == expected.get("reason")
        ):
            correct_reasons += 1

    reason_score = correct_reasons / len(violations) if violations else 1.0

    # 5. Data thoroughness (all 3 sources)
    thoroughness = 0.0
    sources_checked = 0
    for source in ["invoice", "scan", "cold_chain"]:
        if data_requested.get(source, False):
            sources_checked += 1
    thoroughness = sources_checked / 3.0

    score = (
        (violation_score * 0.35)
        + (cold_chain_score * 0.25)
        + (accept_score * 0.20)
        + (reason_score * 0.10)
        + (thoroughness * 0.10)
    )

    # Anti-gaming: penalize agents that blindly reject clean items
    # This prevents the "reject everything" exploit (previously scored 73%)
    if clean_items:
        false_rejections = sum(
            1 for sku_id in clean_items
            if agent_decisions.get(sku_id, {}).get("action") in ("reject", "flag_shortage")
        )
        false_rejection_rate = false_rejections / len(clean_items)
        score *= (1.0 - 0.5 * false_rejection_rate)

    return round(min(max(score, 0.0), 1.0), 4)


# =============================================================================
# Dispatcher
# =============================================================================

TASK_GRADERS = {
    "clean_delivery": grade_clean_delivery,
    "quantity_mismatch": grade_quantity_mismatch,
    "hidden_violation": grade_hidden_violation,
}


def grade_episode(
    task_name: str,
    agent_decisions: Dict[str, Dict],
    ground_truth: Dict[str, Dict],
    data_requested: Dict[str, bool],
) -> float:
    """
    Grade an episode for the given task.

    Args:
        task_name: One of 'clean_delivery', 'quantity_mismatch', 'hidden_violation'
        agent_decisions: Agent's per-SKU decisions
        ground_truth: Expected correct decisions from scenario generator
        data_requested: Which data sources the agent requested

    Returns:
        Score between 0.0 and 1.0

    Raises:
        ValueError: if task_name is not recognized
    """
    if task_name not in TASK_GRADERS:
        raise ValueError(
            f"Unknown task: '{task_name}'. Must be one of: {list(TASK_GRADERS.keys())}"
        )
    try:
        return TASK_GRADERS[task_name](agent_decisions, ground_truth, data_requested)
    except Exception:
        # Safety net — never crash, never return NaN
        return 0.0
