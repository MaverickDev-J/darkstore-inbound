# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scenario Generator for the DarkStore Inbound Environment.

Generates seeded, reproducible delivery scenarios with deliberate discrepancies.
Each generator returns all documents + the ground truth expected from the agent.

Three difficulty levels:
  - Easy (clean_delivery): Everything matches, no violations.
  - Medium (quantity_mismatch): Shortages + unauthorized substitution.
  - Hard (hidden_violation): Surface looks clean, cold chain spike + shelf life issue.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

try:
    from ..models import (
        ColdChainReading,
        LineItem,
        PolicyRules,
        ScanItem,
    )
except ImportError:
    from models import (
        ColdChainReading,
        LineItem,
        PolicyRules,
        ScanItem,
    )

# =============================================================================
# Product catalog — realistic Indian quick-commerce dark store items
# =============================================================================

PRODUCT_CATALOG = [
    ("SKU001", "Amul Toned Milk 500ml", 28.0),
    ("SKU002", "Britannia White Bread 400g", 45.0),
    ("SKU003", "Haldiram Aloo Bhujia 200g", 85.0),
    ("SKU004", "Parle-G Biscuits 800g", 55.0),
    ("SKU005", "Mother Dairy Curd 400g", 35.0),
    ("SKU006", "Nestle Maggi 2-Min Noodles 4-Pack", 52.0),
    ("SKU007", "Dabur Real Mixed Fruit Juice 1L", 110.0),
    ("SKU008", "Lays Classic Salted Chips 52g", 20.0),
    ("SKU009", "Amul Butter 100g", 56.0),
    ("SKU010", "Surf Excel Easy Wash 500g", 125.0),
    ("SKU011", "Kissan Mixed Fruit Jam 200g", 99.0),
    ("SKU012", "Tata Salt 1kg", 28.0),
    ("SKU013", "Fortune Sunlite Oil 1L", 155.0),
    ("SKU014", "Cadbury Dairy Milk Silk 150g", 170.0),
    ("SKU015", "Paper Boat Aamras 200ml 6-Pack", 180.0),
]


DELIVERY_DATE = "2026-04-01"  # Fixed simulated delivery date


def _make_batch_number(rng: random.Random) -> str:
    """Generate a realistic batch number like 'B2026-0315-A7'."""
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    suffix = rng.choice("ABCDEFGH") + str(rng.randint(1, 9))
    return f"B2026-{month:02d}{day:02d}-{suffix}"


def _make_expiry(rng: random.Random, days_ahead: int) -> str:
    """Generate an expiry date N days after the delivery date."""
    base = datetime.strptime(DELIVERY_DATE, "%Y-%m-%d")
    expiry = base + timedelta(days=days_ahead)
    return expiry.strftime("%Y-%m-%d")


def _generate_clean_cold_chain(rng: random.Random, n_readings: int = 12) -> List[ColdChainReading]:
    """Generate a cold chain log with all temperatures within safe range (2-6°C)."""
    base_time = datetime.strptime(f"{DELIVERY_DATE}T06:00:00", "%Y-%m-%dT%H:%M:%S")
    readings = []
    for i in range(n_readings):
        ts = base_time + timedelta(minutes=30 * i)
        temp = round(rng.uniform(2.0, 6.0), 1)
        readings.append(
            ColdChainReading(
                timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S"),
                temperature_celsius=temp,
            )
        )
    return readings


# =============================================================================
# EASY: Clean Delivery — everything matches, no violations
# =============================================================================


def generate_easy_scenario(
    seed: int,
) -> Tuple[
    List[LineItem],       # purchase_order
    List[LineItem],       # invoice
    List[ScanItem],       # scan_data
    List[ColdChainReading],  # cold_chain_log
    PolicyRules,          # policy_rules
    Dict[str, Dict],     # ground_truth
]:
    """
    Generate a clean delivery scenario (Easy task).

    5 SKUs, all quantities match across PO/Invoice/Scan.
    All items have good shelf life and condition.
    Cold chain is clean.
    Correct agent response: accept all 5 SKUs.
    """
    rng = random.Random(seed)
    products = rng.sample(PRODUCT_CATALOG, 5)

    purchase_order = []
    invoice = []
    scan_data = []
    ground_truth = {}

    for sku_id, name, price in products:
        qty = rng.randint(10, 50)

        purchase_order.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
        invoice.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
        scan_data.append(
            ScanItem(
                sku_id=sku_id,
                scanned_qty=qty,
                batch_number=_make_batch_number(rng),
                expiry_date=_make_expiry(rng, days_ahead=rng.randint(30, 180)),
                condition="good",
            )
        )
        ground_truth[sku_id] = {"action": "accept"}

    cold_chain_log = _generate_clean_cold_chain(rng)

    policy_rules = PolicyRules(
        min_shelf_life_days=4,
        max_transit_temp_celsius=8.0,
        approved_substitutions={},
    )

    return purchase_order, invoice, scan_data, cold_chain_log, policy_rules, ground_truth


# =============================================================================
# MEDIUM: Quantity Mismatch — shortages + unauthorized substitution
# =============================================================================


def generate_medium_scenario(
    seed: int,
) -> Tuple[
    List[LineItem],
    List[LineItem],
    List[ScanItem],
    List[ColdChainReading],
    PolicyRules,
    Dict[str, Dict],
]:
    """
    Generate a scenario with discrepancies (Medium task).

    8 SKUs total:
    - 5 clean items (quantities match everywhere)
    - 2 items with quantity shortages (scan < PO, invoice matches scan — supplier short-shipped)
    - 1 item is an unauthorized substitution (invoice has different SKU than PO)

    Correct agent response:
    - Accept 5 clean items
    - Flag shortages for 2 items with correct shortage amounts
    - Reject 1 item as unauthorized substitution
    """
    rng = random.Random(seed)
    products = rng.sample(PRODUCT_CATALOG, 9)  # 8 used + 1 for substitution

    purchase_order = []
    invoice = []
    scan_data = []
    ground_truth = {}

    # Pick indices for discrepancies
    shortage_indices = [0, 1]  # First two products will have shortages
    sub_index = 2              # Third product will be substituted
    sub_product = products[8]  # Use 9th product as the unauthorized substitution

    for i, (sku_id, name, price) in enumerate(products[:8]):
        qty = rng.randint(20, 60)

        if i in shortage_indices:
            # Shortage: PO says qty, but scan shows fewer
            short_amount = rng.randint(3, min(10, qty - 1))
            actual_qty = qty - short_amount

            purchase_order.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            # Invoice matches the short amount (supplier invoiced what they sent)
            invoice.append(LineItem(sku_id=sku_id, name=name, quantity=actual_qty, unit_price=price))
            scan_data.append(
                ScanItem(
                    sku_id=sku_id,
                    scanned_qty=actual_qty,
                    batch_number=_make_batch_number(rng),
                    expiry_date=_make_expiry(rng, days_ahead=rng.randint(30, 180)),
                    condition="good",
                )
            )
            ground_truth[sku_id] = {
                "action": "flag_shortage",
                "shortage_qty": short_amount,
            }

        elif i == sub_index:
            # Unauthorized substitution: PO has one product, invoice/scan has another
            sub_sku, sub_name, sub_price = sub_product

            purchase_order.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            # Invoice shows substituted product
            invoice.append(LineItem(sku_id=sub_sku, name=sub_name, quantity=qty, unit_price=sub_price))
            scan_data.append(
                ScanItem(
                    sku_id=sub_sku,
                    scanned_qty=qty,
                    batch_number=_make_batch_number(rng),
                    expiry_date=_make_expiry(rng, days_ahead=rng.randint(30, 180)),
                    condition="good",
                )
            )
            ground_truth[sku_id] = {
                "action": "reject",
                "reason": "unauthorized_substitution",
            }

        else:
            # Clean item — everything matches
            purchase_order.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            invoice.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            scan_data.append(
                ScanItem(
                    sku_id=sku_id,
                    scanned_qty=qty,
                    batch_number=_make_batch_number(rng),
                    expiry_date=_make_expiry(rng, days_ahead=rng.randint(30, 180)),
                    condition="good",
                )
            )
            ground_truth[sku_id] = {"action": "accept"}

    cold_chain_log = _generate_clean_cold_chain(rng)

    policy_rules = PolicyRules(
        min_shelf_life_days=4,
        max_transit_temp_celsius=8.0,
        approved_substitutions={},  # No approved substitutions — the sub is unauthorized
    )

    return purchase_order, invoice, scan_data, cold_chain_log, policy_rules, ground_truth


# =============================================================================
# HARD: Hidden Violation — surface clean, cold chain + shelf life problems
# =============================================================================


def generate_hard_scenario(
    seed: int,
) -> Tuple[
    List[LineItem],
    List[LineItem],
    List[ScanItem],
    List[ColdChainReading],
    PolicyRules,
    Dict[str, Dict],
]:
    """
    Generate a scenario with hidden safety violations (Hard task).

    10 SKUs total:
    - 7 clean items (everything matches, good condition, good shelf life)
    - 1 item with a damaged condition (visible in scan data)
    - 1 item with a shelf life violation (expiry too close — must check against policy)
    - 1 item that looks clean in docs, BUT cold chain log has a hidden temperature spike

    The cold chain spike affects ALL perishable items, but the agent must:
    1. Proactively request the cold chain log
    2. Detect the temperature spike
    3. Reject the affected perishable SKU(s)

    This tests whether the agent proactively checks auxiliary data that nobody asked about.
    """
    rng = random.Random(seed)
    products = rng.sample(PRODUCT_CATALOG, 10)

    purchase_order = []
    invoice = []
    scan_data = []
    ground_truth = {}

    damaged_index = 0     # First item: physically damaged
    shelf_life_index = 1  # Second item: shelf life violation
    cold_chain_index = 2  # Third item: perishable affected by cold chain spike

    # Perishable SKUs that would be affected by cold chain violation
    perishable_names = {"milk", "curd", "butter", "juice", "aamras"}

    for i, (sku_id, name, price) in enumerate(products):
        qty = rng.randint(15, 50)

        if i == damaged_index:
            # Damaged item — visible in scan condition
            purchase_order.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            invoice.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            scan_data.append(
                ScanItem(
                    sku_id=sku_id,
                    scanned_qty=qty,
                    batch_number=_make_batch_number(rng),
                    expiry_date=_make_expiry(rng, days_ahead=rng.randint(30, 180)),
                    condition="damaged",
                )
            )
            ground_truth[sku_id] = {"action": "reject", "reason": "damaged"}

        elif i == shelf_life_index:
            # Shelf life violation — expiry is only 2 days away (policy requires 4)
            purchase_order.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            invoice.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            scan_data.append(
                ScanItem(
                    sku_id=sku_id,
                    scanned_qty=qty,
                    batch_number=_make_batch_number(rng),
                    expiry_date=_make_expiry(rng, days_ahead=2),  # Only 2 days! Policy requires 4.
                    condition="good",
                )
            )
            ground_truth[sku_id] = {"action": "reject", "reason": "shelf_life_violation"}

        elif i == cold_chain_index:
            # This item looks perfectly clean on paper, but the cold chain has a spike
            # making it unsafe. Agent must proactively check cold chain log.
            purchase_order.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            invoice.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            scan_data.append(
                ScanItem(
                    sku_id=sku_id,
                    scanned_qty=qty,
                    batch_number=_make_batch_number(rng),
                    expiry_date=_make_expiry(rng, days_ahead=rng.randint(30, 180)),
                    condition="good",
                )
            )
            ground_truth[sku_id] = {"action": "reject", "reason": "cold_chain_violation"}

        else:
            # Clean item
            purchase_order.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            invoice.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            scan_data.append(
                ScanItem(
                    sku_id=sku_id,
                    scanned_qty=qty,
                    batch_number=_make_batch_number(rng),
                    expiry_date=_make_expiry(rng, days_ahead=rng.randint(30, 180)),
                    condition="good",
                )
            )
            ground_truth[sku_id] = {"action": "accept"}

    # Cold chain log: mostly clean BUT with a hidden temperature spike
    cold_chain_log = _generate_cold_chain_with_spike(rng)

    policy_rules = PolicyRules(
        min_shelf_life_days=4,
        max_transit_temp_celsius=8.0,
        approved_substitutions={},
    )

    return purchase_order, invoice, scan_data, cold_chain_log, policy_rules, ground_truth


def _generate_cold_chain_with_spike(
    rng: random.Random, n_readings: int = 18
) -> List[ColdChainReading]:
    """
    Generate a cold chain log with a hidden temperature spike.

    Most readings are in safe range (2-6°C).
    3 consecutive readings in the middle spike to 12-15°C (above 8°C threshold).
    This simulates a refrigeration failure during transit.
    """
    base_time = datetime.strptime(f"{DELIVERY_DATE}T04:00:00", "%Y-%m-%dT%H:%M:%S")

    # Spike happens roughly in the middle of transit
    spike_start = n_readings // 3
    spike_end = spike_start + 3

    readings = []
    for i in range(n_readings):
        ts = base_time + timedelta(minutes=20 * i)

        if spike_start <= i < spike_end:
            # Temperature spike! Above the 8°C threshold
            temp = round(rng.uniform(12.0, 15.5), 1)
        else:
            # Normal safe temperature
            temp = round(rng.uniform(2.0, 6.0), 1)

        readings.append(
            ColdChainReading(
                timestamp=ts.strftime("%Y-%m-%dT%H:%M:%S"),
                temperature_celsius=temp,
            )
        )

    return readings


# =============================================================================
# Generator dispatcher
# =============================================================================


TASK_GENERATORS = {
    "clean_delivery": generate_easy_scenario,
    "quantity_mismatch": generate_medium_scenario,
    "hidden_violation": generate_hard_scenario,
}

TASK_DIFFICULTY = {
    "clean_delivery": "easy",
    "quantity_mismatch": "medium",
    "hidden_violation": "hard",
}


def generate_scenario(task_name: str, seed: int) -> Tuple:
    """
    Generate a scenario for the given task.

    Args:
        task_name: One of 'clean_delivery', 'quantity_mismatch', 'hidden_violation'
        seed: Random seed for reproducibility

    Returns:
        Tuple of (purchase_order, invoice, scan_data, cold_chain_log, policy_rules, ground_truth)

    Raises:
        ValueError: if task_name is not recognized
    """
    if task_name not in TASK_GENERATORS:
        raise ValueError(
            f"Unknown task: '{task_name}'. Must be one of: {list(TASK_GENERATORS.keys())}"
        )
    return TASK_GENERATORS[task_name](seed)
