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


def _get_delivery_date() -> str:
    """Return today's date as simulated delivery date."""
    return datetime.now().strftime("%Y-%m-%d")


def _make_batch_number(rng: random.Random) -> str:
    """Generate a realistic batch number like 'B2026-0315-A7'."""
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    suffix = rng.choice("ABCDEFGH") + str(rng.randint(1, 9))
    return f"B2026-{month:02d}{day:02d}-{suffix}"


def _make_expiry(rng: random.Random, days_ahead: int) -> str:
    """Generate an expiry date N days after the delivery date."""
    base = datetime.strptime(_get_delivery_date(), "%Y-%m-%d")
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

    4-6 SKUs, all quantities match. Good shelf life and condition.
    Cold chain is clean. Correct agent response: accept all SKUs.
    """
    rng = random.Random(seed)
    n_skus = rng.randint(4, 6)
    products = rng.sample(PRODUCT_CATALOG, n_skus)

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

    6-10 SKUs total. Violations include 1-3 quantity shortages and
    0-2 unauthorized substitutions at randomized positions.

    Correct agent response:
    - Accept clean items
    - Flag shortages with correct amount
    - Reject substitutions
    """
    rng = random.Random(seed)
    n_skus = rng.randint(6, 10)
    n_shortages = rng.randint(1, min(3, n_skus - 2))
    n_subs = rng.randint(0, min(2, n_skus - n_shortages - 2))
    n_violations = n_shortages + n_subs

    # Need extra products for substitution replacements
    products = rng.sample(PRODUCT_CATALOG, min(n_skus + n_subs, len(PRODUCT_CATALOG)))
    base_products = products[:n_skus]
    sub_replacements = products[n_skus:n_skus + n_subs]

    # Randomize which positions get each violation type
    all_indices = list(range(n_skus))
    rng.shuffle(all_indices)
    shortage_indices = set(all_indices[:n_shortages])
    sub_indices = set(all_indices[n_shortages:n_shortages + n_subs])
    sub_iter = iter(sub_replacements)

    purchase_order = []
    invoice = []
    scan_data = []
    ground_truth = {}

    for i, (sku_id, name, price) in enumerate(base_products):
        qty = rng.randint(20, 60)

        if i in shortage_indices:
            short_amount = rng.randint(3, min(10, qty - 1))
            actual_qty = qty - short_amount

            purchase_order.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
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

        elif i in sub_indices:
            sub_sku_id, sub_name, sub_price = next(sub_iter)
            purchase_order.append(LineItem(sku_id=sku_id, name=name, quantity=qty, unit_price=price))
            invoice.append(LineItem(sku_id=sub_sku_id, name=sub_name, quantity=qty, unit_price=sub_price))
            scan_data.append(
                ScanItem(
                    sku_id=sub_sku_id,
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

    8-12 SKUs total with randomized violation positions:
    - 1-2 damaged items (visible in scan condition)
    - 1 item with a shelf life violation (expiry too close)
    - 1-2 items in the affected cold chain compartment

    The cold chain spike is localized to one truck compartment.
    Only SKUs assigned to that compartment are affected.
    The agent must: (1) request the cold chain log, (2) find the spike,
    (3) reject all SKUs whose compartment was affected.
    """
    rng = random.Random(seed)
    n_skus = rng.randint(8, 12)
    products = rng.sample(PRODUCT_CATALOG, n_skus)

    # Randomize how many of each violation type
    n_damaged = rng.randint(1, 2)
    n_cold_chain = rng.randint(1, 2)
    # Always exactly 1 shelf life violation

    # Shuffle indices and assign violations
    all_indices = list(range(n_skus))
    rng.shuffle(all_indices)
    damaged_indices = set(all_indices[:n_damaged])
    shelf_life_index = all_indices[n_damaged]
    cold_chain_indices = set(all_indices[n_damaged + 1:n_damaged + 1 + n_cold_chain])

    purchase_order = []
    invoice = []
    scan_data = []
    ground_truth = {}

    for i, (sku_id, name, price) in enumerate(products):
        qty = rng.randint(15, 50)

        if i in damaged_indices:
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

        elif i in cold_chain_indices:
            # These items are in the affected cold chain compartment.
            # Surface documents look fine, but cold chain log has a spike.
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
