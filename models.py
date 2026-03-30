# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the DarkStore Inbound Receiving Environment.

Defines typed Pydantic models for:
- Sub-models: LineItem, ScanItem, ColdChainReading, PolicyRules
- Action: 7 action types for inspecting and deciding on deliveries
- Observation: Progressive-reveal view of delivery documents
- State: Internal ground truth and episode configuration
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


# =============================================================================
# Sub-Models (data structures used inside Action/Observation)
# =============================================================================


class LineItem(BaseModel):
    """A single item line from a Purchase Order or Invoice."""

    sku_id: str = Field(..., description="Unique SKU identifier, e.g. 'SKU001'")
    name: str = Field(..., description="Product name, e.g. 'Amul Toned Milk 500ml'")
    quantity: int = Field(..., description="Number of units ordered/invoiced")
    unit_price: float = Field(..., description="Price per unit in INR")


class ScanItem(BaseModel):
    """Physical scan data for a single SKU at the receiving dock."""

    sku_id: str = Field(..., description="SKU identifier matching PO/Invoice")
    scanned_qty: int = Field(..., description="Actual physical count at dock")
    batch_number: str = Field(..., description="Manufacturer batch ID")
    expiry_date: str = Field(
        ..., description="Expiry date in ISO format, e.g. '2026-04-05'"
    )
    condition: str = Field(
        ..., description="Physical condition: 'good', 'damaged', or 'leaking'"
    )


class ColdChainReading(BaseModel):
    """A single temperature reading from the refrigerated transit log."""

    timestamp: str = Field(
        ..., description="ISO timestamp of the reading, e.g. '2026-03-30T08:15:00'"
    )
    temperature_celsius: float = Field(
        ..., description="Recorded temperature in Celsius"
    )


class PolicyRules(BaseModel):
    """Company receiving policy rules that the agent must enforce."""

    min_shelf_life_days: int = Field(
        ...,
        description="Minimum days until expiry required for acceptance (e.g. 4)",
    )
    max_transit_temp_celsius: float = Field(
        ...,
        description="Maximum allowed transit temperature in Celsius (e.g. 8.0)",
    )
    approved_substitutions: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of approved SKU substitutions: {original_sku: replacement_sku}",
    )


# =============================================================================
# Core OpenEnv Types
# =============================================================================


class DarkstoreInboundAction(Action):
    """
    Agent's action in the DarkStore Inbound environment.

    The agent must first request data sources (invoice, scan, cold_chain),
    then make per-SKU decisions (accept, flag_shortage, reject),
    and finally call finalize to end the episode.

    Action types:
        - request_invoice: Reveal the supplier invoice data
        - request_scan: Reveal the physical scan/count data
        - request_cold_chain: Reveal the cold chain temperature log
        - accept_sku: Accept a specific SKU as correctly delivered
        - flag_shortage: Flag a quantity shortage for a specific SKU
        - reject_sku: Reject a specific SKU with a reason
        - finalize: Complete the inspection and end the episode
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: 'request_invoice', 'request_scan', 'request_cold_chain', "
            "'accept_sku', 'flag_shortage', 'reject_sku', 'finalize'"
        ),
    )
    sku_id: Optional[str] = Field(
        default=None,
        description="Target SKU for accept/flag/reject actions",
    )
    reason: Optional[str] = Field(
        default=None,
        description=(
            "Reason for rejection. One of: 'expired', 'damaged', 'leaking', "
            "'cold_chain_violation', 'unauthorized_substitution', 'shelf_life_violation'"
        ),
    )
    shortage_qty: Optional[int] = Field(
        default=None,
        description="Quantity short for flag_shortage action (ordered - received)",
    )


class DarkstoreInboundObservation(Observation):
    """
    What the agent sees after each action.

    Starts with only the purchase_order and policy_rules visible.
    Invoice, scan_data, and cold_chain_log are revealed when requested.
    """

    # --- Always visible ---
    phase: str = Field(
        default="inspect",
        description="Current phase: 'inspect' (gathering data) or 'complete' (episode over)",
    )
    purchase_order: List[LineItem] = Field(
        default_factory=list,
        description="The purchase order — what was ordered",
    )
    policy_rules: Optional[PolicyRules] = Field(
        default=None,
        description="Company policy rules the agent must enforce",
    )

    # --- Revealed on request ---
    invoice: Optional[List[LineItem]] = Field(
        default=None,
        description="Supplier invoice — revealed after 'request_invoice' action",
    )
    scan_data: Optional[List[ScanItem]] = Field(
        default=None,
        description="Physical scan data — revealed after 'request_scan' action",
    )
    cold_chain_log: Optional[List[ColdChainReading]] = Field(
        default=None,
        description="Temperature log — revealed after 'request_cold_chain' action",
    )

    # --- Status tracking ---
    pending_skus: List[str] = Field(
        default_factory=list,
        description="SKU IDs that haven't been acted on yet",
    )
    resolved_skus: List[str] = Field(
        default_factory=list,
        description="SKU IDs that have been accepted/flagged/rejected",
    )
    message: str = Field(
        default="",
        description="Feedback message from the last action",
    )
    steps_remaining: int = Field(
        default=25,
        description="Number of steps the agent has left before forced termination",
    )


class DarkstoreInboundState(State):
    """
    Internal environment state — not visible to the agent.

    Contains ground truth answers and episode configuration.
    Inherits episode_id and step_count from State.
    """

    task_name: str = Field(
        default="",
        description="Task identifier: 'clean_delivery', 'quantity_mismatch', or 'hidden_violation'",
    )
    difficulty: str = Field(
        default="easy",
        description="Difficulty level: 'easy', 'medium', or 'hard'",
    )
    scenario_seed: int = Field(
        default=0,
        description="Random seed used to generate this scenario",
    )
    max_steps: int = Field(
        default=25,
        description="Maximum number of steps allowed in this episode",
    )
    ground_truth: Dict[str, Dict] = Field(
        default_factory=dict,
        description=(
            "Expected correct action per SKU. Format: "
            "{'SKU001': {'action': 'accept'}, "
            "'SKU002': {'action': 'flag_shortage', 'shortage_qty': 5}, ...}"
        ),
    )
    delivery_date: str = Field(
        default="2026-04-01",
        description="Simulated date of delivery (for shelf life checks)",
    )
