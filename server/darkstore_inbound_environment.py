# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DarkStore Inbound Environment — Core Logic.

Simulates a dark store receiving dock where an AI agent must reconcile
delivery documents (PO, Invoice, Scan Data, Cold Chain Log) and make
accept/reject/chargeback decisions per SKU.

Implements the OpenEnv 3-method interface: reset(), step(), state.
"""

from uuid import uuid4
from datetime import datetime
from typing import Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        DarkstoreInboundAction,
        DarkstoreInboundObservation,
        DarkstoreInboundState,
        LineItem,
        ScanItem,
        ColdChainReading,
        PolicyRules,
    )
except ImportError:
    from models import (
        DarkstoreInboundAction,
        DarkstoreInboundObservation,
        DarkstoreInboundState,
        LineItem,
        ScanItem,
        ColdChainReading,
        PolicyRules,
    )

try:
    from .scenario_generator import generate_scenario, TASK_DIFFICULTY
    from .graders import grade_episode
except ImportError:
    from server.scenario_generator import generate_scenario, TASK_DIFFICULTY
    from server.graders import grade_episode


# Valid action types the agent can take
VALID_ACTION_TYPES = {
    "request_invoice",
    "request_scan",
    "request_cold_chain",
    "accept_sku",
    "flag_shortage",
    "reject_sku",
    "finalize",
}

# Valid reject reasons
VALID_REJECT_REASONS = {
    "expired",
    "damaged",
    "leaking",
    "cold_chain_violation",
    "unauthorized_substitution",
    "shelf_life_violation",
    "price_discrepancy",
}

# Default task if none specified
DEFAULT_TASK = "clean_delivery"
DEFAULT_SEED = 42
MAX_STEPS = 25


class DarkstoreInboundEnvironment(Environment):
    """
    Dark Store Inbound Receiving Agent Environment.

    The agent inspects delivery documents and makes per-SKU decisions.
    Documents are progressively revealed (agent must request them).

    Episode flow:
      1. reset() → agent sees PO + policy rules only
      2. Agent requests data: invoice, scan, cold_chain
      3. Agent makes decisions: accept/flag_shortage/reject per SKU
      4. Agent calls finalize → episode ends, grader runs
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize with empty state — actual setup happens in reset()."""
        self._state = DarkstoreInboundState(episode_id=str(uuid4()), step_count=0)

        # Scenario data (populated on reset)
        self._purchase_order: List[LineItem] = []
        self._invoice: List[LineItem] = []
        self._scan_data: List[ScanItem] = []
        self._cold_chain_log: List[ColdChainReading] = []
        self._policy_rules: Optional[PolicyRules] = None

        # Progressive reveal flags
        self._invoice_revealed: bool = False
        self._scan_revealed: bool = False
        self._cold_chain_revealed: bool = False

        # Agent's decisions and tracking
        self._agent_decisions: Dict[str, Dict] = {}
        self._pending_skus: List[str] = []
        self._resolved_skus: List[str] = []
        self._cumulative_reward: float = 0.0
        self._episode_done: bool = False
        self._last_message: str = ""

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> DarkstoreInboundObservation:
        """
        Reset the environment and generate a new delivery scenario.

        Args:
            seed: Random seed for reproducibility (default: 42)
            episode_id: Custom episode ID (auto-generated if not provided)
            **kwargs: May include 'task' to select difficulty:
                      'clean_delivery' (easy), 'quantity_mismatch' (medium),
                      'hidden_violation' (hard)

        Returns:
            Initial observation with PO and policy rules visible
        """
        # Parse task from kwargs or extra reset data
        task_name = kwargs.get("task", DEFAULT_TASK)
        actual_seed = seed if seed is not None else DEFAULT_SEED
        actual_episode_id = episode_id or str(uuid4())

        # Generate the scenario
        (
            self._purchase_order,
            self._invoice,
            self._scan_data,
            self._cold_chain_log,
            self._policy_rules,
            ground_truth,
        ) = generate_scenario(task_name, actual_seed)

        # Initialize state
        self._state = DarkstoreInboundState(
            episode_id=actual_episode_id,
            step_count=0,
            task_name=task_name,
            difficulty=TASK_DIFFICULTY.get(task_name, "easy"),
            scenario_seed=actual_seed,
            max_steps=MAX_STEPS,
            ground_truth=ground_truth,
            delivery_date=datetime.now().strftime("%Y-%m-%d"),
        )

        # Reset reveal flags
        self._invoice_revealed = False
        self._scan_revealed = False
        self._cold_chain_revealed = False

        # Reset agent tracking
        self._agent_decisions = {}
        self._pending_skus = [item.sku_id for item in self._purchase_order]
        self._resolved_skus = []
        self._cumulative_reward = 0.0
        self._episode_done = False
        self._last_message = (
            f"New delivery arrived at the receiving dock. "
            f"Task: {task_name} | {len(self._purchase_order)} SKUs to inspect. "
            f"Review the Purchase Order and policy rules, then request additional "
            f"data sources (invoice, scan, cold_chain) before making decisions."
        )

        return self._build_observation(reward=0.0, done=False)

    def step(self, action: DarkstoreInboundAction) -> DarkstoreInboundObservation:
        """
        Process one agent action and return the updated observation.

        Args:
            action: The agent's action (request data, accept/flag/reject SKU, or finalize)

        Returns:
            Updated observation with feedback, reward, and done status
        """
        if self._episode_done:
            return self._build_observation(
                reward=0.0,
                done=True,
                message="Episode already complete. Call reset() to start a new episode.",
            )

        self._state.step_count += 1
        action_type = action.action_type.strip().lower()

        # Check step budget
        if self._state.step_count > self._state.max_steps:
            return self._finalize_episode(
                message="Step budget exhausted. Episode ended automatically."
            )

        # Validate action type
        if action_type not in VALID_ACTION_TYPES:
            self._last_message = (
                f"Invalid action type: '{action_type}'. "
                f"Valid types: {sorted(VALID_ACTION_TYPES)}"
            )
            reward = -0.05
            self._cumulative_reward += reward
            return self._build_observation(reward=reward, done=False, message=self._last_message)

        # Dispatch action
        if action_type == "request_invoice":
            return self._handle_request_invoice()
        elif action_type == "request_scan":
            return self._handle_request_scan()
        elif action_type == "request_cold_chain":
            return self._handle_request_cold_chain()
        elif action_type == "accept_sku":
            return self._handle_accept_sku(action)
        elif action_type == "flag_shortage":
            return self._handle_flag_shortage(action)
        elif action_type == "reject_sku":
            return self._handle_reject_sku(action)
        elif action_type == "finalize":
            return self._finalize_episode(message="Agent finalized the inspection.")
        else:
            return self._build_observation(reward=-0.05, done=False, message="Unknown action.")

    # =========================================================================
    # Action Handlers
    # =========================================================================

    def _handle_request_invoice(self) -> DarkstoreInboundObservation:
        """Reveal the supplier invoice data."""
        if self._invoice_revealed:
            msg = "Invoice data already revealed."
            return self._build_observation(reward=0.0, done=False, message=msg)

        self._invoice_revealed = True
        msg = f"Invoice revealed: {len(self._invoice)} line items."
        return self._build_observation(reward=0.0, done=False, message=msg)

    def _handle_request_scan(self) -> DarkstoreInboundObservation:
        """Reveal the physical scan data."""
        if self._scan_revealed:
            msg = "Scan data already revealed."
            return self._build_observation(reward=0.0, done=False, message=msg)

        self._scan_revealed = True
        msg = f"Scan data revealed: {len(self._scan_data)} items scanned."
        return self._build_observation(reward=0.0, done=False, message=msg)

    def _handle_request_cold_chain(self) -> DarkstoreInboundObservation:
        """Reveal the cold chain temperature log."""
        if self._cold_chain_revealed:
            msg = "Cold chain log already revealed."
            return self._build_observation(reward=0.0, done=False, message=msg)

        self._cold_chain_revealed = True
        msg = f"Cold chain log revealed: {len(self._cold_chain_log)} temperature readings."
        return self._build_observation(reward=0.0, done=False, message=msg)

    def _handle_accept_sku(self, action: DarkstoreInboundAction) -> DarkstoreInboundObservation:
        """Accept a SKU as correctly delivered."""
        sku_id = action.sku_id
        if not sku_id:
            return self._build_observation(
                reward=-0.05, done=False,
                message="accept_sku requires a sku_id."
            )

        if sku_id not in self._pending_skus:
            if sku_id in self._resolved_skus:
                msg = f"SKU {sku_id} already resolved."
            else:
                msg = f"SKU {sku_id} not found in this delivery."
            return self._build_observation(reward=-0.05, done=False, message=msg)

        # Record decision
        self._agent_decisions[sku_id] = {"action": "accept"}
        self._pending_skus.remove(sku_id)
        self._resolved_skus.append(sku_id)

        # Uniform progress reward — does not leak ground truth.
        # Correctness is only revealed at finalize() via the grader.
        reward = 0.02
        msg = f"SKU {sku_id} accepted. Decision recorded."

        self._cumulative_reward += reward
        return self._build_observation(reward=reward, done=False, message=msg)

    def _handle_flag_shortage(self, action: DarkstoreInboundAction) -> DarkstoreInboundObservation:
        """Flag a quantity shortage for a SKU."""
        sku_id = action.sku_id
        shortage_qty = action.shortage_qty

        if not sku_id:
            return self._build_observation(
                reward=-0.05, done=False,
                message="flag_shortage requires a sku_id."
            )
        if shortage_qty is None or shortage_qty <= 0:
            return self._build_observation(
                reward=-0.05, done=False,
                message="flag_shortage requires a positive shortage_qty."
            )

        if sku_id not in self._pending_skus:
            if sku_id in self._resolved_skus:
                msg = f"SKU {sku_id} already resolved."
            else:
                msg = f"SKU {sku_id} not found in this delivery."
            return self._build_observation(reward=-0.05, done=False, message=msg)

        # Record decision
        self._agent_decisions[sku_id] = {
            "action": "flag_shortage",
            "shortage_qty": shortage_qty,
        }
        self._pending_skus.remove(sku_id)
        self._resolved_skus.append(sku_id)

        # Uniform progress reward — does not leak ground truth.
        # Correctness is only revealed at finalize() via the grader.
        reward = 0.02
        msg = f"SKU {sku_id} shortage flagged: {shortage_qty} units. Decision recorded."

        self._cumulative_reward += reward
        return self._build_observation(reward=reward, done=False, message=msg)

    def _handle_reject_sku(self, action: DarkstoreInboundAction) -> DarkstoreInboundObservation:
        """Reject a SKU with a reason."""
        sku_id = action.sku_id
        reason = action.reason

        if not sku_id:
            return self._build_observation(
                reward=-0.05, done=False,
                message="reject_sku requires a sku_id."
            )
        if not reason or reason not in VALID_REJECT_REASONS:
            return self._build_observation(
                reward=-0.05, done=False,
                message=f"reject_sku requires a valid reason. Valid: {sorted(VALID_REJECT_REASONS)}"
            )

        if sku_id not in self._pending_skus:
            if sku_id in self._resolved_skus:
                msg = f"SKU {sku_id} already resolved."
            else:
                msg = f"SKU {sku_id} not found in this delivery."
            return self._build_observation(reward=-0.05, done=False, message=msg)

        # Record decision
        self._agent_decisions[sku_id] = {"action": "reject", "reason": reason}
        self._pending_skus.remove(sku_id)
        self._resolved_skus.append(sku_id)

        # Uniform progress reward — does not leak ground truth.
        # Correctness is only revealed at finalize() via the grader.
        reward = 0.02
        msg = f"SKU {sku_id} rejected (reason: {reason}). Decision recorded."

        self._cumulative_reward += reward
        return self._build_observation(reward=reward, done=False, message=msg)

    # =========================================================================
    # Episode Finalization
    # =========================================================================

    def _finalize_episode(self, message: str) -> DarkstoreInboundObservation:
        """
        End the episode and compute the final grader score.

        Any pending SKUs that weren't acted on count as missed (no decision).
        The grader computes the final 0.0–1.0 score.
        """
        self._episode_done = True

        # Build data_requested dict for grader
        data_requested = {
            "invoice": self._invoice_revealed,
            "scan": self._scan_revealed,
            "cold_chain": self._cold_chain_revealed,
        }

        # Run the grader
        final_score = grade_episode(
            task_name=self._state.task_name,
            agent_decisions=self._agent_decisions,
            ground_truth=self._state.ground_truth,
            data_requested=data_requested,
        )

        # Build summary message
        total_skus = len(self._state.ground_truth)
        resolved = len(self._resolved_skus)
        missed = total_skus - resolved
        sources_checked = sum([self._invoice_revealed, self._scan_revealed, self._cold_chain_revealed])

        final_message = (
            f"{message} "
            f"| Resolved: {resolved}/{total_skus} SKUs "
            f"| Missed: {missed} "
            f"| Data sources checked: {sources_checked}/3 "
            f"| Final score: {final_score:.4f}"
        )

        return self._build_observation(
            reward=final_score,
            done=True,
            message=final_message,
        )

    # =========================================================================
    # Observation Builder
    # =========================================================================

    def _build_observation(
        self,
        reward: float,
        done: bool,
        message: Optional[str] = None,
    ) -> DarkstoreInboundObservation:
        """Build the observation object from current environment state."""
        return DarkstoreInboundObservation(
            # Always visible
            phase="complete" if done else "inspect",
            purchase_order=self._purchase_order,
            policy_rules=self._policy_rules,
            # Progressively revealed
            invoice=self._invoice if self._invoice_revealed else None,
            scan_data=self._scan_data if self._scan_revealed else None,
            cold_chain_log=self._cold_chain_log if self._cold_chain_revealed else None,
            # Status
            pending_skus=list(self._pending_skus),
            resolved_skus=list(self._resolved_skus),
            message=message or self._last_message,
            steps_remaining=max(0, self._state.max_steps - self._state.step_count),
            # OpenEnv base fields
            done=done,
            reward=reward,
        )

    # =========================================================================
    # State Property
    # =========================================================================

    @property
    def state(self) -> DarkstoreInboundState:
        """Get the current internal environment state."""
        return self._state
