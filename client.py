# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DarkStore Inbound Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DarkstoreInboundAction, DarkstoreInboundObservation


class DarkstoreInboundEnv(
    EnvClient[DarkstoreInboundAction, DarkstoreInboundObservation, State]
):
    """
    Client for the DarkStore Inbound Environment.

    Example:
        >>> with DarkstoreInboundEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset()
        ...     print(result.observation.message)
        ...     result = env.step(DarkstoreInboundAction(action_type="request_invoice"))
        ...     print(result.observation.invoice)
    """

    def _step_payload(self, action: DarkstoreInboundAction) -> Dict:
        """Convert action to JSON payload."""
        payload = {"action_type": action.action_type}
        if action.sku_id is not None:
            payload["sku_id"] = action.sku_id
        if action.reason is not None:
            payload["reason"] = action.reason
        if action.shortage_qty is not None:
            payload["shortage_qty"] = action.shortage_qty
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[DarkstoreInboundObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = DarkstoreInboundObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
