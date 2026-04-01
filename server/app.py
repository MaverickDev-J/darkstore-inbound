# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the DarkStore Inbound Environment.

Endpoints:
    - POST /reset: Reset the environment (accepts task, seed)
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uv run server
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    from ..models import DarkstoreInboundAction, DarkstoreInboundObservation
    from .darkstore_inbound_environment import DarkstoreInboundEnvironment
except (ImportError, ModuleNotFoundError):
    from models import DarkstoreInboundAction, DarkstoreInboundObservation
    from server.darkstore_inbound_environment import DarkstoreInboundEnvironment


app = create_app(
    DarkstoreInboundEnvironment,
    DarkstoreInboundAction,
    DarkstoreInboundObservation,
    env_name="darkstore_inbound",
    max_concurrent_envs=1,
)

def main():
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
