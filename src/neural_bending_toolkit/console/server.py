"""WebSocket server/session handling for the Neural Bending Console runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol

from neural_bending_toolkit.console.curate import curate_take
from neural_bending_toolkit.console.runtime import ConsoleRuntime, node_specs
from neural_bending_toolkit.console.schema import PatchGraph, validate_patch_graph


class JsonSocket(Protocol):
    async def send_json(self, payload: dict[str, Any]) -> None: ...


class ConsoleSession:
    """Holds singleton runtime session for websocket clients."""

    def __init__(self) -> None:
        self.patch: PatchGraph | None = None
        self.runtime: ConsoleRuntime | None = None
        self._lock = asyncio.Lock()

    async def handle_message(self, websocket: JsonSocket, message: dict[str, Any]) -> None:
        msg_type = message.get("type")
        try:
            if msg_type == "PATCH_LOAD":
                await self._patch_load(websocket, message["patch"])
            elif msg_type == "PATCH_UPDATE":
                patch_payload = message.get("full_patch") or message.get("patch_delta")
                await self._patch_load(websocket, patch_payload)
            elif msg_type == "RUNTIME_START":
                await self._runtime_start(websocket, message)
            elif msg_type == "RUNTIME_STOP":
                await self._runtime_stop(websocket)
            elif msg_type == "TAKE":
                await self._take(websocket, message.get("label"))
            elif msg_type == "PARAM_SET":
                await self._param_set(websocket, message)
            elif msg_type == "CURATE_TAKE":
                await self._curate_take(websocket, message)
            else:
                await websocket.send_json(
                    {
                        "type": "ERROR",
                        "message": f"unknown message type '{msg_type}'",
                    }
                )
                return
            await websocket.send_json({"type": "ACK", "message_type": msg_type})
        except Exception as err:
            await websocket.send_json({"type": "ERROR", "message": str(err)})

    async def _patch_load(self, websocket: JsonSocket, patch_payload: dict[str, Any]) -> None:
        patch = validate_patch_graph(patch_payload, node_specs=node_specs())
        self.patch = patch
        await websocket.send_json({"type": "PATCH_VALIDATION", "errors": []})

    async def _runtime_start(self, websocket: JsonSocket, message: dict[str, Any]) -> None:
        async with self._lock:
            patch_payload = message.get("patch")
            if patch_payload:
                self.patch = validate_patch_graph(patch_payload, node_specs=node_specs())
            if not self.patch:
                raise ValueError("no patch loaded")

            if self.runtime and self.runtime.running:
                await self.runtime.stop()

            self.runtime = ConsoleRuntime(
                self.patch,
                tick_rate=float(message.get("options", {}).get("tick_rate", 30.0)),
                event_cb=websocket.send_json,
            )
            await self.runtime.start(message.get("options", {}))

    async def _runtime_stop(self, websocket: JsonSocket) -> None:
        if self.runtime:
            await self.runtime.stop()
            await websocket.send_json(
                {
                    "type": "RUNTIME_STATUS",
                    "running": False,
                    "tick": self.runtime.tick,
                    "latch_state": {},
                }
            )

    async def _take(self, websocket: JsonSocket, label: str | None) -> None:
        if not self.runtime:
            raise ValueError("runtime not initialized")
        run_dir = self.runtime.recorder.save_take(label)
        if run_dir:
            await websocket.send_json({"type": "TAKE_SAVED", "run_dir": str(run_dir)})

    async def _param_set(self, websocket: JsonSocket, message: dict[str, Any]) -> None:
        if not self.runtime:
            raise ValueError("runtime not initialized")
        self.runtime.update_param(
            node_id=message["node_id"],
            param=message["param"],
            value=message["value"],
        )

    async def _curate_take(self, websocket: JsonSocket, message: dict[str, Any]) -> None:
        run_dir_value = message.get("run_dir")
        if not run_dir_value and self.runtime and self.runtime.recorder.run_dir:
            run_dir_value = str(self.runtime.recorder.run_dir)
        if not run_dir_value:
            raise ValueError("no run_dir available for curation")
        slug = str(message.get("slug") or "console_take")
        curated_dir = curate_take(Path(run_dir_value), slug)
        await websocket.send_json({"type": "CURATE_SAVED", "export_dir": str(curated_dir)})


session = ConsoleSession()


try:  # pragma: no cover - optional runtime dependency
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect

    app = FastAPI(title="Neural Bending Console")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_json()
                await session.handle_message(websocket, message)
        except WebSocketDisconnect:
            if session.runtime and session.runtime.running:
                await session.runtime.stop()
except ImportError:  # pragma: no cover - optional runtime dependency
    app = None
