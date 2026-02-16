import asyncio
import time

from neural_bending_toolkit.console.server import ConsoleSession


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.messages.append(payload)


def test_ws_protocol_smoke_runtime_events() -> None:
    patch = {
        "nodes": [
            {
                "id": "prompt",
                "type": "PromptSourceNode",
                "params": {"text": "smoke test"},
                "ui": {"x": 0, "y": 0},
                "enabled": True,
            },
            {
                "id": "gen",
                "type": "DummyTextGenNode",
                "params": {},
                "ui": {"x": 1, "y": 0},
                "enabled": True,
            },
            {
                "id": "metric",
                "type": "MetricProbeNode",
                "params": {},
                "ui": {"x": 2, "y": 0},
                "enabled": True,
            },
        ],
        "edges": [
            {
                "id": "e1",
                "from_node": "prompt",
                "from_port": "text",
                "to_node": "gen",
                "to_port": "prompt",
            },
            {
                "id": "e2",
                "from_node": "gen",
                "from_port": "text",
                "to_node": "metric",
                "to_port": "text",
            },
        ],
        "globals": {},
    }

    async def scenario() -> list[dict]:
        session = ConsoleSession()
        ws = FakeWebSocket()
        await session.handle_message(ws, {"type": "PATCH_LOAD", "patch": patch})
        await session.handle_message(
            ws, {"type": "RUNTIME_START", "options": {"tick_rate": 120}}
        )

        deadline = time.time() + 0.5
        while time.time() < deadline:
            if any(msg.get("type") == "TEXT_UPDATE" for msg in ws.messages) and any(
                msg.get("type") == "METRIC_UPDATE" for msg in ws.messages
            ):
                break
            await asyncio.sleep(0.01)

        await session.handle_message(ws, {"type": "RUNTIME_STOP"})
        return ws.messages

    messages = asyncio.run(scenario())
    assert any(msg.get("type") == "PATCH_VALIDATION" for msg in messages)
    assert any(msg.get("type") == "TEXT_UPDATE" for msg in messages)
    assert any(msg.get("type") == "METRIC_UPDATE" for msg in messages)
