from neural_bending_toolkit.console.recorder import ConsoleRecorder
from neural_bending_toolkit.console.runtime import (
    MixerNode,
    RuntimeContext,
    SovereignSwitchboardNode,
)


def _ctx() -> RuntimeContext:
    return RuntimeContext(
        tick=1,
        tick_rate=30.0,
        latch_state={"token_boundary_latch": True, "diffusion_step_latch": True},
        recorder=ConsoleRecorder(),
    )


def test_mixer_routing_with_solo_and_mute() -> None:
    node = MixerNode(
        "mix",
        {
            "ch1_volume": 1.0,
            "ch2_volume": 0.5,
            "ch2_mute": True,
            "ch3_solo": True,
            "ch3_volume": 0.8,
        },
    )
    out = node.process(
        _ctx(),
        {
            "ch1_text": "alpha",
            "ch2_text": "beta",
            "ch3_text": "gamma",
            "ch3_image": "img.png",
        },
    )
    assert "gamma" in out["text"]
    assert "alpha" not in out["text"]
    assert out["image_path"] == "img.png"


def test_cv_modulation_on_numeric_param() -> None:
    node = MixerNode("mix", {"ch1_volume": 1.0, "ch1_volume_cv_att": 0.5, "ch1_volume_cv_offset": 0.2})
    out = node.process(_ctx(), {"ch1_text": "hello", "param:ch1_volume": 0.4})
    assert "ch1" in out["metric"]
    assert abs(float(out["metric"]["ch1"]) - 1.4) < 1e-6


def test_switchboard_metrics_with_stub_adapters() -> None:
    node = SovereignSwitchboardNode("switch", {})
    out = node.process(_ctx(), {"prompt": "policy futures"})
    assert "refusal_delta_cv" in out
    assert "framing_delta_cv" in out
    assert "ontology_distance_delta_cv" in out
    assert "metric" in out
