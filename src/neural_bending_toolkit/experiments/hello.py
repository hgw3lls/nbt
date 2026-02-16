"""Built-in hello world style experiment."""

from __future__ import annotations

from pydantic import Field

from neural_bending_toolkit.experiment import Experiment, ExperimentSettings, RunContext


class HelloExperimentConfig(ExperimentSettings):
    """Configuration for the hello experiment."""

    name: str = "world"
    repeats: int = Field(default=1, ge=1, le=100)


class HelloExperiment(Experiment):
    """A tiny experiment that emits greetings and metrics."""

    name = "hello-experiment"
    config_model = HelloExperimentConfig

    def run(self, context: RunContext) -> None:
        for idx in range(self.config.repeats):
            step = idx + 1
            message = f"Hello experiment, {self.config.name}!"
            context.pre_intervention_snapshot(
                name="greeting_state",
                data={"step": step, "name": self.config.name, "phase": "before"},
            )
            context.log_event(message)
            context.log_metric(
                step=step,
                metric_name="greetings_emitted",
                value=step,
                metadata={"name": self.config.name},
            )
            context.post_intervention_snapshot(
                name="greeting_state",
                data={"step": step, "name": self.config.name, "phase": "after"},
            )

        context.save_text_artifact(
            "greeting.txt",
            f"Hello experiment, {self.config.name}!\n",
        )
