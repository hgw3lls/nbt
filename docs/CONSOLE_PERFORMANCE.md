# Console Performance Mode (Modular Synth / Mixer)

This guide describes live modular console features for circuit-bending performances and dissertation capture.

## New performance modules

- `MixerNode`
  - 4 channels with per-channel volume, mute, solo, and analysis send.
  - Mixed text/image + analysis metrics output.
- `FeedbackBusNode`
  - Latch-safe feedback with safety gate threshold and token window limit.
- `SovereignSwitchboardNode`
  - Runs prompt across region-tagged model voices and emits CV metrics:
    - `refusal_delta_cv`
    - `framing_delta_cv`
    - `ontology_distance_delta_cv`

## Universal CV modulation

Any numeric parameter can be modulated by wiring a CV signal to a dynamic input port:

- `param:<param_name>`

Parameter resolution:

```text
value = base + (cv * <param_name>_cv_att) + <param_name>_cv_offset
```

The UI inspector exposes these controls for numeric params.

## Patch memory + takes timeline

UI enhancements include:

- named patch memory in localStorage
- import/export JSON patch workflows
- take timeline list with quick recall
- `Curate Take` action to export artifacts to dissertation exports

## Dissertation export

Use the UI `Curate Take` button (sends `CURATE_TAKE`) to export the latest run into:

- `dissertation/exports/<slug>/`

Copied artifacts include memo, outputs, figure specs, patch, metrics, and report (when present).
