# Neural Bending Console UI

The Console UI is a Vite + React + TypeScript app in `console_ui/` that connects to the backend websocket API (`/ws`).

## Run locally

### 1) Start backend server

From repo root:

```bash
nbt console serve --host 127.0.0.1 --port 8000
```

### 2) Start frontend dev server

In a second terminal:

```bash
cd console_ui
npm install
npm run dev
```

By default it connects to:

- `ws://127.0.0.1:8000/ws`

Override with env var:

```bash
VITE_CONSOLE_WS_URL=ws://localhost:9000/ws npm run dev
```

## Features

- Rack canvas built with React Flow (draggable modules + patch cables).
- Node palette includes base + real-model/bend modules:
  - PromptSource
  - CV_LFO
  - DummyTextGen
  - MetricProbe
  - Recorder
  - LLMVoice
  - DiffusionVoice
  - EmbeddingContamination
  - StratigraphySampler
  - GovernanceDissonance
  - JusticeReweighting
  - Compare
- Inspector with parameter editing:
  - numeric sliders/inputs
  - boolean toggles
  - CV attenuator controls (`<param>_cv_att`).
- Live output stream from `TEXT_UPDATE`.
- Image preview pane from `IMAGE_UPDATE` / `IMAGE_PATH` outputs.
- Bottom metrics + scope chart from `METRIC_UPDATE`.
- Header controls:
  - start/stop runtime
  - save patch
  - take capture
  - backend status indicator
- Patch persistence:
  - load from file
  - save to file
  - autosave to localStorage

## Tests

```bash
cd console_ui
npm test
```

Covers:

- patch serialization/parsing
- websocket message parsing


Performance mode details: see `docs/CONSOLE_PERFORMANCE.md`.
