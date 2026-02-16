export type PortType =
  | 'TEXT'
  | 'CV'
  | 'METRIC'
  | 'EMBEDDING'
  | 'LATENTS'
  | 'ATTENTION'
  | 'RESIDUAL'
  | 'TRIGGER'
  | 'IMAGE_PATH'
  | 'AUDIO_PATH';

export interface PatchNode {
  id: string;
  type: string;
  params: Record<string, unknown>;
  ui: { x: number; y: number };
  enabled: boolean;
}

export interface PatchEdge {
  id: string;
  from_node: string;
  from_port: string;
  to_node: string;
  to_port: string;
}

export interface PatchGraph {
  nodes: PatchNode[];
  edges: PatchEdge[];
  globals: Record<string, unknown>;
}

export const starterPatch: PatchGraph = {
  nodes: [
    {
      id: 'prompt_1',
      type: 'PromptSourceNode',
      params: { text: 'neural bending console' },
      ui: { x: 50, y: 100 },
      enabled: true,
    },
    {
      id: 'gen_1',
      type: 'DummyTextGenNode',
      params: { temperature: 0.2 },
      ui: { x: 350, y: 100 },
      enabled: true,
    },
    {
      id: 'metric_1',
      type: 'MetricProbeNode',
      params: {},
      ui: { x: 650, y: 100 },
      enabled: true,
    },
  ],
  edges: [
    {
      id: 'edge_prompt_gen',
      from_node: 'prompt_1',
      from_port: 'text',
      to_node: 'gen_1',
      to_port: 'prompt',
    },
    {
      id: 'edge_gen_metric',
      from_node: 'gen_1',
      from_port: 'text',
      to_node: 'metric_1',
      to_port: 'text',
    },
  ],
  globals: { tick_rate: 30 },
};
