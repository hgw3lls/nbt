import { PortType } from '../types/patch';

export interface ModuleDef {
  label: string;
  type: string;
  inputs: Array<{ key: string; type: PortType }>;
  outputs: Array<{ key: string; type: PortType }>;
  defaultParams: Record<string, unknown>;
}

export const moduleDefs: ModuleDef[] = [
  { label: 'PromptSource', type: 'PromptSourceNode', inputs: [], outputs: [{ key: 'text', type: 'TEXT' }], defaultParams: { text: 'enter prompt' } },
  { label: 'CV_LFO', type: 'CV_LFONode', inputs: [], outputs: [{ key: 'cv', type: 'CV' }], defaultParams: { waveform: 'sine', frequency_hz: 1.0 } },
  { label: 'DummyTextGen', type: 'DummyTextGenNode', inputs: [{ key: 'prompt', type: 'TEXT' }], outputs: [{ key: 'text', type: 'TEXT' }], defaultParams: { temperature: 0.2 } },
  { label: 'MetricProbe', type: 'MetricProbeNode', inputs: [{ key: 'text', type: 'TEXT' }], outputs: [{ key: 'metric', type: 'METRIC' }], defaultParams: {} },
  { label: 'Recorder', type: 'RecorderNode', inputs: [{ key: 'trigger', type: 'TRIGGER' }], outputs: [], defaultParams: {} },
  {
    label: 'LLMVoice',
    type: 'LLMVoiceNode',
    inputs: [
      { key: 'prompt', type: 'TEXT' },
      { key: 'temp_cv', type: 'CV' },
      { key: 'top_p_cv', type: 'CV' },
      { key: 'top_k_cv', type: 'CV' },
    ],
    outputs: [
      { key: 'text', type: 'TEXT' },
      { key: 'metric', type: 'METRIC' },
    ],
    defaultParams: { temperature: 0.8, top_p: 0.9, top_k: 40, max_new_tokens: 24 },
  },
  {
    label: 'DiffusionVoice',
    type: 'DiffusionVoiceNode',
    inputs: [
      { key: 'prompt', type: 'TEXT' },
      { key: 'guidance_cv', type: 'CV' },
      { key: 'embedding', type: 'EMBEDDING' },
    ],
    outputs: [
      { key: 'image_path', type: 'IMAGE_PATH' },
      { key: 'metric', type: 'METRIC' },
    ],
    defaultParams: { guidance_scale: 7.5, num_steps: 20 },
  },
  {
    label: 'EmbeddingContamination',
    type: 'EmbeddingContaminationNode',
    inputs: [
      { key: 'embedding_a', type: 'EMBEDDING' },
      { key: 'embedding_b', type: 'EMBEDDING' },
    ],
    outputs: [{ key: 'embedding', type: 'EMBEDDING' }],
    defaultParams: { enabled: true, dry_wet: 0.5, concept_a: 'law', concept_b: 'care' },
  },
  {
    label: 'StratigraphySampler',
    type: 'StratigraphySamplerNode',
    inputs: [{ key: 'cv', type: 'CV' }],
    outputs: [{ key: 'cv', type: 'CV' }],
    defaultParams: { enabled: true, dry_wet: 1.0 },
  },
  {
    label: 'GovernanceDissonance',
    type: 'GovernanceDissonanceNode',
    inputs: [
      { key: 'text', type: 'TEXT' },
      { key: 'embedding_a', type: 'EMBEDDING' },
      { key: 'embedding_b', type: 'EMBEDDING' },
      { key: 'cv', type: 'CV' },
    ],
    outputs: [
      { key: 'text', type: 'TEXT' },
      { key: 'embedding', type: 'EMBEDDING' },
      { key: 'cv', type: 'CV' },
    ],
    defaultParams: { enabled: true, dry_wet: 0.5, inject: 'yet the opposite also holds' },
  },
  {
    label: 'JusticeReweighting',
    type: 'JusticeReweightingNode',
    inputs: [
      { key: 'text', type: 'TEXT' },
      { key: 'embedding', type: 'EMBEDDING' },
    ],
    outputs: [
      { key: 'text', type: 'TEXT' },
      { key: 'embedding', type: 'EMBEDDING' },
      { key: 'metric', type: 'METRIC' },
    ],
    defaultParams: { enabled: true, dry_wet: 0.6, attractor_lexicon: ['care', 'justice', 'repair'] },
  },

  {
    label: 'Mixer',
    type: 'MixerNode',
    inputs: [
      { key: 'ch1_text', type: 'TEXT' },
      { key: 'ch2_text', type: 'TEXT' },
      { key: 'ch3_text', type: 'TEXT' },
      { key: 'ch4_text', type: 'TEXT' },
      { key: 'ch1_image', type: 'IMAGE_PATH' },
      { key: 'ch2_image', type: 'IMAGE_PATH' },
      { key: 'ch3_image', type: 'IMAGE_PATH' },
      { key: 'ch4_image', type: 'IMAGE_PATH' },
      { key: 'analysis_bus', type: 'METRIC' },
    ],
    outputs: [
      { key: 'text', type: 'TEXT' },
      { key: 'image_path', type: 'IMAGE_PATH' },
      { key: 'metric', type: 'METRIC' },
    ],
    defaultParams: {
      ch1_volume: 1, ch2_volume: 1, ch3_volume: 1, ch4_volume: 1,
      ch1_mute: false, ch2_mute: false, ch3_mute: false, ch4_mute: false,
      ch1_solo: false, ch2_solo: false, ch3_solo: false, ch4_solo: false,
      ch1_send_analysis: true, ch2_send_analysis: true, ch3_send_analysis: true, ch4_send_analysis: true
    },
  },
  {
    label: 'FeedbackBus',
    type: 'FeedbackBusNode',
    inputs: [{ key: 'text', type: 'TEXT' }, { key: 'cv', type: 'CV' }],
    outputs: [{ key: 'text', type: 'TEXT' }, { key: 'metric', type: 'METRIC' }],
    defaultParams: { gate_threshold: 0.8, max_feedback_tokens: 12 },
  },
  {
    label: 'SovereignSwitchboard',
    type: 'SovereignSwitchboardNode',
    inputs: [{ key: 'prompt', type: 'TEXT' }],
    outputs: [
      { key: 'refusal_delta_cv', type: 'CV' },
      { key: 'framing_delta_cv', type: 'CV' },
      { key: 'ontology_distance_delta_cv', type: 'CV' },
      { key: 'metric', type: 'METRIC' },
    ],
    defaultParams: {
      voices: [
        { name: 'us_voice', region: 'NA' },
        { name: 'eu_voice', region: 'EU' },
        { name: 'cn_voice', region: 'APAC' },
      ],
    },
  },
  {
    label: 'Compare',
    type: 'CompareNode',
    inputs: [
      { key: 'baseline_text', type: 'TEXT' },
      { key: 'bent_text', type: 'TEXT' },
      { key: 'baseline_image', type: 'IMAGE_PATH' },
      { key: 'bent_image', type: 'IMAGE_PATH' },
    ],
    outputs: [{ key: 'metric', type: 'METRIC' }],
    defaultParams: { attractor_lexicon: ['care', 'justice', 'repair'] },
  },
];

export const portTypeStyles: Record<PortType, { color: string; icon: string }> = {
  TEXT: { color: '#60a5fa', icon: '📝' },
  CV: { color: '#f59e0b', icon: '〰️' },
  METRIC: { color: '#34d399', icon: '📊' },
  TRIGGER: { color: '#f43f5e', icon: '⚡' },
  EMBEDDING: { color: '#a78bfa', icon: '🧠' },
  LATENTS: { color: '#2dd4bf', icon: '🌫️' },
  ATTENTION: { color: '#fb7185', icon: '👁️' },
  RESIDUAL: { color: '#22c55e', icon: '♻️' },
  IMAGE_PATH: { color: '#f97316', icon: '🖼️' },
  AUDIO_PATH: { color: '#eab308', icon: '🔊' },
};
