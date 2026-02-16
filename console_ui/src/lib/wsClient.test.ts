import { describe, expect, it } from 'vitest';

import { parseServerMessage } from './wsClient';

describe('ws message parsing', () => {
  it('parses known message', () => {
    const msg = parseServerMessage(
      JSON.stringify({ type: 'METRIC_UPDATE', ts: 't', metrics: { entropy_proxy: 0.3 } })
    );
    expect(msg.type).toBe('METRIC_UPDATE');
  });


  it('parses image update message', () => {
    const msg = parseServerMessage(
      JSON.stringify({ type: 'IMAGE_UPDATE', ts: 't', image_path: '/tmp/a.png' })
    );
    expect(msg.type).toBe('IMAGE_UPDATE');
  });


  it('parses curate saved message', () => {
    const msg = parseServerMessage(
      JSON.stringify({ type: 'CURATE_SAVED', export_dir: 'dissertation/exports/x' })
    );
    expect(msg.type).toBe('CURATE_SAVED');
  });

  it('throws when type missing', () => {
    expect(() => parseServerMessage(JSON.stringify({ hello: 'world' }))).toThrow(/missing type/);
  });
});
