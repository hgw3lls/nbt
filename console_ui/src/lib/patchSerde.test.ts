import { describe, expect, it } from 'vitest';

import { parsePatch, serializePatch } from './patchSerde';
import { starterPatch } from '../types/patch';

describe('patchSerde', () => {
  it('roundtrips patch serialization', () => {
    const encoded = serializePatch(starterPatch);
    const decoded = parsePatch(encoded);
    expect(decoded.nodes.length).toBe(starterPatch.nodes.length);
    expect(decoded.edges.length).toBe(starterPatch.edges.length);
  });

  it('throws on invalid patch shape', () => {
    expect(() => parsePatch('{"foo":1}')).toThrow(/Invalid patch format/);
  });
});
