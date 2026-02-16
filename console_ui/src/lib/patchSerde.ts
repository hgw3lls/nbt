import { PatchGraph } from '../types/patch';

export function serializePatch(patch: PatchGraph): string {
  return JSON.stringify(patch, null, 2);
}

export function parsePatch(raw: string): PatchGraph {
  const parsed = JSON.parse(raw) as PatchGraph;
  if (!Array.isArray(parsed.nodes) || !Array.isArray(parsed.edges)) {
    throw new Error('Invalid patch format: expected nodes and edges arrays');
  }
  return parsed;
}
