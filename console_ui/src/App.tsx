import { useEffect, useMemo, useRef, useState } from 'react';
import ReactFlow, {
  Background,
  Connection,
  Controls,
  Edge,
  MarkerType,
  Node,
  addEdge,
  useEdgesState,
  useNodesState,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { moduleDefs, portTypeStyles } from './components/moduleDefs';
import { RackNode } from './components/RackNode';
import { parsePatch, serializePatch } from './lib/patchSerde';
import { ConnectionStatus, ConsoleWsClient, ServerMessage } from './lib/wsClient';
import { PatchGraph, PatchNode, starterPatch } from './types/patch';

const nodeTypes = { rackNode: RackNode };
const WS_URL =
  (import.meta.env.VITE_CONSOLE_WS_URL as string | undefined) ??
  'ws://127.0.0.1:8000/ws';

function toRfNodes(patchNodes: PatchNode[]): Node[] {
  return patchNodes.map((node) => {
    const mod = moduleDefs.find((m) => m.type === node.type);
    return {
      id: node.id,
      type: 'rackNode',
      position: node.ui,
      data: {
        label: mod?.label ?? node.type,
        type: node.type,
        inputs: mod?.inputs ?? [],
        outputs: mod?.outputs ?? [],
      },
    };
  });
}

function toRfEdges(patch: PatchGraph): Edge[] {
  return patch.edges.map((edge) => {
    const outType = moduleDefs
      .find((m) => m.type === patch.nodes.find((n) => n.id === edge.from_node)?.type)
      ?.outputs.find((p) => p.key === edge.from_port)?.type;
    return {
      id: edge.id,
      source: edge.from_node,
      sourceHandle: `out:${edge.from_port}`,
      target: edge.to_node,
      targetHandle: `in:${edge.to_port}`,
      animated: true,
      markerEnd: { type: MarkerType.ArrowClosed },
      style: outType ? { stroke: portTypeStyles[outType].color } : undefined,
    };
  });
}

function fromRf(patch: PatchGraph, nodes: Node[], edges: Edge[]): PatchGraph {
  return {
    ...patch,
    nodes: patch.nodes.map((node) => {
      const rf = nodes.find((n) => n.id === node.id);
      return rf ? { ...node, ui: { x: rf.position.x, y: rf.position.y } } : node;
    }),
    edges: edges.map((edge) => ({
      id: edge.id,
      from_node: edge.source,
      from_port: (edge.sourceHandle ?? 'out:text').replace('out:', ''),
      to_node: edge.target,
      to_port: (edge.targetHandle ?? 'in:text').replace('in:', ''),
    })),
  };
}

export default function App() {
  const [patch, setPatch] = useState<PatchGraph>(() => {
    const cached = window.localStorage.getItem('nbt_console_patch');
    return cached ? parsePatch(cached) : starterPatch;
  });
  const [nodes, setNodes, onNodesChange] = useNodesState(toRfNodes(patch.nodes));
  const [edges, setEdges, onEdgesChange] = useEdgesState(toRfEdges(patch));
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');

  const [textOutput, setTextOutput] = useState('');
  const [imageOutput, setImageOutput] = useState('');
  const [takePath, setTakePath] = useState('');
  const [takeTimeline, setTakeTimeline] = useState<string[]>([]);
  const [curatePath, setCuratePath] = useState('');
  const [runtimeRunning, setRuntimeRunning] = useState(false);

  const [patchLibrary, setPatchLibrary] = useState<Record<string, PatchGraph>>(() => {
    try {
      return JSON.parse(
        window.localStorage.getItem('nbt_console_patch_library') ?? '{}'
      ) as Record<string, PatchGraph>;
    } catch {
      return {};
    }
  });
  const [selectedPatchName, setSelectedPatchName] = useState('');

  const [metrics, setMetrics] = useState<Record<string, number>>({
    coherence: 0,
    entropy: 0,
    divergence: 0,
    refusal: 0,
    attractor_density: 0,
  });
  const [scopePoints, setScopePoints] = useState<
    Array<{ entropy: number; coherence: number }>
  >([]);

  const wsRef = useRef<ConsoleWsClient | null>(null);

  useEffect(() => {
    const client = new ConsoleWsClient(WS_URL, {
      onStatus: setStatus,
      onMessage: (msg: ServerMessage) => {
        if (msg.type === 'TEXT_UPDATE') {
          setTextOutput((prev) => `${prev}${msg.text_chunk}`);
        }
        if (msg.type === 'IMAGE_UPDATE') {
          setImageOutput(msg.image_path);
        }
        if (msg.type === 'METRIC_UPDATE') {
          const entropy = Number(msg.metrics.entropy_proxy ?? 0);
          const coherence = Number(msg.metrics.text_length ?? 0) / 100;
          setMetrics({
            coherence,
            entropy,
            divergence: Number(
              msg.metrics.divergence_proxy ?? Math.abs(entropy - coherence)
            ),
            refusal: Number(msg.metrics.refusal_delta ?? msg.metrics.refusal_proxy ?? 0),
            attractor_density: Number(
              msg.metrics.attractor_density_delta ??
                msg.metrics.attractor_density ??
                entropy / 2
            ),
          });
          setScopePoints((prev) => [...prev.slice(-59), { entropy, coherence }]);
        }
        if (msg.type === 'TAKE_SAVED') {
          setTakePath(msg.run_dir);
          setTakeTimeline((prev) => [msg.run_dir, ...prev.filter((v) => v !== msg.run_dir)].slice(0, 20));
        }
        if (msg.type === 'CURATE_SAVED') {
          setCuratePath(msg.export_dir);
        }
        if (msg.type === 'RUNTIME_STATUS') {
          setRuntimeRunning(msg.running);
        }
      },
    });
    wsRef.current = client;
    client.connect();
    return () => client.disconnect();
  }, []);

  useEffect(() => {
    setNodes(toRfNodes(patch.nodes));
    setEdges(toRfEdges(patch));
  }, [patch, setEdges, setNodes]);

  const selectedNode = useMemo(
    () => patch.nodes.find((n) => n.id === selectedNodeId),
    [patch.nodes, selectedNodeId]
  );

  function safeSend(message: Parameters<ConsoleWsClient['send']>[0]): void {
    try {
      wsRef.current?.send(message);
    } catch {
      // ignore transient disconnects
    }
  }

  function updatePatch(next: PatchGraph, sendUpdate = true): void {
    setPatch(next);
    window.localStorage.setItem('nbt_console_patch', serializePatch(next));
    if (sendUpdate) {
      safeSend({ type: 'PATCH_UPDATE', full_patch: next });
    }
  }

  function onConnect(connection: Connection): void {
    const nextEdges = addEdge({ ...connection, id: `edge_${crypto.randomUUID()}` }, edges);
    setEdges(nextEdges);
    updatePatch(fromRf(patch, nodes, nextEdges));
  }

  function addModule(moduleType: string): void {
    const mod = moduleDefs.find((m) => m.type === moduleType);
    if (!mod) return;
    const id = `${moduleType.replace('Node', '').toLowerCase()}_${Date.now()}`;
    const newNode: PatchNode = {
      id,
      type: moduleType,
      params: { ...mod.defaultParams },
      enabled: true,
      ui: { x: 180 + Math.random() * 240, y: 100 + Math.random() * 200 },
    };
    updatePatch({ ...patch, nodes: [...patch.nodes, newNode] });
  }

  function setNodeParam(key: string, value: unknown): void {
    if (!selectedNode) return;
    const next = {
      ...patch,
      nodes: patch.nodes.map((n) =>
        n.id === selectedNode.id ? { ...n, params: { ...n.params, [key]: value } } : n
      ),
    };
    updatePatch(next, false);
    safeSend({ type: 'PARAM_SET', node_id: selectedNode.id, param: key, value });
  }

  function toggleNodeEnabled(enabled: boolean): void {
    if (!selectedNode) return;
    updatePatch({
      ...patch,
      nodes: patch.nodes.map((n) => (n.id === selectedNode.id ? { ...n, enabled } : n)),
    });
  }

  function startRuntime(): void {
    safeSend({ type: 'PATCH_LOAD', patch });
    safeSend({ type: 'RUNTIME_START', patch, options: { tick_rate: 30 } });
  }

  function stopRuntime(): void {
    safeSend({ type: 'RUNTIME_STOP' });
  }

  function take(): void {
    safeSend({ type: 'TAKE' });
  }

  function curateTake(): void {
    const slug = window.prompt('Export slug', 'live_take');
    if (!slug) return;
    safeSend({ type: 'CURATE_TAKE', run_dir: takeTimeline[0] || takePath, slug });
  }

  function savePatchToDisk(): void {
    const blob = new Blob([serializePatch(fromRf(patch, nodes, edges))], {
      type: 'application/json',
    });
    const href = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = href;
    a.download = 'patch.json';
    a.click();
    URL.revokeObjectURL(href);
  }

  async function loadPatchFile(file: File): Promise<void> {
    const loadedPatch = parsePatch(await file.text());
    updatePatch(loadedPatch);
    safeSend({ type: 'PATCH_LOAD', patch: loadedPatch });
  }

  function savePatchNamed(): void {
    const name = window.prompt('Patch name');
    if (!name) return;
    const next = { ...patchLibrary, [name]: fromRf(patch, nodes, edges) };
    setPatchLibrary(next);
    setSelectedPatchName(name);
    window.localStorage.setItem('nbt_console_patch_library', JSON.stringify(next));
  }

  function loadNamedPatch(name: string): void {
    setSelectedPatchName(name);
    const found = patchLibrary[name];
    if (!found) return;
    updatePatch(found);
    safeSend({ type: 'PATCH_LOAD', patch: found });
  }

  return (
    <div className="app-shell">
      <header className="header">
        <h1>Neural Bending Console</h1>
        <div className="status-dot">Backend: {status}</div>
        <button onClick={runtimeRunning ? stopRuntime : startRuntime}>
          {runtimeRunning ? 'Stop' : 'Start'}
        </button>
        <button onClick={savePatchToDisk}>Save Patch</button>
        <button onClick={savePatchNamed}>Save to Memory</button>
        <select value={selectedPatchName} onChange={(e) => loadNamedPatch(e.target.value)}>
          <option value="">Patch Memory</option>
          {Object.keys(patchLibrary).map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
        <button onClick={take}>Take</button>
        <button onClick={curateTake}>Curate Take</button>
        <label className="file-label">
          Load Patch
          <input
            type="file"
            accept="application/json"
            onChange={(e) => e.target.files?.[0] && loadPatchFile(e.target.files[0])}
          />
        </label>
      </header>

      <div className="main-grid">
        <aside className="left-panel">
          <h3>Output Stage</h3>
          <pre>{textOutput || 'Waiting for TEXT_UPDATE...'}</pre>
          <h4>Image Preview</h4>
          {imageOutput ? (
            <img className="image-preview" src={imageOutput} alt="latest output" />
          ) : (
            <p>No IMAGE_PATH output yet.</p>
          )}
          <p className="take-path">{takePath && `Take saved: ${takePath}`}</p>
          <p className="take-path">{curatePath && `Curated export: ${curatePath}`}</p>
          <h4>Take Timeline</h4>
          <ul className="timeline">
            {takeTimeline.map((item) => (
              <li key={item}>
                <button onClick={() => setTakePath(item)}>{item}</button>
              </li>
            ))}
          </ul>
        </aside>

        <section className="canvas-panel">
          <div className="palette">
            <h3>Add Module</h3>
            {moduleDefs.map((mod) => (
              <button key={mod.type} onClick={() => addModule(mod.type)}>
                {mod.label}
              </button>
            ))}
          </div>

          <div className="flow-wrap">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={(_, node) => setSelectedNodeId(node.id)}
              onNodeDragStop={(_, draggedNode) => {
                const nextNodes = nodes.map((n) =>
                  n.id === draggedNode.id ? { ...n, position: draggedNode.position } : n
                );
                updatePatch(fromRf(patch, nextNodes, edges));
              }}
              fitView
            >
              <Background />
              <Controls />
            </ReactFlow>
          </div>
        </section>

        <aside className="right-panel">
          <h3>Inspector</h3>
          {selectedNode ? (
            <>
              <h4>{selectedNode.id}</h4>
              <label>
                Enabled
                <input
                  type="checkbox"
                  checked={selectedNode.enabled}
                  onChange={(e) => toggleNodeEnabled(e.target.checked)}
                />
              </label>
              {Object.entries(selectedNode.params).map(([key, value]) => {
                if (typeof value === 'boolean') {
                  return (
                    <label key={key}>
                      {key}
                      <input
                        type="checkbox"
                        checked={value}
                        onChange={(e) => setNodeParam(key, e.target.checked)}
                      />
                    </label>
                  );
                }

                if (typeof value === 'number') {
                  return (
                    <div key={key} className="inspector-field">
                      <label>{key}</label>
                      <input
                        type="range"
                        min={-2}
                        max={2}
                        step={0.01}
                        value={value}
                        onChange={(e) => setNodeParam(key, Number(e.target.value))}
                      />
                      <input
                        type="number"
                        value={value}
                        onChange={(e) => setNodeParam(key, Number(e.target.value))}
                      />
                      <div className="cv-jack">
                        CV jack: <code>param:{key}</code>
                      </div>
                      <label>
                        CV attenuator
                        <input
                          type="range"
                          min={-1}
                          max={1}
                          step={0.01}
                          value={Number((selectedNode.params[`${key}_cv_att`] as number) ?? 1)}
                          onChange={(e) => setNodeParam(`${key}_cv_att`, Number(e.target.value))}
                        />
                      </label>
                      <label>
                        CV offset
                        <input
                          type="number"
                          value={Number((selectedNode.params[`${key}_cv_offset`] as number) ?? 0)}
                          onChange={(e) => setNodeParam(`${key}_cv_offset`, Number(e.target.value))}
                        />
                      </label>
                    </div>
                  );
                }

                return (
                  <div key={key} className="inspector-field">
                    <label>{key}</label>
                    <input value={String(value)} onChange={(e) => setNodeParam(key, e.target.value)} />
                  </div>
                );
              })}
            </>
          ) : (
            <p>Select a module to edit params.</p>
          )}
        </aside>
      </div>

      <footer className="meters-strip">
        <div className="meter-grid">
          {Object.entries(metrics).map(([k, v]) => (
            <div className="meter" key={k}>
              <span>{k.replace('_', ' ')}</span>
              <progress value={Math.min(1, Math.max(0, Number(v)))} max={1} />
              <strong>{Number(v).toFixed(3)}</strong>
            </div>
          ))}
        </div>
        <div className="scope">
          <h4>Scope (Entropy & Coherence)</h4>
          <svg viewBox="0 0 600 120" preserveAspectRatio="none">
            <polyline
              fill="none"
              stroke="#60a5fa"
              strokeWidth="2"
              points={scopePoints.map((p, i) => `${i * 10},${120 - p.coherence * 100}`).join(' ')}
            />
            <polyline
              fill="none"
              stroke="#34d399"
              strokeWidth="2"
              points={scopePoints.map((p, i) => `${i * 10},${120 - p.entropy * 100}`).join(' ')}
            />
          </svg>
        </div>
      </footer>
    </div>
  );
}
