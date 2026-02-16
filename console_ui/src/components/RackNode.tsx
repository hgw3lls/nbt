import { Handle, NodeProps, Position } from 'reactflow';

import { portTypeStyles } from './moduleDefs';
import { PortType } from '../types/patch';

interface RackNodeData {
  label: string;
  type: string;
  inputs: Array<{ key: string; type: PortType }>;
  outputs: Array<{ key: string; type: PortType }>;
}

function PortBadge({ label, type }: { label: string; type: PortType }) {
  const style = portTypeStyles[type];
  return (
    <span className="port-badge" style={{ borderColor: style.color }}>
      <span>{style.icon}</span>
      <span>{label}</span>
    </span>
  );
}

export function RackNode({ data }: NodeProps<RackNodeData>) {
  return (
    <div className="rack-node">
      <div className="rack-node-title">{data.label}</div>
      <div className="rack-node-type">{data.type}</div>

      {data.inputs.map((port, idx) => {
        const style = portTypeStyles[port.type];
        return (
          <div className="port-row" key={`in-${port.key}`}>
            <Handle
              type="target"
              id={`in:${port.key}`}
              position={Position.Left}
              style={{ top: 56 + idx * 24, background: style.color }}
            />
            <PortBadge label={port.key} type={port.type} />
          </div>
        );
      })}

      {data.outputs.map((port, idx) => {
        const style = portTypeStyles[port.type];
        return (
          <div className="port-row out" key={`out-${port.key}`}>
            <PortBadge label={port.key} type={port.type} />
            <Handle
              type="source"
              id={`out:${port.key}`}
              position={Position.Right}
              style={{ top: 56 + idx * 24, background: style.color }}
            />
          </div>
        );
      })}
    </div>
  );
}
