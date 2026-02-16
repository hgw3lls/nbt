import { PatchGraph } from '../types/patch';

export type ServerMessage =
  | { type: 'ACK'; message_type: string }
  | { type: 'ERROR'; message: string }
  | { type: 'PATCH_VALIDATION'; errors?: string[] }
  | {
      type: 'RUNTIME_STATUS';
      running: boolean;
      tick: number;
      latch_state: Record<string, boolean>;
    }
  | { type: 'METRIC_UPDATE'; ts: string; metrics: Record<string, number> }
  | { type: 'TEXT_UPDATE'; ts: string; text_chunk: string; channel?: string }
  | { type: 'IMAGE_UPDATE'; ts: string; image_path: string; channel?: string }
  | { type: 'TAKE_SAVED'; run_dir: string }
  | { type: 'CURATE_SAVED'; export_dir: string };

export type ClientMessage =
  | { type: 'PATCH_LOAD'; patch: PatchGraph }
  | { type: 'PATCH_UPDATE'; full_patch: PatchGraph }
  | { type: 'RUNTIME_START'; patch?: PatchGraph; options?: Record<string, unknown> }
  | { type: 'RUNTIME_STOP' }
  | { type: 'TAKE'; label?: string }
  | { type: 'PARAM_SET'; node_id: string; param: string; value: unknown }
  | { type: 'CURATE_TAKE'; run_dir?: string; slug?: string };

export interface WsCallbacks {
  onOpen?: () => void;
  onClose?: () => void;
  onMessage?: (message: ServerMessage) => void;
  onError?: (error: string) => void;
  onStatus?: (status: ConnectionStatus) => void;
}

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

export function parseServerMessage(raw: string): ServerMessage {
  const message = JSON.parse(raw) as Partial<ServerMessage>;
  if (!message.type) {
    throw new Error('WS message missing type');
  }
  return message as ServerMessage;
}

export class ConsoleWsClient {
  private ws?: WebSocket;
  private readonly url: string;
  private readonly callbacks: WsCallbacks;
  private reconnectTimer?: number;
  private reconnectAttempts = 0;

  constructor(url: string, callbacks: WsCallbacks = {}) {
    this.url = url;
    this.callbacks = callbacks;
  }

  connect(): void {
    this.updateStatus(this.reconnectAttempts > 0 ? 'reconnecting' : 'connecting');
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.updateStatus('connected');
      this.callbacks.onOpen?.();
    };

    this.ws.onmessage = (event) => {
      try {
        const parsed = parseServerMessage(String(event.data));
        this.callbacks.onMessage?.(parsed);
      } catch (error) {
        this.callbacks.onError?.(String(error));
      }
    };

    this.ws.onerror = () => {
      this.callbacks.onError?.('WebSocket connection error');
    };

    this.ws.onclose = () => {
      this.updateStatus('disconnected');
      this.callbacks.onClose?.();
      this.scheduleReconnect();
    };
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = undefined;
    }
    this.ws?.close();
    this.ws = undefined;
    this.updateStatus('disconnected');
  }

  send(message: ClientMessage): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not connected');
    }
    this.ws.send(JSON.stringify(message));
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      window.clearTimeout(this.reconnectTimer);
    }
    this.reconnectAttempts += 1;
    const delayMs = Math.min(1000 * this.reconnectAttempts, 5000);
    this.reconnectTimer = window.setTimeout(() => this.connect(), delayMs);
  }

  private updateStatus(status: ConnectionStatus): void {
    this.callbacks.onStatus?.(status);
  }
}
