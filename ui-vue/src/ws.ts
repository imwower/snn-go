import type { Pinia } from 'pinia';
import { useUiStore } from './store/ui';
import type { LogPayload, MetricPayload, SocketEnvelope, SpikePayload } from './types';

let socket: WebSocket | null = null;
let reconnectHandle: number | null = null;
let started = false;
let storeInstance: ReturnType<typeof useUiStore> | null = null;
const RECONNECT_DELAY = 1000;

const wsUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws`;
};

const scheduleReconnect = () => {
  if (!started) {
    return;
  }
  if (reconnectHandle !== null) {
    window.clearTimeout(reconnectHandle);
  }
  reconnectHandle = window.setTimeout(() => {
    reconnectHandle = null;
    connect();
  }, RECONNECT_DELAY);
};

const handleMetrics = (payload: MetricPayload) => {
  storeInstance?.pushMetric(payload);
  if (storeInstance && storeInstance.status === 'Idle') {
    storeInstance.setStatus('Training');
  }
};

const handleSpikes = (payload: SpikePayload) => {
  storeInstance?.pushSpike(payload);
  if (storeInstance && storeInstance.status === 'Idle') {
    storeInstance.setStatus('Training');
  }
};

const handleLog = (payload: LogPayload) => {
  storeInstance?.pushLog(payload);
  storeInstance?.pushMessage('log', payload, 'log');
};

const handleEnvelope = (raw: unknown) => {
  if (!raw || typeof raw !== 'object') {
    console.warn('Unexpected websocket payload', raw);
    return;
  }

  const message = raw as Partial<SocketEnvelope> & { type?: string; data?: unknown };

  if (message.type === 'metrics') {
    handleMetrics(message.data as MetricPayload);
    storeInstance?.pushMessage('metrics', message.data, 'metrics');
    return;
  }

  if (message.type === 'spikes') {
    handleSpikes(message.data as SpikePayload);
    storeInstance?.pushMessage('spikes', message.data, 'spikes');
    return;
  }

  if (message.type === 'log') {
    handleLog(message.data as LogPayload);
    return;
  }

  console.warn('Unknown websocket message type', message);
};

const connect = () => {
  if (!storeInstance) {
    return;
  }
  try {
    socket = new WebSocket(wsUrl());
  } catch (err) {
    console.warn('Failed to open websocket', err);
    scheduleReconnect();
    return;
  }

  socket.onmessage = (event: MessageEvent<string>) => {
    try {
      const parsed: unknown = JSON.parse(event.data);
      handleEnvelope(parsed);
    } catch (err) {
      console.warn('Failed to parse websocket payload', err);
    }
  };

  socket.onclose = () => {
    scheduleReconnect();
  };

  socket.onerror = (err) => {
    console.warn('WebSocket error', err);
    socket?.close();
  };
};

export const startSocket = (pinia: Pinia) => {
  if (started) {
    return;
  }
  storeInstance = useUiStore(pinia);
  started = true;
  connect();
};

export const getSocket = () => socket;
