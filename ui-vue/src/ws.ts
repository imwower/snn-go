import type { Pinia } from 'pinia';
import { useUiStore } from './store/ui';
import type { LogPayload, MetricPayload, SpikePayload } from './types';

let source: EventSource | null = null;
let reconnectHandle: number | null = null;
let started = false;
let storeInstance: ReturnType<typeof useUiStore> | null = null;

const RECONNECT_DELAY = 1500;

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

const ensureNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  return fallback;
};

const parseJSON = <T>(raw: string): T | null => {
  try {
    return JSON.parse(raw) as T;
  } catch (err) {
    console.warn('Failed to parse payload', err);
    return null;
  }
};

const normalizeMetric = (payload: Partial<MetricPayload> & { time_unix?: number; residual?: number; K?: number }): MetricPayload => {
  const epoch = ensureNumber(payload.epoch, 0);
  const step = ensureNumber(payload.step, ensureNumber(payload.k, 0));
  const loss = ensureNumber(payload.loss, 0);
  const acc = typeof payload.acc === 'number' && Number.isFinite(payload.acc) ? payload.acc : undefined;
  const residual = typeof payload.residual === 'number' && Number.isFinite(payload.residual) ? payload.residual : undefined;
  const throughput = typeof payload.throughput === 'number' && Number.isFinite(payload.throughput) ? payload.throughput : undefined;
  const lr = typeof payload.lr === 'number' && Number.isFinite(payload.lr) ? payload.lr : undefined;
  const k = typeof payload.k === 'number' && Number.isFinite(payload.k) ? payload.k : undefined;

  return {
    epoch,
    step,
    loss,
    acc,
    residual,
    throughput,
    lr,
    k
  };
};

const handleMetrics = (payload: MetricPayload, subject: string) => {
  storeInstance?.pushMetric(payload);
  storeInstance?.pushMessage(subject, payload, 'metrics');
  if (storeInstance && storeInstance.status === 'Idle') {
    storeInstance.setStatus('Training');
  }
};

const handleSpikes = (payload: SpikePayload) => {
  storeInstance?.pushSpike(payload);
  storeInstance?.pushMessage('spikes', payload, 'spikes');
  if (storeInstance && storeInstance.status === 'Idle') {
    storeInstance.setStatus('Training');
  }
};

const handleLog = (payload: { level?: string; msg?: string; message?: string; time_unix?: number; ts?: number }) => {
  const normalized: LogPayload = {
    level: (typeof payload.level === 'string' ? payload.level.toUpperCase() : 'INFO') as LogPayload['level'],
    message: typeof payload.msg === 'string' ? payload.msg : typeof payload.message === 'string' ? payload.message : '',
    ts:
      typeof payload.ts === 'number'
        ? payload.ts
        : typeof payload.time_unix === 'number'
          ? payload.time_unix
          : Date.now() / 1000
  };
  storeInstance?.pushLog(normalized);
  storeInstance?.pushMessage('log', normalized, 'log');
};

const bindEventHandlers = (es: EventSource) => {
  es.addEventListener('metrics_batch', (event) => {
    const payload = parseJSON<MetricPayload & { k?: number; residual?: number }>((event as MessageEvent<string>).data);
    if (!payload) {
      return;
    }
    handleMetrics(normalizeMetric(payload), 'metrics_batch');
  });

  es.addEventListener('metrics_epoch', (event) => {
    const payload = parseJSON<MetricPayload & { k?: number; residual?: number }>((event as MessageEvent<string>).data);
    if (!payload) {
      return;
    }
    const metric = normalizeMetric({ ...payload, step: payload.step ?? 0 });
    handleMetrics(metric, 'metrics_epoch');
  });

  es.addEventListener('log', (event) => {
    const payload = parseJSON<{ level?: string; msg?: string; message?: string; time_unix?: number }>((event as MessageEvent<string>).data);
    if (!payload) {
      return;
    }
    handleLog(payload);
  });

  es.addEventListener('spikes', (event) => {
    const payload = parseJSON<SpikePayload>((event as MessageEvent<string>).data);
    if (!payload) {
      return;
    }
    handleSpikes(payload);
  });
};

const connect = () => {
  if (!storeInstance) {
    return;
  }

  if (source) {
    source.close();
    source = null;
  }

  try {
    source = new EventSource('/events');
  } catch (err) {
    console.warn('Failed to open SSE channel', err);
    scheduleReconnect();
    return;
  }

  bindEventHandlers(source);

  source.onerror = (err) => {
    console.warn('EventSource error', err);
    source?.close();
    scheduleReconnect();
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

export const getSocket = () => source;
