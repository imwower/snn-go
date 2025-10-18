import type { Pinia } from 'pinia';
import { useUiStore } from './store/ui';
import type {
  MetricPayload,
  TrainInitEvent,
  TrainIterEvent,
  UISysLogEvent,
  LogPayload,
  DatasetDownloadEvent
} from './types';

let source: EventSource | null = null;
let started = false;
let storeInstance: ReturnType<typeof useUiStore> | null = null;

const toLogLevel = (level?: string): LogPayload['level'] => {
  const normalized = (level || 'INFO').toUpperCase();
  if (normalized === 'DEBUG' || normalized === 'INFO' || normalized === 'WARNING' || normalized === 'ERROR') {
    return normalized;
  }
  return 'INFO';
};

const parseJSON = <T>(raw: string): T | null => {
  try {
    return JSON.parse(raw) as T;
  } catch (err) {
    console.warn('无法解析 SSE 负载', err);
    return null;
  }
};

const pushTextLog = (message: string, ts?: number, level: LogPayload['level'] = 'INFO') => {
  storeInstance?.pushPlainLog(message, level, ts);
  storeInstance?.pushMessage('log-text', { message, ts, level }, 'log');
};

const handleMetricsBatch = (payload: MetricPayload) => {
  storeInstance?.pushMetric(payload);
  if (storeInstance && storeInstance.status === 'Idle') {
    storeInstance.setStatus('Training');
  }
  storeInstance?.pushMessage('metrics_batch', payload, 'metrics_batch');
};

const handleMetricsEpoch = (payload: MetricPayload) => {
  const loss = typeof payload.loss === 'number' ? payload.loss.toFixed(4) : 'n/a';
  const acc = typeof payload.acc === 'number' ? payload.acc.toFixed(4) : 'n/a';
  pushTextLog(`[EPOCH] epoch=${payload.epoch ?? '-'} loss=${loss} acc=${acc}`, payload.time_unix);
  storeInstance?.pushMessage('metrics_epoch', payload, 'metrics_epoch');
};

const handleSysLog = (payload: UISysLogEvent) => {
  const level = toLogLevel(payload.level);
  const message = payload.msg ?? '';
  console.log('[UI] 日志', message);
  storeInstance?.pushPlainLog(message, level, payload.time_unix);
  storeInstance?.pushMessage('log', payload, 'log');
};

const handleTrainInit = (payload: TrainInitEvent) => {
  console.log('[UI] train_init', payload);
  const text = `[INIT] dataset=${payload.dataset ?? '-'} epochs=${payload.epochs ?? '-'} batch=${payload.batch_size ?? '-'} T=${payload.timesteps ?? '-'} K=${payload.fixed_point_K ?? '-'} lr=${payload.lr ?? '-'}`;
  storeInstance?.setStatus('Idle');
  if (storeInstance) {
    const current = storeInstance.cfg;
    storeInstance.setCfg({
      dataset: payload.dataset ?? current.dataset,
      lr: typeof payload.lr === 'number' ? payload.lr : current.lr,
      K: typeof payload.fixed_point_K === 'number' ? payload.fixed_point_K : current.K,
      T: typeof payload.timesteps === 'number' ? payload.timesteps : current.T
    });
  }
  storeInstance?.pushPlainLog(text, 'INFO', payload.time_unix);
  storeInstance?.pushMessage('train_init', payload, 'train_init');
};

const handleTrainIter = (payload: TrainIterEvent) => {
  if (typeof payload.residual === 'number') {
    console.log('[UI] train_iter residual', payload.residual);
  }
  const residualText =
    typeof payload.residual === 'number' ? payload.residual.toFixed(6) : String(payload.residual ?? 'n/a');
  pushTextLog(
    `[FPT] epoch=${payload.epoch ?? '-'} step=${payload.step ?? '-'} residual=${residualText}`,
    payload.time_unix
  );
  storeInstance?.pushMessage('train_iter', payload, 'train_iter');
};

const handleDatasetDownload = (payload: DatasetDownloadEvent) => {
  storeInstance?.applyDatasetEvent(payload);
};

const setupListeners = (evSource: EventSource) => {
  evSource.addEventListener('metrics_batch', (event: MessageEvent<string>) => {
    const payload = parseJSON<MetricPayload>(event.data);
    if (payload) {
      handleMetricsBatch(payload);
    }
  });

  evSource.addEventListener('metrics_epoch', (event: MessageEvent<string>) => {
    const payload = parseJSON<MetricPayload>(event.data);
    if (payload) {
      handleMetricsEpoch(payload);
    }
  });

  evSource.addEventListener('log', (event: MessageEvent<string>) => {
    const payload = parseJSON<UISysLogEvent>(event.data);
    if (payload) {
      handleSysLog(payload);
    }
  });

  evSource.addEventListener('train_init', (event: MessageEvent<string>) => {
    const payload = parseJSON<TrainInitEvent>(event.data);
    if (payload) {
      handleTrainInit(payload);
    }
  });

  evSource.addEventListener('train_iter', (event: MessageEvent<string>) => {
    const payload = parseJSON<TrainIterEvent>(event.data);
    if (payload) {
      handleTrainIter(payload);
    }
  });
  evSource.addEventListener('dataset_download', (event: MessageEvent<string>) => {
    const payload = parseJSON<DatasetDownloadEvent>(event.data);
    if (payload) {
      handleDatasetDownload(payload);
    }
  });

  evSource.addEventListener('open', () => {
    storeInstance?.pushMessage('events', { state: 'open' }, 'sse');
  });

  evSource.onerror = (err) => {
    console.warn('SSE 错误', err);
    storeInstance?.pushMessage('events', { state: 'error', error: err }, 'sse');
  };
};

const fetchConfig = async () => {
  const store = storeInstance;
  if (!store) {
    return;
  }
  try {
    const response = await fetch('/api/config', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`status=${response.status}`);
    }
    const cfg = await response.json();
    const training = cfg?.training as TrainInitEvent | undefined;
    if (training) {
      const text = `[CONFIG] dataset=${training.dataset ?? '-'} epochs=${training.epochs ?? '-'} batch=${training.batch_size ?? '-'} T=${training.timesteps ?? '-'} K=${training.fixed_point_K ?? '-'} lr=${training.lr ?? '-'}`;
      pushTextLog(text);
      const current = store.cfg;
      store.setCfg({
        dataset: training.dataset ?? current.dataset,
        lr: typeof training.lr === 'number' ? training.lr : current.lr,
        K: typeof training.fixed_point_K === 'number' ? training.fixed_point_K : current.K,
        T: typeof training.timesteps === 'number' ? training.timesteps : current.T
      });
    }
    store.pushMessage('config', cfg, 'config');
  } catch (err) {
    console.warn('获取配置失败', err);
    pushTextLog('[WARN] 获取配置失败', undefined, 'WARNING');
  }
};

const connect = () => {
  if (!storeInstance) {
    return;
  }
  const url =
    typeof window !== 'undefined' && import.meta.env.DEV
      ? 'http://127.0.0.1:8000/events'
      : '/events';
  source = new EventSource(url, { withCredentials: false });
  setupListeners(source);
};

export const startSocket = (pinia: Pinia) => {
  if (started) {
    return;
  }
  storeInstance = useUiStore(pinia);
  started = true;
  void fetchConfig();
  connect();
};

export const getSocket = () => source;
