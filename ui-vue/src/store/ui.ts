import { defineStore } from 'pinia';
import type {
  DatasetName,
  LayerLayout,
  MetricEntry,
  MetricPayload,
  MessageEntry,
  SpikeEntry,
  SpikePayload,
  TrainingConfig,
  TrainingStatus,
  LogPayload,
  LogEntry,
  DatasetDownloadEvent
} from '../types';

const MAX_METRICS = 500;
const MAX_SPIKES = 60;
const MAX_MESSAGES = 100;
const MAX_LOGS = 500;
const DEFAULT_TOAST_DURATION = 3000;

const defaultConfig = (): TrainingConfig => ({
  dataset: 'MNIST',
  mode: 'tstep',
  network_size: 1024,
  layers: 4,
  lr: 1e-3,
  K: 10,
  tol: 1e-4,
  T: 20
});

const buildLayout = (cfg: TrainingConfig): LayerLayout[] => {
  const layers: LayerLayout[] = [];
  const perLayer = Math.max(1, Math.ceil(cfg.network_size / Math.max(1, cfg.layers)));
  const layerSpacing = 6;
  const gridSpacing = 1.4;

  for (let layerIndex = 0; layerIndex < cfg.layers; layerIndex += 1) {
    const count = perLayer;
    const positions = new Float32Array(count * 3);
    const columns = Math.ceil(Math.sqrt(count));
    const rows = Math.ceil(count / columns);
    const x = (layerIndex - (cfg.layers - 1) / 2) * layerSpacing;

    for (let i = 0; i < count; i += 1) {
      const row = Math.floor(i / columns);
      const col = i % columns;
      const y = ((rows - 1) / 2 - row) * gridSpacing;
      const z = (col - (columns - 1) / 2) * gridSpacing;
      const offset = i * 3;
      positions[offset] = x;
      positions[offset + 1] = y;
      positions[offset + 2] = z;
    }

    layers.push({
      layer: layerIndex,
      count,
      startIndex: layerIndex * perLayer,
      positions
    });
  }

  return layers;
};

export const useUiStore = defineStore('ui', {
  state: () => ({
    cfg: defaultConfig(),
    status: 'Idle' as TrainingStatus,
    metrics: [] as MetricEntry[],
    lastMetric: null as MetricEntry | null,
    spikes: [] as SpikeEntry[],
    layersLayout: buildLayout(defaultConfig()),
    messages: [] as MessageEntry[],
    logs: [] as LogEntry[],
    toast: null as { message: string; type: 'info' | 'error'; at: number; duration?: number } | null,
    download: {
      active: false,
      name: '',
      progress: 0,
      startedAt: 0
    }
  }),
  getters: {
    totalNodes(state) {
      return state.layersLayout.reduce((acc, layer) => acc + layer.count, 0);
    },
    isDownloadActive(state) {
      return state.download.active;
    },
    downloadPercent(state) {
      return Math.round(state.download.progress * 100);
    },
    isControlLocked(state) {
      return state.download.active || state.status === 'Initializing';
    }
  },
  actions: {
    setCfg(partial: Partial<TrainingConfig>) {
      const current = this.cfg;
      const next: TrainingConfig = { ...current, ...partial };
      next.layers = Math.max(1, Math.round(Number.isFinite(next.layers) ? next.layers : current.layers));
      next.network_size = Math.max(1, Math.round(Number.isFinite(next.network_size) ? next.network_size : current.network_size));
      const lrValue = Number(next.lr);
      next.lr = Number.isFinite(lrValue) ? lrValue : current.lr;
      next.K = Math.max(1, Math.round(Number.isFinite(next.K) ? next.K : current.K));
      const tolValue = Number(next.tol);
      next.tol = Number.isFinite(tolValue) && tolValue > 0 ? tolValue : current.tol;
      if (typeof next.T === 'number' && Number.isFinite(next.T)) {
        next.T = Math.max(1, Math.round(next.T));
      } else {
        next.T = current.T;
      }
      this.cfg = next;
      this.layersLayout = buildLayout(next);
    },
    setDataset(name: DatasetName) {
      this.setCfg({ dataset: name });
    },
    setStatus(status: TrainingStatus) {
      this.status = status;
    },
    pushMetric(payload: MetricPayload) {
      const entry: MetricEntry = { ...payload, at: Date.now() };
      this.metrics.push(entry);
      if (this.metrics.length > MAX_METRICS) {
        this.metrics.splice(0, this.metrics.length - MAX_METRICS);
      }
      this.lastMetric = entry;
    },
    pushSpike(payload: SpikePayload) {
      const entry: SpikeEntry = { ...payload, at: Date.now() };
      this.spikes.push(entry);
      if (this.spikes.length > MAX_SPIKES) {
        this.spikes.splice(0, this.spikes.length - MAX_SPIKES);
      }
    },
    pushMessage(subject: string, payload?: unknown, type?: string) {
      const entry: MessageEntry = {
        subject,
        type,
        payload,
        at: Date.now()
      };
      this.messages.push(entry);
      if (this.messages.length > MAX_MESSAGES) {
        this.messages.splice(0, this.messages.length - MAX_MESSAGES);
      }
    },
    pushLog(payload: LogPayload) {
      const timestamp = typeof payload.ts === 'number' ? payload.ts * 1000 : Date.now();
      const entry: LogEntry = {
        ...payload,
        at: timestamp
      };
      this.logs.push(entry);
      if (this.logs.length > MAX_LOGS) {
        this.logs.splice(0, this.logs.length - MAX_LOGS);
      }
    },
    pushPlainLog(message: string, level: LogPayload['level'] = 'INFO', ts?: number) {
      const unixSeconds = typeof ts === 'number' ? ts : Math.floor(Date.now() / 1000);
      this.pushLog({
        ts: unixSeconds,
        level,
        message
      });
    },
    replaceMetrics(payloads: MetricPayload[]) {
      const entries = payloads.slice(-MAX_METRICS).map((payload) => {
        const historic = payload as MetricPayload & { at?: number };
        const timestamp = typeof historic.at === 'number' ? historic.at : Date.now();
        return { ...payload, at: timestamp };
      });
      this.metrics = entries;
      this.lastMetric = entries.length > 0 ? entries[entries.length - 1] : null;
    },
    replaceSpikes(payloads: SpikePayload[]) {
      const entries = payloads.slice(-MAX_SPIKES).map((payload) => {
        const historic = payload as SpikePayload & { at?: number };
        const timestamp = typeof historic.at === 'number' ? historic.at : Date.now();
        return { ...payload, at: timestamp };
      });
      this.spikes = entries;
    },
    replaceLogs(payloads: LogPayload[]) {
      const entries = payloads.slice(-MAX_LOGS).map((payload) => {
        const ts = typeof payload.ts === 'number' ? payload.ts * 1000 : Date.now();
        return {
          ...payload,
          at: ts
        } as LogEntry;
      });
      this.logs = entries;
    },
    clearSpikes() {
      this.spikes = [];
    },
    showToast(message: string, type: 'info' | 'error' = 'info', duration = DEFAULT_TOAST_DURATION) {
      this.toast = {
        message,
        type,
        at: Date.now(),
        duration
      };
    },
    clearToast(at?: number) {
      if (!this.toast) {
        return;
      }
      if (typeof at === 'number' && this.toast.at !== at) {
        return;
      }
      this.toast = null;
    },
    startDownload(name: string) {
      if (this.download.active && this.download.name === name) {
        return;
      }
      this.download = {
        active: true,
        name,
        progress: 0,
        startedAt: Date.now()
      };
      this.pushMessage('dataset', { name, state: 'start' }, 'dataset_download');
      this.pushPlainLog(`开始下载数据集 ${name}`, 'INFO');
    },
    updateDownloadProgress(progress: number) {
      if (!this.download.active) {
        return;
      }
      const clamped = Number.isFinite(progress) ? Math.min(Math.max(progress, 0), 1) : this.download.progress;
      this.download.progress = clamped;
    },
    finishDownload(success: boolean, message?: string, name?: string) {
      if (!this.download.active || (name && this.download.name !== name)) {
        return;
      }
      const datasetName = this.download.name;
      if (success && this.download.progress < 1) {
        this.download.progress = 1;
      }
      this.download = {
        active: false,
        name: '',
        progress: 0,
        startedAt: 0
      };
      if (success) {
        this.pushPlainLog(`数据集 ${datasetName} 下载完成`, 'INFO');
      } else {
        this.pushPlainLog(`数据集 ${datasetName} 下载失败${message ? `：${message}` : ''}`, 'ERROR');
      }
      this.pushMessage('dataset', { name: datasetName, success, message }, 'dataset_download');
    },
    applyDatasetEvent(event: DatasetDownloadEvent) {
      if (!event || !event.name) {
        return;
      }
      const state = event.state ?? '';
      if (state === 'start') {
        this.startDownload(event.name);
        if (typeof event.progress === 'number') {
          this.updateDownloadProgress(event.progress);
        }
        return;
      }
      if (state === 'progress') {
        this.startDownload(event.name);
        if (typeof event.progress === 'number') {
          this.updateDownloadProgress(event.progress);
        }
        return;
      }
      if (state === 'complete') {
        const progress = typeof event.progress === 'number' ? event.progress : 1;
        this.updateDownloadProgress(progress);
        this.finishDownload(true, undefined, event.name);
        return;
      }
      if (state === 'error') {
        this.finishDownload(false, event.message, event.name);
      }
    },
    reset() {
      this.cfg = defaultConfig();
      this.status = 'Idle';
      this.metrics = [];
      this.lastMetric = null;
      this.spikes = [];
      this.layersLayout = buildLayout(this.cfg);
      this.messages = [];
      this.logs = [];
      this.toast = null;
      this.download = {
        active: false,
        name: '',
        progress: 0,
        startedAt: 0
      };
    }
  }
});
