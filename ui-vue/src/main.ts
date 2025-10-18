import { createApp } from 'vue';
import { createPinia } from 'pinia';
import axios from 'axios';
import App from './App.vue';
import './styles.css';
import { startSocket } from './ws';
import { useUiStore } from './store/ui';
import type { LogPayload, MetricPayload, SpikePayload } from './types';

const pinia = createPinia();
const store = useUiStore(pinia);

const extractItems = <T>(payload: unknown): T[] => {
  if (Array.isArray(payload)) {
    return payload as T[];
  }
  if (payload && typeof payload === 'object') {
    const items = (payload as { items?: unknown }).items;
    if (Array.isArray(items)) {
      return items as T[];
    }
  }
  return [];
};

const hydrateFromHistory = async () => {
  try {
    const { data } = await axios.get<MetricPayload[] | { items?: MetricPayload[] }>('/api/metrics/recent', {
      params: { limit: 200 }
    });
    const metrics = extractItems<MetricPayload>(data);
    if (metrics.length > 0) {
      store.replaceMetrics(metrics);
    }
  } catch (err) {
    console.warn('Failed to fetch recent metrics', err);
    store.showToast('无法获取最新指标', 'error');
  }

  try {
    const { data } = await axios.get<SpikePayload[] | { items?: SpikePayload[] }>('/api/spikes/recent', {
      params: { limit: 30 }
    });
    const spikes = extractItems<SpikePayload>(data);
    if (spikes.length > 0) {
      store.replaceSpikes(spikes);
    }
  } catch (err) {
    console.warn('Failed to fetch recent spikes', err);
    store.showToast('无法获取最新神经元放电数据', 'error');
  }

  try {
    const { data } = await axios.get<LogPayload[] | { items?: LogPayload[] }>('/api/logs/recent', {
      params: { limit: 200 }
    });
    const logs = extractItems<LogPayload>(data);
    if (logs.length > 0) {
      store.replaceLogs(logs);
    }
  } catch (err) {
    console.warn('Failed to fetch recent logs', err);
    store.showToast('无法获取最新训练日志', 'error');
  }
};

void hydrateFromHistory();
startSocket(pinia);

const app = createApp(App);
app.use(pinia);
app.mount('#app');
