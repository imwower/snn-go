<template>
  <aside>
    <div class="scrollable">
      <section>
        <label>
          <span>数据集</span>
          <select v-model="dataset">
            <option v-for="option in datasetOptions" :key="option.value" :value="option.value">
              {{ option.label }}
            </option>
          </select>
        </label>
        <button @click="downloadDataset" :disabled="isBusy">
          下载到本地
        </button>
      </section>
      <fieldset>
        <legend>训练参数</legend>
        <label>
          <span>network_size</span>
          <input type="number" min="1" v-model.number="networkSize" />
          <span class="hint">整体神经元数量，建议 512 ~ 4096，越大越细腻但渲染压力更高。</span>
        </label>
        <label>
          <span>layers</span>
          <input type="number" min="1" v-model.number="layers" />
          <span class="hint">网络层数，2 ~ 6 层较易观察；层数越多可视化深度越大。</span>
        </label>
        <label>
          <span>lr</span>
          <input type="number" step="0.0001" v-model.number="lr" />
          <span class="hint">学习率，默认 1e-3；尝试 5e-4 ~ 2e-3 平衡收敛速度与稳定性。</span>
        </label>
        <label>
          <span>K</span>
          <input type="number" min="1" v-model.number="K" />
          <span class="hint">K 表示突触邻域大小，10~30 可形成适中稀疏连接。</span>
        </label>
        <label>
          <span>tol</span>
          <input type="number" step="0.000001" v-model.number="tol" />
          <span class="hint">容差阈值控制停止条件，越小越精准但耗时更久。</span>
        </label>
      </fieldset>
    </div>
    <div class="buttons">
      <button @click="initTraining" :disabled="isBusy">初始化</button>
      <button @click="startTraining" :disabled="isBusy">训练</button>
      <button @click="stopTraining" :disabled="isBusy">停止</button>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import axios from 'axios';
import { useUiStore } from '../store/ui';
import type { DatasetListPayload, DatasetName } from '../types';

const store = useUiStore();
interface DatasetOption {
  value: DatasetName;
  label: string;
  status?: string;
  message?: string | null;
}
const datasetOptions = ref<DatasetOption[]>([{ value: store.cfg.dataset, label: store.cfg.dataset }]);
const isBusy = ref(false);

const dataset = computed({
  get: () => store.cfg.dataset,
  set: (value: DatasetName) => store.setDataset(value)
});
const networkSize = computed({
  get: () => store.cfg.network_size,
  set: (value: number) => store.setCfg({ network_size: value })
});
const layers = computed({
  get: () => store.cfg.layers,
  set: (value: number) => store.setCfg({ layers: value })
});
const lr = computed({
  get: () => store.cfg.lr,
  set: (value: number) => store.setCfg({ lr: value })
});
const K = computed({
  get: () => store.cfg.K,
  set: (value: number) => store.setCfg({ K: value })
});
const tol = computed({
  get: () => store.cfg.tol,
  set: (value: number) => store.setCfg({ tol: value })
});

const withBusy = async (task: () => Promise<void>) => {
  if (isBusy.value) {
    return;
  }
  isBusy.value = true;
  try {
    await task();
  } finally {
    isBusy.value = false;
  }
};

const loadDatasets = async () => {
  try {
    const { data } = await axios.get<DatasetListPayload | DatasetName[]>('/api/datasets');
    const options = normalizeDatasetList(data);
    if (options.length > 0) {
      datasetOptions.value = options;
      if (!options.some((option) => option.value === store.cfg.dataset)) {
        store.setDataset(options[0]?.value ?? store.cfg.dataset);
      }
      return;
    }
    console.warn('Dataset API returned empty payload', data);
    store.showToast('数据集列表为空', 'error');
  } catch (err) {
    console.warn('Failed to load datasets', err);
    store.showToast('无法加载数据集列表', 'error');
  }
};

onMounted(() => {
  void loadDatasets();
});

const downloadDataset = () =>
  withBusy(async () => {
    try {
      await axios.post('/api/datasets/download', { name: dataset.value });
    } catch (err) {
      console.warn('Download failed', err);
      store.showToast('数据集下载失败', 'error');
    }
  });

const initTraining = () =>
  withBusy(async () => {
    try {
      store.setStatus('Initializing');
      await axios.post('/api/train/init', {
        dataset: store.cfg.dataset,
        mode: store.cfg.mode,
        network_size: store.cfg.network_size,
        layers: store.cfg.layers,
        lr: store.cfg.lr,
        K: store.cfg.K,
        tol: store.cfg.tol,
        T: store.cfg.T
      });
    } catch (err) {
      console.warn('Init failed', err);
      store.showToast('初始化训练失败', 'error');
      store.setStatus('Idle');
    }
  });

const startTraining = () =>
  withBusy(async () => {
    try {
      await axios.post('/api/train/start', {});
      store.setStatus('Training');
    } catch (err) {
      console.warn('Start failed', err);
      store.showToast('启动训练失败', 'error');
    }
  });

const stopTraining = () =>
  withBusy(async () => {
    try {
      await axios.post('/api/train/stop', {});
      store.setStatus('Stopped');
    } catch (err) {
      console.warn('Stop failed', err);
      store.showToast('停止训练失败', 'error');
    }
  });

function normalizeDatasetList(payload: unknown): DatasetOption[] {
  const result: DatasetOption[] = [];
  const seen = new Set<string>();
  const pushOption = (option: DatasetOption | null) => {
    if (!option || seen.has(option.value)) {
      return;
    }
    seen.add(option.value);
    result.push(option);
  };

  if (Array.isArray(payload)) {
    payload.forEach((entry) => pushOption(toDatasetOption(entry)));
    return result;
  }
  if (payload && typeof payload === 'object') {
    const obj = payload as DatasetListPayload;
    if (Array.isArray(obj.datasets)) {
      obj.datasets.forEach((entry) => pushOption(toDatasetOption(entry)));
    }
    if (Array.isArray(obj.available)) {
      obj.available.forEach((entry) => pushOption(toDatasetOption(entry)));
    }
    if (Array.isArray(obj.installed)) {
      obj.installed.forEach((entry) => pushOption(toDatasetOption(entry)));
    }
  }
  return result;
}

function toDatasetOption(entry: unknown): DatasetOption | null {
  if (typeof entry === 'string') {
    const value = entry.trim();
    if (!value) {
      return null;
    }
    return { value: value as DatasetName, label: value };
  }
  if (entry && typeof entry === 'object') {
    const record = entry as Record<string, unknown>;
    const rawSlug = typeof record.slug === 'string' ? record.slug.trim() : undefined;
    const rawName = typeof record.name === 'string' ? record.name.trim() : undefined;
    const rawDataset = typeof record.dataset === 'string' ? record.dataset.trim() : undefined;
    const rawValue = typeof record.value === 'string' ? record.value.trim() : undefined;
    const value = rawSlug || rawDataset || rawValue || rawName;
    const label = rawName || rawSlug || rawDataset || rawValue;
    if (!value || !label) {
      return null;
    }
    return {
      value: value as DatasetName,
      label,
      status: typeof record.status === 'string' ? record.status : undefined,
      message: typeof record.message === 'string' ? record.message : undefined
    };
  }
  return null;
}
</script>
