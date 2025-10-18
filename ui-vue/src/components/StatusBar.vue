<template>
  <footer>
    <span class="status-badge" :class="statusClass">{{ status }}</span>
    <div class="metrics">
      <svg class="sparkline" viewBox="0 0 100 30" preserveAspectRatio="none">
        <polyline
          v-if="sparkPoints"
          :points="sparkPoints"
          fill="none"
          stroke="var(--accent)"
          stroke-width="1.5"
          stroke-linejoin="round"
          stroke-linecap="round"
        />
        <line x1="0" y1="28" x2="100" y2="28" stroke="rgba(255,255,255,0.12)" stroke-width="1" />
      </svg>
      <span>Epoch: {{ metric.epoch }}</span>
      <span>Step: {{ metric.step }}</span>
      <span>Loss: {{ metric.loss }}</span>
      <span>Acc: {{ metric.acc }}</span>
      <span>Throughput: {{ metric.throughput }}</span>
      <span>Residual: {{ metric.residual }}</span>
    </div>
    <div class="message-pill">
      <strong>{{ message.subject }}</strong>
      <span>{{ message.summary }}</span>
    </div>
  </footer>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useUiStore } from '../store/ui';
import type { LogPayload, MetricPayload } from '../types';

const store = useUiStore();

const status = computed(() => store.status);

const statusClass = computed(() => `status-${status.value.toLowerCase()}`);

const formatNumber = (value: number | undefined, digits: number) =>
  typeof value === 'number' ? value.toFixed(digits) : '--';

const formatResidual = (value: number | undefined) =>
  typeof value === 'number' ? value.toExponential(2) : '--';

const metric = computed(() => {
  const last = store.lastMetric;
  if (!last) {
    return {
      epoch: '--',
      step: '--',
      loss: '--',
      acc: '--',
      throughput: '--',
      residual: '--'
    };
  }
  return {
    epoch: last.epoch ?? '--',
    step: last.step ?? '--',
    loss: formatNumber(last.loss, 4),
    acc: formatNumber(last.acc, 3),
    throughput: formatNumber(last.throughput, 1),
    residual: formatResidual(last.residual)
  };
});

const sparkPoints = computed(() => {
  const entries = store.metrics.slice(-60);
  if (entries.length < 2) {
    return '';
  }
  const values = entries.map((entry) => (
    typeof entry.loss === 'number' ? entry.loss : 0
  ));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  return entries
    .map((entry, index) => {
      const x = (index / (entries.length - 1)) * 100;
      const value = typeof entry.loss === 'number' ? entry.loss : min;
      const normalized = range === 0 ? 0.5 : (value - min) / range;
      const y = 28 - normalized * 24;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(' ');
});

const describeMessage = (entry: { subject: string; type?: string; payload?: unknown } | undefined) => {
  if (!entry) {
    return { subject: '无消息', summary: '等待流数据...' };
  }
  if (entry.type === 'metrics' && entry.payload && typeof entry.payload === 'object') {
    const metricPayload = entry.payload as MetricPayload;
    const loss = formatNumber(metricPayload.loss, 4);
    return {
      subject: entry.subject || 'metrics',
      summary: `loss ${loss}`
    };
  }
  if (entry.type === 'log' && entry.payload && typeof entry.payload === 'object') {
    const logPayload = entry.payload as LogPayload;
    const metricPayload = logPayload.metric as MetricPayload | undefined;
    const loss = metricPayload ? formatNumber(metricPayload.loss, 4) : '--';
    const acc = metricPayload ? formatNumber(metricPayload.acc, 3) : '--';
    return {
      subject: logPayload.message || 'log',
      summary: `loss ${loss} · acc ${acc}`
    };
  }
  if (entry.type === 'spikes' && entry.payload && typeof entry.payload === 'object') {
    const spike = entry.payload as { layer?: number; neurons?: number[] };
    const count = Array.isArray(spike.neurons) ? spike.neurons.length : 0;
    return {
      subject: entry.subject || 'spikes',
      summary: `layer ${spike.layer ?? '?'} · ${count} spikes`
    };
  }
  const preview = JSON.stringify(entry.payload)?.slice(0, 40) ?? '';
  return {
    subject: entry.subject || entry.type || 'event',
    summary: preview || '收到事件'
  };
};

const message = computed(() => {
  const last = store.messages[store.messages.length - 1];
  const base = describeMessage(last);
  if (last) {
    const time = new Date(last.at).toLocaleTimeString('zh-CN', { hour12: false });
    base.summary = `${base.summary} · ${time}`;
  }
  return base;
});
</script>
