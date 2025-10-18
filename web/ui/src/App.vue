<template>
  <div class="layout">
    <section class="left">
      <h3>训练日志</h3>
      <div class="badges">
        <span class="badge">loss: {{ lastLoss }}</span>
        <span class="badge">acc: {{ lastAcc }}</span>
        <span class="badge">epoch: {{ epoch }}</span>
      </div>
      <LogPanel :logs="logs" />
    </section>
    <section class="right">
      <h3>Loss / Acc</h3>
      <MetricChart :loss="lossPoints" :acc="accPoints" />
    </section>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import LogPanel from './components/LogPanel.vue'
import MetricChart from './components/MetricChart.vue'

const logs = ref([])
const lossPoints = ref([])
const accPoints = ref([])
const lastLoss = ref('-')
const lastAcc = ref('-')
const epoch = ref('-')

onMounted(() => {
  const es = new EventSource('/events')

  es.addEventListener('metrics_batch', ev => {
    const o = JSON.parse(ev.data)
    lastLoss.value = o.loss.toFixed(4)
    lastAcc.value = o.acc.toFixed(4)
    epoch.value = String(o.epoch)
    lossPoints.value.push({ x: o.step, y: o.loss })
    accPoints.value.push({ x: o.step, y: o.acc })
  })

  es.addEventListener('metrics_epoch', ev => {
    const o = JSON.parse(ev.data)
    logs.value.push(`[EPOCH] epoch=${o.epoch} loss=${o.loss.toFixed(4)} acc=${o.acc.toFixed(4)}`)
  })

  es.addEventListener('log', ev => {
    const o = JSON.parse(ev.data)
    logs.value.push(o.msg)
  })
})
</script>

<style scoped>
.layout { display: flex; height: 100vh; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
.left  { width: 40%; border-right: 1px solid #e5e7eb; padding: 12px; overflow: auto; }
.right { flex: 1; padding: 12px; }
.badges { display:flex; gap:8px; margin-bottom: 8px; }
.badge { background:#f3f4f6; border-radius: 6px; padding: 2px 8px; font-size: 12px; }
h3 { margin: 0 0 8px; font-weight: 600; }
</style>
