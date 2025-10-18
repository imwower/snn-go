<template>
  <div class="log">
    <p v-for="(l,i) in logs" :key="i" class="line">{{ l }}</p>
  </div>
</template>

<script setup>
defineProps({ logs: { type: Array, default: () => [] } })
</script>

<style scoped>
.log { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; height: calc(100vh - 100px); overflow: auto; }
.line { margin: 2px 0; white-space: pre; }
</style>
