<template>
  <div class="app-shell">
    <div class="content-row">
      <Sidebar class="sidebar" />
      <Network3D class="network-view" />
    </div>
    <StatusBar class="status-bar" />
    <div class="logs-wrapper">
      <button class="logs-toggle" type="button" @click="toggleLogs">
        {{ showLogs ? '隐藏日志' : '显示日志' }}
      </button>
      <LogsPanel v-if="showLogs" />
    </div>
    <Toast />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import Sidebar from './components/Sidebar.vue';
import Network3D from './components/Network3D.vue';
import StatusBar from './components/StatusBar.vue';
import Toast from './components/Toast.vue';
import LogsPanel from './components/LogsPanel.vue';
import { useUiStore } from './store/ui';

useUiStore();

const showLogs = ref(false);
const toggleLogs = () => {
  showLogs.value = !showLogs.value;
};
</script>
