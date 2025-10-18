<template>
  <div class="app-shell" :class="{ 'logs-open': showLogs }">
    <div class="content-row">
      <Sidebar class="sidebar" />
      <div class="main-area">
        <Network3D class="network-view" />
        <StatusBar
          class="status-bar"
          :class="{ expanded: showLogs }"
          :show-logs="showLogs"
          @toggle-logs="toggleLogs"
        />
        <transition name="logs-overlay">
          <div v-if="showLogs" class="logs-overlay">
            <LogsPanel />
          </div>
        </transition>
      </div>
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
