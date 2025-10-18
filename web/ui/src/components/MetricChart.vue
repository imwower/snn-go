<template>
  <canvas ref="cv"></canvas>
</template>

<script setup>
import { onMounted, watch, ref } from 'vue'
const props = defineProps({
  loss: { type: Array, default: () => [] },
  acc:  { type: Array, default: () => [] }
})
const cv = ref(null)
let ctx

onMounted(() => {
  ctx = cv.value.getContext('2d')
  resize()
  window.addEventListener('resize', () => { resize(); draw() })
  draw()
})
watch(() => [props.loss.length, props.acc.length], draw)

function resize(){
  const parent = cv.value.parentElement
  cv.value.width  = parent.clientWidth - 6
  cv.value.height = 420
}

function draw(){
  if(!ctx) return
  ctx.clearRect(0, 0, cv.value.width, cv.value.height)
  const pad = {l:40, t:10, r:20, b:30}
  const W = cv.value.width - pad.l - pad.r
  const H = cv.value.height - pad.t - pad.b
  ctx.strokeStyle = '#e5e7eb'
  ctx.strokeRect(pad.l, pad.t, W, H)

  // scales
  const maxLoss = Math.max(1e-6, ...props.loss.map(d=>d.y))
  const maxAcc  = 1.0
  // plot loss
  ctx.beginPath()
  props.loss.forEach((d,i)=>{
    const x = pad.l + W*(i/Math.max(1,props.loss.length-1))
    const y = pad.t + H*(1 - d.y/maxLoss)
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y)
  })
  ctx.strokeStyle = '#111827'; ctx.stroke()
  // plot acc
  ctx.beginPath()
  props.acc.forEach((d,i)=>{
    const x = pad.l + W*(i/Math.max(1,props.acc.length-1))
    const y = pad.t + H*(1 - d.y/maxAcc)
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y)
  })
  ctx.strokeStyle = '#6b7280'; ctx.stroke()
}
</script>
