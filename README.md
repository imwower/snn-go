# SNN‑GO (3‑Comp, Fixed‑T, NATS)

事件驱动的三隔室脉冲神经网络（Basal / Apical / Soma），采用固定时间步训练；所有事件消息通过 Go 标准库 `encoding/json` 编解码；前端 UI 使用 **Vue 3 + Vite** 实时展示训练日志与 Loss/Acc。

- 代码目录：`cmd/{api,trainer}`、`internal/{config,events,natsbus,data,snn}`、`ui-vue`（Vue 前端源码）
- JetStream 流：`SNN_EVENTS`
- 默认主题：`snn.train.*`、`snn.metrics.*`、`snn.params.*`、`snn.ui.log.training`

## 快速开始

1. **启动 NATS + JetStream**
   ```bash
   docker run -it --rm -p 4222:4222 -p 8222:8222 nats:2 -js
   # JetStream 控制台: http://127.0.0.1:8222
   ```
2. **构建前端（ui-vue）**  
   API 会直接从 `ui-vue/dist` 提供静态资源。
   ```bash
   cd ui-vue
   npm install
   npm run build   # 生成 ui-vue/dist
   ```
3. **启动后端（Go）**
   ```bash
   go mod tidy
   # UI + SSE 网关
   go run ./cmd/api
   # 新终端：训练服务
   go run ./cmd/trainer
   ```
4. **打开浏览器**  
   访问 `http://127.0.0.1:8000`，左侧为日志，右侧为 Loss / Acc。

## 配置（`config.yaml`，JSON 语法）

- `nats.url`：NATS 地址（默认 `nats://127.0.0.1:4222`）
- `nats.stream`：JetStream 流名（默认 `SNN_EVENTS`）
- `training.*`：数据集、epoch、batch、时间步 T、固定点迭代（K、tol）、学习率、三隔室参数等
- `training.dataset`：可选 `"MNIST"`、`"FASHION"` 或 `"SYNTH"`
- `model.input` / `model.output`：输入 / 输出维度
- `ui.addr`：UI 监听地址（默认 `:8000`）

## 模型与训练要点

- 三隔室离散化实现参见 `internal/snn/model.go`
- 固定时间步 T 前向；读出层 `logits = Σ_t v_s_out[t]`
- 训练与评估解码一致：`CE(Σ_t v_s, y)` + `argmax(Σ_t v_s)`，避免 Loss 低 Acc 低
- 先仅训练读出层（`W_out`, `B_out`）即可获得稳定收敛，之后可扩展至 `W_in` / `W_b2s` / `W_a2s` 的 STE‑BPTT

## JetStream 事件

- `snn.train.init`：训练启动时发布超参
- `snn.train.fpt.iter`：固定点迭代（K 次）剩余误差
- `snn.metrics.batch` / `snn.metrics.epoch`：批次 / 轮次指标
- `snn.params.apply` / `snn.params.snap`：参数应用与快照
- `snn.ui.log.training`：面向人类的训练日志

`/events` SSE 会把以上消息推送给浏览器前端。

## 示例日志

```text
[INFO] epoch=1 step=1   loss=2.3191 acc=0.086
[INFO] epoch=1 step=50  loss=1.8427 acc=0.362
[EPOCH] epoch=1 loss=1.3872 acc=0.585
[INFO] epoch=2 step=50  loss=0.9731 acc=0.773
[EPOCH] epoch=2 loss=0.8420 acc=0.804
[EPOCH] epoch=3 loss=0.7214 acc=0.835
training finished
```

## 常见问题

- Loss 降但 Acc 不升：确认训练 / 评估都使用 `Σ_t v_s_out[t]`，评估禁用噪声，并检查标签对齐与 NATS 去重窗口
- SSE 无数据：检查 NATS 是否运行，并确认订阅了 `snn.metrics.*` / `snn.ui.log.training`
- `config.yaml` 解析失败：文件必须保持 JSON 语法，不能添加注释

## 许可

MIT

## 开发脚本

- 清空 JetStream 流：`./scripts/purge-nats.sh [容器名] [流名]`  
  默认目标为容器 `nats-js` 和流 `SNN_EVENTS`。脚本会在容器内安装 `nats` CLI（若缺失），然后执行 `nats stream purge <流名> --force`。
