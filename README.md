# SNN‑GO (3‑Comp, Fixed‑T, NATS)

事件驱动的三隔室脉冲神经网络（Basal/Apical/Soma），固定时间步训练；所有消息使用 Go 标准库 `encoding/json` 编解码；UI 采用 **Vue 3 + Vite** 实时展示训练日志与 Loss/Acc。

> 代码目录：`cmd/{api,trainer}`、`internal/{config,events,natsbus,data,snn}`、`web/ui`（Vue）。  
> NATS JetStream 流：`SNN_EVENTS`；主题：`snn.train.*`、`snn.metrics.*`、`snn.params.*`、`snn.ui.log.training`。

## 快速开始

1. **启动 NATS + JetStream**
   ```bash
   docker run -it --rm -p 4222:4222 -p 8222:8222 nats:2 -js
   # JetStream 控制台: http://127.0.0.1:8222
   ```


前端（Vue 3）

cd web/ui
npm i
npm run build     # 生成 web/ui/dist


后端（Go）

go mod tidy
# 启动 UI + SSE 网关
go run ./cmd/api
# 另启终端：启动训练服务
go run ./cmd/trainer


浏览器访问：http://127.0.0.1:8000（左侧日志、右侧 Loss/Acc）

配置（config.yaml，JSON 语法）

nats.url：NATS 地址（默认 nats://127.0.0.1:4222）

nats.stream：JetStream 流名（默认 SNN_EVENTS）

training.*：数据集、epoch、batch、T 步、固定点迭代（K、tol）、学习率、三隔室参数等

model.input/output：输入/输出维度

ui.addr：UI 监听地址（默认 :8000）

模型与训练要点

三隔室离散化（见 internal/snn/model.go）；

固定时间步 T 前向；读出层 logits=Σ_t v_s_out[t]；

损失与评估解码一致：交叉熵 CE(Σ_t v_s, y) + argmax(Σ_t v_s)，避免 “Loss 低 Acc 低”；

先训练读出层（W_out,B_out）即得可观的收敛；可在此基础上扩展 STE‑BPTT 到 W_in/W_b2s/W_a2s。

事件机制（JetStream）

snn.train.init：训练启动超参；

snn.train.fpt.iter：固定点每次迭代残差（K 次）；

snn.metrics.batch / snn.metrics.epoch：批/轮指标；

snn.params.apply / snn.params.snap：参数应用/快照；

snn.ui.log.training：人读友好日志。
UI 的 /events SSE 会推送上述事件到浏览器。

示例关键日志
[INFO] epoch=1 step=1   loss=2.3191 acc=0.086
[INFO] epoch=1 step=50  loss=1.8427 acc=0.362
[EPOCH] epoch=1 loss=1.3872 acc=0.585
[INFO] epoch=2 step=50  loss=0.9731 acc=0.773
[EPOCH] epoch=2 loss=0.8420 acc=0.804
[EPOCH] epoch=3 loss=0.7214 acc=0.835
training finished

常见问题

Loss 降 Acc 不升：确认训练/评估都使用 Σ_t v_s_out[t]；评估禁用噪声；检查标签对齐与 NATS 去重窗口。

SSE 无数据：检查 NATS 是否启动、是否订阅 snn.metrics.*/snn.ui.log.training。

config 解析失败：config.yaml 必须是 JSON 语法。

许可

MIT
