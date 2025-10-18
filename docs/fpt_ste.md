# 固定点残差与 STE‑BPTT 说明

## 1. 固定点残差
- 函数残差： \( r_{\text{fp}}(\mathbf{v}) = \|\mathcal{F}(\mathbf{v})-\mathbf{v}\|_2 / (\|\mathbf{v}\|_2 + \varepsilon) \)
- 步间差分残差： \( r_{\Delta}(t) = \|\mathbf{v}^{t}-\mathbf{v}^{t-1}\|_2 / (\|\mathbf{v}^{t-1}\|_2 + \varepsilon) \)，并在 epoch 或 batch 内做时间平均。

本项目在日志事件 `snn.train.fpt.iter` 中上报 **步间差分** 形式（对 soma 电位）。

## 2. 三隔室与近似梯度
离散更新见代码 `internal/snn/model.go`。读出层使用时间聚合解码（Σ_t v_s^t）。为了对不可导的发放函数求梯度，采用三角 surrogate：
\[
\psi_\gamma(x) = \max(0, 1-|x|/\gamma)/\gamma,\quad
\frac{\partial H(x)}{\partial x} \approx \psi_\gamma(x).
\]

有 \(\partial v_s^{t+1}/\partial u_s^{t+1} \approx I - \theta \psi_\gamma(u_s^{t+1}-\theta)\)。  
时间反传与参数梯度骨架请参考源码注释（Forward/Backprop 上方）。

## 3. 实用建议
- 先只训练读出层（W_out, b_out）快速得到可用 Acc；  
- 若开启端到端：小学习率、γ∈[0.1,0.3]、梯度裁剪到 [-5,5]；  
- 按需把 basal/apical 状态也纳入残差统计用于更严格的 FPT 监控。
