# 本轮已实现方法与口径

## 执行对象

InternVL3-8B-hf，实际推理 dtype FP16，固定图像视觉前缀，之后复用缓存而不重新编码图像。所有视觉位置保持可见，原始 post-RoPE 位置不变。历史文本 KV 保持 FP16；本轮不把“提高已错误计算的文本 KV 存储位宽”作为修复。

首轮视觉/文本前缀使用公共 native FP16 prefill，其第一个答案 token 的 logits 已经形成。压缩从该前缀之后的缓存读取生效；后续问题 prefill 和逐 token decode 都可能受到影响。

## 表示

- `full`：同执行器 FP16 视觉 KV。
- `base2`：官方 KIVI 编码算子的视觉适配，K 按 channel、V 按 token，g=128；非完整 KIVI serving 复现。
- `native4`：单套独立 4-bit，同样的融合执行器。
- `asym`：K2 + K残差2，V2 + sign/组均值幅度。含 scale/min/alpha，真实 packed 张量驻留；默认读基础，按事件选择残差。
- `base1/asym1`：本地 1-bit 基底扩展，独立 GPU 数值检查；不能称官方 KIVI1。
- `_k/_v`：只读相应残差轴。后续 compact 版本只拥有该轴残差，未用参数以 alias 占位；旧试验的双轴存储按其原始实测账本保留。
- `kvarn2/kvarn4`：锁定官方底层旋转/双轴方差/量化函数的 mechanism-port。当前在统一 attention 前物化，packed reference 与 decoded owner 都计入；仅作为质量参照。

## 预算化事件策略

统一预算为固定常驻渐进表示，加每会话额外逻辑读取额度。额外读包括残差，以及 probe/replay/repair 额外发生的基底与文本读取。每次融合调用的原始账本与累计预算计数必须完全一致。

| 策略 | 在线信息与动作 |
|---|---|
| base | 仅基础读取 |
| uniform/random/periodic/prefill/early | 冻结的层组/时间规则，不看未来 |
| margin | 低精度试算后，根据当前 logits 决定回滚重放前驱，改变下一词选择；两次前向完整计费 |
| margin_write/carrier_write | 已选输入 token 被消费时，依当前 margin/词法内容候选提高视觉读取精度，影响新状态形成 |
| table_current/table_future | 相同8开发图像、相同动作库，分别按当前/未来参考 KL 保真收益标定；在线仅查当前 phase 与 layer group 表 |
| probe_l{L}_t{η} | 同一 q 在 L 层双读基础与精化表示，以相对 L2 决定是否使用精化输出并升级后续层；当前层已有 K/V 和更早层状态不会自动重算 |
| repair_once | 同一生成文本按原 suffix/逐 token 顺序重算最近一轮，单列额外成本；低额度下可因真实重算成本而跳过 |

双读 probe 的 baseline/residual 差是当前表示的纠偏幅度，不是可观测的 FP16 真误差。controller 的两读总量含正常基底读，而全局额外账本只额外计第二读，不能再次整笔相加。

收益表的动作扫描在开发轨迹的第 2/4 轮写入，未来 horizon 为 1/2。Current 与 Future 使用同 8 图、同完整轨迹和同候选动作库，但目标所覆盖的 query 位置/权重并非逐项相同。本轮尚未运行覆盖所有 query 位置的最强 current-only 校准对照；即使某一外部任务出现收益，也不能据此断言“平均失真目标本质失效”。此外 reference fidelity 与下游答案质量是不同目标。

## 机制与应用

同 token 四臂与单位置手术只作机制诊断。连续剂量插值、参考状态切片及未来 oracle 均不冒充部署算法。应用质量来自各方法自由生成并自喂的轨迹；teacher-forced 计时只用于等工作量成本，不对其文本计算应用准确率。

## 资源输出

`owned_initial/final` 为实际张量 owner 字节，包含文本容量预留与位置；`peak_allocated` 是 CUDA 整会话峰值，含模型和临时 workspace。`read_events` 是每次融合调用的逻辑 tensor payload，未包含硬件缓存/重复 query tile 等实际 HBM 事务，不能标作硬件带宽。自然生成时间与固定 token、warmup 后的重复计时分开报告。
