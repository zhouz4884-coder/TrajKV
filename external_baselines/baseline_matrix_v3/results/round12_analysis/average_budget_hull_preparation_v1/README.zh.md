# Round12完整500后的平均预算端点诊断（仅准备，尚未读真实500）

主helper：`round12_average_budget_hull_v1.py`。本目录目前只有代码与CPU合成检查，没有真实500诊断结果；未读取正在运行的collector state、参考pack或候选raw。

待Root确认两域各500闭合，使用实际最终快照路径运行：

```text
python -X utf8 -B ${EVIDENCE_ROOT}/results/round12/average_budget_hull_preparation_v1/round12_average_budget_hull_v1.py --state ${EVIDENCE_ROOT}/results/round12/fixed500_harvest_v1/collector_runs/FINAL_SNAPSHOT/ROOT_STATE.json --output ${EVIDENCE_ROOT}/results/round12/fixed500_average_budget_hull_v1
```

默认pack为`fixed500_harvest_v1/reference_pack_v1/REFERENCE_PACK_v1.json`。可显式`--reference-pack`；仍须与最终state绑定SHA一致。默认source-cluster bootstrap为2000次、seed20261124。

## 准入与来源

先读取state顶层，必须同时满足：

- `status=COMPLETE_FIXED500_1000_NEW_ANSWERS`，`result_status=COMPLETE_MATCHED1000`；
- 两域完整闭合各500，1000个新终止回答，0运行中记录、0收取/评分错误；
- 所有分片官方评分完整PASS，冻结R与exact500身份不变。

不完整时在读取pack/raw或创建结果目录**之前**拒绝。随后重核实际raw triplet SHA、scored行SHA及精确原始行绑定、完整owner费用、相同图像/问题/模型/prompt/positions、各域同500 ID与原始full-KV分母；所有新旧输出仍是既有推理，新增推理0。

## 数学与范围

按域计算真实总完整视觉owner字节：

`lambda = (sum B_R - sum B_low)/(sum B_high - sum B_low)`

设每张图片在编码前用独立于图片/答案的Bernoulli(lambda)选择旧high，否则选旧low。复用锁定旧输出时，期望平均质量是：

`S_mix = (1-lambda) S_low + lambda S_high`

其期望总视觉owner费用与R相同；它是一条**平均预算**端点线，既不是新跑的一种方法，也不是完整SOTA凸包，更不是逐图片同上限的比较。混合可能使用旧约1.8bit高档；R约1.55bit是均值，逐图最大值另报。这里不匹配text KV、峰值、编码时间、传输或延迟。

每次source bootstrap重采样后重新计算lambda、R和端点质量，再算R−混合；不能先固定lambda再只bootstrap答案差。区间外lambda不裁剪、不外推；若点或某些resample不在凸组合段，区间明确不可用，不悄悄丢样本。

输出JSON、逐源成本/分数CSV与中文表。结果是事后辅助诊断，未调整多重比较，不是新的预注册primary或独立确认，不据此修改当前候选、阈值或样本。

## 已做的CPU检查

`cpu_checks_v1/CHECKS.json`：4项通过，全部为显式合成算术/准入检查，实际GPU调用0、实际任务记录读取0、真实500 state/pack读取0。

1. 解析中点质量与预算相等时差值为0。
2. 预算在端点段外时不裁剪、不外推。
3. 两个来源簇的合成例中，重采样lambda确实在0.25至0.75间变化，差值区间为[-15,+15]pp；不是固定lambda的bootstrap。
4. 不完整fixture state在读取不存在的pack、raw或创建结果目录之前被拒绝。
