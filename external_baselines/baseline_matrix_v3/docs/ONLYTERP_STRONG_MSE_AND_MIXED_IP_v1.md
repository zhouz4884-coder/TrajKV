# OnlyTerp：较高位 MSE 与单次 Gaussian mixed-IP 源变体

Hmix12 的64题负结果只描述该低位 MSE/QJL0 变体，不能当作 TurboQuant 论文主表或所有 IP 路线失败。本轮准备两个明确区分的参考，不修改已有 MSE、0xSero、evaluate、bridge 或 controller，也没有启动 GPU。

## 已有 Hmix23，不重新生成冲突配置

`configs/matrix/OnlyTerp_MSE_Hmix23_readiness_v1.json` 核对了原先五份配置与 job 的精确 SHA。直接使用现有 `M1_OT_Hmix23_*`：先 Text2，再 Text4+Info4，最后 Text32+Info32。三阶段的集合相互包含；获得真实 same-contract source 后应复用已有行，只跑 missing，而不是把五个完整 draft 合计74条当作74个独立问题。

该点是源 MSE2/3、QJL0、32 outlier/96 regular、Hadamard。regular96补到128，故 payload 为 `(128×2+32×3)/128=2.75`。两个源 FP16 group norms 加0.25，得到3.0 core bits，尚未加所有 map/sign/codebook/boundary/CPU-GPU常量/位置费用。只能以实际 GPU owner 报完整位数，不能填“论文2.5”或把3.0当完整字节。已有数学 CUDA门覆盖2/3；本轮核对没有把它扩展成整模型成功。

## 正确 public API 能实现单次 IP

固定 [OnlyTerp primary `src/cache.py`](https://github.com/OnlyTerp/turboquant/blob/1891cdd2ee60514dbd6dd0172ea3dcf00a3b0306/src/cache.py) 的 `turboquant_encode_internal`、`qjl_encode` 和 `turboquant_decode_single` 提供了正确的 Gaussian 残差编码/解码。主 Cache 的默认 `compute_attention(qjl_score_weight=1)` 是 K=MSE+一次QJL、V=MSE，尽管 `store_batch` 将 V 的 QJL 符号和残差范数也存下来。

新模块 `methods.turboquant_onlyterp_mixed_prod_adapter.OnlyTerpMixedProdAdapter` 保持这一实际源默认：

- K：旧 MSE 源 alias 的同一分组、旋转、码本、FP16 group norms，再按源 API 计算原空间残差，用同一 seed规则生成的全128维 Gaussian S编码一次QJL。
- V：同样编码并**保留/计费**源 V-QJL；默认读取 `source_cache_mse`。`full_prod_decode` 是另一个明确标注的 public-vector-decode 变体，不偷换为高层 Cache 默认。
- QJL signs 的原 int64 0/1 值仅作无损 bitpack，读取恢复原值；QJL residual norm保持源 FP32，不另降为FP16。投影矩阵保持 FP32。
- 所有实际持久 Gaussian 矩阵均在 `named_tensors` 中计费。CPU Gaussian生成只是临时 scratch，精确上传后不保留隐藏 CPU副本；旧 MSE组件实际持有的CPU/GPU码本、边界、符号/映射仍完整列出并按真实storage去重。没有将投影矩阵当免费常数。

新的默认宽度是源高层 Cache 的 MSE regular3/outlier4：Hadamard padding使 MSE payload=4，再加每KV标量1个QJL sign、源group/residual norms合计0.5，**core已是5.5 bits**，还未计S与其它常量。实现也覆盖2/3，但没有假设这些完整点低于2.5或3.5。

本模块是“OnlyTerp primary mixed-MSE + 一个全128维 Gaussian QJL”的视觉源移植，不是把论文的两个完整独立 prod 实例参数补齐。论文 §4.3 的 `32×3+96×2` 实际除128是2.25，且3.5点完整比例未给全；没有据此伪造“论文2.5/3.5复现”标签。

## Reader 与模型 probe

reader明确命名 `torch_source_ip_fp32_reference`。它临时解码MSE FP32 K/V，计算源分开的 MSE score 与**恰好一次** `S q` 残差 score，再同精确文本合并一次因果softmax。full-prod V模式调用对应源vector decode。实现未使用已证双加QJL的buffer manager，也不使用不兼容的Rademacher experimental kernel。

这是数学参考，不是论文优化packed kernel。临时dense sign、FP32 KV、bit展开、矩阵读取/投影以及当前参考实现的重复materialize都会进入真实峰值/时延；不得把该速度当论文优化内核速度。

CPU结果：`results/p0/onlyterp_mixed_prod_cpu_v1.json`，五组覆盖2/3、3/4、H1/H2与零值。K/V源PQ重构、QJL signs/FP32 norms、Gaussian矩阵和两种完整vector decode逐值一致；默认读法对 literal primary Cache单次QJL输出、GQA/原位置因果FP64参考、无原KV alias、无持久decode、V-QJL保留和完整owner计费均通过。它不是GPU/任务质量结果。

供Root安排的草案：

```text
configs/jobs/bm_onlyterp_mixed_prod_cuda_gate_v1.json
configs/jobs/bm_M1_onlyterp_mixed_prod_Hmix34_probe2_v1.json
```

第二个 job 是独立 native-protocol 诊断 driver，不改 evaluate 的 ALLOWLIST。每个分支完整重算问题suffix并独立生成首token；native suffix检查和生成也使用不同prefix clone，不把参考question KV或首token送给压缩臂。默认2QA同时列 source-cache V-MSE与public V-prod两种读法。保留native原型用于控制，明确不是进程内存/速度benchmark，也不是可直接进入正式 prediction cache 的生产者。后续只有GPU数学门与真实模型结果能够决定正式admission与是否扩样。
