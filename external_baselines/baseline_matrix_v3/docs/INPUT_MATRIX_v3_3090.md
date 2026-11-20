# TrajKV: RTX 3090 可执行 SOTA 复现矩阵 v3

日期：2026-11-19。根据用户《基线矩阵设计 v2》、TrajKV main `5e9f4f19051b0fc2db5b1c55b19776d690b322d9`、模型原始配置与已检索的公开源码制定。**这是执行设计，不代表新模型及外部方法已在当前 3090 上跑通。**

## 一、论文的主要比较协议

- 目标：固定模型及输入视觉密度，比较 **视觉缓存真实驻留字节—公开任务质量** 的 Pareto 前沿；另报编码、prefill、decode、总请求时间与实际 GPU 内存。
- 主协议：与现有 TrajKV 对齐，建立图像视觉前缀后应用 KV 压缩，再以压缩视觉 KV 重算全部问题 suffix；所有对照都需要从相同问题状态形成阶段开始受压缩影响。标准评测主表单问、自由生成；多轮在独立表中测试，不混入单轮平均。
- 单模型固定同一 checkpoint revision、权重 dtype（FP16；如原生 BF16 需所有方法一致且说明）、processor 版本、图像原始 bytes、cap/max_pixels、完整 chat prompt、输出上限、贪心解码、EOS、视觉 token 位置及 text KV 处理。模型间允许各自**原生** processor、提示模板和图像分辨率，但模型内的方法间必须相同。
- 压缩范围：**视觉 KV**，文本 KV 始终为相同精度；对无法忠实视觉化的原方法，另标原始完整系统运行，不能冒充同轴复现。FP16/视觉KIVI-style/每个方法必须有真实 owner 字节。
- 匹配 1.3/1.8/2.25 等容量档的实际 bytes，而不是名义 bit；各方法同时报原论文默认配置和最接近容量的合法配置，只有**预算不超出**的点可称 iso-byte；离散预算未精确命中则给前沿/区间，不插值产生虚构实测值。

## 二、三个主要模型及硬件门禁

| ID | 主模型及原始权重链接 | KV几何与意义 | RTX3090硬门禁 | 定位 |
|---|---|---|---|---|
| M1 | [OpenGVLab/InternVL3-8B-hf](https://huggingface.co/OpenGVLab/InternVL3-8B-hf) | 28L×4KVH×128D，现有真实packed主模型 | **已有单卡FP16全链证据**；沿用冻结环境 | 主结果与全部外部方法 |
| M2 | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 28L×4KVH×128D；不同视觉前端及 mRoPE，现代强 VLM | FP16/BF16、低max_pixels、单卡实际prefill + 第一个自由回答及峰值<24GB；否则限制分辨率或记录不可行 | 视觉前端和位置编码迁移 |
| M3 | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 36L×8KVH×128D、DeepStack、多模态 mRoPE；与M1/M2语言骨干不同 | 原生配置先单卡FP16/BF16加载 + 图像prefill、decode/完整cache检查，不能因4B推定成功 | 跨语言骨干和更多head；必须先过兼容门 |
| 备选 | [LLaVA-v1.6-vicuna-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf) | 32L×32KVH×128D，MHA；TrajKV已在3090真实跑通 | 可直接做技术压力对照，但现有低bit质量掉得较多 | M3不兼容时顶替或附录结构压力实验 |

**执行前先运行原生模型单卡图像+问答 smoke（10例不同分辨率），并记录 weights resident、peak alloc、KV，检查视觉 token 位置和复杂 mRoPE/DeepStack 是否能在真实 packed reader 中等价保真。** OOM、未处理的局部注意力、因不兼容必须丢掉额外视觉层，均记为明确状态，不得用权重量化改变研究自变量。更大的 Qwen3-VL-8B 可在两卡、相同权重精度和实际可运行的条件下作为后续扩展；先不用它阻塞主矩阵。

## 三、公开数据集

| ID | 数据集和官方/主流入口 | 最终主协议 | 官方/明确评分 | 适合的作用 |
|---|---|---|---|---|
| D1 | [TextVQA](https://huggingface.co/datasets/lmms-lab-encoder/textvqa) | validation约5000题，先每模型128源smoke，再500/1000分层pilot | **官方十票 soft VQA accuracy**，不是ANLS | 场景文字、细粒度事实，当前主要短板 |
| D2 | [InfoVQA/InfographicVQA](https://huggingface.co/datasets/lmms-lab-encoder/DocVQA) | validation2801题，先128源smoke，再500/1000 pilot | ANLS（严格一致实现） | 密集文本、版面、图文混合 |
| D3 | [DocVQA](https://huggingface.co/datasets/lmms-lab-encoder/DocVQA) | **官方validation5349题**，不要用已高接近天花板的train子集替代 | ANLS + normalized exact | 文档保真、传统核心数据集 |
| D4 | [ChartQA](https://huggingface.co/datasets/ahmed-masry/ChartQA) | test2500题，报告human/aug两组，不因差异而只取高分组 | relaxed accuracy，数字容忍及 yes/no 按官方规则 | 图表读数与数值逻辑；跨域检验 |
| D5 | OCRBench（论文补强） | 仅M1以及通过兼容门的M2，公开约1000题 | 官方分类及总体得分 | 解释1-bit是否伤害OCR；不强制纳入首轮 |
| 后续 | [MMLongBench-Doc](https://github.com/mayubo2333/MMLongBench-Doc) | 仅在多页/长上下文真实适配后，先2/4/8/16页预定pilot | 其官方评价可能需要闭源模型辅助，须分开“可复现离线代理”和原版官方协议 | 长文档容量价值，不作为本轮默认主图 |

**不得把任意项目子集的 Doc train 98–99 ANLS 称为官方 DocVQA 标准成绩。** 模型间实际视觉token数不同，单模型内同一问题下所有方法须完全匹配。

## 四、真正开源且能够开始审计的外部方法

| ID | 方法及源码 | 出处/性质 | 当前可用性与端口边界 | 主矩阵建议 |
|---|---|---|---|---|
| F | FP16视觉KV | 原生代码 | 现有模型全支持；压缩问答协议同一比较 | **必选** |
| K | [KIVI](https://github.com/jy-yuan/KIVI)，其中用户现有`base2`为KIVI-style视觉轴 | ICML2024；官方仓库 | 当前`base2`为真实packed但无原方法全精度近窗；标为`KIVI-style visual port`，可再加全官方策略作为不同实验 | **必选**，容量基准；保留准确版本标签 |
| C | [CalibQuant](https://github.com/insuhan/calibquant) | 视觉KV 1-bit，官方代码 | 原README目标InternVL2.5，往InternVL3/Qwen迁移需要调整Hooks、视觉索引和模型校准，需先数值/预算验收 | **M1优先**，M2有条件；不可强制M3 |
| T | [TurboQuant](https://github.com/0xSero/turboquant) 或 [OnlyTerp参考](https://github.com/OnlyTerp/turboquant) | ICLR2026；**第三方公开实现**，须明确不是Google官方已验证VLM | 核对3090上的实际pack、随机旋转/QJL和 reader；默认合法2.5/3.5或附近配置，统计实际encoder元数据 | **必选一个实现**，以可重复执行为准 |
| E | [PyramidKV](https://github.com/Linking-ai/PyramidKV) + [NVIDIA kvpress](https://github.com/NVIDIA/kvpress) 的可核查选择器接口；或 [VisCache/PruneKV](https://github.com/Wlklk/VisCache) 单图子部件 | token选择/驱逐邻轴 | 原PyramidKV与kvpress偏LLM，VisCache偏视频；在同一视觉前缀协议适配须标`faithful visual adapter`，保留其允许的query/窗口语义 | **选择其中一种可忠实移植者**，其他附录 |
| V（P2） | [KVTC第三方实现](https://github.com/OnlyTerp/kvtc) | ICLR2026论文，**非NVIDIA官方源码**；PCA/变宽+熵编码 | CUDA5090测量、ANS解码开销较高，3090真实部署与prefill仅在通过门禁后；K4V6名义均值5bit不是2.5bit | 主要做变宽对照，独立P2，不能阻塞主矩阵 |
| V（P2） | [CommVQ官方实现](https://github.com/UMass-Embodied-AGI/CommVQ) | ICML2025；加性VQ/可交换RoPE | 公开完整代码但目前训练与融合内核主要为LLaMA-3.1-8B，需要新VLM码本重训、mRoPE适配 | 强学术对手，成功移植后进主表，否则保留P2 |
| 不强制 | GSRQ、HybridKV、TriAttention、EpiCache | 相关论文或特定服务方法 | GSRQ/HybridKV暂未找到可核实的通用VLM直接复现路径；TriAttention现成TensorRT-LLM版本需要SM100且保留prompt而清理decode | 引用，不能以非忠实替代冒称完整复现 |

## 五、可执行核心矩阵（逐格统一记录）

方法缩写 F=FP16; K=KIVI-style视觉轴; C=CalibQuant正式移植；T=TurboQuant第三方忠实移植；E=选中的一个视觉token驱逐方法；O=我们的冻结R7@约1.3bit与R5/R7@约1.8bit（**以后单独跑，不计入先行外部任务**）。`C?`表示只有真正通过M3适配协议时才运行，不能用替代算法偷偷填格。

| 模型 | TextVQA val (D1) | InfoVQA val (D2) | DocVQA val (D3) | ChartQA test (D4) | 备注 |
|---|---|---|---|---|---|
| M1 InternVL3-8B | F K C T E → O | F K C T E → O | F K C T E → O | F K C T E → O | 所有基线主锚；4个数据集 |
| M2 Qwen2.5-VL-7B | F K C? T E → O | F K C? T E → O | F K C? T E → O | F K C? T E → O | CalibQuant移植通过才有C |
| M3 Qwen3-VL-4B | F K T E (C?) → O | F K T E (C?) → O | F K T E (C?) → O | F K T E (C?) → O | DeepStack与mRoPE先过严格预算/等价门；否则由LLaVA替代 |

外部必需基线图约为3×4×4～5=48–60格（包含FP16），每种方法可能另需第二预算点；先做pilot再做必要大规模确认。**不要同时要求114格×全部数据集×所有码率×四模型。**

### 分阶段规模

- **P0（硬门禁，不打主表分）**：3模型×10张不同尺寸单图，原生FP16、真实prefill/first-answer；各方法2–8图量化往返、位置/tail/多层reader算术、owner核查；不支持列真实状态并跳过，非自动替代。
- **P1（可比性）**：M1×D1/D2×F K C T E（先128或256独立图）；M2/M3×D1×F K T E（每格至少64/128真实图）；所有方法同源重算，建立代码/预算一致性。保持模型官方原生输出与“无压缩桥”的一致性。
- **P2（主体预赛）**：M1×D1–D4×F K C T E，M2/M3×D1–D4×F K T E，各格固定500个来源/问题（不足全量则全用），采每源成对差及其方差；决定哪些方法应全面扩充。
- **P3（论文正式结果）**：优先完成 M1 的 D1/D2/D3/D4 全官方split，M2/M3至少 D1/D2/D3 的原定足量子集且采用相同方法集合；能承担时补全部split，所有主结果明确称full或subset，不报不同split的横向差。OCRBench以M1优先，MMLongBench仅在多页预检后增设。
- **P4（我们的方法）**：基线库冻结后，分别在原定模型/数据/输入协议上运行R7和新算法，用缓存的原始预测做同源配对比较。做真实速度判断时，在相同环境中交错重跑性能关键基线。

### 两种不可混淆的公平比较

A. **同一完整视觉输入、同一推理范围的视觉KV存储约束**：量化与视觉缓存驱逐（后prefill）的真实保留字节，统计回答准确率；与完全不压缩的同N FP16比较，并给不同预算实际点及前沿。

B. **固定总资源下的输入/执行能力**：允许不同crop/grid/保留视觉信息，必须同时计vision encoder成本、完整原生prefill、真实GPU峰值、缓存占用和历史状态，不能拿B的分数冒充A的单变量优势。

Eviction若需要当前query的attention来选择token，与我们“未来问题未知时编码图像”属于不同信息条件：单列`query-known`和`query-unknown`（只用可获得的历史/视觉前缀统计），不可让查询已知的方法偷偷取得独特信息却称严格同协议。

## 六、统一记录格式及重用要求

每条记录至少包含：`model_id`,`model_commit`,`dtype`,`model_arch`,`dataset`,`dataset_revision`,`split`,`source_id`,`question_id`,`original_image_sha256`,`processor_id`,`processor_version`,`cap_or_max_pixels`,`actual_visual_token_count`,`chat_template_hash`,`full_prompt_hash`,`question_suffix_sha`,`compression_boundary`,`method`,`method_paper`,`upstream_repo`,`upstream_commit`,`port_name`,`faithfulness_status`,`bits_nominal`,`full_kv_byte_denominator`,`stored_visual_KV_bytes`,`coder_metadata_bytes`,`positions_bytes`,`encoder_constants_bytes`,`lookup_bytes`,`actual_effective_visual_bits`,`text_kv_bytes`,`peak_allocated_bytes`,`input_tokens`,`output_token_ids`,`raw_prediction`,`official_score`,`source_cluster_id`,`wall_ms`,`encode_ms`,`prefill_ms`,`ttft_ms`,`inter_token_ms`,`hardware`,`cuda_version`,`torch_version`,`hashes`,`status`。

- `status`: `PASS` / `UNSUPPORTED_MODEL` / `HARDWARE_UNSUPPORTED` / `OOM_AT_DECLARED_INPUT` / `METHOD_PORT_UNVERIFIED` / `INPUT_MISMATCH` / `OWNER_AUDIT_FAILED` / `FAILED_RUN`. 绝不填造的分数或虚构的“0”。
- 固定任务评分：TextVQA十票soft accuracy；InfoVQA/DocVQA ANLS；ChartQA relaxed accuracy（human/aug分报）。用同原图/问题ID配对，文档同源多问按源聚类bootstrap+sign-flip；报告差值、95%CI、清晰的比较家族，不以 CI 跨零为等价证据。MDE先根据真实pilot估计，不照抄过小子集推断的0.05pp保证。
- 优先报告 `quality vs actual visual bytes` 与同N `FP16` loss；`quality@isoB` 用实测容量不超过目标的点，而非将名义bit或压缩率强行兑换成结果。未覆盖同一容量范围的方法不外推。
- **复用规则**：保存源图/图像哈希、Q/A原始模型答案、token IDs、score、完整费用、方法+环境manifest及输出不可变哈希。后续仅当所有比较合同一致，允许新算法只跑一次并与历史基线离线逐项配对；时间和吞吐涉及GPU波动，需要同日交错重跑关键基线；任何模型、cap、processor、prompt、dtype、边界变化均须使旧基线进入 `STALE_PROTOCOL` 状态。
- 保留低成本 KIVI-style、uniform4 RTN 与FP16 的至少一个已冻结版本作为可信技术锚点。不能为了省时间删掉最容易复核的对照。

## 七、交给 Codex 的主指令

> 以本文件为主执行合同，在原 TrajKV 仓库旁创建独立、可恢复的 `baseline_matrix_v3` 项目入口，严禁覆盖 Round1–8 的源码、冻结配置及证据。优先完成 P0→P1；成功后沿 P2/P3 的既定矩阵执行，不能在单个方案失败时静默改成它的“简化版”或用 CPU 模拟/GPU假压缩替换真实结果。
>
> 实现方法注册、模型前缀捕获适配、统一真实字节 owner、官方评分、分层数据拉取和三阶段 pilot/full 驱动。对于每个第三方方法，下载并锁定源码SHA、给出源论文/官方代码/第三方 port 身份，隔离旧依赖环境；在 3090 上先做真实CUDA算术、缓存存储及视觉问题后缀重放门禁。不得引入权重量化，更不能默认让更多GPU替换3090门禁。全周期固定输入与算法选择，失败留原始诊断，不编造缺失数据。每日任务只由可验证的硬件和协议通过情况解锁；若发生不兼容，输出缺口矩阵和已完成有效实验，不要伪造全部成功。外部基线冻结后输出可供新算法追加的不可变 benchmark manifest/score cache，确认数据原文不用于训练或选方法。实际数据集正式全量在适当的pilot和显存诊断通过后再启动。

## 八、核查来源（2026-11-19）

- NVIDIA RTX3090 为SM86/24GB，没有原生FP8 Tensor Core（CUDA Compute Capabilities Table 36）。
- Qwen2.5-VL-7B config：https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json
- Qwen3-VL-4B config：https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json
- KIVI：https://github.com/jy-yuan/KIVI ; CalibQuant：https://github.com/insuhan/calibquant
- TurboQuant（社区）：https://github.com/0xSero/turboquant ; KVTC（社区）：https://github.com/OnlyTerp/kvtc
- CommVQ（官方，原Llama）：https://github.com/UMass-Embodied-AGI/CommVQ
- PyramidKV（官方）：https://github.com/Linking-ai/PyramidKV ; NVIDIA kvpress：https://github.com/NVIDIA/kvpress
- VisCache：https://github.com/Wlklk/VisCache ; TriAttention TensorRT-LLM：https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/kv_cache_compression/triattention.md
- ChartQA：https://github.com/vis-nlp/ChartQA ; DocVQA/InfoVQA：https://huggingface.co/datasets/lmms-lab-encoder/DocVQA
- MMLongBench-Doc：https://github.com/mayubo2333/MMLongBench-Doc （原评测流程涉及外部模型辅助）。
