# 第二周期执行状态

本文件是执行记录，不改变用户授权或冻结数据的成员。

## 已完成

- chat.md 的最终 GPT 第三轮回复已放在文末，以 gptdone 收口，按用户要求结束讨论后开工。
- configs/a2/isolation.json 冻结 dev8 与新 confirm64。确认集排除旧 dev32/explore16/confirm64；ID 不按新损伤挑选。确认文件在推理前补齐 Linux 图像路径，成员不变。
- audit/a2/assets_identity.json：11 个模型文件和 72 张开发/确认图像全量 SHA256 一致，运行时 torch 2.4.1+cu121、transformers 5.12.0、triton 3.0.0。
- a2_kernel_gpu1_v1：144 个 GPU mixed-representation 检查通过，空/全 mask 与旧端点完全一致。
- a2_grid_dev8_gpu0_v2：96 会话完成，8 文档、37 题、3 caps × 4 codec。v1 因漏传 save_states 启动失败，保留日志，v2 修复后完整重跑。
- a2_select_dev8_gpu1_v1：128 会话完成，16 selector/dose 组合。
- a2_aux_gpu0_v1：D0 diag4、onebit40 会话、C 16 构造会话 ×9臂×2目标=288 自由目标回答，完成。
- a2_refine_dev_gpu1_v1：全局读上限的 A2、448 个同 q 局部校准点、14 个实际等存储配比 ×dev8，完成。

## 当前结果

- 分辨率 Full ANLS：cap4=59.0734、cap8=84.7490、cap16=84.7490；Base2 cap16=87.4517。不能把此开发集的量化正翻转当作 Full 数学上界。
- dev8 Doc 的 Base1=0、V-only=0、K-only=81.3810、K/V refined=80.6306；Full=84.7490。K-only 有额外存储，不能声称同 1-bit。
- 2-bit 页选择当前未胜随机；128-token 页消融与新确认尚待收口。
- 原始确定性 CS 视觉 score top1 证书在 diag4 的采样 q 中通过率0，平均上界12.44、实际改变量均值0.85，仅说明证书松。
- 14 个等存储层配比在有效视觉位宽约1.49/1.73/1.97处大多仍崩溃；局部 MSE knapsack 倾向 V，和 K-only 整体恢复不同。不能把少量层高精度当成已成立的低比特算法。

## 已启动的后续有限包

- a2_global_dev_gpu0_v1：全 K 工作点逐层删除的固定 token KL 校准；校准规则和128页对照；base1四臂机制重测。其代码/配置在 job_sources 独立快照。
- a2_rates_dev_gpu1_v1：等前述 dev 包结束，39 个真实 mixed K/V bits 与 K-residual1 GPU检查，再跑真实 K2/V1、K1/V2、K1+r1/V1。新核 fused_rates.py 与原 fused_masked.py 分开，旧结果不改。
- a2_calib_post_dev_gpu1_v1：等 rates 结束，先与官方 packed QK/WV 算子比对，再跑5个 Calib post-scale/clip/tau 开发配置。用于排查直接 dequant 数值端口与 post-scale 运算次序及超参迁移差异，避免只比较弱配置。

## 尚须完成才收口

1. 检查上述工作包完成/失败回执，必要时修复并独立新名重跑。
2. 对 C 的 fact 目标使用明确短答案指令复核，原版结果保留；不把过预算的 Native4 升级当作同预算收益。
3. 冻结一个 A2 候选与低比特候选/强对照、参数与代码哈希。此时才能对新 confirm64 推理。确认还没有启动。
4. 双卡各32文档确认，报告所有新错、原始自由轨迹、文档聚类配对 CI、多重比较、同真实字节的资源账。
5. 选定方法做独立固定 token 同步耗时测量；逻辑读账和物理 DRAM 分开。选择开销为预声明 metadata/query/scratch 费用模型，不称实测 HBM。
6. 统一中文报告、Python科学图、可复跑配置与独立清单；核验第一周期冻结文件未修改。整个有限包完成后再向用户最终汇报。

## 操作

`python code/a2/manage.py status RUN...` 查看权威 PID/complete；组合队列的子目录同步必须 `python code/sync_results.py RUN --sources`，不加 --sources 只取根文件。

分析：code/a2/analyze.py 仅接自由生成 variant；code/a2/analyze_c.py 用 C 的独立目标参考；旧分析/旧原始结果都不改。新确认候选名称、主比较与正式图待上述开发结果后确定。

## 最新推进（取代上面的“尚须”状态描述，仍未跑新64确认）

用户已显式要求设置目标，已创建 active goal（无token预算）；目标细化见 GOAL_REFINEMENT_2026-11-13.md：含元数据/辅助的平均视觉KV≤1.8bit，争取距强质量参考≤0.5pp；读取在固定存储/质量下减少≥50%额外精化费用；+2pp为同低预算点的争取项。

### 新协议与已完成工作

- session_boundary.py 已实现视觉边界压缩：原生因果视觉prefix提取后删除其后的native问题状态，压缩后重读整个问题suffix，首答案token也经压缩读取。原生提取/重读工作计时；此协议不与旧共享首token混合。
- 官方 InternVL eval 的固定短答提示 `Answer the question using a single word or phrase.` 已统一加到所有方法/每轮；来源见 audit/a2/short_prompt_precommit.json。dev8_short.json、confirm64_short.json 均已落地，确认成员不变，确认推理仍未启动。
- 新64的Windows→Linux图像路径、content输入结构、确认split标签已在推理前补齐；configs/a2/isolation.json 更新hash，图像成员不变。assets_identity验证过11模型文件与72图像SHA。
- 20读数题/9图，完全FP16另一轴：Full93.0357、K1=0、K2=94.2857、V1-token90.2363、V1-channel91.9562、V2-token93.0357 ANLS。a2_value_probe20_gpu1_v1 共120回答完成。额外clipped V1控制在factorial队列完成。
- a2_rates_dev_gpu1_v1：39个mixed真实位宽/1bit K residual检查+48会话完成。Kr1不是原Kr2重命名。
- Calib post-scale 与官方packed QK/WV算子两形状误差均0；a2_calib_post_dev_gpu1_v1 五配置40会话完成。Calib为数值移植，materialized/packed均计，未当官方serving。
- C补短答目标后a2_c_recheck_gpu0_v2：288回答完成。keep Base2事实96.875、私有绑定100；drop_fact事实83.333、绑定100，平均owner16.717→14.206MB；drop_secret绑定0。Native4+dropfact恢复100但owner22.004MB，超预算，不能算同预算收益。分析a2_analysis_c_v2。
- a2_hybrid_dev_gpu0_v1 数值组合48会话完成；KVarN2 K + clipped-channel V1有线索。已写fused_hybrid.py真正packed reader（KVarN2和native K2两条K路径，V1 clipped-channel）；对应独立GPU检查通过。
- 真实模型same-q比较 a2_model_kernel_gpu1_v1：28层，原frame materialized vs rotated packed最大abs0.0557、最大relativeL2 .00312；synthetic .0005量级。裸问题时很多ANLS损失来自完整解释句，未改评分，后续统一标准短答提示。
- a2_factorial_dev_gpu0_v1 首次因两个load_official名称相撞启动失败；修复别名后a2_factorial_dev_gpu0_v2完成32会话+20 clipped-V诊断，保留失败日志。没有因此篡改数值结果。
- a2_boundary_dev4_s0/s1_v1（裸问题、split）各88会话完成；a2_short_dev4_s0/s1_v1（标准短答、split）各104会话完成。
- 标准短答dev8：Full8=85.1995、Full16=84.7490、Base2-16=82.0463、KVarN2-16=86.5508；nativeK2V1-16=71.4369；native2+clippedV1-16=81.1454、约1.63实际bit；同组合cap8=78.8288。V1码率与质量均按真实存储，尚未达到0.5pp主目标。
- A2页128最后q metadata选择：none82.0463、random79.3436、mass/risk81.3707、thumbnail-priority82.0463、all85.1351。Periodic独立预算控制已追加。
- GotOcr2ImageProcessor源码已核实thumbnail追加在最后，audit/a2/processor_thumbnail.json。A2全base始终含thumbnail；thumb_random是预算内优先精化的对照。
- 压缩VP索引现在所有层共用不可变backing；owned分列visual_position，所有实际bpp包括该项。位置读费用单列position_bytes。
- a2_vfloor_dev_gpu0_v1：K2/V1 +2层V-sign residual/小范围K clipping，56会话完成；真实bpp两层约1.7924。cap16未优于81.1454组合。
- a2_vfloor8_dev_gpu1_v1：相同7配置cap8，56会话完成；最佳vlate2_kclip08_cap8约80.7745，未超81.1454。

### 正在收尾的最后开发包

- a2_final_repr_dev_gpu0_v1（PID550271）：11配置×dev8=88会话，最后状态76/88，近期应完成。包括V双质心（Lloyd3）、W_o列范数加权双质心、近似folded-sign元数据、K1/V1双质心、native V2前/后2层，以及K降2层/V升4层的等字节交换。safeK=[5,7]来自原global K3→K1删除KL的最低敏感层，仅为surrogate。
- 新quantizers.py只改编码，仍1bit索引+2个FP16仿射重构参数，原fused_rates解码。check_quantizers.py CPU signed-word/constant-group/加权目标检查PASS，不能当模型质量证据。
- a2_periodic_dev_gpu1_v1（PID551702）8会话COMPLETE。Periodic语义：问题prefill完整精化，decode每4步精化，统一全会话extra cap；此为强时间策略对照。
- 新CALFA-KV arXiv2611.10511已核对（2026-11-07，有公开代码）：联合码本/中心/径向参数，离线CE平方梯度Fisher，在线FP16 anchors。我们当前W_o权重不是Fisher，不冒称其复现。

### 接下来必须做

1. 同步最后88与Periodic；分析完整dev。不要再增开算法候选。
2. freeze_confirmation.py尚未执行。需要把最后11新encoder/mixed-layer配置纳入eligible；目前helper已覆盖v_layers/k_clip/layer_bits列表，尚需k_encoder/v_encoder。确认参照含同cap FP16/Native4/KVarN2；主对照含同cap nativeK2V1及A2 random/mass，补最强同码率centroid对照。
3. 冻结后上传启动a2_confirm64_s0/s1_v1（每卡32图）。新64结果未看、未跑；严禁确认后改参数再冒称盲确认。所有正式工作点保留。
4. 在正式确认期间完成统计、图、审计和报告代码；之后跑少量固定token同步耗时，codec编码开销已加入encoding_wall_ms。物理HBM旧ncu2020失败Error36，当前未取得有效counter，不能称逻辑字节为实测HBM。
5. 完成新64、图像聚类CI/primary Holm、条件分层、失败案例、字节与源码审计，核验第一周期冻结文件未改，写统一中文报告与可复跑产物，再完成goal。

### 运行辅助

当前本地分析脚本默认UTF8读取非ASCII数据要显式encoding，避免Windows GBK错误；远端Linux默认UTF8。组合队列同步用--sources才会取子目录。

freeze_confirmation.py会拒绝覆盖既有audit/a2/frozen_confirmation.json，便于守住确认边界。checkpoint和源代码分别在audit/job_sources/与远端jobs/*.source，不覆写历史运行。
