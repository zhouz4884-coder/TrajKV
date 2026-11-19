# 真实 backend 的连续诊断与 quiet timing 薄入口

新增 `round8_backend_continuous.py`、`round8_backend_timing.py` 与共用 `round8_backend_support.py`，不改任何原连续诊断、timer、bank、codec、reader 或既有结果。本轮仅本地 CPU 准备/检查，没有 SSH、GPU、模型推理或任务调度。

配置必须显式声明 `backend_module` 与 `backend_family`，允许组合只有 round8_candidate_runner 的 spherical/field、round8_postres_runner 的 postres、round8_final_ray_runner 的 final_ray。先执行对应 family 的逐 variant/common-input/price 校验；不支持的 CD、V refit、postres 或 final-ray 标记会提前拒绝，避免默认 Round8Bank 忽略新参数。当前 wrapper 固定所有方法 cap8。

实际运行先 import 原 round8_runner，使 R8/R7/R6 注册链缓存，再进入既有 backend_registration context。原 main 内部再次 import 是 cached import，其随后 `from factory import bank_class` 得到选定 subclass。入口断言 factory/executor 返回该 class，记录真实 module、qualname、source SHA 与 MRO。finally 恢复 factory/executor、包括内部 Full teacher/canonical class 在内的 settings，以及 session_boundary.ProgressiveBank。

Continuous CLI 保留 `--config --output [--references] [--plan-only]`，原 round8_continuous.main 的完整 Full-teacher prefix、独立 QA、可选 canonical-GT 独立 pass、EOS/closure 范围、物理模型 forward 计数、source-macro 逻辑不变。它是连续误差诊断，不计算 task accuracy，也不是 E2E timing。旧 records 并不承诺包含逐 pass 的新 refit stats；新 sidecar 把 name→variant 声明、原 config SHA 和真实 class 固定下来。

Timing CLI 保留 `--config --output [--plan-only]`，直接执行原 round7_timing.main。只在 context 内以新 set 临时把 round8_joint 加入 SUPPORTED，finally 恢复原 set 对象。仍是5/6方法、3source、warm1/measure3、每 turn 固定8 IDs、一次模型加载和原 NVIDIA quiet 边界检查。没有 profile 的 WHT/PQ/allocator/CUDA-event 函数仪器，也没有改 timer 的数值或记录。新入口要求显式 frozen_variant_contract，hash 与每个 variant 字典必须逐项匹配真实冻结质量配置；本轮没有猜 winner 或生成实际测量 job。未来6臂可由 root 从冻结质量配置复制 Full/native4/base2/oldK1/oldK2/newwinner。

两入口在旧 main 前只把 before 快照留在内存，不创建 output，保留原 `mkdir(exist_ok=False)`。快照补齐 backend/helper 及原 baseline source、config、samples、model config（不重哈希权重）、selected image bytes、legacy Python source、KIVI source、所有声明的外部书/reference/factor/policy 和可选 contract/reference 文件。旧 main 成功后再做 after 核验，并核原 manifest 实际消费的 config/samples/model_config/optional-reference SHA 与 before 一致。wrapper 的预检/哈希在原 timing trials 外；实际新算法编码仍走原 run_session 计时路径。

不改旧 manifest、complete、receipt、records、summary、schedule 或控制文件的一个字节。新增 backend_diagnostic_sidecar.json 或 backend_timing_sidecar.json，包含 before/after、原全部 artifact SHA、class/source/配置声明与明确范围。最后分别写 `complete_backend_diagnostic.json` / `complete_backend_timing.json`，绑定原 complete/manifest、sidecar、wrapper/support 和实际消费输入 SHA；下游只能在相应新 marker 后接纳。timing 的 producer 是 timing_records.jsonl，diagnostic 是 records.jsonl 与 passes.jsonl，两者均不是新自由生成质量回答。

CPU v3 证据 `outputs/round8_backend_wrappers_cpu_v3/check.json`：四 family×两个原入口共8个真实原 plan（continuous refs 透传与原 pass 计划；timing72trial 协议）、8个实际 late-factory-binding 选定 class stub、8个原 main 异常恢复、6个配置负控及旧 timer 漏列的 reference book 前后变化负控通过。原文件 SHA 和旧产物字节保持，SUPPORTED 恢复原对象。stub mains 不进行模型 forward，不是实际连续诊断或 quiet 测量证据。计划模式也不执行 runtime registration；不可把其成功称为 GPU 或模型验收。
