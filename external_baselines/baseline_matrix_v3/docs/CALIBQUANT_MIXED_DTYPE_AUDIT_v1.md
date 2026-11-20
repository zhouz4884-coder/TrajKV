# CalibQuant源码的混合dtype语义

独立执行原作者AST的CPU类型追踪确认：共同FP16模型端口与源码默认BF16模型，经过硬FP16 kernel返回值后，确有不同的中间精度。这个判断来自源表达式与`torch.promote_types`，没有利用试评分数选择解释。

|源阶段|默认BF16模型与metadata|共同FP16模型与metadata|
|---|---|---|
|q×K scale、K bias产品/输出|BF16|FP16（旧端口SUM/DIV单独扩FP32）|
|原QK kernel输出及其除sqrt(D)|FP16|FP16|
|QK结果减bias|FP32（Half与BF16提升）|FP16（旧端口cast回Half）|
|视觉校准high/min/−3 map|FP32|FP16|
|softmax之后概率|显式cast BF16|公共bridge cast FP16|
|原WV kernel输出|FP16（输入概率显式Half）|FP16|
|WV结果乘V scale|FP32|FP16|
|V-bias子产品|BF16|FP16|
|视觉结果减bias、合并text结果|FP32|FP16|
|源函数最终cast|硬BF16|公共bridge保持FP16模型接口|

因此作者默认并非“全部仿射都FP32”：V-bias子项仍在BF16。`infer.py`的模型dtype为BF16，但示例`quant_method`字符串为qjl；`patch_model`默认calibquant，且Calib decode硬转BF16，表明该分支面向这一模型精度编写。不能把当前qjl示例本身当作已执行CalibQuant的证明。

旧FP32-bias端口保留了共同Half模型的操作边界。之前对冻结case的23项前向误差闭环及74.53125% Text128结果仍然有效；它们验证的是那个明确端口，并不证明与作者默认BF16混合语义位级一致。这不是改写旧实验为bug。

值得增加真实端到端两QA精度对照。独立原型`methods/calibquant_fp32_affine_adapter.py`保持原量化器、clip、packed bytes、metadata FP16、原QK/WV kernel和−3 map，临时bias/仿射/校准采用FP32，最后恢复公共Half接口；q×K scale送入原kernel前仍保持Half。它比作者自动promotion位置更统一，必须标为显式FP32 affine数值端口，而非作者BF16位级复现，也不作为本方算法创新。

该最小原型为适配现有bridge，会把视觉weighted结果先cast Half再与text项相加；原作者是在joint输出后cast BF16，这一剩余边界差异也明确保留。没有改旧state、bridge或evaluate。CPU四种bits/N组合的编码和owner完全相同，边界条件化FP64 coded-math检查通过；零query校准在FP32下按原有1e-8自然有限，无clamp。CPU检查不代表目标CUDA或任务质量已通过。

证据：`audit/calibquant_source_dtype_trace_v1.json`保存源文件SHA、逐句dtype与typed-kernel-stub范围；`audit/calibquant_fp32_affine_cpu_gate_v1.json`保存格式和算术检查。后续GPU诊断只比较原生F、旧bias端口和新affine端口的各自完整suffix及自由生成，不读取GT。
