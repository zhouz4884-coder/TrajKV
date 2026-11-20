# C2精度合同的独立复核（source replay待执行）

**当前判断：C2值得立即继续单题及固定128源pilot；现有证据支持它是可有限运行的明确精度移植，但最终operator误差合同尚待一次同源重放。** 不能因一个FP64输出阈值失败便排除这个更强外部bitwidth，也不能只凭全局L2小或输出接近零便认定误差全部合理。

|已冻结观察|数值|能说明什么|
|---|---:|---|
|原FP16版本C2|有限bias乘积的Half归约溢出|原始失败保留；是数值范围问题，不是packed码错误|
|FP32 bias SUM/DIV端口|两图完整28层suffix和独立freegen有限|可运行；尚非大规模任务质量证据|
|source0/layer26对FP64 instrument|最大绝对差0.015625；17/78,848元素不同；相对L2约7.717e-5|不是位级一致；也不是由此即可认定算法移植错误|
|同图对作者AST单query执行|最大差0.015625；16元素不同|同精度不等于相同GEMM/reduction形状|
|原QK kernel对FP64-rounded rawdot|最大0.0625，最大差位置为1个Half ULP|源FP32归约后Half存档可以跨舍入边界|
|原WV kernel，同一实际p|对FP64-rounded原始加权码差为0|不能把最终输出差直接归因于WV原始kernel有大误差|

source的`calibquant.py`在QK和WV两个Triton kernel中明确使用FP32乘积/归约后存入FP16。FP64 instrument提供独立高精度计算以及指定的Half边界，但它并非源kernel浮点归约树的逐条模拟。现有端口只把bias SUM/DIV改为FP32、再cast回Half；quantizer、bits2时K/V clip=1、校准max偏移−3、packed格式、原QK/WV kernel及模型精度均不变。

在无溢出且普通浮点误差模型成立的范围，FP32归约满足`|fl(sum a)-sum a| <= gamma_n sum|a|`，其中`gamma_n=n*u32/(1-n*u32)`；Half存档另加Half舍入项。这种界应在**原始kernel输出与相同输入的精确和**之间检查，而不是给最终输出随意加一个绝对阈值。bias的标量分母转换、FP32除法与Half cast同样分开计入。

设两次执行在共同顺序下的Half权重为p和p'，同一存档值网格为v。其理想线性加权差逐通道满足`|sum_i (p_i-p'_i)v_i| <= sum_i |p_i-p'_i| |v_i|`。然后分别加入两次执行实际经过的、逐步验证在IEEE舍入界内的weighted-code归约、scale/zero仿射、概率和、text matmul以及最终相加误差。该方法无需假设CUDA expf等同于精确实数指数，也不依赖把最终117个ULP解释成某一个原始算子的1个ULP。

旧pt未保存实际p和各仿射中间量，因此这些观察**目前不能独立完成上述误差分账**。新驱动`code/gates/calibquant_c2_roundoff_capture.py`只重放这一个已冻结层，不加载8B模型、不读GT；比较生产cache.read与同形源操作重放、作者AST和FP64 instrument，并把实际/参考p及原始归约结果留在远端8TB。CPU审计器`code/gates/calibquant_c2_roundoff_bound_cpu.py`先检查原始算子预算，再验证完整前向界；任一步越界便输出未验证，不修改算法或扩大最终输出阈值。

只有新重放生产输出一致、各源算子在指定范围内以及完整前向界闭合时，才可把这个case记为数值合同支持。该case通过仍不推出所有输入、不同CUDA库或所有head/shape都通过；后续pilot照常保留nonfinite/zero-range等源算法失败，不能做clamp或nan替换。FP16产品自身若溢出，也超出这次只扩展bias归约精度的端口范围。

本轮按源规则新增C2，而不是因为C1试评分低才把其clip或−3偏移调整得更有利。C2最终应与相同模型、原图、processor、prefix边界和自由生成协议下的F/K配对比较。所有quality结论来自后续真实3090问答；这里的CPU界是数值分析，不产生新的任务分数。
