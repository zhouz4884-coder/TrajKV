# Round8 最终 residual 状态的混合域 KL 后校准：静态复用审查

结论：可以复用现有成熟的「实际 packed forward + dense STE + Full teacher KL」梯度框架，不需要先重写 autograd 或 Triton kernel；不能直接替换旧脚本配置就运行。需要一个小的新校准入口和一个新 bank 后处理候选，固定原 J/sign/完整 ranking，在最终 residual 编码结束后同时更新 base 与所有相应 alpha。此次只读源码及答案无关的样本元数据，没有 GPU、SSH、模型前向、GT 或新数值方法实现。

## 现有可复用点与实际缺口

| 现有位置 | 可以复用 | 不能原样沿用 |
|---|---|---|
| `code/a2/calibrate_direction_metadata.py:28`、`:63`、`:101`、`:110`、`:116` | 冻结模型；336个 log-Kgain/log-Vgain/Voffset 参数；Adam、范围约束；exact-forward/dense-STE；Full teacher-to-student KL；有限/非零梯度检查 | `book==(256,4)`；手写32B K码；只含base K/V，无RC/RA/lookup；不能代表K1/R43最终状态 |
| `code/a2/fit_functional_v.py:82`、`:125`、`:134` | 冻结f0后附加参数、channel-group V fold、同样的STE与KL | `kg=None`，只训练V；旧G4_256；没有residual，不能直接覆盖新的336参数族 |
| `code/a2/round6_fit_gamma.py:92`、`:112`、`:127`、`:138`、`:159`、`:194` | 更近的骨架：固定编码/residual compact pool、实际packed前向、完整问题suffix重放、Full KL、每次前向位级控制；KL路径保留q/text梯度 | 旧MixedResidual单stage、旧book/32B码、`[2,H]` lookup、FP16 gamma参数计费，不能直接当R8的`[2,H,2]`与FP32 post336 |
| `code/a2/round6_student_codec.py:177` | `teacher_case`仅从Full greedy产生目标token与teacher logits，保存图像边界前prefix、问题suffix及teacher历史；不读参考答案 | 旧 `encode_layer`/`select_units` 是旧布局，不能用来分配K1/R42 |
| `code/a2/round8_codec.py:37`、`:51`、`:79`；`round6_depth_codec.py:14`、`:22` | 当前K1 nibble解码/编码、逐级真实FP16 residual；配合 `binary_centroids` 与旧 `fold` 可精确形成初始字段 | 不要由128-token native dump重建全图attention；采样MSE不是本目标 |
| `code/a2/round8_attention.py:74` | 原reader已经接受packed K、binary V、RC/RA和二级lookup，足够执行新字段；serving无需新增kernel或参数输入 | 原reader没有backward；训练继续用现有dense STE，而不是把Triton输出当作有梯度 |
| `code/a2/round8_continuous.py:49` | `independent_qa`把同源各问题独立序列化，不携带前问答案 | 不能直接把grouped train32交旧 `features`，否则只用 `content` 中的首问 |

旧f0实据：`results/metadata_followup_v1/calibration_s0/all_kv/factors.json`记录32图、83case、400步、336参数、`key_gain_mode=centered`，原书为G4_256。原400步日志的优化段合计148.286秒，中位0.3541秒/步；这是历史已记录耗时，未包括完整收集/加载，不能当作新R42校准的计时预测。新K1书继承旧f0的流程见 `round7_key_artifacts.py:35`，明确数值未refit。

## 最小参数与物理字段合同

建议第一版只定义 centered-K 与现有V中心锚定fold，避免增加whole-K或visual bias等额外候选。对每层、每个KV head，新增

\[
u_K,u_V,o_V\in\mathbb R,\qquad a_K=e^{u_K}>0,\ a_V=e^{u_V}>0.
\]

共 \(28\times4\times3=336\) 个FP32 scalar，初始化全部为0（实际倍率1、offset0）。旧f0保持原值，并在形成所有原始base与residual后才应用post：

\[
g'=Q_{16}(a_Kg_0),\qquad
\alpha'_{K,s}=Q_{16}(a_K\alpha_{K,s,0}),\qquad \mu'=\mu_0.
\]

固定J和每级sign；理想实数语义为 \(\mu_0+a_K(\hat K_0-\mu_0)\)。若未来要whole-K scale，必须另声明 \(\mu'=Q_{16}(a_K\mu_0)\)，否则两种操作不能混称。第一版沿用旧centered含义，均值不变，不加入Mean/Ray的新组合。

V采用已存在 `affine_rate_bank.fold:11` 的锚定含义，令 \(M_0=m_0+S_0/2\)：

\[
S'=Q_{16}(a_VS_0),\qquad
m'=Q_{16}\{M_0+(o_V-1/2)\operatorname{FP32}(S')\},\qquad
\alpha'_{V,s}=Q_{16}(a_V\alpha_{V,s,0}).
\]

该offset是新scale单位，理想语义为 \(M_0+a_V(\hat V_0-M_0)+o_Va_VS_0\)，并非统一native标量偏置 \(a_VV+b\)。若改用literal affine，则应另定义 \(m'=Q_{16}(a_Vm_0+b_c)\) 及b的单位；一个head只有一个offset参数，不能暗中新增128个自由channel偏置。

身份初始化还有一个具体数值条件：直接照搬 `M0+(o−.5)*S'` 的FP32加减，在极端scale/min量级差下不具有通用的bitwise identity保证。新的post实现可明确使用 \(m'=Q_{16}[m_0+\tfrac12(S_0-\operatorname{FP32}(S'))+o_V\operatorname{FP32}(S')]\)，使a=1/o=0时差项严格为0；它保持上述理想中心锚语义，但与旧fold的非零参数FP32运算顺序不同，须作为新post合同记录，不修改旧fold。另一简单选择是literal \(b_c=o_VS_{0,c}\)、\(m'=Q_{16}(a_Vm_0+o_VS_0)\)，也能保持零参数身份。二者只选一种；静态审查不声称已经跑过实际字段身份检查。

全部新字段按原部署的nested FP16次序解码，不重新求J、sign、MAD或重编码residual。只变base而把alpha固定会改变所声称的全状态仿射；先变base再重估residual则可能由R抵消变换。一般倍率/offset不能与逐级FP16舍入交换，所以这不是「对旧最终解码向量做affine」的位级等价式；优化和验证都必须使用实际新FP16字段。

## 训练前向与梯度路径

每个calibration source以旧f0、旧book和明确R42初始状态，使用当前生产编码函数形成全部真实Nv的固定packed字段。可用生产bank仅做 `no_grad` 收集并及时关闭其PhaseState hook；不要用serving的in-place text append路径直接反传。loss沿用 `round6_student_codec.teacher_case` + fresh `make_prefix_cache` + `round6_fit_gamma.loss_for` 的普通DynamicCache和 `_p185_read` callback。

训练时从固定初始fields函数式构造g'/S'/m'/alpha'，不能先 `copy_` 到无梯度state后再当作梯度输入。dense分支以 `key_template`、binary unpack和每一级 `decode_sign_stage`重建相同字段；保留q、当前student text K/V梯度。exact分支在 `no_grad` 下调用原 `round8_attention.attention`，两者桥接：

\[
O=Q_{16}\left(O_{dense}+\operatorname{stopgrad}(O_{packed}-O_{dense})\right).
\]

必须继续逐调用 `torch.equal(O.detach(),O_packed)`；现有 `round6_fit_gamma.py:159` 已有这个控制。STE仅是dense surrogate导数；实际硬量化KL并不保证逐步下降，不能宣称精确量化梯度或无偏梯度。冻结所有模型权重，参数梯度须存在、有限且整体非零；记录K/Vgain/Voffset各族梯度，个别头为零不自动构成代码失败。

calTrain32的答案无关样本SHA为 `5b3f97ea175103903d988480ee6977d07e17f688a21126545d3fa4b2be39466d`。实查为16 Doc/45问、16 Text/25问，共70问；样本顶层无answer/answers/references。展开全部70问时采用

\[
L=\frac12\sum_{d}\frac1{16}\sum_{s\in d}\frac1{|Q_s|}\sum_{q\in Q_s}\frac1{T_q}\sum_t
\mathrm{KL}(p^{Full}_{sqt}\Vert p^{post}_{sqt}).
\]

不能对70case简单等权循环而声称域/源等权，否则Doc45问和多问source被过度加权。使用预先固定的均衡source/question schedule或相应权重；校准与checkpoint选择只用train teacher KL，不能用select或新GT。teacher上限可沿用8 token；每步是完整compressed-question+teacher-history重放，不是native-MSE。

## 真实计费与三个最小variant

旧f0的336个FP32 scalar仍计1,344B；新post336再计1,344B，两者不可合并少计：f0先影响base和原residual的生成，post则在最终字段形成后应用，且存在不同FP16舍入点。post即使仅encoder使用也按本项目完整owner规则计费；JSON文件长度不充当tensor字节。

保留完整224 ranking与原lookup，明确stage_limit42。实际第43项是 `[v,27,3,0]`，所以R42保留23K+19V。相对于旧R43的实际B0：

\[
B_{new}-B_0=1344-18N_v.
\]

Nv1280时为 −21,696B（5,964,933→5,943,237B）；Nv1792时为 −30,912B（8,315,013→8,284,101B）。一般需Nv≥75，合法32倍数下至少96；Nv32/64不能据此声称≤旧B0。新planner必须先预留1344B、保留完整ranking/policy计费，再验证实际owner等于预测且不高于对应旧B0。可复用 `round6_gamma_bank.py:46` 的整数reserve/`nextafter`模式，再接当前 `plan_key_stages`；不能把post表漏在planner/owner之外。现field/postres/mean的专用validator硬要求43，应新增很小的新validator，不放宽冻结规则。

最小比较是：旧 `R43/f0`；`R42/f0 + identity-post336`；`R42/f0 + learned-post336`。后两者保留同形post表并同样计1,344B，identity须对纯R42的字段/reader逐位一致。训练增益比较以后两者为主，旧R43用于核对总体预算与阶段删减代价。固定旧book/f0/J/sign/224 ranking，模型权重不动，Mean/Ray旧方法与旧结果不动。

## 工作量与停止条件

实现范围是两个小适配面：校准收集/forward adapter，以及encode后fold+计费bank/validator；可复用现有teacher、optimizer、STE、原reader、硬decoder和预算逻辑，不需要新训练平台。未实测，不能保证24GB GPU激活/CPUcase缓存峰值：旧脚本逐case保存native prefix K/V，70问若重复存同源大prefix会占多GB主存；可先仅校准32个首问source或按source复用冻结prefix，但必须如实声明问题覆盖，不能把32首问叫70问。

若复用现有400步计划，工作是最多70个Full prefill+最多490个teacher decode outer-forward、400次compressed suffix forward/backward，以及完整70问初末KL评估；这是操作量，不是时长承诺。最先只需一问检验θ=0字段/reader identity、exact-forward identity、336参数有限梯度与模型无梯度，再开展已授权的完整训练。若plain DynamicCache/当前Transformers路径使student text梯度断开或OOM，先记为真实实现/资源阻塞；此次静态审查没有把未跑的梯度或质量宣称为通过。
