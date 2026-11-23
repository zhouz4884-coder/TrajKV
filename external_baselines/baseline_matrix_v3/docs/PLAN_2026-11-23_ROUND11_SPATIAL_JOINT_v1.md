# Round11执行稿v1：原预算下的联合V编码与空间条件残差

2026-11-23。用户已授权“方案够即可开始”，核心方法与对照已足够明确，进入实现和真实GPU验证。本文冻结研究合同；源码、配置、具体run与技术修复版本由各自manifest绑定，失败证据保留。当前尚未取得任何新自由回答质量结果。

## 阶段目标

在完整InternVL3-8B、实际packed reader、question_unknown视觉前缀协议中，检验已有低比特base与residual字段能否重新组织成更有效的表示，在原R7low逐ID完整B0费用下恢复更多真实VQA答案。完成四候选D96、必要的有界迭代/独立来源复测、完整质量/费用/时间报告与自动GitHub交付后收尾。阶段执行完成不等于新方法胜出或论文已完成。

## 模型与数据合同

M1 revision259a3b64a14623c0ec91a045cb43f7c5af5fa6af；完整FP16模型，图像cap8，原模板与suffix、max_new_tokens64、seed0、原position IDs/visual边界及视觉token全支持。编码在问题到来前完成；GT只进入独立评分。旧R7资产、R9/R10源与结果不覆盖。

D96为原D160不可变顺序的Text前32、Info前32、Doc16、Chart16，不按预测或得分选图，只作开发。SourceGroup manifest SHA663076f21fc7df7d00253d06604e9f46da06ffe9d62532cd66d82794c05fa9b4。原R10输入manifest SHAb2ba5f54516abf96344e02356c7cee079b4ff48e2a270ea8fc61ea1c78baaecf；完整旧文件先验后slice。相同来源/图像/模板/位置/生成及评分合同下，复用FP16、R7low、R7high的288条旧预测，保原完整producer/count/source/PID/exit0证明与子集逐行身份，不计新推理。

Text官方十票soft，Info/Doc冻结ANLS，Chart冻结relaxed主指标。四域分别报告，source配对CI仅作探索描述；不把96来源×4候选称384独立来源，不跨异质指标池化成3pp非劣证明。

## 四候选

- JointV：固定实际Half base端点，对V20联合重选基础bits、sign、alpha；理想实数内层按128阈值扫描至多129区间。真实Half候选含原alpha、round及相邻Half，四实际电平择优，strict改进才换，tie原字段。不宣称Half域全局最优或任务非劣。
- Spatial：全部base冻结，仅对V20选原P0或P1。P1=.5Bi+.125BL+.125BR+.25Bvert，vert为两行配对邻居，水平边界回退中心。同32word的n=4bi+bL+bR+2bvert，canonical P1=Half(((8-n)*A0+n*A1)/8)，实际端点是已fold的Half值。
- JointSpatial：先JointV写完最终base，再冻结它做空间mode选择；不保留第二份旧base，不逐token改b后继续引用旧邻居。
- Shuffled：原base冻结，固定解析同word置换共轭原邻接，保持度数、边界自回退、同权重/同算术，作空间对照。

Spatial/JointSpatial/Shuffled用原FP16 alpha signbit作每token mode，幅度用abs，-0按bitcast处理。mode0保原decode；mode1表示空间预测＋sign残差。K23与全部K字段不变，未覆V不变，原V0只fold一次。V没有16词方向书；该书属于K。

## 完整预算与数值

原43槽全stage0，K23/V20；complete visual owner B0(N)=4590N+89733，payload、mean/gain、centroid、codebook、positions、lookup、ranking/policy、f0资产全计。四候选应与同输入原B0精确相等；约1.3不是严格逐题≤1.3。无新增persistent Tensor/矩阵/模式表；真实新版本reader解alpha符号语义。原FP16视觉cache、解包bits、预测/编码workspace不能留在serving owner。时间和临时峰值另计。

先核当前M1实际pixel_shuffle/flatten的16×16网格及tile/thumbnail断开，再真实CUDA scalar/packed数值；flag0须与旧reader逐位一致。随后做实际8B原R7+四no-op+四encoded共9源级TF、原生teacher cap4，QA0/成本单列；不得以局部损失门控正式任务。算子/执行错误用新版本新run修复，原技术失败保留，不能写成已完成新质量。

## 两卡任务与首批计数

四新方法×Text32/Info32/Doc16/Chart16=384次新增自由回答，16个新quality cells。三共享参照288是复用；主视图共672事件/28格，仍只有96开发来源。两GPU按完整cell分为各192回答，无无意义重复基线。

模型、原始数据、结果及TRITON/CUDA/TorchInductor/TMP等全部新目录在远端项目8TB数据盘，本地大文件在E。采用现有baseline框架ops/stage.py与ops/launch.py创建不可变全框架source snapshot和数据盘cache；该框架包含新runner及legacy依赖。旧TrajKV launch_remote.py/remote.py不改。启动前检查现有worker，已授权任务之外的GPU进程不终止。

## 有界自主迭代与确认

根据完整D96真实答案、mode使用、重构/读出变化及成本，自主决定是否追加至多两版有实际机制差异的读出度量候选（SpatialWO、JointSpatialWO，最多另192开发回答）。优先使用原模型W_O导出的度量，若临时计算则记encoding/workspace，若常驻新参数则真实收费；无偷偷增加新bit/FP16缓存/监督信息。新增参数/源码需新配置与run，不覆盖首批。

最多两候选在全部D96后冻结进入R128：Text48、Info48、Doc16、Chart16。选择规则seed=Round11_confirm128_20261123_v1在primary结果前冻结；排除旧历史、formal500以及R10 cal64/D160/reserve224，按ID/source/Doc-parent/imageSHA，先源选择后解question/refs；真split与既有native-full曝光明示。manifest SHAacdfb720120d55793a4f6e7dc2b43e5681c0727aff28e8c906ecbc6696578380已物化独立核。R128不用于新参数选择。

确认重新真实运行三参照与所选一/两候选，512/640次新增自由回答，不复用D96预测。只有原字符串变化而任务/成本无优势的版本不为填表强行扩跑；是否值得复测由实际研究证据判断，不以p<.05或BEC阈值作为许可。未触发的确认/追加明列NOT_RUN与原因，质量空值，不能补0。若仍未形成优势，保留负结果，不由此宣称1.3编码族物理封顶。

本阶段不默认扩到官方四域500、跨模型、整图递归DPCM或低分辨率base新格式；那些是冻结有竞争力版本后的下一步或新设计。不得把未定费用/新reader的储备方案塞入这384上限。

## 收尾与论文推进

报告全部实际方法、源/模型身份、逐题预测、冻结评分、改善/损伤、真实完整字节和原型计时；cal/TF/QA0与free答案分开，发展与确认cohort分开。争取同B0质量提升，或同原高档质量下更省费用；不从CPU失真外推正式得分。

阶段结束自动增量同步既有公开TrajKV仓库：代码、配置、可浏览表/报告与可复核归档，核验远端commit/tree/正文及归档SHA，保旧namespace/config/results/releases/tags/history。私聊、凭据/连接信息、原图、模型weights、私有PT/WO/raw logits不公开。source/public/LF/Git/archive身份分开，refs/input lists显式进入Git树。无实际上传成功不报告已同步。
