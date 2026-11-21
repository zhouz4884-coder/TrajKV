# Round9 新来源复测阶段：12次运行 / 384个回答

本包仅包含 TextVQA validation 与 InfoVQA TRAIN 各32个固定来源，FP16、原R7、单层Drop4/Exchange、DP3 DropD/Exchange六种方法共12个真实运行，384个答案事件、64个独立问题。此前18次/704个开发答案不重复计为本批新推理。正式四域500和完整Round9目标另行交付。

来源名单01f029…在推理前冻结；候选编码及单层/DP3策略来自旧cal32，未用本批新答案重新拟合。此次数据现已被观察并参与候选比较，是新来源复测与选择集，不是仍未见的盲确认集，不声明模型预训练未见过这些图像。

主要结果分别见两套REPORT/summary/paired原件。所有资源来自真实owner容量和共享backing去重；Info的压缩点最大请求bit均超过1.3，不能用mean bit声称全请求严格1.3。不同候选实际字节不等，不能把质量差当作等字节机制因果。CI未校正多重比较，不以跨零宣等价或统计赢家。质量运行时延不是独占受控速度。

DP3升层[14,22,26]、11个K供体固定来自实际896个旧cal事件的条件损失表。CPU有限表动态规划与原表、脚本、summary均归档；此最优仅针对固定q/head-layer可加代理，不是全模型或任务最优。部署完整字节差为Exchange 765−6N、DropD −198N−1283，K12/V20、32个残差单位，448字节全局int8 lookup、2048字节共享高码本仅收一次。

数值支持为168次实际cal策略reader/field比较及56次独立nonempty-KRE诊断比较，共224次；后56次使用诊断fixture，不是任务候选，不添加任何问答。私有校准PT/W_O只保留生产者记录的哈希，不声称独立重hash，也不发布原图、权重、PT、认证信息或私聊。

旧R7/R9 v3/v4、18/704阶段与归档均保持原字节。V5仅新增custom producer contract；不伪装SourceRun canonical admission。旧流式stat/SHA/逐行SHA/LF/ZIP机制保持。
