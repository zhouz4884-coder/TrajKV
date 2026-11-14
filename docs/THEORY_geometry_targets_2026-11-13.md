# 注意力需要保真的量，以及可验证的径向／方向分解

本文件整理数学事实和当前研究的可测量解释，不把现有编码的成本点写成信息论最优。

## 1. 固定方向时，MSE 最优半径通常收缩

设真实 key 为 $k=r u$，单位量化方向为 $\hat u$，$c=u^T\hat u$。固定方向且半径非负，

$$\|r u-\hat r\hat u\|^2=r^2+\hat r^2-2r\hat r c,
\quad \hat r_{MSE}=\max(0,rc).$$

在 $c\ge0$ 时，真实范数恢复 $\hat r=r$ 相对 MSE 最优半径增加 $r^2(1-c)^2$ 的平方误差。由此可知“更小的元素 MSE”与“更真实的幅度”不是同一目标。是否更有利于任务，要由相同方向下的真实模型干预决定。原点 LS 增益 $k^T\hat k/\|\hat k\|^2$ 也与 norm ratio 不同；中心化 LS 则只拟合相对中心的残差，不等于恢复整个 key 的范数。

## 2. 范数／方向的精确误差恒等式

对任意非零真实和量化 key，$k=ru,\hat k=\hat r\hat u$：

$$\|k-\hat k\|^2=(r-\hat r)^2+2r\hat r(1-u^T\hat u).$$

这是可在 GPU 原始 K 上逐行核对的径向与方向能量分解。两个能量之和是重构平方误差，但能量份额不等于任务因果份额。四臂使用**实际量化范数**与**实际量化方向**，分别替换为真实量，不把真实量化表示偷换成常数半径。MAD/Lloyd 有非零组中心且 g128 参数跨块变化，量化范数不必为常数；常数数组的相关系数未定义。

## 3. 注意力 KL 的目标是 logit 扰动的相对结构

令 $a=\mathrm{softmax}(z)$，$\hat a=\mathrm{softmax}(z+\delta)$，有

$$D_{KL}(a\|\hat a)=\log\mathbb E_{i\sim a}e^{\delta_i}-\mathbb E_{i\sim a}\delta_i.$$

加到所有位置上的同一常数完全抵消。在小扰动下，二阶项为 $\frac12\operatorname{Var}_{i\sim a}(\delta_i)$；这是展开式，不是对大扰动的近似保证。对有界扰动 Hoeffding 上界为 $D_{KL}\le(\max\delta-\min\delta)^2/8$。准确性取决于沿实际 query 方向的误差及注意力权重，单一元素 MSE 或全局 SNR 不能给出普适可用门槛。

若所有视觉 logits 同加偏置，视觉内部排序不变，却会改变视觉与 FP16 文本之间的质量分配。正比例乘数保持视觉内部 argmax，但能改变 softmax 浓度与跨模态竞争。因此分段温度必须保留文本对照；输出词表温度不能用于 greedy 修复。

稳定 top-1 的精确条件是：对真实赢家 $i^*$，所有 $j$ 满足 $\delta_j-\delta_{i^*}<z_{i^*}-z_j$。它依赖真实间隙，而非固定 SNR=1 或5。正比例放大可保序而重构误差很大；极小间隙也能被很小扰动翻转。

## 4. 与先行工作的关系

MSE 与内积目标的区别已有 [TurboQuant](https://arxiv.org/html/2504.19874v1) 的理论和算法；[QJL](https://arxiv.org/html/2406.03482v1) 已用 sign sketch 估计内积。[PolarQuant（Han 等）](https://arxiv.org/html/2502.02617v1) 使用随机预条件与递归极坐标，[同名 PolarQuant（Wu 等）](https://arxiv.org/html/2502.00527v1) 对 post-RoPE 二维子向量编码半径和角度。[AsymKV](https://aclanthology.org/2025.coling-main.158/) 已研究 K/V 结构角色与逐层不同精度。不能将“范数和方向可分”“K比V敏感”“MSE不是内积目标”单独宣称为首次。

还查到 [Villines 的 N+D 预印本](https://papers.ssrn.com/sol3/Delivery.cfm/6828039.pdf?abstractid=6828039&mirid=1&type=2)，目前核验到作者原始摘要，全文访问未成功；其摘要明确讨论逐头最小余弦与幅度／方向编码。未独立复现其数值。

本项目可主张的新增证据需落在：视觉文档精确读取、同格式电平变化、真实量化范数×方向的四臂端到端效应、状态形成时点对结论的影响，以及包含辅助成本的实际码率—质量结果。它们是待确认的具体贡献组合，不能由上述数学恒等式直接推出论文新颖性。
