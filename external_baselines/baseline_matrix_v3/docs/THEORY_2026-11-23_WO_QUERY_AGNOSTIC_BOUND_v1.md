# W_O度量能保证什么：不依赖未来问题的直接V误差界

这是一条局部模型代数结论，不是答案质量保证。实际GPU使用Half解码值与FP64 Gram作为编码目标，有限精度attention乘加的舍入误差另由数值控制核验，不包含在以下实数公式中。

## 定义与上界

在固定一层，j为query head，g(j)为其GQA对应的KV head；M1有28个query heads、每个KV head对应7个。W_j是该head在原W_O中的输入列块，e_gi为第i个视觉位置解码V与原native前缀V的差。文本V不改变。对任意非负、对全部可见位置归一化的attention权重a_ji，定义直接值误差：

Delta_y_V(a) = sum_(j,i in visual) a_ji W_j e_g(j),i。

向量Cauchy不等式给出：

||Delta_y_V(a)||^2 <= (sum_(j,i in visual) a_ji^2) * (sum_(j,i in visual) ||W_j e_g(j),i||^2)
                     <= 28 * sum_(g,i in visual) e_gi^T H_g e_gi，
H_g = sum_(j:g(j)=g) W_j^T W_j。

第二步使用每个head的visual attention质量至多1，故sum_i a_ji^2<=1。该界对所有合法query/attention模式都成立，不需要预测未来query。N越大时该总量界可能很松，它不等于精确最坏attention损失。

在实数模型中，若对原覆盖V20逐token按上述精确Gram严格接受、其余视觉V单元不变，则每层这一共同Cauchy上界的目标项不增加。R11实际程序比较的是FP64计算Gram和二次式，其严格验收仅保证该计算目标不增；没有计算Gram与求值的严格舍入余量，不能据此声称原W_j精确数学上界也逐位不增。若要转成数值证书，需要另计Gram误差与二次式求值误差，本阶段没有该证书。固定码字与sign时的条件alpha，以及单坐标实际Half变分式，见WO补充执行稿；它们是局部求解，不是全局最优。

## 不能把这个界写成整个模型误差不增加

在同一token历史、同一位置索引下，用128维value列向量表示两分支，一层的attention读出差可以精确分成三项：

Y_hat - Y_native = sum_j W_j [sum_(i in all) (a_hat_ji-a_native_ji) v_native_g(j),i
                             + sum_(i in visual) a_hat_ji e_visual_g(j),i
                             + sum_(i in text) a_hat_ji e_text_g(j),i]。

上述H界只控制第二项的一个统一上界。只有额外固定两分支的text V时，第三项才为零；真实后缀重放或自由生成中text V也可能因低层状态改变，不能省略。它也不控制第一项、各项抵消或后续层query的变化。上界下降不意味着实际第二项下降，更不意味着整个模型KL、ANLS或准确率提高。自由生成若token历史已经不同，也不能把上述同历史局部分解直接当作整个轨迹的比较。当前JointV的开发集改善与WO完整自由答案分别报告，不能由定理预言其大小。

## 对下一步的数学启发（未执行）

若V编码仅修复少量错误而主要缺口仍在K，下一步应将合法K码字索引也纳入离散选择。E[(q^T e_K)^2]=e_K^T Sigma_q e_K给出一个路由logit代理；W_Q/合法前缀hidden协方差可以提供无需未来问题的先验，但必须处理RoPE位置分布和visual/text attention竞争。该代理与H_V不同，不能宣称一个统一元素MSE已经解出K/V最优比特。应先用同表示、同搜索预算的H=I控制和固定V底座验证真实任务，再决定是否形成下一阶段。
