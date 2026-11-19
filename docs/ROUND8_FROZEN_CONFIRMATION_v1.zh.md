# 冻结四方法的新来源首问确认

在fixed-final200参数冻结后完成256个来源的256道首问、四方法共1,024条真实回答。所有方法、参数及40个冻结文件保持原字节；没有依据新确认结果调参。Doc来自官方TRAIN的项目source holdout，不能称官方validation/test；Info和Text分别为官方validation来源留出。三域独立报告，不计算跨域平均。

| 方法 | Doc TRAIN holdout ANLS（64） | InfoVQA validation ANLS（64） | TextVQA validation accuracy（128） |
|---|---:|---:|---:|
| Full | 99.4420% | 74.2898% | 81.7969% |
| 冻结 R7 K1/R43 | 99.2299% | 72.2325% | 76.3281% |
| 旧 K2 1.8-bit预算锚 | 99.2217% | 71.1277% | 78.2031% |
| 新 postcal R42 | 98.6273% | 71.8419% | 75.6250% |

| 新方法 − 旧R7 | 点差（pp） | paired 95% CI（pp） | Holm |
|---|---:|---:|---:|
| Doc TRAIN holdout | −0.6025 | [−2.0833, +0.2757] | 1 |
| Info validation | −0.3906 | [−2.3438, +1.1719] | 1 |
| Text validation | −0.7031 | [−2.9688, +1.0938] | 0.7506 |

新方法在三个确认域均没有高于主比较器R7的点估计，区间均跨0；不能声称确认到质量收益。Text相对Full为−6.1719pp、CI[−10.5469,−2.2656]pp，报告比较族内Holm=0.01263。统计沿用原source-cluster bootstrap和100,000次paired sign flip，每域对候选与3控制做Holm，不池化不同数据集分数。

价格为包括实际encoder常数的完整visual owner：Doc中旧R7/新方法约1.29473/1.28991bit，Info约1.29642/1.29163bit，Text约1.29563/1.29083bit。新增336个FP32数值表完整计1,344B，旧f0仍计费；每来源新R42相对旧K1/R43的字节差严格等于`1344−18Nv≤0`。旧K2锚是Gaussian256×4 K2＋binary/all V1，按实际Nv选择可负担前缀，实际预算约1.8bit；不是固定43stage或与K1同价。三个域实际K2阶段范围分别42–44、26–44和40–44。

全部候选对同一source使用相同图像、真实Nv、cap8和唯一首问。Doc/Info/Text实际Nv范围为1280–1792、256–2304、768–1792；小Nv也以真实常数与实际残差池计费，不填充dummy。Text保留全部10个原始reference votes；参考答案只在本地CPU、合法final marker后用于评分，没有进入远端runner。

每个producer的`complete_postcal.json`、R8/engine、records、config、样本顺序、训练参数图及来源SHA均严格通过。Doc和Info的live `/proc`入口已实际抓到，Text使用根任务保存的live证据；三者Popen及snapshot也已核对。256张图像在推理后远端只读重算SHA，仍与冻结样本一致；没有下载原图、权重或tensor。用于本地评分/准备的两个非运行时源码虽在40项冻结清单中，原确认snapshot未包含它们；它们并不被renderer执行。实际消耗的runtime源码、全局codec/训练资产和本地40项分别明确核验，未把未执行文件库存当成执行证明。

额外首target仪器只取两个旧train QA、identity/learned各一次，共4pairs，不看GT、不拟合参数。训练分支完整teacher-forced suffix的首target logit与真实部署question-only首logit的top1在4/4相同，最大绝对差0.015625、最大相对L2约6.28e−5；identity也不位级相等。部署捕获SHA严格等于实际`initial_logits_sha256`，使用同一绝对FP32表且没有log/exp往返。该结果未见所测首步的巨大协议不一致，但不能保证全轨迹一致或解释全部开发质量变化。仪器在首次轮询前已经完成，执行证据如实为Popen＋实际snapshot/输出SHA，未重跑获取live画面。

总入口为`results/round8_v1/confirmation_firstq4_v1_summary_v1.json`，SHA `4645e9a4fa80e478e435d038267770899b61a63e8cc047ad55f5f02b9d236ada`。各域正式分析位于`confirm_firstq4_v1_{pool}_analysis_v1`，原始producer、价格审计、live/runtime、图像后验SHA及纯诊断报告均由总入口绑定。

Doc PID538666、Info PID543063、Text PID542273全部完成退出；本子任务GPU1已释放，短仪器PID548438退出后GPU0也已释放。后续continuous/quiet和最终公开交付由根任务统一完成，本文不提供时延、goodput或服务SLO结论。
