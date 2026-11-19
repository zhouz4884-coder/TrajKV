# R42 最终状态 post336：固定200步校准结果

固定seed1212、70问、200次更新的训练已完成。完整70问源平衡 Full-teacher KL 从 **0.09217983596 降至 0.02301817567（下降75.03%）**。这是calTrain上的校准目标结果，不是任务准确率或新源泛化结果；最终step200参数在后续质量反馈前冻结，不按select或GT挑迭代。

| 校准域 | source / 问题 | 初始 KL | 最终 KL |
|---|---:|---:|---:|
| Doc | 16 / 45 | 0.0743391 | 0.0306188 |
| Text | 16 / 25 | 0.1100205 | 0.0154176 |
| 域/源/源内问题等权合计 | 32 / 70 | 0.0921798 | 0.0230182 |

输入只含既有calTrain32的独立问题，经`independent_qa`展开。每步一个问题取teacher token均值；预定source均衡schedule共有Doc100步、Text100步，每源6或7次、源内问题均衡。初末汇总对每域、每源、源内问题等权，未使用select、参考答案或新确认GT。

模型权重、旧Gauss16书、旧f0、J和sign固定。先形成完整R42初始字段，再应用绝对FP32 `[fK,fV,o]`：K的gain和每级alpha同乘fK，K mean不动；V的scale乘fV，`min'=min+[.5*(1-fV)+o*fV]*old_scale`，每级V alpha同乘fV。没有重新编码R。旧f0是在Doc32/G4_256上校准后继承到K1/R43，并未声称是新codec的共同最优；新组合是最终状态coherent fold与teacher-aware校准，不是对原冻结编码器的bug修复。

新增336个FP32 scalar真实计 **1,344B**，旧f0仍计费。保留完整224排序，R42删原第43个V stage，保留23K/19V；相对旧R43为`1344−18Nv`字节，当前实际输入满足不超过旧B0。

真实GPU证据包括9,548/9,548次STE桥接输出与实际packed输出位级一致，1,988/1,988次零参数actual reader与plain旧R42位级一致，dense shadow最大相对L2为0.0001950（最大绝对差0.0078125，按声明的绝对+相对容差验收）。首个真实问题与首个optimizer step的q和三参数族梯度均非零、有限。前向使用未改的packed reader，dense分支只给surrogate derivative，不能描述为硬量化或全模型的精确/无偏梯度。

v1新trainer错误地要求每个问题的q梯度都非零，在step51停止。v2没有修改物理变换、forward、loss、optimizer或schedule，从头重跑200步；旧source/run保留。前51步的schedule、KL和q hook/范数均51/51精确复现。step51原例有27个hook、q梯度全部有限但范数为0，KL约−4.26e−10（数值零附近的FP32舍入），三族含regularization的参数梯度范数均约4.1e−5且有限。它属于有限零梯度，不是缺hook或NaN；不能仅凭零值判定autograd断开。v2仍硬拒missing/nonfinite，首probe/step0仍要求非零，之后有限零只记录。本轮6个有限q零步骤为51、59、87、101、137、165，均来自两个近数值零KL问题；200步参数梯度全部存在且有限，q hook均存在且有限。

成功run为`results/round8_v1/postcal_train32_v2`，包含10个canonical小JSON/JSONL和最终`complete_postcal_train.json`。最终marker SHA为 `9646c6c396deeb365d48fbdbb4eaef4e397d82d1716be63a7b04c4e436fcb7a5`；`factors.json` SHA为 `f68dfe87199ff0215ee4aed85a203744577a76e70c97b03ba7079aaa04c8b090`；336个FP32表值SHA为 `142879fdb6f6afe599d867aee2bc8e49f9ff0a7cdbe3a5c3816a267d227f28c9`。已通过部署owner的独立artifact图、旧f0/book、448B packed完整排序和共享helper SHA检查。

汇总入口为`results/round8_v1/postcal_training_v2_summary_v1.json`，SHA `887e58c6dc54c51ac2ceef49aad6982bf15b891f9467fcbd3ea67c865b073a77`。失败v1的7个小JSON已原路径保存在`results/round8_v1/postcal_train32_v1`，没有最终marker或learned参数；两个run的实际专用launcher job/launch/snapshot库存另保存在`outputs/round8_v1/postcal_train_runtime_v1`。运行中曾发生本地stage回执重名并过早依赖启动的操作错误，旧回执未覆盖；已单独保存成功的23源、config和live实际snapshot argv补核回执，不据此重写任何source/run。

有效训练GPU0 PID532871已退出并明确释放。200个优化step日志合计122.64秒、中位0.599秒/步，不包含加载、teacher收集及初末评估；不能当作服务时延。此任务未启动额外GPU分支、未下载权重/图片/tensor、未发布。后续QA、最终确认及quiet由根任务独立执行。
