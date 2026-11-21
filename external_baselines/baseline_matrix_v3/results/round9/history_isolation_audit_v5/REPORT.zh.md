# Round9历史隔离与新确认池

本轮cal32不是全历史独立：8 Doc＋16 Text来自已登记的历史校准/调参源池，8 Info TRAIN是本轮新源。Text开发64全部与旧R7/R8确认/压缩评测来源重合（原始内容SHA、原sourceID及image cluster均64/64），不应称全历史未见；Info开发64与旧Info validation来源无交集。cal/dev与当前四域正式500仍按来源及raw SHA隔离。

当前Text/Info各64，包括MSE后32，已是观察过的开发/复核面板。后32 Text0、Info−4.08的已观察结果不能再为后续交换candidate当盲确认。本次不改变这些输入、不按其正确/错误挑新样本。

旧Text validation5000×8只是计划：两worker在GPU前DEFERRED，free_answers_generated=0/model_children=0；没有把它误当完整压缩评测。旧全部64个source-list、272份压缩/采样manifest元数据经来源索引核查，270有COMPLETE、2是attempt；登记但未证明执行的cal/confirm保留为保守排除。仅FP16官方full benchmark单列，不等于压缩调参。

早期legacy图是重新编码JPEG，不能只比raw SHA：通过原Doc first1000 order/SHA证明与Text原imageID映射恢复source/Doc parent，同时保留不同原bytes SHA，不伪称字节相等。完整交集/原件SHA/排除身份见JSON和JSONL。

## 已保留但未运行的新确认32＋32

- Text：validation，可用排除后来源1676；固定seed hash source，再独立hash源内QA取32。引用现有8TB canonical原图，无TextTRAIN下载。
- Info：TRAIN，复用全24个已经LFS SHA验证的公开shard；排当前72等后可用sourceID4334，以同规则取32。仅解出这32张新原图到E。
- 全64均不与当前160、正式2000、旧明确确认/压缩/校准来源相交；没有读取FP16正确性或压缩掉分选样。Refs独立。

这些是按已枚举项目暴露隔离的保留确认来源，不表示基础模型预训练没见过，也不由高分推断普适泛化。交换candidate先在当前开发面板有信号并冻结参数，再由Root启用此确认池；当前0新推理/0GPU/0远端写/0Git。
