# R9 public v2 → diag v3／BRPE v4 最小扩展建议（只读建议）

当前源码未修改。已审v2的入口指纹另录JSON；不要把其435b/cebb whitelist偷偷放宽为任意runner，也不要退回SourceRun或伪造source_evaluate。旧R7已发布归档继续保持原样。

1. **diagdriver_v3实际源身份**：从四个已闭diagnostic source_snapshot取得round9_runner_v3.py bb6f766167c13f47b59f5c32aeb35e0556d1b9137596d9ea69ef009bebe38a8c，以及其16项实际source_hashes（新增key_controls_v2 44500c37…dfdf4）。在新版contract和standalone cold同步追加一个明确epoch；不可只拿当前工作树hash填进去。原launch/job/config/raw3、纠正后的producer receipt、实际角色源SHA守卫全保留。

2. **mode与归类**：实际新mode是K-Book_FZJ，不是旧K-Func_FZJ。当前FZJ用的是MSE训练书facc8c…，应从book.objective/实际fit kind记录来源，不能因旧命名写成functional。K-Mu和K-Book_FZJ必须同时保持diagnostic_only=true及method_frontier_eligible=false。576端到端事件分为448参考/候选/复核和128诊断，不将四诊断run加入方法Pareto；给每组单独计数。

3. **现v2具体修改位置**：round9_public_contract_v2.py的DRIVERS/expected_sources/MODES及prepare_diagnostic_support目前不认识v3与K-Book_FZJ；verify_public_round9_v2.py的R9_DRIVERS/R9_EPOCHS和verify_round9也需同样明确映射；export_public_round9_v2.py只需消费新strict module与新cold状态、保持stream/stat/SHA/line/LF/ZIP机械块。建议另存v3入口，保留v2 CPU原件与来源证明。

4. **未来BRPE driver_v4不得预填成功**：当前没有其实际端到端raw/source/scored，不能声明SHA或通过。待真实snapshot后固定driver全SHA和实际依赖集合，包括precision_exchange helper原15e075…、实际calibrator_v2的8b9867…或Root实际最终源；禁止用失败v1的1ce11d冒成功profile。新candidate的profile_path/profile_sha应单独绑定cal32 profile原件、实际cal source/launch/inventory/selection objective和低/高Gaussian静态书。

5. **BRPE不是另一本已拟合64参数书**：其高G4_256是冻结Gaussian书9d3ad3…，低G4_16仍af29f2…，原K112/V224因子及V20不变。不要把profile/high_book字段送入只支持K-MSE/K-Func fitted-book的validate_fit；新增明确的precision-exchange profile closure。profile须给出升级层、恰四个原K stage0 donor、route-only选择与选后整4重放校验，取真实calv2 complete和source proof，不造未运行profile。

6. **完整物理审计**：原int32 lookup1792B→共享signed-int8 lookup448B，所有28层views与全局字段都要重绑、旧backing释放。保留原低书128B并新增高书2048B、全层4head的索引增长64Nv、四K residual减少72Nv、format28B+donor12B，得delta=744−8Nv。cold保留旧621的真实storage平衡并增加上述typed/profile/alias检查；Lookup8／Drop4／Exchange分别标明，不能以view逻辑大小代实际驻留。GPU reader/byte controls须使用真实取回证据，不把CPUfake owner当GPU通过。

7. **失败与数值修复单列**：calibrator_v1因特定环境/宽度797 F64 fused log_softmax失败，保留错误与最小GPU/CPU观察作为technical-support，0任务QA。v2显式max/shift/exp/sum/log及行和/KL/chain/整4阈值未放松；待Root实际cal32成功后再归档其profile。它不是当前576里的BRPE方法得分。

8. **数据和计数**：Info开发true TRAIN、Text开发validation；后32复核已观察，不能再称后续candidate盲确认。两个MSE64点仍由各自2×32真实source组成，不伪raw64。未来selected Doc/Info formal500要保留原500输入及主metric/source lock，独立于32/64开发逻辑点；新保留确认32/域仍未使用，不能靠已熟悉TRAIN高分宣泛化。

9. **发布前实际正例仍必要**：新版本只能在Root确定阶段cutoff后运行真实source closure/export/归档内standalone ZIP+metrics/cluster-CI冷核验，再由Root发布。未执行MV/FB/density/跨层不出结果表，WO/PT/原图/权重不入包。当前仅建议，不改任何Root/Delivery执行源，不GPU、远端写或Git。
