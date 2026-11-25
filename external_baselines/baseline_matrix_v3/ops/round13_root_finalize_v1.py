"""Root-authored actual Round13 cutoff, report and dual identity seal.

Consumes only closed, explicit components. Does not export, authenticate or
publish. Full public cold replay and GitHub verification remain separate.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import round13_public_adapter_v1 as adapter
import round13_public_semantics_v1 as semantic
from export_public import Sanitizer

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def textfile(path,text):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
def add_file(scope,path,public,role,channel='BROWSER_AND_ARCHIVE'):
    path=Path(path).resolve();digest=adapter.sha(path)
    old=next((r for r in scope['files'] if r['public_path']==public),None)
    if old:
        adapter.require(old['sha256']==digest,'Root file collision: '+public)
        return
    scope['files'].append({'source_path':str(path),'public_path':public,'sha256':digest,
        'role':role,'delivery_channel':channel,'expected_in_git':channel=='BROWSER_AND_ARCHIVE'})
def resource(q,key):
    x=q.get(key);return x['mean'] if isinstance(x,dict) else None
def num(x,n=2):return '—' if x is None else f'{x:.{n}f}'
def interval(v):return '—' if v is None else '['+', '.join(num(x) for x in v)+']'

def human_report(quality,pairs,counts):
    q={(r['label'],r['dataset']):r for r in quality}
    p={(r['candidate'],r['reference'],r['dataset']):r for r in pairs}
    bench=[r for r in quality if r['evaluation_phase']=='observed_fixed500_comparison']
    candidate=next(r for r in bench if r['label']=='FIXED500_Pbest')
    high=q['FIXED500_R7high','textvqa'];low=q['FIXED500_R7low','textvqa'];native=q['FIXED500_FP16','textvqa']
    rel=lambda ref:100*(1-resource(candidate,'stored_visual_KV_bytes')/resource(ref,'stored_visual_KV_bytes'))
    lines=['# Round13 完整研究复盘：支持集与精度交换，及 TextVQA 固定500验证','',
      '日期：2026-11-25。GPU实验已按实际执行范围闭合；本报告的 GitHub 交付以另外的远端核验收据为准。', '',
      f"完成{counts['new_successful_terminal_answers']}条新端到端回答：{counts['r13_development_method_successful_terminal_answers']}条开发方法回答、{counts['r12_reference_successful_terminal_answers']}条缺失R参考补齐、{counts['r13_benchmark_successful_terminal_answers']}条新方法固定500回答；失败/不可信终止回答均为0。所有任务使用完整原始M1/InternVL3-8B FP16权重、两张RTX3090，所有远端模型/数据/缓存/结果在8TB项目盘。",'',
      '本轮得到一个值得保留的TextVQA低码率点。Info/Doc的支持删减和全头共享token精度交换没有形成通用升级；RoPE位置中心的数学可行性也没有自动带来更优质量—字节收益。下一阶段应把支持集与每行精度共同优化，并以真实任务结果定价。','',
      '## 一、冻结算法后的同题TextVQA 500题','',
      'P_best在Text64开发扩样完成后冻结，随后才产生这500条新输出。问题、模型、原图、原生输入身份与主评分器同已有参照绑定；旧FP16/R7及外部配置均复用历史推理并重新核验，不计为本轮新推理。该500为已观察benchmark，未称全新独立确认。','',
      '| 方法/具体配置 | 题数 | 分数% | 平均/最大有效bit | 平均完整视觉字节 |',
      '|---|---:|---:|---:|---:|']
    order={'FIXED500_Pbest':0,'FIXED500_R7low':1,'FIXED500_R7high':2,'FIXED500_FP16':3}
    for r in sorted(bench,key=lambda x:(order.get(x['label'],4),-x['primary_mean'])):
        bits=r['actual_effective_visual_bits'];label=r['label'].removeprefix('FIXED500_')
        lines.append(f"| {label} | {r['questions']} | {num(100*r['primary_mean'])} | {num(bits['mean'],4)} / {num(bits['max'],4)} | {num(resource(r,'stored_visual_KV_bytes'),1)} |")
    lines+=['','| P_best相对参照 | 同题数 | 分数差pp | 配对95% CI(pp) |','|---|---:|---:|---|']
    for label in ('FIXED500_R7low','FIXED500_R7high','FIXED500_FP16'):
        pair=p['FIXED500_Pbest',label,'textvqa']
        lines.append(f"| {label.removeprefix('FIXED500_')} | {pair['n']} | {num(pair['primary_delta_pp'])} | {interval(pair['ci95_pp'])} |")
    lines += ['',f"P_best平均完整视觉缓存相对R7高档减少{rel(high):.3f}%，相对FP16减少{rel(native):.3f}%；相对R7低档减少{rel(low):.3f}%。三个主参照的配对区间均跨0，当前支持有希望的质量—缓存工作点，尚未证明显著提升、质量等价或非劣。",'',
      '外部行代表表中已实际实现的具体配置。KIVI_style、source_example、CalibQuant与OnlyTerp等保留各自设定名称，不将某个配置泛化为整篇原论文。完整14个旧参照的原始core、评分和必要来源证据纳入本阶段闭包。','',
      '## 二、用了什么方法','',
      '| 路线 | 具体改变 | 资源与对照 |','|---|---|---|',
      '| P_best / P_random | 全原生图像前缀计算后，保留全局thumbnail与部分位置；K2/V1的完整A编码按真实行重新压紧 | 两选择器同M、同完整low43上限；全N统计保留、位置不重编号、删除行不能回流FP16文本 |',
      '| EX_best | 普通行保留low43，少量行存原生post-RoPE K/V，双路径统一softmax | 每图在R完整字节cap下解合法例外数，计V字填充与全部路由；N256显式回退low |',
      '| TOK_best | 退掉R的K残差，部分token改G4_256 K2，其余G4_16 K1 | V基码与R的V逻辑残差冻结；真实紧凑两池、两码本和int32路由均收费 |',
      '| CAL_AFF | 旧二值实际Half端点的完整仿射迁移到V2R，原V残差重编码 | K与原43覆盖保持，同V2R完整字节 |',
      '| PC / PC_DONOR | 在保留旧post中心a的基础上增加位置相关均值，固定centered-f0；同删最低优先K槽的donor控制 | 新mu/freq/scaling/format共28,936B；逐图退最少K槽支付，全部原R-V字段保持 |','',
      'P的图像侧选择器只读取原生、问题未知的图像前缀KV（层0/13/27的重要度与多样性）；全局thumbnail常驻。每个自由生成步记录raw top1/top2与margin，记录不参与策略。PC读取的是图像prefill当时复制冻结的RoPE频率。编码器不读取参考答案、旧预测或未来问题query。','',
      '有效bit统一以原始完整视觉K/V元素数为分母，包含删除位置带来的存储减少；P实际保留行仍是K2/V1及相应残差，并非把每个保留元素量化为1.286位。比较的主资源量是完整持久视觉owner字节。','',
      '## 三、开发与扩样结果','',
      '均为此前已观察的R11/R12来源，作为开发数据；原Info64/Doc32/Text32及额外固定Text32都保留来源与native绑定。表内分母不同的条目只通过公开的同题配对交集比较。','',
      '| 方法 | Info 分数%(n) | Doc 分数%(n) | Text 分数%(n) |', '|---|---:|---:|---:|']
    for label in ('P_best','P_random','EX_best','TOK_best','CAL_AFF','PC','PC_DONOR','R7low','R7high','FP16','R'):
        cells=[]
        for ds in ('infovqa','docvqa','textvqa'):
            r=q.get((label,ds));cells.append(f"{100*r['primary_mean']:.4f} ({r['questions']})" if r else '—')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    pc=q['PC','textvqa'];ps=q['P_best','textvqa'];donor=q['PC_DONOR','textvqa']
    pc_pair=p['PC','P_best','textvqa']
    premium=100*(resource(pc,'stored_visual_KV_bytes')/resource(ps,'stored_visual_KV_bytes')-1)
    lines += ['',f"PC在同Text64相对P_best仅{pc_pair['primary_delta_pp']:+.5f}pp，CI {interval(pc_pair['ci95_pp'])}，完整字节却多{premium:.3f}%。它的分数点估计稍高，不能说被P严格点支配；当前收益不足以支持本轮继续PC500。Donor的成本更高、点估计分数更低。R只有Text32，所有与R的Text比较保留n32，不制造R64。",'',
      '## 四、实验改变了哪些判断','',
      '1. 少量原生例外没有把局部误差修复变成强任务收益；本次low43＋图像侧选择的配置在Info64只比low提高约0.205pp，相对同预算R仍落后。不能从少数视觉sink或attention质量直接算出答案收益。',
      '2. token层面的精度细化不自动更优。TOK在Info64相对R约−8.98pp，同时改变了K1＋符号残差与K2的表示动作、head/token覆盖粒度及K/V误差交互；本轮没有将失败单独归因到某一个因素。',
      '3. 完整仿射迁移主要保持旧V2R任务结果，仅Text16出现小幅部分得分变化；编码常数可修，并不保证带来值得发表的性能增量。',
      '4. pre-RoPE统计确实更集中，但这只是模型结构信号。PC保留旧f0目标并给出可执行码流，任务收益仍呈域差异；以真实回答而非谱/MSE做晋级依据。',
      '5. Text64的正向趋势在500上缩小：P相对low由+3.4375pp变成+1.66pp，相对FP16由开发集的正差变成−1.96pp。扩样有必要；Native FP16是参照点，也不是任务正确率的数学上界。','',
      '## 五、数学认识与后续抓手','',
      '固定同一层、同一query/text及全支持编码A，令D为删除的视觉位置、a_i为全联合注意力、m_D=Σ_D a_i、o_A为全输出。在实数运算下：','',
      r'\[o_S-o_A=\frac{\sum_{i\in D}a_i(o_A-v_i)}{1-m_D}.\]','',
      '因此删除代价还依赖值向量与当前输出的偏离及跨位置抵消；attention质量本身不是代价。精度变化与删除变化可在固定局部状态写成(o_S−o_A)+(o_A−o_low)，不能将各层局部差线性当成全模型答案损失。', '',
      '当前P只是一个固定字节下的K2＋约72%支持点。对于非回退图像，真实完整字节为6382M＋91657＋8N＋16，M按32对齐且保留thumbnail256行；原low cap为4590N＋89733。N256回退说明“约1.3bit”不是全域统一的理论硬上界；本次Text500实测最大1.2969bit。', '',
      '下一阶段优先两项质量研究：在同一真实字节cap下联动已有K1.5/K2/K2.5表示与保留位置数，测支持—精度曲线；将图像前缀/全局thumbnail query的功能贡献作为问题未知的选择特征，与当前norm＋多样性做同M、同reader对照。后者是待验证代理，不能偷用未来问题query，也不能因局部误差下降就宣称任务改善。', '',
      '以N=1792、同一low cap=8,315,013B为例，冻结factory的实际格式系数给出以下待测曲线；只有A行已在当前稀疏实现中执行。G8还改变block维数，不能伪装成纯bit消融。', '',
      '| 码本 / K位数（V1） | 可保留M / 原N | 真实预算公式 | 此点完整字节 | 当前状态 |',
      '|---|---:|---|---:|---|',
      '| G8_4096 / K1.5 | 1472 / 1792 | 5486M+155145+8N+16 | 8,244,889 | 稀疏compactor待实现，未GPU评测 |',
      '| G4_256 / K2 | 1280 / 1792 | 6382M+91657+8N+16 | 8,274,969 | 本轮P的已实现点 |',
      '| G4_1024 / K2.5 | 1120 / 1792 | 7278M+97801+8N+16 | 8,263,513 | 稀疏compactor待实现，未GPU评测 |', '',
      '质量工作点稳定后，直接编码保留行并保留全N统计，替代当前完整A编码后再压紧的冗余路径；先用逐位控制保持码流等价，再测真实运行代价。之后在第二模型与其它域检验收益。本轮尚不构成跨模型、四域或高水平论文完成证明。', '',
      '## 六、真实资源与证据边界','',
      f"P的500题平均encode为{resource(candidate,'encode_ms')/1000:.3f}s、wall为{resource(candidate,'wall_ms')/1000:.3f}s；这些是带审计/日志的原型质量运行，不是受控速度、HBM流量或吞吐基准。PC还有每次attention临时Half相位表及FP32 scratch。持久视觉KV字节包含真实底层容量、码本、元数据、位置、路由、残差和padding；模型权重、文本KV与prefill峰值分别记录，不能用视觉压缩比替代整机显存节约。",'',
      'GPU数值检查、原始位置与文本排除控制、旧源存储释放以及完整自由生成在不同层次验证。S0成功32次prefix＋32次问题TF，失败尝试另1＋1，总66次非任务模型前向；已闭合CUDA测试109个case（含必要重试），不计作新回答。四次技术/采集失败尝试保留原始来源与日志。', '',
      '完整质量、配对、资源、计数与来源清单以FINAL_SCALAR_REPORT和CSV为准；公开封包保留source/public/Git LF/ZIP各自身份。原图、原模型权重、私有激活、凭据、连接信息与私聊不上传。公开冷核验重放标量评分、来源与完整字节，不声称重新运行GPU或私有tensor。','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('--components',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--version',default='v1');a=p.parse_args()
    started=time.perf_counter();a.output.mkdir(parents=True,exist_ok=False)
    comp=json.loads(a.components.read_text(encoding='utf-8-sig'))
    adapter.require(not comp['cheap_preflight']['blockers'],'Root cutoff has pending dependencies')
    scope={key:comp[key] for key in ('files','cells','views','comparisons','qa0','attempts')}
    scope.update(schema=adapter.SCHEMA,namespace=adapter.NAMESPACE,fixture_only=False,
        research_cutoff_complete=True,stage_state='READY_FOR_ROOT_SEAL',phase_label='round13_20261125',
        whole_external_library_completion_claimed=False,SourceRun_admission_claimed=False,
        metric_contract='matrix_v3_metrics_v1')
    if not any(x['candidate_view']=='PC_textvqa' and x['reference_view']=='P_best_textvqa' for x in scope['comparisons']):
        scope['comparisons'].append({'candidate_view':'PC_textvqa','reference_view':'P_best_textvqa','bootstrap':2000,'seed':20261125})
    groups={}
    for cell in scope['cells']:
        if cell['kind']=='reused_reference':continue
        if cell['kind']=='new_round12':group='reference_R32'
        elif cell.get('evaluation_phase')=='observed_fixed500_comparison' or 'fixed500' in cell['cell_id'].lower():group='benchmark_P_Text500'
        else:
            cfg=json.loads(Path(next(r['source_path'] for r in scope['files'] if r['public_path']==cell['producer']['source_config'])).read_text(encoding='utf-8-sig'))
            mode=cfg['candidate']['mode'];group=('support_P320' if mode.startswith('P_') else 'PC_geometry224' if mode in ('PC','PC_DONOR') else mode)
        groups.setdefault(group,[]).append(cell['cell_id'])
    scope['groups']=[{'group_id':key,'disposition':'EXECUTED','cell_ids':ids} for key,ids in groups.items()]
    for name,reason in (
        ('PC_fixed500','Text64 only +0.15625pp over lower-byte P while costing20.67% more; no additional benchmark was selected.'),
        ('other_domain_fixed500','The new candidate was selected only for Text; Info/Doc development did not support a general upgrade.'),
        ('second_model_adaptation','Current actual research cutoff is M1; cross-model generalization remains next-stage work.'),
        ('controlled_serving_speed','Instrumented quality runs do not substitute for a controlled speed/HBM/throughput experiment.'),
        ('HEAD_tier_new_format','Historical exchange and physical-budget analysis only; no new HEAD-tier GPU implementation was run.')):
        scope['groups'].append({'group_id':name,'disposition':'NOT_RUN','cell_ids':[],'planned_cell_names':[],'quality':None,'reason':reason})
    scope['research_decision']={'schema':'trajkv.round13.research_cutoff_decision.v1','status':'CLOSED',
        'reason':'Completed all actual development, bounded expansions, R references and the frozen P Text500 comparison. Preserve the useful low-rate point with its uncertainty; defer uncompetitive or untested branches.',
        'independent_confirmation_claimed':False,'required_group_ids':[g['group_id'] for g in scope['groups']],
        'required_qa0_ids':[x['qa0_id'] for x in scope['qa0']],
        'required_attempt_ids':[x['attempt_id'] for x in scope['attempts']],
        'plan_bindings':comp['scope_plan_bindings'],
        'unused_schedule_context':'Original15 full-cell configs and old sequential PC schedule were replaced before execution; source_context bindings carry zero new inference.'}
    snapshot=a.output/'round13_root_finalize_executed.py';snapshot.write_bytes(Path(__file__).read_bytes())
    add_file(scope,snapshot,'ops/round13_root_finalize_v1.py','code')
    for name in ('SUPPORT_PRECISION_MATH_AND_NEXT_v1.zh.md','SUPPORT_PRECISION_MATH_VERIFICATION_v1.json'):
        extra=Path('${EVIDENCE_ROOT}/results/round13/next_design_v1')/name
        add_file(scope,extra,'results/round13/next_design/'+name,'support')
    source=adapter.LocalSource(scope,(adapter.ROOT,adapter.EVIDENCE))
    with adapter.metric_modules(source) as (metric,scorer,reporter):
        admitted,totals=semantic.admit_cells(scope,source,metric,scorer)
        quality,pairs,views=semantic.recompute_views(scope,admitted,scorer,reporter)
    qa=semantic.verify_qa0(scope,source,totals);attempts=semantic.verify_attempts(scope,source,totals)
    semantic.verify_research_decision(scope,admitted,qa,attempts,totals,source)
    totals['QA0_plus_attempt_native_prefix_forwards']=totals['QA0_native_prefix_model_forwards']+totals['attempt_native_prefix_model_forwards']
    totals['QA0_plus_attempt_model_TF_forwards']=totals['QA0_model_TF_forwards']+totals['attempt_model_TF_forwards']
    totals['QA0_plus_attempt_model_forwards']=totals['QA0_native_prefix_model_forwards']+totals['QA0_model_TF_forwards']+totals['attempt_model_forwards']
    adapter.require(totals['new_successful_terminal_answers']==1396 and totals['new_failure_status_rows']==0,'Root actual final1396 ledger differs')
    adapter.require(totals['r13_development_method_successful_terminal_answers']==864 and totals['r13_benchmark_successful_terminal_answers']==500
        and totals['r12_reference_successful_terminal_answers']==32,'Root development/reference/benchmark counts differ')
    summary={'schema':'trajkv.round13.Root_final_scalar_report.v1','quality':quality,'paired':pairs,'counts':dict(totals)}
    scalar=a.output/'FINAL_SCALAR_REPORT_v1.json';write(scalar,summary)
    scope['report_summary']='results/round13/FINAL_SCALAR_REPORT_v1.json';add_file(scope,scalar,scope['report_summary'],'report')
    table=a.output/'FINAL_METHOD_DOMAIN_TABLE_v1.csv'
    with table.open('x',encoding='utf-8',newline='') as f:
        fields=['phase','method','dataset','n','score_percent','ci95_percent','mean_bits','max_bits','weighted_bits','mean_visual_bytes','mean_encode_ms','mean_wall_ms','cap_violations']
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in quality:
            w.writerow({'phase':r['evaluation_phase'],'method':r['label'],'dataset':r['dataset'],'n':r['questions'],
                'score_percent':100*r['primary_mean'],'ci95_percent':json.dumps([100*x for x in r['primary_ci95']]),
                'mean_bits':resource(r,'actual_effective_visual_bits'),'max_bits':r['actual_effective_visual_bits']['max'],
                'weighted_bits':r['element_weighted_effective_visual_bits'],'mean_visual_bytes':resource(r,'stored_visual_KV_bytes'),
                'mean_encode_ms':resource(r,'encode_ms'),'mean_wall_ms':resource(r,'wall_ms'),'cap_violations':r['cap_violations']})
    add_file(scope,table,'results/round13/FINAL_METHOD_DOMAIN_TABLE_v1.csv','quality_table')
    paired=a.output/'FINAL_PAIRED_v1.json';write(paired,pairs);add_file(scope,paired,'results/round13/FINAL_PAIRED_v1.json','report')
    report=a.output/f'REPORT_ROUND13_FINAL_{a.version}.zh.md';textfile(report,human_report(quality,pairs,totals))
    add_file(scope,report,'docs/'+report.name,'report')
    scope['expected_accounting']=dict(totals)
    semantic.verify_report(summary,quality,pairs,totals)
    # Check the final report additions against the same explicit role/path
    # inventory contract before issuing either identity seal.
    adapter.LocalSource(scope,(adapter.ROOT,adapter.EVIDENCE))
    scope['stage_state']='ROOT_SEALED'
    scope['root_seal']={'schema':'trajkv.round13.root_seal.v1','created_hkt':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'source_payload_sha256':adapter.frozen.canonical(adapter.payload(scope)),
        'public_payload_sha256':adapter.frozen.canonical(Sanitizer().tree(adapter.payload(scope))),
        'Root_finalizer_source_sha256':adapter.sha(snapshot),'semantic_verifier_sha256':adapter.sha(semantic.__file__),
        'actual_new_answers':1396,'whole_external_library_completion_claimed':False}
    sealed=a.output/'ROUND13_SCOPE_SEALED_v1.json';write(sealed,scope)
    receipt={'status':'PASS_ROOT_RECOMPUTED_1396_AND_SEALED_CUTOFF','counts':dict(totals),'quality_views':len(quality),'paired_comparisons':len(pairs),
        'component_sha256':adapter.sha(a.components),'sealed_scope_sha256':adapter.sha(sealed),'report_sha256':adapter.sha(report),
        'new_GPU_or_model_calls':0,'GitHub_publication_performed':False,'wall_seconds':time.perf_counter()-started}
    write(a.output/'ROOT_CUTOFF_RECEIPT_v1.json',receipt)
    user=adapter.ROOT.parent/'TrajKV/docs'/report.name
    if user.exists():adapter.require(user.read_bytes()==report.read_bytes(),'preserve prior user report')
    else:user.write_bytes(report.read_bytes())
    print(json.dumps({'status':receipt['status'],'scope':str(sealed),'report':str(user),'counts':dict(totals),'seconds':receipt['wall_seconds']}))
if __name__=='__main__':main()
