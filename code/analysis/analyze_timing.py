"""Independent offline audit and figures for fixed-token quiet timing.

Reads raw timing JSONL, not only the producer's summary. No model is loaded,
no GPU is used, and no answer-quality score is calculated.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


MIB=2**20
LABELS={'full:base':'Full FP16','base2:base':'Base 2-bit','native4:base':'Native 4-bit',
        'asym:uniform':'Uniform','asym:periodic':'Periodic',
        'asym:table_current':'Current table','asym:table_future':'Future table',
        'asym:probe_l7_t0.02':'L7 probe (0.02)','asym:margin_write':'Margin write'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def stats(values):
    values=[float(value) for value in values if value is not None]
    return dict(n=len(values),median=statistics.median(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None,
                values=values)


def signature(turns):
    payload=[{key:turn[key] for key in ('token_ids','forced_chat_closure','logical_end')} for turn in turns]
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def recalculate_reads(events):
    fields=('base_visual_bytes','residual_bytes','text_bytes','reference_bytes')
    result={key:sum(int(row.get(key,0)) for row in events) for key in fields}
    result['total_logical_read_bytes']=sum(result.values())
    result['probe_total_bytes']=sum(sum(int(row.get(key,0)) for key in fields) for row in events if row['pass_kind']=='probe')
    result['replay_or_repair_total_bytes']=sum(sum(int(row.get(key,0)) for key in fields) for row in events if row['pass_kind'] in ('replay','repair'))
    result['probe_base_text_extra_bytes']=sum(int(row.get('base_visual_bytes',0))+int(row.get('text_bytes',0))
                                                for row in events if row['pass_kind']=='probe')
    result['replay_base_text_extra_bytes']=sum(int(row.get('base_visual_bytes',0))+int(row.get('text_bytes',0))
                                                 for row in events if row['pass_kind'] in ('replay','repair'))
    result['extra_logical_read_bytes']=(result['residual_bytes']+result['probe_base_text_extra_bytes']+
                                         result['replay_base_text_extra_bytes'])
    return result


def weighted_interval_ms(records):
    numerator=0.
    denominator=0
    for record in records:
        for turn in record['runner_result']['turns']:
            count=max(len(turn['token_ids'])-1,0)
            if count:
                numerator+=count*turn['mean_inter_token_ms']
                denominator+=count
    return numerator/denominator if denominator else None


def audit_inputs(folder):
    complete=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    records=jsonl(folder/'records.jsonl')
    warmups=jsonl(folder/'warmup_records.jsonl')
    allocator=jsonl(folder/'allocator_audits.jsonl')
    canonical=jsonl(folder/'canonical_full.jsonl')
    schedule=json.loads((folder/'schedule.json').read_text(encoding='utf-8'))
    contract=manifest['contract']
    ids=contract['selected_ids']
    configs=[c['configuration'] for c in contract['configurations']]
    repeats=contract['repeats']
    expected=len(ids)*len(configs)*repeats
    issues=[]
    def check(condition,kind,detail):
        if not condition:issues.append(dict(kind=kind,detail=detail))
    check(complete.get('measured_trials')==expected and complete.get('expected_measurements')==expected,
          'complete_count',complete)
    check(complete.get('quality')=='not_scored','complete_quality',complete)
    expected_records={(sid,config,repeat) for sid in ids for config in configs for repeat in range(repeats)}
    record_keys=[(r['session_id'],r['configuration'],r['repeat']) for r in records]
    check(len(records)==expected and len(set(record_keys))==expected and set(record_keys)==expected_records,
          'measurement_coverage',dict(actual=len(records),unique=len(set(record_keys)),expected=expected))
    warmup_keys=[(r['session_id'],r['configuration'],r['repeat']) for r in warmups]
    expected_warm={(sid,config,repeat) for sid in ids for config in configs for repeat in range(contract['warmups'])}
    check(len(warmup_keys)==len(expected_warm) and set(warmup_keys)==expected_warm,
          'warmup_coverage',dict(actual=len(warmups),expected=len(expected_warm)))
    planned={r['trial_id']:r for r in schedule}
    check([r['trial_id'] for r in records]==[r['trial_id'] for r in schedule if r['phase']=='measurement'],
          'measurement_schedule_order','observed records must follow the frozen randomized schedule')
    check([r['trial_id'] for r in warmups]==[r['trial_id'] for r in schedule if r['phase']=='warmup'],
          'warmup_schedule_order','warmups are independently randomized and separated')
    canon={r['session_id']:r for r in canonical}
    check(set(canon)==set(ids),'canonical_coverage',list(canon))
    audit_by_id={a['audit_id']:a for a in allocator}
    check(len(audit_by_id)==len(allocator),'allocator_unique_ids',len(allocator))
    audit_pairs=[(a['session_id'],a['configuration']) for a in allocator]
    check(len(audit_pairs)==len(ids)*len(configs) and len(set(audit_pairs))==len(audit_pairs),
          'allocator_pair_coverage',dict(actual=len(audit_pairs),expected=len(ids)*len(configs)))
    for number,(source,expected_hash) in enumerate(contract['sources'].items()):
        name=source.replace('\\','/').split('/')[-1]
        snapshot=folder/'source_snapshot'/f'{number:02d}_{name}'
        check(snapshot.exists() and digest(snapshot)==expected_hash,'source_snapshot_hash',str(snapshot))
    quiet_count=token_count=ledger_count=budget_count=0
    for r in records+warmups:
        key=r['trial_id']
        schedule_row=planned.get(key,{})
        check(all(r.get(field)==schedule_row.get(field) for field in ('phase','repeat','session_id','configuration','method','policy')),
              'schedule_identity',key)
        source=canon[r['session_id']]['canonical_full']['turns']
        actual=r['runner_result']['turns']
        identical=(len(actual)==len(source) and all(all(a[field]==b[field] for field in ('token_ids','forced_chat_closure','logical_end'))
                                                            for a,b in zip(actual,source)))
        check(identical,'actual_tokens_positions',key)
        check(signature(actual)==r['token_identity']['sha256'],'transcript_signature',key)
        check(r.get('quality')=='not_scored' and r.get('application_accuracy') is None,
              'quality_not_scored',key)
        quiet=True
        for boundary in ('quiet_before','quiet_after'):
            q=r[boundary]
            verified=(q.get('verified') is True and q.get('quiet') is True and not q.get('foreign_processes')
                      and all(process.get('self') is True for process in q.get('processes',[])))
            quiet=quiet and verified
        check(quiet and r.get('quiet_verified') is True,'quiet_boundaries',key)
        reads=recalculate_reads(r['runner_result']['read_events'])
        ledger=all(r['logical_reads'].get(field)==reads[field] for field in
                   ('base_visual_bytes','residual_bytes','text_bytes','reference_bytes',
                    'total_logical_read_bytes','probe_total_bytes','replay_or_repair_total_bytes','extra_logical_read_bytes'))
        ledger=ledger and reads['extra_logical_read_bytes']==r['runner_result']['extra_logical_read_bytes']
        check(ledger and r.get('read_ledger_exact') is True,'actual_call_byte_ledger',key)
        budget=contract.get('read_budget_bytes')
        within=budget is None or 0<=reads['extra_logical_read_bytes']<=budget
        check(within and r.get('read_budget_ok') is True,'read_budget',key)
        embedded=r['memory_allocator_audit']
        check(audit_by_id.get(embedded['audit_id'])==embedded,'allocator_embedded_identity',key)
        check(embedded['session_id']==r['session_id'] and embedded['configuration']==r['configuration'],
              'allocator_pair_identity',key)
        recomputed_it=weighted_interval_ms([r])
        saved_it=r['timing']['mean_inter_token_ms']
        check((recomputed_it is None and saved_it is None) or (recomputed_it is not None and saved_it is not None and abs(recomputed_it-saved_it)<1e-9),
              'inter_token_weighting',key)
        if r['phase']=='measurement':
            token_count+=int(identical);quiet_count+=int(quiet);ledger_count+=int(ledger);budget_count+=int(within)
    for a in allocator:
        before=a['before_constructor'];after=a['after_constructor'];base=a['model_only_before_runner']
        check(a['owned_unique_bytes']==a['bank_owned_after']['total_unique'],'owner_unique_identity',a['audit_id'])
        for field in ('allocated','reserved'):
            check(a[f'constructor_{field}_change_bytes']==after[field+'_bytes']-before[field+'_bytes'],
                  'constructor_allocator_difference',a['audit_id']+':'+field)
            check(a[f'steady_{field}_increment_from_model_only_bytes']==after[field+'_bytes']-base[field+'_bytes'],
                  'post_constructor_model_baseline_difference',a['audit_id']+':'+field)
        check(a['session_peak_minus_model_only_bytes']==a['session_peak_allocated_bytes']-base['allocated_bytes'],
              'peak_model_baseline_difference',a['audit_id'])
        check(a.get('collection_phase')=='separate unmeasured warmup/dry repeat','allocator_nonmeasurement_phase',a['audit_id'])
        source_trial=a['audit_id'].removeprefix('allocator_')
        if source_trial in planned:
            check(planned[source_trial]['phase']=='warmup','allocator_warmup_source',a['audit_id'])
    producer=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    check(producer.get('quality')=='not_scored' and producer.get('warmups_included') is False,
          'producer_summary_scope','quality and warmup exclusion')
    for group in producer['configurations']:
        subset=[r for r in records if r['configuration']==group['configuration']]
        check(group['measured_trials']==len(subset) and group['expected_trials']==len(ids)*repeats,
              'producer_summary_counts',group['configuration'])
        for entry in group['per_session']:
            selected=[r for r in subset if r['session_id']==entry['session_id']]
            expected_stats=stats([r['timing']['session_ms'] for r in selected])
            check(all(entry['session_ms'][key]==expected_stats[key] for key in ('n','median','minimum','maximum')),
                  'producer_summary_session_stats',group['configuration']+':'+entry['session_id'])
        for total in group['repeat_totals']:
            selected=[r for r in subset if r['repeat']==total['repeat']]
            check(abs(total['session_ms_sum']-sum(r['timing']['session_ms'] for r in selected))<1e-9,
                  'producer_summary_repeat_totals',group['configuration']+':'+str(total['repeat']))
    facts=dict(measurements=len(records),expected_measurements=expected,warmups=len(warmups),
               allocator_audits=len(allocator),canonical_sessions=len(canonical),
               actual_token_identical_measurements=token_count,quiet_verified_measurements=quiet_count,
               independently_reconciled_ledgers=ledger_count,within_budget_measurements=budget_count,
               quality='not_scored',paid_calls=complete.get('paid_calls'),issues=issues,
               producer_summary_independently_checked=True,
               status='PASS' if not issues else 'FAIL')
    return manifest,records,warmups,allocator,canonical,facts


def summarize(manifest,records,allocator,canonical):
    contract=manifest['contract'];ids=contract['selected_ids'];repeats=contract['repeats']
    rows=[];session_rows=[]
    for config in contract['configurations']:
        name=config['configuration']
        selected=[r for r in records if r['configuration']==name]
        audits=[a for a in allocator if a['configuration']==name]
        round_rows=[]
        for repeat in range(repeats):
            current=[r for r in selected if r['repeat']==repeat]
            reads=[recalculate_reads(r['runner_result']['read_events']) for r in current]
            round_rows.append(dict(repeat=repeat,sessions=len(current),
                four_session_ms=sum(r['timing']['session_ms'] for r in current),
                mean_first_ttft_ms=statistics.mean(r['timing']['first_turn_ttft_ms'] for r in current),
                weighted_inter_token_ms=weighted_interval_ms(current),
                mean_extra_read_mib=statistics.mean(x['extra_logical_read_bytes']/MIB for x in reads),
                mean_base_visual_read_mib=statistics.mean(x['base_visual_bytes']/MIB for x in reads),
                mean_text_read_mib=statistics.mean(x['text_bytes']/MIB for x in reads),
                mean_probe_total_read_mib=statistics.mean(x['probe_total_bytes']/MIB for x in reads),
                mean_replay_total_read_mib=statistics.mean(x['replay_or_repair_total_bytes']/MIB for x in reads),
                mean_residual_read_mib=statistics.mean(x['residual_bytes']/MIB for x in reads),
                mean_probe_base_text_extra_mib=statistics.mean(x['probe_base_text_extra_bytes']/MIB for x in reads),
                mean_replay_base_text_extra_mib=statistics.mean(x['replay_base_text_extra_bytes']/MIB for x in reads)))
        for sid in ids:
            subset=sorted((r for r in selected if r['session_id']==sid),key=lambda x:x['repeat'])
            session_rows.append(dict(configuration=name,session_id=sid,
                                     session_ms=stats([r['timing']['session_ms'] for r in subset]),
                                     first_ttft_ms=stats([r['timing']['first_turn_ttft_ms'] for r in subset]),
                                     inter_token_ms=stats([r['timing']['mean_inter_token_ms'] for r in subset]),
                                     peak_allocated_bytes=stats([r['memory']['peak_allocated_bytes'] for r in subset])))
        probes=[r['runner_result'].get('intra_step_probe_summary') for r in selected]
        probes=[p for p in probes if p is not None]
        row=dict(configuration=name,label=LABELS.get(name,name),rounds=round_rows,
                 four_session_ms=stats([r['four_session_ms'] for r in round_rows]),
                 first_ttft_ms=stats([r['mean_first_ttft_ms'] for r in round_rows]),
                 inter_token_ms=stats([r['weighted_inter_token_ms'] for r in round_rows]),
                 later_turn_ttft_ms=stats([t['ttft_ms'] for r in selected for t in r['timing']['per_turn'][1:]]),
                 extra_read_mib=stats([r['mean_extra_read_mib'] for r in round_rows]),
                 base_visual_read_mib=stats([r['mean_base_visual_read_mib'] for r in round_rows]),
                 text_read_mib=stats([r['mean_text_read_mib'] for r in round_rows]),
                 probe_total_read_mib=stats([r['mean_probe_total_read_mib'] for r in round_rows]),
                 replay_total_read_mib=stats([r['mean_replay_total_read_mib'] for r in round_rows]),
                 residual_read_mib=stats([r['mean_residual_read_mib'] for r in round_rows]),
                 probe_base_text_extra_mib=stats([r['mean_probe_base_text_extra_mib'] for r in round_rows]),
                 replay_base_text_extra_mib=stats([r['mean_replay_base_text_extra_mib'] for r in round_rows]),
                 owner_mib=stats([a['owned_unique_bytes']/MIB for a in audits]),
                 allocated_increment_mib=stats([a['steady_allocated_increment_from_model_only_bytes']/MIB for a in audits]),
                 reserved_increment_mib=stats([a['steady_reserved_increment_from_model_only_bytes']/MIB for a in audits]),
                 peak_minus_model_mib=stats([a['session_peak_minus_model_only_bytes']/MIB for a in audits]),
                 measured_peak_allocated_mib=stats([r['memory']['peak_allocated_bytes']/MIB for r in selected]),
                 visual_payload_ratio=stats([a['bank_owned_after']['packed_visual_ratio'] for a in audits]),
                 visual_owner_mib=stats([(a['bank_owned_after']['visual_base']+a['bank_owned_after']['visual_residual'])/MIB for a in audits]),
                 text_capacity_mib=stats([a['bank_owned_after']['text_capacity']/MIB for a in audits]),
                 probe_events=sum(p['events'] for p in probes),executed_probes=sum(p['probes'] for p in probes),
                 probe_upgrades=sum(p['upgrades'] for p in probes),
                 probe_skipped_events=sum(len(r['runner_result'].get('intra_step_probe_skipped',[])) for r in selected))
        row['executed_probe_relative_l2']=stats([event['decision']['relative_l2'] for r in selected
            for event in r['runner_result'].get('intra_step_probes',[]) if event.get('probe_observed')])
        rows.append(row)
    by={r['configuration']:r for r in rows}
    comparisons=[]
    if 'asym:probe_l7_t0.02' in by:
        probe=by['asym:probe_l7_t0.02']
        for name in ('asym:periodic','asym:table_current','asym:table_future','full:base'):
            if name not in by:continue
            base=by[name]
            comparisons.append(dict(method='asym:probe_l7_t0.02',reference=name,
                median_four_session_time_ratio=probe['four_session_ms']['median']/base['four_session_ms']['median'],
                median_four_session_time_change_percent=100*(probe['four_session_ms']['median']/base['four_session_ms']['median']-1),
                paired_round_time_ratios=[a['four_session_ms']/b['four_session_ms'] for a,b in zip(probe['rounds'],base['rounds'])],
                median_first_ttft_change_percent=100*(probe['first_ttft_ms']['median']/base['first_ttft_ms']['median']-1),
                median_inter_token_change_percent=100*(probe['inter_token_ms']['median']/base['inter_token_ms']['median']-1)))
    return dict(schema='trajkv.independent_timing_analysis.v1',quality='not_scored',
                gpu=manifest['gpu'],sessions=ids,turns=contract['turns'],max_tokens=contract['max_tokens'],
                visual_tokens=[r['canonical_full']['visual_tokens'] for r in canonical],
                output_tokens=[sum(len(t['token_ids']) for t in r['canonical_full']['turns']) for r in canonical],
                read_budget_bytes=contract.get('read_budget_bytes'),warmups=contract['warmups'],repeats=repeats,
                configurations=rows,per_session=session_rows,probe_comparisons=comparisons,
                aggregation='one round sums the four session times; table/figure show median and full range across three measured rounds',
                inter_token_scope='interval-count-weighted means, includes EOS and prototype instrumentation; not raw latency quantiles or pure kernel time',
                memory_scope='unique owned bank storages, synchronized post-constructor allocator increment, and whole-session peak are distinct; allocator audit is unmeasured warmup',
                inference_limit='four short 256-visual-token development images, fixed Full output strings; no task quality, throughput, long-context or serving-steady claim')


def figure(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':11.5,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.2,
                         'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    rows=summary['configurations'];n=len(rows);ys=list(range(n))
    colors=['#767676' if r['configuration']=='full:base' else '#CFCECE' if not r['configuration'].startswith('asym:')
            else '#0F4D92' if ':probe_' in r['configuration'] else '#42949E' if 'table_future' in r['configuration']
            else '#3775BA' if 'table_current' in r['configuration'] else '#E9A6A1' for r in rows]
    fig,axes=plt.subplots(2,2,figsize=(17.5,11.6),gridspec_kw={'width_ratios':[1,1]})
    def horizontal(ax,key,scale=1.):
        med=[r[key]['median']/scale for r in rows]
        ax.barh(ys,med,color=colors,edgecolor='#272727',linewidth=.7,height=.62)
        for y,r in zip(ys,rows):
            values=[v/scale for v in r[key]['values']]
            ax.plot([min(values),max(values)],[y,y],color='#272727',lw=1.3)
            for offset,value in zip([-.12,0,.12],values):
                ax.scatter(value,y+offset,s=15,color='white',edgecolor='#272727',linewidth=.6,zorder=4)
        ax.set_yticks(ys,[r['label'] for r in rows]);ax.invert_yaxis()
        ax.set_xlim(left=0);ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
    horizontal(axes[0,0],'four_session_ms',1000.)
    axes[0,0].set_title('a  Same-output workload time',loc='left',fontweight='bold')
    axes[0,0].set_xlabel('Four-session time (s); dots: three measured rounds')
    residual=[r['residual_read_mib']['median'] for r in rows]
    probe=[r['probe_base_text_extra_mib']['median'] for r in rows]
    replay=[r['replay_base_text_extra_mib']['median'] for r in rows]
    ax=axes[0,1]
    ax.barh(ys,residual,color='#3775BA',edgecolor='#272727',linewidth=.7,height=.62,label='All residual reads')
    ax.barh(ys,probe,left=residual,color='#E9A6A1',edgecolor='#272727',linewidth=.7,height=.62,hatch='//',label='Extra probe base/text reads')
    ax.barh(ys,replay,left=[a+b for a,b in zip(residual,probe)],color='#B64342',height=.62,label='Extra replay base/text reads')
    if summary['read_budget_bytes'] is not None:
        ax.axvline(summary['read_budget_bytes']/MIB,color='#767676',ls='--',lw=1,label='Read cap')
    ax.set_yticks(ys,[r['label'] for r in rows]);ax.invert_yaxis();ax.set_xlim(left=0)
    ax.set_title('b  Extra logical reads',loc='left',fontweight='bold')
    ax.set_xlabel('MiB per session; metadata included; not HBM traffic')
    ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
    handles,labels=ax.get_legend_handles_labels()
    if not any(replay):
        keep=[i for i,label in enumerate(labels) if label!='Extra replay base/text reads']
        handles=[handles[i] for i in keep];labels=[labels[i] for i in keep]
    read_handles,read_labels=handles,labels
    names=['full:base','base2:base','native4:base','asym:periodic']
    memory=[next(r for r in rows if r['configuration']==name) for name in names if any(r['configuration']==name for r in rows)]
    ax=axes[1,0]
    for offset,key,color,hatch,label in [(-.24,'owner_mib','#3775BA','', 'Unique bank owners (incl. text reserve)'),
                                         (0,'allocated_increment_mib','#8BCF8B','//','Allocator: post-constructor minus model'),
                                         (.24,'peak_minus_model_mib','#E9A6A1','..','Allocator: session peak minus model')]:
        ax.barh([i+offset for i in range(len(memory))],[r[key]['median'] for r in memory],height=.21,
                color=color,edgecolor='#272727',linewidth=.7,hatch=hatch,label=label)
    ax.set_yticks(range(len(memory)),['Full FP16','Base 2-bit','Native 4-bit','Asymmetric' ][:len(memory)])
    ax.invert_yaxis();ax.set_xlim(left=0);ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
    ax.set_title('c  Distinct memory accounting scopes',loc='left',fontweight='bold')
    ax.set_xlabel('MiB above / outside resident model; common prefill affects peak')
    memory_handles,memory_labels=ax.get_legend_handles_labels()
    horizontal(axes[1,1],'inter_token_ms')
    axes[1,1].set_title('d  Instrumented inter-token interval',loc='left',fontweight='bold')
    axes[1,1].set_xlabel('Weighted mean interval (ms); includes prototype overhead')
    fig.suptitle('Quiet single-GPU timing: four development images, identical Full-generated tokens',fontsize=15,fontweight='bold')
    fig.legend(read_handles,read_labels,loc='lower center',bbox_to_anchor=(.5,.078),ncol=3,fontsize=10)
    fig.legend(memory_handles,memory_labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,fontsize=10)
    fig.text(.5,.02,'10 turns/image; 256 visual tokens; 1 warmup + 3 measured repeats/config/image. Quality: not_scored. No throughput or long-context inference.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.135,1,.96),pad=1.7)
    for suffix in ('png','pdf'):
        fig.savefig(output/f'timing_cost_comparison.{suffix}',dpi=300,facecolor='white')
    plt.close(fig)


def report(summary,audit,folder):
    text=['# 独立计时复核：四张开发图像的固定 token 成本','',
          f"数据源：`{folder}`。独立复核 **{audit['status']}**；{audit['measurements']} 次测量、{audit['warmups']} 次 warmup、{audit['allocator_audits']} 份 allocator 审计、4 条 Full 轨迹。",'',
          f"全部 {audit['actual_token_identical_measurements']} 次测量的实际 token、闭合符和逻辑位置与 Full 轨迹一致；{audit['quiet_verified_measurements']} 次通过运行前后跨物理 GPU compute-process 安静检查；{audit['independently_reconciled_ledgers']} 份实际调用字节账本独立对平，{audit['within_budget_measurements']} 次在预算内。quality 始终为 **not_scored**。",'',
          '## 范围与聚合','',
          f"GPU：{summary['gpu']}。四张 CLEVR 开发图像，各 {summary['turns']} 轮、256 个视觉 token；Full 生成的输出 token 数依次为 {summary['output_tokens']}，包含 EOS。每配置/图像先 1 次 warmup，再 3 次测量；warmup 完全排除。",'',
          '下表先在每个测量轮中对四会话求时间总和，再报告三轮的中位数和完整范围。TTFT 是四会话首轮 TTFT 的轮内均值，再跨三轮汇总；inter-token 以实际区间数加权。范围是三次实测范围，不是置信区间。','',
          '| 配置 | 四会话总时间 s，中位数 [min,max] | 首轮 TTFT ms，中位数 | inter-token ms，中位数 | 额外逻辑读取 MiB/会话 |',
          '|---|---:|---:|---:|---:|']
    for r in summary['configurations']:
        s=r['four_session_ms']
        text.append(f"| {r['label']} | {s['median']/1000:.4f} [{s['minimum']/1000:.4f}, {s['maximum']/1000:.4f}] | {r['first_ttft_ms']['median']:.3f} | {r['inter_token_ms']['median']:.3f} | {r['extra_read_mib']['median']:.3f} |")
    text+=['','## Probe 与简单控制','']
    for c in summary['probe_comparisons']:
        text.append(f"- Probe 相对 {LABELS.get(c['reference'],c['reference'])} 的四会话总时间中位数变化为 **{c['median_four_session_time_change_percent']:+.2f}%**；inter-token 中位数变化为 {c['median_inter_token_change_percent']:+.2f}%。这是本组固定轨迹上的成本差，不能据此判断答案质量。")
    probe=next((r for r in summary['configurations'] if ':probe_' in r['configuration']),None)
    if probe:
        text+=['',f"Probe 的执行探针/升级数为 {probe['executed_probes']}/{probe['probe_upgrades']}（跨 12 个测量会话实例），总 forward event 为 {probe['probe_events']}，预算跳过 {probe['probe_skipped_events']} 次。已执行探针全部触发，不能把预算截断当成阈值具有选择性。",
               f"其额外读取由全部残差读取 {probe['residual_read_mib']['median']:.3f} MiB/会话与探针额外 base/text 读取 {probe['probe_base_text_extra_mib']['median']:.3f} MiB/会话组成。probe 总读取本身也含残差，不能再次与残差总项相加。"]
        signal=probe['executed_probe_relative_l2']
        text.append(f"已执行探针的纠偏 relative-L2 范围为 {signal['minimum']:.4f}–{signal['maximum']:.4f}（中位数 {signal['median']:.4f}），均超过固定阈值 0.02。它是 base/residual 差，不是 FP16 真实误差。")
    text+=['','## 三种内存口径','',
           '| 配置 | 视觉 payload / FP16 | Unique bank owner MiB | 构造后 allocated−model MiB | 构造后 reserved−model MiB | 会话 peak allocated−model MiB |',
           '|---|---:|---:|---:|---:|---:|']
    for r in summary['configurations']:
        text.append(f"| {r['label']} | {100*r['visual_payload_ratio']['median']:.2f}% | {r['owner_mib']['median']:.3f} | {r['allocated_increment_mib']['median']:.3f} | {r['reserved_increment_mib']['median']:.3f} | {r['peak_minus_model_mib']['median']:.3f} |")
    text+=['',
           '- Unique owner 是 bank 的实际去重存储，包含常驻视觉表示、文本预留容量和位置；视觉 payload 百分比不能替代整个 KV owner 的百分比。',
           '- 构造后 allocator 增量在独立非计时 warmup 中同步采样，包含该时点仍存活的 setup 张量；它不是严格的 bank-only 量，也不等于最终服务稳态。reserved 是 allocator 保留池，不能当作活跃张量。',
           '- bank.owned 不包含后续 repair_start 快照与临时 workspace；整会话 peak 包含首次视觉 prefill、编码、probe/replay 与临时工作区。此次相同 peak 不能被改写成压缩降低了峰值。',
           '- 三次正式计时没有构造包装；allocator 审计来自另外的 warmup。每条正式记录链接原始 allocator audit ID。',
           '', '## 可支持的结论与限制','',
           '本数据支持：在此四图、短上下文、同输出协议与当前原型中，压缩减少了视觉表示和 bank owner 的部分驻留字节，但没有观察到相对 Full 的会话时间优势；probe 没有比 periodic 更快。首轮 TTFT 还包含输入准备、视觉 prefill 和量化/残差编码，不能当作 decode 稳态开销。',
           'inter-token 指标仍包含模型前向、Python 调度、margin 读取、probe 双读与同步等原型开销；没有逐 token 原始分布，不报告 P95，也不把它当作纯 kernel 时间。',
           '四张已用于开发的图像不足以支持总体显著性、吞吐、多用户负载、长视觉上下文、跨模型或部署收益。强制输出 Full 的文本不代表压缩方法自然生成时答对了这些题；本报告不计算或推断任何应用准确率。',
           '', '## 文件','',
           '- `audit_checks.json`：独立原始证据复核。',
           '- `independent_summary.json`：全部三重复、逐会话 med/range 与三口径内存。',
           '- `configuration_table.csv`：主表机器可读版本。',
           '- `timing_cost_comparison.png` / `.pdf`：成本对照图。']
    return '\n'.join(text)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();folder=args.input.resolve();output=(args.output or folder/'independent_analysis').resolve()
    output.mkdir(parents=True,exist_ok=True)
    manifest,records,warmups,allocator,canonical,audit=audit_inputs(folder)
    (output/'audit_checks.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    if audit['issues']:
        raise RuntimeError('Independent timing audit failed; see audit_checks.json')
    summary=summarize(manifest,records,allocator,canonical)
    (output/'independent_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    figure(summary,output)
    (output/'TIMING_INDEPENDENT_REPORT.zh.md').write_text(report(summary,audit,folder),encoding='utf-8')
    fields=['configuration','four_session_median_s','four_session_min_s','four_session_max_s','first_ttft_median_ms',
            'inter_token_median_ms','base_visual_read_mib','residual_read_mib','probe_total_read_mib','replay_total_read_mib',
            'extra_read_mib','unique_owner_mib','post_constructor_allocated_increment_mib',
            'post_constructor_reserved_increment_mib','session_peak_minus_model_mib','visual_payload_ratio']
    with (output/'configuration_table.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for r in summary['configurations']:
            writer.writerow(dict(configuration=r['configuration'],four_session_median_s=r['four_session_ms']['median']/1000,
                four_session_min_s=r['four_session_ms']['minimum']/1000,four_session_max_s=r['four_session_ms']['maximum']/1000,
                first_ttft_median_ms=r['first_ttft_ms']['median'],inter_token_median_ms=r['inter_token_ms']['median'],
                base_visual_read_mib=r['base_visual_read_mib']['median'],residual_read_mib=r['residual_read_mib']['median'],
                probe_total_read_mib=r['probe_total_read_mib']['median'],replay_total_read_mib=r['replay_total_read_mib']['median'],
                extra_read_mib=r['extra_read_mib']['median'],unique_owner_mib=r['owner_mib']['median'],
                post_constructor_allocated_increment_mib=r['allocated_increment_mib']['median'],
                post_constructor_reserved_increment_mib=r['reserved_increment_mib']['median'],
                session_peak_minus_model_mib=r['peak_minus_model_mib']['median'],visual_payload_ratio=r['visual_payload_ratio']['median']))
    provenance=dict(script_sha256=digest(Path(__file__)),inputs={name:digest(folder/name) for name in
        ('complete.json','manifest.json','records.jsonl','warmup_records.jsonl','allocator_audits.jsonl','canonical_full.jsonl','schedule.json','summary.json')},
        quality='not_scored',gpu_used_by_analysis=False)
    (output/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=audit['status'],measurements=len(records),quality='not_scored',output=str(output))))


if __name__=='__main__':
    main()
