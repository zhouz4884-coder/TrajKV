"""Summarize the four frozen Round6 comparisons from completed analyses only.

No raw inference record, unfinished run, or new answer/reference annotation is
read. Raw two-sided paired sign-flip p-values receive a new Holm correction over
exactly the four prespecified pairs separately in Doc and Text. Existing broad
exploratory-family Holm values are deliberately not reused.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PAIRS = [('r6_depth_v_pair', 'r6_depth_incumbent'),
    ('r6_depth_v_pair', 'r6_depth_RD_actual'), ('r6_depth_v_pair', 'v_impact_max'),
    ('r6_depth_v_pair', 'native4_8')]
NI_MARGIN = -.005


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, field):
    value = float(value)
    require(math.isfinite(value), 'Nonfinite ' + field)
    return value


def holm_four(pvalues):
    """Stable Holm step-down adjusted p-values, exactly one four-test family."""
    require(len(pvalues) == 4, 'Prespecified family must contain four p-values')
    values = [finite(p, 'p') for p in pvalues]
    require(all(0 <= p <= 1 for p in values), 'Invalid raw two-sided p')
    order = sorted(range(4), key=lambda index: (values[index], index))
    adjusted, previous = [None]*4, 0.
    for rank, index in enumerate(order):
        previous = min(1., max(previous, (4-rank)*values[index]))
        adjusted[index] = previous
    return adjusted


def complete_preflight(folder, domain):
    """Only final analysis summary/marker files establish eligibility here."""
    path = folder/'summary.json'
    require(path.is_file(), domain + ' analysis has no final summary; no run results will be opened')
    # Text writes its complete marker after the summary. Doc's analyzer writes
    # the final COMPLETE summary last and does not emit a separate marker.
    if domain == 'Text':
        require((folder/'complete.json').is_file(), 'Text analysis final completion marker is absent')
    marker = load(folder/'complete.json') if (folder/'complete.json').is_file() else None
    if marker is not None:
        require(marker.get('status') == 'COMPLETE', domain+' analysis completion marker is not COMPLETE')
    summary = load(path)
    require(summary.get('status') == 'COMPLETE', domain+' analysis summary is not COMPLETE')
    if marker is not None and marker.get('sessions') is not None:
        require(int(marker['sessions']) == int(summary['sessions']), domain+' completion/session count mismatch')
    return summary


def primary_rows(domain, summary, pairs, table):
    left_key, right_key = ('variant', 'baseline') if domain == 'Doc' else ('candidate', 'comparator')
    indexed = {}
    for comparison in summary['comparisons']:
        key = (comparison[left_key], comparison[right_key])
        require(key not in indexed, domain+' contains duplicate exact comparison '+str(key))
        indexed[key] = comparison
    require(all(pair in indexed for pair in pairs), domain+' lacks one or more frozen exact directional comparison pairs')
    rows=[]
    for left,right in pairs:
        source = indexed[(left,right)]
        delta,lo,hi,p = [finite(source[key],key) for key in ('delta','ci_low','ci_high','p')]
        require(-1 <= lo <= hi <= 1 and -1 <= delta <= 1 and 0 <= p <= 1, 'Invalid paired score/CI/p range')
        require(int(source['questions']) == int(summary['questions_per_variant']), domain+' paired question coverage mismatch')
        quality = 'anls' if domain=='Doc' else 'accuracy'
        require(math.isclose(delta, float(table[left][quality])-float(table[right][quality]), rel_tol=1e-9, abs_tol=1e-12),
            domain+' paired difference disagrees with same-input absolute quality')
        rows.append(dict(domain=domain,variant=left,baseline=right,metric='ANLS' if domain=='Doc' else 'official TextVQA soft accuracy',
            questions=int(source['questions']),source_clusters=int(source['images']),
            cluster_unit='source_document_id' if domain=='Doc' else 'source_image_id',
            delta=delta,delta_pp=100*delta,ci_low=lo,ci_high=hi,ci_low_pp=100*lo,ci_high_pp=100*hi,
            two_sided_p=p,holm_prespecified_four=None,holm_family_size=4,
            holm_family=domain+' frozen four primary comparisons',
            ci_contains_zero=lo <= 0 <= hi))
    for row,adjusted in zip(rows,holm_four([row['two_sided_p'] for row in rows])):
        row['holm_prespecified_four']=adjusted
        row['holm_rejects_zero_difference_at_0_05']=adjusted <= .05
    return rows


def noninferiority(domain, rows):
    row=next(x for x in rows if x['baseline']=='native4_8')
    delta,lo,hi = row['delta'],row['ci_low'],row['ci_high']
    if lo >= NI_MARGIN:
        conclusion='reported two-sided 95% CI lower bound meets the -0.5pp noninferiority yardstick'
    elif hi < NI_MARGIN:
        conclusion='reported two-sided 95% CI is entirely below the -0.5pp yardstick'
    else:
        conclusion='reported 95% CI overlaps the -0.5pp yardstick; noninferiority remains unresolved'
    return dict(domain=domain,variant=row['variant'],baseline=row['baseline'],margin=NI_MARGIN,margin_pp=-.5,
        delta=row['delta'],delta_pp=row['delta_pp'],ci_low=lo,ci_high=hi,ci_low_pp=100*lo,ci_high_pp=100*hi,
        point_estimate_meets_yardstick=delta >= NI_MARGIN,
        reported_95pct_CI_supports_noninferiority=lo >= NI_MARGIN,
        reported_95pct_CI_entirely_below_yardstick=hi < NI_MARGIN,
        interpretation=conclusion,point_estimate_alone_establishes_noninferiority=False,
        dedicated_one_sided_noninferiority_p_computed=False,equivalence_test_performed=False,
        scope='Descriptive noninferiority yardstick applied to the saved paired two-sided 95% cluster CI; the two-sided difference p is not a noninferiority p.')


def complete_owner(table_row):
    images = int(table_row['images'])
    base = int(table_row['visual_bytes_total'])
    encoder_per_image = finite(table_row['encoder_constants_bytes_per_session'], 'encoder constant bytes')
    encoder_total = encoder_per_image*images
    require(math.isclose(encoder_total,round(encoder_total),rel_tol=0,abs_tol=1e-5), 'Noninteger complete encoder-owner total')
    complete = base+int(round(encoder_total))
    bits = finite(table_row['budget_bits_including_encoder_constants'],'complete bit rate')
    require(images > 0 and complete > 0 and bits > 0, 'Invalid complete owner')
    return dict(images=images,visual_base_residual_lookup_position_bytes_total=base,
        encoder_constant_bytes_total=int(round(encoder_total)),complete_visual_bytes_total=complete,
        complete_visual_bytes_per_image=complete/images,complete_visual_bits=bits,
        inferred_full_visual_denominator_bytes=16*complete/bits)


def byte_savings(domain, table, protocol):
    full = [v['name'] for v in protocol['variants'] if v['method']=='full']
    require(len(full)==1, 'Frozen protocol does not contain exactly one Full baseline')
    pair=protocol['primary'];candidate=complete_owner(table[pair]);rows=[]
    require(table[pair].get('eligible_lowbit_frontier') is True and candidate['complete_visual_bits']<=1.8+1e-12,
        domain+' pair is not an eligible complete-owner <=1.8bit candidate')
    for baseline in ('native4_8',full[0]):
        reference=complete_owner(table[baseline])
        require(candidate['images']==reference['images'] and math.isclose(candidate['inferred_full_visual_denominator_bytes'],
            reference['inferred_full_visual_denominator_bytes'],rel_tol=1e-10,abs_tol=1e-4),
            domain+' cost controls do not share the same actual visual denominator')
        saved=reference['complete_visual_bytes_total']-candidate['complete_visual_bytes_total']
        rows.append(dict(domain=domain,variant=pair,baseline=baseline,images=candidate['images'],
            candidate_complete_visual_bits=candidate['complete_visual_bits'],baseline_complete_visual_bits=reference['complete_visual_bits'],
            candidate_complete_visual_bytes_total=candidate['complete_visual_bytes_total'],
            baseline_complete_visual_bytes_total=reference['complete_visual_bytes_total'],saved_bytes_total=saved,
            saved_bytes_per_image=saved/candidate['images'],saved_fraction=saved/reference['complete_visual_bytes_total'],
            saved_percent=100*saved/reference['complete_visual_bytes_total'],
            baseline_over_candidate_byte_ratio=reference['complete_visual_bytes_total']/candidate['complete_visual_bytes_total'],
            cost_scope='actual visual base/residual, lookup, positions and encoder constants; per-session constants counted for each image; excludes common text KV and model weights'))
    return rows


def validate_domain(domain,summary,membership,protocol,protocol_sha):
    require(membership['method_freeze_sha256']==protocol_sha, domain+' membership predates a different method freeze')
    require(membership.get('source_content_overlap')==0 and membership.get('selection_fixed_before_answers_accessed') is True,
        domain+' frozen source membership is not qualified')
    for key in ('samples_sha256','references_sha256'):
        require(summary[key]==membership[key], domain+' completed analysis uses a different frozen cohort '+key)
    require(int(summary['questions_per_variant'])==int(membership['questions']), domain+' analysis question coverage differs from frozen cohort')
    require(int(summary['images'])==int(membership['images']), domain+' analysis image coverage differs from frozen cohort')
    table={row['variant']:row for row in summary['table']}
    names={v['name'] for v in protocol['variants']}
    require(len(table)==len(summary['table']) and set(table)==names, domain+' analysis does not contain exactly all frozen variants')
    require(int(summary['variants'])==len(names) and int(summary['sessions'])==len(names)*int(membership['images']),
        domain+' completed analysis/session count differs from full frozen coverage')
    for name,row in table.items():
        require(int(row['images'])==int(membership['images']) and int(row['questions'])==int(membership['questions']), domain+' variant coverage differs: '+name)
        require(row.get('materialized_quality_port') is False, domain+' analysis contains a materialized quality port')
    if domain=='Text':
        audit=summary.get('audit',{})
        require(audit.get('status')=='PASS' and audit.get('complete_coverage') is True, 'Text completed analysis audit failed')
    return table


def csv_write(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--doc-analysis',type=Path,required=True)
    p.add_argument('--text-analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'audit/a2/round6_confirmation_v1.json')
    args=p.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite existing summary output')
    # Check that both domains have final analysis artifacts before interpreting
    # any result. No run directory or unfinished raw record is ever traversed.
    for domain,folder in [('Doc',args.doc_analysis),('Text',args.text_analysis)]:
        require((folder/'summary.json').is_file(),domain+' final summary is absent')
        if domain=='Text': require((folder/'complete.json').is_file(),'Text final analysis marker is absent')
    completed={'Doc':complete_preflight(args.doc_analysis,'Doc'),'Text':complete_preflight(args.text_analysis,'Text')}
    protocol=load(args.protocol);protocol_sha=sha(args.protocol)
    pairs=[tuple(x) for x in protocol['primary_comparisons']]
    require(pairs==EXPECTED_PAIRS and protocol['primary']=='r6_depth_v_pair', 'Frozen primary comparison protocol differs from this four-pair summary')
    membership_paths={'Doc':ROOT/'audit/a2/round6_doc54_membership.json','Text':ROOT/'audit/a2/round6_text_membership_v1.json'}
    primary,noninfer,bytes_,domains=[],[],[],{}
    for domain in ('Doc','Text'):
        summary=completed[domain];membership=load(membership_paths[domain])
        table=validate_domain(domain,summary,membership,protocol,protocol_sha)
        rows=primary_rows(domain,summary,pairs,table)
        primary.extend(rows);noninfer.append(noninferiority(domain,rows));bytes_.extend(byte_savings(domain,table,protocol))
        metric='anls' if domain=='Doc' else 'accuracy'
        domains[domain]=dict(analysis_role=summary.get('analysis_role'),metric='ANLS' if domain=='Doc' else 'official TextVQA soft accuracy',
            frozen_images=membership['images'],frozen_questions=membership['questions'],
            absolute_quality=[dict(variant=name,score=table[name][metric],score_percent=100*table[name][metric],
                ci_low=table[name]['ci_low'],ci_high=table[name]['ci_high']) for name in table],
            primary_comparisons=rows,noninferiority_to_native4=noninfer[-1])
    input_paths={'protocol':args.protocol,'doc_membership':membership_paths['Doc'],'text_membership':membership_paths['Text']}
    for domain,folder in [('doc',args.doc_analysis),('text',args.text_analysis)]:
        for filename in ('summary.json','complete.json','comparisons.csv','table.csv'):
            if (folder/filename).is_file():input_paths[domain+'_'+filename]=folder/filename
    result=dict(schema='trajkv.round6.frozen-primary-summary.v1',status='COMPLETE',primary=protocol['primary'],
        primary_comparisons=[list(x) for x in pairs],holm_family='four frozen primary comparisons separately within each domain',
        holm_source='new correction of saved raw two-sided paired sign-flip p values; broad exploratory-family adjusted p values are not reused',
        uncertainty='Saved question-weighted paired source-cluster 95% intervals; no raw-record rescoring or uncertainty recomputation',
        equivalence_claimed=False,interpretation='A CI containing zero does not establish equivalence. A point estimate within -0.5pp alone does not establish noninferiority; the native4 yardstick is reported separately for the point and CI.',
        noninferiority_margin_pp=-.5,domains=domains,comparisons=primary,noninferiority=noninfer,byte_savings=bytes_,
        input_sha256={name:sha(path) for name,path in input_paths.items()},input_paths={name:str(path) for name,path in input_paths.items()},
        summarizer_sha256=sha(__file__),raw_inference_records_read=False,method_reselection=False)
    args.output.mkdir(parents=True,exist_ok=False)
    csv_write(args.output/'primary_comparisons.csv',primary)
    csv_write(args.output/'native4_noninferiority_yardstick.csv',noninfer)
    csv_write(args.output/'complete_byte_savings.csv',bytes_)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (args.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','domains':2,'primary_comparisons_per_domain':4}))
    print(json.dumps({'status':'COMPLETE','comparisons':primary,'noninferiority':noninfer,'byte_savings':bytes_}))


if __name__=='__main__':
    main()
