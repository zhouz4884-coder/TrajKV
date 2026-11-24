"""Post-hoc development-only numeric/formatting strata and exact future costs."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent
source = ROOT/'COMPACT_ANALYSIS_ROWS_v1.jsonl'
rows = [json.loads(v) for v in source.read_text(encoding='utf-8').splitlines()]
info = {(v['method'],v['id']):v for v in rows if v['dataset']=='infovqa'}
numeric = lambda text: bool(re.fullmatch(r'[+-]?\d+(?:\.\d+)?%?', re.sub(r'[,\s]','',str(text))))
numeric_key = lambda text: re.sub(r'[,\s]','',text) if numeric(text) else None
strata, cases = [], []
for level in (True,False):
    ids = [i for (m,i),v in info.items() if m=='R' and all(numeric(t) for t in v['references']) == level]
    deltas = [info[('R',i)]['score']-info[('A',i)]['score'] for i in ids]
    strata.append({'all_refs_numeric_after_removing_commas_whitespace':level,'n':len(ids),
        'R_better':sum(v>1e-12 for v in deltas),'A_better':sum(v< -1e-12 for v in deltas),
        'R_minus_A_pp':100*statistics.mean(deltas),'posthoc_heuristic_not_semantic_annotation':True})
for (method,ident),a in info.items():
    if method!='A' or ('R',ident) not in info:
        continue
    r = info[('R',ident)]
    if a['score'] == r['score']:
        continue
    cases.append({'id':ident,'N':a['N'],'m_R':r['m'],'question':a['question'],
        'references':a['references'],'A_prediction':a['prediction'],'R_prediction':r['prediction'],
        'A_score':a['score'],'R_score':r['score'],
        'same_numeric_string_after_removing_commas_whitespace':
            numeric_key(a['prediction']) is not None and numeric_key(a['prediction'])==numeric_key(r['prediction'])})
costs = []
for n in (256,768,1024,1280,1536,1792,2304):
    target = 5486*n+155145
    m_R = min(224,(1670*n+65408)//(18*n))
    m_head = min(224,(1670*n+51200)//(18*n))
    r_bytes = (3816+18*m_R)*n+89737
    head_bytes = (3816+18*m_head)*n+89737+14208
    assert head_bytes <= target and m_head >= 43
    costs.append({'N':n,'R_slots':m_R,'proposed_headbook_slots':m_head,
        'original_R_bytes':r_bytes,'G8_budget_bytes':target,'proposed_headbook_bytes':head_bytes,
        'proposed_minus_R_bytes':head_bytes-r_bytes,'hypothetical_not_implemented':True})
result = {'schema':'trajkv.round12.additional_error_views.v1',
    'source_compact_rows_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'fixed500_outputs_accessed':False,'new_GPU_calls':0,'new_free_answers':0,
    'Info64_numeric_format_strata':strata,'Info64_A_R_different_scores':cases,
    'hypothetical_headbook_costs':costs,
    'cost_premise':'All112 G4_16 books stored FP16, shared128B replaced by14336B. New reader uses a fixed head-specific layout and the already charged4B format descriptor; any extra actual descriptor/workspace must be added if implementation needs it.',
    'headbook_m_formula':'min(224,floor((1670*N+51200)/(18*N)))',
    'selection_policy_unchanged':True,'all_new_methods_unimplemented':True}
with (ROOT/'ADDITIONAL_PATTERN_VIEWS_v1.json').open('x',encoding='utf-8',newline='\n') as f:
    json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'numeric_strata':strata,'differences':len(cases),'hypothetical_costs':costs},ensure_ascii=False))
