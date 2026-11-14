"""Freeze image isolation and descriptive error groups before new inference."""
from pathlib import Path,PureWindowsPath
import csv, hashlib, json
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'configs/a2'; OUT.mkdir(exist_ok=True)
def load(name):return json.loads((ROOT/'configs/data'/name).read_text(encoding='utf-8'))
dev=load('docvqa_dev32_remote_runner.json')
explore=load('docvqa_external_eval16_remote_runner.json')
oldconfirm=load('docvqa_confirm64_remote_runner.json')
pool=load('docvqa_heldout_candidate_runner.json')+dev
excluded={r['image_id'] for r in dev+explore+oldconfirm}
remaining={r['id']:r for r in pool if r['image_id'] not in excluded}
new=[remaining[k] for k in sorted(remaining)][:64]
assert len(new)>=16 and not ({r['image_id'] for r in new}&excluded)
for name,rows in [('dev8',dev[:8]),('confirm64',new)]:
    for row in rows:
        row['a2_split']='development' if name=='dev8' else 'new_image_confirmation'
        if 'remote_images' not in row:
            row['remote_images']=['/path/to/storage/VLAtest/gsm8k_server_bundle/benchmark_samples/docvqa_images/'+PureWindowsPath(p).name for p in row['images']]
        if 'content' not in row:row['content']=[{'type':'image'} for _ in row['images']]+[{'type':'text','text':row['questions'][0]}]
        if name=='confirm64':
            row['source_split']=row['split'];row['split']='a2_confirmation_frozen_20261113'
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
contract={'schema':'trajkv.a2.isolation.v1','created':'2026-11-13','dev_ids':[r['id'] for r in dev[:8]],
 'confirm_ids':[r['id'] for r in new],'confirm_images':len(new),
 'excluded_images':len(excluded),'remaining_total':len(remaining),
 'historical_exposure':'isolated from first cycle; older project exposure not exhaustively audited',
 'sampling':'stable ID order after image exclusions, no new outcomes or answer labels used',
 'sha256':{n:hashlib.sha256((OUT/(n+'.json')).read_bytes()).hexdigest() for n in ('dev8','confirm64')}}
(OUT/'isolation.json').write_text(json.dumps(contract,indent=2))
folder=ROOT/'results/a2_d0';folder.mkdir(exist_ok=True)
rows=list(csv.DictReader((ROOT/'results/analysis_docvqa_confirm64_v1/fixed_score/scored_turns.csv').open(encoding='utf-8-sig')))
index={(r['session_id'],r['reference_turn'],r['variant']):r for r in rows}
cases=[]
for sid,turn in sorted({(r['session_id'],r['reference_turn']) for r in rows}):
    f,b,k=[index[(sid,turn,v)] for v in ('full:base','base2:base','kvarn2:base')]
    for metric in ('exact','score'):
        fv,bv,kv=[float(r[metric]) for r in (f,b,k)]
        group=('full_correct_base_wrong_kvarn_correct' if fv==1 and bv<1 and kv==1 else
               'full_correct_base_and_kvarn_wrong' if fv==1 and bv<1 and kv<1 else
               'full_wrong_base_better' if fv<1 and bv>fv else 'other')
        cases.append({'session_id':sid,'turn':int(turn),'metric':metric,'group':group,
                     'full':fv,'base':bv,'kvarn':kv,'full_text':f['prediction'],'base_text':b['prediction'],'kvarn_text':k['prediction']})
(folder/'old_error_cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False),encoding='utf-8')
summary={m:dict(Counter(r['group'] for r in cases if r['metric']==m)) for m in ('exact','score')}
(folder/'summary.json').write_text(json.dumps({'old_confirm_only':summary,'new_split':contract},indent=2))
print(json.dumps({'old_confirm_only':summary,'new_confirm_images':len(new),'dev_images':8}))
