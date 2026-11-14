"""100 independent old-pool images; fresh64 excluded; digit-bearing QA prioritized."""
import copy,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
pool=sum([read(ROOT/'configs/data'/n) for n in ('docvqa_dev32_remote_runner.json','docvqa_confirm64_remote_runner.json','docvqa_external_eval16_remote_runner.json')],[])
pool={r['image_id']:r for r in pool};fresh={r['image_id'] for r in read(ROOT/'configs/a2/confirm64.json')}
assert not set(pool)&fresh
refs={r['id']:r['references'] for r in map(json.loads,(ROOT/'configs/data/docvqa_references.jsonl').read_text(encoding='utf-8').splitlines())}
samples=[];labels=[]
for image in sorted(pool)[:100]:
 source=pool[image];choices=list(enumerate(source['questions'][:6],1))
 def digits(t):
  target=refs[source['id']][t-1]['answer'];return bool(re.search(r'[0-9]',target if isinstance(target,str) else ' '.join(target)))
 turn,question=next(((t,q) for t,q in choices if digits(t)),choices[0]);row=copy.deepcopy(source)
 row['id']=source['id']+'_mechanism100_q'+str(turn);row['questions']=[question+'\nAnswer the question using a single word or phrase.'];row['turns']=1
 row['content']=[{'type':'image'},{'type':'text','text':row['questions'][0]}];row['split']='old_pool_mechanism_development100';row['source_session_id']=source['id'];row['source_turn']=turn
 samples.append(row);labels.append({'id':row['id'],'image_id':image,'source_id':source['id'],'source_turn':turn,'answer':refs[source['id']][turn-1]['answer'],'digit_bearing':digits(turn)})
assert len(samples)==100 and len({r['image_id'] for r in samples})==100
folder=ROOT/'configs/a2';(folder/'mechanism100.json').write_text(json.dumps(samples,indent=2,ensure_ascii=False),encoding='utf-8');(folder/'mechanism100_refs.json').write_text(json.dumps(labels,indent=2))
audit={'schema':'trajkv.mechanism100.v1','images':100,'questions':100,'digit_bearing':sum(r['digit_bearing'] for r in labels),'fresh64_overlap':0,
 'selection':'first100 old-pool images in stable content-ID order; first digit-bearing question if available, otherwise first question; no new-method outcomes used',
 'samples_sha256':hashlib.sha256((folder/'mechanism100.json').read_bytes()).hexdigest(),'role':'development/mechanism, not blind confirmation'}
(ROOT/'audit/a2/mechanism100_selection.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
