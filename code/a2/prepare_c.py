"""Controlled same-user-role visual facts/private bindings; development images only."""
from pathlib import Path
import copy,json,hashlib
ROOT=Path(__file__).resolve().parents[2]
rows=json.loads((ROOT/'configs/a2/dev8.json').read_text(encoding='utf-8'))
refs={r['id']:r['references'] for r in map(json.loads,(ROOT/'configs/data/docvqa_references.jsonl').read_text().splitlines())}
contexts=[];targets=[]
for row in rows:
    q=row['questions'][1];gold=refs[row['id']][1]['answer']
    if isinstance(gold,list):gold=gold[0]
    nonce='BIND-'+hashlib.sha256(row['id'].encode()).hexdigest()[:6].upper()
    for order in ('fact_first','secret_first'):
        r=copy.deepcopy(row);r['id']=row['id']+'_'+order;r['context_order']=order
        for content in r['content']:
            if content['type']=='text':content['text']='Examine this document. Reply only OK.'
        r['fact_question']=q
        r['fact_statement']=f'Remember this fact about the document. Question: {q} The answer is: {gold}. Reply only OK.'
        r['secret_statement']=f'Remember this private binding for our conversation: {nonce}. It is assigned by the user and is unrelated to the document. Reply only OK.'
        r['secret_question']='What is the private binding assigned by the user for this conversation? Answer only the binding.'
        contexts.append(r);targets.append({'id':r['id'],'fact':gold,'secret':nonce})
(ROOT/'configs/a2/c_contexts.json').write_text(json.dumps(contexts,indent=2,ensure_ascii=False),encoding='utf-8')
(ROOT/'configs/a2/c_references.json').write_text(json.dumps(targets,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'contexts':len(contexts),'same_role':'user','fixed_acknowledgements':'OK','label':'constructed diagnostic, not benchmark'}))
