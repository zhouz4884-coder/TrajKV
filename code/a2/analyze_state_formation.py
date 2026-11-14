"""Matched questions and codecs, three causal state-formation protocols."""
import argparse,csv,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/analysis'))
from analyze_external import score_docvqa
from analyze import paired,write_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
 refs={r['id']:r for r in json.loads((ROOT/'configs/a2/mechanism100_refs.json').read_text())}
 runs={'compressed_question':['mad100_e2e_s0_v1','mad100_e2e_s1_v1','mad_reference100_gpu1_v1'],
       'native_question_replay_last':['state100_native_history_v1'],'native_question_shared_first':['state100_native_first_logit_v1']}
 wanted={'rtn_k1v1','base2_reference','native_k2v1','mad_k1_vmadc','rtn_k1_vmadc_headgain','fp16_reference100'}
 groups=defaultdict(dict);rows=[]
 for protocol,names in runs.items():
  for run in names:
   for f in (ROOT/'results'/run).rglob('records.jsonl'):
    for line in f.read_text().splitlines():
     r=json.loads(line)
     if r['variant'] not in wanted:continue
     score=score_docvqa(r['turns'][0]['answer'],refs[r['session_id']]['answer'])
     key=(r['image_id'],1);group=(protocol,r['variant'])
     if key in groups[group]:raise RuntimeError('Duplicate protocol/image/codec')
     groups[group][key]=score['score']
     rows.append({'protocol':protocol,'variant':r['variant'],'image_id':r['image_id'],'score':score['score'],'exact':score['exact'],
                  'first_token':r['turns'][0]['token_ids'][0],'token_count':len(r['turns'][0]['token_ids']),'prediction':r['turns'][0]['answer']})
 table=[]
 for (protocol,v),g in sorted(groups.items()):
  assert len(g)==100
  table.append({'protocol':protocol,'variant':v,'images':len(g),'anls':float(np.mean(list(g.values()))),
   'exact':float(np.mean([r['exact'] for r in rows if r['protocol']==protocol and r['variant']==v]))})
 contrasts=[]
 for v in sorted(wanted):
  for left,right in [('native_question_replay_last','compressed_question'),('native_question_shared_first','native_question_replay_last'),('native_question_shared_first','compressed_question')]:
   contrasts.append({'variant':v,'left':left,'right':right,**paired(groups[left,v],groups[right,v])})
 differences=[]
 for v in sorted(wanted-{'fp16_reference100'}):
  for left,right in [('native_question_replay_last','compressed_question'),('native_question_shared_first','native_question_replay_last')]:
   effect={k:groups[left,v][k]-groups[right,v][k]-(groups[left,'fp16_reference100'][k]-groups[right,'fp16_reference100'][k]) for k in groups[left,v]}
   differences.append({'variant':v,'left':left,'right':right,**paired(effect,{k:0. for k in effect})})
 first_tokens={(r['protocol'],r['variant'],r['image_id']):r['first_token'] for r in rows}
 first_agreement=[]
 for (protocol,v),g in sorted(groups.items()):
  ids=[k[0] for k in g];agree=sum(first_tokens[protocol,v,k]==first_tokens[protocol,'fp16_reference100',k] for k in ids)/len(ids)
  first_agreement.append({'protocol':protocol,'variant':v,'first_token_agreement_with_same_protocol_fp16':agree})
  if protocol=='native_question_shared_first' and agree!=1.:raise RuntimeError('Supposedly shared first logit did not share first token')
 write_csv(a.output/'scores.csv',rows);write_csv(a.output/'table.csv',table);write_csv(a.output/'contrasts.csv',contrasts)
 write_csv(a.output/'differences_in_differences.csv',differences);write_csv(a.output/'first_token_agreement.csv',first_agreement)
 result={'table':table,'contrasts':contrasts,'differences_in_differences':differences,'first_token_agreement':first_agreement,
  'scope':'matched100 development; native-question protocols are diagnostic, excluded from primary compression quality'}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
