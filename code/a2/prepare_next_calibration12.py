"""Unused old-pool images for a later calibration study, no current-test images."""
import copy,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 current=load(ROOT/'configs/a2/mechanism100.json');used={r['image_id'] for r in current};fresh={r['image_id'] for r in load(ROOT/'configs/a2/confirm64_short.json')}
 pool={r['image_id']:r for f in ('docvqa_dev32_remote_runner.json','docvqa_confirm64_remote_runner.json','docvqa_external_eval16_remote_runner.json') for r in load(ROOT/'configs/data'/f)}
 remaining=[pool[k] for k in sorted(pool) if k not in used and k not in fresh]
 assert len(remaining)==12;samples=[]
 for r in remaining:
  for turn,question in enumerate(r['questions'][:3],1):
   row=copy.deepcopy(r);row['id']=r['id']+'_nextcal_q'+str(turn);question=question+'\nAnswer the question using a single word or phrase.'
   row.update(questions=[question],content=[{'type':'image'},{'type':'text','text':question}],turns=1,split='unused_old_pool_calibration12_20261114',source_session_id=r['id'],source_turn=turn)
   samples.append(row)
 dest=ROOT/'configs/a2/next_calibration12.json'
 if dest.exists():raise RuntimeError('Next calibration pool already fixed')
 dest.write_text(json.dumps(samples,indent=2,ensure_ascii=False),encoding='utf-8')
 info={'images':12,'questions':len(samples),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'overlap_mechanism100':0,'overlap_new64':0,
  'selection':'all12 unused old-pool image hashes, first<=3 source questions; no model outcome selection','uses_ground_truth_answers':False,
  'role':'next research phase only, not used to select current frozen candidate','planned_change':'compare first-token vs multi-token teacher calibration, identity initialization for K2, and matched question-state formation'}
 (ROOT/'audit/a2/next_calibration12.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
if __name__=='__main__':main()
