"""Quiet matched-output timing of actual frozen multi-question cache paths."""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
import session,session_boundary
from factory import bank_class
from run_timing import quiet
from transformers import AutoProcessor,AutoModelForImageTextToText

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text());quiet();a.output.mkdir(exist_ok=False,parents=True)
 for k,v in c['arguments'].items():setattr(a,k,Path(v) if k in ('model','samples','legacy_scripts','kivi_source') else v)
 for k,v in {'turns':6,'max_tokens':32,'max_patches':8,'margin':0.,'max_replays':0,'save_states':False,'read_budget_bytes':None,'gain_table_data':None}.items():
  if not hasattr(a,k):setattr(a,k,v)
 session.install_legacy(a.legacy_scripts);torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 rows=json.loads(a.samples.read_text(encoding='utf-8'))[:c.get('images',3)];processor=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
 ext={v[k]:sha(v[k]) for v in c['variants'] for k in ('direction_book_path','query_scale_path','k_fold_factor_path','v_fold_factor_path','affine_factor_path','decoder_book_path') if v.get(k)}
 manifest={'config':c,'config_sha256':sha(a.config),'samples_sha256':sha(a.samples),'selected_ids':[r['id'] for r in rows],
  'external_codec_inputs':ext,'sources':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'quality':'not_scored',
  'scope':'model load excluded; native causal-prefix extraction, cache construction, question replay and decode included',
  'length_control':'same8 forced output IDs per actual question, generated from Full, repeated/truncated to7+EOS; accuracy not scored',
  'quiet_scope':'both-GPU compute-process checks at boundaries; no claim of continuous thermal isolation'}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 def run(row,v,forced=None):
  cls=bank_class(v['method']);cls.settings=v;session_boundary.ProgressiveBank=cls;a.max_patches=v.get('cap',8)
  return session_boundary.run_session(model,processor,row,a,v['method'],'base',forced=forced)[0]
 canonical={};length=c.get('tokens_per_turn',8)
 with torch.inference_mode():
  for row in rows:
   result=run(row,{'method':'full','cap':8,'selector':'none','store_selector_meta':False});targets=[]
   for turn in result['turns']:
    ids=[t for t in turn['token_ids'] if t!=processor.tokenizer.eos_token_id]
    if not ids:ids=processor.tokenizer.encode('0',add_special_tokens=False)
    targets.append({'token_ids':(ids*length)[:length-1]+[processor.tokenizer.eos_token_id]})
   canonical[row['id']]=targets
  (a.output/'length_controls.json').write_text(json.dumps(canonical,indent=2))
  schedule=[]
  for phase,n in [('warmup',1),('measurement',c.get('repeats',4))]:
   for repeat in range(n):
    block=[(row,v) for row in rows for v in c['variants']];random.Random(20261114+repeat+(10000 if phase=='measurement' else 0)).shuffle(block)
    schedule += [(phase,repeat,row,v) for row,v in block]
  planned=[{'phase':phase,'repeat':repeat,'session_id':row['id'],'variant':v['name']} for phase,repeat,row,v in schedule]
  (a.output/'schedule.json').write_text(json.dumps(planned,indent=2));positions={};count=0
  with (a.output/'timing_records.jsonl').open('w') as log:
   for phase,repeat,row,v in schedule:
    quiet();start=time.perf_counter();r=run(row,v,canonical[row['id']]);wall=1000*(time.perf_counter()-start);quiet()
    if [t['token_ids'] for t in r['turns']]!=[t['token_ids'] for t in canonical[row['id']]]:raise RuntimeError('Output IDs differ')
    current=[(t['logical_end'],t['forced_chat_closure']) for t in r['turns']];key=(row['id'],v.get('cap',8))
    if key in positions and current!=positions[key]:raise RuntimeError('Same-cap logical positions differ')
    positions[key]=current
    record={'phase':phase,'repeat':repeat,'session_id':row['id'],'variant':v['name'],'quality':'not_scored','task_accuracy_eligible':False,
     'quiet_verified':True,'wall_ms_including_runner_setup':wall,'runner_result':r}
    log.write(json.dumps(record)+'\n');log.flush();count+=1
    print(json.dumps({'trial':count,'phase':phase,'variant':v['name'],'session_ms':r['session_ms']}),flush=True)
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','trials':count,'measurement_trials':len(rows)*len(c['variants'])*c.get('repeats',4),'quality':'not_scored'}))
if __name__=='__main__':main()
