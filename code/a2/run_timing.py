"""Quiet GPU timing of the same packed model paths with fixed-length text.

No accuracy is scored. The 32-token length control repeats each image's native
FP16 answer tokens and ends with EOS; every method consumes identical IDs.
"""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
import session,session_boundary
from factory import bank_class
from timing_control import quiet_snapshot,verify_tokens
from transformers import AutoProcessor,AutoModelForImageTextToText

def quiet():
 q=quiet_snapshot()
 if not q['verified'] or not q['quiet']:raise RuntimeError('Another GPU worker is active; timing is deferred')
 return True
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text())
 quiet();a.output.mkdir(exist_ok=False,parents=True)
 for k,v in c['arguments'].items():setattr(a,k,Path(v) if k in ('model','samples','legacy_scripts','kivi_source') else v)
 for k,v in {'turns':1,'max_tokens':32,'max_patches':8,'margin':0.,'max_replays':0,'save_states':False,'read_budget_bytes':None,'gain_table_data':None}.items():
  if not hasattr(a,k):setattr(a,k,v)
 session.install_legacy(a.legacy_scripts);torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 rows=json.loads(a.samples.read_text())[:c.get('images',3)];proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
 manifest={'config':c,'config_sha256':sha(a.config),'sample_sha256':sha(a.samples),'sources':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
  'scope':'full-model, model-load excluded; native visual prefill + encode + complete question replay + fixed32 decode included',
  'quality':'not_scored','length_control':'repeat each FP16 free answer non-EOS IDs to31 then appendEOS; not a benchmark answer',
  'quiet_scope':'compute-process check on both GPUs at trial boundaries; no continuous thermal isolation'}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 canonical={}
 def run(row,v,forced=None):
  cls=bank_class(v['method']);cls.settings=v;session_boundary.ProgressiveBank=cls;a.max_patches=v.get('cap',8)
  return session_boundary.run_session(model,proc,row,a,v['method'],'base',forced=forced)[0]
 with torch.inference_mode():
  for row in rows:
   r=run(row,{'method':'full','cap':8,'selector':'none','store_selector_meta':False})
   ids=[x for x in r['turns'][0]['token_ids'] if x!=proc.tokenizer.eos_token_id]
   if not ids:ids=proc.tokenizer.encode('0',add_special_tokens=False)
   target=(ids*32)[:31]+[proc.tokenizer.eos_token_id]
   canonical[row['id']]={'token_ids':target}
  (a.output/'length_controls.json').write_text(json.dumps(canonical,indent=2))
  schedule=[]
  for phase,repeats in [('warmup',1),('measurement',c.get('repeats',4))]:
   for repeat in range(repeats):
    block=[(row,v) for row in rows for v in c['variants']];random.Random(20261113+repeat+(10000 if phase=='measurement' else 0)).shuffle(block)
    schedule += [(phase,repeat,row,v) for row,v in block]
  tokens_seen={};n=0
  with (a.output/'timing_records.jsonl').open('w') as log:
   for phase,repeat,row,v in schedule:
    quiet();t=time.perf_counter();r=run(row,v,[canonical[row['id']]]);wall_ms=1000*(time.perf_counter()-t);quiet()
    if r['turns'][0]['token_ids']!=canonical[row['id']]['token_ids']:raise RuntimeError('Timing tokens differ')
    key=row['id'];current=(r['turns'][0]['logical_end'],r['turns'][0]['forced_chat_closure'])
    # Same cap implies identical logical position/closure; cross-cap positions
    # differ because resolution is deliberately a separate resource setting.
    key=(key,v.get('cap',8))
    if key in tokens_seen and current!=tokens_seen[key]:raise RuntimeError('Timing positions differ at same cap')
    tokens_seen[key]=current
    record={'phase':phase,'repeat':repeat,'session_id':row['id'],'variant':v['name'],'quality':'not_scored','task_accuracy_eligible':False,
      'quiet_verified':True,'wall_ms_including_runner_setup':wall_ms,'runner_result':r}
    log.write(json.dumps(record)+'\n');log.flush();n+=1
    print(json.dumps({'trial':n,'phase':phase,'variant':v['name'],'session_ms':r['session_ms']}),flush=True)
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','trials':n,'measurement_trials':len(rows)*len(c['variants'])*c.get('repeats',4),'quality':'not_scored'}))
if __name__=='__main__':main()
