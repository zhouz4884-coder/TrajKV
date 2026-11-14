"""C: real KV eviction vs clean replay, original positions, controlled acknowledgements.

Only final answers are freely generated and evaluated. The exposed context
uses fixed assistant OK acknowledgements to avoid explicit answer leakage into
later text. Deployment deletion can still leave information in later hidden KV.
Clean replay skips removed complete message blocks, preserving original RoPE
positions. Image/model re-prefill and retained-text replay are explicitly timed.
"""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from run_cycle import features,consume,sync
from masked_bank import A2Bank
from bank import install_legacy
from transformers import AutoProcessor,AutoModelForImageTextToText

ARMS=[('keep_base2','base2','deploy',[]),('drop_fact','base2','deploy',['fact']),
 ('drop_secret','base2','deploy',['secret']),('drop_both','base2','deploy',['fact','secret']),
 ('clean_drop_fact','base2','clean',['fact']),('clean_drop_secret','base2','clean',['secret']),
 ('clean_drop_both','base2','clean',['fact','secret']),('drop_fact_native4','native4','deploy',['fact']),
 ('keep_full','full','deploy',[])]

def run(model,proc,row,a,arm,target):
 from p180_multiturn_core import attention_bridge,next_suffix
 from p184_working_set import attention
 import run_p175_modern_replication as old
 name,method,mode,drop=arm
 gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();sync();start=time.perf_counter()
 f={k:v.to(model.device,dtype=model.dtype if v.is_floating_point() else v.dtype) for k,v in features(proc,row,a.max_patches).items()}
 visual=(f['input_ids'][0]==model.config.image_token_id).nonzero().flatten()
 with attention_bridge(model,native=True):o=model(**f,use_cache=True,return_dict=True,logits_to_keep=1)
 cache=o.past_key_values;logical=f['input_ids'].shape[-1];del o,f
 A2Bank.settings={'selector':'none'};bank=A2Bank(model,cache,visual,method,a.legacy_scripts,a.kivi_source)
 ack=proc.tokenizer.encode('OK',add_special_tokens=False)+[proc.tokenizer.eos_token_id]
 ranges={};explicit_tokens=0;skipped_tokens=0
 try:
  def feed(ids):
   nonlocal logical,explicit_tokens
   for chunk in ids:
    with attention(model):out,logical=consume(model,cache,chunk,logical)
    explicit_tokens+=chunk.shape[-1]
    del out
  feed([torch.tensor([[v]],device=model.device) for v in ack])
  order=['fact','secret'] if row['context_order']=='fact_first' else ['secret','fact']
  blocks=[(kind,row[kind+'_statement']) for kind in order]+[('distractor','Please keep replies concise, with no explanation. Reply only OK.')]
  for number,(kind,text) in enumerate(blocks,2):
   bank.turn=number;bank.phase='context';begin=logical
   suffix=next_suffix(proc,text).to(model.device);size=suffix.shape[-1]+len(ack)
   ranges[kind]=(begin,begin+size)
   if mode=='clean' and kind in drop:
    logical+=size;skipped_tokens+=size
    for s in bank.states:s.length=logical
   else:feed([suffix]+[torch.tensor([[v]],device=model.device) for v in ack])
  sync();context_ms=1000*(time.perf_counter()-start)
  remove=[ranges[k] for k in drop] if mode=='deploy' else []
  logical_before=logical;bank.compact(remove)
  assert all(s.length==logical_before for s in bank.states)
  expected_remove=[ranges[k] for k in drop]
  for s in bank.states:
   pos=s.text_pos[:s.text_length]
   for lo,hi in expected_remove:
    if bool(((pos>=lo)&(pos<hi)).any()):raise RuntimeError('Evicted position remains')
  owner=bank.owned();sync();allocated=torch.cuda.memory_allocated();query_start=time.perf_counter()
  bank.turn=5;bank.phase='target';question=row[target+'_question']
  ids=next_suffix(proc,question).to(model.device)
  with attention(model):out,logical=consume(model,cache,ids,logical)
  logits=out.logits[:,-1];del out;pred=[];eos=old.stop_ids(model,proc)
  for step in range(a.max_tokens):
   token=int(logits.argmax(-1).item());pred.append(token)
   if token in eos:break
   with attention(model):out,logical=consume(model,cache,torch.tensor([[token]],device=model.device),logical)
   logits=out.logits[:,-1];del out
  sync()
  return {'id':row['id'],'image_id':row['image_id'],'ordering':row['context_order'],'arm':name,'method':method,'mode':mode,'drop':drop,'target':target,
   'prediction':proc.tokenizer.decode(pred,skip_special_tokens=True).strip(),'token_ids':pred,'visual_tokens':int(visual.numel()),
   'context_ranges':ranges,'context_tokens_consumed':explicit_tokens,'context_tokens_skipped':skipped_tokens,
   'owned_before_target':owner,'allocated_before_target':allocated,'logical_before_target':logical_before,
   'peak_allocated':torch.cuda.max_memory_allocated(),'context_ms':context_ms,'target_ms':1000*(time.perf_counter()-query_start),
   'session_ms':1000*(time.perf_counter()-start),'read_events':bank.event_rows(),'eviction_events':bank.evictions,
   'legal_original_positions':True,'later_deployment_kv_not_reencoded':mode=='deploy',
   'protocol':'constructed_fixed_acknowledgements_free_target','labels_visible':'fact and private binding explicitly supplied in user context; no hidden evaluator labels accessed'}
 finally:bank.close();del bank,cache;gc.collect()

def main():
 p=argparse.ArgumentParser()
 for n in ('model','samples','legacy-scripts','kivi-source','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--max-patches',type=int,default=4);p.add_argument('--max-tokens',type=int,default=24);p.add_argument('--count',type=int,default=16)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);install_legacy(a.legacy_scripts)
 torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 rows=json.loads(a.samples.read_text())[:a.count]
 (a.output/'manifest.json').write_text(json.dumps({'arms':ARMS,'selected_ids':[r['id'] for r in rows],'targets':['fact','secret'],'scope':'controlled mechanism diagnostic, not benchmark','samples_sha256':hashlib.sha256(a.samples.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval();n=0
 with torch.inference_mode(),(a.output/'records.jsonl').open('w') as log:
  for row in rows:
   for arm in ARMS:
    for target in ('fact','secret'):
     r=run(model,proc,row,a,arm,target);log.write(json.dumps(r)+'\n');log.flush();n+=1
     print(json.dumps({'done':n,'id':row['id'],'arm':arm[0],'target':target,'prediction':r['prediction']}),flush=True)
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','records':n,'expected':len(rows)*len(ARMS)*2}))
if __name__=='__main__':main()
