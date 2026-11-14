"""GPU model forwards for canonical-reference log-prob and FP16 logit KL.

Labels are intentionally visible for teacher forcing. No answers from these
records enter task accuracy, and the file is named proxies.jsonl accordingly.
The free-generation end-to-end jobs are separate and remain primary.
"""
import argparse,gc,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
import session_teacher as executor
from masked_bank import A2Bank
from rate_bank import RateBank
from single_axis_bank import SingleAxisBank
from factory import bank_class
from transformers import AutoProcessor,AutoModelForImageTextToText
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();c=json.loads(a.config.read_text())
 a.output.mkdir(exist_ok=False,parents=True)
 for k,v in c['arguments'].items():setattr(a,k,Path(v) if k in ('model','samples','legacy_scripts','kivi_source','references') else v)
 for k,v in {'turns':1,'max_tokens':32,'max_patches':8,'margin':0.,'max_replays':0,'save_states':False,'read_budget_bytes':None,'gain_table_data':None}.items():
  if not hasattr(a,k):setattr(a,k,v)
 executor.install_legacy(a.legacy_scripts);torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 rows=json.loads(a.samples.read_text());rows=rows[c.get('start',0):c.get('start',0)+c.get('count',len(rows))]
 refs={r['id']:r for r in json.loads(a.references.read_text())}
 manifest={'config':c,'samples_sha256':sha(a.samples),'references_sha256':sha(a.references),'selected_ids':[r['id'] for r in rows],
  'reference_answers_visible':True,'scope':'canonical first reference teacher forcing; not task accuracy or serving timing',
  'reference_choice':'first provided string, original case/format, content positions scored; EOS separate',
  'gpu':torch.cuda.get_device_name(),'sources':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval();n=0
 def run(row,v,forced):
  cls=bank_class(v['method'])
  cls.settings=v;executor.ProgressiveBank=cls;a.max_patches=v.get('cap',8)
  result,traces=executor.run_session(model,proc,row,a,v['method'],'base',forced=forced)
  return result,traces
 with torch.inference_mode(),(a.output/'proxies.jsonl').open('w') as log:
  for row in rows:
   gold=refs[row['id']]['answer'];gold=gold[0] if isinstance(gold,list) else gold
   tokens=proc.tokenizer.encode(gold,add_special_tokens=False)
   if not tokens:raise RuntimeError('Empty canonical target')
   forced=[{'token_ids':tokens+[proc.tokenizer.eos_token_id]}]
   _,baseline=run(row,{'method':'full','selector':'none','cap':8},forced)
   assert len(baseline)==len(tokens)+1
   ref_lp=[z.reshape(-1).double().log_softmax(-1) for z in baseline]
   ref_gold=[float(lp[t]) for lp,t in zip(ref_lp,tokens+[proc.tokenizer.eos_token_id])]
   base_row={'session_id':row['id'],'image_id':row['image_id'],'variant':'full_reference','teacher_forced':True,'task_accuracy_eligible':False,
    'content_tokens':len(tokens),'mean_reference_token_logprob':sum(ref_gold[:-1])/len(tokens),'mean_kl_fp16_to_method':0.,'argmax_changes':0,'eos_logprob':ref_gold[-1]}
   log.write(json.dumps(base_row)+'\n');log.flush();n+=1
   for v in c['variants']:
    result,traces=run(row,v,forced)
    if len(traces)!=len(baseline):raise RuntimeError('Token schedule differs')
    kl=[];gold_lp=[];flips=0
    for z,reference,lp_ref,t in zip(traces,baseline,ref_lp,tokens+[proc.tokenizer.eos_token_id]):
     lp=z.reshape(-1).double().log_softmax(-1);kl.append(float((lp_ref.exp()*(lp_ref-lp)).sum()));gold_lp.append(float(lp[t]));flips+=int(z.argmax()!=reference.argmax())
    r={'session_id':row['id'],'image_id':row['image_id'],'variant':v['name'],'teacher_forced':True,'task_accuracy_eligible':False,
      'content_tokens':len(tokens),'mean_reference_token_logprob':sum(gold_lp[:-1])/len(tokens),'mean_kl_fp16_to_method':sum(kl[:-1])/len(tokens),
      'gold_logprob_delta_to_fp16':(sum(gold_lp[:-1])-sum(ref_gold[:-1]))/len(tokens),'eos_logprob':gold_lp[-1],
      'argmax_changes':flips,'positions_including_eos':len(traces),'read_ledger_exact':result['read_ledger_exact'],'variant_config':v}
    log.write(json.dumps(r)+'\n');log.flush();n+=1
   print(json.dumps({'completed_image':row['id'],'proxy_records':n}),flush=True)
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','proxy_records':n,'images':len(rows),'paid_calls':0}))
if __name__=='__main__':main()
