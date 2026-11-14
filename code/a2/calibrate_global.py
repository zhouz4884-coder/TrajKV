"""Global logit sensitivity of removing one K refinement from a working K-only codec.

Development-only, fixed token/position replay, no ground-truth answers or future
queries. The reference is the all-K-refined codec itself, not a quality oracle.
"""
import argparse,gc,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from run_cycle import features,consume,sync
from bank import install_legacy
from allocation_bank import AllocationBank
from transformers import AutoProcessor,AutoModelForImageTextToText
def main():
 p=argparse.ArgumentParser()
 for n in ('model','samples','legacy-scripts','kivi-source','evaluation-config','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True);install_legacy(a.legacy_scripts)
 torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 from p180_multiturn_core import attention_bridge
 from p184_working_set import attention
 rows=json.loads(a.samples.read_text())[:8];proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
 manifest={'samples_sha256':hashlib.sha256(a.samples.read_bytes()).hexdigest(),'selected_ids':[r['id'] for r in rows],
  'scope':'fixed first-answer token replay; all-K-refined baseline vs one K layer removed','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'answers_visible':False}
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));allrows=[]
 with torch.inference_mode(),(a.output/'records.jsonl').open('w') as log:
  for row in rows:
   f={k:v.to(model.device,dtype=model.dtype if v.is_floating_point() else v.dtype) for k,v in features(proc,row,16).items()}
   visual=(f['input_ids'][0]==model.config.image_token_id).nonzero().flatten()
   with attention_bridge(model,native=True):out=model(**f,use_cache=True,return_dict=True,logits_to_keep=1)
   cache=out.past_key_values;first=out.logits[:,-1].clone();logical=f['input_ids'].shape[-1];del out,f
   AllocationBank.settings={'selector':'all','fraction':1,'store_selector_meta':False,'k_layers':list(range(28)),'v_layers':[]}
   bank=AllocationBank(model,cache,visual,'alloc1',a.legacy_scripts,a.kivi_source)
   try:
    initial=bank.snapshot(cpu=True);tokens=[];reference=[];logits=first
    for step in range(4):
     token=int(logits.argmax(-1).item());tokens.append(token)
     with attention(model):out,logical=consume(model,cache,torch.tensor([[token]],device=model.device),logical)
     logits=out.logits[:,-1];reference.append(logits.float().cpu().clone());del out
     if token==proc.tokenizer.eos_token_id:break
    stats=[]
    for layer in [-1]+list(range(28)):
     bank.restore(initial);bank.axis_flags=[(i!=layer,False) for i in range(28)];pos=initial[0]['length'];kl=[];maxdiff=[];flips=0
     for token,ref in zip(tokens,reference):
      with attention(model):out,pos=consume(model,cache,torch.tensor([[token]],device=model.device),pos)
      logits=out.logits[:,-1].float().cpu();del out
      lp=logits.double().log_softmax(-1);lq=ref.double().log_softmax(-1)
      kl.append(float((lq.exp()*(lq-lp)).sum()));maxdiff.append(float((logits-ref).abs().max()));flips+=int(logits.argmax()!=ref.argmax())
     stats.append({'removed_k_layer':layer,'mean_kl':sum(kl)/len(kl),'argmax_changes':flips,'positions':len(tokens),'max_abs':max(maxdiff)})
    if stats[0]['max_abs']!=0:raise RuntimeError('Same-state replay null failed')
    r={'session_id':row['id'],'tokens':tokens,'statistics':stats,'reference':'all K-r2 + 1-bit V','visual_tokens':int(visual.numel()),'owned_diagnostic':bank.owned()}
    allrows.append(r);log.write(json.dumps(r)+'\n');log.flush();print(json.dumps({'done':len(allrows),'id':row['id']}),flush=True)
   finally:bank.close();del bank,cache;gc.collect()
 sensitivity={i:sum(next(x['mean_kl'] for x in r['statistics'] if x['removed_k_layer']==i) for r in allrows)/len(allrows) for i in range(28)}
 ranking=sorted(sensitivity,key=lambda i:(-sensitivity[i],i))
 cfg=json.loads(a.evaluation_config.read_text());rules=[]
 for slots in (12,24,36):
  rules.append({'name':f'global_k_s{slots}','method':'alloc1','selector':'all','fraction':1,'store_selector_meta':False,
   'k_layers':sorted(ranking[:slots//2]),'v_layers':[],'slots':slots,'calibration_scope':'first-answer fixed-token KL sensitivity to removing K layer from all-K codec, dev8'})
 cfg['variants']+=rules
 (a.output/'fitted_variants.json').write_text(json.dumps(cfg,indent=2));(a.output/'frozen_table.json').write_text(json.dumps({'sensitivity':sensitivity,'ranking':ranking,'rules':rules},indent=2))
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','records':len(allrows),'same_state_null_max':0}))
if __name__=='__main__':main()
