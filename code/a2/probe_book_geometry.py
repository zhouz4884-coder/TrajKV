"""Actual packed model queries with extra FP16 diagnostic references only."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch
from transformers import AutoProcessor,AutoModelForImageTextToText
from types import SimpleNamespace
import session_boundary
from masked_bank import A2Bank
from learned_direction_bank import LearnedDirectionBank
from fast_wht import wht,torch_wht

REFERENCE_QUERIES={};METRICS=[]

class CapturedFull(A2Bank):
 def read(self,q,i):
  if self.turn==1 and self.phase=='question_prefill' and i not in REFERENCE_QUERIES:REFERENCE_QUERIES[i]=q[:,:,-1:].detach().float().cpu()
  return super().read(q,i)

class CapturedLearned(LearnedDirectionBank):
 def _initialize(self,model,cache,visual,method,legacy_scripts,kivi_source,keep_reference=False):
  self.diagnostic_keys=[layer.keys.index_select(2,visual).detach().float().clone() for layer in cache.layers]
  self.measured_layers=set()
  super()._initialize(model,cache,visual,method,legacy_scripts,kivi_source,False)
 def read(self,q,i):
  if self.turn==1 and self.phase=='question_prefill' and i not in self.measured_layers:
   self.measured_layers.add(i);state=self.states[i];nh=state.direction_code.shape[1];group=q.shape[1]//nh
   book=self.book if self.decoder_books is None else self.decoder_books[i];codes=state.direction_code.long()
   template=book[codes] if self.decoder_books is None else book[torch.arange(nh,device=q.device)[None,:,None,None],codes]
   rotated=(template.float().reshape(1,nh,len(state.visual_pos),128)*state.direction_gain.float()).half()
   original=self.diagnostic_keys[i];mean=state.direction_mean.float();centered=original-mean
   reconstructed_centered=torch_wht(rotated.float());reconstructed=reconstructed_centered+mean
   qr=REFERENCE_QUERIES[i].to(q.device);qc=q[:,:,-1:].float();keys=original.repeat_interleave(group,1)
   def scores(query):
    return (wht(query.half()).float()@rotated.float().repeat_interleave(group,1).transpose(-1,-2)+(query*mean.repeat_interleave(group,1)).sum(-1,keepdim=True))/(128**.5)
   native=qr@keys.transpose(-1,-2)/(128**.5);native_changed=qc@keys.transpose(-1,-2)/(128**.5)
   fixed=scores(qr);actual=scores(qc);direct=fixed-native;state_change=native_changed-native;interaction=actual-native_changed-direct
   residual=(direct+state_change+interaction-(actual-native)).abs().max();assert float(residual)<.001
   p=native.log_softmax(-1);ph=fixed.log_softmax(-1)
   cosine=lambda x,y:torch.nn.functional.cosine_similarity(x,y,dim=-1).mean()
   rmse=lambda x:float(x.square().mean().sqrt())
   METRICS.append({'layer':i,'visual_tokens':len(state.visual_pos),'centered_key_cosine':float(cosine(centered,reconstructed_centered)),
    'full_key_cosine':float(cosine(original,reconstructed)),'centered_key_mse':float((centered-reconstructed_centered).square().mean()),
    'centered_norm_ratio':float((reconstructed_centered.norm(dim=-1)/centered.norm(dim=-1).clamp_min(1e-10)).mean()),
    'query_cosine_to_full':float(cosine(qr,qc)),'direct_score_rmse_fixed_full_query':rmse(direct),'state_score_rmse':rmse(state_change),
    'interaction_score_rmse':rmse(interaction),'total_score_rmse':rmse(actual-native),'decomposition_max_residual':float(residual),
    'visual_only_attention_kl_fixed_full_query':float((p.exp()*(p-ph)).sum(-1).mean()),
    'visual_top1_match_fixed_full_query':float((native.argmax(-1)==fixed.argmax(-1)).float().mean()),'task_accuracy_eligible':False})
  return super().read(q,i)

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();c=json.loads(args.config.read_text());args.output.mkdir(parents=True,exist_ok=False)
 a=SimpleNamespace(**c['arguments'])
 for key in ['model','samples','legacy_scripts','kivi_source']:setattr(a,key,Path(getattr(a,key)))
 for key,value in {'turns':1,'max_tokens':1,'margin':0.,'max_replays':0,'save_states':False,'read_budget_bytes':None,'gain_table_data':None}.items():setattr(a,key,value)
 session_boundary.install_legacy(a.legacy_scripts);torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 processor=AutoProcessor.from_pretrained(a.model,local_files_only=True);model=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
 rows=json.loads(a.samples.read_text())[c['start']:c['start']+c['count']];total=0
 manifest={'config':c,'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),'samples_sha256':hashlib.sha256(a.samples.read_bytes()).hexdigest(),'task_accuracy_eligible':False,
  'external_codec_inputs':{variant[key]:hashlib.sha256(Path(variant[key]).read_bytes()).hexdigest() for variant in c['variants'] for key in ['direction_book_path','decoder_book_path'] if variant.get(key)},
  'scope':'first-question last-prefill query from actual packed model execution; FP16 source keys retained only for this instrument; no storage/timing claim; visual-only score algebra is a diagnostic, not vocabulary logit KL',
  'sources':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob('*.py')}}
 (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 with torch.inference_mode(),(args.output/'geometry.jsonl').open('w') as stream:
  for row in rows:
   REFERENCE_QUERIES.clear();CapturedFull.settings={'method':'full','selector':'none','cap':8,'store_selector_meta':False};session_boundary.ProgressiveBank=CapturedFull;a.max_patches=8
   session_boundary.run_session(model,processor,row,a,'full','base',forced=[{'token_ids':[processor.tokenizer.eos_token_id]}]);assert len(REFERENCE_QUERIES)==28
   for variant in c['variants']:
    METRICS.clear();CapturedLearned.settings=variant;session_boundary.ProgressiveBank=CapturedLearned
    session_boundary.run_session(model,processor,row,a,'learned_direction','base',forced=[{'token_ids':[processor.tokenizer.eos_token_id]}]);assert len(METRICS)==28
    for entry in METRICS:stream.write(json.dumps({'session_id':row['id'],'variant':variant['name'],**entry})+'\n');total+=1
    stream.flush()
   print(json.dumps({'done_image':row['id'],'diagnostic_layer_records':total}),flush=True)
 (args.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','images':len(rows),'diagnostic_layer_records':total,'task_accuracy_eligible':False}))

if __name__=='__main__':main()
