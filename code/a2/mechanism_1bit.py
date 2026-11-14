"""First-cycle four-arm causal replay, repeated at the actual damaged 1-bit point."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/experiments'))
import torch,mechanism
from bank import ProgressiveBank,install_legacy
from transformers import AutoProcessor,AutoModelForImageTextToText
def onebit(*args,**kwargs):
 x=list(args);assert x[3]=='asym';x[3]='asym1';return ProgressiveBank(*x,**kwargs)
def main():
 p=argparse.ArgumentParser()
 for n in ('model','samples','legacy-scripts','kivi-source','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.count=4;a.turns=4;a.max_tokens=12;a.max_patches=16;a.future_horizon=1;a.min_target_turn=2;a.action_turns=[];a.parsed_actions=[];a.save_logits=False
 a.output.mkdir(exist_ok=False,parents=True);install_legacy(a.legacy_scripts)
 torch.manual_seed(1212);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 rows=json.loads(a.samples.read_text())[:a.count];mechanism.ProgressiveBank=onebit
 (a.output/'manifest.json').write_text(json.dumps({'selected_ids':[r['id'] for r in rows],'base':'1bit K/V g128, no refinement reads','max_tokens':a.max_tokens,'turns':a.turns,
  'canonical_tokens':'own base1 generated, no ground-truth labels','original_executor_sha256':hashlib.sha256(Path(mechanism.__file__).read_bytes()).hexdigest(),
  'scope':'fixed-token/position four-arm contrast, not downstream answer quality or natural mediation'},indent=2))
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 model=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
 with torch.inference_mode(),(a.output/'records.jsonl').open('w') as log:
  for row in rows:
   r=mechanism.run_mechanism(model,proc,row,a);r['base_precision']=1;log.write(json.dumps(r)+'\n');log.flush()
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','records':len(rows)}))
if __name__=='__main__':main()
