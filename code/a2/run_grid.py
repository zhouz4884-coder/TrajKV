"""New resolution workpoints; frozen original executor, no answer access."""
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/experiments'))
import run_cycle as core
import torch
from transformers import AutoProcessor,AutoModelForImageTextToText
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for n in ('model','samples','legacy-scripts','kivi-source','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--caps',nargs='+',type=int,default=[4,8,16]);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=8)
 p.add_argument('--methods',nargs='+',default=['full','base2','native4','kvarn2'])
 a=p.parse_args();a.turns=6;a.max_tokens=32;a.margin=0.;a.max_replays=0;a.read_budget_bytes=None;a.gain_table_data=None;a.save_states=False
 a.output.mkdir(parents=True,exist_ok=False)
 core.install_legacy(a.legacy_scripts);torch.set_num_threads(4);torch.manual_seed(1212);torch.backends.cuda.matmul.allow_tf32=False
 rows=json.loads(a.samples.read_text())[a.start:a.start+a.count]
 (a.output/'manifest.json').write_text(json.dumps({'schema':'trajkv.a2.grid.v1','selected_ids':[r['id'] for r in rows],
  'caps':a.caps,'methods':a.methods,'answers_visible':False,'samples_sha256':sha(a.samples),
  'script_sha256':sha(__file__),'executor_sha256':sha(core.__file__),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__},indent=2))
 proc=AutoProcessor.from_pretrained(a.model,local_files_only=True)
 with (a.output/'processor_inventory.jsonl').open('w') as out:
  for row in rows:
   for cap in a.caps:
    t=time.perf_counter();f=core.features(proc,row,cap)
    v={'id':row['id'],'cap':cap,'shapes':{k:list(x.shape) for k,x in f.items()},'processor_ms':1000*(time.perf_counter()-t)}
    # The processor inventory never calls the model or reads answer labels.
    v['input_ids_count']=int(f['input_ids'].numel());out.write(json.dumps(v)+'\n');out.flush();del f
 model=AutoModelForImageTextToText.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,device_map='cuda:0',attn_implementation='sdpa').eval()
 with torch.inference_mode(),(a.output/'records.jsonl').open('w') as log:
  for row in rows:
   for cap in a.caps:
    a.max_patches=cap
    for method in a.methods:
     t=time.perf_counter();r,_=core.run_session(model,proc,row,a,method,'base');r['max_patches']=cap
     r['variant']=f'{method}:cap{cap}';log.write(json.dumps(r)+'\n');log.flush()
     print(json.dumps({'id':row['id'],'cap':cap,'method':method,'nv':r['visual_tokens'],'seconds':round(time.perf_counter()-t,2)}),flush=True)
 (a.output/'complete.json').write_text(json.dumps({'status':'COMPLETE','records':len(rows)*len(a.caps)*len(a.methods)}))
if __name__=='__main__':main()
