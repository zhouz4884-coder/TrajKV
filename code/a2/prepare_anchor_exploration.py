"""A bounded native-prefix transfer test, selected on already-open dev64 only."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def main():
 path=ROOT/'audit/a2/anchor_development_design_v1.json'
 if path.exists():raise RuntimeError('Anchor exploration already declared')
 dev=json.loads((ROOT/'configs/a2/trajectory_development64_s0_v1.json').read_text());available={v['name']:v for v in dev['variants']}
 prompt='Observe the complete image and all visible text carefully. Remember the visual details, names, numbers, and spatial relationships for the questions that follow.\n'
 variants=[{'name':'anchor_full_original_prompt','method':'anchor_full','anchor_mode':'none','anchor_prompt':'','cap':8,'selector':'none','store_selector_meta':False},
  {'name':'anchor_full_fixed_prompt','method':'anchor_full','anchor_mode':'none','anchor_prompt':prompt,'cap':8,'selector':'none','store_selector_meta':False}]
 for label,base_name,method in [('affine','warm_multi_gain','anchor_affine'),('direction','direction_vq2_v1','anchor_direction')]:
  for suffix,mode,prefix in [('original','none',''),('prompt_only','none',prompt),('native','native',prompt),('compressed','compressed',prompt)]:
   cfg=copy.deepcopy(available[base_name]);cfg.update(name='anchor_'+label+'_'+suffix,method=method,anchor_mode=mode,anchor_prompt=prefix,anchor_budget_bits=1.8,anchor_max_tokens=16)
   variants.append(cfg)
 jobs=[]
 for shard in (0,1):
  name=f'anchors_development64_s{shard}_v1';cfg=copy.deepcopy(dev);cfg.update(variants=variants,start=32*shard,count=32)
  cp=ROOT/'configs/a2'/f'{name}.json';cp.write_text(json.dumps(cfg,indent=2))
  wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/cached_reads_dev64_s{shard}_v1/complete.json','launch':REMOTE_ROOT+f'/jobs/cached_reads_dev64_s{shard}_v1.launch.json'}],
   'entrypoint':'run_variants.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
  wp=ROOT/'configs/a2'/f'{name}_wait.json';wp.write_text(json.dumps(wait,indent=2))
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
  (ROOT/'configs/jobs'/f'{name}.json').write_text(json.dumps(job,indent=2));jobs.append(name)
 report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'motivation':'six metadata fits reached84.79–85.25 on new dev64 while Full92.00; test a different repair: transfer precision to a small fixed-prefix state before any actual question',
  'variants':variants,'images':64,'questions':255,'sessions':640,'jobs':jobs,
  'controls':'same fixed prompt, native vs compressed-image anchor formation, prompt-only m0, original-prompt m0, and Full matching both prompts',
  'native_prefix_scope':'model sees only image plus budgeted prefix tokens before packing; actual question tokens are truncated out of the native model input; independent sentinel tokenization check every session',
  'budget':'per-image m=floor((1.8bit budget−actual base/metadata/positions/shared encoder constants)/(two FP16 KV arrays per state+shared position)); max16; all dedicated compact anchor storage charged',
  'runtime':'actual packed base and FP16 compact anchor tensors, no retained full visual KV; temporary text concat cost charged/timed',
  'data_scope':'already-open development64 only; confirm256 and external balanced128 still unused',
  'source_sha256':{name:hashlib.sha256((ROOT/'code/a2'/name).read_bytes()).hexdigest() for name in ('anchor_bank.py','anchor_protocol.py','session_anchor.py','run_variants.py','factory.py')}}
 path.write_text(json.dumps(report,indent=2));print(json.dumps({'variants':len(variants),'sessions':640,'jobs':jobs}))
if __name__=='__main__':main()
