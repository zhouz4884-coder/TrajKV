"""Prespecify held-out continuous diagnostics for every learned-book ablation."""
import copy,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code'))
from remote import REMOTE_ROOT,PYTHON

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):
 if path.exists():raise RuntimeError('Do not overwrite staged diagnostics: '+str(path))
 path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
 marker=ROOT/'audit/a2/book_continuous64_design_v1.json'
 if marker.exists():raise RuntimeError('Diagnostics already frozen')
 base=json.loads((ROOT/'configs/a2/book_development64_s0_v2.json').read_text(encoding='utf-8'));samples=json.loads((ROOT/'configs/a2/docvqa_next_dev64.json').read_text(encoding='utf-8'));ids={x['id'] for x in samples}
 references=[{'id':r['id'],'answer':r['references'][0]['answer']} for r in map(json.loads,(ROOT/'configs/data/docvqa_next_references.jsonl').read_text(encoding='utf-8').splitlines()) if r['id'] in ids]
 assert len(references)==64;rp=ROOT/'configs/data/docvqa_bookdev_first_refs.json';write(rp,references)
 variants=[v for v in base['variants'] if v['method']!='full'];external={}
 for v in variants:
  for field in ['direction_book_path','decoder_book_path']:
   if v.get(field):
    path=v[field];assert path.startswith(REMOTE_ROOT+'/');external[path]=sha(ROOT/path[len(REMOTE_ROOT)+1:])
 design={'created_at_utc':datetime.now(timezone.utc).isoformat(),'images':64,'proxy_records':64*(1+len(variants)),
  'variants':variants,'sample_sha256':sha(ROOT/'configs/a2/docvqa_next_dev64.json'),'references_sha256':sha(rp),'external_codec_inputs_sha256':external,
  'scope':'first-question canonical reference token mean log-prob and Full-to-method KL; ground truth used for diagnostics only; all 11 free-development ablations retained; candidate selection uses complete free ANLS, not these proxies'}
 write(marker,design);jobs=[]
 for shard in [0,1]:
  name=f'book_continuous64_s{shard}_v1';cfg=copy.deepcopy(base);cfg.update(start=32*shard,count=32,variants=variants,
   usage='canonical_reference_diagnostics_only',external_codec_inputs_sha256=external,frozen_contract_sha256=sha(marker))
  cfg['arguments'].update(turns=1,max_patches=8,references=REMOTE_ROOT+'/configs/data/'+rp.name)
  cp=ROOT/'configs/a2'/f'{name}.json';write(cp,cfg)
  dependency=f'book_development64_s{shard}_v2'
  wait={'dependencies':[{'complete':REMOTE_ROOT+f'/results/{dependency}/complete.json','launch':REMOTE_ROOT+f'/jobs/{dependency}.launch.json'}],
   'entrypoint':'run_pinned_continuous.py','arguments':['--config',REMOTE_ROOT+'/configs/a2/'+cp.name]}
  wp=ROOT/'configs/a2'/f'{name}_wait.json';write(wp,wait)
  job={'name':name,'env':{'CUDA_VISIBLE_DEVICES':str(shard),'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1','TOKENIZERS_PARALLELISM':'false'},
   'argv':[PYTHON,'-u',REMOTE_ROOT+'/code/a2/run_after_dependencies.py','--config',REMOTE_ROOT+'/configs/a2/'+wp.name,'--output',REMOTE_ROOT+'/results/'+name]}
  write(ROOT/'configs/jobs'/f'{name}.json',job);jobs.append(name)
 print(json.dumps({'jobs':jobs,'proxy_records':design['proxy_records'],'reference_file_upload_required':str(rp)}))

if __name__=='__main__':main()
